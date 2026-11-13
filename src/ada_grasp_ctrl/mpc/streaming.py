"""Clocked, bounded mock commands; no ROS or hardware imports."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .contracts import JointFeedback, MPCProposal, fresh, named_vector
from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES
from ada_grasp_ctrl.ros_execution.settings import bridge_config


@dataclass
class CommandOwnership:
    """Exclusive owner shared by legacy/mock paths; transport also checks DDS graph."""

    owner: str | None = None

    def claim(self, owner: str) -> None:
        if self.owner is not None or owner not in ("baseline", "mpc_mock", "mpc_active"):
            raise ValueError("Command owner collision")
        self.owner = owner

    def release(self, owner: str) -> None:
        if self.owner != owner:
            raise ValueError("Command owner mismatch")
        self.owner = None


class GuardedStream:
    """Reject unsafe/stale inputs and bound position, step, speed and acceleration.

    Failures latch command silence, retaining the last bounded target at the mock
    backend. This is not a claim about physical emergency stopping behavior.
    """

    def __init__(
        self,
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        model_speed: tuple[float, ...],
        *,
        config: dict | None = None,
        ownership: CommandOwnership | None = None,
        hardware_enabled: bool = False,
    ) -> None:
        self.config = bridge_config(config)
        if self.config.get("side", "right") != "right" or (
            self.config["environment"] == "hardware" and not hardware_enabled
        ):
            raise ValueError("Streaming is right-side only; hardware requires explicit enable")
        if self.config["rate_hz"] != 100:
            raise ValueError("Phase 2 command rate must be 100 Hz")
        self.lower = named_vector(JOINT_NAMES, lower)
        self.upper = named_vector(JOINT_NAMES, upper)
        model_speed = named_vector(JOINT_NAMES, model_speed)
        if any(lo >= hi or v <= 0 for lo, hi, v in zip(self.lower, self.upper, model_speed)):
            raise ValueError("Invalid model limits")
        groups = ["arm"] * 6 + ["hand"] * 22
        self.speed = tuple(min(v, self.config[g + "_speed"]) for v, g in zip(model_speed, groups))
        self.acceleration = tuple(self.config[g + "_acceleration"] for g in groups)
        self.step = tuple(self.config[g + "_step"] for g in groups)
        self.ownership = ownership or CommandOwnership()
        self.q: tuple[float, ...] | None = None
        self.velocity = (0.0,) * 28
        self.target: MPCProposal | None = None
        self.last_tick: float | None = None
        self.last_source: float | None = None
        self.fault: str | None = None
        self.running = False
        self.dropped_ticks = 0

    def check_limits(self, q: tuple[float, ...]) -> None:
        if any(not lo - 1e-9 <= v <= hi + 1e-9 for v, lo, hi in zip(q, self.lower, self.upper)):
            raise ValueError("Joint-limit violation")

    def start(self, feedback: JointFeedback, now: float, *, owner: str = "mpc_mock") -> None:
        """Start from fresh measured positions without publishing a target jump."""
        if self.running or self.fault:
            raise ValueError("Stream cannot be restarted")
        fresh(feedback.source_timestamp, feedback.receive_timestamp, now, self.config["feedback_timeout"])
        self.check_limits(feedback.positions_rad)
        self.ownership.claim(owner)
        self.owner = owner
        self.q, self.last_tick, self.last_source = feedback.positions_rad, now, feedback.source_timestamp
        self.running = True

    def stop(self, reason: str) -> None:
        """Latch failure/cancel and release ownership; no further commands are returned."""
        self.fault = reason
        if self.running:
            self.ownership.release(self.owner)
            self.running = False

    def accept(self, proposal: MPCProposal, now: float) -> None:
        """Only valid, fresh absolute targets can replace the last accepted target."""
        try:
            if not self.running:
                raise ValueError("Stopped stream")
            if not proposal.valid or proposal.positions_rad is None:
                raise ValueError(
                    "Rejected MPC proposal: " + str(proposal.diagnostics.get("reason", proposal.contact_status))
                )
            fresh(proposal.computed_at, proposal.computed_at, now, 0.25)
            if proposal.solve_duration_s > 0.1:
                raise ValueError("MPC deadline missed")
            if self.target and proposal.computed_at <= self.target.computed_at:
                raise ValueError("Duplicate or reordered proposal")
            self.check_limits(proposal.positions_rad)
            self.target = proposal
        except ValueError as error:
            self.stop(str(error))
            raise

    def tick(self, now: float, feedback: JointFeedback, *, transport_ok: bool = True) -> tuple[float, ...] | None:
        """Return at most one bounded command; a delayed timer never emits catch-up bursts."""
        try:
            if not self.running or self.fault or not transport_ok or self.ownership.owner != self.owner:
                raise ValueError("Stopped, lost transport or command owner")
            fresh(feedback.source_timestamp, feedback.receive_timestamp, now, self.config["feedback_timeout"])
            self.check_limits(feedback.positions_rad)
            if feedback.source_timestamp < self.last_source:
                raise ValueError("Reordered feedback")
            self.last_source = feedback.source_timestamp
            dt = now - self.last_tick
            if not math.isfinite(dt) or dt < 0 or dt > 0.05 + 1e-9:
                raise ValueError("Command-rate stall or clock regression")
            if dt < 0.01 - 1e-9:
                return None
            if self.target is None:
                raise ValueError("No accepted target")
            fresh(self.target.computed_at, self.target.computed_at, now, 0.25)
            self.dropped_ticks += max(0, int((dt + 1e-9) / 0.01) - 1)
            output, velocities = [], []
            for q, target, old_v, speed, accel, step in zip(
                self.q, self.target.positions_rad, self.velocity, self.speed, self.acceleration, self.step
            ):
                distance = target - q
                # Discrete stopping envelope. Reversals decelerate under the same bound;
                # a target may lag or be overshot, but no unchecked new step is emitted.
                braking = max(0.0, math.sqrt(2 * accel * abs(distance) + (accel * dt / 2) ** 2) - accel * dt / 2)
                desired_v = math.copysign(min(speed, step / dt, abs(distance) / dt, braking), distance)
                velocity = max(old_v - accel * dt, min(old_v + accel * dt, desired_v))
                if abs(velocity) > speed + 1e-9 or abs(velocity * dt) > step + 1e-9:
                    raise ValueError("No feasible bounded command")
                velocities.append(velocity)
                output.append(q + velocity * dt)
            command = named_vector(JOINT_NAMES, output)
            self.check_limits(command)
            self.q, self.velocity, self.last_tick = command, tuple(velocities), now
            return command
        except ValueError as error:
            self.stop(str(error))
            raise
