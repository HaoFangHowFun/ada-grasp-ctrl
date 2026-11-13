"""MPC linear resampling and execution guards; no movement constraints or ROS imports."""

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
    """Resample accepted MPC targets over 100 ms, with explicit publish commit.

    Operational guards latch faults; position/speed/acceleration constraints belong
    to Ours and the downstream stack. No physical delivery is inferred from commit.
    """

    def __init__(
        self, *, config: dict | None = None, ownership: CommandOwnership | None = None, hardware_enabled: bool = False
    ) -> None:
        self.config = bridge_config(config)
        if self.config.get("side", "right") != "right" or (
            self.config["environment"] == "hardware" and not hardware_enabled
        ):
            raise ValueError("Streaming is right-side only; hardware requires explicit enable")
        if self.config["rate_hz"] != 100:
            raise ValueError("MPC transport requires 100 Hz")
        self.ownership = ownership or CommandOwnership()
        self.initial_q: tuple[float, ...] | None = None
        self.q_last_published: tuple[float, ...] | None = None
        self.q_start: tuple[float, ...] | None = None
        self.target: MPCProposal | None = None
        self.segment_started: float | None = None
        self.last_tick: float | None = None
        self.last_source: float | None = None
        self.last_now: float | None = None
        self.last_published_at: float | None = None
        self.pending: tuple[float, tuple[float, ...]] | None = None
        self.running = False
        self.fault: str | None = None
        self.dropped_ticks = 0

    @property
    def q(self) -> tuple[float, ...] | None:
        """Last locally published pair, or the measured seed before first send."""
        return self.initial_q if self.q_last_published is None else self.q_last_published

    def start(self, feedback: JointFeedback, now: float, *, owner: str = "mpc_mock") -> None:
        """Seed from fresh canonical measured joints; create no publication."""
        try:
            if self.running or self.fault:
                raise ValueError("Stream cannot be restarted")
            fresh(feedback.source_timestamp, feedback.receive_timestamp, now, self.config["feedback_timeout"])
            self.initial_q = named_vector(feedback.joint_names, feedback.positions_rad)
            self.ownership.claim(owner)
            self.owner, self.running = owner, True
            self.last_tick = self.last_now = now
            self.last_source = feedback.source_timestamp
        except ValueError as error:
            self.stop(str(error))
            raise

    def stop(self, reason: str) -> None:
        """Latch silence even if ownership has already been lost; never reclaim it."""
        self.fault = self.fault or reason
        self.pending = None
        if self.running and self.ownership.owner == self.owner:
            self.ownership.release(self.owner)
        self.running = False

    def _clock_owner(self, now: float) -> None:
        if not self.running or self.fault or self.ownership.owner != self.owner:
            raise ValueError("Stopped or lost command owner")
        if not math.isfinite(now) or now < self.last_now:
            raise ValueError("Stream clock regression")
        self.last_now = now

    def accept(self, proposal: MPCProposal, now: float) -> None:
        """Anchor a new segment at the last published pair, never an unsent sample."""
        try:
            self._clock_owner(now)
            if not proposal.valid or proposal.positions_rad is None:
                raise ValueError(
                    "Rejected MPC proposal: " + str(proposal.diagnostics.get("reason", proposal.contact_status))
                )
            named_vector(proposal.joint_names, proposal.positions_rad)
            fresh(proposal.computed_at, proposal.computed_at, now, 0.25)
            if not math.isfinite(proposal.solve_duration_s) or not 0 <= proposal.solve_duration_s <= 0.1:
                raise ValueError("MPC deadline missed")
            if self.target and proposal.computed_at <= self.target.computed_at:
                raise ValueError("Duplicate or reordered proposal")
            self.q_start, self.target, self.segment_started = self.q, proposal, now
            self.pending = None
        except ValueError as error:
            self.stop(str(error))
            raise

    def check(self, now: float, feedback: JointFeedback, *, transport_ok: bool = True) -> None:
        """Check execution health without generating or committing a command."""
        try:
            self._clock_owner(now)
            if not transport_ok:
                raise ValueError("Lost transport")
            named_vector(feedback.joint_names, feedback.positions_rad)
            fresh(feedback.source_timestamp, feedback.receive_timestamp, now, self.config["feedback_timeout"])
            if feedback.source_timestamp < self.last_source:
                raise ValueError("Reordered feedback")
            self.last_source = feedback.source_timestamp
            if now - self.last_tick > 0.05 + 1e-9:
                raise ValueError("Command-rate stall")
            if self.target is None:
                raise ValueError("No accepted target")
            fresh(self.target.computed_at, self.target.computed_at, now, 0.25)
        except ValueError as error:
            self.stop(str(error))
            raise

    def tick(self, now: float, feedback: JointFeedback, *, transport_ok: bool = True) -> tuple[float, ...] | None:
        """Generate at most one sample; only published() may advance the anchor."""
        self.check(now, feedback, transport_ok=transport_ok)
        try:
            if self.pending is not None:
                raise ValueError("Previous sample has not been published")
            dt = now - self.last_tick
            if dt < 0.01 - 1e-9:
                return None
            alpha = 1.0 if now >= self.segment_started + 0.1 else max(0.0, (now - self.segment_started) / 0.1)
            goal = self.target.positions_rad
            if alpha == 0:
                command = self.q_start
            elif alpha == 1:
                command = goal
            else:
                # Subtract from the nearer endpoint for same-sign values; use a
                # convex sum across zero to avoid overflow for finite extremes.
                def blend(a: float, b: float) -> float:
                    if a == b:
                        return a
                    if (a <= 0 <= b) or (b <= 0 <= a):
                        return (1 - alpha) * a + alpha * b
                    return a + alpha * (b - a) if alpha <= 0.5 else b - (1 - alpha) * (b - a)

                command = tuple(blend(a, b) for a, b in zip(self.q_start, goal))
            command = named_vector(JOINT_NAMES, command)
            self.pending = (now, command)
            return command
        except ValueError as error:
            self.stop(str(error))
            raise

    def published(self, command: tuple[float, ...], now: float) -> None:
        """Commit after BOTH local publisher calls returned; not physical delivery."""
        try:
            self._clock_owner(now)
            command = named_vector(JOINT_NAMES, command)
            if self.pending is None or command != self.pending[1] or now < self.pending[0]:
                raise ValueError("Publication does not match the pending command")
            sampled_at = self.pending[0]
            self.dropped_ticks += max(0, int((sampled_at - self.last_tick + 1e-9) / 0.01) - 1)
            self.q_last_published, self.last_tick = command, sampled_at
            self.last_published_at, self.pending = now, None
        except ValueError as error:
            self.stop(str(error))
            raise
