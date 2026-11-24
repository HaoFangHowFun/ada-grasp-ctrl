"""One clock and one completion decision for paired, bounded non-MPC references."""

from __future__ import annotations

import math
from typing import Callable

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, finite_vector, side_names
from ada_grasp_ctrl.mpc.baseline_stream import BaselineStream
from ada_grasp_ctrl.mpc.contracts import JointFeedback, MPCProposal, fresh
from ada_grasp_ctrl.pipeline.plan import check_start
from .motion import arrival_reached
from .shared_packet import SIDES, number, receiver_config, validate_shared_packet, stage_reference


class SharedStageStream:
    """Prepare complete paired frames before publication; any fault latches combined silence.

    BaselineStream is reused as an unchanged 28-coordinate numerical guard. Its
    legacy right-name template is internal only: side names and model limits are
    checked here before adaptation, and every emitted name uses the actual side.
    No MPC solver, independent side timer, ROS publisher or driver lives here.
    """

    def __init__(self, packet: dict, limits: dict, event: Callable[..., None] | None = None) -> None:
        if packet.get("environment") == "hardware":
            from .shared_hardware import validate_hardware_packet

            self.packet = validate_hardware_packet(packet)
        else:
            self.packet = validate_shared_packet(packet)
        self.arms = self.packet["plan_identity"]["active_arms"]
        self.schedule = self.packet["schedule"]
        self.configs = {side: receiver_config(self.packet, side) for side in SIDES}
        self.event = event or (lambda kind, **data: None)
        if set(limits) != set(SIDES):
            raise ValueError("Both side model limits are required")
        self.limits = {}
        for side in SIDES:
            if limits[side]["joint_names"] != list(sum(side_names(side), ())):
                raise ValueError(f"{side}: model joint order differs")
            self.limits[side] = {
                key: finite_vector(limits[side][key], 28, f"{side}.{key}") for key in ("lower", "upper", "velocity")
            }
            if any(lo >= hi or v <= 0 for lo, hi, v in zip(*self.limits[side].values())):
                raise ValueError(f"{side}: invalid model limits")
        self.index = 0
        self.guards = {}
        self.fault = None
        self.done = False
        self.last_tick = None
        self.sources = {}
        self.settled = None
        # Reject every endpoint before creating a command publisher.
        for side in SIDES:
            self._limits(side, self.packet["snapshot"][side]["positions_rad"])
        for stage in self.schedule["stages"]:
            for side, target in stage["arms"].items():
                self._limits(side, target["positions_rad"])
                previous = target["start_positions_rad"]
                for point, duration in zip(stage.get("waypoints_rad", []), stage.get("segment_durations_s", [])):
                    self._limits(side, point[side])
                    if duration + 1e-6 < self._minimum(side, previous, point[side]):
                        raise ValueError(f"lift.{side}: waypoint violates model duration limits")
                    previous = point[side]
                if stage["duration_s"] + 1e-6 < self._minimum(
                    side, target["start_positions_rad"], target["positions_rad"]
                ):
                    raise ValueError(f"{stage['phase']}.{side}: schedule violates model duration limits")

    def _limits(self, side: str, positions: list[float]) -> None:
        values = finite_vector(positions, 28, side)
        limits = self.limits[side]
        if any(not lo - 1e-9 <= q <= hi + 1e-9 for q, lo, hi in zip(values, limits["lower"], limits["upper"])):
            raise ValueError(f"{side}: joint-limit violation")

    def _minimum(self, side: str, start: list[float], end: list[float], *, include_hand: bool = True) -> float:
        """Use the receiver's existing quintic-derived minimum, including model speed caps."""
        c, times = self.configs[side], []
        for group, sl in (("arm", slice(0, 6)), ("hand", slice(6, 28))):
            if group == "hand" and not include_hand:
                continue
            delta = max(abs(a - b) for a, b in zip(start[sl], end[sl]))
            speed = min(c[group + "_speed"], min(self.limits[side]["velocity"][sl]))
            times.extend([1.875 * delta / speed, math.sqrt(5.774 * delta / c[group + "_acceleration"])])
        return max(times)

    def _feedback(self, feedback: dict, now: float) -> dict[str, list[float]]:
        if set(feedback) != set(SIDES):
            raise ValueError("Both side feedback streams are required")
        actual = {}
        for side in SIDES:
            sample = feedback[side]
            if sample.get("fault"):
                raise ValueError(f"{side}: {sample['fault']}")
            if sample["joint_names"] != list(sum(side_names(side), ())):
                raise ValueError(f"{side}: feedback routing/names differ")
            source = number(sample["source_timestamp"], f"{side}.source_timestamp")
            received = number(sample["receive_timestamp"], f"{side}.receive_timestamp")
            fresh(source, received, now, self.configs[side]["feedback_timeout"])
            if source < self.sources.get(side, source):
                raise ValueError(f"{side}: reordered feedback")
            self.sources[side] = source
            actual[side] = finite_vector(sample["positions_rad"], 28, f"{side}.feedback")
            self._limits(side, actual[side])
            if side not in self.arms:
                check_start(actual[side], self.packet["snapshot"][side]["positions_rad"])
        return actual

    @staticmethod
    def _numeric_feedback(sample: dict, side: str) -> JointFeedback:
        return JointFeedback(
            tuple(JOINT_NAMES),
            tuple(sample["positions_rad"]),
            sample["source_timestamp"],
            sample["receive_timestamp"],
            f"validated_{side}_coordinates",
        )

    def start(self, feedback: dict, now: float) -> None:
        """Bind measured starts and model limits before the first paired frame."""
        if self.last_tick is not None or self.fault or self.done:
            raise ValueError("Shared stream cannot restart")
        try:
            now = number(now, "now")
            actual = self._feedback(feedback, now)
            for side in SIDES:
                check_start(actual[side], self.packet["snapshot"][side]["positions_rad"])
            self._begin(actual, feedback, now)
        except Exception as error:
            self.abort(str(error))
            raise

    def _begin(self, actual: dict, feedback: dict, now: float) -> None:
        stage = self.schedule["stages"][self.index]
        duration = stage["duration_s"]
        for side in self.arms:
            check_start(actual[side], stage["arms"][side]["start_positions_rad"], ignore_hand_error=self.index > 0)
            if duration + 1e-6 < self._minimum(
                side, actual[side], stage["arms"][side]["positions_rad"], include_hand=False
            ):
                raise ValueError(f"{side}: fresh arm state requires a longer reviewed duration")
            # Preserve legacy nonblocking-hand policy: refresh starts and extend one
            # shared duration if contact left fingers away from their previous targets.
            duration = max(duration, self._minimum(side, actual[side], stage["arms"][side]["positions_rad"]))
            if "waypoints_rad" in stage:
                first = self._minimum(side, actual[side], stage["waypoints_rad"][0][side])
                duration = max(duration, first * stage["duration_s"] / stage["segment_durations_s"][0])
        if duration > 300:
            raise ValueError("Fresh state requires more than the existing 300 second bound")
        self.starts = {side: actual[side].copy() for side in self.arms}
        for side in self.arms:
            limits = self.limits[side]
            config = {
                k: v
                for k, v in self.configs[side].items()
                if k not in ("side", "active_arms", "tip_frame", "mode", "path_verified")
            }
            guard = BaselineStream(
                tuple(limits["lower"]),
                tuple(limits["upper"]),
                tuple(limits["velocity"]),
                config=config,
                hardware_enabled=self.packet["environment"] == "hardware",
            )
            guard.start(self._numeric_feedback(feedback[side], side), now, owner="baseline")
            self.guards[side] = guard
        self.began = self.last_tick = now
        self.duration, self.settled = duration, None
        self.event(
            "stage_started",
            phase=stage["phase"],
            index=self.index,
            stage_start_time=now,
            reviewed_duration_s=stage["duration_s"],
            duration_s=duration,
            active_arms=self.arms,
        )

    def abort(self, reason: str) -> None:
        """Latch whole-stage silence; the ROS owner requests existing bounded arm stops."""
        if self.fault is None:
            self.fault = reason
            for guard in self.guards.values():
                guard.stop(reason)
            self.event("shared_abort", index=self.index, reason=reason)

    def tick(self, now: float, feedback: dict, *, transport_ok: bool = True) -> dict | None:
        """Prepare all sides atomically at 100 Hz; emit no partial frame on a guard failure.

        Returns:
            One paired command/audit frame, or None before the next cadence tick or after completion.

        Raises:
            ValueError: On any side's fault, stale feedback, unsafe output or shared timeout.
        """
        if self.fault:
            raise ValueError(self.fault)
        if self.done:
            return None
        try:
            now = number(now, "now")
            if self.last_tick is None or not transport_ok:
                raise ValueError("Stream not started or command routing/ownership failed")
            actual = self._feedback(feedback, now)
            dt = now - self.last_tick
            if not 0 <= dt <= 0.05 + 1e-9:
                raise ValueError("Command-rate stall or clock regression")
            if dt < 0.01 - 1e-9:
                return None
            stage = self.schedule["stages"][self.index]
            elapsed = now - self.began
            progress = min(1.0, elapsed / self.duration)
            commands, references, errors = {}, {}, {}
            for side in self.arms:
                target = stage["arms"][side]["positions_rad"]
                desired = stage_reference(stage, side, self.starts[side], elapsed, self.duration)
                guard = self.guards[side]
                previous, velocity = guard.q, guard.velocity
                guard.accept(MPCProposal(tuple(JOINT_NAMES), tuple(desired), True, now, 0, "linear"), now)
                bounded = guard.tick(now, self._numeric_feedback(feedback[side], side))
                if bounded is None:
                    raise ValueError("Side cadence differs from common stage cadence")
                self._limits(side, list(bounded))
                for q, old, old_v, speed, acceleration, step in zip(
                    bounded, previous, velocity, guard.speed, guard.acceleration, guard.step
                ):
                    v = (q - old) / dt
                    if abs(v) > speed + 1e-8 or abs(v - old_v) > acceleration * dt + 1e-8 or abs(q - old) > step + 1e-8:
                        raise ValueError(f"{side}: bounded output violates speed/acceleration/step limits")
                commands[side], references[side] = list(bounded), desired
                errors[side] = [t - q for t, q in zip(target, actual[side])]
            stationary = self.packet["snapshot"]
            arm_positions = [(commands[s] if s in self.arms else stationary[s]["positions_rad"])[:6] for s in SIDES]
            frame = dict(
                phase=stage["phase"],
                index=self.index,
                monotonic_time=now,
                stage_start_time=self.began,
                stage_end_time=self.began + self.duration,
                progress=progress,
                duration_s=self.duration,
                arm=dict(
                    joint_names=list(side_names("left")[0] + side_names("right")[0]),
                    positions_rad=sum(arm_positions, []),
                ),
                hands={s: dict(joint_names=list(side_names(s)[1]), positions_rad=commands[s][6:]) for s in self.arms},
                reference=references,
                commands=commands,
                feedback=feedback,
                target_error_rad=errors,
            )
            self.last_tick = now
            reached = elapsed >= self.duration and all(
                arrival_reached(errors[s], self.configs[s])
                and max(abs(q - t) for q, t in zip(commands[s], stage["arms"][s]["positions_rad"])) < 1e-8
                for s in self.arms
            )
            self.settled = (now if self.settled is None else self.settled) if reached else None
            self.event("prepared_frame", **frame)
            if (
                self.settled is not None
                and now - self.settled >= self.schedule["settle_time_s"]
                and all(feedback[s]["source_timestamp"] > self.settled for s in SIDES)
            ):
                self.event(
                    "stage_completed",
                    phase=stage["phase"],
                    index=self.index,
                    monotonic_time=now,
                    target_error_rad=errors,
                    hand_arrival_required=False,
                )
                for guard in self.guards.values():
                    guard.stop("stage_complete")
                self.index += 1
                if self.index == len(self.schedule["stages"]):
                    self.done = True
                else:
                    self._begin(actual, feedback, now)
            elif elapsed > self.duration + self.configs[self.arms[0]]["settle_timeout"]:
                raise ValueError("Shared stage arrival timeout")
            return frame
        except Exception as error:
            self.abort(str(error))
            raise
