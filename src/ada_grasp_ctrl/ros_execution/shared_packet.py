"""Explicit mock-only transport projection of a Task 3 schedule; standard library only."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from ada_grasp_ctrl.experiments.right_ros_contract import PHASE_SEQUENCES, finite_vector, side_names
from ada_grasp_ctrl.experiments.right_ros_sender import motion_duration
from .settings import DEFAULTS, bridge_config

KIND = "ada_shared_mock_joint_targets"
SIDES = ("left", "right")
LIMIT_KEYS = ("arm_speed", "hand_speed", "arm_acceleration", "hand_acceleration")


def waypoint_timing(starts: dict, waypoints: list[dict], policies: dict) -> tuple[list[float], dict]:
    """Time common lift segments using the existing slower-side motion policy."""
    if not isinstance(waypoints, list) or not 2 <= len(waypoints) <= 10:
        raise ValueError("Lift requires 2..10 common waypoints")
    previous = starts
    durations, requirements = [], dict.fromkeys(starts, 0.0)
    for point in waypoints:
        if set(point) != set(starts):
            raise ValueError("Lift waypoint arms differ")
        required = {}
        for side in starts:
            end = finite_vector(point[side], 28, f"lift.{side}.waypoint")
            if end[6:] != starts[side][6:]:
                raise ValueError("Lift waypoints must retain squeeze fingers")
            required[side] = motion_duration(previous[side], end, argparse.Namespace(**policies[side]))
            requirements[side] += required[side]
        durations.append(max(required.values()))
        previous = point
    return durations, requirements


def stage_reference(stage: dict, side: str, start: list[float], elapsed: float, duration: float) -> list[float]:
    """Sample one common stage clock, including paired lift translation waypoints."""
    from ada_grasp_ctrl.mpc.reference import linear_reference

    if "waypoints_rad" not in stage:
        return linear_reference(start, stage["arms"][side]["positions_rad"], elapsed, duration)
    cursor = max(0.0, elapsed) * stage["duration_s"] / duration
    previous = start
    for point, segment in zip(stage["waypoints_rad"], stage["segment_durations_s"]):
        if cursor <= segment:
            return linear_reference(previous, point[side], cursor, segment)
        cursor -= segment
        previous = point[side]
    return previous.copy()


def number(value: object, field: str, *, positive: bool = False) -> float:
    """Validate one finite clock, duration or policy number without bool coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field}: expected a finite number")
    if positive and value <= 0:
        raise ValueError(f"{field}: expected a positive number")
    return float(value)


def arm_list(value: object, field: str) -> list[str]:
    """Require explicit unique left/right names, never infer arms from availability."""
    if not isinstance(value, list) or not value or any(s not in SIDES for s in value) or len(set(value)) != len(value):
        raise ValueError(f"{field}: expected nonempty unique left/right arms")
    return value


def receiver_config(packet: dict, side: str) -> dict:
    """Retain existing runtime guards; hand tracking remains observable and nonblocking."""
    policy = packet["schedule"]["policies_by_arm"].get(side, {})
    return bridge_config(
        dict(
            side=side,
            active_arms=list(SIDES),
            environment=packet["environment"],
            hand_arrival_required=False,
            settle_time=packet["schedule"]["settle_time_s"],
            **{key: policy.get(key, DEFAULTS[key]) for key in LIMIT_KEYS},
        )
    )


def validate_shared_packet(packet: dict) -> dict:
    """Validate transport names, shared clocks, duration policy and explicit mock scope.

    The Ada exporter validates the complete GraspPlan and Task 3 schedule before
    producing this numeric transport projection. ROS does not import robot planners.

    Returns:
        Independent transport data with preserved source-plan identity and provenance.

    Raises:
        ValueError: On incompatible schema, hardware scope or inconsistent targets/timing.
    """
    return _validate_packet(packet, environment="mock", kind=KIND)


def _validate_packet(packet: dict, *, environment: str, kind: str) -> dict:
    """Common numeric transport contract; callers separately enforce their execution scope."""
    try:
        if (
            type(packet.get("schema_version")) is not int
            or packet["schema_version"] != 1
            or packet.get("kind") != kind
            or packet.get("environment") != environment
            or packet.get("hardware_ready") is not False
        ):
            raise ValueError(f"Expected explicit schema-1 {environment}-only shared packet")
        identity, schedule = packet["plan_identity"], packet["schedule"]
        if type(identity["schema_version"]) is not int or identity["schema_version"] != 1:
            raise ValueError("plan_identity.schema_version must be 1")
        for name in ("candidate_id", "mode"):
            if not isinstance(identity[name], str) or not identity[name].strip():
                raise ValueError(f"plan_identity.{name}: missing identity")
        if not isinstance(identity["provenance"], dict):
            raise ValueError("Missing selected candidate provenance")
        arms = arm_list(identity["active_arms"], "active_arms")
        available = arm_list(identity["available_arms"], "available_arms")
        if not set(arms) <= set(available) or schedule["active_arms"] != arms:
            raise ValueError("Plan/schedule arms disagree or an active arm is unavailable")
        if (
            type(schedule["schema_version"]) is not int
            or schedule["schema_version"] != 1
            or schedule["reference"] != "linear"
            or schedule["joint_unit"] != "rad"
        ):
            raise ValueError("Expected schema-1 linear schedule in radians")
        for field in ("execution_ready", "hardware_ready", "continuous_path_verified", "acceleration_continuous"):
            if schedule[field] is not False:
                raise ValueError(f"Schedule cannot claim {field}")
        if set(packet["snapshot"]) != set(SIDES) or set(packet["scene_kinematics"]) != set(SIDES):
            raise ValueError("Both measured sides and their model contracts are required")
        initial = {}
        for side in SIDES:
            data = packet["snapshot"][side]
            if data["joint_names"] != list(sum(side_names(side), ())):
                raise ValueError(f"snapshot.{side}: joint order differs")
            initial[side] = finite_vector(data["positions_rad"], 28, f"snapshot.{side}")
            if not packet["scene_kinematics"][side]:
                raise ValueError(f"Missing {side} kinematics")
        settle = number(schedule["settle_time_s"], "settle_time_s", positive=True)
        hold = number(schedule["hold_s"], "hold_s", positive=True)
        if max(settle, hold) > 300:
            raise ValueError("Invalid hold/settle duration")
        policies = schedule["policies_by_arm"]
        if set(policies) != set(arms):
            raise ValueError("Duration policies must exactly match active_arms")
        for side, policy in policies.items():
            if set(policy) != {*LIMIT_KEYS, "minimum_duration"}:
                raise ValueError(f"{side}: unknown or missing duration policy")
            for field in LIMIT_KEYS:
                if number(policy[field], field, positive=True) > DEFAULTS[field]:
                    raise ValueError("Cannot relax existing speed/acceleration limits")
            if not 2 <= number(policy["minimum_duration"], "minimum_duration") <= 300:
                raise ValueError("Existing minimum-duration floor is required")
        phases = tuple(s["phase"] for s in schedule["stages"])
        if phases not in PHASE_SEQUENCES:
            raise ValueError("Unknown, missing or reordered scheduled phases")
        cursor = number(schedule["stage_start_time"], "stage_start_time")
        if cursor < 0:
            raise ValueError("Negative stage start")
        previous = {side: initial[side] for side in arms}
        for stage in schedule["stages"]:
            phase = stage["phase"]
            if set(stage["arms"]) != set(arms) or set(stage["required_durations_s"]) != set(arms):
                raise ValueError(f"{phase}: every active side must have complete targets and timing")
            required = {}
            for side in arms:
                data = stage["arms"][side]
                if data["joint_names"] != list(sum(side_names(side), ())):
                    raise ValueError(f"{phase}.{side}: joint order differs")
                start = finite_vector(data["start_positions_rad"], 28, f"{phase}.{side}.start")
                end = finite_vector(data["positions_rad"], 28, f"{phase}.{side}.end")
                if start != previous[side]:
                    raise ValueError(f"{phase}.{side}: discontinuous start")
                if (phase == "hold" and end != start) or (phase == "lift" and end[6:] != start[6:]):
                    raise ValueError(f"{phase}.{side}: changed retained joints")
                if phase == "release" and end != start[:6] + [0.0] * 22:
                    raise ValueError(f"{phase}.{side}: invalid release")
                if phase == "return" and end != initial[side][:6] + [0.0] * 22:
                    raise ValueError(f"{phase}.{side}: invalid return")
                required[side] = (
                    hold if phase == "hold" else motion_duration(start, end, argparse.Namespace(**policies[side]))
                )
                if "waypoints_rad" not in stage and (
                    number(stage["required_durations_s"][side], f"{phase}.{side}.duration", positive=True)
                    != required[side]
                ):
                    raise ValueError(f"{phase}.{side}: duration policy differs")
                previous[side] = end
            duration = number(stage["duration_s"], f"{phase}.duration", positive=True)
            expected_duration = max(required.values())
            if "waypoints_rad" in stage:
                if phase != "lift":
                    raise ValueError("Only lift supports translation waypoints")
                starts = {s: stage["arms"][s]["start_positions_rad"] for s in arms}
                segments, waypoint_required = waypoint_timing(starts, stage["waypoints_rad"], policies)
                if (
                    stage["segment_durations_s"] != segments
                    or stage["waypoints_rad"][-1] != previous
                    or stage["required_durations_s"] != waypoint_required
                ):
                    raise ValueError("Lift waypoint endpoint/timing differs")
                expected_duration = sum(segments)
            if duration != expected_duration or duration > 300:
                raise ValueError(f"{phase}: invalid common duration")
            if (
                number(stage["stage_start_time"], phase) != cursor
                or number(stage["stage_end_time"], phase) != cursor + duration
                or number(stage["settled_time"], phase) != cursor + duration + settle
            ):
                raise ValueError(f"{phase}: inconsistent shared stage clock")
            cursor = stage["settled_time"]
        if number(schedule["end_time"], "end_time") != cursor:
            raise ValueError("Schedule end time differs")
        json.dumps(packet, allow_nan=False)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError(f"Malformed shared packet: {error}") from error
    return deepcopy(packet)


def load_shared_packet(path: Path) -> dict:
    """Read strict JSON without changing or auto-upgrading legacy packet schemas."""

    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")

    return validate_shared_packet(json.loads(path.read_text(), parse_constant=reject))
