"""Offline shared linear references for explicit, already-solved joint endpoints.

Durations retain the existing conservative sender policy. Linear endpoint velocity
jumps are not acceleration-limited by time scaling; receiver protections still apply.
The schedule is nominal data, never permission to advance an execution stage.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from ada_grasp_ctrl.experiments.right_ros_contract import PHASE_SEQUENCES, CYCLE_PHASES, finite_vector, side_names
from ada_grasp_ctrl.experiments.right_ros_sender import motion_duration
from ada_grasp_ctrl.grasp_selection.contracts import GraspPlan, validate_grasp_plan
from ada_grasp_ctrl.mpc.reference import linear_reference
from ada_grasp_ctrl.ros_execution.settings import DEFAULTS

POLICY_KEYS = ("arm_speed", "hand_speed", "arm_acceleration", "hand_acceleration")


def _number(value: object, field: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(f"{field}: expected a {'positive ' if positive else ''}finite number")
    return float(value)


def _duration(value: object, field: str) -> float:
    result = _number(value, field, positive=True)
    if result > 300:
        raise ValueError(f"{field}: exceeds the existing 300 second duration bound")
    return result


def _named(value: object, side: str, field: str) -> list[float]:
    names = sum(side_names(side), ())
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{field}: expected exactly the named 6 CRX + 22 Sharpa joints for {side}")
    return finite_vector([value[n] for n in names], 28, field)


def _policies(value: object, arms: list[str]) -> dict[str, dict[str, float]]:
    if value is None:
        value = dict.fromkeys(arms, {})
    if not isinstance(value, dict) or set(value) != set(arms):
        raise ValueError("policies_by_arm must exactly match active_arms")
    policies = {}
    for side in arms:
        overrides = value[side]
        if not isinstance(overrides, dict) or set(overrides) - {*POLICY_KEYS, "minimum_duration"}:
            raise ValueError(f"policies_by_arm.{side}: unknown duration policy fields")
        policy = {
            key: _number(overrides.get(key, DEFAULTS[key]), f"{side}.{key}", positive=True) for key in POLICY_KEYS
        }
        if any(policy[key] > DEFAULTS[key] for key in POLICY_KEYS):
            raise ValueError(f"policies_by_arm.{side}: cannot increase existing speed/acceleration limits")
        policy["minimum_duration"] = _duration(overrides.get("minimum_duration", 2.0), f"{side}.minimum_duration")
        if policy["minimum_duration"] < 2.0:
            raise ValueError(f"{side}.minimum_duration: cannot reduce the existing 2 second floor")
        policies[side] = policy
    return policies


def build_stage_schedule(
    plan: GraspPlan,
    start_positions_rad: dict[str, dict[str, float]],
    stage_targets_rad: dict[str, dict[str, dict[str, float]]],
    *,
    policies_by_arm: dict[str, dict[str, float]] | None = None,
    stage_start_time: float = 0.0,
    settle_time_s: float = DEFAULTS["settle_time"],
    hold_s: float = 5.0,
) -> dict[str, Any]:
    """Build a shared stage timeline using all commanded joints, without solving IK.

    Args:
        plan: Validated Task 1 grasp data; only its active arms participate.
        start_positions_rad: Measured named joints by side; a known passive side is ignored.
        stage_targets_rad: Already-solved named joints, keyed by established phase then
            active side. Required phases are pregrasp/grasp/squeeze; optional above_object,
            lift or the full hold/release/return cycle follow the existing phase sequences.
        policies_by_arm: Optional stricter per-active-side sender duration policies.
        stage_start_time: Nominal shared start on one caller-owned clock, in seconds.
        settle_time_s: Existing receiver settle interval between phases, also after the last.
        hold_s: Stationary hold duration; retains the existing exception to movement floors.

    Returns:
        A schema-1 JSON-compatible schedule with shared start/end timestamps, ordered
        joint vectors and each side's duration requirement. All readiness flags remain false.

    Raises:
        ValueError: On malformed plans, names, phases, policy violations, mismatched selected
            fingers, invalid cycle endpoints or a duration exceeding the existing 300s bound.
    """
    grasp = validate_grasp_plan(plan)
    arms = list(grasp["active_arms"])
    policies = _policies(policies_by_arm, arms)
    began = _number(stage_start_time, "stage_start_time")
    if began < 0:
        raise ValueError("stage_start_time must be nonnegative")
    settle = _duration(settle_time_s, "settle_time_s")
    hold = _duration(hold_s, "hold_s")
    if (
        not isinstance(start_positions_rad, dict)
        or not set(arms) <= set(start_positions_rad)
        or set(start_positions_rad) - {"left", "right"}
    ):
        raise ValueError("start_positions_rad must include all active arms and only known sides")
    initial = {side: _named(start_positions_rad[side], side, f"start_positions_rad.{side}") for side in arms}
    if not isinstance(stage_targets_rad, dict):
        raise ValueError("stage_targets_rad: expected a phase-to-target object")
    phases = tuple(p for p in ("above_object", *CYCLE_PHASES) if p in stage_targets_rad)
    if set(phases) != set(stage_targets_rad) or phases not in PHASE_SEQUENCES:
        raise ValueError("stage_targets_rad: expected an established phase sequence")
    previous, cursor, stages = initial, began, []
    for phase in phases:
        targets = stage_targets_rad[phase]
        if not isinstance(targets, dict) or set(targets) != set(arms):
            raise ValueError(f"{phase}: targets must exactly match active_arms")
        positions = {side: _named(targets[side], side, f"{phase}.{side}") for side in arms}
        requirements = {}
        for side in arms:
            start, end = previous[side], positions[side]
            if phase in ("pregrasp", "grasp", "squeeze"):
                fingers = grasp["targets"][side][phase]["joint_positions_rad"]
                if end[6:] != [fingers[name] for name in side_names(side)[1]]:
                    raise ValueError(f"{phase}.{side}: finger endpoints differ from the selected GraspPlan")
            if phase == "hold" and end != start:
                raise ValueError(f"hold.{side}: must retain all lift joints")
            if phase == "release" and end != start[:6] + [0.0] * 22:
                raise ValueError(f"release.{side}: must retain lifted arm and open fingers to zero")
            if phase == "return" and end != initial[side][:6] + [0.0] * 22:
                raise ValueError(f"return.{side}: must restore measured arm start with open fingers")
            if phase == "lift" and end[6:] != start[6:]:
                raise ValueError(f"lift.{side}: must preserve squeeze fingers")
            try:
                required = (
                    hold if phase == "hold" else motion_duration(start, end, argparse.Namespace(**policies[side]))
                )
            except (OverflowError, ValueError) as error:
                raise ValueError(f"{phase}.{side}: invalid computed duration") from error
            requirements[side] = _duration(required, f"{phase}.{side}.duration_s")
        duration = max(requirements.values())
        end_time, settled_time = cursor + duration, cursor + duration + settle
        if not math.isfinite(settled_time) or not cursor < end_time < settled_time:
            raise ValueError(f"{phase}: clock cannot represent the stage and settle intervals")
        stages.append(
            dict(
                phase=phase,
                stage_start_time=cursor,
                stage_end_time=end_time,
                settled_time=settled_time,
                duration_s=duration,
                required_durations_s=requirements,
                arms={
                    side: dict(
                        joint_names=list(sum(side_names(side), ())),
                        start_positions_rad=previous[side],
                        positions_rad=positions[side],
                    )
                    for side in arms
                },
            )
        )
        previous, cursor = positions, settled_time
    return dict(
        schema_version=1,
        kind="ada_shared_stage_schedule",
        plan=grasp,
        active_arms=arms,
        reference="linear",
        joint_unit="rad",
        stage_start_time=began,
        settle_time_s=settle,
        hold_s=hold,
        policies_by_arm=policies,
        stages=stages,
        end_time=cursor,
        timeline="nominal_until_preceding_stage_succeeds",
        completion_policy="retain_existing_arrival_settle_timeout_and_abort_checks",
        acceleration_continuous=False,
        continuous_path_verified=False,
        execution_ready=False,
        hardware_ready=False,
    )


def validate_stage_schedule(schedule: dict) -> dict[str, Any]:
    """Rebuild serialized timing data and reject altered clocks, vectors or policies.

    Returns:
        Independent schedule verified against its embedded plan and original duration policy.

    Raises:
        ValueError: If the schedule is malformed or inconsistent. This validates planning
            data only, not endpoint IK, continuous collision checks or execution readiness.
    """
    if (
        not isinstance(schedule, dict)
        or type(schedule.get("schema_version")) is not int
        or schedule["schema_version"] != 1
    ):
        raise ValueError("schedule.schema_version: expected 1")
    try:
        arms = schedule["active_arms"]
        stages = schedule["stages"]
        if not isinstance(stages, list) or not stages:
            raise ValueError("schedule.stages must be a nonempty list")
        starts, targets = {}, {}
        for index, stage in enumerate(stages):
            phase = stage["phase"]
            if phase in targets:
                raise ValueError("Duplicate scheduled phase")
            for field in ("stage_start_time", "stage_end_time", "settled_time", "duration_s"):
                _number(stage[field], f"{phase}.{field}")
            targets[phase] = {}
            for side in arms:
                data = stage["arms"][side]
                names = list(sum(side_names(side), ()))
                if data["joint_names"] != names:
                    raise ValueError(f"{phase}.{side}: noncanonical joint ordering")
                start = finite_vector(data["start_positions_rad"], 28, f"{phase}.{side}.start_positions_rad")
                end = finite_vector(data["positions_rad"], 28, f"{phase}.{side}.positions_rad")
                if index == 0:
                    starts[side] = dict(zip(names, start))
                targets[phase][side] = dict(zip(names, end))
        expected = build_stage_schedule(
            schedule["plan"],
            starts,
            targets,
            policies_by_arm=schedule["policies_by_arm"],
            stage_start_time=schedule["stage_start_time"],
            settle_time_s=schedule["settle_time_s"],
            hold_s=schedule["hold_s"],
        )
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Malformed stage schedule: {error}") from error
    if schedule != expected:
        raise ValueError("Schedule differs from its plan, joint endpoints or existing timing policy")
    return expected


def sample_stage(
    schedule: dict,
    phase: str,
    now: float,
    *,
    stage_start_time: float | None = None,
) -> dict[str, Any]:
    """Sample one shared linear progress value for all active arms, with endpoint clipping.

    Args:
        schedule: Serialized output from ``build_stage_schedule``.
        phase: Explicit phase chosen by the caller; no automatic execution transitions.
        now: Time on the same caller-owned clock as the start timestamp.
        stage_start_time: Optional common actual start after predecessor arrival succeeds.
            Rebase both arms together; a nominal clock never authorizes skipping arrival gates.

    Returns:
        Common phase/start/end/progress and named, canonically ordered per-arm positions.

    Raises:
        ValueError: On invalid schedule, phase or clock values.
    """
    checked = validate_stage_schedule(schedule)
    stage = next((s for s in checked["stages"] if s["phase"] == phase), None)
    if stage is None:
        raise ValueError(f"Phase is not scheduled: {phase}")
    now = _number(now, "now")
    began = stage["stage_start_time"] if stage_start_time is None else _number(stage_start_time, "stage_start_time")
    duration = stage["duration_s"]
    if began < 0 or not math.isfinite(began + duration) or began + duration <= began:
        raise ValueError("Invalid shared stage start time")
    progress = 0.0 if now <= began else 1.0 if now >= began + duration else (now - began) / duration
    references = {}
    for side, target in stage["arms"].items():
        start, end = target["start_positions_rad"], target["positions_rad"]
        positions = (
            start.copy()
            if progress == 0
            else end.copy()
            if progress == 1
            else linear_reference(start, end, now - began, duration)
        )
        references[side] = dict(joint_names=target["joint_names"], positions_rad=positions)
    return dict(
        phase=phase, stage_start_time=began, stage_end_time=began + duration, progress=progress, arms=references
    )
