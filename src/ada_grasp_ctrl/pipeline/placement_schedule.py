"""Simulation-only extension of the shared linear schedule, with explicit release dwell."""

import argparse
from copy import deepcopy

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import side_names
from ada_grasp_ctrl.experiments.right_ros_sender import motion_duration
from .shared_timing import build_stage_schedule, _duration, _named, _policies

PHASES = (
    "above_object",
    "pregrasp",
    "grasp",
    "squeeze",
    "lift",
    "reorient",
    "transport",
    "place",
    "settle",
    "release_1",
    "release_2",
    "retreat_lift",
    "retreat",
)
PREVIOUS_PHASES = tuple(phase for phase in PHASES if phase != "retreat_lift")
LEGACY_PHASES = tuple(phase for phase in PREVIOUS_PHASES if phase != "reorient")
PHASE_SEQUENCES = (PHASES, PREVIOUS_PHASES, LEGACY_PHASES)
DEFAULT_PARAMETERS = dict(
    preplace_height_m=0.15,
    settle_s=0.5,
    release_1_s=0.5,
    release_2_s=1.0,
    retreat_distance_m=0.10,
    retreat_s=2.0,
)
RETREAT_LIFT_PARAMETERS = dict(retreat_lift_height_m=0.10, retreat_lift_s=2.0)


def parameters(overrides: dict, *, include_retreat_lift: bool = True) -> dict:
    """Resolve positive finite simulation parameters without changing motion policies."""
    defaults = {**DEFAULT_PARAMETERS, **(RETREAT_LIFT_PARAMETERS if include_retreat_lift else {})}
    if set(overrides) - set(defaults):
        raise ValueError("Unknown simulation.placement parameter")
    return {key: _duration(overrides.get(key, value), key) for key, value in defaults.items()}


def build_schedule(grasp: dict, start: dict, targets: dict, settings: dict) -> dict:
    """Preserve the original first five stages and append the winner's solved endpoints.

    Reorient/Transport/Place use the existing duration policy. Fixed-duration stages have
    no extra endpoint dwell; the explicit Settle stage owns the Place dwell.
    """
    phases = next((p for p in PHASE_SEQUENCES if set(targets) == set(p)), None)
    if phases is None or len(grasp["active_arms"]) != 1:
        raise ValueError("Placement requires the complete single-arm phase sequence")
    settings = parameters(settings, include_retreat_lift="retreat_lift" in phases)
    schedule = build_stage_schedule(grasp, start, {p: targets[p] for p in PHASES[:5]})
    side = schedule["active_arms"][0]
    names = list(sum(side_names(side), ()))
    held = schedule["stages"][-1]["arms"][side]["positions_rad"][6:]
    cursor = schedule["end_time"]
    previous = schedule["stages"][-1]["arms"][side]["positions_rad"]
    place = _named(targets["place"][side], side, "place")
    for phase in phases[5:]:
        if set(targets[phase]) != {side}:
            raise ValueError(f"{phase}: targets must match the single active arm")
        end = _named(targets[phase][side], side, phase)
        fingers = held
        if phase in ("release_1", "release_2", "retreat_lift", "retreat"):
            source = "grasp" if phase == "release_1" else "pregrasp"
            fingers = [grasp["targets"][side][source]["joint_positions_rad"][n] for n in names[6:]]
        if end[6:] != fingers:
            raise ValueError(f"{phase}: incorrect finger endpoint")
        if phase in ("settle", "release_1", "release_2") and end[:6] != place[:6]:
            raise ValueError(f"{phase}: must retain the exact Place arm")
        duration = (
            motion_duration(previous, end, argparse.Namespace(**schedule["policies_by_arm"][side]))
            if phase in ("reorient", "transport", "place")
            else settings[phase + "_s"]
        )
        duration = _duration(duration, phase + ".duration_s")
        # Place is immediately followed by the explicit 0.5s stationary Settle.
        dwell = schedule["settle_time_s"] if phase in ("reorient", "transport") else 0.0
        schedule["stages"].append(
            dict(
                phase=phase,
                stage_start_time=cursor,
                stage_end_time=cursor + duration,
                settled_time=cursor + duration + dwell,
                duration_s=duration,
                required_durations_s={side: duration},
                arms={side: dict(joint_names=names, start_positions_rad=previous, positions_rad=end)},
            )
        )
        previous, cursor = end, cursor + duration + dwell
    schedule.update(
        kind="ada_simulation_placement_schedule",
        parameters=settings,
        end_time=cursor,
        timeline="timed_simulation_reference",
        completion_policy="timed_with_explicit_dwell",
    )
    schedule.pop("hold_s")
    return schedule


def validate_schedule(schedule: dict) -> None:
    """Rebuild and compare the simulation variant; leave the hardware validator unchanged."""
    if (
        not isinstance(schedule, dict)
        or type(schedule.get("schema_version")) is not int
        or schedule["schema_version"] != 1
    ):
        raise ValueError("placement schedule.schema_version: expected 1")
    try:
        if tuple(s["phase"] for s in schedule["stages"]) not in PHASE_SEQUENCES:
            raise ValueError("Incorrect grasp-to-place phase sequence")
        side = schedule["active_arms"][0]
        names = list(sum(side_names(side), ()))
        start = {side: dict(zip(names, schedule["stages"][0]["arms"][side]["start_positions_rad"]))}
        targets = {s["phase"]: {side: dict(zip(names, s["arms"][side]["positions_rad"]))} for s in schedule["stages"]}
        expected = build_schedule(schedule["plan"], start, targets, schedule["parameters"])
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Malformed placement schedule: {error}") from error
    if deepcopy(schedule) != expected:
        raise ValueError("Placement schedule differs from its endpoints or timing policy")


def planned_stage(
    side: str | list[str],
    phase: str,
    path: list,
    cursor: float,
    *,
    fixed_duration: float | None = None,
    dwell: float = 0.0,
) -> dict:
    """Project a checked full-configuration path onto the existing linear reference.

    Each edge retains the existing motion-duration floor and velocity policy.
    The full frozen state is recorded for audit; passive commands remain unchanged.
    """
    from ada_grasp_ctrl.robot_environment.model import side_slice

    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 56 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected a finite full-configuration path")
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    sides = [side] if isinstance(side, str) else list(side)
    policies = _policies(None, sides)
    durations = [
        max(
            _duration(
                motion_duration(a[side_slice(s)], b[side_slice(s)], argparse.Namespace(**policies[s])), "edge duration"
            )
            for s in sides
        )
        for a, b in zip(points, points[1:])
    ]
    if len(sides) == 2 and phase in ("lift", "transport", "place"):
        # Dense object-path samples are one motion, not separately stopped moves.
        # Apply the existing duration policy once to accumulated joint travel and
        # allocate a common clock using the larger arm increment on each edge.
        travel = np.sum(np.abs(np.diff(points, axis=0)), axis=0)
        total = max(
            motion_duration(np.zeros(28), travel[side_slice(s)], argparse.Namespace(**policies[s])) for s in sides
        )
        weights = np.max(np.abs(np.diff(points, axis=0)), axis=1)
        weights = np.maximum(weights, 1e-12)
        durations = (total * weights / weights.sum()).tolist()
    if fixed_duration is not None:
        if len(durations) != 1:
            raise ValueError("Fixed duration is only for existing settle/release transitions")
        durations = [_duration(fixed_duration, phase)]
    duration = _duration(sum(durations), phase)
    return dict(
        phase=phase,
        stage_start_time=cursor,
        stage_end_time=cursor + duration,
        settled_time=cursor + duration + dwell,
        duration_s=duration,
        required_durations_s={s: duration for s in sides},
        arms={
            s: dict(
                joint_names=list(sum(side_names(s), ())),
                start_positions_rad=points[0, side_slice(s)].tolist(),
                positions_rad=points[-1, side_slice(s)].tolist(),
            )
            for s in sides
        },
        waypoints_rad=[{s: point[side_slice(s)].tolist() for s in sides} for point in points[1:]],
        segment_durations_s=durations,
        fixed_q56=points[0].tolist(),
    )


def validate_rrt_schedule(plan: dict) -> None:
    """Validate the reviewed prefix without claiming future actual-state paths exist."""
    from ada_grasp_ctrl.robot_environment.model import side_slice
    from ada_grasp_ctrl.robot_environment.kinematics import rigid

    schedule, sides = plan["schedule"], plan["active_arms"]
    if sides not in (["left"], ["right"], ["left", "right"]) or schedule["active_arms"] != sides:
        raise ValueError("RRT placement requires canonical active arms")
    cursor = 0.0
    stages = schedule["stages"]
    phases = [s["phase"] for s in stages]
    if phases != ["pregrasp", "pregrasp_fingers", "grasp", "squeeze"]:
        raise ValueError("RRT review requires approach and nominal grasp/squeeze only")
    if (
        plan["pending_segments"]
        not in (["lift", "carry", "place", "retreat"], ["lift", "carry", "retreat"], ["carry", "retreat"])
        or plan["execution_phases"] != phases
    ):
        raise ValueError("Future paths must remain pending actual-state planning")
    if plan.get("stage_completion") != "timed":
        raise ValueError("RRT placement retains timed execution")
    expected_object = rigid(plan["target"]["Delta_T"]) @ rigid(plan["source_object_pose"])
    if (
        not np.allclose(rigid(plan["target_object_pose"]), expected_object, atol=1e-12, rtol=0)
        or plan["target_revision"] != plan["target"]["target_revision"]
        or plan["target_sha256"] != plan["target"]["target_sha256"]
    ):
        raise ValueError("Confirmed object target changed")
    previous = np.asarray(plan["initial_q56"])
    blocks = [side_slice(s) for s in sides]
    for stage in stages:
        fixed = np.asarray(stage["fixed_q56"])
        if not np.array_equal(fixed, previous):
            raise ValueError("Reviewed path discontinuity or frozen joints changed")
        points = [fixed.copy()]
        for row in stage["waypoints_rad"]:
            q = fixed.copy()
            if set(row) != set(sides):
                raise ValueError("Waypoint active arm changed")
            for side in sides:
                q[side_slice(side)] = row[side]
            points.append(q)
        dwell = schedule["settle_time_s"]
        expected = planned_stage(sides, stage["phase"], points, cursor, dwell=dwell)
        if expected != stage:
            raise ValueError("Reviewed path timing or endpoints changed")
        if stage["phase"] == "pregrasp" and any(
            not np.array_equal(q[block][6:], fixed[block][6:]) for q in points for block in blocks
        ):
            raise ValueError("Approach must hold fingers fixed")
        if stage["phase"] in ("pregrasp_fingers", "grasp", "squeeze"):
            endpoint = "pregrasp" if stage["phase"] == "pregrasp_fingers" else stage["phase"]
            if points[-1].tolist() != plan["nominal_endpoints"][endpoint]:
                raise ValueError("Selected grasp endpoints changed")
        if stage["phase"] == "pregrasp":
            if any(
                not np.array_equal(points[-1][block][:6], np.asarray(plan["nominal_endpoints"]["pregrasp"])[block][:6])
                for block in blocks
            ):
                raise ValueError("Selected pregrasp arm endpoint changed")
        previous, cursor = points[-1], stage["settled_time"]
    if cursor != schedule["end_time"]:
        raise ValueError("Reviewed path clock changed")
