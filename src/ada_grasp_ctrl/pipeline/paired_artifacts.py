"""Native paired selection to the existing synchronized schedule and transport core."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import side_names
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.grasp_selection.contracts import plan_from_candidate
from ada_grasp_ctrl.grasp_selection.multi_arm import forbidden_contacts, solve_paired
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.robot_environment.kinematics import residual
from ada_grasp_ctrl.ros_execution.settings import DEFAULTS
from .approach import above_object_target
from .dual_scene import SideAdapter
from .shared_timing import build_stage_schedule
from .shared_artifacts import shared_mock_packet, shared_hardware_packet

SIDES = ("left", "right")


def named_state(q: np.ndarray) -> dict:
    """Project a canonical combined configuration into the existing per-side named ABI."""
    return {s: dict(zip(sum(side_names(s), ()), q[side_slice(s)].tolist())) for s in SIDES}


def export_paired_selection(
    robot: object,
    sim: object,
    selection: dict,
    state_path: Path,
    object_root: Path,
    config: dict,
    output: Path,
    candidates_path: Path,
) -> Path:
    """Build and check shared endpoints and sampled paths; never authorize physical movement.

    Lift uses five common world-Z translations with retained squeeze fingers. Static
    samples are diagnostics of the nominal joint reference, not continuous path certification.
    """
    chosen = selection["selected"]
    if chosen["required_arms"] != list(SIDES):
        raise ValueError("Paired export requires both arms")
    state = read_json(state_path)
    current = robot.configuration(state["full_positions_rad"])
    stages = {p: robot.configuration(chosen["stages"][p]["q56"]) for p in ("pregrasp", "grasp", "squeeze")}
    with np.load(object_root / "object.npz", allow_pickle=False) as archive:
        points = archive["points_world_m"]
    above = stages["pregrasp"].copy()
    above_checks = {}
    for side in SIDES:
        block = side_slice(side)
        adapter = SideAdapter(robot, sim, stages["pregrasp"], side)
        result = above_object_target(
            adapter,
            stages["pregrasp"][block],
            points,
            config["above_object_clearance_m"],
            grasp=stages["grasp"][block],
            open_hand=config["above_object_open_hand"],
            hand_extrapolation_factor=config["above_object_hand_factor"],
        )
        above[block] = result["positions"]
        above_checks[side] = result
    squeeze = stages["squeeze"]
    poses = {s: robot.fk(squeeze, s) for s in SIDES}
    relative = np.linalg.inv(poses["left"]) @ poses["right"]
    height = config["lift_height_m"]
    if isinstance(height, bool) or not np.isfinite(height) or height <= 0:
        raise ValueError("Lift height must be finite and positive")
    lift, lift_checks, q = [], [], squeeze.copy()
    for fraction in np.linspace(0.2, 1.0, 5):
        targets = {s: poses[s].copy() for s in SIDES}
        for target in targets.values():
            target[2, 3] += height * fraction
        q, ik = solve_paired(robot, q, targets)
        actual_relative = np.linalg.inv(robot.fk(q, "left")) @ robot.fk(q, "right")
        error = residual(actual_relative, relative)
        if error["position_m"] > 1e-3 or error["orientation_rad"] > 1e-2:
            raise ValueError(f"Lift relative wrist residual exceeds existing IK tolerance: {error}")
        lift.append(q.copy())
        lift_checks.append(
            dict(
                fraction=float(fraction),
                ik=ik,
                relative_residual=error,
                targets={s: t.tolist() for s, t in targets.items()},
                q56=q.tolist(),
            )
        )
    opened = lift[-1].copy()
    returned = current.copy()
    for side in SIDES:
        block = side_slice(side)
        opened[block.start + 6 : block.stop] = 0
        returned[block.start + 6 : block.stop] = 0
    endpoints = dict(above_object=above, **stages, lift=lift[-1], hold=lift[-1], release=opened, **{"return": returned})
    checks, previous = [], current
    for phase, endpoint in endpoints.items():
        # Sample nominal segments at the existing arm/hand command-step resolutions.
        segments = lift if phase == "lift" else [endpoint]
        count = 0
        for end in segments:
            end = robot.configuration(end)
            arm_delta = max(np.max(np.abs(end[side_slice(s, True)] - previous[side_slice(s, True)])) for s in SIDES)
            hand_delta = max(np.max(np.abs(end[side_slice(s)][6:] - previous[side_slice(s)][6:])) for s in SIDES)
            samples = max(
                2, int(np.ceil(max(arm_delta / DEFAULTS["arm_step"], hand_delta / DEFAULTS["hand_step"]))) + 1
            )
            for alpha in np.linspace(0, 1, samples):
                sample = robot.configuration(previous + alpha * (end - previous))
                contact_phase = phase if phase in ("grasp", "squeeze", "lift", "hold") else "pregrasp"
                bad = forbidden_contacts(sim.check_static_contacts(sample), list(SIDES), phase=contact_phase)
                if bad:
                    raise ValueError(f"Paired {phase} nominal path collision at {alpha:.3f}: {bad}")
                count += 1
            previous = end
        checks.append(dict(phase=phase, sampled_configurations=count, forbidden_contacts=[]))
    plan = plan_from_candidate(chosen["candidate"], available_arms=selection["enabled_arms"])
    schedule = build_stage_schedule(
        plan,
        named_state(current),
        {p: named_state(q) for p, q in endpoints.items()},
        hold_s=config["hold_s"],
        lift_waypoints_rad=[{s: q[side_slice(s)].tolist() for s in SIDES} for q in lift],
    )
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "motion_plan.json"
    write_json(plan_path, schedule)
    write_json(
        output / "path_checks.json",
        dict(
            above_object=above_checks,
            lift=lift_checks,
            phases=checks,
            continuous_path_verified=False,
            physical_execution_verified=False,
            geometry_note="Measured table and robots; partial object cloud is not a collision solid",
            relative_wrist_note=(
                "Constrained at common IK waypoints; interpolation and driver tracking are not rigid-object control"
            ),
        ),
    )
    paths = [
        plan_path,
        output / "path_checks.json",
        state_path,
        candidates_path,
        output.parent / "selection.json",
        object_root / "object.json",
        object_root / "object.npz",
        robot.urdf_path,
        robot.directory / "manifest.json",
    ]
    paths += [
        Path(r["provenance"]["predictions"])
        for r in read_json(candidates_path)["candidates"]
        if "predictions" in r["provenance"]
    ]
    source = dict(plan_path=str(plan_path), references={str(p): file_hash(p) for p in paths})
    description = robot.urdf_path.read_text()
    if state["environment"] == "hardware":
        packet = shared_hardware_packet(schedule, named_state(current), description, source)
    elif state["environment"] == "mock":
        packet = shared_mock_packet(schedule, named_state(current), description)
        packet["source"] = source
    else:
        raise ValueError("Unknown snapshot environment")
    path = output / "targets.json"
    write_json(path, packet)
    return path
