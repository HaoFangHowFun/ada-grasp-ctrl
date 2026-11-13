"""Bind automatic side selection to the same eight-stage ROS playback contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ada_grasp_ctrl.experiments.crx_sharpa_right_model import matrix_pose
from ada_grasp_ctrl.experiments.right_ros_contract import CYCLE_PHASES, side_names, kinematic_contract, validate_packet
from ada_grasp_ctrl.experiments.right_ros_sender import motion_duration, load_packet
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.ros_execution.settings import bridge_config
from .approach import above_object_target
from .lift import lift_target
from .plan import validate_plan
from .dual_scene import SideAdapter


def export_dual_selection(
    robot: object,
    sim: object,
    selection: dict,
    state_path: Path,
    object_root: Path,
    config: dict,
    output: Path,
    candidates_path: Path,
) -> Path:
    """Export one selected side while retaining both measured branches in the reviewed plan."""
    chosen = selection["selected"]
    if chosen is None:
        raise ValueError("No feasible candidate on either enabled side")
    side = chosen["selected_side"]
    passive = "left" if side == "right" else "right"
    state = read_json(state_path)
    q56 = robot.configuration(state["full_positions_rad"])
    adapter = SideAdapter(robot, sim, q56, side)
    start = q56[side_slice(side)].tolist()
    poses = {
        s: np.asarray(chosen["stages"][s]["q56"])[side_slice(side)].tolist() for s in ("pregrasp", "grasp", "squeeze")
    }
    with np.load(object_root / "object.npz", allow_pickle=False) as archive:
        points = archive["points_world_m"]
    above = above_object_target(
        adapter,
        poses["pregrasp"],
        points,
        config["above_object_clearance_m"],
        grasp=poses["grasp"],
        open_hand=config["above_object_open_hand"],
        hand_extrapolation_factor=config["above_object_hand_factor"],
    )
    lift = lift_target(adapter, poses["squeeze"], config["lift_height_m"])
    phases = ["above_object", *CYCLE_PHASES]
    frames = [
        start,
        above["positions"],
        poses["pregrasp"],
        poses["grasp"],
        poses["squeeze"],
        lift["positions"],
        lift["positions"].copy(),
        lift["positions"][:6] + [0.0] * 22,
        start[:6] + [0.0] * 22,
    ]
    settings = bridge_config(
        dict(
            side=side,
            active_arms=["left", "right"],
            environment=state["environment"],
            hand_tolerance=config["hand_tolerance"],
            hand_arrival_required=config["hand_arrival_required"],
        )
    )
    args = argparse.Namespace(minimum_duration=2.0, **settings)
    durations = [motion_duration(a, b, args) for a, b in zip(frames, frames[1:])]
    durations[phases.index("hold")] = float(config["hold_s"])
    flange = {s: matrix_pose(adapter.forward(np.asarray(q))).tolist() for s, q in zip(phases, frames[1:])}
    flange["above_object"], flange["lift"] = above["target_flange_wxyz"], lift["target_flange_wxyz"]
    plan = dict(
        schema_version=1,
        side=side,
        environment=state["environment"],
        candidate_id=chosen["candidate_id"],
        joint_names=list(sum(side_names(side), ())),
        phases=phases,
        keyframes=frames,
        durations_s=durations,
        settle_time_s=settings["settle_time"],
        interpolation="quintic_smoothstep",
        receiver_settings=settings,
        above_object=above,
        lift=lift,
        passive_positions_rad=q56[side_slice(passive)].tolist(),
        continuous_path_verified=False,
        hardware_execution_ready=False,
        cycle=dict(
            hold_s=float(config["hold_s"]),
            hold_starts_after="lift_succeeded",
            release_policy="zero_joint_open",
            return_policy="measured_start_arm_open_hand",
            return_path_checked=False,
            flange_targets_wxyz={s: flange[s] for s in ("hold", "release", "return")},
        ),
    )
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    plan_path, selected_path, snapshot_path = (
        output / n for n in ("motion_plan.json", "selection.json", "snapshot.json")
    )
    selected = dict(
        selected_side=side,
        selected_candidate_id=chosen["candidate_id"],
        selected=dict(stage_checks={s: dict(q_rad=q) for s, q in poses.items()}),
        source_selection=str(output.parent / "selection.json"),
    )
    write_json(selected_path, selected)
    write_json(
        snapshot_path,
        dict(
            side=side,
            environment=state["environment"],
            full_positions_rad=start,
            passive_positions_rad=plan["passive_positions_rad"],
            full_scene_state=str(state_path),
        ),
    )
    write_json(plan_path, plan)
    info = read_json(object_root / "object.json")
    paths = [
        robot.urdf_path,
        robot.directory / "manifest.json",
        state_path,
        selected_path,
        snapshot_path,
        candidates_path,
        output.parent / "selection.json",
        object_root / "object.json",
        object_root / "object.npz",
    ]
    for field in ("plane_path",):
        if field in info:
            paths.append(Path(info[field]))
    if "path" in info["extrinsics"]:
        paths.append(Path(info["extrinsics"]["path"]))
    for candidate in read_json(candidates_path)["candidates"]:
        provenance = candidate["provenance"]
        if "predictions" in provenance:
            paths.append(Path(provenance["predictions"]))
    source = dict(
        kind="grasp_selection",
        model_path=str(robot.urdf_path),
        model_sha256=file_hash(robot.urdf_path),
        selection_path=str(selected_path),
        snapshot_path=str(snapshot_path),
        plan_path=str(plan_path),
        plan_sha256=file_hash(plan_path),
        references={str(p): file_hash(p) for p in paths},
    )
    description = robot.urdf_path.read_text()
    packet = dict(
        schema_version=1,
        kind="ada_side_joint_targets",
        side=side,
        environment=state["environment"],
        sample_id=chosen["candidate_id"],
        units="rad",
        joint_names=plan["joint_names"],
        reference_frame="world",
        pose_frame=side + "_hand_flange",
        quaternion_order="WXYZ",
        execution_phases=phases,
        source=source,
        kinematics=kinematic_contract(description, side),
        scene_kinematics={s: kinematic_contract(description, s) for s in ("left", "right")},
        reviewed_plan=plan,
        collision_path_checked=False,
        targets=[
            dict(phase=s, positions=q, duration_s=t, flange_target_wxyz=flange[s])
            for s, q, t in zip(phases, frames[1:], durations)
        ],
    )
    validate_packet(packet)
    path = output / "targets.json"
    write_json(path, packet)
    write_json(
        output / "run_manifest.yaml",
        dict(
            schema_version=1,
            mode="ada_ros_target_export",
            sample_id=packet["sample_id"],
            source=source,
            targets_sha256=file_hash(path),
        ),
    )
    load_packet(path)
    return path
