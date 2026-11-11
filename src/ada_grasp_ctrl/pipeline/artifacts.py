"""Candidate contact sheets and hash-bound targets from actual selector results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, PHASES, kinematic_contract, validate_packet
from ada_grasp_ctrl.experiments.right_ros_sender import motion_duration
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.grasp_selection.preview import render_images
from ada_grasp_ctrl.ros_execution.settings import bridge_config
from .plan import check_start, validate_plan
from .lift import lift_target
from .approach import above_object_target
from ada_grasp_ctrl.experiments.crx_sharpa_right_model import matrix_pose


def export_selection(
    entry: dict,
    snapshot: Path,
    output: Path,
    *,
    lift_height_m: float = 0.0,
    complete_cycle: bool = False,
    hold_s: float = 5.0,
    above_object_clearance_m: float = 0.0,
    above_object_hand_factor: float = 1.0,
    hand_tolerance: float | None = None,
    hand_arrival_required: bool = True,
) -> Path:
    """Export verified selected endpoints, preserving full arm and hand targets.

    Args:
        entry: Entry produced by the hash/FK-checking selection preview loader.
        snapshot: Full 28-joint live snapshot used for ranking.
        output: Fresh export directory.
        lift_height_m: Optional post-selection lift, excluded from candidate gates.
        complete_cycle: Append hold, open-hand release and return after lift.
        hold_s: Time maintaining the lift target after confirmed arrival.
        above_object_clearance_m: Optional hand-to-object overhead clearance; zero preserves legacy plans.
        above_object_hand_factor: Finger extrapolation from grasp toward pregrasp; 1 preserves pregrasp.
        hand_tolerance: Optional pipeline finger-arrival tolerance in radians.
        hand_arrival_required: Whether finger position error gates arrival and phase transitions.

    Returns:
        Path to the sender packet, bound to its selection and preview plan.
    """
    if entry["keyframes"] is None:
        raise ValueError("No feasible selected candidate to export")
    state = read_json(snapshot)
    if state.get("environment") not in ("mock", "hardware") or state.get("full_joint_names") != list(JOINT_NAMES):
        raise ValueError("Execution export requires a measured full 28-joint snapshot")
    start = state["full_positions_rad"]
    if start[:6] != entry["keyframes"][0, :6].tolist():
        raise ValueError("Selection start differs from snapshot")
    frames = [start, *entry["keyframes"][1:].tolist()]
    if isinstance(lift_height_m, bool) or not np.isfinite(lift_height_m) or lift_height_m < 0:
        raise ValueError("Lift height must be finite and nonnegative; zero disables lift")
    phases = list(PHASES)
    lift = None
    if not isinstance(complete_cycle, bool):
        raise ValueError("complete_cycle must be boolean")
    if isinstance(hold_s, bool) or not np.isfinite(hold_s) or not 0 < hold_s <= 300:
        raise ValueError("hold_s must be finite and within (0, 300]")
    if lift_height_m:
        lift = lift_target(entry["adapter"], frames[-1], lift_height_m)
        frames.append(lift["positions"])
        phases.append("lift")
    cycle = None
    if complete_cycle and lift is not None:
        frames.extend([frames[-1].copy(), frames[-1][:6] + [0.0] * 22, start[:6] + [0.0] * 22])
        phases.extend(["hold", "release", "return"])
        world_model = np.linalg.inv(entry["adapter"].model_T_world)
        cycle = {
            "hold_s": float(hold_s),
            "hold_starts_after": "lift_succeeded",
            "release_policy": "zero_joint_open",
            "return_policy": "measured_start_arm_open_hand",
            "flange_targets_wxyz": {
                phase: matrix_pose(world_model @ entry["adapter"].kin.forward(np.asarray(q))).tolist()
                for phase, q in zip(phases[4:], frames[5:])
            },
            "return_path_checked": False,
        }
    check_start(start, start)
    settings = bridge_config(
        {
            "environment": state["environment"],
            "hand_arrival_required": hand_arrival_required,
            **({"hand_tolerance": hand_tolerance} if hand_tolerance is not None else {}),
        }
    )
    args = argparse.Namespace(minimum_duration=2.0, **settings)
    plan = {
        "schema_version": 1,
        "environment": state["environment"],
        "candidate_id": entry["result"]["selected_candidate_id"],
        "joint_names": list(JOINT_NAMES),
        "phases": phases,
        "keyframes": frames,
        "durations_s": [motion_duration(a, b, args) for a, b in zip(frames, frames[1:])],
        "settle_time_s": settings["settle_time"],
        "interpolation": "quintic_smoothstep",
        "continuous_path_verified": False,
        "hardware_execution_ready": False,
        "receiver_settings": settings,
    }
    if lift is not None:
        plan["lift"] = lift
    if cycle is not None:
        plan["cycle"] = cycle
        plan["durations_s"][4] = float(hold_s)
    validate_plan(plan)
    run = Path(entry["run"])
    manifest = read_json(run / "manifest.json")
    references = dict(manifest["input_sha256"])
    for path in (run / "selection_result.json", run / "candidate_evaluations.json", run / "manifest.json", snapshot):
        references[str(path)] = file_hash(path)
    batch = manifest["candidate_metadata"]
    references[batch["calibration"]["path"]] = batch["calibration"]["sha256"]
    references[batch["metadata"]["path"]] = batch["metadata"]["sha256"]
    asset_root = Path(entry["asset_root"])
    object_info = read_json(asset_root / "object.json")
    source_snapshot = Path(object_info["source_snapshot"]) / "snapshot.npz"
    if file_hash(source_snapshot) != object_info["snapshot_sha256"]:
        raise ValueError("Environment point cloud changed since extraction")
    references[str(source_snapshot)] = object_info["snapshot_sha256"]
    for name, digest in read_json(asset_root / "asset.json")["files_sha256"].items():
        references[str(asset_root / name)] = digest
    model = Path(entry["adapter"].info["urdf"])
    references[str(model)] = file_hash(model)
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "motion_plan.json"
    packet = {
        "schema_version": 1,
        "kind": "ada_right_joint_targets",
        "environment": state["environment"],
        "sample_id": f"group{entry['group']}_{plan['candidate_id']}",
        "units": "rad",
        "joint_names": list(JOINT_NAMES),
        "reference_frame": "world",
        "pose_frame": "right_hand_flange",
        "quaternion_order": "WXYZ",
        "collision_path_checked": False,
        "execution_phases": phases,
        "source": {
            "kind": "grasp_selection",
            "model_path": str(model),
            "model_sha256": file_hash(model),
            "selection_path": str(run / "selection_result.json"),
            "snapshot_path": str(snapshot),
            "plan_path": str(plan_path),
            "plan_sha256": "",
            "references": references,
        },
        "kinematics": kinematic_contract(model.read_text()),
        "targets": [
            {
                "phase": phase,
                "positions": frames[index + 1],
                "flange_target_wxyz": (
                    entry["result"]["selected"]["stages"][phase]["position_m"]
                    + entry["result"]["selected"]["stages"][phase]["quaternion_wxyz"]
                ),
                "duration_s": plan["durations_s"][index],
            }
            for index, phase in enumerate(PHASES)
        ],
        "reviewed_plan": plan,
    }
    if lift is not None:
        packet["targets"].append(
            {
                "phase": "lift",
                "positions": lift["positions"],
                "flange_target_wxyz": lift["target_flange_wxyz"],
                "duration_s": plan["durations_s"][3],
            }
        )
    if cycle is not None:
        for index, phase in enumerate(phases[4:], start=4):
            packet["targets"].append(
                {
                    "phase": phase,
                    "positions": frames[index + 1],
                    "flange_target_wxyz": cycle["flange_targets_wxyz"][phase],
                    "duration_s": plan["durations_s"][index],
                }
            )
    if (
        isinstance(above_object_clearance_m, bool)
        or not np.isfinite(above_object_clearance_m)
        or above_object_clearance_m < 0
    ):
        raise ValueError("Above-object clearance must be finite and nonnegative")
    if above_object_clearance_m:
        world_model = np.linalg.inv(entry["adapter"].model_T_world)
        points = np.asarray(entry["surface"]["vertices"])
        points_world = points @ world_model[:3, :3].T + world_model[:3, 3]
        above = above_object_target(
            entry["adapter"],
            frames[1],
            points_world,
            above_object_clearance_m,
            grasp=frames[2],
            hand_extrapolation_factor=above_object_hand_factor,
        )
        first_duration = motion_duration(start, above["positions"], args)
        descent_duration = motion_duration(above["positions"], frames[1], args)
        frames.insert(1, above["positions"])
        phases.insert(0, "above_object")
        plan["durations_s"][:1] = [first_duration, descent_duration]
        plan["above_object"] = above
        packet["targets"][0]["duration_s"] = descent_duration
        packet["targets"].insert(
            0,
            {
                "phase": "above_object",
                "positions": above["positions"],
                "flange_target_wxyz": above["target_flange_wxyz"],
                "duration_s": first_duration,
            },
        )
    validate_plan(plan)
    write_json(plan_path, plan)
    packet["source"]["plan_sha256"] = file_hash(plan_path)
    validate_packet(packet)
    path = output / "targets.json"
    write_json(path, packet)
    write_json(
        output / "run_manifest.yaml",
        {
            "schema_version": 1,
            "mode": "ada_ros_target_export",
            "sample_id": packet["sample_id"],
            "source": packet["source"],
            "targets_sha256": file_hash(path),
        },
    )
    return path


def contact_sheets(entry: dict, output: Path) -> dict:
    """Render each feasible candidate's best branch and pack six grasp views per page."""
    from PIL import Image, ImageDraw

    output.mkdir(parents=True, exist_ok=False)
    evaluations = json.loads((Path(entry["run"]) / "candidate_evaluations.json").read_text())
    feasible = [e for e in evaluations if e["best_solution_index"] is not None]
    tiles = []
    for evaluation in feasible:
        option = next(o for o in evaluation["options"] if o["solution_index"] == evaluation["best_solution_index"])
        frames = np.array([entry["keyframes"][0], *[option["stages"][phase]["q_rad"] for phase in PHASES]])
        item = {
            **entry,
            "group": evaluation["candidate_id"],
            "keyframes": frames,
            "result": {**entry["result"], "selected_candidate_id": evaluation["candidate_id"]},
        }
        render_images([item], output)
        image = Image.open(output / f"group{evaluation['candidate_id']}_closeup.png").convert("RGB")
        image = image.resize((600, 450))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 600, 42), fill=(20, 30, 40))
        chosen = evaluation["candidate_id"] == entry["result"]["selected_candidate_id"]
        draw.text(
            (10, 8),
            f"{evaluation['candidate_id']} | cost {option['score']:.5f}" + (" | SELECTED" if chosen else ""),
            fill="white",
        )
        tiles.append(image)
    pages = []
    for offset in range(0, max(1, len(tiles)), 6):
        sheet = Image.new("RGB", (1800, 940), (35, 40, 48))
        ImageDraw.Draw(sheet).text(
            (12, 10),
            f"Group {entry['group']} | feasible {len(feasible)} / {len(evaluations)} | six grasps per page",
            fill="white",
        )
        for index, tile in enumerate(tiles[offset : offset + 6]):
            sheet.paste(tile, ((index % 3) * 600, 40 + (index // 3) * 450))
        path = output / f"candidates_{offset // 6 + 1:03d}.png"
        sheet.save(path)
        pages.append(str(path))
    result = {
        "total": len(evaluations),
        "feasible": len(feasible),
        "pages": pages,
        "selected": entry["result"]["selected_candidate_id"],
    }
    write_json(output / "index.json", result)
    return result
