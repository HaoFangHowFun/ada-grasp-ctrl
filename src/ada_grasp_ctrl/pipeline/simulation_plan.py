"""Existing inference/selection/planning adapted to a transport-neutral simulation reference."""

from copy import deepcopy
from pathlib import Path
import numpy as np

from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json, file_hash
from ada_grasp_ctrl.grasp_selection.contracts import candidate_from_schema1, plan_from_candidate
from ada_grasp_ctrl.grasp_selection.multi_arm import select_candidates, validate_candidate
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.ros_execution.shared_packet import stage_reference
from .dual_artifacts import single_endpoints
from .dual_inference import infer_live_sides
from .dual_scene import live_scene
from .dual_selection import select_live_hierarchical
from .paired_artifacts import paired_schedule, named_state
from .shared_timing import build_stage_schedule, validate_stage_schedule

PHASES = ("above_object", "pregrasp", "grasp", "squeeze", "lift", "hold")


def adjust_squeeze_candidates(rows: list[dict], ratio: float) -> list[dict]:
    """Extrapolate world wrist translation and fingers before IK, retaining wrist rotation.

    Args:
        rows: Raw schema-1 inference candidates, including mirrored and paired modes.
        ratio: Positive multiplier of the grasp-to-squeeze displacement.

    Returns:
        Independent candidates with adjustment provenance; raw predictions are untouched.

    Raises:
        ValueError: On invalid targets, invalid ratios, or already-adjusted candidates.
    """
    if isinstance(ratio, bool) or not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("simulation.squeeze_extrapolate_ratio must be finite and positive")
    adjusted = []
    for original in rows:
        row = validate_candidate(original)
        if "simulation_pose_adjustment" in row["provenance"]:
            raise ValueError("Squeeze candidates have already been adjusted")
        for side in row["required_arms"]:
            grasp, squeeze = (row["targets"][side][stage] for stage in ("grasp", "squeeze"))
            g, q = np.array(grasp["T_world_hand"]), np.array(squeeze["T_world_hand"])
            q[:3, 3] = g[:3, 3] + ratio * (q[:3, 3] - g[:3, 3])
            squeeze["T_world_hand"] = q.tolist()
            for name, value in squeeze["joint_positions_rad"].items():
                origin = grasp["joint_positions_rad"][name]
                squeeze["joint_positions_rad"][name] = origin + ratio * (value - origin)
        row["provenance"]["simulation_pose_adjustment"] = dict(
            squeeze_extrapolate_ratio=float(ratio),
            formula="grasp + ratio * (squeeze - grasp)",
            fields=["world_wrist_translation", "finger_joint_positions_rad"],
            wrist_rotation="preserve_original_squeeze",
        )
        adjusted.append(validate_candidate(row))
    return adjusted


def prepare_plan(config: dict, selector: str, root: Path, processes: object, state: dict) -> dict | None:
    """Run the unchanged inference and selector against an object-free static scene."""
    obj = root / "object"
    rows = infer_live_sides(obj, config, processes, root)
    rows = adjust_squeeze_candidates(rows, config.get("simulation", {}).get("squeeze_extrapolate_ratio", 2.0))
    write_json(root / "simulation_candidates.json", dict(schema_version=1, candidates=rows))
    robot, static = live_scene(Path(config["robot_snapshot"]), state, read_json(Path(config["table_plane"])))
    if selector == "hierarchical":
        native = read_json(root / "candidates.json") if config["grasp_model"] == "native_multimode" else None
        if native is not None:
            native = deepcopy(native)
            native["candidates"] = rows
        selection, batch, audit = select_live_hierarchical(
            robot,
            static,
            rows,
            config["enabled_arms"],
            state["full_positions_rad"],
            config["hierarchical_selector"],
            native_diagnostics=native,
            grasp_mode=config["grasp_mode"],
        )
        write_json(root / "canonical_candidates.json", batch)
        write_json(root / "hierarchical_selection.json", audit)
    else:
        selection = select_candidates(robot, static, rows, config["enabled_arms"], state["full_positions_rad"])
    write_json(root / "selection.json", selection)
    if selection["selected"] is None:
        return None
    return build_plan(robot, static, selection, root, config)


def build_plan(robot: object, static: object, selection: dict, root: Path, config: dict) -> dict:
    """Use the shared endpoints and duration policy; bind the reviewed six-stage prefix."""
    chosen = selection["selected"]
    args = (robot, static, selection, root / "robot_state.json", root / "object", config)
    if len(chosen["required_arms"]) == 2:
        schedule, checks = paired_schedule(*args)
    else:
        q, _, _, _, above, lift, phases, frames = single_endpoints(*args)
        side = chosen["selected_side"]
        targets = {}
        for phase, frame in zip(phases, frames[1:]):
            full = q.copy()
            full[side_slice(side)] = frame
            targets[phase] = {side: named_state(full)[side]}
        schedule = build_stage_schedule(
            plan_from_candidate(
                candidate_from_schema1(chosen["candidate"], mode=chosen["candidate"].get("mode", side + "_full")),
                available_arms=selection["enabled_arms"],
            ),
            named_state(q),
            targets,
            hold_s=config["hold_s"],
        )
        checks = dict(above_object=above, lift=lift, continuous_path_verified=False)
    validate_stage_schedule(schedule)
    state = read_json(root / "robot_state.json")
    references = {
        str(p): file_hash(p)
        for p in (
            root / "robot_state.json",
            root / "object/object.npz",
            root / "object/object.json",
            root / "scene.mjb",
            root / "settled_state.npz",
            root / "selection.json",
            root / "scene_config.yaml",
            robot.urdf_path,
            robot.directory / "manifest.json",
            Path(config["table_plane"]),
        )
    }
    adjusted_path = root / "simulation_candidates.json"
    if adjusted_path.exists():
        references[str(adjusted_path)] = file_hash(adjusted_path)
    result = dict(
        reference_hz=config.get("simulation", {}).get("reference_hz", 100),
        stage_completion=config.get("simulation", {}).get("stage_completion", "timed"),
        pose_adjustment=deepcopy(chosen["candidate"]["provenance"].get("simulation_pose_adjustment", {})),
        schema_version=1,
        kind="ada_simulation_reference",
        environment="simulation",
        schedule=schedule,
        execution_phases=list(PHASES),
        initial_q56=state["full_positions_rad"],
        full_joint_names=list(robot.joint_names),
        references=references,
        candidate_id=chosen["candidate_id"],
        active_arms=chosen["required_arms"],
        effort=chosen["effort"],
        candidate_counts=[
            {k: row[k] for k in ("mode", "candidate_count", "evaluated_candidate_count", "feasible_candidate_count")}
            for row in selection.get("hierarchical_report", {}).get("mode_trace", [])
        ],
        source_mode=chosen["candidate"]["provenance"].get("source_model_mode", "right_full"),
        execution_mode=chosen["candidate"].get("execution_mode", chosen["selected_side"] + "_full"),
    )
    verify_plan(result)
    write_json(root / "path_checks.json", checks)
    write_json(root / "simulation_plan.json", result)
    return result


def verify_plan(plan: dict) -> None:
    """Reject changed observations, physical snapshot, selection or reviewed reference."""
    if plan.get("kind") != "ada_simulation_reference" or plan.get("environment") != "simulation":
        raise ValueError("Expected a simulation reference, never a relabeled transport packet")
    if plan.get("stage_completion", "arrival") not in ("timed", "arrival"):
        raise ValueError("Unknown simulation stage_completion policy")
    hz = float(plan.get("reference_hz", 100))
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("Simulation reference_hz must be finite and positive")
    if plan.get("workflow") == "grasp_to_place":
        from .placement_schedule import validate_schedule

        validate_schedule(plan["schedule"])
        if (
            plan["execution_phases"] != [s["phase"] for s in plan["schedule"]["stages"]]
            or plan.get("stage_completion") != "timed"
        ):
            raise ValueError("Placement must execute the complete timed schedule")
        if plan["active_arms"] != plan["schedule"]["active_arms"]:
            raise ValueError("Placement active arm changed")
        side = plan["active_arms"][0]
        stages = plan["schedule"]["stages"]
        if stages[0]["arms"][side]["start_positions_rad"] != plan["initial_q56"][side_slice(side)]:
            raise ValueError("Placement schedule initial joints changed")
        place = next(s for s in stages if s["phase"] == "place")
        if not np.array_equal(reference(plan, place, place["duration_s"]), plan["accepted_place_q56"]):
            raise ValueError("Placement schedule changed the accepted Place solution")
    else:
        validate_stage_schedule(plan["schedule"])
        if (
            plan["execution_phases"] != list(PHASES)
            or tuple(s["phase"] for s in plan["schedule"]["stages"][:6]) != PHASES
        ):
            raise ValueError("Simulation must execute the identical six-stage prefix")
    for path, digest in plan["references"].items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f"Stale reviewed source: {path}")


def reference(plan: dict, stage: dict, elapsed: float) -> np.ndarray:
    """Sample the same pure shared linear reference for preview and physical execution."""
    q = np.array(plan["initial_q56"], dtype=float)
    for side in plan["active_arms"]:
        q[side_slice(side)] = stage_reference(
            stage, side, stage["arms"][side]["start_positions_rad"], elapsed, stage["duration_s"]
        )
    return q


def preview(plan: dict, progress: float) -> np.ndarray:
    """Scrub the executed stages, leaving the physics data untouched."""
    stage, elapsed = preview_stage(plan, progress)
    return reference(plan, stage, elapsed)


def preview_stage(plan: dict, progress: float) -> tuple[dict, float]:
    """Resolve the on-demand timeline location shared by joint and object previews."""
    stages = plan["schedule"]["stages"][: len(plan["execution_phases"])]
    now = np.clip(progress, 0, 100) / 100 * stages[-1]["settled_time"]
    stage = next((s for s in stages if now <= s["settled_time"]), stages[-1])
    return stage, now - stage["stage_start_time"]
