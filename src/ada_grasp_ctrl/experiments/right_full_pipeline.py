"""ROS-free selection and execution of frozen right_full candidates on real CRX."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf

from ada_grasp_ctrl.grasp_selection.adapters.crx_sharpa import CrxSharpaAdapter
from ada_grasp_ctrl.grasp_selection.selector import GraspSelector
from ada_grasp_ctrl.grasp_selection.types import (
    GraspCandidate,
    RobotState,
    SelectionContext,
    StageTarget,
    STAGES,
    jsonable,
)
from ada_grasp_ctrl.paths import project_root, resolve_from_root
from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record
from .crx_sharpa_right_evaluation import make_config, run_episode, sample_preflight
from .crx_sharpa_right_preparation import RightKinematics, json_write, load_preview
from .crx_sharpa_right_model import JOINTS
from .hugs_sharpa_input import FIELDS
from .phase_comparison import sha256
from .right_full_control import MODES, Controller, physics_substeps
from .right_full_episode import RightFullEvaluator


def select_frozen(report: dict, preview: Path, sample_ids: list[str]) -> tuple[dict, dict, dict]:
    """Invoke the existing real-CRX selector on an explicit frozen candidate set.

    Frozen registration and world flange targets already exist; this adapter only
    packages them as StageTarget. Selection recomputes bounded IK with the existing
    criteria, weights and tie-breaking. Different physical scenes cannot compete.
    """
    rows = [next((r for r in report["samples"] if r["sample_id"] == name), None) for name in sample_ids]
    if not rows or any(r is None for r in rows) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Expected distinct sample IDs present in the verified preview")
    if len({(r["object_id"], r["scale"]) for r in rows}) != 1:
        raise ValueError("Candidates must share the same object and scale")
    records = [
        validate_grasp_record(
            load_npy_record(r["prepared_path"]),
            r["prepared_path"],
            expected_joint_dim=28,
            expected_joint_names=JOINTS,
            require_joint_names=True,
            require_assets=True,
        )
        for r in rows
    ]
    first = records[0]
    for record in records[1:]:
        for key in ("obj_pose", "initial_qpos"):
            if not np.array_equal(record[key], first[key]):
                raise ValueError(f"Candidate scene mismatch: {key}")
    base = np.asarray(report["model"]["world_T_base"])
    adapter = CrxSharpaAdapter(
        preview / "preview_report.json",
        {"frame_id": "world", "unit": "m", "normal_xyz": [0.0, 0.0, 1.0], "d": 0.0},
        base,
    )
    # Frozen preview's table is the world-Z plane; verify that against the actual scene.
    from .crx_sharpa_right_simulation import ApprovedScene

    scene = ApprovedScene(Path(rows[0]["scene_path"]))
    try:
        scene.reset_pose_qpos(first["initial_qpos"], first["obj_pose"])
        table = scene.data.body("table")
        if not np.allclose(table.xpos[2], 0, atol=1e-12) or not np.allclose(table.xmat.reshape(3, 3)[:, 2], [0, 0, 1]):
            raise ValueError("Frozen selector bridge requires the existing world-Z tabletop")
        physics_substeps(scene.model.opt.timestep)
        physics_dt = float(scene.model.opt.timestep)
    finally:
        scene.close_view_and_render()
    source = sha256(preview / "preview_report.json")
    state = RobotState(first["initial_qpos"][:6], 1.0, 1.0, adapter.model_id, "frozen_simulation_state", "fixture")
    context = SelectionContext(state, base, "verified frozen preview", source, 1.0)
    candidates = [
        GraspCandidate(
            row["sample_id"],
            {
                stage: StageTarget(pose[:3], pose[3:], record[field][6:])
                for stage, field, pose in zip(STAGES, FIELDS, record["flange_targets_wxyz"])
            },
            row["prepared_path"],
            source,
            {"frozen_sample": row["sample_id"]},
        )
        for row, record in zip(rows, records)
    ]
    result = GraspSelector(adapter).select(candidates, context)
    if result.selected is None:
        raise ValueError(f"Existing CRX selector found no feasible candidate: {jsonable(result.evaluations)}")
    index = sample_ids.index(result.selected_candidate_id)
    row, record = deepcopy(rows[index]), deepcopy(records[index])
    checks = result.selected["stage_checks"]
    for stage, field in zip(STAGES, FIELDS):
        record[field] = np.asarray(checks[stage]["q_rad"])
    row["all_targets_reachable"] = True
    row["targets"] = [
        {
            "joint_limit_violations": c["joint_limit_violations"],
            "arm_table_clear": c["collision"]["arm_penetration_m"] <= 1e-5,
        }
        for c in checks.values()
    ]
    # Prove real CRX world frames/Jacobians before optimization, at all three targets.
    kin = RightKinematics(report["model"])
    audit = {field: kin.audit(record[field]) for field in FIELDS}
    return row, record, {"selection": jsonable(result), "compatibility": audit, "physics_timestep_s": physics_dt}


def run(preview: Path, output: Path, sample_ids: list[str], settings: object, *, compare: bool = False) -> dict:
    """Select once, then run identical scenes under either or both policies."""
    Controller(settings.controller.mode)
    if settings.trajectory.interpolation != "linear" or settings.contact.source != "mujoco":
        raise ValueError("Only linear joint references and MuJoCo observations are supported")
    if settings.controller.ours.frequency_hz != 10:
        raise ValueError("Original ours frequency must remain 10 Hz")
    report = load_preview(preview)
    row, record, selection = select_frozen(report, preview, sample_ids)
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "selection.json", selection)
    prepared = output / "selected_grasp.npy"
    np.save(prepared, record, allow_pickle=True)
    row["prepared_path"] = str(prepared)
    preflight = sample_preflight(row, record, report["model"])
    json_write(output / "preflight.json", preflight)
    modes = MODES if compare else (settings.controller.mode,)
    manifest = {
        "schema_version": 1,
        "controller_modes": list(modes),
        "settings": OmegaConf.to_container(settings),
        "preview_root": str(preview),
        "preview_sha256": sha256(preview / "preview_report.json"),
        "selected_candidate_id": row["sample_id"],
        "candidate_ids": sample_ids,
        "selected_grasp_sha256": sha256(prepared),
        "scene_sha256": sha256(Path(row["scene_path"])),
        "model": report["model"],
        "physics_timestep_s": selection["physics_timestep_s"],
        "control_period_s": 0.1,
        "physics_substeps": physics_substeps(selection["physics_timestep_s"]),
        "evaluation_seed": 12,
        "inference_rerun": False,
        "selection": "GraspSelector(CrxSharpaAdapter)",
        "code_sha256": {
            str(p.relative_to(project_root())): sha256(p)
            for p in sorted((project_root() / "src/ada_grasp_ctrl").rglob("*.py"))
        },
    }
    json_write(output / "run_manifest.yaml", manifest)
    results, evaluators = [], []
    for mode in modes:
        config = make_config(report, row, output / row["sample_id"] / mode, "op" if mode == "linear" else "ours")
        config = OmegaConf.merge(OmegaConf.to_container(config, resolve=True), settings, {"controller": {"mode": mode}})
        holder = []

        def factory(*args: object) -> RightFullEvaluator:
            evaluator = RightFullEvaluator(*args)
            holder.append(evaluator)
            return evaluator

        result = run_episode(row, report, output, mode, preflight, evaluator_factory=factory, config=config)
        ticks = holder[0].ticks if holder else []
        physical = result.get("physical_lift_pass")
        result.update(
            controller_mode=mode,
            grasp_success=physical,
            lift_success=physical,
            max_contact_force=max((t["max_contact_force"] for t in ticks), default=0.0),
            sum_normal_force=ticks[-1]["sum_normal_force"] if ticks else None,
            completion=holder[0].completion if holder else None,
            solver_status=None
            if mode == "linear"
            else (
                "failed"
                if result["status"] == "execution_error"
                else "degraded"
                if result.get("solver_degraded")
                else "success"
                if ticks
                else None
            ),
        )
        folder = output / row["sample_id"] / mode
        json_write(folder / "diagnostics.json", ticks)
        json_write(folder / "result.json", result)
        results.append(result)
        evaluators.extend(holder)
    paired = None
    if len(evaluators) == 2:
        frames = [[f for f in e.episode_recorder.frames if f["segment"] == "control_initial"] for e in evaluators]
        if all(frames):
            for key in ("qpos", "qvel", "ctrl", "act", "object_pose"):
                np.testing.assert_array_equal(frames[0][0][key], frames[1][0][key])
            paired = True
    summary = {"schema_version": 1, "results": results, "identical_control_initial_state": paired}
    json_write(output / "run_report.json", summary)
    with (output / "failures.jsonl").open("w") as stream:
        import json

        for result in results:
            if result["status"] in ("execution_error", "solver_degraded"):
                stream.write(json.dumps({k: result.get(k) for k in ("controller_mode", "status", "reason")}) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Execute a frozen selected right_full grasp without ROS or hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-root", required=True)
    parser.add_argument(
        "--sample-id", action="append", required=True, help="Candidate subset; repeat only within one scene"
    )
    parser.add_argument("--output-root", required=True, help="Fresh directory; mode subdirectories never overwrite")
    parser.add_argument("--config", type=Path, default=project_root() / "src/ada_grasp_ctrl/config/right_full.yaml")
    parser.add_argument("--controller", choices=MODES, help="Override controller.mode before simulation")
    parser.add_argument("--compare", action="store_true", help="Select once, execute linear and ours serially")
    args = parser.parse_args(argv)
    try:
        settings = OmegaConf.load(resolve_from_root(args.config))
        if args.controller:
            settings.controller.mode = args.controller
        summary = run(
            resolve_from_root(args.preview_root),
            resolve_from_root(args.output_root, "output"),
            args.sample_id,
            settings,
            compare=args.compare,
        )
        for result in summary["results"]:
            print(f"{result['controller_mode']}: {result['status']}; lift_success={result['lift_success']}")
        return int(any(r["status"] in ("execution_error", "solver_degraded") for r in summary["results"]))
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"right_full preflight error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
