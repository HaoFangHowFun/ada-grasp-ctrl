"""Frozen Sharpa predictions evaluated with native op/ours and native statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time
from typing import Any

import mujoco
import numpy as np
from omegaconf import open_dict
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.batch import write_batch_report
from ada_grasp_ctrl.optimization import SOLVER_FAILURE_POLICIES
from ada_grasp_ctrl.paths import source_checkout_root
from ada_grasp_ctrl.runtime import configure_runtime, write_run_manifest
from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record
from ada_grasp_ctrl.tasks.control_eval import _validate_object_assets, safe_eval_one
from ada_grasp_ctrl.utils.grasp_controller import GraspControllerParameters
from ada_grasp_ctrl.utils.hand_util import collision_mesh_paths
from .episode_recording import load_episode
from .hugs_sharpa_input import FIELDS, SCALES, adapt_prediction
from .phase_comparison import (
    PHASE_METHODS,
    build_config,
    compare_phase_reports,
    execute_task,
    sha256,
    verify_paired_inputs,
    write_json,
)
from .sharpa_model import ACTUATORS, FLANGE, JOINTS, ROBOT, build_models
from .shadow_regression import baseline_delta

PROTOCOL = {
    "name": "sharpa-right-100k-tabletop-v1",
    "selection_seed": 42,
    "evaluation_seed": 12,
    "scales": list(SCALES),
    "objects": 25,
    "candidates_per_scene": 10,
    "candidate_selection": "seeded sample of existing top-20; no outcome filtering",
    "pregrasp_ratio": 2,
    "squeeze_ratio": 1,
    "friction": [0.8, 0.02],
    "mass_kg": 0.1,
    "max_penetration_m": 0.01,
    "initial_force_gate": False,
    "offsets": [0.0],
    "final_sum_force_n": 15,
    "force_calibration": "provisional",
    "hand_kp": 7.5,
    "hand_max_velocity_rad_s": 2.0,
    "training_split_verified": False,
}


def select_sources(predictions: Path, scene_manifest: Path, data_root: Path, smoke: bool) -> list[dict]:
    """Select by scene identity and seed before consulting any execution outcome.

    Args:
        predictions: Root of the saved right-full candidates.
        scene_manifest: Sharpa four-scale scene list, relative to scene_cfg.
        data_root: Dataset root containing object/DGN_2k.
        smoke: Keep one stable full-suite index per scale.

    Returns:
        Frozen sample identities, including full-suite indices.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for relative in sorted(json.loads(scene_manifest.read_text())):
        p = Path(relative)
        if len(p.parts) != 3 or p.parent.name != "tabletop_ur10e":
            continue
        scene = (data_root / "object/DGN_2k/scene_cfg" / p).resolve(strict=True)
        raw = np.load(scene, allow_pickle=True).item()
        obj = raw["scene"][raw["task"]["obj_name"]]
        scale = float(np.asarray(obj["scale"])[0])
        matches = [s for s in SCALES if np.isclose(scale, s, atol=1e-6, rtol=0)]
        if len(matches) != 1:
            raise ValueError(f"Unexpected scene scale: {scene}")
        groups[p.parts[0]][matches[0]].append((p, scene))
    eligible = sorted(k for k, v in groups.items() if set(v) == set(SCALES))
    if len(eligible) < 25:
        raise ValueError(f"Need 25 four-scale tabletop objects, found {len(eligible)}")
    rng = np.random.default_rng(42)
    selected = sorted(rng.choice(eligible, size=25, replace=False).tolist())
    entries = []
    for object_id in selected:
        for scale in SCALES:
            candidates = groups[object_id][scale]
            relative, scene = candidates[int(rng.integers(len(candidates)))]
            files = sorted((predictions / relative.with_suffix("")).glob("*.npy"))
            if len(files) != 20:
                raise ValueError(f"Expected 20 existing candidates: {relative}, found {len(files)}")
            for index in sorted(rng.choice(20, size=10, replace=False).tolist()):
                source = files[index].resolve()
                i = len(entries)
                entries.append(
                    {
                        "sample_id": f"g{i:04d}",
                        "sample_index": i,
                        "source": str(source),
                        "scene_path": str(scene),
                        "object_id": object_id,
                        "scale": scale,
                    }
                )
    if smoke:
        return [next(e for e in entries if e["scale"] == scale) for scale in SCALES]
    return entries


class KinematicsAudit:
    """Compare generated MuJoCo/Pinocchio frame poses and contact Jacobians."""

    def __init__(self, model_dir: Path) -> None:
        from ada_grasp_ctrl.utils.pin_helper import PinocchioHelper

        self.model = mujoco.MjModel.from_xml_path(str(model_dir / "sharpa_right.xml"))
        self.data = mujoco.MjData(self.model)
        self.pin = PinocchioHelper(str(model_dir / "sharpa_right.urdf"), "urdf")
        if set(self.pin.dof_joint_names) != set(JOINTS):
            raise ValueError("Pinocchio must retain all 28 named joints")
        self.to_pin = [JOINTS.index(n) for n in self.pin.dof_joint_names]
        self.to_user = [self.pin.dof_joint_names.index(n) for n in JOINTS]
        self.frames = [
            self.model.body(i).name for i in range(self.model.nbody) if self.model.body(i).name.startswith("right_")
        ]

    def check(self, record: dict) -> dict:
        """Audit all three flange targets and named hand-body frames.

        Args:
            record: Prepared common record with expected WXYZ flange targets.

        Returns:
            Maximum pose and Jacobian discrepancies.

        Raises:
            ValueError: If frame conversion or either kinematic model disagrees.
        """
        errors = {"position_m": 0.0, "orientation_rad": 0.0, "jacobian": 0.0}
        for stage, field in enumerate(FIELDS):
            q = record[field]
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            self.pin.compute_jacobians(q[self.to_pin])
            expected = record["flange_targets_wxyz"][stage]
            errors["position_m"] = max(
                errors["position_m"], float(np.linalg.norm(self.data.body(FLANGE).xpos - expected[:3]))
            )
            rotation = Rotation.from_quat(expected[[4, 5, 6, 3]]).as_matrix()
            errors["orientation_rad"] = max(
                errors["orientation_rad"],
                float(Rotation.from_matrix(rotation.T @ self.data.body(FLANGE).xmat.reshape(3, 3)).magnitude()),
            )
            for frame in self.frames:
                pose = self.pin.get_frame_pose(frame)
                body = self.data.body(frame)
                errors["position_m"] = max(errors["position_m"], float(np.linalg.norm(pose[:3, 3] - body.xpos)))
                errors["orientation_rad"] = max(
                    errors["orientation_rad"],
                    float(Rotation.from_matrix(pose[:3, :3].T @ body.xmat.reshape(3, 3)).magnitude()),
                )
                jp, jr = np.zeros((3, 28)), np.zeros((3, 28))
                mujoco.mj_jacBody(self.model, self.data, jp, jr, self.model.body(frame).id)
                pin_jac = self.pin.get_frame_space_jacobian(frame)[:, self.to_user]
                errors["jacobian"] = max(errors["jacobian"], float(np.max(np.abs(pin_jac - np.vstack([jp, jr])))))
        if errors["position_m"] > 0.001 or errors["orientation_rad"] > 0.01 or errors["jacobian"] > 1e-5:
            raise ValueError(f"Kinematic contract failed: {errors}")
        return errors


def prepare_suite(
    destination: Path,
    predictions: Path,
    metadata: Path,
    checkpoint: Path,
    scenes: Path,
    data_root: Path,
    source_xml: Path,
    source_urdf: Path,
    smoke: bool,
) -> Path:
    """Build immutable inputs from existing predictions, without inference.

    Args:
        destination: New suite directory.
        predictions: Existing 100K right-full candidates.
        metadata: Exact right-hand metadata file.
        checkpoint: Provenance checkpoint; never loaded as executable pickle.
        scenes: Four-scale source scene manifest.
        data_root: Dataset root.
        source_xml: Bench right-hand model.
        source_urdf: Matching source URDF.
        smoke: Prepare only four frozen samples rather than the full suite.

    Returns:
        Validated suite manifest path.
    """
    sources = select_sources(predictions, scenes, data_root, smoke)
    metadata_value = json.loads(metadata.read_text())
    checkpoint_hash = sha256(checkpoint)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "selection.json", sources)
    dependencies = set(build_models(source_xml, source_urdf, destination / "model"))
    dependencies.update([metadata, checkpoint, scenes])
    audit = KinematicsAudit(destination / "model")
    prepared = destination / "prepared_grasps"
    prepared.mkdir()
    rows = []
    for entry in sources:
        row = dict(entry, status="preparation_error")
        rows.append(row)
        try:
            source = Path(row["source"])
            record = adapt_prediction(load_npy_record(source), metadata_value, data_root)
            if record["scene_path"] != row["scene_path"] or record["object_id"] != row["object_id"]:
                raise ValueError("Candidate scene/object does not match frozen selection")
            row["fk_audit"] = audit.check(record)
            path = prepared / f"{row['sample_id']}.npy"
            np.save(path, record)
            obj = Path(record["obj_path"])
            dependencies.update([obj / "info/simplified.json", *collision_mesh_paths(obj)])
            row.update(
                status="prepared",
                prepared_path=str(path),
                prepared_sha256=sha256(path),
                source_sha256=sha256(source),
                scene_sha256=sha256(Path(row["scene_path"])),
            )
        except Exception as error:
            row["message"] = f"{row['source']}: {type(error).__name__}: {error}"
    write_json(destination / "preparation_report.json", rows)
    if any(row["status"] != "prepared" for row in rows):
        raise ValueError(f"Preparation failed; no samples replaced. See {destination}/preparation_report.json")
    dependencies.update((destination / "model").glob("*"))
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "robot": ROBOT,
        "smoke": smoke,
        "data_root": str(data_root),
        "asset_root": str(destination / "model"),
        "prepared_root": str(prepared),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "samples": rows,
        "dependencies": [{"path": str(p), "sha256": sha256(p)} for p in sorted(dependencies)],
    }
    path = destination / "suite_manifest.json"
    write_json(path, manifest)
    path.with_suffix(".sha256").write_text(sha256(path) + "\n")
    return path


def load_suite(path: Path) -> dict:
    """Verify every frozen input and dependency, rejecting partial or changed suites."""
    if path.with_suffix(".sha256").read_text().strip() != sha256(path):
        raise ValueError("Frozen manifest changed; use a new suite")
    suite = json.loads(path.read_text())
    if suite["protocol"] != PROTOCOL or suite["robot"] != ROBOT:
        raise ValueError("Incompatible Sharpa protocol")
    rows = suite["samples"]
    if len(rows) != (4 if suite["smoke"] else 1000):
        raise ValueError("Incomplete suite")
    if len({r["sample_id"] for r in rows}) != len(rows) or len({r["sample_index"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample identity")
    checked = set()
    for row in rows:
        if row["status"] != "prepared":
            raise ValueError("Suite contains preparation failures")
        for name in ("source", "scene", "prepared"):
            source = Path(row["source"] if name == "source" else row[f"{name}_path"])
            expected = row[f"{name}_sha256"]
            if (str(source), expected) not in checked:
                if sha256(source) != expected:
                    raise ValueError(f"Frozen input changed: {source}")
                checked.add((str(source), expected))
    for item in suite["dependencies"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Dependency changed: {item['path']}")
    return suite


def phase_config(suite: dict, phase: str, output: Path, workers: int):
    """Compose private fixed settings while retaining native control objectives."""
    config = build_config(
        "control_eval",
        ROBOT,
        output,
        Path(suite["data_root"]),
        seed=12,
        asset_root=suite["asset_root"],
        grasp_dir=suite["prepared_root"],
        n_worker=workers,
    )
    config.task.method = PHASE_METHODS[phase]
    config.task.arm_pregrasp_is_grasp = False
    config.task.graspdata.pregrasp_t = config.task.graspdata.squeeze_t = 0
    config.task.miu_coef = [0.8, 0.02]
    config.task.simulation_metrics.max_pene = 0.01
    config.task.simulation_metrics.max_force = float("inf")
    with open_dict(config):
        config.task.record_episode = True
    return configure_runtime(config)


def preflight(config: Any, samples: list[dict]) -> None:
    """Validate the private contract without expanding or bypassing the public CLI."""
    if config.hand_name != ROBOT or config.setting != "tabletop" or config.task.method not in PHASE_METHODS.values():
        raise ValueError("Invalid private Sharpa hand/setting/method")
    if list(config.task.offsets) != [0.0] or config.task.input_data != "grasp_dir":
        raise ValueError("Expected the frozen zero-offset grasp protocol")
    if config.task.control.solver_failure_policy not in SOLVER_FAILURE_POLICIES:
        raise ValueError("Invalid solver failure policy")
    GraspControllerParameters.from_config(config.task.control)
    audit = KinematicsAudit(Path(config.asset_root))
    for row in samples:
        record = validate_grasp_record(
            load_npy_record(row["prepared_path"]),
            row["prepared_path"],
            expected_joint_dim=28,
            expected_joint_names=JOINTS,
            require_joint_names=True,
            require_assets=True,
        )
        audit.check(record)
    model = audit.model
    if [model.actuator(i).name for i in range(model.nu)] != list(ACTUATORS):
        raise ValueError("Actuator order mismatch")
    _validate_object_assets([e["prepared_path"] for e in samples], config)


def episode_metrics(path: Path) -> dict:
    """Summarize recorded execution without Shadow-specific palm assumptions."""
    record = load_npy_record(path)
    replay = load_episode(path.with_suffix(".replay.npz"))
    frames = replay["frames"]
    if not frames or frames[-1]["segment"] != "terminal":
        raise ValueError(f"Incomplete replay: {path}")
    for frame in frames:
        for field in ("qpos", "qvel", "ctrl", "act", "object_pose"):
            if not np.isfinite(frame[field]).all():
                raise ValueError(f"Nonfinite replay {field}: {path}")
    diagnostics = record.get("solver_diagnostics", [])
    contacts = record["contacts"]
    adaptive_contact_steps = sum(
        bool(contact) and np.linalg.norm(np.asarray(opt["q_a"]) - np.asarray(target)) > 1e-4
        for contact, opt, target in zip(contacts, record.get("opt_res", []), record["planned_dof"])
    )
    return {
        "initialization": replay["metadata"]["initialization"],
        "control_steps": len(record["dof"]),
        "contact_steps": sum(bool(s) for s in contacts),
        "max_contacts": max(map(len, contacts), default=0),
        "adaptive_contact_steps": int(adaptive_contact_steps),
        "stage_values": sorted({int(s) for s in record["stage"]}),
        "rejected_solves": sum(not d["accepted"] for d in diagnostics),
        "solver_diagnostics": diagnostics,
        "lift_delta_m": float(frames[-1]["object_pose"][2] - frames[0]["object_pose"][2]),
        "recorded_frames": len(frames),
        "terminal_frame": True,
    }


def _eval_one(params: tuple) -> tuple:
    """Run native evaluation with process-local private robot registration."""
    start = time.monotonic()
    return safe_eval_one(params), time.monotonic() - start


def run_suite(path: Path, output: Path, smoke: bool, workers: int, baseline: Path | None = None) -> int:
    """Run paired native policies and report failures without replacing inputs.

    Args:
        path: Frozen suite manifest.
        output: Fresh run directory.
        smoke: Execute four stable inputs only.
        workers: Positive CPU process count; default CLI uses one.
        baseline: Compatible previous regression JSON, if requested.

    Returns:
        Zero for clean execution, one for degradation, errors or regression.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    suite = load_suite(path)
    if suite["smoke"] and not smoke:
        raise ValueError("A smoke-only suite cannot stand in for the full evaluation")
    if smoke:
        suite["samples"] = [next(e for e in suite["samples"] if e["scale"] == s) for s in SCALES]
    info = {
        "suite_sha256": sha256(path),
        "suite_path": str(path),
        "protocol": PROTOCOL,
        "sample_ids": [e["sample_id"] for e in suite["samples"]],
        "workers": workers,
        "smoke": smoke,
    }
    previous = None
    if baseline:
        previous = json.loads(baseline.read_text())
        for key in ("suite_sha256", "protocol", "sample_ids"):
            if previous[key] != info[key]:
                raise ValueError(f"Baseline mismatch: {key}")
    # Validate shared assets and mappings before reserving an execution directory.
    preflight(phase_config(suite, "phase1", output / "phase1", workers), suite["samples"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment_manifest.json", info)
    repo = source_checkout_root()
    write_json(
        output / "code_state.json",
        {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "file_hashes": {
                str(p.relative_to(repo)): sha256(p)
                for folder in ("src", "script")
                for p in sorted((repo / folder).rglob("*"))
                if p.is_file() and p.suffix in (".py", ".yaml", ".sh")
            },
            "asset_dependencies": suite["dependencies"],
        },
    )
    phases, exits = {}, {}
    for phase, method in PHASE_METHODS.items():
        directory = output / phase
        config = phase_config(suite, phase, directory, workers)
        write_run_manifest(config, [e["prepared_path"] for e in suite["samples"]])
        params = [(e["prepared_path"], config, None, e["sample_index"]) for e in suite["samples"]]
        results, rows = [], []
        context = mp.get_context("spawn").Pool(workers) if workers > 1 else nullcontext()
        with context as pool, (directory / "progress.jsonl").open("w") as stream:
            evaluated = pool.imap(_eval_one, params) if pool else map(_eval_one, params)
            for entry, (result, elapsed) in zip(suite["samples"], evaluated):
                results.append(result)
                row = dict(entry, status=result.status.value, evaluation=result.to_dict(), wall_seconds=elapsed)
                if result.output_paths:
                    try:
                        row["metrics"] = episode_metrics(Path(result.output_paths[0]))
                    except Exception as error:
                        row.update(status="execution_error", metrics_error=str(error))
                rows.append(row)
                stream.write(json.dumps({"sample_id": entry["sample_id"], "status": row["status"]}) + "\n")
                stream.flush()
                print(f"{phase} {len(rows)}/{len(params)} {entry['sample_id']}: {row['status']}", flush=True)
        write_batch_report(Path(config.log_dir), "control_eval", results, num_discovered=len(results), num_skipped=0)
        stat = build_config(
            "control_stat", ROBOT, directory, Path(suite["data_root"]), seed=12, asset_root=suite["asset_root"]
        )
        stat.task.method = method
        stat.task.input_report = str(Path(config.log_dir) / "run_report.json")
        stat_code = execute_task(stat)
        statistics = json.loads((Path(stat.log_dir) / "run_report.json").read_text())
        by_path = {r["input_path"]: r for r in statistics["results"]}
        for row in rows:
            if row["status"] == "completed":
                result = by_path.get(row["evaluation"]["output_paths"][0])
                row["status"] = (
                    result["details"].get("scientific_outcome", result["status"]) if result else "missing_result"
                )
        write_json(directory / "phase_results.json", rows)
        phases[phase] = rows
        exits[phase] = {
            "control_stat": stat_code,
            "control_eval": int(any(r.status.value in ("execution_error", "solver_degraded") for r in results)),
        }
        write_json(output / "exit_codes.json", exits)
    comparison = compare_phase_reports(suite["samples"], phases)
    comparison["verified_identical_inputs"] = verify_paired_inputs(phases)
    write_json(output / "comparison.json", comparison)
    for field, filename in (("scale", "scale_results.json"), ("object_id", "object_results.json")):
        write_json(
            output / filename,
            {
                str(key): {
                    phase: dict(Counter(r["status"] for r in rows if r[field] == key)) for phase, rows in phases.items()
                }
                for key in sorted({r[field] for r in suite["samples"]})
            },
        )
    info["comparison"] = comparison
    write_json(output / "regression.json", info)
    delta = baseline_delta(info, previous) if previous else None
    if delta is not None:
        write_json(output / "baseline_delta.json", delta)
    write_json(
        output / "validation.json",
        {
            "paired_inputs": comparison["verified_identical_inputs"],
            "fk_audits": {r["sample_id"]: r["fk_audit"] for r in suite["samples"]},
        },
    )
    (output / "summary.md").write_text(
        "# Sharpa controller experiment\n\n"
        + ("Smoke feasibility only; not a performance estimate.\n\n" if smoke else "")
        + "```json\n"
        + json.dumps(comparison["summary"], indent=2)
        + "\n```\n\n"
        + "Provisional force target. Training split provenance is not verified. Solver degradation remains separate.\n"
    )
    failed = any(
        r["status"] in ("execution_error", "solver_degraded", "missing_result")
        for rows in phases.values()
        for r in rows
    )
    failed = failed or any(code for codes in exits.values() for code in codes.values())
    failed = failed or bool(
        delta and (delta["lost_successes"] or delta["new_solver_degradation"] or delta["new_execution_errors"])
    )
    print(f"REPORT={output / 'summary.md'}", flush=True)
    return int(failed)
