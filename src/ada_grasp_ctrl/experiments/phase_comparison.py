"""Paired raw-prediction protocol using the existing op and ours policies."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from hydra import compose, initialize_config_dir
import mujoco
import numpy as np
from omegaconf import DictConfig, open_dict

from ada_grasp_ctrl.batch import write_batch_report
from ada_grasp_ctrl.errors import BatchExecutionError
from ada_grasp_ctrl.paths import source_checkout_root
from ada_grasp_ctrl.runtime import configure_runtime, write_run_manifest
from ada_grasp_ctrl.tasks import TASK_REGISTRY
from ada_grasp_ctrl.tasks.control_eval import _validate_control_preflight, _validate_object_assets, safe_eval_one
from ada_grasp_ctrl.utils.viewer_session import create_debug_viewer_session

from .hugs_shadow_input import adapt_prediction
from .episode_recording import load_episode

PHASE_METHODS = {"phase1": "op", "phase2": "ours"}


def sha256(path: Path) -> str:
    """Hash a file without loading large checkpoints into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON; undefined metrics must be None, never NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_config(task: str, hand: str, output: Path, data_root: Path, **values: Any) -> DictConfig:
    """Compose native defaults with explicit runtime roots and task identity."""
    checkout = source_checkout_root()
    if checkout is None:
        raise ValueError("This experimental entry point requires a source checkout")
    with initialize_config_dir(version_base="1.3", config_dir=str(checkout / "src/ada_grasp_ctrl/config")):
        config = compose(config_name="base", overrides=[f"task={task}", f"hand={hand}"])
    with open_dict(config):
        config.task_name, config.hand_name = task, hand
        config.output_root = str(output)
        config.save_dir = str(output)
        config.log_dir = str(output / "log" / task)
        config.data_root = str(data_root)
        config.n_worker = 1
        for key, value in values.items():
            config[key] = value
    return config


def execute_task(config: DictConfig) -> int:
    """Run one native task, retaining completed reports on batch errors."""
    configure_runtime(config)
    try:
        TASK_REGISTRY[config.task_name](config)
        return 0
    except BatchExecutionError:
        return 1


def prepare_grasp_suite(
    input_root: Path,
    metadata_path: Path,
    data_root: Path,
    output: Path,
    selection_manifest: Path | None = None,
) -> Path:
    """Adapt a frozen selection and solve CPU IK once for both policies.

    Args:
        input_root: Directory of trusted local HUGS predictions.
        metadata_path: Explicit training metadata JSON.
        data_root: HUGS dataset root.
        output: New suite directory; must not exist.
        selection_manifest: Optional JSON with ordered relative ``samples`` paths.

    Returns:
        Suite manifest path, including unsuccessful preparation entries.
    """
    metadata = json.loads(metadata_path.read_text())
    paths = (
        json.loads(selection_manifest.read_text())["samples"]
        if selection_manifest
        else [str(p.relative_to(input_root)) for p in sorted(input_root.rglob("*.npy"))]
    )
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Selection must contain nonempty, unique prediction paths")
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, relative in enumerate(paths):
        source = (input_root / relative).resolve()
        if not source.is_relative_to(input_root.resolve()):
            raise ValueError(f"Selection path escapes input root: {relative}")
        entry = {
            "sample_id": f"g{index:04d}",
            "sample_index": index,
            "source": str(source),
            "status": "preparation_error",
        }
        entries.append(entry)
        try:
            entry["source_sha256"] = sha256(source)
            adapted = adapt_prediction(np.load(source, allow_pickle=True).item(), metadata, data_root)
            entry["scene_path"] = adapted["scene_path"]
            entry["scene_sha256"] = sha256(Path(adapted["scene_path"]))
            dest = output / "adapted_learning" / f"{entry['sample_id']}.npy"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, adapted)
            entry["adapted_path"] = str(dest)
        except Exception as error:
            entry["message"] = f"{source}: {type(error).__name__}: {error}"
    # Freeze selected IDs before any IK or physics outcomes exist.
    write_json(output / "selection.json", entries)
    if any("adapted_path" in item for item in entries):
        config = build_config("format", "shadow", output, data_root, grasp_dir=str(output / "formatted"))
        config.task.data_path = str(output / "adapted_learning")
        config.task.data_name, config.task.max_num = "Learning", -1
        execute_task(config)
        if list((output / "formatted").rglob("*.npy")):
            config = build_config(
                "dummy_arm_qpos",
                "shadow",
                output,
                data_root,
                grasp_dir=str(output / "formatted"),
                dummy_arm_grasp_dir=str(output / "prepared_grasps"),
            )
            config.task.device, config.task.max_num = "cpu", -1
            execute_task(config)
    errors = {}
    for task in ("format", "dummy_arm_qpos"):
        report = output / "log" / task / "run_report.json"
        if report.is_file():
            for item in json.loads(report.read_text())["results"]:
                if item["status"] != "completed":
                    errors[Path(item["input_path"]).stem] = item
    for entry in entries:
        prepared = output / "prepared_grasps" / f"{entry['sample_id']}.npy"
        if prepared.is_file():
            entry.update(status="prepared", prepared_path=str(prepared), prepared_sha256=sha256(prepared))
        elif entry["sample_id"] in errors:
            entry["error"] = errors[entry["sample_id"]]
    manifest = {
        "schema_version": 1,
        "robot": "dummy_arm_shadow",
        "source": "hugs-shadow-right-full",
        "data_root": str(data_root),
        "prepared_root": str(output / "prepared_grasps"),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "samples": entries,
    }
    write_json(output / "preparation_report.json", entries)
    write_json(output / "suite_manifest.json", manifest)
    return output / "suite_manifest.json"


def build_phase_config(
    suite: dict[str, Any], phase: str, output: Path, seed: int = 12, viewer: str | None = None
) -> DictConfig:
    """Freeze the raw three-pose protocol while retaining native control settings."""
    config = build_config(
        "control_eval", suite["robot"], output, Path(suite["data_root"]), grasp_dir=suite["prepared_root"], seed=seed
    )
    config.task.method = PHASE_METHODS[phase]
    config.task.arm_pregrasp_is_grasp = False
    config.task.graspdata.pregrasp_t = 0
    config.task.graspdata.squeeze_t = 0
    config.task.offsets = [0.0]
    config.task.debug_viewer = viewer is not None
    config.task.debug_viewer_backend = viewer or "mujoco"
    with open_dict(config):
        config.task.record_episode = True
        config.task.hold_terminal_viewer = viewer is not None
    return configure_runtime(config)


def episode_metrics(path: Path) -> dict[str, Any]:
    """Summarize actual control, contact, tracking and solver observations."""
    record = np.load(path, allow_pickle=True).item()
    replay = load_episode(path.with_suffix(".replay.npz"))
    frames = replay["frames"]
    contacts = record["contacts"]
    forces = [sum(float(c["contact_force"][0]) for c in step) for step in contacts]
    diagnostics = record.get("solver_diagnostics", [])
    metrics = {
        "control_steps": len(record["dof"]),
        "simulation_seconds": frames[-1]["time"],
        "contact_steps": sum(bool(step) for step in contacts),
        "max_sum_normal_force": max(forces, default=0.0),
        "initial_object_z": float(frames[0]["object_pose"][2]),
        "final_object_z": float(frames[-1]["object_pose"][2]),
        "solver_diagnostics": diagnostics,
        "solver_decisions": dict(Counter(d.get("decision", "diagnostic") for d in diagnostics)),
        "rejected_solves": sum(not d["accepted"] for d in diagnostics),
        "initialization": replay.get("metadata", {}).get("initialization"),
    }
    control_frames = [f for f in frames if f["segment"] in ("approach", "squeeze")]
    metrics["prelift_object_z"] = float((control_frames[-1] if control_frames else frames[0])["object_pose"][2])
    if len(record["dof"]):
        errors = np.asarray(record["dof"]) - np.asarray(record["planned_dof"])
        metrics["base_xyz_tracking_rmse_m"] = np.sqrt(np.mean(errors[:, :3] ** 2, axis=0)).tolist()
        metrics["finger_tracking_rmse_rad"] = float(np.sqrt(np.mean(errors[:, 6:] ** 2)))
    names = replay.get("metadata", {}).get("reference_joint_names")
    if names and control_frames:
        model = mujoco.MjModel.from_xml_path(replay["scene_path"])
        data = mujoco.MjData(model)
        addresses = [int(model.joint(f"child-{name}").qposadr[0]) for name in names]
        palm = model.body("child-rh_palm").id
        palm_errors = []
        for frame in control_frames:
            data.qpos[:] = frame["qpos"]
            mujoco.mj_forward(model, data)
            actual = data.xpos[palm].copy()
            data.qpos[addresses] = frame["reference"]
            mujoco.mj_forward(model, data)
            palm_errors.append(actual - data.xpos[palm])
        metrics["postaction_palm_xyz_tracking_rmse_m"] = np.sqrt(np.mean(np.square(palm_errors), axis=0)).tolist()
    return metrics


def verify_paired_inputs(phases: dict[str, list[dict[str, Any]]]) -> int:
    """Reject comparisons with mismatched seeds, targets or initial simulator state."""
    if not all(phase in phases for phase in PHASE_METHODS):
        return 0
    right = {r["sample_id"]: r for r in phases["phase2"]}
    verified = 0
    for left in phases["phase1"]:
        other = right[left["sample_id"]]
        if not all(row.get("evaluation", {}).get("output_paths") for row in (left, other)):
            continue
        if left["prepared_sha256"] != other["prepared_sha256"]:
            raise ValueError(f"Mismatched prepared hash: {left['sample_id']}")
        if left["evaluation"]["details"]["sample_seed"] != other["evaluation"]["details"]["sample_seed"]:
            raise ValueError(f"Mismatched sample seed: {left['sample_id']}")
        records = [
            load_episode(Path(row["evaluation"]["output_paths"][0]).with_suffix(".replay.npz")) for row in (left, other)
        ]
        stored = np.load(left["prepared_path"], allow_pickle=True).item()
        for key in ("qpos", "qvel", "ctrl", "object_pose"):
            np.testing.assert_array_equal(
                records[0]["frames"][0][key], records[1]["frames"][0][key], err_msg=f"{left['sample_id']} initial {key}"
            )
        for key in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos"):
            for record in records:
                np.testing.assert_array_equal(
                    record["targets"][key], stored[key], err_msg=f"{left['sample_id']} modified {key}"
                )
        verified += 1
    return verified


def run_phase_suite(
    suite: dict[str, Any], phase: str, output: Path, seed: int = 12, viewer: str | None = None
) -> list[dict[str, Any]]:
    """Run each frozen sample independently and collect native eval/stat reports."""
    config = build_phase_config(suite, phase, output, seed, viewer)
    ready = [e for e in suite["samples"] if e["status"] == "prepared"]
    for entry in ready:
        if sha256(Path(entry["prepared_path"])) != entry["prepared_sha256"]:
            raise ValueError(f"Prepared input changed: {entry['sample_id']}")
    _validate_control_preflight(config)
    _validate_object_assets([e["prepared_path"] for e in ready], config)
    write_run_manifest(config, [e["prepared_path"] for e in ready])
    results, rows = [], []
    session = create_debug_viewer_session(config.task)
    try:
        for entry in suite["samples"]:
            row = dict(entry)
            rows.append(row)
            if entry["status"] != "prepared":
                continue
            start = time.monotonic()
            result = safe_eval_one((entry["prepared_path"], config, session, entry["sample_index"]))
            results.append(result)
            row.update(status=result.status.value, evaluation=result.to_dict(), wall_seconds=time.monotonic() - start)
            if result.output_paths:
                row["metrics"] = episode_metrics(Path(result.output_paths[0]))
            print(f"{phase} {entry['sample_id']}: {row['status']}", flush=True)
            write_json(output / "phase_results.json", rows)
    finally:
        if session is not None:
            session.close()
    write_batch_report(Path(config.log_dir), "control_eval", results, num_discovered=len(ready), num_skipped=0)
    stat = build_config("control_stat", suite["robot"], output, Path(suite["data_root"]), seed=seed)
    stat.task.method = PHASE_METHODS[phase]
    stat.task.input_report = str(Path(config.log_dir) / "run_report.json")
    execute_task(stat)
    stats = json.loads((Path(stat.log_dir) / "run_report.json").read_text())
    by_path = {r["input_path"]: r for r in stats["results"]}
    for row in rows:
        if row["status"] == "completed":
            result = by_path.get(row["evaluation"]["output_paths"][0])
            row["status"] = (
                result["details"].get("scientific_outcome", result["status"]) if result else "missing_result"
            )
    write_json(output / "phase_results.json", rows)
    return rows


def compare_phase_reports(samples: list[dict[str, Any]], phases: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Join selected IDs, preserving preparation failures and missing outcomes."""
    indices = {phase: {r["sample_id"]: r for r in rows} for phase, rows in phases.items()}
    rows = []
    for sample in samples:
        row = {
            "sample_id": sample["sample_id"],
            "scene_path": sample.get("scene_path"),
            "prepared_sha256": sample.get("prepared_sha256"),
        }
        for phase in PHASE_METHODS:
            row[phase] = (
                indices.get(phase, {})
                .get(sample["sample_id"], {})
                .get("status", "preparation_error" if sample["status"] != "prepared" else "missing_result")
            )
        rows.append(row)
    summaries = {}
    for phase in PHASE_METHODS:
        counts = Counter(row[phase] for row in rows)
        valid = counts["success"] + counts["failure"]
        scenes = {row["scene_path"] for row in rows if row["scene_path"]}
        successful_scenes = {row["scene_path"] for row in rows if row[phase] == "success"}
        summaries[phase] = {
            "counts": dict(counts),
            "success_over_valid": counts["success"] / valid if valid else None,
            "success_over_selected": counts["success"] / len(rows) if rows else None,
            "successful_scenes": len(successful_scenes),
            "scene_count": len(scenes),
        }
    return {
        "selected_count": len(rows),
        "summary": summaries,
        "pairs": rows,
        "cross_table": dict(Counter(f"{r['phase1']} / {r['phase2']}" for r in rows)),
    }


def run_comparison(
    suite_path: Path,
    output: Path,
    phase: str = "both",
    seed: int = 12,
    sample_id: str | None = None,
    viewer: str | None = None,
) -> int:
    """Execute a fresh comparison, or one independent phase, without overwriting."""
    suite = json.loads(suite_path.read_text())
    if sample_id:
        suite["samples"] = [s for s in suite["samples"] if s["sample_id"] == sample_id]
        if not suite["samples"]:
            raise ValueError(f"Unknown sample ID: {sample_id}")
    if viewer and (len(suite["samples"]) != 1 or phase == "both"):
        raise ValueError("Live viewer requires one sample and one phase")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "experiment_manifest.json",
        {
            "suite_path": str(suite_path),
            "suite_sha256": sha256(suite_path),
            "protocol": "predicted-three-pose",
            "seed": seed,
            "phase": phase,
            "suite": suite,
        },
    )
    phases = {}
    for name in PHASE_METHODS if phase == "both" else [phase]:
        phases[name] = run_phase_suite(suite, name, output / name, seed, viewer)
    comparison = compare_phase_reports(suite["samples"], phases)
    comparison["verified_identical_inputs"] = verify_paired_inputs(phases)
    write_json(output / "comparison.json", comparison)
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "scene_path", "prepared_sha256", *PHASE_METHODS])
        writer.writeheader()
        writer.writerows(comparison["pairs"])
    (output / "summary.md").write_text(
        "# Paired grasp pilot\n\n```json\n"
        + json.dumps(comparison["summary"], indent=2)
        + "\n```\n\nNative op/ours durations differ. No extra squeeze extrapolation.\n"
    )
    return int(
        any(
            r["status"] in ("preparation_error", "execution_error", "solver_degraded", "missing_result")
            for rows in phases.values()
            for r in rows
        )
    )
