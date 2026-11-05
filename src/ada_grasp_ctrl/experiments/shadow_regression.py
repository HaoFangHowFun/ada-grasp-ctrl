"""Frozen native-asset Shadow regression inputs and headless paired execution."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from ada_grasp_ctrl.batch import write_batch_report
from ada_grasp_ctrl.paths import source_checkout_root
from ada_grasp_ctrl.runtime import write_run_manifest
from ada_grasp_ctrl.tasks.control_eval import _validate_control_preflight, _validate_object_assets, safe_eval_one
from .phase_comparison import (
    PHASE_METHODS,
    build_config,
    build_phase_config,
    compare_phase_reports,
    episode_metrics,
    execute_task,
    prepare_grasp_suite,
    sha256,
    verify_paired_inputs,
    write_json,
)

PROTOCOL = {
    "id": "shadow-native-100x10-v1",
    "scene_count": 100,
    "object_count": 89,
    "grasps_per_scene": 10,
    "sample_count": 1000,
    "sliding_friction": 0.8,
    "torsional_friction": 0.02,
    "pregrasp_ratio": 2,
    "squeeze_ratio": 1,
    "max_pene_m": 0.01,
    "initial_force_gate": "disabled",
    "seed": 12,
    "sampling_seed": 42,
    "proposals_per_scene": 100,
}


def checkout() -> Path:
    """Return the required source checkout or fail before doing work."""
    root = source_checkout_root()
    if root is None:
        raise ValueError("Shadow regression requires a source checkout")
    return root


def matched_pregrasp(prepared: dict, raw: dict, metadata: dict) -> dict:
    """Derive benchmark pregrasp while retaining grasp and squeeze exactly.

    Args:
        prepared: Validated dummy-arm record.
        raw: Corresponding world-palm prediction.
        metadata: Training joint-order metadata.

    Returns:
        Independent record with extrapolated pregrasp translation and fingers.
    """
    result = copy.deepcopy(prepared)
    pre, grasp = raw["pregrasp_qpos"], raw["grasp_qpos"]
    xyz = grasp[:3] + 2 * (pre[:3] - grasp[:3])
    result["pregrasp_qpos"][:3] = xyz - Rotation.from_quat(pre[[4, 5, 6, 3]]).apply([0, 0, 0.034])
    joints = grasp[7:29] + 2 * (pre[7:29] - grasp[7:29])
    result["pregrasp_qpos"][6:] = [joints[metadata["joint_names"].index(n)] for n in prepared["joint_names"][6:]]
    result["approach_qpos"] = result["pregrasp_qpos"][None, :].copy()
    return result


def prepare_suite(destination: Path, dexlearn: Path, checkpoint: Path, metadata_path: Path) -> Path:
    """Create a new immutable 100-scene, 1000-prediction suite once.

    Args:
        destination: New output directory; existing partial suites are rejected.
        dexlearn: Compatible HUGS-DexLearn checkout with its GPU environment.
        checkpoint: Frozen right_full_752k checkpoint.
        metadata_path: Corresponding right_full training metadata.

    Returns:
        Path to the complete, frozen suite manifest.
    """
    repo = checkout()
    native = repo / "examples/assets/object/DGN_2k"
    scenes = sorted((native / "scene_cfg").rglob("*.npy"))
    fixtures = json.loads((repo / "examples/quick_manifest.json").read_text())
    expected_hashes = {e["path"]: e["sha256"] for e in fixtures["assets"]["object_files"]}
    names = sorted({p.parts[-3] for p in scenes})
    if len(scenes) != 100 or len(names) != 89:
        raise ValueError(f"Expected native 100 scenes/89 objects; found {len(scenes)}/{len(names)}")
    for required in (checkpoint, metadata_path, dexlearn / ".venv/bin/python"):
        if not required.is_file():
            raise FileNotFoundError(required)
    destination.mkdir(parents=True, exist_ok=False)
    data_root = destination / "dataset"
    objects = data_root / "object/DGN_2k"
    dependencies, selected = [], []
    for index, name in enumerate(names):
        src, target = native / "processed_data" / name, objects / "processed_data" / name
        target.mkdir(parents=True)
        for path in sorted(src.rglob("*")):
            if path.is_file():
                digest = sha256(path)
                if expected_hashes.get(str(path.relative_to(repo))) != digest:
                    raise ValueError(f"Native fixture dependency hash mismatch: {path}")
                dependencies.append({"path": str(path), "sha256": digest})
        for folder in ("mesh", "urdf", "info"):
            (target / folder).symlink_to(src / folder, target_is_directory=True)
        mesh = trimesh.load(src / "mesh/simplified.obj", force="mesh", process=False)
        points, _ = trimesh.sample.sample_surface(mesh, 16384, seed=42 + index)
        pc = target / "complete_point_cloud.npy"
        np.save(pc, points.astype(np.float32))
        dependencies.append({"path": str(pc), "sha256": sha256(pc)})
    for scene_path in scenes:
        record = np.load(scene_path, allow_pickle=True).item()
        obj = record["scene"][record["task"]["obj_name"]]
        scale = np.asarray(obj["scale"])
        if not np.all(scale == scale[0]) or float(scale[0]) not in (0.06, 0.08, 0.10, 0.12):
            raise ValueError(f"Unexpected native scale in {scene_path}")
        target = objects / scene_path.relative_to(native)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scene_path, target)
        dependencies.append({"path": str(scene_path), "sha256": sha256(scene_path)})
        selected.append(
            {
                "scene_path": str(target),
                "native_scene": str(scene_path),
                "object_id": record["task"]["obj_name"],
                "scale": float(scale[0]),
            }
        )
    write_json(objects / "valid_split/test.json", names)
    scene_manifest = destination / "scene_manifest.json"
    write_json(
        scene_manifest,
        {
            "scene_count": 100,
            "scene_paths": [s["scene_path"] for s in selected],
            "selected": selected,
            "split": "All native fixtures; not a held-out ML evaluation",
        },
    )
    overrides = [
        "task=sample",
        "algo=robotMultiHierar",
        "data=shadowMulti",
        "test_data=shadowMulti",
        "exp_name=shadow_native_regression_v1",
        f"output_folder={destination}/sampling",
        f"ckpt={checkpoint}",
        f"data_root={data_root}",
        f"asset_root={dexlearn}/assets",
        f"data.paths.grasp_path={metadata_path.parent.parent}",
        "data.robot.metadata_group=right_full",
        "data.point_cloud.source=complete",
        "data.point_cloud.preload=false",
        "test_data.pc_source=complete",
        "test_data.preload_point_clouds=false",
        "test_data.metadata_group=right_full",
        'test_data.grasp_type_lst=["3_right_full"]',
        "test_data.test_scene_num=0",
        f"+test_data.test_scene_list_path={scene_manifest}",
        "test_data.test_subset_seed=42",
        "algo.test_grasp_num=100",
        "algo.test_topk=10",
        "num_workers=0",
        "persistent_workers=false",
        "seed=42",
    ]
    command = [str(dexlearn / ".venv/bin/python"), "-m", "dexlearn.main", *overrides]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["OMP_NUM_THREADS"] = "2"
    write_json(
        destination / "generation.json",
        {
            "command": command,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "metadata_sha256": sha256(metadata_path),
            "protocol": PROTOCOL,
        },
    )
    print("Generating 1000 frozen predictions; see sampling.log", flush=True)
    with (destination / "sampling.log").open("w") as log:
        code = subprocess.run(
            command, cwd=dexlearn, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        ).returncode
    write_json(destination / "sampling_exit.json", {"exit_code": code})
    if code:
        raise ValueError(f"DexLearn sampling failed ({code}); see {destination}/sampling.log")
    predictions = destination / "sampling/shadowMulti_robotMultiHierar_shadow_native_regression_v1/tests/step_050000"
    files = sorted(predictions.rglob("*.npy"))
    counts = Counter(str(np.load(p, allow_pickle=True).item()["scene_path"]) for p in files)
    if len(files) != 1000 or len(counts) != 100 or set(counts.values()) != {10}:
        raise ValueError(
            f"Expected exactly 10 predictions per scene; got {len(files)} predictions/{len(counts)} scenes"
        )
    raw_manifest = prepare_grasp_suite(predictions, metadata_path, data_root, destination / "raw_suite")
    suite = json.loads(raw_manifest.read_text())
    if any(e["status"] != "prepared" for e in suite["samples"]):
        raise ValueError("Preparation errors: inspect raw_suite/preparation_report.json; no incomplete suite promoted")
    suite["prepared_root"] = str(destination / "prepared_grasps")
    Path(suite["prepared_root"]).mkdir()
    metadata = json.loads(metadata_path.read_text())
    by_scene = {e["scene_path"]: e for e in selected}
    for entry in suite["samples"]:
        old = Path(entry["prepared_path"])
        record = matched_pregrasp(
            np.load(old, allow_pickle=True).item(), np.load(entry["source"], allow_pickle=True).item(), metadata
        )
        entry["parent_prepared_path"], entry["parent_prepared_sha256"] = str(old), sha256(old)
        target = Path(suite["prepared_root"]) / f"{entry['sample_id']}.npy"
        np.save(target, record)
        entry.update(
            prepared_path=str(target),
            prepared_sha256=sha256(target),
            scale=by_scene[entry["scene_path"]]["scale"],
            object_id=by_scene[entry["scene_path"]]["object_id"],
        )
    suite.update(
        protocol=PROTOCOL,
        dependencies=dependencies,
        generation=json.loads((destination / "generation.json").read_text()),
    )
    result = destination / "suite_manifest.json"
    write_json(result, suite)
    (destination / "suite_manifest.sha256").write_text(sha256(result) + "\n")
    print(f"Frozen suite ready: {result}", flush=True)
    return result


def load_suite(path: Path) -> dict:
    """Reject altered or incomplete frozen inputs before evaluating anything."""
    digest_path = path.with_suffix(".sha256")
    if not digest_path.is_file() or digest_path.read_text().strip() != sha256(path):
        raise ValueError("Frozen manifest digest missing or changed; do not regenerate inputs during regression")
    suite = json.loads(path.read_text())
    if suite.get("protocol") != PROTOCOL or len(suite["samples"]) != 1000:
        raise ValueError("Expected frozen native 100x10 v1 protocol")
    entries = suite["samples"]
    if [e["sample_index"] for e in entries] != list(range(1000)) or len({e["sample_id"] for e in entries}) != 1000:
        raise ValueError("Frozen sample identity/order mismatch")
    checked = set()
    for entry in entries:
        if entry["status"] != "prepared":
            raise ValueError(f"Incomplete prepared sample: {entry['sample_id']}")
        for key in ("prepared", "source", "scene"):
            p = Path(entry["source"] if key == "source" else entry[f"{key}_path"])
            expected = entry[f"{key}_sha256"]
            if (str(p), expected) not in checked:
                if sha256(p) != expected:
                    raise ValueError(f"Frozen input hash mismatch: {p}")
                checked.add((str(p), expected))
    for dependency in suite["dependencies"]:
        if sha256(Path(dependency["path"])) != dependency["sha256"]:
            raise ValueError(f"Frozen asset hash mismatch: {dependency['path']}")
    return suite


def select_samples(suite: dict, smoke: bool) -> dict:
    """Select four stable smoke IDs without renumbering full-suite sample seeds."""
    if not smoke:
        return suite
    result = copy.deepcopy(suite)
    chosen = []
    for scale in (0.06, 0.08, 0.10, 0.12):
        chosen.append(next(e for e in suite["samples"] if e["scale"] == scale))
    result["samples"] = sorted(chosen, key=lambda e: e["sample_index"])
    return result


def baseline_delta(current: dict, baseline: dict) -> dict:
    """Compare strict sample identities; reject incompatible protocols or subsets."""
    for key in ("suite_sha256", "protocol", "sample_ids"):
        if current[key] != baseline[key]:
            raise ValueError(f"Baseline mismatch in {key}")
    old = {r["sample_id"]: r for r in baseline["comparison"]["pairs"]}
    changes = []
    for row in current["comparison"]["pairs"]:
        for phase in PHASE_METHODS:
            before, after = old[row["sample_id"]][phase], row[phase]
            if before != after:
                changes.append({"sample_id": row["sample_id"], "phase": phase, "before": before, "after": after})
    return {
        "changes": changes,
        "lost_successes": [r for r in changes if r["before"] == "success"],
        "new_solver_degradation": [r for r in changes if r["after"] == "solver_degraded"],
        "new_execution_errors": [r for r in changes if r["after"] == "execution_error"],
    }


def _timed_eval_one(params: tuple) -> tuple:
    """Evaluate one stable-index sample and return its worker-local elapsed time.

    Args:
        params: Native safe_eval_one arguments, including the full-suite index.

    Returns:
        Native result and elapsed seconds, excluding time waiting in the pool.
    """
    started = time.monotonic()
    return safe_eval_one(params), time.monotonic() - started


def run_suite(
    suite_path: Path,
    output: Path,
    smoke: bool = False,
    baseline_path: Path | None = None,
    workers: int = 1,
) -> int:
    """Execute both phases headlessly, retaining failures and immutable provenance.

    Args:
        suite_path: Complete frozen manifest from prepare_suite.
        output: Fresh directory for this controller revision.
        smoke: Run one stable sample per scale instead of 1000 samples.
        baseline_path: Optional previous run's regression.json for paired comparison.
        workers: Positive CPU process count; one retains direct serial evaluation.

    Returns:
        0 for completed application execution, 1 for degradation/errors/regression.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    suite = select_samples(load_suite(suite_path), smoke)
    run_info = {
        "suite_sha256": sha256(suite_path),
        "suite_path": str(suite_path),
        "protocol": PROTOCOL,
        "sample_ids": [e["sample_id"] for e in suite["samples"]],
        "smoke": smoke,
        "workers": workers,
    }
    if baseline_path:
        previous = json.loads(baseline_path.read_text())
        for key in ("suite_sha256", "protocol", "sample_ids"):
            if previous.get(key) != run_info[key]:
                raise ValueError(f"Baseline mismatch in {key}")
    output.mkdir(parents=True, exist_ok=False)
    repo = checkout()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    code_hashes = {
        str(p.relative_to(repo)): sha256(p)
        for folder in ("src", "script", "assets/hand")
        for p in sorted((repo / folder).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yaml", ".xml", ".sh")
    }
    write_json(output / "code_state.json", {"git_head": revision, "working_tree_file_hashes": code_hashes})
    write_json(output / "experiment_manifest.json", run_info)
    phases, exits = {}, {}
    for phase, method in PHASE_METHODS.items():
        directory = output / phase
        config = build_phase_config(suite, phase, directory, seed=12)
        config.n_worker = workers
        config.task.miu_coef = [0.8, 0.02]
        config.task.simulation_metrics.max_pene = 0.01
        config.task.simulation_metrics.max_force = float("inf")
        _validate_control_preflight(config)
        _validate_object_assets([e["prepared_path"] for e in suite["samples"]], config)
        write_run_manifest(config, [e["prepared_path"] for e in suite["samples"]])
        results, rows = [], []
        params = [(e["prepared_path"], config, None, e["sample_index"]) for e in suite["samples"]]
        pool_context = mp.Pool(processes=workers) if workers > 1 else nullcontext()
        with pool_context as pool, (directory / "progress.jsonl").open("w") as stream:
            # Ordered delivery keeps reports stable while evaluation runs concurrently.
            evaluated = pool.imap(_timed_eval_one, params) if pool else map(_timed_eval_one, params)
            for entry, (result, elapsed) in zip(suite["samples"], evaluated):
                results.append(result)
                row = dict(
                    entry,
                    status=result.status.value,
                    evaluation=result.to_dict(),
                    wall_seconds=elapsed,
                )
                if result.output_paths:
                    row["metrics"] = episode_metrics(Path(result.output_paths[0]))
                rows.append(row)
                stream.write(
                    json.dumps(
                        {"sample_id": entry["sample_id"], "status": row["status"], "wall_seconds": row["wall_seconds"]}
                    )
                    + "\n"
                )
                stream.flush()
                print(f"{phase} {len(rows)}/{len(suite['samples'])} {entry['sample_id']}: {row['status']}", flush=True)
        write_batch_report(Path(config.log_dir), "control_eval", results, num_discovered=len(results), num_skipped=0)
        stat = build_config("control_stat", suite["robot"], directory, Path(suite["data_root"]), seed=12)
        stat.task.method, stat.task.input_report = method, str(Path(config.log_dir) / "run_report.json")
        stat_code = execute_task(stat)
        stats = json.loads((Path(stat.log_dir) / "run_report.json").read_text())
        by_path = {r["input_path"]: r for r in stats["results"]}
        for row in rows:
            if row["status"] == "completed":
                result = by_path.get(row["evaluation"]["output_paths"][0])
                row["status"] = (
                    result["details"].get("scientific_outcome", result["status"]) if result else "missing_result"
                )
        write_json(directory / "phase_results.json", rows)
        phases[phase] = rows
        exits[phase] = {
            "control_eval": int(any(r.status.value in ("execution_error", "solver_degraded") for r in results)),
            "control_stat": stat_code,
        }
        write_json(output / "exit_codes.json", exits)
    comparison = compare_phase_reports(suite["samples"], phases)
    verified = verify_paired_inputs(phases)
    comparison["verified_identical_inputs"] = verified
    write_json(output / "comparison.json", comparison)
    per_scale = {
        str(scale): {
            phase: dict(Counter(r["status"] for r in rows if r["scale"] == scale)) for phase, rows in phases.items()
        }
        for scale in (0.06, 0.08, 0.10, 0.12)
    }
    write_json(output / "scale_results.json", per_scale)
    run_info["comparison"] = comparison
    write_json(output / "regression.json", run_info)
    delta = baseline_delta(run_info, previous) if baseline_path else None
    if delta is not None:
        write_json(output / "baseline_delta.json", delta)
    (output / "summary.md").write_text(
        "# Shadow controller regression\n\n```json\n"
        + json.dumps({"summary": comparison["summary"], "per_scale": per_scale, "baseline_changes": delta}, indent=2)
        + "\n```\n\nSolver degradation remains a separate outcome. Raw targets have no extra squeeze extrapolation.\n"
    )
    failed = any(
        r["status"] in ("execution_error", "solver_degraded", "missing_result", "preparation_error")
        for rows in phases.values()
        for r in rows
    )
    failed = failed or any(v for phase in exits.values() for v in phase.values())
    failed = failed or bool(
        delta and (delta["lost_successes"] or delta["new_solver_degradation"] or delta["new_execution_errors"])
    )
    print(json.dumps(comparison["summary"], indent=2), flush=True)
    print(f"REPORT={output / 'summary.md'}", flush=True)
    return int(failed)
