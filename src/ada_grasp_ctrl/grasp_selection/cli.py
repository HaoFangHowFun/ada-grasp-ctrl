"""Offline-first selector over saved DexLearn candidates and named state snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

import yaml

from ada_grasp_ctrl.paths import project_root, resolve_from_root
from .adapters.dexlearn import file_hash, load_candidates, read_json, verify_reference
from .adapters.state import load_state
from .criteria.joint_effort import JointDisplacement
from .selector import GraspSelector
from .types import SelectionContext, jsonable


def write_json(path: Path, value: object) -> None:
    """Write finite, reproducible JSON; never silently replace invalid diagnostics."""
    import json

    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(args: argparse.Namespace, output: Path) -> dict:
    """Verify model/input provenance and produce structured selection diagnostics."""
    from .adapters.crx_sharpa import CrxSharpaAdapter

    candidate_path = resolve_from_root(args.candidates)
    state_path = resolve_from_root(args.state)
    report_path = resolve_from_root(args.model_report)
    batch = load_candidates(candidate_path)
    state_payload = read_json(state_path)
    asset_root = resolve_from_root(batch.payload["object_root"])
    asset = read_json(asset_root / "asset.json")
    for name, expected in asset["files_sha256"].items():
        verify_reference({"path": str(asset_root / name), "sha256": expected})
    table = read_json(asset_root / "table_plane.json")
    calibration_id = batch.payload["calibration"]["sha256"]
    if table["extrinsics"]["sha256"] != calibration_id or asset["calibration"]["sha256"] != calibration_id:
        raise ValueError("Table/asset/candidate calibration mismatch")
    declared = batch.payload["calibration"]["declared_frame"]
    if (
        declared
        != "world = deployment URDF root (left arm base); camera = colour optical frame (x right, y down, z forward)"
    ):
        raise ValueError("Camera-world to deployment-world relationship is not explicitly traced")
    adapter = CrxSharpaAdapter(report_path, table, state_payload["world_T_right_base"])
    state, when = load_state(
        state_payload, adapter.model_id, replay=args.replay, now_s=time.time(), max_age_s=args.max_age_s
    )
    context = SelectionContext(
        state,
        state_payload["world_T_right_base"],
        state_payload["transform_source"],
        calibration_id,
        when,
        args.max_age_s,
        args.replay,
        {
            "camera_world": batch.payload["calibration"].get("independently_verified") is True,
            "world_right_base": state_payload.get("base_independently_verified") is True,
            "hand_mount": adapter.info["mount_calibrated"] is True,
            "table": table["status"] == "accepted",
            "environment": False,
        },
        False,
    )
    result = GraspSelector(adapter, scoring_criteria=[JointDisplacement(args.weights)]).select(
        batch.candidates, context
    )
    result.evaluations.extend(batch.rejected)
    result.evaluations.sort(key=lambda e: e.candidate_id)
    source_dir = Path(__file__).parent
    reused = project_root() / "src/ada_grasp_ctrl/experiments"
    code = list(source_dir.rglob("*.py")) + [
        reused / (n + ".py")
        for n in ("crx_sharpa_right_model", "crx_sharpa_right_preparation", "crx_sharpa_right_simulation")
    ]
    manifest = {
        "schema_version": 1,
        "arguments": vars(args),
        "input_sha256": {
            str(p): file_hash(p) for p in (candidate_path, state_path, report_path, asset_root / "asset.json")
        },
        "code_sha256": {str(p): file_hash(p) for p in sorted(code)},
        "git_head": subprocess.check_output(["git", "-C", str(project_root()), "rev-parse", "HEAD"], text=True).strip(),
        "configuration": result.configuration,
        "candidate_metadata": {k: v for k, v in batch.payload.items() if k != "candidates"},
        "snapshot_mode": "recorded_state_replay" if args.replay else "current_hardware_snapshot",
        "inference_rerun": False,
        "motion_commands": 0,
    }
    write_json(output / "manifest.json", manifest)
    (output / "run_manifest.yaml").write_text(yaml.safe_dump(jsonable(manifest), sort_keys=True))
    write_json(output / "candidate_evaluations.json", result.evaluations)
    result_json = jsonable(result)
    result_json.pop("evaluations")
    write_json(output / "selection_result.json", result_json)
    summary = {
        "status": result.status,
        "candidates": len(result.evaluations),
        "feasible_candidates": sum(e.best_solution_index is not None for e in result.evaluations),
        "selected_candidate_id": result.selected_candidate_id,
        "cost": result.cost,
        "execution_ready": False,
        "motion_commands": 0,
    }
    write_json(output / "run_report.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run selection in a new output directory; no-feasible is a structured normal result."""
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--model-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--replay", action="store_true", help="Check freshness at captured receipt time, not wall clock"
    )
    parser.add_argument("--max-age-s", type=float, default=0.5)
    parser.add_argument("--weights", nargs=6, type=float, default=[1.0] * 6)
    args = parser.parse_args(argv)
    output, created = resolve_from_root(args.output_root, "output"), False
    try:
        JointDisplacement(args.weights)
        output.mkdir(parents=True, exist_ok=False)
        created = True
        summary = run(args, output)
        print(json.dumps(summary, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        summary = {"status": "preflight_error", "error": str(error), "execution_ready": False}
        if created:
            write_json(output / "run_report.json", summary)
        print(json.dumps(summary))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
