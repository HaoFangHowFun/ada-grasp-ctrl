"""Preview verified pipeline targets on the single reserved port 8089."""

import argparse
import json
from pathlib import Path

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.grasp_selection.preview import load_entries, serve
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json
from ada_grasp_ctrl.grasp_selection.preview import transform_points
from ada_grasp_ctrl.experiments.crx_sharpa_right_model import pose_matrix


def main(argv: list[str] | None = None) -> int:
    """Serve selected entries and record which dataset is currently displayed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument(
        "--enter-gate", action="store_true", help="The owning pipeline is awaiting terminal confirmation"
    )
    args = parser.parse_args(argv)
    rows = json.loads(args.summary.read_text())
    entries = load_entries(args.summary)
    for row, entry in zip(rows, entries):
        entry["execution_error"] = row.get("execution_error")
        metadata = read_json(Path(entry["asset_root"]) / "object.json")
        snapshot = Path(metadata["source_snapshot"]) / "snapshot.npz"
        if file_hash(snapshot) != metadata["snapshot_sha256"]:
            raise ValueError("Environment point cloud changed since capture")
        with np.load(snapshot, allow_pickle=False) as archive:
            points = archive["points_world_m"]
            stride = max(1, int(np.ceil(len(points) / 40000)))
            entry["environment_points"] = transform_points(points[::stride], entry["adapter"].model_T_world)
        if row.get("packet_path"):
            packet = load_packet(Path(row["packet_path"]))
            entry["plan"] = packet["reviewed_plan"]
            entry["keyframes"] = np.asarray(entry["plan"]["keyframes"])
            if "above_object" in entry["plan"]["phases"]:
                entry["targets"]["above_object"] = entry["adapter"].model_T_world @ pose_matrix(
                    np.asarray(entry["plan"]["above_object"]["target_flange_wxyz"])
                )
            if "lift" in entry["plan"]["phases"]:
                entry["targets"]["lift"] = entry["adapter"].model_T_world @ pose_matrix(
                    np.asarray(entry["plan"]["lift"]["target_flange_wxyz"])
                )
            for phase, pose in entry["plan"].get("cycle", {}).get("flange_targets_wxyz", {}).items():
                entry["targets"][phase] = entry["adapter"].model_T_world @ pose_matrix(np.asarray(pose))
            # JSON output sorts mapping keys; retain the execution phase order for ghosts.
            entry["targets"] = {phase: entry["targets"][phase] for phase in entry["plan"]["phases"]}
    args.status.parent.mkdir(parents=True, exist_ok=True)
    serve(entries, 8089, status_path=args.status, execution_gate=args.enter_gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
