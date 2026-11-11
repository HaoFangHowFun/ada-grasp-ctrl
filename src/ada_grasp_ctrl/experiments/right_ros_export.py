"""Export existing right-CRX prepared Ada targets; never invent replacement poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ada_grasp_ctrl.paths import resolve_external_root
from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record
from .crx_sharpa_right_preparation import load_preview
from .phase_comparison import sha256
from .right_ros_contract import JOINT_NAMES, PHASES, kinematic_contract, validate_packet


def export_targets(preview_root: Path, sample_id: str, output_root: Path) -> Path:
    """Export one hash-verified prepared sample, preserving all three 28D targets.

    Args:
        preview_root: Existing preview directory whose hashes and diagnostics are verified.
        sample_id: Explicit sample selection; no success-based reselection is performed.
        output_root: New, empty export directory.

    Returns:
        Path of the JSON packet usable by the system-Python ROS sender.

    Raises:
        ValueError: If preparation, IK, limits, static arm clearance or metadata are invalid.
        FileExistsError: If output_root exists.
    """
    report = load_preview(preview_root)
    matches = [s for s in report["samples"] if s["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"{sample_id}: expected exactly one prepared sample")
    sample = matches[0]
    if not sample["all_targets_reachable"] or len(sample["targets"]) != 3:
        raise ValueError(f"{sample_id}: preparation contains invalid IK targets")
    for phase, diagnostic in zip(PHASES, sample["targets"]):
        if (
            not diagnostic["solver_success"]
            or not diagnostic["reachable"]
            or diagnostic["joint_limit_violations"]
            or not diagnostic["arm_table_clear"]
            or diagnostic["static_contacts"]["arm_table_penetration"]
        ):
            raise ValueError(f"{sample_id}.{phase}: invalid IK/limits or static arm-table penetration")
        if (
            diagnostic["position_m"] > report["tolerances"]["ik_position_m"]
            or diagnostic["orientation_rad"] > report["tolerances"]["ik_angle_rad"]
        ):
            raise ValueError(f"{sample_id}.{phase}: IK residual exceeds preparation tolerance")
    record = load_npy_record(sample["prepared_path"])
    validate_grasp_record(
        record,
        sample_id,
        expected_joint_dim=28,
        expected_joint_names=JOINT_NAMES,
        require_joint_names=True,
        require_assets=True,
    )
    if record.get("sample_id") != sample_id or record.get("quaternion_order") != "WXYZ":
        raise ValueError(f"{sample_id}: prepared identity/quaternion convention differs")
    if not np.array_equal(record["flange_targets_wxyz"], sample["evaluation_flange_targets_wxyz"]):
        raise ValueError(f"{sample_id}: record/report target mismatch")
    model = Path(report["model"]["urdf"])
    packet: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ada_right_joint_targets",
        "environment": "mock",
        "sample_id": sample_id,
        "units": "rad",
        "joint_names": list(JOINT_NAMES),
        "reference_frame": "world",
        "pose_frame": "right_hand_flange",
        "quaternion_order": "WXYZ",
        "registration": report["registration"],
        "collision_path_checked": False,
        "source": {
            "preview_report": str(preview_root / "preview_report.json"),
            "preview_sha256": sha256(preview_root / "preview_report.json"),
            "prepared_path": sample["prepared_path"],
            "prepared_sha256": sha256(Path(sample["prepared_path"])),
            "model_path": str(model),
            "model_sha256": sha256(model),
        },
        "kinematics": kinematic_contract(model.read_text()),
        "targets": [
            {
                "phase": phase,
                "positions": np.asarray(record[phase + "_qpos"]).tolist(),
                "flange_target_wxyz": np.asarray(record["flange_targets_wxyz"][i]).tolist(),
                "preparation_diagnostics": sample["targets"][i],
            }
            for i, phase in enumerate(PHASES)
        ],
    }
    validate_packet(packet)
    output_root.mkdir(parents=True, exist_ok=False)
    path = output_root / "targets.json"
    path.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    manifest = {
        "schema_version": 1,
        "mode": "ada_ros_target_export",
        "sample_id": sample_id,
        "output_root": str(output_root),
        "source": packet["source"],
        "targets_sha256": sha256(path),
    }
    (output_root / "run_manifest.yaml").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_root / "run_report.json").write_text(json.dumps({"status": "exported", **manifest}, indent=2) + "\n")
    return path


def main() -> int:
    """Run preparation export under the maintained Ada Python 3.10 environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        preview = resolve_external_root("data", args.preview_root)
        output = resolve_external_root("output", args.output_root)
        print(export_targets(preview, args.sample_id, output))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
