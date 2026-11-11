"""Load existing fixed-camera right_full artifacts without inference or pose changes."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ada_grasp_ctrl.paths import resolve_from_root
from ..types import CandidateEvaluation, GraspCandidate, HAND_NAMES, STAGES, StageTarget, rigid


def file_hash(path: Path) -> str:
    """Hash a file incrementally, including large model/point-cloud dependencies."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    """Read strict JSON, rejecting nonstandard NaN/Infinity constants."""

    def reject(value: str) -> None:
        raise ValueError(f"{path}: nonfinite JSON constant {value}")

    result = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected JSON object")
    return result


def verify_reference(reference: dict) -> Path:
    """Resolve and verify an explicitly named file, never search for a replacement."""
    path = resolve_from_root(reference["path"])
    if file_hash(path) != reference["sha256"]:
        raise ValueError(f"Source hash mismatch: {path}")
    return path


@dataclass
class CandidateBatch:
    """Valid candidates plus attributable malformed rows and unchanged metadata."""

    candidates: list[GraspCandidate]
    rejected: list[CandidateEvaluation]
    payload: dict
    source_id: str
    source_sha256: str


def decode(payload: dict, source_id: str, source_sha256: str = "in_memory") -> CandidateBatch:
    """Validate shared conventions, retaining row-specific input failures."""
    required = {
        "schema_version": 1,
        "frame_id": "world",
        "unit": "m",
        "joint_unit": "rad",
        "quaternion_order": "wxyz",
        "pose_frame": "right_hand_flange",
        "stages": list(STAGES),
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"{source_id}: {key} must be {expected!r}")
    names = payload.get("metadata", {}).get("joint_names", [])
    if len(names) != 22 or set(names) != set(HAND_NAMES):
        raise ValueError("Expected exactly 22 named canonical right-hand joints")
    calibration_id = payload.get("calibration", {}).get("sha256")
    if not calibration_id:
        raise ValueError("Missing calibration identity")
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("Missing candidate list")
    ids = [row.get("candidate_id") if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate candidate IDs")
    candidates, rejected = [], []
    for row in rows:
        try:
            if set(row["stages"]) != set(STAGES):
                raise ValueError("Missing original phase; extrapolation is not performed")
            stages = {}
            for stage in STAGES:
                raw = row["stages"][stage]
                named = raw["joint_positions_rad"]
                if not isinstance(named, dict) or set(named) != set(HAND_NAMES):
                    raise ValueError(stage + ": expected 22 named hand joint positions")
                target = StageTarget(
                    raw["position_m"],
                    raw["quaternion_wxyz"],
                    [named[n] for n in HAND_NAMES],
                    raw["frame_id"],
                    raw["pose_frame"],
                )
                if not np.allclose(rigid(raw["T_world_hand"], stage), target.transform(), atol=1e-6, rtol=0):
                    raise ValueError(stage + ": matrix and XYZ/WXYZ pose disagree")
                stages[stage] = target
            candidates.append(GraspCandidate(row["candidate_id"], stages, source_id, calibration_id, row))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            rejected.append(
                CandidateEvaluation(
                    row["candidate_id"], source_id, reasons=["invalid_candidate"], diagnostics={"error": str(error)}
                )
            )
    return CandidateBatch(candidates, rejected, payload, source_id, source_sha256)


def load_candidates(path: Path) -> CandidateBatch:
    """Verify the saved right_full metadata, calibration and original measured object."""
    path = resolve_from_root(path)
    payload = read_json(path)
    batch = decode(payload, str(path), file_hash(path))
    verify_reference(payload["calibration"])
    metadata = read_json(verify_reference(payload["metadata"]))
    if (
        metadata.get("hand_family") != "sharpa_right"
        or metadata.get("grasp_type") != "right_full"
        or metadata.get("wrist_body_names") != ["right_hand_flange"]
        or metadata.get("joint_names") != payload["metadata"]["joint_names"]
    ):
        raise ValueError("Source metadata is not the declared Sharpa right_full flange contract")
    root = resolve_from_root(payload["object_root"])
    if file_hash(root / "object.npz") != payload["object_sha256"]:
        raise ValueError("Measured object hash mismatch")
    obj = read_json(root / "object.json")
    if obj["extrinsics"]["sha256"] != payload["calibration"]["sha256"]:
        raise ValueError("Object/candidate calibration mismatch")
    return batch
