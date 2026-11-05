"""Adapter for HUGS Shadow right_full world-frame palm predictions."""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ADA_JOINTS = [
    f"rh_{finger}J{joint}"
    for finger, joints in (("FF", 4), ("MF", 4), ("RF", 4), ("LF", 5), ("TH", 5))
    for joint in range(joints, 0, -1)
]
FIELDS = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")


def adapt_prediction(raw: dict[str, Any], metadata: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Validate a 58-D right-full prediction and convert palm poses to bases.

    Args:
        raw: Trusted local HUGS prediction record.
        metadata: Training metadata with explicit joint and wrist names.
        data_root: Root for scene paths stored in predictions.

    Returns:
        Learning input with three unchanged finger targets in Ada joint order.

    Raises:
        ValueError: If the prediction or metadata violates this exact contract.
        FileNotFoundError: If the referenced scene does not exist.
    """
    names = metadata["joint_names"]
    if len(names) != 22 or set(names) != set(ADA_JOINTS):
        raise ValueError("Expected exactly the 22 named right Shadow finger joints")
    if metadata["wrist_body_names"] != ["rh_palm"]:
        raise ValueError("Expected rh_palm source poses")
    if np.asarray(raw["pred_grasp_type_id"]).item() != 3:
        raise ValueError("Expected predicted grasp type 3 (right_full)")
    scene = Path(str(np.asarray(raw["scene_path"]).item()))
    scene = scene if scene.is_absolute() else data_root / scene
    if not scene.is_file():
        raise FileNotFoundError(scene)
    result = {"scene_path": str(scene.resolve()), "joint_names": ADA_JOINTS.copy()}
    permutation = [names.index(name) for name in ADA_JOINTS]
    for field in FIELDS:
        value = np.asarray(raw[field], dtype=float)
        if value.shape != (58,) or not np.isfinite(value).all():
            raise ValueError(f"{field}: expected finite (58,) array")
        quat = value[3:7]
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-4, rtol=0):
            raise ValueError(f"{field}: expected normalized WXYZ quaternion")
        base = value[:3] - Rotation.from_quat(quat[[1, 2, 3, 0]]).apply([0, 0, 0.034])
        result[field] = np.r_[base, quat, value[7:29][permutation]]
    return result
