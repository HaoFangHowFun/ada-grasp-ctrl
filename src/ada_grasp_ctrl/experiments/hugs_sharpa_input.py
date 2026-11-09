"""Exact HUGS right-full Sharpa flange-pose adapter, without Shadow offsets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.schema import validate_grasp_record, with_current_schema
from .sharpa_model import FLANGE, HAND_JOINTS, JOINTS

FIELDS = ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos")
SCALES = (0.060, 0.081, 0.106, 0.133)


def adapt_prediction(raw: dict[str, Any], metadata: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Convert three 58D world-flange targets into a named 28D record.

    Args:
        raw: Trusted local single-candidate DexLearn record.
        metadata: Explicit right-hand training metadata.
        data_root: Root for relative scene references.

    Returns:
        Common grasp record with pregrasp extrapolated exactly once.

    Raises:
        ValueError: If frame, hand, dimensions, scene or scale differ.
    """
    names = metadata.get("joint_names", [])
    if len(names) != 22 or set(names) != set(HAND_JOINTS):
        raise ValueError("Expected the 22 named right Sharpa joints")
    if metadata.get("wrist_body_names") != [FLANGE]:
        raise ValueError(f"Expected source frame {FLANGE}")
    if np.asarray(raw["pred_grasp_type_id"]).item() != 3:
        raise ValueError("Expected right_full type 3")
    scene_path = Path(str(np.asarray(raw["scene_path"]).item()))
    scene_path = (scene_path if scene_path.is_absolute() else data_root / scene_path).resolve(strict=True)
    if scene_path.parent.name != "tabletop_ur10e":
        raise ValueError(f"Only tabletop_ur10e scenes are supported: {scene_path}")
    scene = np.load(scene_path, allow_pickle=True).item()
    object_id = scene["task"]["obj_name"]
    obj = scene["scene"][object_id]
    scale = np.asarray(obj["scale"], dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not np.allclose(scale, scale[0], atol=1e-9, rtol=0):
        raise ValueError("Expected finite uniform object scale")
    if not any(np.isclose(scale[0], s, atol=1e-6, rtol=0) for s in SCALES):
        raise ValueError(f"Unexpected scale {scale[0]}")
    mesh = (scene_path.parent / obj["file_path"]).resolve(strict=True)
    permutation = [names.index(j) for j in HAND_JOINTS]
    targets = []
    for field in FIELDS:
        q = np.asarray(raw[field], dtype=float)
        if q.shape != (58,) or not np.isfinite(q).all():
            raise ValueError(f"{field}: expected finite (58,) vector")
        if not np.isclose(np.linalg.norm(q[3:7]), 1, atol=1e-4, rtol=0):
            raise ValueError(f"{field}: expected normalized WXYZ quaternion")
        targets.append(np.r_[q[:7], q[7:29][permutation]])
    targets = np.asarray(targets)
    targets[0, :3] = targets[1, :3] + 2 * (targets[0, :3] - targets[1, :3])
    targets[0, 7:] = targets[1, 7:] + 2 * (targets[0, 7:] - targets[1, 7:])
    # Nested X/Y/Z hinge bodies use intrinsic XYZ, not extrinsic xyz.
    eulers = Rotation.from_quat(targets[:, [4, 5, 6, 3]]).as_euler("XYZ")
    eulers = np.unwrap(eulers, axis=0)
    result = with_current_schema(
        {
            "obj_path": str(mesh.parent.parent),
            "obj_pose": np.asarray(obj["pose"], dtype=float),
            "obj_scale": float(scale[0]),
            "scene_path": str(scene_path),
            "object_id": object_id,
            "joint_names": list(JOINTS),
            "source_frame": FLANGE,
            "flange_targets_wxyz": targets[:, :7].copy(),
            "pregrasp_extrapolation": 2,
        }
    )
    for i, field in enumerate(FIELDS):
        result[field] = np.r_[targets[i, :3], eulers[i], targets[i, 7:]]
    return validate_grasp_record(
        result,
        str(scene_path),
        expected_joint_dim=28,
        expected_joint_names=JOINTS,
        require_joint_names=True,
        require_assets=True,
    )
