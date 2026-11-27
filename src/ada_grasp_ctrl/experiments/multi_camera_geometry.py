"""Metric RGB-D geometry and seed-associated fusion for stationary cameras."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .fixed_camera_geometry import clusters, project_color, roi_mask, signed_height, transform

ROLES = ("fixed", "wrist_left", "wrist_right")
SERIALS = dict(zip(ROLES, ("347322060270", "260522275388", "260522273536")))


def select_bundle(queues: dict, now: float, after: float, max_age: float, tolerance: float) -> dict:
    """Select a fresh global timestamp window from all concurrently subscribed streams.

    Args:
        queues: Stream keys mapped to (timestamp, payload) sequences.
        now: ROS wall time in seconds.
        after: Capture request time; earlier sensor frames are ineligible.
        max_age: Maximum sensor age in seconds.
        tolerance: Maximum difference across the entire bundle, including CameraInfo.

    Returns:
        One selected payload per stream.

    Raises:
        ValueError: If any stream is absent, stale, future, or outside the common window.
    """
    if not np.isfinite([now, after, max_age, tolerance]).all() or min(max_age, tolerance) <= 0:
        raise ValueError("Invalid capture timing limits")
    fresh = {
        key: [
            (stamp, msg)
            for stamp, msg in queue
            if np.isfinite(stamp) and after < stamp <= now and now - stamp <= max_age
        ]
        for key, queue in queues.items()
    }
    missing = [key for key, queue in fresh.items() if not queue]
    if missing:
        raise ValueError(f"Missing/future/stale streams: {missing}")
    for anchor in sorted({stamp for queue in fresh.values() for stamp, _ in queue}, reverse=True):
        windows = {
            key: [pair for pair in queue if anchor - tolerance <= pair[0] <= anchor] for key, queue in fresh.items()
        }
        if all(windows.values()):
            return {key: max(queue, key=lambda pair: pair[0])[1] for key, queue in windows.items()}
    raise ValueError("No global synchronized RGB-D/CameraInfo window")


def backproject(depth: np.ndarray, encoding: str, info: dict, limits: dict) -> tuple[np.ndarray, np.ndarray]:
    """Backproject rectified depth with its own intrinsics and ROS metric encoding.

    Returns finite depth-optical XYZ and their flattened pixel indices. RealSense
    ROS 16UC1 images are millimetres (including D405, regardless of SDK depth units).
    Float 32FC1 images are metres. Nonzero distortion is rejected rather than ignored.
    """
    if encoding not in ("16UC1", "32FC1"):
        raise ValueError(f"Unsupported depth encoding: {encoding}")
    if depth.shape != (info["height"], info["width"]):
        raise ValueError("Depth image and depth CameraInfo dimensions differ")
    if info["distortion_model"] not in ("plumb_bob", "rational_polynomial") or not np.allclose(info["d"], 0):
        raise ValueError("Require rectified depth CameraInfo with zero distortion")
    k = np.asarray(info["k"], dtype=float).reshape(3, 3)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("Invalid depth intrinsics")
    z = depth.astype(float).ravel() * (0.001 if encoding == "16UC1" else 1.0)
    valid = np.flatnonzero(np.isfinite(z) & (z > 0) & (z >= limits["min_depth_m"]) & (z <= limits["max_depth_m"]))
    v, u = np.divmod(valid, info["width"])
    points = np.column_stack(((u - k[0, 2]) * z[valid] / k[0, 0], (v - k[1, 2]) * z[valid] / k[1, 1], z[valid]))
    return points, valid


def bounds_mask(points: np.ndarray, bounds: list) -> np.ndarray:
    """Select a finite ordered axis-aligned box in the existing world frame."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError("workspace/exclusion bounds must be ordered finite 2x3 metre coordinates")
    return ((points >= bounds[0]) & (points <= bounds[1])).all(axis=1)


def fuse_clouds(
    clouds: dict, fixed: dict, metadata: dict, plane: dict, config: dict, camera_config: dict
) -> tuple[dict, dict]:
    """Reuse the full D415 workspace and table filter; keep representative source indices."""
    points = np.concatenate([clouds[role] for role in ROLES])
    labels = np.concatenate([np.full(len(clouds[role]), i, dtype=np.int8) for i, role in enumerate(ROLES)])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Fusion requires finite Nx3 calibrated clouds")
    color = transform(points, np.linalg.inv(fixed["T_world_color"]))
    pixels = project_color(color, metadata["color_info"])
    h, w = fixed["rgb"].shape[:2]
    workspace = camera_config.get("automatic_workspace_roi", "full_valid_camera")
    workspace = [0, 0, w, h] if workspace == "full_valid_camera" else workspace
    keep = roi_mask(pixels, workspace, w, h)
    keep &= (color[:, 2] >= camera_config["min_depth_m"]) & (color[:, 2] <= camera_config["max_depth_m"])
    if config.get("workspace_bounds_m") is not None:
        keep &= bounds_mask(points, config["workspace_bounds_m"])
    for bounds in config.get("exclude_bounds_m", []):
        keep &= ~bounds_mask(points, bounds)
    cropped = np.flatnonzero(keep)
    _, reps = np.unique(np.floor(points[cropped] / config["voxel_m"]).astype(np.int64), axis=0, return_index=True)
    voxel_indices = cropped[reps]
    heights = signed_height(points[voxel_indices], plane)
    indices = voxel_indices[
        (heights > camera_config["object_min_height_m"]) & (heights < camera_config["object_max_height_m"])
    ]
    return dict(points_world_m=points[indices], source_camera=labels[indices], concatenated_indices=indices), dict(
        input_points={role: len(clouds[role]) for role in ROLES},
        cropped_points=len(cropped),
        voxel_points=len(voxel_indices),
        foreground_points=len(indices),
    )


def associate_seed(
    points: np.ndarray, seed: np.ndarray, other_seed: np.ndarray, config: dict, camera_config: dict
) -> tuple[np.ndarray, dict]:
    """Choose the unique seed-supported component; reject competing visible objects.

    Connectivity uses existing clustering. Other D415 ROI clusters veto merged
    components. Touching or entirely occluded unrelated objects still require a
    clear stationary workspace and visual review, as in the original workflow.
    """
    if not len(seed) or not len(points):
        raise ValueError("Empty D415 seed or fused foreground")
    groups = clusters(
        points, config["voxel_m"], camera_config["object_cluster_radius_m"], camera_config["object_min_points"]
    )
    distance = config["association_distance_m"]
    support = [float(np.mean(cKDTree(points[group]).query(seed)[0] <= distance)) for group in groups]
    eligible = [i for i, fraction in enumerate(support) if fraction >= config["min_seed_support"]]
    # Even a secondary partial association is ambiguous, never choose the largest.
    if len(eligible) != 1 or sum(fraction >= 0.15 for fraction in support) != 1:
        raise ValueError(f"Ambiguous or empty fused seed association; support={support}")
    index = eligible[0]
    selected = groups[index]
    if len(other_seed) and np.count_nonzero(cKDTree(points[selected]).query(other_seed)[0] <= distance) >= 3:
        raise ValueError(
            "Fused component connects another D415 foreground cluster; clear workspace or tighten clustering"
        )
    return selected, dict(cluster_id=index, seed_support=support, selected_points=len(selected))
