"""ROS-independent geometry for measured tables and single-object snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def file_hash(path: Path) -> str:
    """Return a streaming SHA256 identifier for an input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rigid(value: Any, name: str = "transform") -> np.ndarray:
    """Validate a finite homogeneous proper rigid transform; never repair it."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: expected finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError(f"{name}: invalid homogeneous row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5, rtol=0
    ):
        raise ValueError(f"{name}: rotation must be orthonormal with determinant +1")
    return matrix


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply one validated source-to-target transform to row-vector XYZ meters."""
    matrix = rigid(matrix)
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def load_extrinsics(path: Path) -> tuple[np.ndarray, dict]:
    """Read the supplied color-optical-to-world calibration and provenance."""
    data = json.loads(path.read_text())
    matrix = rigid(data["T_world_cam"], "T_world_cam")
    if data.get("converged") is False:
        raise ValueError("Extrinsics solver did not converge")
    frame = data.get("frame", "").lower()
    if "optical" not in frame or not ("color" in frame or "colour" in frame) or "world" not in frame:
        raise ValueError("Extrinsics must explicitly identify color optical and world frames")
    return matrix, {
        "path": str(path),
        "sha256": file_hash(path),
        "declared_frame": data["frame"],
        "independently_verified": False,
    }


def signed_height(points: np.ndarray, plane: dict) -> np.ndarray:
    """Compute signed orthogonal distance; world +Z-facing side is positive."""
    n = np.asarray(plane["normal_xyz"], dtype=float)
    d = float(plane["d"])
    if (
        n.shape != (3,)
        or not np.isfinite(n).all()
        or not np.isfinite(d)
        or not np.isclose(np.linalg.norm(n), 1, atol=1e-6, rtol=0)
        or n[2] <= 0
    ):
        raise ValueError("Plane must have a finite unit upward normal and finite d")
    return np.asarray(points) @ n + d


def project_color(points_color: np.ndarray, info: dict) -> np.ndarray:
    """Project optical XYZ with the live CameraInfo plumb-bob distortion model."""
    import cv2

    if info["distortion_model"] not in ("plumb_bob", "rational_polynomial"):
        raise ValueError(f"Unsupported color distortion model: {info['distortion_model']}")
    points = np.asarray(points_color, dtype=np.float64)
    pixels = np.full((len(points), 2), np.nan)
    good = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    k = np.asarray(info["k"], dtype=float).reshape(3, 3)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid color CameraInfo K")
    if good.any():
        pixels[good] = cv2.projectPoints(points[good], np.zeros(3), np.zeros(3), k, np.asarray(info["d"], dtype=float))[
            0
        ].reshape(-1, 2)
    return pixels


def roi_mask(pixels: np.ndarray, roi: list[int], width: int, height: int) -> np.ndarray:
    """Select an explicit half-open RGB rectangle after validating image bounds."""
    if len(roi) != 4 or any(int(v) != v for v in roi):
        raise ValueError("ROI must be four integer coordinates x0 y0 x1 y1")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI is outside the RGB image or empty")
    return (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= x0)
        & (pixels[:, 0] < x1)
        & (pixels[:, 1] >= y0)
        & (pixels[:, 1] < y1)
    )


def fit_plane(points: np.ndarray, config: dict) -> tuple[dict, np.ndarray]:
    """Fit RANSAC then least-squares inliers, reporting explicit quality failures.

    Args:
        points: Finite world XYZ meters from a manually selected RGB ROI.
        config: Public fitting and quality thresholds.

    Returns:
        Full unit plane and final inlier mask; failed quality remains a preview.

    Raises:
        ValueError: If input is insufficient, degenerate, or configuration invalid.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Plane points must be finite Nx3")
    if len(points) < config["min_points"]:
        raise ValueError(f"Insufficient ROI points: {len(points)} < {config['min_points']}")
    rng = np.random.default_rng(config["seed"])
    sample = points[rng.choice(len(points), min(len(points), config["ransac_max_points"]), replace=False)]
    threshold = config["ransac_threshold_m"]
    best = None
    best_count = 0
    for _ in range(config["ransac_iterations"]):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-10:
            continue
        n /= length
        d = -float(n @ a)
        count = np.count_nonzero(np.abs(sample @ n + d) <= threshold)
        if count > best_count:
            best, best_count = (n, d), count
    if best is None:
        raise ValueError("Degenerate points: no plane")
    n, d = best
    mask = np.abs(points @ n + d) <= threshold
    for _ in range(5):
        inliers = points[mask]
        if len(inliers) < 3:
            raise ValueError("Too few RANSAC inliers")
        center = inliers.mean(axis=0)
        _, singular, vh = np.linalg.svd(inliers - center, full_matrices=False)
        if singular[1] < 1e-8:
            raise ValueError("Plane points are collinear")
        n = vh[-1]
        if n[2] < 0:
            n = -n
        d = -float(n @ center)
        updated = np.abs(points @ n + d) <= threshold
        if np.array_equal(updated, mask):
            break
        mask = updated
    # Refine on exactly the saved final inlier set.
    inliers = points[mask]
    center = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    n = vh[-1] * (1 if vh[-1, 2] >= 0 else -1)
    d = -float(n @ center)
    residual = np.abs(inliers @ n + d)
    span = np.ptp((inliers - center) @ vh[:2].T, axis=0)
    tilt = float(np.rad2deg(np.arccos(np.clip(n[2], -1, 1))))
    quality = dict(
        point_count=len(points),
        inlier_count=int(mask.sum()),
        inlier_ratio=float(mask.mean()),
        rms_m=float(np.sqrt(np.mean(residual**2))),
        p95_m=float(np.percentile(residual, 95)),
        max_residual_m=float(residual.max()),
        tilt_deg=tilt,
        span_m=span.tolist(),
    )
    failures = []
    for key, limit, sign in [
        ("inlier_count", config["min_points"], -1),
        ("inlier_ratio", config["min_inlier_ratio"], -1),
        ("rms_m", config["max_rms_m"], 1),
        ("p95_m", config["max_p95_m"], 1),
        ("max_residual_m", config["max_residual_m"], 1),
        ("tilt_deg", config["max_tilt_deg"], 1),
    ]:
        if sign * quality[key] > sign * limit:
            failures.append(f"{key}={quality[key]:.6g} violates limit {limit}")
    if min(span) < config["min_span_m"]:
        failures.append("Observed plane coverage is too small")
    if not config["height_range_m"][0] <= center[2] <= config["height_range_m"][1]:
        failures.append("Observed table height is outside configured range")
    plane = dict(
        schema_version=1,
        frame_id="world",
        unit="m",
        normal_xyz=n.tolist(),
        d=d,
        normal_convention="dot(normal, world +Z) > 0; signed height positive above table",
        quality=quality,
        quality_passed=not failures,
        quality_failures=failures,
        status="preview_unaccepted",
        centroid_world_m=center.tolist(),
        observed_bounds_world_m=[inliers.min(axis=0).tolist(), inliers.max(axis=0).tolist()],
        observed_tangent_axes=vh[:2].tolist(),
        observed_span_m=span.tolist(),
        bounds_semantics="observed support only; not full table boundaries",
        horizontal_tolerance_deg=config["horizontal_tolerance_deg"],
    )
    if tilt <= config["horizontal_tolerance_deg"]:
        plane["z_table"] = -d / float(n[2])
        plane["z_table_scope"] = "optional approximation over observed support only; retain full plane"
    return plane, mask


def clusters(points: np.ndarray, voxel_m: float, radius_m: float, min_points: int) -> list[np.ndarray]:
    """Return deterministic voxel-connected components, largest first, as input indices."""
    if voxel_m <= 0 or radius_m < voxel_m or min_points < 1:
        raise ValueError("Invalid object clustering thresholds")
    if len(points) == 0:
        return []
    _, representatives, inverse = np.unique(
        np.floor(points / voxel_m).astype(np.int64), axis=0, return_index=True, return_inverse=True
    )
    coarse = points[representatives]
    tree = cKDTree(coarse)
    labels = np.full(len(coarse), -1, dtype=int)
    label = 0
    for seed in range(len(coarse)):
        if labels[seed] >= 0:
            continue
        labels[seed] = label
        pending = [seed]
        while pending:
            current = pending.pop()
            neighbors = np.asarray(tree.query_ball_point(coarse[current], radius_m), dtype=int)
            new = neighbors[labels[neighbors] < 0]
            labels[new] = label
            pending.extend(new.tolist())
        label += 1
    expanded = labels[inverse]
    groups = [np.flatnonzero(expanded == i) for i in range(label)]
    return sorted((g for g in groups if len(g) >= min_points), key=lambda g: (-len(g), int(g[0])))


def validate_timing(stamps: dict[str, float], now: float, max_age: float, tolerance: float) -> tuple[dict, dict]:
    """Reject stale, future, or unsynchronized sensor bundles before processing."""
    if not np.isfinite([now, *stamps.values()]).all() or "cloud" not in stamps:
        raise ValueError("Invalid sensor timestamps")
    ages = {key: now - value for key, value in stamps.items()}
    offsets = {key: value - stamps["cloud"] for key, value in stamps.items()}
    if any(age < -0.05 or age > max_age for age in ages.values()):
        raise ValueError(f"stale or future timestamps: {ages}")
    if any(abs(offset) > tolerance for offset in offsets.values()):
        raise ValueError(f"unsynchronized timestamps: {offsets}")
    return ages, offsets


def valid_depth_mask(points: np.ndarray, depth_pixels: int, config: dict) -> tuple[np.ndarray, float, float]:
    """Reject insufficient valid metric XYZ, including sparse unorganized clouds."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or depth_pixels <= 0:
        raise ValueError("Invalid pointcloud dimensions or depth CameraInfo resolution")
    mask = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= config["min_depth_m"])
        & (points[:, 2] <= config["max_depth_m"])
    )
    fraction = float(mask.mean()) if len(mask) else 0.0
    coverage = float(mask.sum() / depth_pixels)
    if mask.sum() < config["min_points"] or min(fraction, coverage) < config["min_valid_fraction"]:
        raise ValueError(
            f"insufficient valid depth: {mask.sum()} points; fraction={fraction:.3f}, coverage={coverage:.3f}"
        )
    return mask, fraction, coverage
