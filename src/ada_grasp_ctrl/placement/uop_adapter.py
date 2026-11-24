"""Local multi-plane adapter; run with the existing UOP environment, never patch upstream."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import random
import sys
from typing import Callable

import numpy as np

from .session import rotation, sha256, write_json


def align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a deterministic proper rotation, including parallel/antiparallel vectors."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
        raise ValueError("Alignment requires finite 3D vectors")
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-12:
        raise ValueError("Alignment vectors must be nonzero")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cross, dot = np.cross(a, b), np.clip(np.dot(a, b), -1, 1)
    if np.linalg.norm(cross) < 1e-9:
        if dot > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew * ((1 - dot) / np.dot(cross, cross))


def fit_candidates(
    points: np.ndarray,
    labels: np.ndarray,
    fit_plane: Callable,
    official_rotation: Callable,
    center_world: np.ndarray,
    radius: float,
    table_normal: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Fit each eligible official instance once, retaining all valid plane/rotation pairs.

    The installed upstream rule is positive label, at least 50 points and successful
    RANSAC (0.001 normalized distance, 3 samples, 1000 iterations). No ratio/ranking
    threshold or candidate deduplication is added. Nonfinite geometry is not displayable.
    """
    rows, diagnostics = [], []
    table_alignment = align_vectors(np.array([0, 0, 1]), table_normal)
    for instance in np.unique(labels):
        indices = np.flatnonzero(labels == instance)
        if instance <= 0 or len(indices) < 50:
            continue
        try:
            plane, inliers = fit_plane(points[indices])
            plane, inliers = np.asarray(plane, float), np.asarray(inliers, int)
            if plane.shape != (4,) or not np.isfinite(plane).all() or not len(inliers):
                raise ValueError("Empty inliers or nonfinite plane")
            plane_center = points[indices[inliers]].mean(0)
            normal = plane[:3].copy()
            if np.dot(normal, plane_center) < 0:
                normal *= -1
            normal /= np.linalg.norm(normal)
            if np.linalg.norm(np.cross(normal, [0, 0, -1])) < 1e-9:
                raw_rotation = align_vectors(normal, np.array([0, 0, -1]))
                numerical_fix = "parallel_axis_guard"
            else:
                raw_rotation = official_rotation(plane, plane_center, np.zeros(3))
                numerical_fix = None
            relative = rotation(table_alignment @ rotation(raw_rotation))
            world_plane = np.r_[plane[:3], radius * plane[3] - np.dot(plane[:3], center_world)]
            rows.append(
                dict(
                    relative_rotation=relative.tolist(),
                    support_normal=normal.tolist(),
                    plane_parameters=world_plane.tolist(),
                    instance_id=int(instance),
                    inlier_count=len(inliers),
                    inlier_ratio=len(inliers) / len(indices),
                    metadata=dict(
                        plane_equation="a*x+b*y+c*z+d=0; world metres",
                        plane_normalized=plane.tolist(),
                        plane_center_world_m=(plane_center * radius + center_world).tolist(),
                        instance_point_count=len(indices),
                        inlier_input_indices=indices[inliers].tolist(),
                        uop_rotation_to_minus_z=np.asarray(raw_rotation).tolist(),
                        table_alignment=table_alignment.tolist(),
                        numerical_fix=numerical_fix,
                    ),
                )
            )
        except Exception as error:
            diagnostics.append(dict(instance_id=int(instance), error=f"{type(error).__name__}: {error}"))
    return rows, diagnostics


def infer_snapshot(uop_root: Path, snapshot: Path, output: Path, seed: int = 4) -> dict:
    """Run the installed official segmentation/clustering exactly once on saved observations."""
    sys.path.insert(0, str(uop_root))
    import torch
    import open3d as o3d
    from placement_module import UOPModule
    from utils.placement_utils import calculate_placement_transform, plane_fitting_from_points

    mean_shift = importlib.import_module("utils.meanshift.mean_shift_gpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    o3d.utility.random.seed(seed)
    mean_shift.SEED_NUM = 128
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with np.load(snapshot, allow_pickle=False) as data:
        original = data["object_points_world_m"].copy()
        table_normal = data["table_normal_world"].copy()
    if original.ndim != 2 or original.shape[1] != 3 or not np.isfinite(original).all() or len(original) < 2048:
        raise ValueError("UOP requires at least 2048 finite observed points; no duplicate padding is used")
    indices = np.sort(np.random.default_rng(seed).choice(len(original), 2048, replace=False))
    points = original[indices]
    center = points.mean(0)
    radius = float(np.linalg.norm(points - center, axis=1).max())
    if radius < 1e-9:
        raise ValueError("Degenerate UOP input")
    normalized = ((points - center) / radius).astype(np.float32)
    checkpoint = uop_root / "uop_net/partial_best.pth"
    module = UOPModule(weight=str(checkpoint))
    with torch.inference_mode():
        labels, logits, embedded = module.inference_gpu(normalized)
    if not np.isfinite(logits).all() or not np.isfinite(embedded).all():
        raise ValueError("Nonfinite UOP output")
    rows, diagnostics = fit_candidates(
        normalized, labels, plane_fitting_from_points, calculate_placement_transform, center, radius, table_normal
    )
    np.savez_compressed(
        output / "inference.npz",
        points_normalized=normalized,
        source_indices=indices,
        center_world_m=center,
        normalization_radius_m=radius,
        labels=labels,
        logits=logits,
        embedded=embedded,
    )
    result = dict(
        schema_version=1,
        candidates=rows,
        diagnostics=diagnostics,
        seed=seed,
        inference_calls=1,
        checkpoint=dict(path=str(checkpoint), sha256=sha256(checkpoint)),
        upstream_source_sha256=sha256(uop_root / "placement_module/__init__.py"),
        adapter_source_sha256=sha256(Path(__file__)),
        snapshot_sha256=sha256(snapshot),
        status="completed" if rows else "no_plane",
    )
    write_json(output / "uop_result.json", result)
    return result


def main() -> None:
    """Subprocess boundary keeps Ada and UOP dependencies in their existing environments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uop-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    infer_snapshot(args.uop_root, args.snapshot, args.output, args.seed)


if __name__ == "__main__":
    main()
