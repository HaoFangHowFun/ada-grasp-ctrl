"""Experimental passive camera CLI: capture, fit, accept, extract, replay and preview."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

from ada_grasp_ctrl.paths import resolve_external_root, resolve_from_root
from .fixed_camera_geometry import (
    clusters,
    file_hash,
    fit_plane,
    load_extrinsics,
    project_color,
    rigid,
    roi_mask,
    signed_height,
    transform,
)


def write_json(path: Path, value: Any) -> None:
    """Write portable finite JSON, rejecting accidental NaN/Infinity."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    """Load one explicit JSON record."""
    return json.loads(path.read_text())


def load_config(path: Path) -> dict:
    """Load the declared experimental config and reject invalid numerical limits."""
    config = yaml.safe_load(path.read_text())
    for key, value in config.items():
        if isinstance(value, (float, int)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key}: expected finite nonnegative value")
    for key in ("min_points", "ransac_max_points", "ransac_iterations", "object_min_points"):
        if config[key] < 3 or int(config[key]) != config[key]:
            raise ValueError(f"{key}: expected integer >= 3")
    for key in ("ransac_threshold_m", "object_voxel_m", "object_cluster_radius_m", "capture_timeout_s"):
        if config[key] <= 0:
            raise ValueError(f"{key}: must be positive")
    for key in ("min_valid_fraction", "min_inlier_ratio"):
        if not 0 < config[key] <= 1:
            raise ValueError(f"{key}: must be in (0, 1]")
    if config["min_depth_m"] >= config["max_depth_m"] or config["world_frame"] != "world":
        raise ValueError("Require ordered depth interval and world frame")
    if config["max_tilt_deg"] >= 90 or config["horizontal_tolerance_deg"] > config["max_tilt_deg"]:
        raise ValueError("Invalid plane tilt tolerances")
    return config


def load_snapshot(path: Path) -> tuple[dict, dict]:
    """Verify saved snapshot integrity and camera/world round-trip before reuse."""
    metadata = read_json(path / "snapshot.json")
    if metadata["schema_version"] != 1 or metadata["unit"] != "m" or metadata["frame_id"] != "world":
        raise ValueError("Unsupported snapshot schema/frame/units")
    if file_hash(path / "snapshot.npz") != metadata["snapshot_sha256"]:
        raise ValueError("Snapshot hash mismatch")
    with np.load(path / "snapshot.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    matrix = rigid(arrays["T_world_source"])
    if not np.allclose(matrix, rigid(arrays["T_world_color"]) @ rigid(arrays["T_color_source"]), atol=1e-8):
        raise ValueError("Snapshot transform composition mismatch")
    source = arrays["points_source_m"][arrays["valid_source_indices"]]
    if not np.allclose(transform(source, matrix), arrays["points_world_m"], atol=1e-7, rtol=0):
        raise ValueError("Snapshot source/world round-trip mismatch")
    if not np.allclose(transform(source, arrays["T_color_source"]), arrays["points_color_m"], atol=1e-7, rtol=0):
        raise ValueError("Snapshot source/color round-trip mismatch")
    return arrays, metadata


def load_plane(path: Path, require_accepted: bool = False) -> dict:
    """Validate a saved plane and optional acceptance state without approximating tilt."""
    plane = read_json(path)
    if plane.get("schema_version") != 1 or plane.get("frame_id") != "world" or plane.get("unit") != "m":
        raise ValueError("Unsupported plane schema/frame/units")
    signed_height(np.zeros((1, 3)), plane)
    if require_accepted and (plane.get("status") != "accepted" or not plane.get("quality_passed")):
        raise ValueError("A quality-passed, explicitly accepted plane is required")
    return plane


def choose_roi(arrays: dict, supplied: list[int] | None) -> list[int]:
    """Use explicit ROI coordinates or a single manual RGB selection window."""
    if supplied is not None:
        return supplied
    import os
    import cv2

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("No display: inspect rgb.png and supply --roi x0 y0 x1 y1")
    x, y, width, height = cv2.selectROI("Select table/object ROI; Enter accepts", arrays["rgb"][:, :, ::-1])
    cv2.destroyAllWindows()
    if not width or not height:
        raise ValueError("ROI selection cancelled")
    return [int(x), int(y), int(x + width), int(y + height)]


def select_points(arrays: dict, metadata: dict, roi: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Project existing XYZ into RGB; do not re-deproject cloud coordinates."""
    pixels = project_color(arrays["points_color_m"], metadata["color_info"])
    h, w = arrays["rgb"].shape[:2]
    mask = roi_mask(pixels, roi, w, h)
    return np.flatnonzero(mask), pixels


def save_overlay(path: Path, arrays: dict, pixels: np.ndarray, indices: np.ndarray, roi: list[int]) -> None:
    """Save the exact selected points over RGB plus the manual ROI rectangle."""
    import cv2

    image = arrays["rgb"][:, :, ::-1].copy()
    projected = np.rint(pixels[indices]).astype(int)
    valid = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < image.shape[1])
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < image.shape[0])
    )
    projected = projected[valid]
    image[projected[:, 1], projected[:, 0]] = (0, 255, 0)
    cv2.rectangle(image, tuple(roi[:2]), tuple(roi[2:]), (0, 180, 255), 2)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write {path}")


def fit_snapshot(snapshot: Path, output: Path, config: dict, roi: list[int] | None) -> dict:
    """Fit and save an unaccepted plane, source support indices and RGB overlay."""
    arrays, metadata = load_snapshot(snapshot)
    roi = choose_roi(arrays, roi)
    selected, pixels = select_points(arrays, metadata, roi)
    plane, mask = fit_plane(arrays["points_world_m"][selected], config)
    plane.update(
        snapshot=str(snapshot),
        snapshot_sha256=metadata["snapshot_sha256"],
        roi=roi,
        capture_stamp_s=metadata["capture_stamp_s"],
        serial=metadata["serial"],
        extrinsics=metadata["extrinsics"],
        source_frame=metadata["source_frame"],
        color_info=metadata["color_info"],
        depth_info=metadata["depth_info"],
        topics=metadata["topics"],
        T_world_source=metadata["T_world_source"],
        T_world_color=metadata["T_world_color"],
        config=config,
    )
    indices = selected[mask]
    np.save(output / "inlier_indices.npy", indices)
    plane["inliers_file"] = str(output / "inlier_indices.npy")
    plane["inliers_sha256"] = file_hash(output / "inlier_indices.npy")
    write_json(output / "table_plane.json", plane)
    save_overlay(output / "inlier_overlay.png", arrays, pixels, indices, roi)
    return plane


def accept_plane(plane_path: Path, evidence_path: Path, output: Path, reviewed: bool) -> dict:
    """Accept only reviewed geometry with independent references and repeated captures.

    Evidence contains measured world XYZ table points, two independent repeat plane
    paths, and explicit confirmation of unchanged camera/base and world definition.
    """
    plane = load_plane(plane_path)
    if not reviewed or not plane["quality_passed"]:
        raise ValueError("Explicit --reviewed and a quality-passed preview are required")
    evidence = read_json(evidence_path)
    if evidence.get("camera_base_unchanged") is not True or evidence.get("world_definition_confirmed") is not True:
        raise ValueError("Evidence must confirm camera/base relation and world definition")
    config = plane["config"]
    references = np.asarray(evidence.get("reference_points_world_m"), dtype=float)
    if (
        references.ndim != 2
        or references.shape[1] != 3
        or len(references) < config.get("min_reference_points", 3)
        or not np.isfinite(references).all()
    ):
        raise ValueError("Provide at least three independently measured table reference points in world meters")
    if np.linalg.matrix_rank(references - references.mean(axis=0), tol=0.01) < 2:
        raise ValueError("Reference points must span a table region, not a line")
    camera_references = np.asarray(evidence.get("reference_points_color_m"), dtype=float)
    if camera_references.shape != references.shape or not np.isfinite(camera_references).all():
        raise ValueError("Provide corresponding measured color-optical XYZ for the independent world references")
    reference_xyz_errors = np.linalg.norm(transform(camera_references, plane["T_world_color"]) - references, axis=1)
    if reference_xyz_errors.max() > config["reference_tolerance_m"]:
        raise ValueError("Independent camera-to-world XYZ reference check failed")
    errors = np.abs(signed_height(references, plane))
    if errors.max() > config["reference_tolerance_m"]:
        raise ValueError(f"Independent reference error {errors.max():.5f} m exceeds tolerance")
    repeats = evidence.get("repeat_plane_files", [])
    if len(repeats) < config.get("min_repeat_captures", 2):
        raise ValueError("Provide at least two additional captures of distributed table regions")
    seen = {plane["snapshot_sha256"]}
    centers = [np.asarray(plane["centroid_world_m"])]
    comparisons = []
    for entry in repeats:
        path = resolve_from_root(entry)
        other = load_plane(path)
        if not other["quality_passed"] or other["extrinsics"]["sha256"] != plane["extrinsics"]["sha256"]:
            raise ValueError("Repeat plane quality/calibration mismatch")
        if other["snapshot_sha256"] in seen:
            raise ValueError("Repeat evidence must use independently captured snapshots")
        seen.add(other["snapshot_sha256"])
        center = np.asarray(other["centroid_world_m"])
        if min(np.linalg.norm(center - old) for old in centers) < config.get("repeat_region_separation_m", 0.05):
            raise ValueError("Repeat observed regions must be separated by at least 5 cm")
        centers.append(center)
        angle = float(np.rad2deg(np.arccos(np.clip(np.dot(other["normal_xyz"], plane["normal_xyz"]), -1, 1))))
        error = float(np.max(np.abs(signed_height(references, other) - signed_height(references, plane))))
        if angle > config["repeat_normal_tolerance_deg"] or error > config["repeat_height_tolerance_m"]:
            raise ValueError("Repeat normal/height consistency failed")
        comparisons.append(dict(path=str(path), sha256=file_hash(path), normal_error_deg=angle, height_error_m=error))
    plane.update(
        status="accepted",
        accepted_at=datetime.now(timezone.utc).isoformat(),
        validation=dict(
            evidence=evidence,
            evidence_sha256=file_hash(evidence_path),
            reference_errors_m=errors.tolist(),
            reference_xyz_errors_m=reference_xyz_errors.tolist(),
            repeats=comparisons,
        ),
        invalidation="Revalidate after camera, table or world/base relation changes",
    )
    plane["extrinsics"]["independently_verified"] = True
    write_json(output / "table_plane.json", plane)
    # Reload the persisted full plane and prove signed heights are unchanged.
    reloaded = load_plane(output / "table_plane.json", require_accepted=True)
    if not np.array_equal(signed_height(references, plane), signed_height(references, reloaded)):
        raise RuntimeError("Plane serialization changed representative signed heights")
    return plane


def extract_object(
    snapshot: Path,
    plane_path: Path,
    output: Path,
    config: dict,
    roi: list[int] | None,
    cluster_id: int | None,
    allow_unaccepted: bool,
) -> dict:
    """Extract one explicitly selected connected object above the full measured plane."""
    arrays, metadata = load_snapshot(snapshot)
    plane = load_plane(plane_path, require_accepted=not allow_unaccepted)
    if plane["extrinsics"]["sha256"] != metadata["extrinsics"]["sha256"] or plane["serial"] != metadata["serial"]:
        raise ValueError("Object capture and table must share camera and calibration")
    if not allow_unaccepted and metadata["capture_stamp_s"] <= plane["capture_stamp_s"]:
        raise ValueError("Object capture must be newer than the accepted empty-table capture")
    roi = choose_roi(arrays, roi)
    selected, pixels = select_points(arrays, metadata, roi)
    heights = signed_height(arrays["points_world_m"][selected], plane)
    selected = selected[(heights > config["object_min_height_m"]) & (heights < config["object_max_height_m"])]
    groups = clusters(
        arrays["points_world_m"][selected],
        config["object_voxel_m"],
        config["object_cluster_radius_m"],
        config["object_min_points"],
    )
    write_json(
        output / "clusters.json",
        {
            "clusters": [
                dict(
                    id=i,
                    point_count=len(g),
                    centroid_world_m=arrays["points_world_m"][selected[g]].mean(axis=0).tolist(),
                )
                for i, g in enumerate(groups)
            ]
        },
    )
    if not groups:
        raise ValueError("No object cluster above table in selected ROI")
    if cluster_id is None and len(groups) != 1:
        for i, group in enumerate(groups):
            save_overlay(output / f"cluster_{i}.png", arrays, pixels, selected[group], roi)
        raise ValueError(f"{len(groups)} clusters found; inspect cluster overlays and rerun with --cluster ID")
    cluster_id = 0 if cluster_id is None else cluster_id
    if not 0 <= cluster_id < len(groups):
        raise ValueError("Invalid cluster ID")
    indices = selected[groups[cluster_id]]
    np.savez_compressed(
        output / "object.npz",
        points_world_m=arrays["points_world_m"][indices],
        points_color_m=arrays["points_color_m"][indices],
        source_indices=indices,
        T_world_color=arrays["T_world_color"],
    )
    save_overlay(output / "object_overlay.png", arrays, pixels, indices, roi)
    result = dict(
        schema_version=1,
        unit="m",
        frame_id="world",
        source_snapshot=str(snapshot),
        snapshot_sha256=metadata["snapshot_sha256"],
        plane_path=str(plane_path),
        plane_sha256=file_hash(plane_path),
        table_status=plane["status"],
        roi=roi,
        serial=metadata["serial"],
        extrinsics=metadata["extrinsics"],
        capture_stamp_s=metadata["capture_stamp_s"],
        point_count=len(indices),
        cluster_id=cluster_id,
        object_sha256=file_hash(output / "object.npz"),
        config=config,
    )
    write_json(output / "object.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run an isolated experiment; preflight failures exit 2 and preserve diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["capture", "fit", "accept", "object", "preview"])
    parser.add_argument("--config", default="src/ada_grasp_ctrl/config/experiments/fixed_camera.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--snapshot")
    parser.add_argument("--object-root", help="Optional object run for RViz preview")
    parser.add_argument("--extrinsics")
    parser.add_argument("--plane")
    parser.add_argument("--evidence")
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument(
        "--allow-unaccepted", action="store_true", help="Object preview only; records unaccepted status"
    )
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--cluster", type=int)
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args(argv)
    output = None
    try:
        config = load_config(resolve_from_root(args.config))
        if args.command == "preview":
            if not args.plane:
                raise ValueError("--plane is required")
            plane = load_plane(resolve_from_root(args.plane))
            arrays, _ = load_snapshot(resolve_from_root(plane["snapshot"]))
            from .fixed_camera_ros import publish_preview

            if args.object_root:
                object_root = resolve_from_root(args.object_root)
                info = read_json(object_root / "object.json")
                if info["plane_sha256"] != file_hash(resolve_from_root(args.plane)):
                    raise ValueError("Object and preview plane differ")
                if info["object_sha256"] != file_hash(object_root / "object.npz"):
                    raise ValueError("Object hash mismatch")
                with np.load(object_root / "object.npz", allow_pickle=False) as archive:
                    points = archive["points_world_m"]
                topic = "/ada/object/points"
            else:
                selected, _ = select_points(
                    arrays, read_json(resolve_from_root(plane["snapshot"]) / "snapshot.json"), plane["roi"]
                )
                points = arrays["points_world_m"][selected]
                points = points[np.abs(signed_height(points, plane)) <= plane["config"]["ransac_threshold_m"]]
                topic = "/ada/table/points"
            publish_preview(points, plane, args.duration, topic)
            return 0
        if not args.output_root:
            raise ValueError("--output-root must name a new run directory")
        destination = resolve_external_root("output", args.output_root)
        destination.mkdir(parents=True, exist_ok=False)
        output = destination
        (output / "run_manifest.yaml").write_text(
            yaml.safe_dump(
                dict(command=vars(args), resolved_config=config, started_at=datetime.now(timezone.utc).isoformat()),
                sort_keys=True,
            )
        )
        if args.command == "capture":
            if not args.extrinsics:
                raise ValueError("--extrinsics is required")
            matrix, provenance = load_extrinsics(resolve_from_root(args.extrinsics))
            from .fixed_camera_ros import capture

            arrays, result = capture(config, matrix, provenance, output)
            np.savez_compressed(output / "snapshot.npz", **arrays)
            result["snapshot_sha256"] = file_hash(output / "snapshot.npz")
            write_json(output / "snapshot.json", result)
            import cv2

            if not cv2.imwrite(str(output / "rgb.png"), arrays["rgb"][:, :, ::-1]):
                raise RuntimeError("RGB snapshot write failed")
        elif args.command == "fit":
            if not args.snapshot:
                raise ValueError("--snapshot is required")
            result = fit_snapshot(resolve_from_root(args.snapshot), output, config, args.roi)
        elif args.command == "accept":
            if not args.plane or not args.evidence:
                raise ValueError("--plane and --evidence are required")
            result = accept_plane(
                resolve_from_root(args.plane), resolve_from_root(args.evidence), output, args.reviewed
            )
        else:
            if not args.snapshot or not args.plane:
                raise ValueError("--snapshot and --plane are required")
            result = extract_object(
                resolve_from_root(args.snapshot),
                resolve_from_root(args.plane),
                output,
                config,
                args.roi,
                args.cluster,
                args.allow_unaccepted,
            )
        rejected = result.get("quality_passed") is False
        write_json(
            output / "run_report.json", {"status": "quality_rejected" if rejected else "completed", "result": result}
        )
        print(
            json.dumps(
                {
                    "output_root": str(output),
                    "status": result.get("status", "completed"),
                    "quality": result.get("quality"),
                    "quality_failures": result.get("quality_failures"),
                },
                indent=2,
            )
        )
        return 1 if rejected else 0
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        if output is not None:
            write_json(output / "run_report.json", {"status": "preflight_failed", "error": str(exc)})
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
