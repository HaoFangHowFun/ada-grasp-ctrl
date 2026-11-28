"""One-shot three-camera observation, retaining the existing D415 selection UI."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

from ada_grasp_ctrl.calibration import DEFAULT_CALIBRATION
from ada_grasp_ctrl.paths import resolve_from_root
from .fixed_camera import (
    choose_roi,
    extract_object,
    load_config as load_fixed_config,
    load_plane,
    read_json,
    write_json,
)
from .fixed_camera_geometry import clusters, file_hash, signed_height, transform
from .multi_camera_calibration import load_cameras
from .multi_camera_geometry import ROLES, SERIALS, associate_seed, fuse_clouds

DEFAULT_CONFIG = "src/ada_grasp_ctrl/config/experiments/multi_camera.yaml"


def load_config(path: Path) -> dict:
    """Validate the opt-in configuration before any subscription or inference."""
    config = yaml.safe_load(path.read_text())
    selected = config.get("accepted_pending_calibrations", {})
    if not isinstance(selected, dict) or any(
        role not in ROLES[1:]
        or not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for role, value in selected.items()
    ):
        raise ValueError("accepted_pending_calibrations must map D405 roles to exact entry SHA256 values")
    for key in (
        "capture_timeout_s",
        "max_age_s",
        "sync_tolerance_s",
        "voxel_m",
        "association_distance_m",
        "min_seed_support",
    ):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0.5 < config["min_seed_support"] <= 1:
        raise ValueError("min_seed_support must be in (0.5, 1]")
    if set(config["cameras"]) != set(ROLES):
        raise ValueError("Multi-camera mode requires exactly the three configured roles")
    for role in ROLES:
        camera = config["cameras"][role]
        if (
            not np.isfinite(camera["min_points"])
            or camera["min_points"] < 3
            or int(camera["min_points"]) != camera["min_points"]
        ):
            raise ValueError(f"{role}: min_points must be an integer >= 3")
        if not np.isfinite(camera["min_valid_fraction"]) or not 0 < camera["min_valid_fraction"] <= 1:
            raise ValueError(f"{role}: min_valid_fraction must be in (0, 1]")
        if camera["serial"] != SERIALS[role]:
            raise ValueError(f"{role}: serial differs from the verified deployment")
        if (
            not np.isfinite([camera["min_depth_m"], camera["max_depth_m"]]).all()
            or not 0 < camera["min_depth_m"] < camera["max_depth_m"]
        ):
            raise ValueError(f"{role}: invalid depth range")
    from .multi_camera_geometry import bounds_mask

    for bounds in ([config["workspace_bounds_m"]] if config.get("workspace_bounds_m") is not None else []) + config.get(
        "exclude_bounds_m", []
    ):
        bounds_mask(np.zeros((1, 3)), bounds)
    return config


def calibrated_arrays(raw: dict, metadata: dict, cameras: dict, provenance: dict) -> tuple[dict, dict, dict]:
    """Apply one optical-to-leftbase transform per camera, preserving D415 geometry."""
    clouds = {}
    for role in ROLES:
        source = raw[role]
        calibration = cameras[role]
        world_source = calibration["matrix"]
        if calibration["calibrated_optical_frame"] == metadata[role]["color_frame"]:
            world_source = world_source @ source["T_color_source"]
        elif calibration["calibrated_optical_frame"] != metadata[role]["source_frame"]:
            raise ValueError(f"{role}: captured frame differs from calibrated optical frame")
        source["points_world_m"] = transform(source["points_source_m"], world_source)
        source["T_world_source"] = world_source
        clouds[role] = source["points_world_m"]
    fixed = raw["fixed"]
    arrays = dict(
        points_source_m=fixed["points_source_m"],
        valid_source_indices=np.arange(len(fixed["points_source_m"])),
        points_color_m=transform(fixed["points_source_m"], fixed["T_color_source"]),
        points_world_m=clouds["fixed"],
        rgb=fixed["rgb"],
        T_color_source=fixed["T_color_source"],
        T_world_source=fixed["T_world_source"],
        T_world_color=cameras["fixed"]["matrix"],
    )
    info = {
        **metadata["fixed"],
        "schema_version": 1,
        "unit": "m",
        "frame_id": "world",
        "extrinsics": provenance,
        "capture_stamp_s": metadata["fixed"]["timestamps_s"]["depth"],
        "T_world_color": arrays["T_world_color"].tolist(),
        "T_world_source": arrays["T_world_source"].tolist(),
        "transform_mode": "leftbase calibration @ verified internal TF; applied once",
    }
    return arrays, info, clouds


def table_for_capture(plane_path: Path, output: Path, metadata: dict) -> Path:
    """Rebind unchanged table geometry to an additive calibration snapshot.

    Existing single-camera extraction binds the entire calibration-file hash.
    Adding D405 entries changes that hash without moving the D415 or table. This
    per-run copy explicitly verifies the D415 matrix and records the old file;
    the authoritative table and its acceptance record are never rewritten.
    """
    plane = load_plane(plane_path)
    if plane["serial"] != metadata["serial"] or not np.allclose(
        plane["T_world_color"], metadata["T_world_color"], atol=1e-8, rtol=0
    ):
        raise ValueError("Table/D415 transform mismatch; revalidate the table before multi-camera capture")
    result = deepcopy(plane)
    result.update(
        extrinsics=metadata["extrinsics"],
        source_table=dict(path=str(plane_path), sha256=file_hash(plane_path)),
        calibration_rebinding="Unchanged measured table and D415 matrix; additive camera calibration snapshot",
    )
    write_json(output, result)
    return output


def other_object_seed(arrays: dict, seed_indices: np.ndarray, plane: dict, camera_config: dict) -> np.ndarray:
    """Use other D415 connected components to veto merges with visible neighbors."""
    points = arrays["points_world_m"]
    heights = signed_height(points, plane)
    indices = np.flatnonzero(
        (heights > camera_config["object_min_height_m"]) & (heights < camera_config["object_max_height_m"])
    )
    groups = clusters(
        points[indices],
        camera_config["object_voxel_m"],
        camera_config["object_cluster_radius_m"],
        camera_config["object_min_points"],
    )
    rejected = [indices[group] for group in groups if not np.intersect1d(indices[group], seed_indices).size]
    return points[np.concatenate(rejected)] if rejected else np.empty((0, 3))


def observe(args: argparse.Namespace, root: Path, report: dict) -> Path | None:
    """Capture once, select the D415 seed, then persist one unnormalized fused object."""
    from .multi_camera_ros import capture

    config = load_config(resolve_from_root(args.config))
    camera_config = load_fixed_config(resolve_from_root(args.camera_config))
    if args.command == "observe":
        for key in ("min_depth_m", "max_depth_m", "min_points", "min_valid_fraction"):
            config["cameras"]["fixed"][key] = camera_config[key]
    cameras, provenance = None, None
    if args.command == "observe":
        master = resolve_from_root(args.calibration)
        cameras = load_cameras(master, config)
        report["stationary_camera_calibration"] = {
            role: {key: cameras[role][key] for key in ("status", "independent_validation_passed")} for role in ROLES[1:]
        }
        if not args.plane:
            raise ValueError("--plane is required for object observation")
        saved = root / "calibration.json"
        shutil.copyfile(master, saved)
        provenance = dict(
            path=str(saved),
            sha256=file_hash(saved),
            source_path=str(master),
            calibration_id=read_json(master)["calibration_id"],
            stationary_camera_calibration=report["stationary_camera_calibration"],
        )
    raw, metadata = capture(config, report.setdefault("capture", {}), observe_s=args.observe_seconds)
    report["timings"] = dict(capture_s=report["capture"]["capture_s"])
    capture_root = root / "capture"
    capture_root.mkdir()
    write_json(capture_root / "cameras.json", metadata)
    if args.command == "probe":
        for role in ROLES:
            np.savez_compressed(capture_root / f"{role}.npz", **raw[role])
        report.update(status="streams_verified", calibrated_alignment_verified=False)
        return None
    began = time.monotonic()
    arrays, fixed_metadata, clouds = calibrated_arrays(raw, metadata, cameras, provenance)
    transform_s = time.monotonic() - began
    np.savez_compressed(capture_root / "snapshot.npz", **arrays)
    fixed_metadata["snapshot_sha256"] = file_hash(capture_root / "snapshot.npz")
    write_json(capture_root / "snapshot.json", fixed_metadata)
    import cv2

    if not cv2.imwrite(str(capture_root / "rgb.png"), arrays["rgb"][:, :, ::-1]):
        raise RuntimeError("D415 RGB write failed")
    for role in ROLES[1:]:
        np.savez_compressed(capture_root / f"{role}.npz", **raw[role])
    plane_path = table_for_capture(resolve_from_root(args.plane), root / "table_plane.json", fixed_metadata)
    plane = load_plane(plane_path)
    began = time.monotonic()
    fused, counts = fuse_clouds(clouds, arrays, fixed_metadata, plane, config, camera_config)
    report["timings"]["transform_fusion_s"] = transform_s + time.monotonic() - began
    report["point_counts"] = counts
    began = time.monotonic()
    roi = choose_roi(arrays, args.roi)
    report["timings"]["human_roi_s"] = time.monotonic() - began
    began = time.monotonic()
    seed_root = root / "d415_seed"
    seed_root.mkdir()
    seed_info = extract_object(capture_root, plane_path, seed_root, camera_config, roi, args.cluster, True)
    with np.load(seed_root / "object.npz", allow_pickle=False) as archive:
        seed, seed_indices = archive["points_world_m"], archive["source_indices"]
    other_seed = other_object_seed(arrays, seed_indices, plane, camera_config)
    selected, association = associate_seed(fused["points_world_m"], seed, other_seed, config, camera_config)
    report["association"] = association
    report["timings"]["segmentation_s"] = time.monotonic() - began
    report["timings"]["perception_s"] = sum(
        report["timings"][key] for key in ("capture_s", "transform_fusion_s", "segmentation_s")
    )
    np.savez_compressed(capture_root / "fused.npz", **fused, selected_indices=selected)
    object_root = root / "object"
    object_root.mkdir()
    points = fused["points_world_m"][selected]
    np.savez_compressed(
        object_root / "object.npz",
        points_world_m=points,
        points_color_m=transform(points, np.linalg.inv(arrays["T_world_color"])),
        T_world_color=arrays["T_world_color"],
        source_indices=selected,
        source_camera=fused["source_camera"][selected],
    )
    info = {
        **seed_info,
        "source_type": "multi_camera_fused",
        "extraction_policy": "D415_seed_supported_fused_component",
        "point_count": len(points),
        "object_sha256": file_hash(object_root / "object.npz"),
        "multi_camera": dict(
            config=config,
            seed_path=str(seed_root),
            seed_sha256=seed_info["object_sha256"],
            fused_path=str(capture_root / "fused.npz"),
            fused_sha256=file_hash(capture_root / "fused.npz"),
            cameras_path=str(capture_root / "cameras.json"),
            cameras_sha256=file_hash(capture_root / "cameras.json"),
            point_counts=counts,
            association=association,
            timings=report["timings"],
            color_coordinates="D415 color-frame coordinates only; no invented RGB for occluded surfaces",
        ),
    }
    write_json(object_root / "object.json", info)
    report.update(status="observed", object_root=str(object_root))
    return object_root


def main(argv: list[str] | None = None) -> int:
    """Run passive stream validation or a calibrated object observation; never command robots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "observe"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--camera-config", default="src/ada_grasp_ctrl/config/experiments/fixed_camera.yaml")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--plane")
    parser.add_argument("--roi", nargs=4, type=int)
    parser.add_argument("--cluster", type=int)
    parser.add_argument("--observe-seconds", type=float, default=0)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    root, created = resolve_from_root(args.output_root, "output"), False
    report = dict(schema_version=1, status="preflight", robot_commands_sent=False)
    try:
        if not np.isfinite(args.observe_seconds) or args.observe_seconds < 0:
            raise ValueError("--observe-seconds must be finite and nonnegative")
        root.mkdir(parents=True, exist_ok=False)
        created = True
        (root / "run_manifest.yaml").write_text(yaml.safe_dump(dict(arguments=vars(args))))
        observe(args, root, report)
        print(f"{report['status']}: {root}")
        return 0
    except KeyboardInterrupt:
        report.update(status="canceled")
        return 130
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr)
        return 2
    finally:
        if created:
            write_json(root / "run_report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
