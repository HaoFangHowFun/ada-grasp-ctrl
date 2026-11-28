"""Validate measured fixed-D405 extrinsics; reuse the existing ROS-Industrial solver."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.calibration import DEFAULT_CALIBRATION, load_calibration
from ada_grasp_ctrl.paths import resolve_from_root
from .fixed_camera_geometry import file_hash, rigid
from .multi_camera_geometry import ROLES, SERIALS


def load_cameras(path: Path, config: dict) -> dict:
    """Require validated metadata or an explicitly pinned pending solver result.

    Returns optical-to-leftbase transforms. The caller composes internal optical
    TF when calibration and depth use different optical frames. Missing D405s
    never fall back to identity, arm FK, or the D415 transform.
    """
    calibration = load_calibration(path)
    data = json.loads(path.read_text())
    source = data["authoritative"]
    if source["camera"]["serial"] != config["cameras"]["fixed"]["serial"]:
        raise ValueError("D415 calibration serial mismatch")
    cameras = dict(fixed=dict(matrix=calibration["camera"], calibrated_optical_frame="fixed_color_optical_frame"))
    for role in ROLES[1:]:
        entry = source.get("stationary_cameras", {}).get(role)
        if not entry:
            raise ValueError(f"{role}: missing measured fixed-D405 extrinsics; run multi_camera_calibration")
        if (entry.get("serial"), entry.get("reference_frame"), entry.get("transform_direction"), entry.get("unit")) != (
            config["cameras"][role]["serial"],
            "left_base_link",
            "optical_to_leftbase",
            "m",
        ):
            raise ValueError(f"{role}: calibration serial/reference/direction/units mismatch")
        if entry.get("calibrated_optical_frame") not in (role + "_depth_optical_frame", role + "_color_optical_frame"):
            raise ValueError(f"{role}: unsupported calibrated optical frame")
        quality = entry.get("quality", {})
        from .stationary_camera_install import pending_entry_allowed

        if pending_entry_allowed(role, entry, config):
            if entry.get("mounting") != "stationary":
                raise ValueError(f"{role}: calibration must describe a stationary camera")
            cameras[role] = dict(
                matrix=rigid(entry["T_leftbase_optical"], role),
                calibrated_optical_frame=entry["calibrated_optical_frame"],
                status=entry["status"],
                independent_validation_passed=False,
            )
            continue
        if entry.get("status") != "validated" or quality.get("independent_validation_passed") is not True:
            raise ValueError(f"{role}: independent calibration validation required")
        for key in (
            "sample_count",
            "validation_count",
            "translation_rms_m",
            "rotation_rms_deg",
            "reprojection_rms_px",
            "overlap_rms_m",
        ):
            if key not in quality or not np.isfinite(quality[key]) or quality[key] < 0:
                raise ValueError(f"{role}: invalid quality field {key}")
        if quality["sample_count"] < 3 or quality["validation_count"] < 3 or not entry.get("provenance"):
            raise ValueError(f"{role}: insufficient calibration evidence")
        for metric, limit in (
            ("translation_max_m", "translation_m"),
            ("rotation_max_deg", "rotation_deg"),
            ("reprojection_rms_px", "reprojection_px"),
            ("overlap_rms_m", "overlap_m"),
        ):
            value, threshold = quality.get(metric), quality.get("limits", {}).get(limit)
            if (
                value is None
                or threshold is None
                or not np.isfinite([value, threshold]).all()
                or not 0 <= value <= threshold
                or threshold <= 0
            ):
                raise ValueError(f"{role}: calibration quality exceeds recorded limits: {metric}")
        if entry.get("mounting") != "stationary":
            raise ValueError(f"{role}: calibration must describe a stationary camera")
        cameras[role] = dict(
            matrix=rigid(entry["T_leftbase_optical"], role),
            calibrated_optical_frame=entry["calibrated_optical_frame"],
            status="validated",
            independent_validation_passed=True,
        )
    return cameras


def validated_entry(candidate: dict, evidence: dict, master: dict, limits: dict) -> dict:
    """Validate an existing solver result on held-out measured board poses and overlap.

    Args:
        candidate: Export of the existing solver, with optical frame and fit IDs.
        evidence: Held-out board poses in leftbase and camera plus overlap residuals.
        master: Existing authoritative calibration for right-to-left base conversion.
        limits: Explicit metric, angular, and pixel error thresholds.

    Returns:
        Additive stationary-camera schema entry; no file is modified here.

    Raises:
        ValueError: For malformed, reused, degenerate, or failed observations.
    """
    role = candidate["role"]
    if role not in ROLES[1:] or candidate["serial"] != SERIALS[role]:
        raise ValueError("Expected one explicitly identified fixed D405")
    frame = candidate["calibrated_optical_frame"]
    if frame not in (role + "_color_optical_frame", role + "_depth_optical_frame") or candidate["unit"] != "m":
        raise ValueError("Explicit optical frame and metre units required")
    if candidate.get("transform_direction") != "optical_to_reference" or candidate.get("converged") is not True:
        raise ValueError("Require converged existing solver result with explicit direction")
    matrix = rigid(candidate["T_reference_optical"])
    if candidate["reference_frame"] == "right_base_link":
        base = master["authoritative"]["dual_crx"]["T_leftbase_rightbase"]
        left_right = np.eye(4)
        left_right[:3, :3] = Rotation.from_euler("xyz", base["rpy_rad"]).as_matrix()
        left_right[:3, 3] = base["xyz_m"]
        matrix = rigid(left_right @ matrix)
    elif candidate["reference_frame"] != "left_base_link":
        raise ValueError("Use explicit left_base_link/right_base_link, never an unverified world alias")
    fit_ids = candidate["fit_observation_ids"]
    observations = evidence["observations"]
    validation_ids = [row["id"] for row in observations]
    if len(fit_ids) < 3 or len(set(fit_ids)) != len(fit_ids) or len(observations) < 3:
        raise ValueError("Require at least three distinct fit and three independent validation poses")
    if len(set(validation_ids)) != len(validation_ids) or set(fit_ids) & set(validation_ids):
        raise ValueError("Validation observations must be unique and excluded from fitting")
    translations, angles, pixels, poses = [], [], [], []
    for row in observations:
        known = rigid(row["T_leftbase_board"])
        measured = matrix @ rigid(row["T_optical_board"])
        poses.append(known)
        translations.append(float(np.linalg.norm(known[:3, 3] - measured[:3, 3])))
        angles.append(float(np.rad2deg(Rotation.from_matrix(known[:3, :3].T @ measured[:3, :3]).magnitude())))
        pixels.append(float(row["reprojection_rms_px"]))
    overlap = np.asarray(evidence["overlap_distances_m"], dtype=float)
    if overlap.ndim != 1 or len(overlap) < 3:
        raise ValueError("Require measured geometric overlap distances in metres")
    values = np.r_[translations, angles, pixels, overlap]
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Residuals must be finite and nonnegative")
    span = float(np.max(np.linalg.norm(np.array([pose[:3, 3] for pose in poses]) - poses[0][:3, 3], axis=1)))
    rotation_span = max(float(Rotation.from_matrix(poses[0][:3, :3].T @ pose[:3, :3]).magnitude()) for pose in poses)
    if span < 0.02 or rotation_span < np.deg2rad(10):
        raise ValueError("Independent board observations need diverse positions and orientations")
    if (
        max(translations) > limits["translation_m"]
        or max(angles) > limits["rotation_deg"]
        or max(pixels) > limits["reprojection_px"]
        or float(np.sqrt(np.mean(overlap**2))) > limits["overlap_m"]
    ):
        raise ValueError(
            f"Independent calibration validation failed: translation={translations}, rotation={angles}, pixels={pixels}"
        )
    provenance = candidate["provenance"]
    for key in ("solver_result", "board_config", "mount_registration"):
        record = provenance[key]
        if file_hash(resolve_from_root(record["path"])) != record["sha256"]:
            raise ValueError(f"Changed or missing measured calibration provenance: {key}")
    quality = dict(
        independent_validation_passed=True,
        sample_count=len(fit_ids),
        validation_count=len(observations),
        translation_rms_m=float(np.sqrt(np.mean(np.square(translations)))),
        translation_max_m=max(translations),
        rotation_rms_deg=float(np.sqrt(np.mean(np.square(angles)))),
        rotation_max_deg=max(angles),
        reprojection_rms_px=float(np.sqrt(np.mean(np.square(pixels)))),
        overlap_rms_m=float(np.sqrt(np.mean(overlap**2))),
        limits=limits,
        validation_observation_ids=validation_ids,
        fit_observation_ids=fit_ids,
    )
    return dict(
        serial=candidate["serial"],
        calibrated_optical_frame=frame,
        reference_frame="left_base_link",
        transform_direction="optical_to_leftbase",
        unit="m",
        mounting="stationary",
        status="validated",
        T_leftbase_optical=matrix.tolist(),
        quality=quality,
        provenance=provenance,
    )


def main(argv: list[str] | None = None) -> int:
    """Print the existing collection command or validate/save user-acquired calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collection-command", "validate"))
    parser.add_argument("--role", choices=ROLES[1:], required=True)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--target-config")
    parser.add_argument("--target-mount-frame", default="left_ee_mount")
    parser.add_argument("--candidate")
    parser.add_argument("--evidence")
    parser.add_argument("--output", required=True, help="New collection directory or validated calibration JSON")
    parser.add_argument(
        "--save", action="store_true", help="Also replace authoritative JSON after validation; writes backup"
    )
    parser.add_argument("--max-translation-m", type=float, default=0.01)
    parser.add_argument("--max-rotation-deg", type=float, default=2.0)
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--max-overlap-m", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        output = resolve_from_root(args.output, "output")
        if args.command == "collection-command":
            if not args.target_config:
                raise ValueError("--target-config must identify the existing measured board YAML")
            board = resolve_from_root(args.target_config)
            if not board.is_file():
                raise ValueError("Measured board config does not exist; do not invent dimensions")
            print(
                shlex.join(
                    [
                        "ros2",
                        "launch",
                        "bimanual_manipulation",
                        "fixed_camera_calibration.launch.py",
                        f"image_topic:=/camera/{args.role}/color/image_raw",
                        f"camera_info_topic:=/camera/{args.role}/color/camera_info",
                        f"camera_frame:={args.role}_color_optical_frame",
                        "camera_mount_frame:=left_base_link",
                        f"target_mount_frame:={args.target_mount_frame}",
                        f"target_config:={board}",
                        f"save_path:={output}",
                    ]
                )
            )
            return 0
        if not args.candidate or not args.evidence:
            raise ValueError("--candidate and --evidence are required")
        path = resolve_from_root(args.calibration)
        load_calibration(path)
        original = path.read_bytes()
        master = json.loads(original)
        candidate_path, evidence_path = resolve_from_root(args.candidate), resolve_from_root(args.evidence)
        candidate, evidence = json.loads(candidate_path.read_text()), json.loads(evidence_path.read_text())
        if candidate["role"] != args.role:
            raise ValueError("Candidate role differs from --role")
        limits = dict(
            translation_m=args.max_translation_m,
            rotation_deg=args.max_rotation_deg,
            reprojection_px=args.max_reprojection_px,
            overlap_m=args.max_overlap_m,
        )
        if not all(np.isfinite(value) and value > 0 for value in limits.values()):
            raise ValueError("Validation limits must be positive finite values")
        entry = validated_entry(candidate, evidence, master, limits)
        entry["provenance"] = {
            **entry["provenance"],
            "candidate": dict(path=str(candidate_path), sha256=file_hash(candidate_path)),
            "validation": dict(path=str(evidence_path), sha256=file_hash(evidence_path)),
        }
        backup = output.with_suffix(output.suffix + ".previous")
        entry["provenance"]["previous_calibration"] = dict(path=str(backup), sha256=file_hash(path))
        result = deepcopy(master)
        result["authoritative"].setdefault("stationary_cameras", {})[args.role] = entry
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with backup.open("xb") as stream:
            stream.write(original)
        if args.save:
            if path.read_bytes() != original:
                raise ValueError("Authoritative calibration changed during validation")
            temporary = path.with_suffix(".multi_camera.tmp")
            with temporary.open("xb") as stream:
                stream.write(output.read_bytes())
            temporary.replace(path)
        print(json.dumps(dict(output=str(output), saved=args.save, quality=entry["quality"]), indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Calibration rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
