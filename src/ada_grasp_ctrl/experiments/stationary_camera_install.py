"""Import converged static-camera app results with explicit pending-validation policy."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ada_grasp_ctrl.paths import resolve_from_root
from .fixed_camera_geometry import file_hash, rigid
from .multi_camera_geometry import SERIALS


def entry_digest(entry: dict) -> str:
    """Bind an operator-selected pending entry to its exact geometry and provenance."""
    return hashlib.sha256(json.dumps(entry, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pose_matrix(pose: dict) -> np.ndarray:
    """Convert the official app's metre / XYZW pose without guessing RPY conventions."""
    values = np.asarray([pose[key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")], dtype=float)
    if not np.isfinite(values).all() or not np.isclose(np.linalg.norm(values[3:]), 1.0, atol=1e-6):
        raise ValueError("Invalid solver pose or quaternion")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
    matrix[:3, 3] = values[:3]
    return rigid(matrix)


def pending_entry_allowed(role: str, entry: dict, config: dict) -> bool:
    """Allow only explicitly pinned converged results; never label them validated.

    Returns:
        False when no matching operator selection exists.

    Raises:
        ValueError: If a selected entry's solver or provenance has changed.
    """
    expected = config.get("accepted_pending_calibrations", {}).get(role)
    if expected is None or expected != entry_digest(entry):
        return False
    quality = entry.get("quality", {})
    if (
        entry.get("status") != "pending_independent_validation"
        or quality.get("independent_validation_passed") is not False
        or quality.get("solver_converged") is not True
        or quality.get("sample_count", 0) < 3
    ):
        raise ValueError(f"{role}: selected pending entry is not a converged measured result")
    for name, source in entry["provenance"].items():
        if file_hash(resolve_from_root(source["path"])) != source["sha256"]:
            raise ValueError(f"{role}: changed calibration source: {name}")
    result = yaml.safe_load(resolve_from_root(entry["provenance"]["solver_result"]["path"]).read_text())
    if result.get("converged") is not True or not np.allclose(
        pose_matrix(result["camera_mount_to_camera"]), entry["T_leftbase_optical"], rtol=0, atol=1e-12
    ):
        raise ValueError(f"{role}: pending transform differs from converged solver output")
    return True


def solver_entries(solver_directory: Path, camera_config: Path, previous_calibration: Path) -> dict:
    """Normalize both app outputs after checking capture identity and source files.

    Args:
        solver_directory: Output of prepare_static_camera_calibration with both result YAMLs.
        camera_config: Explicit role-to-serial mapping used during this capture.
        previous_calibration: Preserved master file before additive installation.

    Returns:
        Two pending stationary-camera entries with source file hashes.

    Raises:
        ValueError: For mismatched frames, changed input files, or failed solver results.
    """
    solver_directory = solver_directory.resolve()
    manifest_path = solver_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    session_path = Path(manifest["session"]) / "session.json"
    session = json.loads(session_path.read_text())
    target = Path(session["settings"]["target_config"])
    if session["settings"]["mounting_mode"] != "static" or file_hash(target) != session["board_sha256"]:
        raise ValueError("Expected unchanged static-camera board/session settings")
    serials = yaml.safe_load(camera_config.read_text())
    entries = {}
    for side in ("left", "right"):
        role = f"wrist_{side}"
        if serials[role]["serial_no"] != SERIALS[role]:
            raise ValueError(f"{role}: capture serial differs from deployment")
        result_path = solver_directory / f"{side}_result.yaml"
        result = yaml.safe_load(result_path.read_text())
        if result.get("converged") is not True:
            raise ValueError(f"{role}: solver did not converge")
        context_path = Path(manifest["export"]) / side / "calibration_context.json"
        context = json.loads(context_path.read_text())
        if (
            context["static_camera"] is not True
            or context["pose_parent_frame"] != "left_base_link"
            or context["camera_frame"] != f"{role}_color_optical_frame"
            or context["pose_child_frame"] != session["settings"]["target_mount_frame"]
        ):
            raise ValueError(f"{role}: solver context does not identify the measured static geometry")
        report = manifest["cameras"][side]
        config_path = Path(report["config"])
        app_config = yaml.safe_load(config_path.read_text())
        if (
            app_config["static_camera"] is not True
            or app_config["intrinsics"] != context["intrinsics"]
            or app_config["target_finder"] != yaml.safe_load(target.read_text())["target_finder"]
        ):
            raise ValueError(f"{role}: app configuration differs from capture")
        if report["observations"] != len(report["input_files"]) or report["observations"] < 3:
            raise ValueError(f"{role}: incomplete fit inputs")
        for observation in report["input_files"]:
            for source in observation.values():
                if file_hash(resolve_from_root(source["path"])) != source["sha256"]:
                    raise ValueError(f"{role}: changed solver input file")
        sources = {
            "solver_result": result_path,
            "board_config": target,
            "session": session_path,
            "app_config": config_path,
            "app_manifest": manifest_path,
            "camera_context": context_path,
            "camera_serial_mapping": camera_config.resolve(),
            "previous_calibration": previous_calibration.resolve(),
        }
        entries[role] = dict(
            serial=SERIALS[role],
            calibrated_optical_frame=context["camera_frame"],
            reference_frame="left_base_link",
            transform_direction="optical_to_leftbase",
            unit="m",
            mounting="stationary",
            status="pending_independent_validation",
            T_leftbase_optical=pose_matrix(result["camera_mount_to_camera"]).tolist(),
            target_mount_frame=context["pose_child_frame"],
            T_target_mount_board=pose_matrix(result["target_mount_to_target"]).tolist(),
            quality=dict(
                solver_converged=True,
                independent_validation_passed=False,
                sample_count=report["observations"],
                validation_count=0,
                final_cost_per_obs=float(result["final_cost_per_obs"]),
            ),
            provenance={key: dict(path=str(path), sha256=file_hash(path)) for key, path in sources.items()},
        )
    return entries


def installed_master(master: dict, entries: dict) -> dict:
    """Add stationary cameras while preserving all existing camera and robot geometry."""
    result = deepcopy(master)
    result["authoritative"].setdefault("stationary_cameras", {}).update(entries)
    return result
