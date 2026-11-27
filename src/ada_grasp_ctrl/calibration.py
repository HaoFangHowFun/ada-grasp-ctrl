"""Read authoritative calibration for Ada snapshots and ROS hand-mount configuration."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from ada_grasp_ctrl.experiments.fixed_camera_geometry import file_hash, rigid
from ada_grasp_ctrl.paths import resolve_from_root

DEFAULT_CALIBRATION = "src/ada_grasp_ctrl/config/calibration/dual_system_calibration.json"


def compatible_camera_calibration_hashes(data: dict) -> list[str]:
    """Verify preserved predecessors that differ only by stationary-camera entries.

    Backups are hash-bound actual input files, not declarations that arbitrary
    old calibration is equivalent. Changes to any legacy field invalidate this
    compatibility path, including D415, bases, mounts, conventions and metadata.
    """

    def legacy(value: dict) -> dict:
        return {
            **value,
            "authoritative": {key: item for key, item in value["authoritative"].items() if key != "stationary_cameras"},
        }

    expected, pending, accepted, visited = legacy(data), [data], set(), set()
    while pending:
        current = pending.pop()
        for camera in current["authoritative"].get("stationary_cameras", {}).values():
            reference = camera.get("provenance", {}).get("previous_calibration")
            if not reference or reference.get("sha256") in visited:
                continue
            visited.add(reference.get("sha256"))
            try:
                source = resolve_from_root(reference["path"])
                if file_hash(source) != reference["sha256"]:
                    continue
                previous = json.loads(source.read_text())
                if legacy(previous) != expected:
                    continue
            except (OSError, KeyError, TypeError, ValueError):
                continue
            accepted.add(reference["sha256"])
            pending.append(previous)
    return sorted(accepted)


def load_calibration(path: str | Path = DEFAULT_CALIBRATION) -> dict:
    """Load only authoritative base, optical-camera and hand-mount geometry.

    Args:
        path: Master JSON path, resolved through the existing Ada data root.

    Returns:
        Base/mount XYZ and fixed-axis RPY, camera matrix and file provenance.

    Raises:
        ValueError: On missing or malformed transform data or incompatible world/units.
    """
    path = resolve_from_root(path)
    data = json.loads(path.read_text())
    try:
        convention = data["conventions"]
        if (convention["world_frame"], convention["length_unit"], convention["angle_unit"]) != ("leftbase", "m", "rad"):
            raise ValueError("Calibration must use leftbase world, metres and radians")
        source = data["authoritative"]

        def placement(value: dict) -> dict:
            xyz = np.asarray(value["xyz_m"], dtype=float)
            rpy = np.asarray(value["rpy_rad"], dtype=float)
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
                raise ValueError("Calibration XYZ/RPY must each contain three finite values")
            return dict(xyz_m=xyz.tolist(), rpy_rad=rpy.tolist())

        return dict(
            base=placement(source["dual_crx"]["T_leftbase_rightbase"]),
            camera=rigid(source["camera"]["T_leftbase_color_optical"]["matrix_4x4"], "T_leftbase_color_optical"),
            mounts={side: placement(source["hand_mounts"][side]) for side in ("left", "right")},
            provenance=dict(
                path=str(path),
                sha256=file_hash(path),
                calibration_id=data["calibration_id"],
                **(
                    {"compatible_sha256": compatible_camera_calibration_hashes(data)}
                    if source.get("stationary_cameras")
                    else {}
                ),
            ),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed authoritative calibration: {error}") from error


def calibrated_urdf(description: str, calibration: dict) -> str:
    """Replace known fixed origins absolutely; retain all joint, visual and collision data.

    Args:
        description: Source snapshot URDF text.
        calibration: Authoritative placements returned by load_calibration.

    Returns:
        URDF text with the three calibrated fixed origins.

    Raises:
        ValueError: If the source does not have the inspected Ada parent/child mapping.
    """
    root = ET.fromstring(description)

    def fixed(name: str, parent: str, child: str) -> ET.Element:
        joints = root.findall(f"joint[@name='{name}']")
        if (
            len(joints) != 1
            or joints[0].get("type") != "fixed"
            or joints[0].find("parent") is None
            or joints[0].find("parent").get("link") != parent
            or joints[0].find("child") is None
            or joints[0].find("child").get("link") != child
        ):
            raise ValueError(f"Unresolved calibration frame mapping: {name}, {parent} -> {child}")
        return joints[0]

    left = fixed("left_base_joint", "world", "left_base_link")
    origin = left.find("origin")
    # The installed camera matrix defines world as the left CRX base. Existing
    # snapshots already use identity here; a different world would need explicit conversion.
    if origin is not None and any(
        not np.array_equal(np.fromstring(origin.get(key, "0 0 0"), sep=" "), np.zeros(3)) for key in ("xyz", "rpy")
    ):
        raise ValueError("Calibration requires world == left_base_link in the Ada snapshot")
    placements = [(fixed("right_base_joint", "world", "right_base_link"), calibration["base"])]
    for side in ("left", "right"):
        placements.append(
            (fixed(f"{side}_arm_to_hand", f"{side}_ee_mount", f"{side}_hand_flange"), calibration["mounts"][side])
        )
    for joint, value in placements:
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        origin.set("xyz", " ".join(map(str, value["xyz_m"])))
        origin.set("rpy", " ".join(map(str, value["rpy_rad"])))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def export_ros_hand_mounts(output: str | Path, path: str | Path = DEFAULT_CALIBRATION) -> Path:
    """Export the same absolute mount transforms consumed by calibrated Ada snapshots.

    Args:
        output: New YAML file, resolved through the Ada output root.
        path: Master calibration JSON used by the pipeline.

    Returns:
        Absolute path for the bimanual launch's hand_mounts argument.

    Raises:
        ValueError: If the authoritative calibration is malformed.
        FileExistsError: If the output already exists.
    """
    calibration = load_calibration(path)
    output = resolve_from_root(output, "output")
    mounts = {side: dict(xyz=value["xyz_m"], rpy=value["rpy_rad"]) for side, value in calibration["mounts"].items()}
    mounts["calibration"] = calibration["provenance"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write("# Generated from authoritative calibration; physical validation is separate.\n")
        stream.write(yaml.safe_dump(mounts, sort_keys=False))
    return output


def calibrated_snapshot(source: str | Path, output: str | Path, path: str | Path = DEFAULT_CALIBRATION) -> Path:
    """Create a new portable Ada snapshot for model, preview and existing packet provenance.

    Copy the source assets and historical provenance unchanged. Only model.urdf
    receives calibrated fixed origins; the manifest records this derivation.
    Existing output directories are never reused or overwritten.

    Args:
        source: Portable source snapshot containing model.urdf and manifest.json.
        output: New snapshot directory, resolved through the Ada output root.
        path: Master calibration JSON path.

    Returns:
        Absolute path to the generated snapshot.

    Raises:
        ValueError: If calibration or the source frame mapping is invalid.
        FileExistsError: If the output directory already exists.
    """
    source, output = resolve_from_root(source), resolve_from_root(output, "output")
    calibration = load_calibration(path)
    model = source / "model.urdf"
    description = calibrated_urdf(model.read_text(), calibration)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest.update(
        calibration=calibration["provenance"],
        calibration_source_snapshot=dict(
            path=str(source), model_sha256=file_hash(model), manifest_sha256=file_hash(source / "manifest.json")
        ),
        mount_status="Authoritative JSON applied in Ada; physical validation is separate",
    )
    shutil.copytree(source, output)
    (output / "model.urdf").write_text(description)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    """Export ROS mount configuration without starting ROS or commanding hardware."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", required=True, help="New hand_mounts.yaml path")
    args = parser.parse_args(argv)
    print(export_ros_hand_mounts(args.output, args.calibration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
