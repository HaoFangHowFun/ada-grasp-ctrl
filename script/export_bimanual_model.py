"""Export the installed canonical description without starting ROS nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def absolute(value: str) -> Path:
    """Require explicit absolute paths across the two Python environments."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Expected absolute path: {path}")
    return path.resolve()


def git_info(path: Path) -> dict:
    """Record available source revision and dirty state without changing Git."""

    def run(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(path.parent), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "root": run("rev-parse", "--show-toplevel"),
        "commit": run("rev-parse", "HEAD"),
        "status": run("status", "--short"),
        "remote": run("remote", "get-url", "origin"),
    }


def convert_meshes(directory: Path) -> None:
    """Convert copied DAE visuals to STL, retaining scene-node transforms and units."""
    import trimesh

    root = ET.parse(directory / "resolved.urdf").getroot()
    converted = {}
    for mesh in root.findall(".//mesh"):
        relative = mesh.attrib["filename"]
        if Path(relative).suffix.lower() != ".dae":
            continue
        if relative not in converted:
            source = directory / relative
            scene = trimesh.load(source, force="scene", process=False)
            geometry = scene.to_geometry()
            if not len(geometry.vertices) or not len(geometry.faces):
                raise ValueError(f"Empty DAE geometry: {source}")
            target = source.with_suffix(".stl")
            geometry.export(target)
            converted[relative] = str(target.relative_to(directory))
        mesh.set("filename", converted[relative])
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(
        extension, "compiler", strippath="false", balanceinertia="true", discardvisual="false", fusestatic="false"
    )
    ET.indent(root)
    ET.ElementTree(root).write(directory / "model.urdf", encoding="unicode")
    (directory / "mesh_conversion.json").write_text(
        json.dumps(
            {
                "trimesh_version": trimesh.__version__,
                "dae_to_stl": converted,
                "limitations": "DAE geometry and node transforms retained; textures/materials are not retained in STL. "
                "MuJoCo uses convex hulls for mesh collision; no decomposition or exclusions added.",
            },
            indent=2,
        )
        + "\n"
    )


def export(output: Path, mesh_python: Path) -> None:
    """Create a fresh portable snapshot from the installed ROS description.

    Args:
        output: Nonexistent absolute output directory.
        mesh_python: Existing numerical Python with trimesh and pycollada installed.

    Raises:
        FileExistsError: If output already exists.
        RuntimeError: If source resolution or mesh conversion fails.
    """
    from ament_index_python.packages import get_package_share_directory
    import bimanual_manipulation.composite_description as canonical
    from dual_sharpa_wave.joint_names import joint_names
    import xacro

    output.mkdir(parents=True, exist_ok=False)
    shares = {
        name: Path(get_package_share_directory(name))
        for name in ("bimanual_manipulation", "dual_crx_control", "dual_sharpa_wave", "fanuc_crx_description")
    }
    xml = canonical.build_description(sides="both")
    (output / "source.urdf").write_text(xml)
    root = ET.fromstring(xml)
    sources = [
        Path(canonical.__file__),
        shares["bimanual_manipulation"] / "config/hand_mounts.yaml",
        shares["dual_crx_control"] / "urdf/dual_crx.urdf.xacro",
        shares["dual_crx_control"] / "config/initial_pose.yaml",
        *[Path(p) for p in xacro.all_includes],
        *[shares["dual_sharpa_wave"] / f"urdf/{side}_sharpa_wave_with_flange.urdf" for side in ("left", "right")],
    ]
    provenance = []
    source_dir = output / "sources"
    source_dir.mkdir()
    repos = {}
    for index, path in enumerate(sorted(set(p.resolve(strict=True) for p in sources))):
        target = source_dir / f"{index:03d}_{path.name}"
        shutil.copy2(path, target)
        info = git_info(path)
        if info["root"]:
            repos[info["root"]] = info
        provenance.append(
            {
                "path": str(path),
                "copy": str(target.relative_to(output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    meshes = output / "meshes"
    shutil.copy2(shares["dual_crx_control"] / "config/initial_pose.yaml", output / "initial_pose.yaml")
    meshes.mkdir()
    mesh_sources = {}
    for mesh in root.findall(".//mesh"):
        uri = mesh.attrib["filename"]
        if not uri.startswith("package://"):
            raise ValueError(f"Expected canonical package mesh URI, got {uri}")
        package, relative = uri[len("package://") :].split("/", 1)
        path = (Path(get_package_share_directory(package)) / relative).resolve(strict=True)
        if uri not in mesh_sources:
            target = meshes / f"m{len(mesh_sources):03d}_{path.name}"
            shutil.copy2(path, target)
            mesh_sources[uri] = {
                "path": str(path),
                "copy": str(target.relative_to(output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        mesh.set("filename", mesh_sources[uri]["copy"])
    # Explicit names prevent import-time collisions between left/right mesh basenames.
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for index, geom in enumerate(link.findall(tag)):
                geom.set("name", f"{link.attrib['name']}_{tag}_{index}")
    ET.indent(root)
    ET.ElementTree(root).write(output / "resolved.urdf", encoding="unicode")
    licenses = output / "licenses"
    licenses.mkdir()
    # Asset licenses may live below the ROS package root (e.g. third_party/sharpa_models).
    license_directories = {Path(repository) for repository in repos}
    for record in [*provenance, *mesh_sources.values()]:
        for parent in Path(record["path"]).parents:
            license_directories.add(parent)
            if str(parent) in repos:
                break
    license_sources = []
    for directory in sorted(license_directories):
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")):
                target = licenses / f"{len(license_sources):03d}_{path.name}"
                shutil.copy2(path, target)
                license_sources.append(
                    {
                        "path": str(path),
                        "copy": str(target.relative_to(output)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [str(mesh_python), str(Path(__file__).resolve()), "--convert-meshes", str(output)], env=environment, check=True
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "canonical_builder": str(Path(canonical.__file__).resolve()),
        "package_shares": {k: str(v) for k, v in shares.items()},
        "sources": provenance,
        "repositories": list(repos.values()),
        "meshes": mesh_sources,
        "licenses": license_sources,
        "joint_names": [
            name for side in ("left", "right") for name in (*[f"{side}_J{i}" for i in range(1, 7)], *joint_names(side))
        ],
        "mount_calibrated": False,
        "mount_status": "Upstream mock preview placeholders; uncalibrated for physical motion",
        "urdf": "model.urdf",
        "initial_pose": "initial_pose.yaml",
        "units": {"length": "m", "joint_angle": "rad"},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "manifest.json")


def main() -> None:
    """Run ROS-time export or its isolated numerical mesh conversion subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=absolute)
    parser.add_argument("--mesh-python", type=absolute)
    parser.add_argument("--convert-meshes", type=absolute, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.convert_meshes:
        convert_meshes(args.convert_meshes)
    elif args.output_root and args.mesh_python:
        export(args.output_root, args.mesh_python)
    else:
        parser.error("--output-root and --mesh-python are required")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        sys.exit(2)
