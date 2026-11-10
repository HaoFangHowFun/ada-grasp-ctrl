"""Measured single-view point-cloud and open-surface mesh assets; no shape completion."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

from ada_grasp_ctrl.paths import resolve_external_root, resolve_from_root
from .fixed_camera import load_plane, load_snapshot, read_json, write_json
from .fixed_camera_geometry import file_hash, project_color, transform


def triangulate_depth(
    points: np.ndarray, info: dict, max_edge_m: float, max_depth_jump_m: float
) -> tuple[np.ndarray, dict]:
    """Connect only neighboring original depth pixels, rejecting missing data and jumps.

    Args:
        points: Object vertices in the source depth optical frame, meters.
        info: Live depth CameraInfo; this route requires rectified zero-distortion XYZ.
        max_edge_m: Maximum length of any triangle edge.
        max_depth_jump_m: Maximum camera-depth difference within each triangle.

    Returns:
        Camera-facing triangle indices into unchanged input vertices and diagnostics.

    Raises:
        ValueError: Unsupported intrinsics, non-grid points, duplicates or no valid faces.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError("Expected finite positive-depth Nx3 object points")
    if not np.isfinite([max_edge_m, max_depth_jump_m]).all() or min(max_edge_m, max_depth_jump_m) <= 0:
        raise ValueError("Mesh thresholds must be finite and positive")
    if info["distortion_model"] not in ("plumb_bob", "rational_polynomial") or not np.allclose(info["d"], 0):
        raise ValueError("Depth-grid meshing requires the rectified zero-distortion source cloud")
    k = np.asarray(info["k"], dtype=float).reshape(3, 3)
    width, height = int(info["width"]), int(info["height"])
    if width <= 0 or height <= 0 or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid depth CameraInfo")
    pixels = points[:, :2] / points[:, 2, None] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    rounded = np.rint(pixels).astype(int)
    projection_error = float(np.max(np.abs(pixels - rounded)))
    if projection_error > 0.05:
        raise ValueError("Cloud does not map to original depth pixel centers; cannot infer adjacency")
    if not ((rounded >= 0).all() and (rounded[:, 0] < width).all() and (rounded[:, 1] < height).all()):
        raise ValueError("Depth projection outside image")
    if len(np.unique(rounded, axis=0)) != len(points):
        raise ValueError("Duplicate depth pixel samples")
    grid = np.full((height, width), -1, dtype=int)
    grid[rounded[:, 1], rounded[:, 0]] = np.arange(len(points))
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    faces = np.concatenate([np.stack([a, b, c], axis=-1).reshape(-1, 3), np.stack([b, d, c], axis=-1).reshape(-1, 3)])
    faces = faces[(faces >= 0).all(axis=1)]
    candidate_count = len(faces)
    if not candidate_count:
        raise ValueError("No adjacent measured pixels form a surface")
    triangles = points[faces]
    lengths = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    keep = (
        (lengths.max(axis=1) <= max_edge_m)
        & (np.ptp(triangles[:, :, 2], axis=1) <= max_depth_jump_m)
        & (np.linalg.norm(normals, axis=1) > 1e-12)
    )
    faces, normals = faces[keep], normals[keep]
    if not len(faces):
        raise ValueError("All triangles rejected by metric edge/depth thresholds")
    # Outward for the observed surface means towards the observing optical center.
    backwards = np.einsum("ij,ij->i", normals, points[faces].mean(axis=1)) > 0
    faces[backwards] = faces[backwards][:, [0, 2, 1]]
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return faces, dict(
        candidate_triangles=candidate_count,
        rejected_triangles=candidate_count - len(faces),
        triangles=len(faces),
        vertices_used=len(np.unique(faces)),
        open_boundary_edges=int(np.count_nonzero(counts == 1)),
        nonmanifold_edges=int(np.count_nonzero(counts > 2)),
        depth_pixel_max_error=projection_error,
        max_edge_m=max_edge_m,
        max_depth_jump_m=max_depth_jump_m,
        watertight=False,
        method="original depth-grid adjacency; no smoothing, hole filling or hidden-surface completion",
    )


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray, faces: np.ndarray | None = None) -> None:
    """Write portable ASCII PLY with measured RGB and optional triangle topology."""
    with path.open("w") as stream:
        stream.write(f"ply\nformat ascii 1.0\ncomment units meters\nelement vertex {len(points)}\n")
        stream.write("property double x\nproperty double y\nproperty double z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if faces is not None:
            stream.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\n")
        stream.write("end_header\n")
        for point, color in zip(points, colors):
            stream.write(" ".join([*(f"{v:.12g}" for v in point), *(str(int(v)) for v in color)]) + "\n")
        if faces is not None:
            for face in faces:
                stream.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def build_asset(object_root: Path, output: Path, name: str, max_edge_m: float, max_depth_jump_m: float) -> dict:
    """Export metric local/world points, an open colored surface, and traceable source records."""
    info = read_json(object_root / "object.json")
    if info.get("schema_version") != 1 or info.get("unit") != "m" or info.get("frame_id") != "world":
        raise ValueError("Unsupported object schema/frame/units")
    if file_hash(object_root / "object.npz") != info["object_sha256"]:
        raise ValueError("Object artifact hash mismatch")
    snapshot_path = resolve_from_root(info["source_snapshot"])
    snapshot, metadata = load_snapshot(snapshot_path)
    if metadata["snapshot_sha256"] != info["snapshot_sha256"]:
        raise ValueError("Object source snapshot changed")
    if metadata["source_frame"] != metadata["depth_info"]["header"]["frame_id"]:
        raise ValueError("Surface meshing requires a cloud in the original depth optical frame")
    plane_path = resolve_from_root(info["plane_path"])
    if file_hash(plane_path) != info["plane_sha256"]:
        raise ValueError("Object table-plane hash mismatch")
    plane = load_plane(plane_path)
    with np.load(object_root / "object.npz", allow_pickle=False) as archive:
        indices = archive["source_indices"]
        points = archive["points_world_m"]
        color_points = archive["points_color_m"]
    if not np.array_equal(points, snapshot["points_world_m"][indices]):
        raise ValueError("Object selected-point indices disagree with source snapshot")
    source_points = snapshot["points_source_m"][snapshot["valid_source_indices"][indices]]
    faces, quality = triangulate_depth(source_points, metadata["depth_info"], max_edge_m, max_depth_jump_m)
    pixels = np.rint(project_color(color_points, metadata["color_info"])).astype(int)
    h, w = snapshot["rgb"].shape[:2]
    if not ((pixels >= 0).all() and (pixels[:, 0] < w).all() and (pixels[:, 1] < h).all()):
        raise ValueError("Object RGB projection outside image")
    colors = snapshot["rgb"][pixels[:, 1], pixels[:, 0]]
    center = points.mean(axis=0)
    local = points - center
    world_asset = np.eye(4)
    world_asset[:3, 3] = center
    roundtrip = float(np.max(np.abs(transform(local, world_asset) - points)))
    if roundtrip > 1e-10:
        raise ValueError("Asset local/world conversion lost precision")
    np.save(output / "points.npy", local)
    np.save(output / "points_world.npy", points)
    np.savez_compressed(
        output / "surface.npz",
        vertices=local,
        faces=faces,
        colors_rgb=colors,
        T_world_asset=world_asset,
        T_world_color=snapshot["T_world_color"],
    )
    write_ply(output / "points.ply", local, colors)
    write_ply(output / "points_world.ply", points, colors)
    write_ply(output / "surface.ply", local, colors, faces)
    write_ply(output / "surface_world.ply", points, colors, faces)
    with (output / "surface.obj").open("w") as stream:
        stream.write("# meters; object-local origin is observed centroid; open measured surface\n")
        for point, color in zip(local, colors):
            stream.write("v " + " ".join(f"{v:.12g}" for v in np.r_[point, color / 255.0]) + "\n")
        for face in faces + 1:
            stream.write(f"f {face[0]} {face[1]} {face[2]}\n")
    # Keep the existing object contract so --object <asset directory> works for inference.
    for filename in ("object.json", "object.npz", "object_overlay.png"):
        shutil.copyfile(object_root / filename, output / filename)
    shutil.copyfile(snapshot_path / "rgb.png", output / "rgb.png")
    shutil.copyfile(snapshot_path / "snapshot.json", output / "capture_metadata.json")
    shutil.copyfile(plane_path, output / "table_plane.json")
    report = dict(
        schema_version=1,
        name=name,
        unit="m",
        frame_id="object_local",
        local_frame_definition="centroid of observed points; axes parallel to world; not center of mass",
        T_world_asset=world_asset.tolist(),
        point_count=len(points),
        observed_bounds_local_m=[local.min(axis=0).tolist(), local.max(axis=0).tolist()],
        observed_extent_xyz_m=np.ptp(local, axis=0).tolist(),
        table_normal_height_range_m=(points @ np.asarray(plane["normal_xyz"]) + plane["d"]).tolist(),
        surface=quality,
        local_world_roundtrip_max_m=roundtrip,
        capture_stamp_s=info["capture_stamp_s"],
        table_status=info["table_status"],
        calibration=info["extrinsics"],
        source_object=str(object_root),
        source_object_sha256=file_hash(object_root / "object.json"),
        source_snapshot=str(snapshot_path),
        source_snapshot_sha256=info["snapshot_sha256"],
        purpose="visible-surface inspection and point-cloud grasp inference; no collision/solid asset",
        limitations=[
            "single view: back, underside and occluded regions unobserved",
            f"table-height filter removes points <= {info['config']['object_min_height_m']} m",
            "world placement inherits unaccepted calibration/table status",
        ],
        source="user-requested capture of connected D415; no inferred ownership or redistribution license",
    )
    heights = report["table_normal_height_range_m"]
    report["table_normal_height_range_m"] = [min(heights), max(heights)]
    report["files_sha256"] = {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "asset.json", report)
    return report


def preview_asset(asset_root: Path, output: Path, port: int, duration: float, export_only: bool) -> None:
    """Preview the saved asset with colored mesh, raw points, wireframe and capture overlay."""
    import trimesh

    info = read_json(asset_root / "asset.json")
    if info["schema_version"] != 1 or info["unit"] != "m" or info["frame_id"] != "object_local":
        raise ValueError("Unsupported asset schema/frame/units")
    for name, expected in info["files_sha256"].items():
        if Path(name).name != name or file_hash(asset_root / name) != expected:
            raise ValueError(f"Asset hash mismatch: {name}")
    with np.load(asset_root / "surface.npz", allow_pickle=False) as archive:
        vertices, faces, colors = archive["vertices"], archive["faces"], archive["colors_rgb"]
        world_asset, world_color = archive["T_world_asset"], archive["T_world_color"]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    # Double-sided rendering makes the intentionally open surface easy to inspect from behind.
    mesh.visual = mesh.visual.to_texture()
    mesh.visual.material.doubleSided = True
    mesh.export(output / "surface.glb")
    validation = dict(
        status="preview_ready",
        asset=str(asset_root),
        asset_sha256=file_hash(asset_root / "asset.json"),
        points=len(vertices),
        triangles=len(faces),
        is_watertight=bool(mesh.is_watertight),
        table_status=info["table_status"],
        coordinate_frame="object_local",
        port=port,
    )
    write_json(output / "run_report.json", validation)
    if export_only:
        return
    import viser
    from PIL import Image
    from .fixed_camera_preview import plane_mesh

    server = viser.ViserServer(host="127.0.0.1", port=port, label=info["name"] + " | measured surface")
    view = world_color[:3, 3] - world_asset[:3, 3]
    server.initial_camera.position = tuple(view / np.linalg.norm(view) * max(0.25, np.ptp(vertices, axis=0).max() * 3))
    server.initial_camera.look_at = (0, 0, 0)
    server.initial_camera.up = (0, 0, 1)
    cloud = server.scene.add_point_cloud("/object/points", vertices, colors=colors, point_size=0.0015, visible=False)
    surface = server.scene.add_mesh_trimesh("/object/surface", mesh)
    wire = server.scene.add_mesh_simple(
        "/object/wire", vertices, faces, color=(30, 30, 30), wireframe=True, visible=False
    )
    axes = server.scene.add_frame("/object/origin", axes_length=0.04, axes_radius=0.0006)
    plane = load_plane(asset_root / "table_plane.json")
    table_v, table_f = plane_mesh(plane)
    table = server.scene.add_mesh_simple(
        "/table_observed_patch",
        table_v - world_asset[:3, 3],
        table_f,
        color=(150, 150, 140),
        opacity=0.25,
        side="double",
        visible=False,
    )
    extent = np.asarray(info["observed_extent_xyz_m"]) * 1000
    server.gui.add_markdown(
        f"**{info['name']}**\n\n{len(vertices):,} measured points · {len(faces):,} triangles\n\n"
        f"Observed XYZ span: {extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm\n\n"
        f"Local origin: observed centroid; world-aligned axes. Units: meters.\n\n"
        f"Table: **{info['table_status']}**. Open surface; no hidden-side completion."
    )
    mode = server.gui.add_dropdown(
        "Display", options=["Surface mesh", "Point cloud", "Mesh + points"], initial_value="Surface mesh"
    )
    show_wire = server.gui.add_checkbox("Wireframe", initial_value=False)
    show_axes = server.gui.add_checkbox("Local axes", initial_value=True)
    show_table = server.gui.add_checkbox("Observed table patch", initial_value=False)
    size = server.gui.add_slider("Point size (mm)", min=0.5, max=4.0, step=0.5, initial_value=1.5)
    server.gui.add_image(np.asarray(Image.open(asset_root / "object_overlay.png")), label="Captured object selection")
    server.gui.add_image(np.asarray(Image.open(asset_root / "rgb.png")), label="Original RGB")

    def update(_: object = None) -> None:
        surface.visible = mode.value != "Point cloud"
        cloud.visible = mode.value != "Surface mesh"
        wire.visible = show_wire.value
        axes.visible = show_axes.value
        table.visible = show_table.value
        cloud.point_size = size.value / 1000

    for control in (mode, show_wire, show_axes, show_table, size):
        control.on_update(update)
    update()
    start = time.monotonic()
    try:
        while duration <= 0 or time.monotonic() - start < duration:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> int:
    """Build with system ROS Python, or preview with the existing Ada Python environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "preview"])
    parser.add_argument("--object")
    parser.add_argument("--asset")
    parser.add_argument("--name", default="captured_object")
    parser.add_argument("--max-edge-m", type=float, default=0.008)
    parser.add_argument("--max-depth-jump-m", type=float, default=0.008)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args(argv)
    output = None
    try:
        destination = resolve_external_root("output", args.output_root)
        destination.mkdir(parents=True, exist_ok=False)
        output = destination
        (output / "run_manifest.yaml").write_text(yaml.safe_dump(vars(args)))
        if args.command == "build":
            if not args.object:
                raise ValueError("--object is required")
            report = build_asset(
                resolve_from_root(args.object), output, args.name, args.max_edge_m, args.max_depth_jump_m
            )
            write_json(
                output / "run_report.json",
                dict(
                    status="completed",
                    asset="asset.json",
                    points=report["point_count"],
                    triangles=report["surface"]["triangles"],
                ),
            )
            print({"output_root": str(output), "points": report["point_count"], "surface": report["surface"]})
        else:
            if not args.asset:
                raise ValueError("--asset is required")
            preview_asset(resolve_from_root(args.asset), output, args.port, args.duration, args.export_only)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        if output is not None:
            write_json(output / "run_report.json", dict(status="preflight_failed", error=str(exc)))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
