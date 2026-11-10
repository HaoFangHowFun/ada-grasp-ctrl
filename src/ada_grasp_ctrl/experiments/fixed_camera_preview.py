"""Static world preview of measured geometry and named right Sharpa grasp stages."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.paths import resolve_external_root, resolve_from_root
from .fixed_camera import load_plane, load_snapshot, read_json, write_json
from .fixed_camera_geometry import file_hash, rigid, signed_height, transform
from .fixed_camera_grasp import STAGES

COLORS = ((60, 170, 255), (70, 220, 110), (255, 130, 60))


def origin(element: ET.Element | None) -> np.ndarray:
    """Interpret standard URDF meters and extrinsic XYZ roll/pitch/yaw."""
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return rigid(result)


class HandVisual:
    """Explicit URDF tree FK, retaining mesh origins and the actual flange root."""

    def __init__(self, path: Path, expected_joints: list[str]) -> None:
        import trimesh

        self.path = path
        self.root = ET.parse(path).getroot()
        self.joints = list(self.root.findall("joint"))
        movable = {j.get("name") for j in self.joints if j.get("type") != "fixed"}
        if movable != set(expected_joints):
            raise ValueError("URDF movable joints differ from checkpoint metadata")
        child_links = {j.find("child").get("link") for j in self.joints}
        roots = {link.get("name") for link in self.root.findall("link")} - child_links
        if roots != {"right_hand_flange"}:
            raise ValueError(f"Expected a flange-rooted right-hand URDF, got {roots}")
        self.meshes = []
        self.sources = {str(path): file_hash(path)}
        for link in self.root.findall("link"):
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None:
                    raise ValueError("This Sharpa preview requires mesh visual geometry")
                mesh_path = (path.parent / mesh.get("filename")).resolve(strict=True)
                self.sources[str(mesh_path)] = file_hash(mesh_path)
                loaded = trimesh.load(mesh_path, force="mesh", process=False)
                scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                vertices = transform(np.asarray(loaded.vertices) * scale, origin(visual.find("origin")))
                self.meshes.append((link.get("name"), vertices, np.asarray(loaded.faces)))

    def mesh(self, stage: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        """Place named joints and flange exactly; annotate limits and never clip angles."""
        q = stage["joint_positions_rad"]
        poses = {"right_hand_flange": rigid(stage["T_world_hand"])}
        pending = list(self.joints)
        violations = []
        while pending:
            progressed = False
            for joint in list(pending):
                parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
                if parent not in poses:
                    continue
                local = origin(joint.find("origin"))
                kind = joint.get("type")
                if kind != "fixed":
                    if kind not in ("revolute", "continuous"):
                        raise ValueError(f"Unsupported joint type {kind}")
                    value = float(q[joint.get("name")])
                    if not np.isfinite(value):
                        raise ValueError("Nonfinite hand angle")
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    if not np.isclose(np.linalg.norm(axis), 1, atol=1e-6):
                        raise ValueError("URDF joint axis must be unit")
                    motion = np.eye(4)
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                    local = local @ motion
                    limit = joint.find("limit")
                    if limit is not None and not float(limit.get("lower", "-inf")) <= value <= float(
                        limit.get("upper", "inf")
                    ):
                        violations.append(joint.get("name"))
                poses[child] = poses[parent] @ local
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError("URDF contains a disconnected or cyclic joint tree")
        vertices, faces, offset = [], [], 0
        for link, points, triangles in self.meshes:
            vertices.append(transform(points, poses[link]))
            faces.append(triangles + offset)
            offset += len(points)
        return np.concatenate(vertices), np.concatenate(faces), {"joint_limit_violations": violations}


def plane_mesh(plane: dict) -> tuple[np.ndarray, np.ndarray]:
    """Draw only the observed plane patch, with no inferred table box or thickness."""
    center = np.asarray(plane["centroid_world_m"])
    axes = np.asarray(plane["observed_tangent_axes"])
    half = np.asarray(plane["observed_span_m"]) / 2
    vertices = np.asarray(
        [center + x * half[0] * axes[0] + y * half[1] * axes[1] for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    )
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def preview(args: argparse.Namespace, output: Path) -> dict:
    """Export full visual meshes and optionally serve an interactive candidate selector."""
    import trimesh

    candidates_path = resolve_from_root(args.candidates)
    data = read_json(candidates_path)
    if data["frame_id"] != "world" or data["unit"] != "m" or data["quaternion_order"] != "wxyz":
        raise ValueError("Candidate coordinate convention mismatch")
    object_root = resolve_from_root(data["object_root"])
    object_info = read_json(object_root / "object.json")
    if file_hash(object_root / "object.npz") != data["object_sha256"]:
        raise ValueError("Object hash mismatch")
    plane_path = resolve_from_root(object_info["plane_path"])
    if file_hash(plane_path) != object_info["plane_sha256"]:
        raise ValueError("Table changed since object extraction")
    plane = load_plane(plane_path)
    snapshot, _ = load_snapshot(resolve_from_root(object_info["source_snapshot"]))
    with np.load(object_root / "object.npz", allow_pickle=False) as archive:
        object_points = archive["points_world_m"]
    model = HandVisual(resolve_from_root(args.hand_urdf), data["metadata"]["joint_names"])
    table_vertices, table_faces = plane_mesh(plane)
    rows = []
    meshes = {}
    for candidate in data["candidates"]:
        diagnostics = {}
        parts = []
        for name, color in zip(STAGES, COLORS):
            vertices, faces, diagnostic = model.mesh(candidate["stages"][name])
            diagnostic["visual_mesh_min_table_distance_m"] = float(signed_height(vertices, plane).min())
            diagnostic["table_intersection"] = diagnostic["visual_mesh_min_table_distance_m"] < 0
            diagnostic["scope"] = "static visual mesh vertices; not path or physical collision certification"
            diagnostics[name] = diagnostic
            parts.append((name, vertices, faces, color))
        meshes[candidate["candidate_id"]] = parts
        rows.append(dict(candidate_id=candidate["candidate_id"], stages=diagnostics))
    selected = args.candidate or data["candidates"][0]["candidate_id"]
    if selected not in meshes:
        raise ValueError("Unknown candidate ID")
    scene = trimesh.Scene()
    for name, vertices, faces, color in meshes[selected]:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = np.tile([*color, 210], (len(vertices), 1))
        scene.add_geometry(mesh, node_name=name)
    patch = trimesh.Trimesh(vertices=table_vertices, faces=table_faces, process=False)
    patch.visual.vertex_colors = np.tile([130, 150, 140, 100], (4, 1))
    scene.add_geometry(patch, node_name="observed_table_patch")
    scene.add_geometry(
        trimesh.points.PointCloud(object_points, colors=[240, 220, 80, 255]), node_name="measured_object"
    )
    scene.export(output / "preview.glb")
    report = dict(
        schema_version=1,
        status="preview_only",
        selected_candidate=selected,
        candidates_path=str(candidates_path),
        candidates_sha256=file_hash(candidates_path),
        table_status=plane["status"],
        source_assets=model.sources,
        candidates=rows,
        reachability=data["reachability"],
        calibration=data["calibration"],
    )
    write_json(output / "preview_report.json", report)
    if args.export_only:
        return report
    import viser

    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="Measured table + right Sharpa | preview only")
    center = object_points.mean(axis=0)
    server.initial_camera.look_at = tuple(center)
    server.initial_camera.position = tuple(center + [0.6, -0.6, 0.5])
    server.initial_camera.up = (0, 0, 1)
    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.002)
    server.scene.add_mesh_simple("/table", table_vertices, table_faces, color=(130, 150, 140), opacity=0.3)
    server.scene.add_point_cloud("/object", object_points, colors=(240, 220, 80), point_size=0.002)
    background = snapshot["points_world_m"][::10]
    server.scene.add_point_cloud("/scene", background, colors=(140, 140, 140), point_size=0.001)
    server.gui.add_markdown(f"Table: **{plane['status']}**. Units: m/rad; world +Z.\n\n{data['reachability']}")
    selector = server.gui.add_dropdown("Candidate", options=list(meshes), initial_value=selected)
    stages = server.gui.add_dropdown("Stages", options=["all", *STAGES], initial_value="all")
    summary = server.gui.add_markdown("")
    handles = []

    def update(_: object = None) -> None:
        for handle in handles:
            handle.remove()
        handles.clear()
        for name, vertices, faces, color in meshes[selector.value]:
            if stages.value not in ("all", name):
                continue
            handles.append(server.scene.add_mesh_simple("/hand/" + name, vertices, faces, color=color, opacity=0.65))
        row = next(row for row in rows if row["candidate_id"] == selector.value)
        summary.content = "\n\n".join(
            f"{name}: table distance {d['visual_mesh_min_table_distance_m']:.4f} m; "
            f"joint limits: {d['joint_limit_violations']}"
            for name, d in row["stages"].items()
        )

    selector.on_update(update)
    stages.on_update(update)
    update()
    start = time.monotonic()
    try:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return report


def main(argv: list[str] | None = None) -> int:
    """Open saved candidates without ROS control, robot drivers or physics steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--hand-urdf", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args(argv)
    output = None
    try:
        destination = resolve_external_root("output", args.output_root)
        destination.mkdir(parents=True, exist_ok=False)
        output = destination
        (output / "run_manifest.yaml").write_text(yaml.safe_dump(vars(args)))
        report = preview(args, output)
        write_json(
            output / "run_report.json", {"status": "completed", "selected_candidate": report["selected_candidate"]}
        )
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        if output is not None:
            write_json(output / "run_report.json", {"status": "preflight_failed", "error": str(exc)})
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
