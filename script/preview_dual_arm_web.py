"""Preview both arms in a browser using upstream initial angles and original DAE colors."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import threading
import xml.etree.ElementTree as ET

from mjviser import ViserMujocoScene
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser
import yaml

from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.robot_environment import DualArmModel, DualArmSimulation


def load_initial_configuration(robot: DualArmModel, path: Path) -> np.ndarray:
    """Read upstream J1–J6 URDF degrees and the bimanual default of zero hand angles.

    Args:
        robot: Exported canonical model.
        path: Copy of dual_crx_control/config/initial_pose.yaml; no ROS lookup is needed.

    Returns:
        Finite, bounded 56-vector in radians.

    Raises:
        ValueError: If either side lacks six finite arm angles or violates source limits.
    """
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict) or not {"left", "right"} <= values.keys():
        raise ValueError("Initial pose requires left and right J1–J6 degree arrays")
    q = np.zeros(56)
    for side, offset in (("left", 0), ("right", 28)):
        angles = np.asarray(values[side], dtype=float)
        if angles.shape != (6,) or not np.isfinite(angles).all():
            raise ValueError(f"Expected six finite URDF degree angles for {side}")
        q[offset : offset + 6] = np.deg2rad(angles)
    return robot.configuration(q)


def add_source_visuals(server: viser.ViserServer, sim: DualArmSimulation) -> tuple[dict, dict]:
    """Display retained DAE material parts at the original URDF visual transforms.

    Returns:
        Body-frame handles and a count/color diagnostic. Only colorless visual duplicates
        are hidden; collision geometry and physical parameters are preserved.
    """
    handles, colors = {}, set()
    part_count = 0
    description = ET.parse(sim.robot.directory / "resolved.urdf").getroot()
    for link in description.findall("link"):
        name = link.attrib["name"]
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.attrib["filename"].lower().endswith(".dae"):
                continue
            path = (sim.robot.directory / mesh.attrib["filename"]).resolve(strict=True)
            if not path.is_relative_to(sim.robot.directory):
                raise ValueError("Visual mesh must remain inside the snapshot")
            source = trimesh.load(path, force="scene", process=False)
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            origin = visual.find("origin")
            position = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            angles = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            rotation = Rotation.from_euler("xyz", angles).as_matrix()
            if name not in handles:
                handles[name] = server.scene.add_frame(
                    f"/source_visuals/{name}",
                    show_axes=False,
                    position=sim.data.body(name).xpos,
                    wxyz=sim.data.body(name).xquat,
                )
            for index, node in enumerate(source.graph.nodes_geometry):
                transform, geometry_name = source.graph[node]
                part = source.geometry[geometry_name]
                vertices = trimesh.transform_points(part.vertices, transform) * scale
                vertices = vertices @ rotation.T + position
                rgba = part.visual.material.baseColorFactor
                if rgba is None:
                    raise ValueError(f"Missing source DAE color: {name}/{geometry_name}")
                color = tuple(int(value) for value in rgba[:3])
                colors.add(color)
                server.scene.add_mesh_simple(
                    f"/source_visuals/{name}/{visual.attrib['name']}_{index}",
                    vertices=vertices,
                    faces=part.faces,
                    color=color,
                    opacity=float(rgba[3]) / 255,
                )
                part_count += 1
            # Hide the STL visual duplicate in this preview's private model only.
            sim.model.geom(visual.attrib["name"]).group = 5
    return handles, {"bodies": len(handles), "parts": part_count, "rgb_colors": sorted(colors)}


def serve_preview(robot: DualArmModel, initial_q: np.ndarray, port: int) -> None:
    """Serve static joint sliders until interrupted; no physics steps or ROS commands."""
    # Keep visualization settings private to this scene.
    sim = DualArmSimulation(robot, scene_spec=robot.spec.copy())
    sim.reset(initial_q)
    server = viser.ViserServer(host="127.0.0.1", port=port, label="Dual CRX + Sharpa | offline preview")
    try:
        if server.get_port() != port:
            raise RuntimeError(f"Requested port {port} is occupied")
        server.initial_camera.position = (2.2, -2.6, 1.8)
        server.initial_camera.look_at = (0.2, -0.7, 0.35)
        server.initial_camera.up = (0, 0, 1)
        colored_bodies, material_report = add_source_visuals(server, sim)
        scene = ViserMujocoScene(server, sim.model, num_envs=1)
        scene.camera_tracking_enabled = False
        scene.geom_groups_visible[0] = False
        scene.geom_groups_visible[5] = False
        scene._sync_visibilities()
        arm_degrees = {
            side: np.rad2deg(initial_q[offset : offset + 6]).round(6).tolist()
            for side, offset in (("left", 0), ("right", 28))
        }
        server.gui.add_markdown(
            "**Dual-arm model · 56 joints**\n\nInitial pose: bimanual_manipulation defaults.\n\n"
            f"Left arm (degrees): {arm_degrees['left']}\n\nRight arm (degrees): {arm_degrees['right']}\n\n"
            "Both hands: all joints zero. Sliders use radians.\n\nStatic preview · uncalibrated preview mounts."
        )
        collision = server.gui.add_checkbox("Show collision geometry", initial_value=False)
        reset = server.gui.add_button("Reset to bimanual initial pose")
        info = server.gui.add_markdown("Loading model...")
        dirty = threading.Event()
        sliders = []
        for side, offset in (("Left", 0), ("Right", 28)):
            for part, start, end in (("arm", 0, 6), ("hand", 6, 28)):
                with server.gui.add_folder(f"{side} {part}", expand_by_default=False):
                    for index in range(offset + start, offset + end):
                        slider = server.gui.add_slider(
                            robot.joint_names[index],
                            min=float(robot.limits[index, 0]),
                            max=float(robot.limits[index, 1]),
                            step=0.001,
                            initial_value=float(initial_q[index]),
                        )
                        slider.on_update(lambda event: dirty.set())
                        sliders.append(slider)

        @reset.on_click
        def reset_pose(event: object) -> None:
            """Restore imported initial angles without advancing physics."""
            for index, slider in enumerate(sliders):
                slider.value = float(initial_q[index])
            dirty.set()

        collision.on_update(lambda event: dirty.set())
        dirty.set()
        print(f"PREVIEW_URL=http://localhost:{port}", flush=True)
        print("INITIAL_ARM_DEGREES=" + json.dumps(arm_degrees), flush=True)
        print("SOURCE_DAE_MATERIALS=" + json.dumps(material_report), flush=True)
        while True:
            if not dirty.wait(0.1):
                continue
            dirty.clear()
            sim.reset(np.array([slider.value for slider in sliders]))
            scene.geom_groups_visible[0] = collision.value
            scene._sync_visibilities()
            scene.update_from_mjdata(sim.data)
            for name, handle in colored_bodies.items():
                handle.position = sim.data.body(name).xpos
                handle.wxyz = sim.data.body(name).xquat
            counts = Counter(row["category"] for row in sim.check_static_contacts(sim.configuration()))
            info.content = "Static contact counts: " + ", ".join(
                f"{key}: {value}" for key, value in sorted(counts.items())
            )
    finally:
        server.stop()


def main() -> None:
    """Load explicit offline inputs and serve a local browser preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument(
        "--initial-pose", type=Path, help="Upstream degree YAML; defaults to snapshot/initial_pose.yaml"
    )
    parser.add_argument("--port", type=int, default=8089)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    robot = DualArmModel.load(args.snapshot)
    path = resolve_from_root(args.initial_pose) if args.initial_pose else robot.directory / "initial_pose.yaml"
    serve_preview(robot, load_initial_configuration(robot, path), args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
