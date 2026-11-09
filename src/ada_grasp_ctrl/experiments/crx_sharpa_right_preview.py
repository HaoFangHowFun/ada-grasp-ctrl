"""Interactive static preview and MuJoCo screenshots of prepared right CRX scenes."""

from __future__ import annotations

from pathlib import Path
import threading

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .crx_sharpa_right_model import matrix_pose, pose_matrix
from .sharpa_model import FLANGE, PALM
from .crx_sharpa_right_preparation import body_pose, set_state
from .hugs_sharpa_input import FIELDS

STAGES = ("pregrasp", "grasp", "squeeze")
COLORS = ((60, 180, 255), (70, 220, 110), (255, 110, 80))


def hand_mesh(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place the canonical visual hand exactly at the requested flange target."""
    correction = target @ np.linalg.inv(body_pose(data, FLANGE))
    vertices, faces, offset = [], [], 0
    root_id = model.body(FLANGE).id
    hand_bodies = {root_id}
    for body in range(root_id + 1, model.nbody):
        if model.body_parentid[body] in hand_bodies:
            hand_bodies.add(body)
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] not in hand_bodies or model.geom_group[geom] != 1:
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        v = model.mesh_vert[first : first + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        v = v @ correction[:3, :3].T + correction[:3, 3]
        first, count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices.append(v)
        faces.append(model.mesh_face[first : first + count] + offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def render_screenshots(report: dict, output: Path) -> list[str]:
    """Render overview and hand-object closeups from the exact prepared scene files."""
    paths = []
    for row in report["samples"]:
        model = mujoco.MjModel.from_xml_path(row["scene_path"])
        data = mujoco.MjData(model)
        record = np.load(row["prepared_path"], allow_pickle=True).item()
        set_state(model, data, record, FIELDS[0])
        option = mujoco.MjvOption()
        option.geomgroup[0] = 0
        option.geomgroup[3] = 0
        with mujoco.Renderer(model, height=960, width=1280) as renderer:
            for mode in ("initial", "overview", "closeup"):
                close = mode == "closeup"
                set_state(model, data, record, "initial_qpos" if mode == "initial" else FIELDS[0])
                camera = mujoco.MjvCamera()
                camera.lookat[:] = record["obj_pose"][:3] + np.array([0, 0, 0.08]) if close else [0.42, -0.70, 0.28]
                camera.distance = 0.48 if close else 2.0
                camera.azimuth = 140
                camera.elevation = -25
                renderer.update_scene(data, camera=camera, scene_option=option)
                # Target origins are colored; FK poses remain the opaque robot.
                for target, color in zip(row["evaluation_flange_targets_wxyz"], COLORS):
                    transform = pose_matrix(target)
                    for axis in range(3):
                        scene = renderer.scene
                        geom = scene.geoms[scene.ngeom]
                        scene.ngeom += 1
                        rgba = np.r_[np.asarray(color) / 255, 1].astype(np.float32)
                        mujoco.mjv_initGeom(
                            geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba
                        )
                        mujoco.mjv_connector(
                            geom,
                            mujoco.mjtGeom.mjGEOM_ARROW,
                            0.0015,
                            transform[:3, 3],
                            transform[:3, 3] + 0.035 * transform[:3, axis],
                        )
                picture = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(picture)
                diag = row["targets"][0]
                pose_label = (
                    "CONFIG INITIAL J1-J6 (deg): " + str(np.round(np.rad2deg(record["initial_qpos"][:6]), 2).tolist())
                    if mode == "initial"
                    else f"PREGRASP IK: {diag['position_m']:.3g} m, {diag['orientation_rad']:.3g} rad"
                )
                contact_state = row["initial_state"] if mode == "initial" else diag
                lines = [
                    f"RIGHT CRX + RIGHT SHARPA | {row['sample_id']} | scale {row['scale']:.3f} "
                    "| STATIC (0 physics steps)",
                    pose_label,
                    "m / rad / WXYZ | mount: uncalibrated | blue pregrasp / green grasp / orange squeeze",
                    f"Object bottom/table gap: {row['geometry']['object_bottom_to_table_m']:.6f} m | "
                    f"max penetration: {contact_state['static_contacts']['max_penetration_m']:.6f} m",
                ]
                draw.rectangle((0, 0, 1280, 82), fill=(25, 30, 40))
                for i, line in enumerate(lines):
                    draw.text((14, 8 + 18 * i), line, fill="white")
                path = output / row["sample_id"] / ("initial_overview.png" if mode == "initial" else f"{mode}.png")
                picture.save(path)
                paths.append(str(path))
    return paths


def serve_preview(report: dict, *, port: int = 8086, sample_id: str | None = None) -> None:
    """Serve paused scenes with fixed sample, stage, ghost overlay and camera controls."""
    import viser
    from mjviser import ViserMujocoScene

    server = viser.ViserServer(host="127.0.0.1", port=port, label="Right CRX + right Sharpa | placement review")
    server.initial_camera.position = (1.62, -1.90, 1.18)
    server.initial_camera.look_at = (0.42, -0.70, 0.28)
    server.initial_camera.up = (0, 0, 1)
    entries = {r["sample_id"]: r for r in report["samples"]}
    sample = server.gui.add_dropdown(
        "Sample (four scales)", options=list(entries), initial_value=sample_id or next(iter(entries))
    )
    stage = server.gui.add_dropdown(
        "Robot pose", options=["Initial (config)", *STAGES], initial_value="Initial (config)"
    )
    ghosts = server.gui.add_dropdown(
        "Original target ghosts", options=["Selected target", "All three", "Hidden"], initial_value="Selected target"
    )
    collision = server.gui.add_checkbox("Collision geometry", initial_value=False)
    axes = server.gui.add_checkbox("Coordinate frames", initial_value=True)
    camera = server.gui.add_dropdown("Camera", options=["Overview", "Hand and object"], initial_value="Overview")
    server.gui.add_markdown(
        "**Paused physics · m / rad · quaternion WXYZ**\n\n"
        "Initial (config) uses the configured joint angles; other poses show solved IK/FK.\n\n"
        "Transparent ghosts = original registered targets.\n\n"
        "Blue pregrasp · green grasp · orange squeeze. Axis colors: X red / Y green / Z blue.\n\n"
        "Mount is a reference preview placeholder, not hardware calibration."
    )
    info = server.gui.add_markdown("Loading...")
    dirty = threading.Event()
    state = {}

    def set_camera(client: object) -> None:
        row = entries[sample.value]
        center = np.asarray(row["evaluation_object_wxyz"][:3])
        close = camera.value == "Hand and object"
        look_at = center + [0, 0, 0.07] if close else np.array([0.42, -0.70, 0.28])
        position = look_at + (np.array([0.32, -0.32, 0.25]) if close else np.array([1.2, -1.2, 0.9]))
        with client.atomic():
            client.camera.up_direction = np.array([0, 0, 1])
            client.camera.position = position
            client.camera.look_at = look_at

    server.on_client_connect(set_camera)
    for handle in (sample, stage, ghosts, collision, axes):
        handle.on_update(lambda event: dirty.set())
    camera.on_update(lambda event: [set_camera(client) for client in server.get_clients().values()])
    dirty.set()
    print(f"PREVIEW_URL=http://localhost:{server.get_port()}", flush=True)
    try:
        while True:
            if not dirty.wait(0.1):
                continue
            dirty.clear()
            row = entries[sample.value]
            if state.get("sample") != sample.value:
                server.scene.reset()
                model = mujoco.MjModel.from_xml_path(row["scene_path"])
                data = mujoco.MjData(model)
                record = np.load(row["prepared_path"], allow_pickle=True).item()
                scene = ViserMujocoScene(server, model, num_envs=1)
                scene.camera_tracking_enabled = False
                state = {
                    "sample": sample.value,
                    "model": model,
                    "data": data,
                    "record": record,
                    "scene": scene,
                    "overlay": server.scene.add_frame("/review", show_axes=False),
                }
                for client in server.get_clients().values():
                    set_camera(client)
            model, data, record, scene = (state[k] for k in ("model", "data", "record", "scene"))
            initial = stage.value == "Initial (config)"
            index = 0 if initial else STAGES.index(stage.value)
            state["overlay"].remove()
            state["overlay"] = server.scene.add_frame("/review", show_axes=False)
            for i, name in enumerate(STAGES):
                target = np.asarray(row["evaluation_flange_targets_wxyz"][i])
                frame = server.scene.add_frame(
                    f"/review/{name}",
                    position=target[:3],
                    wxyz=target[3:],
                    axes_length=0.045,
                    axes_radius=0.001,
                    visible=axes.value,
                )
                server.scene.add_label(
                    f"/review/{name}/label", text=f"TARGET {name}", position=(0, 0, 0.05), visible=axes.value
                )
                if ghosts.value == "All three" or (ghosts.value == "Selected target" and i == index):
                    set_state(model, data, record, FIELDS[i])
                    vertices, faces = hand_mesh(model, data, pose_matrix(target))
                    server.scene.add_mesh_simple(
                        f"/review/ghost_{name}",
                        vertices=vertices,
                        faces=faces,
                        color=COLORS[i],
                        opacity=0.3,
                        side="double",
                    )
                frame.visible = axes.value
            set_state(model, data, record, "initial_qpos" if initial else FIELDS[index])
            scene.geom_groups_visible[0] = collision.value
            scene.geom_groups_visible[3] = collision.value
            scene.update_from_mjdata(data)
            scene.request_update()
            frames = {
                "world": np.eye(4),
                "right_CRX_base": np.asarray(report["model"]["world_T_base"]),
                "object": pose_matrix(record["obj_pose"]),
                "FK_flange": body_pose(data, FLANGE),
                "FK_palm": body_pose(data, PALM),
            }
            for name, transform in frames.items():
                pose = matrix_pose(transform)
                server.scene.add_frame(
                    f"/review/frame_{name}",
                    position=pose[:3],
                    wxyz=pose[3:],
                    axes_length=0.055,
                    axes_radius=0.0015,
                    visible=axes.value,
                )
                server.scene.add_label(
                    f"/review/frame_{name}/label", text=name, position=(0, 0, 0.06), visible=axes.value
                )
            bounds = np.asarray(row["geometry"]["world_aabb_m"])
            server.scene.add_box(
                "/review/bounds",
                dimensions=tuple(bounds[1] - bounds[0]),
                position=bounds.mean(axis=0),
                color=(255, 210, 60),
                wireframe=True,
            )
            diag = row["initial_state"] if initial else row["targets"][index]
            pose_info = (
                "**Configured initial pose** · J1–J6 (deg): "
                f"{np.round(np.rad2deg(record['initial_qpos'][:6]), 3).tolist()}\n\n"
                if initial
                else f"IK reachable: **{diag['reachable']}**; position {diag['position_m']:.3g} m; "
                f"angle {diag['orientation_rad']:.3g} rad.\n\n"
            )
            info.content = (
                f"**{row['sample_id']} · {row['object_id']} · scale {row['scale']:.3f}**\n\n"
                f"{pose_info}"
                f"Object dimensions (m): {np.round(row['geometry']['world_dimensions_m'], 5).tolist()}\n\n"
                f"Table z={row['geometry']['table_height_m']:.4f} m; "
                f"bottom gap={row['geometry']['object_bottom_to_table_m']:.6f} m.\n\n"
                f"Penetration: {diag['static_contacts']['max_penetration_m']:.6f} m; "
                f"arm/table collision: {diag['static_contacts']['arm_table_penetration']}.\n\n"
                f"Joint limit violations: {diag['joint_limit_violations']}\n\n"
                f"Relative grasp invariant: {all(c['passed'] for c in row['relative_pose_checks'])}. "
                "No controller or lift has run."
            )
    finally:
        server.stop()
