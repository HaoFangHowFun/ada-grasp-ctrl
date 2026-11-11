"""Inspect selected endpoints and illustrative joint interpolation; no robot control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from ada_grasp_ctrl.experiments.crx_sharpa_right_model import matrix_pose
from ada_grasp_ctrl.experiments.crx_sharpa_right_preview import COLORS, hand_mesh
from ada_grasp_ctrl.experiments.fixed_camera_preview import plane_mesh
from ada_grasp_ctrl.paths import resolve_from_root
from .adapters.crx_sharpa import CrxSharpaAdapter
from .adapters.dexlearn import file_hash, load_candidates, read_json, verify_reference
from .cli import write_json
from .types import STAGES, rigid, vector


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform display geometry into the same model coordinates as the robot."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def display_without_contact_checks(adapter: CrxSharpaAdapter, q: np.ndarray) -> None:
    """Update display geometry only; the optional lift is not collision/path checked."""
    adapter.data.qpos[adapter.addresses] = q
    adapter.data.qvel[:] = 0
    mujoco.mj_kinematics(adapter.model, adapter.data)
    mujoco.mj_comPos(adapter.model, adapter.data)


def load_entries(summary_path: Path) -> list[dict]:
    """Verify saved inputs and selected endpoint FK without rerunning IK or inference."""
    summaries = json.loads(summary_path.read_text())
    entries = []
    for row in summaries:
        run = resolve_from_root(row["runs"][0])
        manifest = read_json(run / "manifest.json")
        for path, expected in manifest["input_sha256"].items():
            verify_reference({"path": path, "sha256": expected})
        result = read_json(run / "selection_result.json")
        configuration = result["configuration"]
        adapter = CrxSharpaAdapter(
            Path(configuration["robot"]["model_report"]),
            configuration["robot"]["table"],
            configuration["context"]["world_T_right_base"],
        )
        batch = load_candidates(resolve_from_root(manifest["arguments"]["candidates"]))
        asset_root = resolve_from_root(batch.payload["object_root"])
        asset = read_json(asset_root / "asset.json")
        for name, expected in asset["files_sha256"].items():
            verify_reference({"path": str(asset_root / name), "sha256": expected})
        with np.load(asset_root / "surface.npz", allow_pickle=False) as archive:
            surface = {name: archive[name].copy() for name in ("vertices", "faces", "colors_rgb", "T_world_asset")}
        surface_world = transform_points(surface["vertices"], rigid(surface["T_world_asset"], "asset"))
        surface["vertices"] = transform_points(surface_world, adapter.model_T_world)
        selected = result["selected"]
        targets, keyframes = {}, None
        if selected is not None:
            candidate = next(c for c in batch.candidates if c.candidate_id == result["selected_candidate_id"])
            if selected["original"] != candidate.original:
                raise ValueError("Selection payload differs from original candidate")
            poses = []
            for stage in STAGES:
                q = vector(selected["stage_checks"][stage]["q_rad"], 28, stage)
                target = adapter.model_T_world @ candidate.stages[stage].transform()
                _, reasons = adapter.check_solution(q, target, selected["stage_checks"][stage]["ik"])
                if reasons:
                    raise ValueError(f"Saved selected endpoint is no longer feasible: {stage}: {reasons}")
                if not np.array_equal(q[6:], candidate.stages[stage].joint_positions_rad):
                    raise ValueError("Selected hand targets differ from candidate")
                poses.append(q)
                targets[stage] = target
            if not np.array_equal(poses[1][:6], result["selected_q_rad"]):
                raise ValueError("Selected arm configuration differs from grasp endpoint")
            initial = np.r_[configuration["context"]["robot_state"]["positions_rad"], poses[0][6:]]
            keyframes = np.array([initial, *poses])
        else:
            # No invalid solution is animated. The original targets remain available as ghosts.
            if batch.candidates:
                targets = {s: adapter.model_T_world @ batch.candidates[0].stages[s].transform() for s in STAGES}
        entries.append(
            {
                "group": row["group"],
                "label": row["label"],
                "run": str(run),
                "result": result,
                "adapter": adapter,
                "surface": surface,
                "asset_root": asset_root,
                "keyframes": keyframes,
                "targets": targets,
                "selection_sha256": file_hash(run / "selection_result.json"),
            }
        )
    return entries


def interpolate(keyframes: np.ndarray, progress: float) -> np.ndarray:
    """Interpolate for display only, without wrapping limited joints or emitting a trajectory."""
    if not np.isfinite(progress) or not 0 <= progress <= 100:
        raise ValueError("Preview progress must be between 0 and 100")
    phase = progress * 3 / 100
    index = min(int(phase), 2)
    alpha = phase - index
    return (1 - alpha) * keyframes[index] + alpha * keyframes[index + 1]


def render_images(entries: list[dict], output: Path) -> None:
    """Save actual selected grasp views using MuJoCo rendering, without advancing physics."""
    from PIL import Image, ImageDraw

    for entry in entries:
        if entry["keyframes"] is None:
            continue
        adapter = entry["adapter"]
        spec = mujoco.MjSpec.from_file(adapter.info["mjcf"])
        spec.add_mesh(name="observed_object", file=str(entry["asset_root"] / "surface.obj"))
        asset = read_json(entry["asset_root"] / "asset.json")
        pose = matrix_pose(adapter.model_T_world @ rigid(asset["T_world_asset"], "asset"))
        obj = spec.worldbody.add_body(name="observed_object", pos=pose[:3], quat=pose[3:])
        obj.add_geom(
            name="observed_surface",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="observed_object",
            contype=0,
            conaffinity=0,
            group=1,
            rgba=[0.35, 0.8, 0.45, 1],
        )
        plane = adapter.provenance["table"]
        vertices, _ = plane_mesh(plane)
        center = transform_points(vertices, adapter.model_T_world).mean(axis=0)
        span = np.asarray(plane["observed_span_m"]) / 2
        quat = matrix_pose(adapter.table_model)[3:]
        spec.worldbody.add_geom(
            name="observed_table_patch",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=center,
            quat=quat,
            size=[*span, 0.01],
            group=1,
            contype=0,
            conaffinity=0,
            rgba=[0.6, 0.65, 0.7, 1],
        )
        spec.worldbody.add_light(pos=[1, -1, 2], dir=[-0.3, 0.2, -1])
        spec.visual.global_.offwidth = 1000
        spec.visual.global_.offheight = 750
        model = spec.compile()
        data = mujoco.MjData(model)
        for name, q in zip(adapter.info["joint_names"], entry["keyframes"][2]):
            data.qpos[model.joint(name).qposadr[0]] = q
        mujoco.mj_forward(model, data)
        option = mujoco.MjvOption()
        option.geomgroup[0] = option.geomgroup[3] = 0
        center = entry["surface"]["vertices"].mean(axis=0)
        with mujoco.Renderer(model, height=750, width=1000) as renderer:
            for close in (False, True):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = center + [0, 0, 0.05] if close else [0.3, -0.65, 0.2]
                camera.distance, camera.azimuth, camera.elevation = (0.5 if close else 1.7), 130, -25
                renderer.update_scene(data, camera=camera, scene_option=option)
                picture = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 1000, 50), fill=(25, 30, 40))
                draw.text(
                    (12, 8),
                    f"Group {entry['group']} | selected {entry['result']['selected_candidate_id']} | GRASP FK",
                    fill="white",
                )
                draw.text(
                    (12, 28),
                    "Static preview | calibration pending | incomplete scene | no motion commands",
                    fill="white",
                )
                picture.save(output / f"group{entry['group']}_{'closeup' if close else 'overview'}.png")


def serve(
    entries: list[dict],
    port: int,
    duration_s: float = 0,
    status_path: Path | None = None,
    execution_gate: bool = False,
) -> None:
    """Serve all five datasets on one port with phase, playback and target controls."""
    import viser
    import trimesh
    from mjviser import ViserMujocoScene
    from ada_grasp_ctrl.pipeline.plan import phase_progress, sample_plan, phase_at_progress

    server = viser.ViserServer(host="127.0.0.1", port=port, label="Grasp candidates · Right CRX + Sharpa")
    if server.get_port() != port:
        server.stop()
        raise RuntimeError(f"Port {port} is occupied; refusing another port")
    labels = {f"{e['group']} · {e['label']}": e for e in entries}
    choice = server.gui.add_dropdown("Dataset", options=list(labels))
    phase = server.gui.add_dropdown("Pose", options=["Current arm", *STAGES], initial_value="grasp")
    play = server.gui.add_checkbox("Play interpolation", initial_value=False)
    progress = server.gui.add_slider("Progress %", min=0, max=100, step=0.1, initial_value=200 / 3)
    speed = server.gui.add_dropdown("Playback speed", options=["0.25x", "0.5x", "1x"], initial_value="0.5x")
    collision = server.gui.add_checkbox("Show collision geometry", initial_value=False)
    ghosts = server.gui.add_checkbox("Show original grasp targets", initial_value=True)
    camera = server.gui.add_dropdown("View", options=["Overview", "Hand and object"], initial_value="Overview")
    planned = any("plan" in entry for entry in entries)
    gate_note = (
        "\n\nSelect a dataset, then press Enter in the terminal to execute it. No motion is sent until you press Enter."
        if execution_gate
        else "\n\nPreview only. No motion commands are sent in this mode."
    )
    banner = server.gui.add_markdown(
        "**Mock grasp preview**\n\nStarts from measured arm and finger positions, "
        "using the execution runtime's quintic interpolation and planned timing."
        + gate_note
        + "\n\nThis previews the commands. The full environment and hardware motion path have not been validated."
        if planned
        else "**Preview only. The execution path has not been validated.**\n\n"
        "Playback shows current arm → pregrasp → grasp → squeeze. Interpolation is not a planned motion path.\n\n"
        "Initial fingers use the pregrasp target; current finger positions were not measured. "
        "The object is a single-view surface; calibration and the table plane await acceptance."
    )
    info = server.gui.add_markdown("Loading...")
    state: dict[str, Any] = {"choice": None, "seek": 200 / 3, "progress": 200 / 3, "phase": "grasp"}

    def set_camera(client: Any) -> None:
        entry = labels[choice.value]
        center = entry["surface"]["vertices"].mean(axis=0)
        close = camera.value == "Hand and object"
        target = center + [0, 0, 0.05] if close else np.array([0.35, -0.65, 0.2])
        with client.atomic():
            client.camera.up_direction = np.array([0, 0, 1])
            client.camera.look_at = target
            client.camera.position = target + (np.array([0.35, -0.35, 0.22]) if close else [1.1, -1.1, 0.8])

    server.initial_camera.position = (1.45, -1.75, 1.0)
    server.initial_camera.look_at = (0.35, -0.65, 0.2)
    server.initial_camera.up = (0, 0, 1)
    server.on_client_connect(set_camera)
    camera.on_update(lambda event: [set_camera(c) for c in server.get_clients().values()])

    @progress.on_update
    def seek(event: Any) -> None:
        if event.client is not None:
            state["seek"] = progress.value

    @play.on_update
    def start(event: Any) -> None:
        if play.value:
            state["seek"] = 0.0

    print(f"PREVIEW_URL=http://localhost:{port}", flush=True)
    started = last = time.monotonic()
    try:
        while not duration_s or time.monotonic() - started < duration_s:
            now = time.monotonic()
            dt, last = min(now - last, 0.1), now
            entry = labels[choice.value]
            adapter = entry["adapter"]
            stages = entry.get("plan", {}).get("phases", list(STAGES))
            if state["choice"] != choice.value:
                phase.options = ["Current arm", *stages]
                if "plan" in entry:
                    environment = "Hardware" if entry["plan"]["environment"] == "hardware" else "Mock"
                    banner.content = (
                        f"**{environment} grasp preview**\n\n"
                        + " → ".join(["Current pose", *stages])
                        + gate_note
                        + "\n\nUses the execution runtime's interpolation and timing. "
                        "The full hardware motion path has not been validated."
                    )
                server.scene.reset()
                scene = ViserMujocoScene(server, adapter.model, num_envs=1)
                scene.camera_tracking_enabled = False
                if "environment_points" in entry:
                    server.scene.add_point_cloud(
                        "/environment_points",
                        points=entry["environment_points"],
                        colors=(110, 130, 150),
                        point_size=0.002,
                    )
                surface = entry["surface"]
                server.scene.add_mesh_trimesh(
                    "/measured_surface",
                    trimesh.Trimesh(
                        vertices=surface["vertices"],
                        faces=surface["faces"],
                        vertex_colors=surface["colors_rgb"],
                        process=False,
                    ),
                )
                server.scene.add_point_cloud(
                    "/measured_points", points=surface["vertices"], colors=surface["colors_rgb"], point_size=0.0015
                )
                table_vertices, table_faces = plane_mesh(adapter.provenance["table"])
                server.scene.add_mesh_simple(
                    "/observed_table",
                    vertices=transform_points(table_vertices, adapter.model_T_world),
                    faces=table_faces,
                    color=(125, 140, 155),
                    opacity=0.5,
                    side="double",
                )
                overlay = server.scene.add_frame("/targets", show_axes=False)
                for i, (stage, target) in enumerate(entry["targets"].items()):
                    target_pose = matrix_pose(target)
                    server.scene.add_frame(
                        f"/targets/{stage}",
                        position=target_pose[:3],
                        wxyz=target_pose[3:],
                        axes_length=0.04,
                        axes_radius=0.001,
                    )
                    if entry["keyframes"] is not None:
                        if stage in ("lift", "hold", "release", "return"):
                            display_without_contact_checks(adapter, entry["keyframes"][i + 1])
                        else:
                            adapter.static_check(entry["keyframes"][i + 1])
                        v, f = hand_mesh(adapter.model, adapter.data, target)
                        server.scene.add_mesh_simple(
                            f"/targets/ghost_{stage}",
                            vertices=v,
                            faces=f,
                            color=COLORS[i % len(COLORS)],
                            opacity=0.18,
                            side="double",
                        )
                state.update(choice=choice.value, scene=scene, overlay=overlay, seek=200 / 3, phase="grasp")
                if "plan" in entry:
                    state["seek"] = phase_progress(entry["plan"], stages.index("grasp") + 1)
                phase.value = "grasp"
                for client in server.get_clients().values():
                    set_camera(client)
                print(
                    f"PREVIEW_READY group={entry['group']} selected={entry['result']['selected_candidate_id']}",
                    flush=True,
                )
                if status_path is not None:
                    temporary = status_path.with_suffix(".tmp")
                    write_json(
                        temporary,
                        {
                            "group": entry["group"],
                            "selected": entry["result"]["selected_candidate_id"],
                            "selection_sha256": entry["selection_sha256"],
                            "ready": True,
                            "phases": list(stages),
                            "phase_progress": {
                                stage: phase_progress(entry["plan"], i + 1) for i, stage in enumerate(stages)
                            }
                            if "plan" in entry
                            else {},
                            "execution_error": entry.get("execution_error"),
                        },
                    )
                    temporary.replace(status_path)
            if phase.value != state["phase"]:
                state["phase"] = phase.value
                state["seek"] = ["Current arm", *stages].index(phase.value) * 100 / len(stages)
                if "plan" in entry:
                    state["seek"] = phase_progress(entry["plan"], ["Current arm", *stages].index(phase.value))
                play.value = False
            if state["seek"] is not None:
                state["progress"], state["seek"] = float(state["seek"]), None
            elif play.value and server.get_clients():
                total = (
                    sum(entry["plan"]["durations_s"]) + len(stages) * entry["plan"]["settle_time_s"]
                    if "plan" in entry
                    else 9
                )
                state["progress"] = (state["progress"] + dt * float(speed.value[:-1]) * 100 / total) % 100
            state["overlay"].visible = ghosts.value
            result = entry["result"]
            if entry["keyframes"] is not None:
                q = (
                    sample_plan(entry["plan"], state["progress"])
                    if "plan" in entry
                    else interpolate(entry["keyframes"], state["progress"])
                )
                plan = entry.get("plan", {})
                lift_start = (
                    (sum(plan["durations_s"][: stages.index("lift")]) + stages.index("lift") * plan["settle_time_s"])
                    / (sum(plan["durations_s"]) + len(stages) * plan["settle_time_s"])
                    * 100
                    if "lift" in stages
                    else 101
                )
                if state["progress"] >= lift_start:
                    display_without_contact_checks(adapter, np.asarray(q))
                    current_phase = phase_at_progress(plan, state["progress"])
                    info.content = (
                        f"**{entry['label']} · {result['selected_candidate_id']} · {current_phase}**\n\n"
                        "Lift and return are excluded from candidate filtering "
                        "and have not been checked for path feasibility or collisions.\n\n"
                        + (
                            f"After reaching the lift target, hold the squeeze for {plan['cycle']['hold_s']:g} s, "
                            "then open the hand and return the arm to its starting pose.\n\n"
                            if "cycle" in plan
                            else "Fingers remain in the squeeze pose.\n\n"
                        )
                        + "Displaying arm targets does not confirm that the object has been grasped or lifted."
                    )
                else:
                    diagnostic = adapter.static_check(np.asarray(q))
                    known = (
                        "Passed"
                        if diagnostic["known_geometry_passed"]
                        else "**Collision exceeds the existing threshold**"
                    )
                    info.content = (
                        f"**{entry['label']} · {result['selected_candidate_id']}**\n\n"
                        f"Joint displacement cost: {result['cost']:.6f} rad²\n\n"
                        f"Existing penetration threshold check for the displayed pose: {known}\n\n"
                        f"Arm penetration: {diagnostic['arm_penetration_m'] * 1000:.3f} mm; "
                        f"Hand self-penetration: {diagnostic['hand_self_penetration_m'] * 1000:.3f} mm; "
                        f"Hand/table penetration: {diagnostic['hand_table_penetration_m'] * 1000:.3f} mm.\n\n"
                        "The hand penetration tolerance is 10 mm, as in the existing experiments; "
                        "this does not guarantee zero collisions.\n\n"
                        "Collisions with the object or other surroundings have not been checked. "
                        "No physics simulation or grasp has been executed."
                    )
            else:
                play.value = False
                info.content = (
                    "**No feasible candidates.** Original targets are shown for reference only; "
                    "failed IK solutions are not played."
                )
            if entry.get("execution_error"):
                info.content = (
                    "**Motion target generation failed. This dataset cannot be executed.**\n\n"
                    + entry["execution_error"]
                    + "\n\nThe originally selected grasp is retained. Only the three grasp stages are displayed.\n\n"
                    + info.content
                )
            scene = state["scene"]
            scene.geom_groups_visible[0] = scene.geom_groups_visible[3] = collision.value
            scene._sync_visibilities()
            scene.update_from_mjdata(adapter.data)
            scene.request_update()
            progress.value = round(state["progress"], 1)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> int:
    """Export review images or serve the unified interactive gallery on port 8089."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--duration-s", type=float, default=0)
    args = parser.parse_args(argv)
    output = resolve_from_root(args.output_root, "output")
    output.mkdir(parents=True, exist_ok=False)
    entries = load_entries(resolve_from_root(args.summary))
    render_images(entries, output)
    write_json(
        output / "preview_manifest.json",
        {
            "schema_version": 1,
            "port": args.port,
            "source_summary": str(resolve_from_root(args.summary)),
            "groups": [
                {"group": e["group"], "run": e["run"], "selection_sha256": e["selection_sha256"]} for e in entries
            ],
            "physics_steps": 0,
            "motion_commands": 0,
            "animation": "illustrative joint interpolation; not a planned path",
            "initial_hand": "pregrasp target; current hand state not measured",
            "execution_ready": False,
        },
    )
    if not args.export_only:
        serve(entries, args.port, args.duration_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
