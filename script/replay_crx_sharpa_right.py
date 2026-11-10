"""Play saved right CRX trajectories without advancing physics."""

import argparse
import json
from pathlib import Path
from typing import Any
import time
import mujoco
from mjviser import ViserMujocoScene
import numpy as np
import viser
from ada_grasp_ctrl.experiments.episode_recording import load_episode
from ada_grasp_ctrl.paths import project_root


def main() -> None:
    """Serve paired saved episodes; no control commands or physics steps are executed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--sample-id", default="g0000")
    args = parser.parse_args()
    args.run_root = (project_root() / args.run_root).resolve()
    report = json.loads((args.run_root / "run_report.json").read_text())
    rows = {r["method"]: r for r in report["results"] if r["sample_id"] == args.sample_id and r.get("replay_path")}
    if set(rows) != {"op", "ours"}:
        raise ValueError(f"Sample {args.sample_id} must have both op and ours saved replays")
    records = {method: load_episode(Path(row["replay_path"])) for method, row in rows.items()}
    models = {method: mujoco.MjModel.from_xml_path(record["scene_path"]) for method, record in records.items()}
    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="Right CRX | grasp and lift replay")
    if server.get_port() != args.port:
        server.stop()
        raise RuntimeError("Requested port is occupied")
    server.initial_camera.position = (1.62, -1.90, 1.18)
    server.initial_camera.look_at = (0.42, -0.70, 0.28)
    server.initial_camera.up = (0, 0, 1)
    method_select = server.gui.add_dropdown("Method", options=["op", "ours"], initial_value="op")
    play = server.gui.add_checkbox("Play", initial_value=True)
    cycle = server.gui.add_checkbox("Loop both methods", initial_value=True)
    speed = server.gui.add_dropdown("Speed", options=["0.25x", "0.5x", "1x", "2x"], initial_value="0.5x")
    progress = server.gui.add_slider("Progress (%)", min=0, max=100, step=0.1, initial_value=0)
    restart = server.gui.add_button("Restart")
    camera = server.gui.add_dropdown("Camera", options=["Overview", "Hand and object"], initial_value="Overview")
    collision = server.gui.add_checkbox("Collision geometry", initial_value=False)
    server.gui.add_markdown(
        f"**{args.sample_id}: recorded grasp and lift**\n\n"
        + " · ".join(f"{method}: {row['status']}" for method, row in rows.items())
    )
    info = server.gui.add_markdown("Loading...")
    state = {"method": None, "time": 0.0, "seek": None}

    def set_camera(client: Any) -> None:
        """Apply an overview or object-centered camera to one browser client."""
        close = camera.value == "Hand and object"
        target = (
            np.asarray(records[method_select.value]["frames"][0]["object_pose"][:3]) + [0, 0, 0.08]
            if close
            else np.array([0.42, -0.70, 0.28])
        )
        offset = np.array([0.4, -0.5, 0.3]) if close else np.array([1.2, -1.2, 0.9])
        with client.atomic():
            client.camera.up_direction = np.array([0, 0, 1])
            client.camera.look_at = target
            client.camera.position = target + offset

    server.on_client_connect(set_camera)
    camera.on_update(lambda event: [set_camera(client) for client in server.get_clients().values()])

    @progress.on_update
    def seek(event: Any) -> None:
        """Seek only on browser input; ignore server progress updates."""
        if event.client is not None:
            state["seek"] = progress.value

    @restart.on_click
    def reset(event: Any) -> None:
        """Restart playback of the selected saved episode."""
        state["seek"] = 0.0

    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            dt, last = min(now - last, 0.2), now
            selected = method_select.value
            if state["method"] != selected:
                server.scene.reset()
                model = models[selected]
                data = mujoco.MjData(model)
                scene = ViserMujocoScene(server, model, num_envs=1)
                scene.camera_tracking_enabled = False
                scene.geom_groups_visible[0] = collision.value
                scene.geom_groups_visible[3] = collision.value
                scene._sync_visibilities()
                frames = records[selected]["frames"]
                times = np.array([f["time"] for f in frames])
                duration = float(times[-1])
                state.update(method=selected, time=0.0)
                print(
                    f"REPLAY_READY method={selected} frames={len(frames)} "
                    f"duration={duration:.2f}s port={server.get_port()}",
                    flush=True,
                )
            if state["seek"] is not None:
                state["time"] = float(state["seek"]) * duration / 100
                state["seek"] = None
            elif play.value and server.get_clients():
                state["time"] += dt * float(speed.value[:-1])
            if state["time"] > duration + 0.8:
                if cycle.value:
                    method_select.value = "ours" if selected == "op" else "op"
                    continue
                state["time"] = duration
                play.value = False
            index = max(0, int(np.searchsorted(times, min(state["time"], duration), side="right") - 1))
            frame = frames[index]
            for field in ("qpos", "qvel", "ctrl", "act"):
                getattr(data, field)[:] = frame[field]
            data.time = frame["time"]
            mujoco.mj_forward(model, data)
            scene.geom_groups_visible[0] = collision.value
            scene.geom_groups_visible[3] = collision.value
            scene._sync_visibilities()
            scene.update_from_mjdata(data)
            scene.request_update()
            progress.value = round(100 * min(state["time"], duration) / duration, 1)
            info.content = (
                f"**{selected} · {frame['segment']}**\n\n"
                f"{frame['time']:.1f} / {duration:.1f} s · frame {index + 1}/{len(frames)}\n\n"
                f"Object Z: {frame['object_pose'][2]:.4f} m\n\n"
                f"Contact steps: {rows[selected]['contact_steps']} · Result: {rows[selected]['status']}"
            )
            time.sleep(0.033)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
