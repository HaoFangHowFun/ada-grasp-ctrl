"""Preview the exact selected-side live cycle in its measured dual-arm scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
import yaml
from mjviser import ViserMujocoScene
import viser

from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json, file_hash
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.robot_environment.model import side_slice
from .dual_scene import live_scene
from .plan import sample_plan, phase_at_progress, phase_progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    config = yaml.safe_load((root / "effective_config.yaml").read_text())
    packet_path = root / "export/targets.json"
    packet = load_packet(packet_path)
    plan = packet["reviewed_plan"]
    state = read_json(root / "state/robot_state.json")
    _, sim = live_scene(
        resolve_from_root(config["robot_snapshot"]), state, read_json(resolve_from_root(config["table_plane"]))
    )
    q56 = np.asarray(state["full_positions_rad"])
    block = side_slice(packet["side"])
    server = viser.ViserServer(host="127.0.0.1", port=8089, label="Live dual-arm grasp")
    try:
        if server.get_port() != 8089:
            raise RuntimeError("Port 8089 is occupied")
        server.initial_camera.position = (1.7, -1.5, 1.1)
        server.initial_camera.look_at = (0.4, -0.4, 0.0)
        server.initial_camera.up = (0, 0, 1)
        scene = ViserMujocoScene(server, sim.model, num_envs=1)
        scene.camera_tracking_enabled = False
        scene.geom_groups_visible[0] = False
        scene.geom_groups_visible[3] = False
        scene._sync_visibilities()
        with np.load(root / "object/object.npz", allow_pickle=False) as data:
            points = data["points_world_m"]
        server.scene.add_point_cloud(
            "/object", points=points, colors=np.tile([80, 200, 80], (len(points), 1)), point_size=0.003
        )
        with np.load(root / "capture/snapshot.npz", allow_pickle=False) as data:
            points = data["points_world_m"][::10]
        server.scene.add_point_cloud(
            "/environment", points=points, colors=np.tile([160, 160, 160], (len(points), 1)), point_size=0.002
        )
        server.gui.add_markdown(
            f"**Selected: {packet['side'].upper()} · {plan['candidate_id']}**\n\n"
            "Review the full cycle, then press Enter in the pipeline terminal to execute."
        )
        progress = server.gui.add_slider("Playback (%)", min=0, max=100, step=0.1, initial_value=0.0)
        playing = server.gui.add_checkbox("Play", initial_value=False)
        phase = server.gui.add_dropdown("Phase", options=tuple(plan["phases"]), initial_value=plan["phases"][0])
        phase.on_update(
            lambda event: setattr(progress, "value", phase_progress(plan, plan["phases"].index(phase.value) + 1))
        )
        label = server.gui.add_markdown("Start")
        duration = sum(plan["durations_s"]) + len(plan["phases"]) * plan["settle_time_s"]
        ready = False
        previous = time.monotonic()
        while True:
            now = time.monotonic()
            if playing.value:
                progress.value = (progress.value + (now - previous) / duration * 100) % 100
            previous = now
            q = q56.copy()
            q[block] = sample_plan(plan, float(progress.value))
            sim.data.qpos[sim.qpos] = q
            mujoco.mj_forward(sim.model, sim.data)
            scene.update_from_mjdata(sim.data)
            label.content = f"**{phase_at_progress(plan, progress.value)}** · {packet['side']}"
            if not ready:
                write_json(
                    root / "viewer_state.json",
                    dict(side=packet["side"], candidate_id=plan["candidate_id"], packet_sha256=file_hash(packet_path)),
                )
                print("PREVIEW_URL=http://localhost:8089", flush=True)
                ready = True
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
