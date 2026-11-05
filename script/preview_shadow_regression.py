"""Browse paired Phase 1 and Phase 2 targets and recorded states."""

import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import viser
from ada_grasp_ctrl.experiments.episode_recording import load_episode
from mjviser import ViserMujocoScene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--label", default="Shadow regression | friction 0.8")
    args = parser.parse_args()
    phase_rows = {
        phase: json.loads((args.run_root / phase / "phase_results.json").read_text()) for phase in ["phase1", "phase2"]
    }
    phase_entries = {phase: {r["sample_id"]: r for r in rows} for phase, rows in phase_rows.items()}
    server = viser.ViserServer(host="127.0.0.1", port=args.port, label=f"{args.label} — Paired phases")
    phase_select = server.gui.add_dropdown("Phase", options=["phase1", "phase2"], initial_value="phase1")
    overview = server.gui.add_markdown("Loading...")
    entries = phase_entries["phase1"]
    sample = server.gui.add_dropdown("Sample", options=list(entries), initial_value=next(iter(entries)))
    mode = server.gui.add_dropdown(
        "View",
        options=["Target poses", "Recorded replay"],
        initial_value="Target poses",
    )
    stage = server.gui.add_dropdown(
        "Target pose",
        options=["pregrasp", "grasp", "squeeze"],
        initial_value="pregrasp",
    )
    progress = server.gui.add_slider("Replay progress (%)", min=0, max=100, step=1, initial_value=0)
    play = server.gui.add_checkbox("Play replay", initial_value=False)
    collision = server.gui.add_checkbox("Show hand collision geometry", initial_value=False)
    reset_camera = server.gui.add_button("Reset camera")
    info = server.gui.add_markdown("Loading...")
    dirty = threading.Event()
    camera_dirty = threading.Event()
    state = {}

    def update(event):
        dirty.set()

    for handle in (phase_select, sample, mode, stage, progress, play, collision):
        handle.on_update(update)
    reset_camera.on_click(lambda event: camera_dirty.set())

    def camera(client):
        center = state.get("center", np.array([0.0, 0.0, 0.1]))
        client.camera.look_at = center
        client.camera.position = center + np.array([0.35, -0.45, 0.3])
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    server.on_client_connect(camera)
    dirty.set()
    last = time.monotonic()
    play_time = 0.0
    print(f"PREVIEW_URL=http://localhost:{server.get_port()}", flush=True)
    try:
        while True:
            now = time.monotonic()
            dt = now - last
            last = now
            if play.value and "record" in state:
                mode.value = "Recorded replay"
                duration = state["record"]["frames"][-1]["time"]
                if duration > 0:
                    play_time += dt
                    progress.value = min(100, int(100 * play_time / duration))
                    if play_time >= duration:
                        play.value = False
                else:
                    play.value = False
            else:
                play_time = progress.value / 100 * state.get("duration", 0.0)
            if dirty.wait(0.025):
                dirty.clear()
                key = sample.value
                phase = phase_select.value
                entry = phase_entries[phase][key]
                counts = Counter(r["status"] for r in phase_rows[phase])
                controller = "No adaptive controller (op)" if phase == "phase1" else "Ada controller enabled (ours)"
                overview.content = (
                    f"## {args.label} · {phase}\n{controller}\n\n{dict(counts)}\n\n"
                    "Matched initialization; pregrasp x2; no extra squeeze."
                )
                if state.get("sample") != key or state.get("phase") != phase:
                    if not entry.get("evaluation", {}).get("output_paths"):
                        server.scene.reset()
                        state.clear()
                        play.value = False
                        info.content = f"**{phase} / {key}: {entry['status']}**\n\nNo replay available."
                        continue
                    control = Path(entry["evaluation"]["output_paths"][0])
                    record = load_episode(control.with_suffix(".replay.npz"))
                    model = mujoco.MjModel.from_xml_path(record["scene_path"])
                    data = mujoco.MjData(model)
                    prepared = np.load(entry["prepared_path"], allow_pickle=True).item()
                    server.scene.reset()
                    scene = ViserMujocoScene(server, model, num_envs=1)
                    scene.camera_tracking_enabled = False
                    state.update(
                        sample=key,
                        phase=phase,
                        model=model,
                        data=data,
                        record=record,
                        scene=scene,
                        prepared=prepared,
                        center=np.asarray(prepared["obj_pose"][:3]) + [0, 0, 0.06],
                        duration=record["frames"][-1]["time"],
                    )
                    progress.value = 0
                    play_time = 0.0
                    play.value = False
                    camera_dirty.set()
                model, data, record = state["model"], state["data"], state["record"]
                if mode.value == "Target poses":
                    frame = record["frames"][0]
                    data.qpos[:] = frame["qpos"]
                    data.qvel[:] = 0
                    addresses = [model.joint("child-" + name).qposadr[0] for name in state["prepared"]["joint_names"]]
                    data.qpos[addresses] = record["targets"][stage.value + "_qpos"]
                    label = f"Target {stage.value} (static, not achieved state)"
                else:
                    target_time = progress.value / 100 * state["duration"]
                    index = int(np.argmin([abs(f["time"] - target_time) for f in record["frames"]]))
                    frame = record["frames"][index]
                    for field in ["qpos", "qvel", "ctrl", "act"]:
                        getattr(data, field)[:] = frame[field]
                    data.time = frame["time"]
                    label = f"Recorded {frame['segment']} · {frame['time']:.1f} s"
                mujoco.mj_forward(model, data)
                state["scene"].geom_groups_visible[3] = collision.value
                state["scene"].update_from_mjdata(data)
                init = entry["metrics"]["initialization"]
                info.content = (
                    f"**{phase} / {key} — {entry['status']}**\n\n{label}\n\n"
                    f"Object: `{Path(entry['scene_path']).parts[-3]}`\n\n"
                    f"Initial hand/object distance: {init['hand_object_distance'] * 1000:.2f} mm\n\n"
                    "Targets follow the experiment settings shown above; no extra squeeze. Replay includes lift."
                )
            if camera_dirty.is_set():
                camera_dirty.clear()
                for client in server.get_clients().values():
                    camera(client)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
