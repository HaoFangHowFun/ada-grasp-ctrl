"""Inspect saved selected pregrasp/grasp/squeeze poses in Viser without stepping physics."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

from mjviser import ViserMujocoScene
import numpy as np
import viser

from ada_grasp_ctrl.experiments.grasp_pose_preview import GraspPosePreview
from ada_grasp_ctrl.paths import resolve_from_root
from preview_dual_arm_web import add_source_visuals


def serve(runs: dict[str, Path], port: int) -> None:
    """Show one saved grasp at a time, hiding collision shells and optional surrounding robot."""
    previews = {label: GraspPosePreview(root) for label, root in runs.items()}
    server = viser.ViserServer(host="127.0.0.1", port=port, label="薄盒 / 起司盒 · Grasp 姿態")
    if server.get_port() != port:
        server.stop()
        raise RuntimeError(f"Requested port {port} is occupied")
    selected = server.gui.add_dropdown("物體", options=list(previews), initial_value=next(iter(previews)))
    stage = server.gui.add_dropdown("姿態", options=["pregrasp", "grasp", "squeeze"], initial_value="grasp")
    context = server.gui.add_checkbox("顯示手臂與桌面", initial_value=False)
    center = server.gui.add_button("回到物體近景")
    info = server.gui.add_markdown("")
    server.gui.add_markdown(
        "滑鼠拖曳旋轉，滾輪縮放。顯示選中的目標姿態，物體保持推論時的位置；不執行模擬。碰撞殼已隱藏。"
    )
    dirty, recenter = threading.Event(), threading.Event()
    current = None

    def camera(client: Any) -> None:
        preview = previews[selected.value]
        target = preview.data.xpos[preview.object_id].copy() + [0, 0, 0.025]
        with client.atomic():
            client.camera.up_direction = np.array([0, 0, 1])
            client.camera.look_at = target
            client.camera.position = target + [0.24, -0.30, 0.20]

    server.on_client_connect(camera)
    selected.on_update(lambda event: (dirty.set(), recenter.set()))
    stage.on_update(lambda event: dirty.set())
    context.on_update(lambda event: dirty.set())
    center.on_click(lambda event: (dirty.set(), recenter.set()))
    dirty.set()
    recenter.set()
    try:
        while True:
            if not dirty.wait(0.1):
                continue
            dirty.clear()
            preview = previews[selected.value]
            preview.set_stage(stage.value)
            if current != selected.value:
                server.scene.reset()
                preview.model.geom_group[:] = preview.original_groups
                wrapper = SimpleNamespace(
                    robot=SimpleNamespace(directory=resolve_from_root(preview.config["scene"]["robot_snapshot"])),
                    model=preview.model,
                    data=preview.data,
                )
                colored, _ = add_source_visuals(server, wrapper)
                preview.assign_context_group()
                scene = ViserMujocoScene(server, preview.model, num_envs=1)
                scene.camera_tracking_enabled = False
                current = selected.value
            scene.geom_groups_visible[0] = False
            scene.geom_groups_visible[3] = False
            scene.geom_groups_visible[5] = False
            scene.geom_groups_visible[4] = context.value
            scene._sync_visibilities()
            scene.update_from_mjdata(preview.data)
            for name, handle in colored.items():
                body = preview.model.body(name).id
                handle.visible = context.value or body in preview.hand_ids
                handle.position = preview.data.xpos[body]
                handle.wxyz = preview.data.xquat[body]
            scene.request_update()
            candidate = preview.selected
            info.content = (
                f"**{selected.value} · {stage.value}**\n\n"
                f"Candidate: `{candidate['candidate_id']}` · effort: {candidate['effort']:.6f} rad²\n\n"
                "物體姿態固定；只切換手與手臂的目標關節角。"
            )
            target = preview.data.xpos[preview.object_id].copy() + [0, 0, 0.025]
            server.initial_camera.look_at = target
            server.initial_camera.position = target + [0.24, -0.30, 0.20]
            server.initial_camera.up = (0, 0, 1)
            if recenter.is_set():
                recenter.clear()
                for client in server.get_clients().values():
                    camera(client)
            print(
                f"POSE_READY object={selected.value} stage={stage.value} "
                f"candidate={candidate['candidate_id']} port={port}",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main() -> None:
    """Open two explicitly provided saved runs; input/output artifacts are never modified."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thin-box-run", required=True, type=Path)
    parser.add_argument("--cheese-box-run", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    serve({"薄盒": args.thin_box_run, "起司盒": args.cheese_box_run}, args.port)


if __name__ == "__main__":
    main()
