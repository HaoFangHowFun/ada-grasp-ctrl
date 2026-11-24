"""Placement controls attached to the existing 8089 main viewport."""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np

from .session import PlacementSession, write_json


class PlacementPanel:
    """One point-cloud handle and cached candidates; no capture or execution on selection."""

    def __init__(self, server: Any, session: PlacementSession, infer: Callable[[], list[dict]] | None) -> None:
        self.session, self.infer = session, infer
        self.lock = threading.RLock()
        with np.load(session.root / "snapshot.npz", allow_pickle=False) as data:
            environment = data["environment_points_world_m"]
            # Hide captured object pixels in the background to avoid a stale duplicate.
            mask = np.ones(len(environment), dtype=bool)
            mask[data["object_source_indices"]] = False
            environment = environment[mask]
        stride = max(1, int(np.ceil(len(environment) / 40000)))
        server.scene.add_point_cloud(
            "/environment", points=environment[::stride], colors=(155, 155, 155), point_size=0.002
        )
        self.cloud = server.scene.add_point_cloud(
            "/object", points=session.points, colors=(80, 200, 80), point_size=0.003, precision="float32"
        )
        center = session.center
        server.initial_camera.position = tuple(center + [0.55, -0.55, 0.4])
        server.initial_camera.look_at = tuple(center)
        server.initial_camera.up = (0, 0, 1)
        server.gui.add_markdown(
            "**Placement Selection**\n\nOrientation preview from the captured partial cloud. "
            "P00 preserves the original orientation. Stability, contact height and collision clearance are unverified."
        )
        self.run_button = server.gui.add_button("Run UOP-Net once")
        self.run_button.disabled = infer is None or session.inference_status != "not_run"
        self.choice = server.gui.add_dropdown(
            "Placement candidate", options=self.options(), initial_value=self.options()[0]
        )
        self.details = server.gui.add_markdown("")
        self.confirm_button = server.gui.add_button("Confirm Placement")
        self.notice = server.gui.add_markdown("")
        self.choice.on_update(self.switch)
        self.run_button.on_click(self.run)
        self.confirm_button.on_click(self.confirm)
        self.refresh()

    def options(self) -> list[str]:
        """Keep P00 first and expose source/instance beside the stable candidate ID."""
        return [
            f"{c.candidate_id} — Original Pose"
            if c.source == "original"
            else f"{c.candidate_id} — UOP-Net instance {c.instance_id}"
            for c in self.session.candidates
        ]

    def refresh(self) -> None:
        """Describe the selection separately from the last confirmed candidate."""
        c = self.session.selected
        text = (
            f"**Selected: {c.candidate_id}** · source: {c.source}\n\n"
            f"**Confirmed: {self.session.confirmed_id or 'none'}**\n\n"
            f"UOP-Net: {self.session.inference_status}"
        )
        if c.source == "uop_net":
            text += (
                f"\n\nInstance {c.instance_id} · inliers {c.inlier_count} ({c.inlier_ratio:.1%})"
                f"\n\nSupport normal (world): {np.round(c.support_normal, 4).tolist()}"
            )
        if self.session.inference_error:
            text += f"\n\nInference failed; P00 remains available. {self.session.inference_error} (see uop.log)"
        self.details.content = text

    def switch(self, _: object = None) -> None:
        """Update only the object geometry; the environment and model remain untouched."""
        with self.lock:
            self.cloud.points = self.session.select(self.choice.value.split(" — ")[0])
            self.refresh()

    def run(self, _: object = None) -> None:
        """Disable the one-shot action before dispatching the existing UOP environment."""
        with self.lock:
            if self.infer is None or self.session.inference_status != "not_run":
                return
            self.run_button.disabled = True
            self.confirm_button.disabled = True
            self.notice.content = "Running UOP-Net on the saved snapshot…"
            self.session.infer_once(self.infer)
            self.choice.options = self.options()
            self.confirm_button.disabled = False
            self.notice.content = "Candidates cached. Select an orientation, then Confirm Placement."
            self.refresh()

    def confirm(self, _: object = None) -> None:
        """Persist the selected orientation and stop at the phase boundary."""
        with self.lock:
            try:
                # Browser updates may be queued on another callback thread.
                self.cloud.points = self.session.select(self.choice.value.split(" — ")[0])
                path = self.session.confirm()
                self.notice.content = (
                    f"Saved **{self.session.confirmed_id}**: `{path}`\n\nPlacement selection complete."
                )
            except (OSError, ValueError) as error:
                self.notice.content = f"Could not save placement: {error}"
            self.refresh()


def serve(session: PlacementSession, server_factory: Callable, infer: Callable | None = None) -> None:
    """Use the same server factory as Dual Live; confirmation never invokes its execution path."""
    server = server_factory()
    try:
        PlacementPanel(server, session, infer)
        write_json(session.root / "viewer_state.json", dict(mode="placement_only", port=8089))
        print("PREVIEW_URL=http://localhost:8089", flush=True)
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def serve_saved(root: Path, server_factory: Callable) -> None:
    """Reopen the captured scene and cached orientations with no model inference."""
    serve(PlacementSession.load(root), server_factory)
