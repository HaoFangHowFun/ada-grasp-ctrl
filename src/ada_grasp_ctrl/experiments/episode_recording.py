"""Passive action-boundary state recording and controller-free replay."""

from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np


class EpisodeRecorder:
    """Keep actual simulator states separately from the legacy control schema."""

    def __init__(self, simulation: Any, targets: dict[str, np.ndarray]):
        self.simulation = simulation
        self.targets = targets
        self.metadata: dict[str, Any] = {}
        self.frames: list[dict[str, Any]] = []

    def capture(self, segment: str, reference: Any = None, command: Any = None) -> None:
        """Copy a state without forwarding or advancing the simulator."""
        data = self.simulation.data
        self.frames.append(
            {
                "segment": segment,
                "time": float(data.time),
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "ctrl": data.ctrl.copy(),
                "act": data.act.copy(),
                "object_pose": self.simulation.get_obj_pose().copy(),
                "reference": None if reference is None else np.asarray(reference).copy(),
                "command": None if command is None else np.asarray(command).copy(),
            }
        )

    def save(self, control_path: Path, status: str) -> None:
        """Save assembled scene XML and a trusted-local NumPy sidecar."""
        scene_path = control_path.with_suffix(".scene.xml")
        scene_path.write_text(self.simulation.spec.to_xml())
        # Use NPZ so native control-stat tree discovery never mistakes a
        # sidecar for a legacy control record.
        np.savez_compressed(
            control_path.with_suffix(".replay.npz"),
            record={
                "schema_version": 1,
                "sampling": "initial and post-action boundaries (10 Hz), terminal",
                "scene_path": str(scene_path.resolve()),
                "status": status,
                "targets": self.targets,
                "metadata": self.metadata,
                "joint_names": [self.simulation.model.joint(i).name for i in range(self.simulation.model.njnt)],
                "frames": self.frames,
            },
        )


def load_episode(path: Path) -> dict[str, Any]:
    """Load a trusted-local replay sidecar with an explicit format boundary."""
    with np.load(path, allow_pickle=True) as archive:
        return archive["record"].item()


def configure_scene_viewer(session: Any) -> None:
    """Use the same Shadow view for live episodes and replay; preserve collisions."""
    if session is not None and session.backend == "mujoco":
        session.set_native_camera(lookat=np.array([0, 0, 0.15]), distance=0.8, azimuth=135, elevation=-25)
        session.set_native_geom_group(3, False)


def hold_viewer(session: Any) -> None:
    """Hold terminal state until the window closes or the user interrupts."""
    if session is None:
        return
    try:
        while session.backend != "mujoco" or session.native_is_running():
            session.sync()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


def replay_episode(path: Path, backend: str = "mujoco", speed: float = 1.0, hold: bool = True) -> None:
    """Restore recorded states, including lift, without executing a controller."""
    from ada_grasp_ctrl.utils.viewer_session import DebugViewerSession

    if speed <= 0:
        raise ValueError("Replay speed must be positive")
    record = load_episode(path)
    model = mujoco.MjModel.from_xml_path(record["scene_path"])
    data = mujoco.MjData(model)
    session = DebugViewerSession(backend=backend)
    try:
        session.attach(model, data)
        configure_scene_viewer(session)
        previous = record["frames"][0]["time"]
        for frame in record["frames"]:
            time.sleep(max(0, frame["time"] - previous) / speed)
            for key in ("qpos", "qvel", "ctrl", "act"):
                getattr(data, key)[:] = frame[key]
            data.time = frame["time"]
            mujoco.mj_forward(model, data)
            session.sync()
            previous = frame["time"]
        if hold:
            hold_viewer(session)
    finally:
        session.close()
