"""Cancelable inference subprocesses with no ROS environment or transport path."""

import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.placement.session import write_json


class SimulationProcesses:
    """Own only model workers; cancellation cannot start a later inference branch."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.child = None

    def run(self, command: list[str], label: str, *, ros: bool = False, timeout: float = 900) -> None:
        """Run a bounded offline worker and preserve actual checkpoint failures."""
        if ros or any("ros_execution" in token or token in ("/bin/bash", "bash") for token in command):
            raise ValueError("Simulation cannot invoke ROS processes")
        environment = {k: v for k, v in os.environ.items() if not k.startswith(("ROS_", "AMENT_", "COLCON_"))}
        environment.pop("PYTHONHOME", None)
        environment["PYTHONPATH"] = str(project_root() / "src")
        environment.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MUJOCO_GL="egl")
        write_json(self.root / (label + "_invocation.json"), dict(command=command, ros=False))
        with (self.root / (label + ".log")).open("w") as log:
            with self.lock:
                if self.cancel.is_set():
                    raise RuntimeError("Attempt canceled")
                child = subprocess.Popen(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                self.child = child
            began = time.monotonic()
            while child.poll() is None:
                if self.cancel.wait(0.05) or time.monotonic() - began > timeout:
                    self.close()
                    raise RuntimeError("Attempt canceled or worker timed out")
            if child.returncode:
                detail = (self.root / (label + ".log")).read_text(errors="replace")[-2000:]
                raise RuntimeError(f"{label} exited {child.returncode}: {detail}")

    def close(self) -> None:
        """Cancel before termination so a completed worker cannot launch another branch."""
        self.cancel.set()
        with self.lock:
            child = self.child
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=3)
