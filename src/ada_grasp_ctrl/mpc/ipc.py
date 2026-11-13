"""Bounded JSON-lines IPC: ROS system Python never imports native MPC libraries."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import select
import subprocess
import time

from .contracts import JointFeedback, MPCProposal


class MPCWorker:
    """One outstanding solve; never accumulate a ROS/solver queue of old targets."""

    def __init__(self, python: str, initialization: dict, log_path: Path) -> None:
        from ada_grasp_ctrl.paths import project_root

        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("LD_LIBRARY_PATH", None)
        env["PYTHONPATH"] = str(project_root() / "src")
        env["OMP_NUM_THREADS"] = "1"
        self.log = log_path.open("w")
        self.process = subprocess.Popen(
            [python, "-m", "ada_grasp_ctrl.mpc.worker"],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.pending: float | None = None
        self.process.stdin.write(json.dumps(initialization, allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 30)[0]:
            self.close()
            raise TimeoutError("MPC worker initialization timeout")
        reply = json.loads(self.process.stdout.readline())
        if reply != {"ready": True, "schema_version": 1}:
            self.close()
            raise ValueError(f"MPC worker initialization rejected: {reply}")

    def submit(
        self, now: float, feedback: JointFeedback, q_cmd: tuple, q_ref: tuple, observation: dict, stage: str
    ) -> None:
        """Send one versioned named-state/contact snapshot without native library objects."""
        if self.pending is not None:
            raise ValueError("MPC request backlog; prior solve is still pending")
        message = {
            "schema_version": 1,
            "now": now,
            "feedback": asdict(feedback),
            "q_cmd": q_cmd,
            "q_ref": q_ref,
            "observation": observation,
            "stage": stage,
        }
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()
        self.pending = time.monotonic()

    def submit_tactile(self, now: float, feedback: JointFeedback, q_cmd: tuple, tactile: dict, stage: str) -> None:
        """Send immutable normalized sensing to the existing native worker."""
        if self.pending is not None:
            raise ValueError("MPC request backlog; prior solve is still pending")
        message = dict(schema_version=1, now=now, feedback=asdict(feedback), q_cmd=q_cmd, tactile=tactile, stage=stage)
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()
        self.pending = time.monotonic()

    def poll(self) -> MPCProposal | None:
        """Return a completed proposal or reject a missed deadline/worker disconnect."""
        if self.process.poll() is not None:
            raise ValueError("MPC worker disconnected")
        if self.pending is None:
            return None
        if time.monotonic() - self.pending > 0.1:
            raise TimeoutError("MPC IPC/solve deadline missed")
        if not select.select([self.process.stdout], [], [], 0)[0]:
            return None
        reply = json.loads(self.process.stdout.readline())
        self.pending = None
        return MPCProposal(**reply)

    def close(self) -> None:
        """Close only the worker owned by this client."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for pipe in (self.process.stdin, self.process.stdout):
            pipe.close()
        self.log.close()
