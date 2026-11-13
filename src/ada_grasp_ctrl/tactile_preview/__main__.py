"""Start the standalone read-only right Sharpa tactile web preview."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from typing import Any

from ada_grasp_ctrl.paths import resolve_from_root
from .data import select_force


class RosProcess:
    """Bounded stdout consumer for the same-host ROS Python 3.12 subprocess."""

    def __init__(self, python: str, joint_topic: str) -> None:
        self.lock = threading.Lock()
        self.packet: dict[str, Any] = {}
        self.error = "Waiting for ROS reader"
        self.received = 0.0
        env = os.environ.copy()
        source = str(resolve_from_root("src"))
        env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            [python, "-m", "ada_grasp_ctrl.tactile_preview.ros_reader", "--joint-topic", joint_topic],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self) -> None:
        """Drain snapshots continuously so browser slowness cannot block DDS callbacks."""
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                packet = json.loads(line)
                if packet.get("kind") != "snapshot":
                    continue
                with self.lock:
                    self.packet, self.received, self.error = packet, time.monotonic(), ""
            except (ValueError, AttributeError) as exc:
                with self.lock:
                    self.error = f"ROS reader output error: {exc}"
        with self.lock:
            self.error = "ROS reader exited; check terminal diagnostics and ROS sourcing"

    def snapshot(self) -> tuple[dict[str, Any], bool, str]:
        """Read the latest immutable packet and worker heartbeat."""
        with self.lock:
            return self.packet, self.process.poll() is None and time.monotonic() - self.received < 0.5, self.error

    def close(self) -> None:
        """Stop only the subscriber child created by this preview."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        self.thread.join(timeout=2)
        if self.process.stdout is not None:
            self.process.stdout.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Validate entrypoint options before opening any subscriptions or HTTP listener."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument(
        "--package-share",
        type=Path,
        default=Path("/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave"),
    )
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--joint-topic", default="/sharpa/right_hand/joint_states")
    parser.add_argument(
        "--geometry-config", type=Path, help="Explicit external NPY geometry profiles and frame evidence"
    )
    parser.add_argument(
        "--allow-unverified-geometry",
        action="store_true",
        help="Opt in to visibly UNVERIFIED diagnostic contact geometry; never calibration",
    )
    parser.add_argument(
        "--force-indices", default="0,1,2", help="Exploratory Fx,Fy,Fz channel indices; always labeled unverified"
    )
    parser.add_argument("--stale-after", type=float, default=0.5)
    parser.add_argument("--pose-skew", type=float, default=0.15)
    parser.add_argument(
        "--receipt-clock-synchronized",
        action="store_true",
        help="Assert driver and subscriber use synchronized ROS clocks; enable receipt-age rejection",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Show neutral model and NO DATA without creating ROS nodes"
    )
    parser.add_argument(
        "--run-seconds", type=float, default=0, help="Optional bounded smoke test; zero runs until Ctrl-C"
    )
    args = parser.parse_args(argv)
    try:
        args.force_indices = tuple(map(int, args.force_indices.split(",")))
        select_force((0,) * 6, args.force_indices)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.port <= 65535 or not 0 < args.stale_after < 60 or not 0 < args.pose_skew < 60:
        parser.error("Invalid port or freshness/skew threshold")
    if not 0 <= args.run_seconds < float("inf"):
        parser.error("run-seconds must be finite and nonnegative")
    args.package_share = args.package_share.expanduser().resolve(strict=True)
    args.urdf = (
        (args.urdf or args.package_share / "urdf/right_sharpa_wave_with_flange.urdf").expanduser().resolve(strict=True)
    )
    return args


def main(argv: list[str] | None = None) -> None:
    """Serve one viewer using existing Viser dependencies, and shut down cleanly."""
    args = parse_args(argv)
    import viser
    from .viewer import Preview, load_hand

    # Viser otherwise silently tries the next port; fail with an actionable error.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((args.host, args.port))
    hand = load_hand(args.urdf, args.package_share)
    server = viser.ViserServer(host=args.host, port=args.port, label="Right Sharpa tactile · read only")
    reader = None
    try:
        if server.get_port() != args.port:
            raise RuntimeError(f"Port {args.port} became occupied")
        preview = Preview(server, hand, args)
        if not args.offline:
            reader = RosProcess(args.ros_python, args.joint_topic)
        print(
            f"PREVIEW_URL=http://{args.host}:{args.port} "
            f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', 'default (0)')}",
            flush=True,
        )
        deadline = time.monotonic() + args.run_seconds if args.run_seconds else float("inf")
        while time.monotonic() < deadline:
            preview.update(*(reader.snapshot() if reader else ({}, False, "Offline mode; no ROS connection")))
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.close()
        server.stop()


if __name__ == "__main__":
    main()
