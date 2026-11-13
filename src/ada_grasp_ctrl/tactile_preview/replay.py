"""Receipt-clock MCAP playback in the existing 8089 hand preview; no ROS runtime."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from ada_grasp_ctrl.experiments.right_ros_contract import ARM_NAMES, HAND_NAMES
from ada_grasp_ctrl.mpc.mcap_replay import ordered
from ada_grasp_ctrl.paths import project_root, resolve_from_root
from .adapter import TactileAdapter
from .data import FINGERS
from .force_viewer import ForcePreview, contact_diagnostic
from .geometry import GeometryCatalog
from .kinematics import load_hand

ARM_TOPIC = "/crx5ia/right/joint_states"
HAND_TOPIC = "/sharpa/right_hand/joint_states"


class ReceiptReplay:
    """Index the existing decoder output by delivery time, using latest-past samples."""

    def __init__(self, decoded: Path) -> None:
        self.rows: dict[str, list[dict]] = {}
        first, last = None, None
        count = 0
        with decoded.open() as stream:
            for line in stream:
                row = json.loads(line)
                stamp = row["bag_ns"]
                if last is not None and stamp < last:
                    raise ValueError("Decoded bag receipt order regressed")
                first = stamp if first is None else first
                last = stamp
                count += 1
                topic = row["topic"]
                if topic not in (ARM_TOPIC, HAND_TOPIC) and "frame" not in row:
                    continue
                rows = self.rows.setdefault(topic, [])
                issue = ""
                if rows:
                    previous = rows[-1]
                    if "frame" in row:
                        a, b = previous["frame"], row["frame"]
                        if (
                            b["session_id"] != a["session_id"]
                            or b["sequence"] <= a["sequence"]
                            or (
                                b["sdk_frame_id"] is not None
                                and a["sdk_frame_id"] is not None
                                and b["sdk_frame_id"] <= a["sdk_frame_id"]
                            )
                        ):
                            issue = "source session/sequence/SDK regression"
                    elif row["source_ros_s"] <= previous["source_ros_s"]:
                        issue = "joint source timestamp regression"
                rows.append(dict(row, source_issue=issue))
        if first is None or last <= first:
            raise ValueError("Empty or zero-duration recording")
        self.origin_ns, self.end_ns, self.message_count = first, last, count
        self.duration = (last - first) * 1e-9
        self.times = {topic: [row["bag_ns"] for row in rows] for topic, rows in self.rows.items()}

    def sample(self, offset_s: float) -> tuple[float, dict]:
        """Seeking never alters source timestamps; pausing freezes the replay clock."""
        if not math.isfinite(offset_s) or not 0 <= offset_s <= self.duration:
            raise ValueError("Replay position outside recording")
        stamp = self.origin_ns + int(offset_s * 1e9)
        now = stamp * 1e-9
        packet: dict[str, Any] = dict(frames={}, joints={}, arm={}, source_issues={})
        for topic, rows in self.rows.items():
            index = bisect_right(self.times[topic], stamp) - 1
            if index < 0:
                continue
            row = rows[index]
            if "frame" in row:
                finger = row["frame"]["finger"]
                packet["frames"][finger] = row["frame"]
                packet["source_issues"][finger] = row["source_issue"]
            else:
                error = row["source_issue"]
                names = HAND_NAMES if topic == HAND_TOPIC else ARM_NAMES
                try:
                    positions = ordered(row, names)
                except ValueError as exc:
                    error, positions = str(exc), ()
                target = "joints" if topic == HAND_TOPIC else "arm"
                packet[target] = dict(
                    names=list(names),
                    positions=list(positions),
                    received=row["bag_ns"] * 1e-9,
                    receipt_ros=row["source_ros_s"],
                    subscriber_ros=row["bag_ns"] * 1e-9,
                    error=error,
                )
        return now, packet


def audit(replay: ReceiptReplay, hand: Any, geometry: GeometryCatalog, mu: float, output: Path) -> dict:
    """Report force diagnostics at 10 Hz across the recording, without a controller."""
    now = 0.0
    adapter = TactileAdapter(
        hand, geometry, stale_after=0.2, pose_skew=0.15, receipt_clock_synchronized=True, clock=lambda: now
    )
    counts = {finger: Counter() for finger in FINGERS}
    values: dict[str, list[dict]] = {finger: [] for finger in FINGERS}
    examples = {}
    ticks = int(replay.duration / 0.1) + 1
    for index in range(ticks):
        offset = index * 0.1
        now, packet = replay.sample(offset)
        snapshot = adapter.update(packet["frames"], packet["joints"], allow_unverified_geometry=True)
        for finger, result in snapshot.fingers:
            row = contact_diagnostic(result, mu, packet["source_issues"].get(finger, ""))
            key = row.get("cone", row["status"])
            counts[finger][key] += 1
            if row["status"] == "VALID":
                values[finger].append(row)
                examples.setdefault(finger + ":" + key, dict(receipt_offset_s=offset, diagnostic=row))
    summary = dict(
        schema_version=1,
        sampling="10 Hz latest-past receipt-time samples; whole bag",
        messages=replay.message_count,
        duration_s=replay.duration,
        ticks=ticks,
        mu=mu,
        frame="right_hand_flange",
        publishers=0,
        ros_context=False,
        fingers={},
        examples=examples,
    )
    for finger in FINGERS:
        rows = values[finger]
        summary["fingers"][finger] = dict(counts=dict(counts[finger]))
        for field in ("fn_n", "ft_n", "angle_deg", "mu_required"):
            data = [r[field] for r in rows if r[field] is not None]
            summary["fingers"][finger][field] = (
                dict(zip(("min", "median", "max"), np.percentile(data, [0, 50, 100]).tolist())) if data else None
            )
    (output / "run_report.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main() -> None:
    """Decode, inspect and serve one port owner; the only child process reads MCAP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", default="/home/hao-fang/ws_fanuc/output/physical_mpc_observe_20261112_194701")
    parser.add_argument("--decoded", help="Reuse existing mcap_decode JSONL instead of decoding again")
    parser.add_argument("--output-root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--start-s", type=float, default=0.0, help="Initial bag receipt offset, seconds")
    parser.add_argument("--run-seconds", type=float, default=0, help="Optional bounded viewer smoke test")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--package-share", default="/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave"
    )
    parser.add_argument("--ros-setup", default="/home/hao-fang/ws_fanuc/install/setup.bash")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not math.isfinite(args.run_seconds) or args.run_seconds < 0:
        parser.error("Invalid port or run-seconds")
    if not args.audit_only:
        # The existing Viser server otherwise silently chooses a different port.
        with socket.socket() as probe:
            probe.bind((args.host, args.port))
    output = resolve_from_root(
        args.output_root or ("tactile_force_replay_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")),
        "output",
    )
    output.mkdir(parents=True, exist_ok=False)
    decoded = resolve_from_root(args.decoded) if args.decoded else output / "decoded.jsonl"
    if not args.decoded:
        env = os.environ.copy()
        env.update(ROS_DOMAIN_ID="199", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
        shell = (
            'source /opt/ros/jazzy/setup.bash; source "$1"; export PYTHONPATH="$2:$PYTHONPATH"; '
            'exec /usr/bin/python3 -m ada_grasp_ctrl.mpc.mcap_decode --bag "$3" --output "$4"'
        )
        subprocess.run(
            [
                "bash",
                "-c",
                shell,
                "offline-decode",
                args.ros_setup,
                str(project_root() / "src"),
                str(resolve_from_root(args.bag)),
                str(decoded),
            ],
            env=env,
            check=True,
        )
    replay = ReceiptReplay(decoded)
    replay.sample(args.start_s)
    # The native right-full factory inherits control.friction_cone_mu unchanged.
    # Do not confuse task.miu_coef (simulated surfaces) with controller mu.
    config = resolve_from_root("src/ada_grasp_ctrl/config/task/control_eval.yaml")
    mu = float(yaml.safe_load(config.read_text())["control"]["friction_cone_mu"])
    package = resolve_from_root(args.package_share)
    hand = load_hand(package / "urdf/right_sharpa_wave_with_flange.urdf", package, load_meshes=not args.audit_only)
    geometry = GeometryCatalog.installed_candidate()
    manifest = dict(
        schema_version=1,
        arguments=vars(args),
        decoded=str(decoded),
        receipt_clock="bag receipt ordering; source acquisition clock remains unknown",
        friction_source=str(config),
        friction_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        mu=mu,
        force_frame="right_hand_flange",
        full_crx_mesh=False,
        ros_initialized=False,
        robot_command_publishers=0,
    )
    (output / "run_manifest.yaml").write_text(json.dumps(manifest, indent=2))
    audit(replay, hand, geometry, mu, output)
    print(f"REPORT={output / 'run_report.json'}", flush=True)
    if args.audit_only:
        return
    import viser

    preview_args = SimpleNamespace(
        geometry_config=None,
        stale_after=0.2,
        pose_skew=0.15,
        receipt_clock_synchronized=True,
        allow_unverified_geometry=True,
        force_indices=(0, 1, 2),
        force_replay=True,
    )
    server = viser.ViserServer(host=args.host, port=args.port, label="Recorded contact forces · read only")
    try:
        if server.get_port() != args.port:
            raise RuntimeError(f"Port {args.port} became occupied; refusing another port")
        with server.gui.add_folder("Recorded playback"):
            play = server.gui.add_checkbox("Play", initial_value=False)
            server.gui.add_dropdown("Playback speed", options=("1×",), initial_value="1×", disabled=True)
            position = server.gui.add_slider(
                "Replay receipt seconds", min=0, max=replay.duration, step=0.01, initial_value=args.start_s
            )
            stamp_label = server.gui.add_markdown("")
            arm_label = server.gui.add_markdown("")
        preview = ForcePreview(server, hand, preview_args, mu, geometry)
        now = replay.origin_ns * 1e-9
        preview.adapter.clock = lambda: now
        print(f"PREVIEW_URL=http://{args.host}:{args.port} (paused; receipt-time replay)", flush=True)
        started = previous = time.monotonic()
        saved = 0.0
        while not args.run_seconds or time.monotonic() - started < args.run_seconds:
            wall = time.monotonic()
            offset = min(replay.duration, position.value + (wall - previous if play.value else 0))
            previous = wall
            position.value = offset
            if offset >= replay.duration:
                play.value = False
            now, packet = replay.sample(offset)
            state = preview.update(packet, True)
            stamp_label.content = (
                f"**{'Playing' if play.value else 'Paused'} · 1× · receipt +{offset:.3f} s**\n\n"
                f"ROS bag receipt time: `{now:.6f}` s. Not sensor acquisition time."
            )
            arm = packet["arm"]
            arm_fresh = bool(arm) and not arm["error"] and 0 <= now - arm["received"] <= 0.2
            arm_values = (
                ", ".join(f"{name}={value:.4f}" for name, value in zip(arm.get("names", ()), arm.get("positions", ())))
                if arm_fresh
                else "UNAVAILABLE"
            )
            arm_label.content = "CRX feedback (rad; hand view fixed at flange):\n\n`" + arm_values + "`"
            if wall - saved >= 0.25:
                state.update(replay_offset_s=offset, playing=play.value, speed=1, port=server.get_port())
                temp = output / "viewer_state.tmp"
                temp.write_text(json.dumps(state, allow_nan=False))
                temp.replace(output / "viewer_state.json")
                saved = wall
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
