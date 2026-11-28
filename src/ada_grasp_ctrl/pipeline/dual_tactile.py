"""Passive dual-hand tactile bridge for the original dual pipeline viewer."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np

from ada_grasp_ctrl.contact.force_display import FINGERS, SIDES, ForceSnapshot
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.tactile_preview.__main__ import RosProcess
from ada_grasp_ctrl.tactile_preview.adapter import TactileAdapter
from ada_grasp_ctrl.tactile_preview.geometry import GeometryCatalog
from ada_grasp_ctrl.tactile_preview.kinematics import load_hand
from ada_grasp_ctrl.tactile_preview.world_forces import world_forces
from .telemetry import full_feedback


def start_reader(config: dict, side: str) -> RosProcess:
    """Own only a subscriber process; reuse the deployment's existing hardware drivers."""
    if side not in SIDES:
        raise ValueError("Expected left or right")
    command = [
        "/bin/bash",
        "-c",
        'source "$1" && source "$2" && shift 2 && exec "$@"',
        "ada-dual-tactile",
        config["ros_setup"],
        config["workspace_setup"],
        config["ros_python"],
        "-m",
        "ada_grasp_ctrl.pipeline.telemetry",
        "--side",
        side,
        "--model-urdf",
        str(resolve_from_root(config["robot_snapshot"]) / "model.urdf"),
    ]
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    environment["PYTHONPATH"] = str(resolve_from_root("src"))
    environment["PATH"] = "/usr/bin:/bin:" + environment.get("PATH", "")
    environment["ROS_DOMAIN_ID"] = str(config["ros_domain_id"])
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return RosProcess(config["ros_python"], "unused", command=command, environment=environment)


def measured_dual(packets: dict, connected: dict, robot: object, now: float) -> dict:
    """Require fresh, coherent measured feedback for both complete 28-joint sides."""
    q, receipts = [], []
    for side in SIDES:
        if not connected.get(side):
            raise ValueError(f"{side}: ROS reader disconnected")
        packet = packets[side]
        q.extend(full_feedback(packet, now, side))
        receipts.extend(packet[group]["received"] for group in ("arm", "joints"))
    if max(receipts) - min(receipts) > 0.15:
        raise ValueError("Incoherent dual-arm/hand feedback receipts")
    robot.configuration(q)
    return dict(joint_names=list(robot.joint_names), positions_rad=q, received=min(receipts))


class DualTactile:
    """Evaluate existing adapters and world forces independently of planned playback."""

    def __init__(self, robot: object, adapters: dict, readers: dict, *, clock: Callable = time.monotonic) -> None:
        self.robot, self.adapters, self.readers, self.clock = robot, adapters, readers, clock

    @classmethod
    def start(cls, config: dict, robot: object) -> DualTactile:
        """Load official side models before starting owned passive ROS children."""
        package = resolve_from_root(config.get("mpc_package_share") or config["active_hardware"]["mpc_package_share"])
        geometry = GeometryCatalog.installed_candidate()
        adapters = {
            side: TactileAdapter(
                load_hand(package / f"urdf/{side}_sharpa_wave_with_flange.urdf", package, side=side, load_meshes=False),
                geometry,
                side=side,
                receipt_clock_synchronized=True,
            )
            for side in SIDES
        }
        readers = {}
        try:
            for side in SIDES:
                readers[side] = start_reader(config, side)
            return cls(robot, adapters, readers)
        except Exception:
            for reader in readers.values():
                reader.close()
            raise

    def sample(self) -> tuple[np.ndarray | None, ForceSnapshot, dict]:
        """Return measured pose, cleared-or-valid forces, and bounded diagnostic evidence."""
        now = self.clock()
        packets, connected, errors = {}, {}, {}
        diagnostics = {}
        for side in SIDES:
            packet, alive, error = self.readers[side].snapshot()
            packets[side], connected[side] = packet, alive
            if error:
                errors[side] = error
            snapshot = self.adapters[side].update(
                packet.get("frames", {}),
                packet.get("joints", {}),
                connected=alive,
                endpoints=packet.get("endpoints", {}),
                errors=packet.get("errors", {}),
                allow_unverified_geometry=True,
            )
            diagnostics[side] = {
                finger: dict(
                    status=row.status,
                    transport=row.transport_status,
                    detail=row.detail,
                    raw_f6=list(row.frame.raw) if row.frame is not None else None,
                    force_local=list(row.force_local) if row.force_local is not None else None,
                    resultant_presumed_n=(
                        float(np.linalg.norm(row.force_local))
                        if row.transport_status == "VALID" and row.force_local is not None
                        else None
                    ),
                    sequence=row.frame.sequence if row.frame is not None else None,
                    session_id=row.frame.session_id if row.frame is not None else None,
                    arrival_age_s=row.arrival_age,
                    pose_skew_s=row.pose_skew,
                )
                for finger, row in snapshot.fingers
            }
        try:
            measured = measured_dual(packets, connected, self.robot, now)
            q = np.asarray(measured["positions_rad"])
            pose_status = "LIVE measured 56 joints"
        except (ValueError, KeyError, TypeError) as error:
            q, measured, pose_status = None, {}, str(error)
        forces = world_forces(self.adapters, self.robot, measured, now=now)
        report = dict(
            schema_version=1,
            timestamp_monotonic=now,
            pose_status=pose_status,
            measured_positions_rad=q.tolist() if q is not None else None,
            connected=connected,
            errors=errors,
            fingers=diagnostics,
            forces=asdict(forces),
            units="presumed N; not calibrated",
            geometry="UNVERIFIED diagnostic",
            normal_load_status="Aggregate F6 cannot reconstruct simulation sum of compressive normal loads",
        )
        return q, forces, report

    def close(self) -> None:
        """Stop subscriber children without changing the user's hardware launch."""
        for reader in self.readers.values():
            reader.close()


def display_frame(
    planned: np.ndarray, previous: np.ndarray, measured: np.ndarray | None, forces: ForceSnapshot, *, live: bool
) -> tuple[np.ndarray, ForceSnapshot | None]:
    """Never draw current forces on a planned or stale pose; retain the last pose on dropout."""
    if not live:
        return planned, None
    if measured is None:
        return previous, None
    return measured, forces


def execution_display_locked(root: Path, locked: bool) -> bool:
    """Latch measured display when execution starts, independently of packet review.

    The runner's report can briefly be incomplete during a write. Retain the
    previous display policy in that case; this observation never grants motion.
    """
    if locked:
        return True
    try:
        report = json.loads((root / "run_report.json").read_text())
    except (OSError, ValueError):
        return locked
    return isinstance(report, dict) and report.get("execution_started") is True


def diagnostic_text(report: dict) -> str:
    """Show transport separately from contact availability, including numeric F6."""
    lines = [
        "**Read-only tactile · provisional N and geometry**",
        "Force directions and geometry are provisional. ΣFn cannot be recovered from aggregate F6.",
        "",
        f"**{report['pose_status']}**",
        "",
        "| Hand / finger | Transport / contact | Resultant (presumed N) | F6 (sensor frame) |",
        "|---|---|---:|---|",
    ]
    for side in SIDES:
        for finger in FINGERS:
            row = report["fingers"][side][finger]
            raw = row["raw_f6"]
            values = ", ".join(f"{x:.3f}" for x in raw) if raw is not None else "unavailable"
            magnitude = row["resultant_presumed_n"]
            magnitude_text = f"{magnitude:.3f}" if magnitude is not None else "unavailable"
            lines.append(f"| {side} / {finger} | {row['transport']} / {row['status']} | {magnitude_text} | {values} |")
    return "\n".join(lines)
