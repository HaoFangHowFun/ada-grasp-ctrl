"""Deterministic synthetic telemetry on the existing 8089 dashboard; no ROS connection."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import yaml
from ada_grasp_ctrl.experiments.right_ros_contract import ARM_NAMES, HAND_NAMES
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.tactile_preview.data import FINGERS, Frame
from ada_grasp_ctrl.tactile_preview.geometry import GeometryCatalog
from .dashboard import serve_dashboard, read_json


class SyntheticReplay:
    """Deterministic actual-state fixtures with rebased same-host receipt clocks."""

    def __init__(self, state: dict, root: Path) -> None:
        self.state, self.root, self.index = state, root, 0
        self.geometry = GeometryCatalog.installed_candidate()
        self.contacts = {}
        for finger in FINGERS:
            profile = self.geometry.maps[finger]
            valid = np.argwhere(profile.mask)
            row, col = valid[len(valid) // 2]
            uv = np.linalg.solve(profile.uv_to_col_row[:, :2], np.array([col, row]) - profile.uv_to_col_row[:, 2])
            self.contacts[finger] = (uv, self.geometry.lookup(finger, uv, True))
        (root / "sender").mkdir()
        (root / "receiver/synthetic").mkdir(parents=True)
        self.phase_events = []
        self.payload_sizes = []

    def snapshot(self) -> tuple[dict, bool, str]:
        now, ros = time.monotonic(), time.time()
        q = np.asarray(self.state["full_positions_rad"]) + 0.002 * np.sin(self.index * 0.1 + np.arange(28))

        def joints(names, values):
            return dict(
                names=list(names),
                positions=values.tolist(),
                received=now,
                receipt_ros=ros,
                subscriber_ros=ros,
                error="",
            )

        packet = dict(
            kind="snapshot",
            arm=joints(ARM_NAMES, q[:6]),
            joints=joints(HAND_NAMES, q[6:]),
            model_valid=True,
            model_sha256=self.state["model_id"],
            world_T_right_base=self.state["world_T_right_base"],
            joint_sources=dict(arm=1, hand=1, description=1),
            worker_monotonic=now,
            errors={},
            endpoints={f: [dict(type="dual_sharpa_wave/msg/TactileFrame")] for f in FINGERS},
            frames={},
        )
        force = 4.0 + 0.1 * np.sin(self.index * 0.1)
        for finger, (uv, contact) in self.contacts.items():
            packet["frames"][finger] = asdict(
                Frame(
                    finger=finger,
                    status="VALID",
                    raw=tuple(-force * contact.n_E) + (0.0, 0.0, 0.0),
                    shape=(6,),
                    error="",
                    received=now,
                    receipt_ros=ros,
                    subscriber_ros=ros,
                    session_id="synthetic-dashboard-replay",
                    sequence=self.index + 1,
                    sdk_frame_id=self.index + 1,
                    sdk_frame_id_raw=str(self.index + 1),
                    sdk_timestamp=None,
                    sdk_timestamp_repr="synthetic",
                    sdk_timestamp_unit="unknown",
                    sdk_timestamp_epoch="unknown",
                    source_frame="synthetic fixture",
                    warnings=(),
                    contact_status="VALID",
                    contact_shape=(1, 3),
                    contact_row=tuple(uv) + (1.0,),
                    contact_count=1,
                )
            )
        phases = (
            "pregrasp",
            "grasp",
            "squeeze",
            "dynamic lift planning",
            "lift",
            "hold",
            "release",
            "return",
            "completed",
        )
        phase = phases[min(self.index // 10, len(phases) - 1)]
        if not self.phase_events or self.phase_events[-1]["phase"] != phase:
            self.phase_events.append(dict(sample=self.index, phase=phase))
        status = dict(
            stage="pipeline" if phase == "completed" else "sender",
            state="completed" if phase == "completed" else "running",
        )
        progress = dict(
            kind="lift_planning_started" if phase == "dynamic lift planning" else "goal_sent",
            phase="grasp" if phase == "squeeze" else phase,
        )
        for name, value in (("pipeline_status", status), ("sender/progress", progress)):
            temporary = self.root / (name + ".tmp")
            temporary.write_text(json.dumps(value))
            temporary.replace(self.root / (name + ".json"))
        if phase in ("grasp", "squeeze"):
            with (self.root / "receiver/synthetic/events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            time=now,
                            kind="native_complete" if self.index == 29 else "mpc_proposal",
                            diagnostics=dict(native_stage=2 if phase == "squeeze" else 1, current_sum_force=5 * force),
                        )
                    )
                    + "\n"
                )
        self.payload_sizes.append(len(json.dumps(packet)))
        self.index += 1
        return packet, True, "Synthetic replay; no ROS or robot connection"

    def close(self) -> None:
        (self.root / "replay_report.json").write_text(
            json.dumps(
                dict(
                    synthetic=True,
                    ros_connected=False,
                    motion_commands=0,
                    samples=self.index,
                    phases=self.phase_events,
                    mean_telemetry_json_bytes=float(np.mean(self.payload_sizes)) if self.payload_sizes else 0,
                ),
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/ada_grasp_ctrl/config/experiments/grasp_pipeline.yaml")
    parser.add_argument("--state", required=True, help="Recorded full robot_state.json; synthetic fixture center")
    parser.add_argument("--object-root", required=True, help="Recorded selected object directory; read once")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error("Replay seconds must be within (0, 60]")
    config = yaml.safe_load(resolve_from_root(args.config).read_text())
    config.update(config.get("active_hardware", {}))
    config["dashboard_replay"] = True
    root = resolve_from_root(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "grouplive").mkdir()
    source_object = resolve_from_root(args.object_root)
    (root / "grouplive/object").symlink_to(source_object, target_is_directory=True)
    source_asset = source_object.parent / "asset"
    if (source_asset / "asset.json").exists():
        (root / "grouplive/asset").symlink_to(source_asset, target_is_directory=True)
    config_path = root / "dashboard_config.json"
    config_path.write_text(json.dumps(config))
    replay = SyntheticReplay(read_json(resolve_from_root(args.state)), root)
    serve_dashboard(
        root / "pipeline_status.json", root / "viewer_state.json", config_path, replay=replay, duration=args.seconds
    )


if __name__ == "__main__":
    main()
