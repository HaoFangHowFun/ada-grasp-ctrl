"""One entry point across Ada, ROS and DexLearn environments, with a terminal execution gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import yaml

from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, load_candidates, read_json, verify_reference
from ada_grasp_ctrl.grasp_selection.cli import run as select_candidates, write_json
from ada_grasp_ctrl.grasp_selection.preview import load_entries
from ada_grasp_ctrl.paths import project_root, resolve_from_root
from .artifacts import contact_sheets, export_selection
from .plan import check_start
from .lift import LiftTargetError
from .approach import AboveObjectTargetError
from ada_grasp_ctrl.ros_execution.settings import validate_domain


class Processes:
    """Own only processes started by this invocation; isolate each Python environment."""

    def __init__(self, config: dict, root: Path) -> None:
        self.config, self.root = config, root
        self.children: list[subprocess.Popen] = []
        self.streams: list[Any] = []
        self.services: list[subprocess.Popen] = []

    def start(self, command: list[str], label: str, *, ros: bool = False, service: bool = False) -> subprocess.Popen:
        """Start a logged child with source-checkout imports and explicit ROS setup."""
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env["PYTHONPATH"] = str(project_root() / "src")
        env["ROS_DOMAIN_ID"] = str(self.config["ros_domain_id"])
        env["MUJOCO_GL"] = "egl"
        env["OMP_NUM_THREADS"] = "4"
        if ros:
            env["PATH"] = "/usr/bin:/bin:" + env.get("PATH", "")
            command = [
                "/bin/bash",
                "-c",
                'source "$1" && source "$2" && shift 2 && exec "$@"',
                "ada-ros",
                self.config["ros_setup"],
                self.config["workspace_setup"],
                *command,
            ]
        with (self.root / "subprocesses.jsonl").open("a") as record:
            record.write(
                json.dumps(
                    {
                        "label": label,
                        "command": command,
                        "ros": ros,
                        "ros_domain_id": env["ROS_DOMAIN_ID"],
                        "started_s": time.time(),
                    }
                )
                + "\n"
            )
        stream = (self.root / (label + ".log")).open("w")
        self.streams.append(stream)
        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(child)
        if service:
            self.services.append(child)
        return child

    def check_services(self) -> None:
        """Fail closed if a driver, receiver or viewer started by this run exits."""
        if any(child.poll() is not None for child in self.services):
            raise RuntimeError("An owned service exited; inspect this run's logs")

    def run(self, command: list[str], label: str, *, ros: bool = False, timeout: float = 900) -> None:
        """Run a bounded stage; never hide a failed child or continue to sending."""
        print(f"[{label}] running", flush=True)
        child = self.start(command, label, ros=ros)
        deadline = time.monotonic() + timeout
        while child.poll() is None:
            self.check_services()
            if time.monotonic() > deadline:
                raise TimeoutError(f"{label} timed out; inspect {self.root / (label + '.log')}")
            time.sleep(0.1)
        if child.returncode:
            raise RuntimeError(f"{label} exited {child.returncode}; inspect {self.root / (label + '.log')}")

    def close(self) -> None:
        """Cancel active sender first, then stop only this run's services."""
        for child in reversed(self.children):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
        for stream in self.streams:
            stream.close()


def wait_for_enter(processes: Processes, stream: Any = None) -> bool:
    """Only an explicitly received empty line approves; EOF and other input cancel."""
    stream = sys.stdin if stream is None else stream
    print("8089 預覽已就緒。按 Enter 執行目前顯示的完整流程；輸入 q 或 Ctrl+C 取消。", flush=True)
    while True:
        processes.check_services()
        if select.select([stream], [], [], 0.2)[0]:
            return stream.readline() in ("\n", "\r\n")


def perceive(group: dict, args: argparse.Namespace, config: dict, processes: Processes, root: Path) -> Path:
    """Capture or replay a cloud, apply calibration and run real checkpoint inference."""
    reference = load_candidates(resolve_from_root(group["candidates_path"]))
    if args.reuse_candidates:
        if args.mode != "replay":
            raise ValueError("--reuse-candidates is only valid for replay")
        return resolve_from_root(group["candidates_path"])
    ros_python = config["ros_python"]
    if args.mode == "live":
        snapshot = root / "capture"
        processes.run(
            [
                ros_python,
                "-m",
                "ada_grasp_ctrl.experiments.fixed_camera",
                "capture",
                "--config",
                str(resolve_from_root(config["camera_config"])),
                "--extrinsics",
                str(resolve_from_root(config["extrinsics"])),
                "--output-root",
                str(snapshot),
            ],
            "capture",
            ros=True,
            timeout=60,
        )
        plane = resolve_from_root(config["table_plane"])
        camera_config = resolve_from_root(config["camera_config"])
        roi, cluster = args.roi, args.cluster
    else:
        old_object = read_json(resolve_from_root(reference.payload["object_root"]) / "object.json")
        snapshot = resolve_from_root(old_object["source_snapshot"])
        if snapshot != resolve_from_root(group["snapshot"]):
            raise ValueError("Gallery capture differs from candidate provenance")
        plane = resolve_from_root(old_object["plane_path"])
        if file_hash(plane) != old_object["plane_sha256"]:
            raise ValueError("Recorded table plane changed")
        camera_config = root / "camera_config.yaml"
        camera_config.write_text(yaml.safe_dump(old_object["config"]))
        roi, cluster = old_object["roi"], old_object["cluster_id"]
    object_root, asset_root = root / "object", root / "asset"
    command = [
        ros_python,
        "-m",
        "ada_grasp_ctrl.experiments.fixed_camera",
        "object",
        "--config",
        str(camera_config),
        "--snapshot",
        str(snapshot),
        "--plane",
        str(plane),
        "--output-root",
        str(object_root),
        "--allow-unaccepted",
    ]
    if roi is not None:
        command += ["--roi", *map(str, roi)]
    if cluster is not None:
        command += ["--cluster", str(cluster)]
    processes.run(command, f"group{group['group']}_object", ros=True)
    processes.run(
        [
            ros_python,
            "-m",
            "ada_grasp_ctrl.experiments.fixed_camera_asset",
            "build",
            "--object",
            str(object_root),
            "--output-root",
            str(asset_root),
            "--name",
            f"group{group['group']}",
        ],
        f"group{group['group']}_asset",
        ros=True,
    )
    command = [
        config["dexlearn_python"],
        "-m",
        "ada_grasp_ctrl.experiments.fixed_camera_grasp",
        "--object",
        str(asset_root),
        "--dexlearn-root",
        config["dexlearn_root"],
        "--output-root",
        str(root / "inference"),
        "--samples",
        str(config["samples"]),
        "--seed",
        str(config["seed"]),
        "--device",
        config["device"],
        "--allow-unaccepted",
    ]
    for field in ("checkpoint", "training_config", "metadata"):
        command += ["--" + field.replace("_", "-"), str(verify_reference(reference.payload[field]))]
    processes.run(command, f"group{group['group']}_inference")
    return root / "inference/candidates.json"


def run(args: argparse.Namespace, config: dict, root: Path, report: dict, processes: Processes) -> None:
    """Prepare independent datasets, preview, then execute only the displayed selection."""
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema-1 pipeline config")
    validate_domain(config.get("environment"), config.get("ros_domain_id"))
    if args.start_mock and config["environment"] != "mock":
        raise ValueError("--start-mock cannot be combined with hardware mode")
    if args.reuse_candidates and args.mode != "replay":
        raise ValueError("--reuse-candidates requires --mode replay; live always captures and infers anew")
    if not args.export_only:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", 8089)) == 0:
                raise RuntimeError("Port 8089 is occupied; close the previous preview before starting this run")
    ros_python = config["ros_python"]
    if args.start_mock:
        processes.run([ros_python, "-m", "ada_grasp_ctrl.pipeline.ros_probe", "--empty-domain"], "probe", ros=True)
        processes.start(
            [
                "ros2",
                "launch",
                "bimanual_manipulation",
                "single_arm_hand.launch.py",
                "prefix:=right_",
                "crx_mock:=true",
                "sharpa_backend:=mock",
                "rviz:=false",
                "viewer:=false",
            ],
            "mock",
            ros=True,
            service=True,
        )
    model_report = resolve_from_root(config["model_report"])
    model = read_json(model_report)["model"]["urdf"]

    def snapshot(label: str) -> Path:
        processes.run(
            [
                ros_python,
                "-m",
                "ada_grasp_ctrl.ros_execution.snapshot",
                "--model-urdf",
                model,
                "--environment",
                config["environment"],
                "--output-root",
                str(root / label),
            ],
            label,
            ros=True,
            timeout=60,
        )
        return root / label / "robot_state.json"

    # Verify ROS/backend availability before expensive perception, then capture a fresh ranking state afterward.
    snapshot("initial_state")
    groups = (
        read_json(resolve_from_root(config["gallery"]))["groups"]
        if args.mode == "replay"
        else [
            {
                "group": "live",
                "label": "即時點雲",
                "candidates_path": str(resolve_from_root(config["inference_reference"])),
            }
        ]
    )
    if args.mode == "replay" and args.group != "all":
        groups = [g for g in groups if g["group"] == args.group]
    if not groups or len({g["group"] for g in groups}) != len(groups):
        raise ValueError("Missing or duplicate input groups")
    rows = []
    report["groups"] = rows
    for group in groups:
        group_id = group["group"]
        if group_id != "live" and not str(group_id).isdecimal():
            raise ValueError("Invalid group ID")
        destination = root / f"group{group_id}"
        destination.mkdir()
        candidate_path = perceive(group, args, config, processes, destination)
        state = snapshot(f"state_{group_id}")
        selection = destination / "selection"
        selection.mkdir()
        select_args = argparse.Namespace(
            candidates=str(candidate_path),
            state=str(state),
            model_report=str(model_report),
            replay=True,
            max_age_s=0.5,
            weights=[1.0] * 6,
            output_root=str(selection),
        )
        print(f"[group{group_id}] selecting candidates", flush=True)
        summary = select_candidates(select_args, selection)
        row = {"group": group_id, "label": group["label"], "runs": [str(selection)], **summary}
        one = destination / "selection_index.json"
        write_json(one, [row])
        entry = load_entries(one)[0]
        row["contact_sheets"] = contact_sheets(entry, destination / "images")
        if entry["keyframes"] is not None:
            try:
                row["packet_path"] = str(
                    export_selection(
                        entry,
                        state,
                        destination / "export",
                        lift_height_m=config.get("lift_height_m", 0.0),
                        complete_cycle=config.get("complete_cycle", False),
                        hold_s=config.get("hold_s", 5.0),
                        above_object_clearance_m=config.get("above_object_clearance_m", 0.0),
                        above_object_hand_factor=config.get("above_object_hand_factor", 1.0),
                        hand_tolerance=config.get("hand_tolerance"),
                        hand_arrival_required=config.get("hand_arrival_required", True),
                    )
                )
            except (LiftTargetError, AboveObjectTargetError) as error:
                row["execution_error"] = str(error)
                stage = "above_object_target" if isinstance(error, AboveObjectTargetError) else "lift_target"
                failure = {"group": group_id, "stage": stage, "reason": str(error)}
                report.setdefault("preparation_errors", []).append(failure)
                write_json(
                    destination
                    / ("above_object_failure.json" if stage == "above_object_target" else "lift_failure.json"),
                    failure,
                )
                with (root / "failures.jsonl").open("a") as stream:
                    stream.write(json.dumps(failure) + "\n")
                print(f"[group{group_id}] {error}", flush=True)
        rows.append(row)
        write_json(root / "integration_summary.json", rows)
        write_json(root / "run_report.json", report)
        print(
            f"[group{group_id}] {summary['feasible_candidates']}/{summary['candidates']} feasible; "
            f"selected {summary['selected_candidate_id']}",
            flush=True,
        )
    report["status"] = "prepared_with_errors" if report.get("preparation_errors") else "prepared"
    if args.export_only:
        return
    status = root / "viewer_state.json"
    processes.start(
        [
            sys.executable,
            "-m",
            "ada_grasp_ctrl.pipeline.viewer",
            "--summary",
            str(root / "integration_summary.json"),
            "--status",
            str(status),
        ]
        + ([] if args.preview_only else ["--enter-gate"]),
        "viewer",
        service=True,
    )
    deadline = time.monotonic() + 60
    while not status.exists():
        processes.check_services()
        if time.monotonic() > deadline:
            raise TimeoutError("Viewer did not become ready on port 8089")
        time.sleep(0.1)
    report["status"] = "preview_only" if args.preview_only else "awaiting_enter"
    write_json(root / "run_report.json", report)
    print("PREVIEW_URL=http://localhost:8089", flush=True)
    print(f"Execution environment: {config['environment']} | ROS domain: {config['ros_domain_id']}", flush=True)
    if args.preview_only:
        print("Preview-only mode; no execution gate. Ctrl+C closes this run.", flush=True)
        while True:
            processes.check_services()
            time.sleep(0.2)
    if not wait_for_enter(processes):
        report["status"] = "canceled_before_execution"
        return
    displayed = read_json(status)
    row = next(r for r in rows if r["group"] == displayed["group"])
    if not row.get("packet_path"):
        raise ValueError(row.get("execution_error", "The displayed dataset has no feasible candidate"))
    packet = load_packet(Path(row["packet_path"]))
    if displayed["selection_sha256"] != packet["source"]["references"][packet["source"]["selection_path"]]:
        raise ValueError("Displayed selection changed")
    actual = read_json(snapshot("before_execute"))
    check_start(actual["full_positions_rad"], packet["reviewed_plan"]["keyframes"][0])
    if read_json(status) != displayed:
        raise ValueError("Displayed dataset changed during confirmation; rerun before execution")
    processes.run([ros_python, "-m", "ada_grasp_ctrl.pipeline.ros_probe"], "receiver_probe", ros=True)
    if read_json(status) != displayed:
        raise ValueError("Displayed dataset changed before receiver startup; rerun before execution")
    receiver_config = root / "receiver_config.yaml"
    receiver_config.write_text(
        yaml.safe_dump(
            {
                "environment": config["environment"],
                "hand_tolerance": packet["reviewed_plan"]["receiver_settings"]["hand_tolerance"],
                "hand_arrival_required": packet["reviewed_plan"]["receiver_settings"].get(
                    "hand_arrival_required", True
                ),
            }
        )
    )
    processes.start(
        [
            ros_python,
            "-m",
            "ada_grasp_ctrl.ros_execution.bridge",
            "--config",
            str(receiver_config),
            "--targets",
            row["packet_path"],
            "--output-dir",
            str(root / "receiver"),
        ],
        "receiver",
        ros=True,
        service=True,
    )
    report.update(status="executing", approved_group=row["group"], approved_candidate=row["selected_candidate_id"])
    write_json(root / "run_report.json", report)
    processes.run(
        [
            ros_python,
            "-m",
            "ada_grasp_ctrl.experiments.right_ros_sender",
            "--targets",
            row["packet_path"],
            "--output-root",
            str(root / "sender"),
            "--execute",
            "--environment",
            config["environment"],
        ],
        "sender",
        ros=True,
        timeout=1000,
    )
    report.update(status="completed", execution=read_json(root / "sender/run_report.json"))


def main(argv: list[str] | None = None) -> int:
    """Run a fresh experimental pipeline; EOF never approves robot motion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/ada_grasp_ctrl/config/experiments/grasp_pipeline.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("replay", "live"), default="live")
    parser.add_argument("--environment", choices=("mock", "hardware"), help="Backend to verify; default from config")
    parser.add_argument("--ros-domain-id", type=int, help="Domain of the already-running camera and robot drivers")
    parser.add_argument("--group", default="01", help="Replay group ID or all; Enter executes only the displayed group")
    parser.add_argument(
        "--start-mock", action="store_true", help="Start and own right-only mock drivers on an unused domain"
    )
    parser.add_argument(
        "--reuse-candidates",
        action="store_true",
        help="Replay saved inference instead of rerunning raw-cloud processing",
    )
    parser.add_argument("--roi", nargs=4, type=int)
    parser.add_argument("--cluster", type=int)
    parser.add_argument("--lift-height-m", type=float, help="Post-selection world-Z lift; 0 disables it")
    parser.add_argument("--hand-tolerance-rad", type=float, help="Finger arrival tolerance; arm tolerance is unchanged")
    parser.add_argument(
        "--above-object-hand-factor",
        type=float,
        help="Finger extrapolation from grasp toward pregrasp; 1 preserves pregrasp",
    )
    parser.add_argument(
        "--above-object-clearance-m",
        type=float,
        help="Hand clearance above observed object before pregrasp; 0 disables it",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-only", action="store_true", help="Produce artifacts without viewer or execution")
    mode.add_argument("--preview-only", action="store_true", help="Keep 8089 open without accepting execution")
    args = parser.parse_args(argv)
    root = resolve_from_root(args.output_root, "output")
    processes = None
    created = False
    report = {
        "schema_version": 1,
        "status": "preflight",
        "hardware_execution_ready": False,
        "inference_rerun": not args.reuse_candidates,
        "mode": args.mode,
        "groups": [],
    }
    try:
        config_path = resolve_from_root(args.config)
        config = yaml.safe_load(config_path.read_text())
        if args.environment is not None:
            config["environment"] = args.environment
        if args.ros_domain_id is not None:
            config["ros_domain_id"] = args.ros_domain_id
        elif config.get("environment") == "hardware":
            config["ros_domain_id"] = validate_domain("hardware", os.environ.get("ROS_DOMAIN_ID"))
        report["environment"] = config.get("environment")
        if args.lift_height_m is not None:
            config["lift_height_m"] = args.lift_height_m
        if args.above_object_clearance_m is not None:
            config["above_object_clearance_m"] = args.above_object_clearance_m
        if args.above_object_hand_factor is not None:
            config["above_object_hand_factor"] = args.above_object_hand_factor
        if args.hand_tolerance_rad is not None:
            config["hand_tolerance"] = args.hand_tolerance_rad
        root.mkdir(parents=True, exist_ok=False)
        created = True
        write_json(
            root / "run_manifest.yaml",
            {
                "schema_version": 1,
                "mode": "grasp_pipeline",
                "arguments": vars(args),
                "config": config,
                "config_sha256": file_hash(config_path),
                "started_s": time.time(),
                "code_sha256": {
                    str(p): file_hash(p)
                    for folder in ("pipeline", "ros_execution")
                    for p in sorted((project_root() / "src/ada_grasp_ctrl" / folder).glob("*.py"))
                },
            },
        )
        processes = Processes(config, root)
        run(args, config, root, report, processes)
        return 1 if report.get("preparation_errors") else 0
    except (KeyboardInterrupt, EOFError):
        report["status"] = "interrupted"
        return 130
    except Exception as error:
        report.update(status="failed", reason=str(error))
        print(str(error), file=sys.stderr)
        return 2
    finally:
        if processes is not None:
            processes.close()
        if created:
            write_json(root / "run_report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
