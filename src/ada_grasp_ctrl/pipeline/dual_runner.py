"""Capture any current object, select a hand automatically, preview and execute one cycle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import time

import yaml

from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json, file_hash
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.grasp_selection.multi_arm import select_candidates, enabled_set
from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.ros_execution.settings import validate_domain, DEFAULTS
from .runner import Processes, perceive, wait_for_enter
from .dual_inference import infer_live_sides
from .dual_scene import live_scene
from .dual_artifacts import export_dual_selection
from .plan import check_start


def configuration(path: str) -> dict:
    """Resolve shared motion defaults and the dual-stack deployment settings."""
    extra = yaml.safe_load(resolve_from_root(path).read_text())
    base = yaml.safe_load(resolve_from_root(extra.pop("base_config")).read_text())
    config = {**base, **extra}
    if "inference_model" in config:
        config.pop("inference_reference", None)
    return config


def run(args: argparse.Namespace, config: dict, root: Path, processes: Processes, report: dict) -> None:
    """Run one fresh live capture, without any saved-candidate fallback."""
    if not args.export_only:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", 8089)) == 0:
                raise RuntimeError("Port 8089 is occupied; close the previous preview")
    model = resolve_from_root(config["robot_snapshot"]) / "model.urdf"

    def snapshot(label: str) -> Path:
        processes.run(
            [
                config["ros_python"],
                "-m",
                "ada_grasp_ctrl.ros_execution.dual_snapshot",
                "--model-urdf",
                str(model),
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

    snapshot("initial_state")
    # Retain the established object ROI/cluster interaction and calibration path.
    perceive_args = argparse.Namespace(mode="live", reuse_candidates=False, roi=args.roi, cluster=args.cluster)
    group = dict(group="live")
    obj = perceive(group, perceive_args, config, processes, root, infer=False)
    rows = infer_live_sides(obj, config, processes, root)
    state_path = snapshot("state")
    state = read_json(state_path)
    robot, sim = live_scene(
        resolve_from_root(config["robot_snapshot"]), state, read_json(resolve_from_root(config["table_plane"]))
    )
    print("Selecting across enabled arms: " + ", ".join(config["enabled_arms"]), flush=True)
    selection = select_candidates(robot, sim, rows, config["enabled_arms"], state["full_positions_rad"])
    selection["geometry_note"] = (
        "Measured table and both robots; partial object cloud is not a verified collision solid"
    )
    write_json(root / "selection.json", selection)
    if selection["selected"] is None:
        report.update(status="no_feasible_candidate")
        return
    selected = selection["selected"]
    report.update(
        selected_side=selected["selected_side"], candidate_id=selected["candidate_id"], effort=selected["effort"]
    )
    packet_path = export_dual_selection(
        robot, sim, selection, state_path, obj, config, root / "export", root / "candidates.json"
    )
    packet = load_packet(packet_path)
    report.update(status="exported", packet_path=str(packet_path))
    write_json(root / "run_report.json", report)
    print(
        f"Selected {report['selected_side']} / {report['candidate_id']} (cost {report['effort']:.6f} rad²)", flush=True
    )
    if args.export_only:
        return
    processes.start(
        [sys.executable, "-m", "ada_grasp_ctrl.pipeline.dual_viewer", "--run", str(root)], "viewer", service=True
    )
    deadline = time.monotonic() + 60
    status_path = root / "viewer_state.json"
    while not status_path.exists():
        processes.check_services()
        if time.monotonic() > deadline:
            raise TimeoutError("Viewer failed to start on 8089")
        time.sleep(0.1)
    print("PREVIEW_URL=http://localhost:8089", flush=True)
    print(f"Environment: {config['environment']} | ROS domain: {config['ros_domain_id']}", flush=True)
    if args.preview_only:
        report["status"] = "preview_only"
        write_json(root / "run_report.json", report)
        while True:
            processes.check_services()
            time.sleep(0.2)
    report["status"] = "awaiting_enter"
    write_json(root / "run_report.json", report)
    if not wait_for_enter(processes):
        report["status"] = "canceled_before_execution"
        return
    displayed = read_json(status_path)
    if displayed != dict(side=packet["side"], candidate_id=packet["sample_id"], packet_sha256=file_hash(packet_path)):
        raise ValueError("Preview differs from selected packet")
    fresh = read_json(snapshot("before_execute"))
    for side in ("left", "right"):
        check_start(fresh["sides"][side]["positions_rad"], state["sides"][side]["positions_rad"])
    if load_packet(packet_path) != packet:
        raise ValueError("Reviewed packet changed")
    processes.run(
        [config["ros_python"], "-m", "ada_grasp_ctrl.pipeline.ros_probe", "--side", packet["side"]],
        "receiver_probe",
        ros=True,
    )
    settings = packet["reviewed_plan"]["receiver_settings"]
    receiver_config = root / "receiver_config.yaml"
    receiver_config.write_text(
        yaml.safe_dump({k: v for k, v in settings.items() if k in DEFAULTS or k in ("side", "active_arms")})
    )
    processes.start(
        [
            config["ros_python"],
            "-m",
            "ada_grasp_ctrl.ros_execution.bridge",
            "--config",
            str(receiver_config),
            "--targets",
            str(packet_path),
            "--output-dir",
            str(root / "receiver"),
        ],
        "receiver",
        ros=True,
        service=True,
    )
    report.update(status="executing", execution_started=True)
    write_json(root / "run_report.json", report)
    processes.run(
        [
            config["ros_python"],
            "-m",
            "ada_grasp_ctrl.experiments.right_ros_sender",
            "--targets",
            str(packet_path),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--environment", choices=("hardware", "mock"))
    parser.add_argument("--ros-domain-id", type=int)
    parser.add_argument("--arms", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--roi", nargs=4, type=int)
    parser.add_argument("--cluster", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-only", action="store_true")
    mode.add_argument("--preview-only", action="store_true")
    args = parser.parse_args(argv)
    root, processes, created = resolve_from_root(args.output_root, "output"), None, False
    report = dict(schema_version=1, status="preflight", object_lifted=None, object_held=None)
    try:
        config = configuration(args.config)
        if args.environment:
            config["environment"] = args.environment
        config["ros_domain_id"] = validate_domain(
            config["environment"], args.ros_domain_id if args.ros_domain_id is not None else os.getenv("ROS_DOMAIN_ID")
        )
        if args.arms != "auto":
            config["enabled_arms"] = [args.arms]
        enabled_set(config["enabled_arms"])
        root.mkdir(parents=True, exist_ok=False)
        created = True
        (root / "effective_config.yaml").write_text(yaml.safe_dump(config))
        (root / "run_manifest.yaml").write_text(yaml.safe_dump(dict(arguments=vars(args), config=config, mode="live")))
        processes = Processes(config, root)
        run(args, config, root, processes, report)
        return 1 if report["status"] == "no_feasible_candidate" else 0
    except KeyboardInterrupt:
        report["status"] = "canceled"
        return 130
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr)
        return 1 if report.get("execution_started") else 2
    finally:
        if processes:
            processes.close()
        if created:
            write_json(root / "run_report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
