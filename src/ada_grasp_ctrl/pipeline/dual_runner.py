"""Capture any current object, select a hand automatically, preview and execute one cycle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import shutil
import sys
import time

import yaml

from ada_grasp_ctrl.calibration import DEFAULT_CALIBRATION, calibrated_snapshot
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.grasp_selection.multi_arm import select_candidates, enabled_set
from .dual_packet import load_execution_packet as load_packet, preview_identity
from ada_grasp_ctrl.ros_execution.settings import validate_domain, DEFAULTS
from .runner import Processes, perceive, wait_for_enter
from .dual_inference import infer_live_sides
from .dual_scene import live_scene
from .dual_artifacts import export_dual_selection
from .dual_selection import selector_settings, select_live_hierarchical
from .plan import check_start
from .lift import inward_offset_limit
from .grasp_models import configure_model


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
    timings = report.setdefault("timings", {})
    began = time.monotonic()
    if config.get("multi_camera"):
        observation_root = root / "multi_camera"
        command = [
            config["ros_python"],
            "-m",
            "ada_grasp_ctrl.experiments.multi_camera",
            "observe",
            "--config",
            str(resolve_from_root(config["multi_camera_config"])),
            "--camera-config",
            str(resolve_from_root(config["camera_config"])),
            "--calibration",
            str(resolve_from_root(config.get("extrinsics") or DEFAULT_CALIBRATION)),
            "--plane",
            str(resolve_from_root(config["table_plane"])),
            "--output-root",
            str(observation_root),
        ]
        if args.roi is not None:
            command += ["--roi", *map(str, args.roi)]
        if args.cluster is not None:
            command += ["--cluster", str(args.cluster)]
        processes.run(command, "multi_camera_observation", ros=True)
        obj = observation_root / "object"
        report["multi_camera"] = read_json(observation_root / "run_report.json")
    else:
        obj = perceive(group, perceive_args, config, processes, root, infer=False)
    timings["capture_s"] = time.monotonic() - began
    began = time.monotonic()
    rows = infer_live_sides(obj, config, processes, root)
    timings["inference_s"] = time.monotonic() - began
    native_diagnostics = None
    if config.get("grasp_model") == "native_multimode":
        native_diagnostics = read_json(root / "candidates.json")
        report.update(
            {
                k: native_diagnostics[k]
                for k in (
                    "checkpoint",
                    "availability_scores",
                    "ranked_available_modes",
                    "eligible_modes",
                    "unsupported_modes",
                )
            }
        )
        timings["native_inference"] = native_diagnostics["timings"]
    else:
        reference = config.get("inference_model") or read_json(resolve_from_root(config["inference_reference"]))
        report["checkpoint"] = reference["checkpoint"]
    state_path = snapshot("state")
    state = read_json(state_path)
    robot, sim = live_scene(
        resolve_from_root(config["robot_snapshot"]), state, read_json(resolve_from_root(config["table_plane"]))
    )
    print("Selecting across enabled arms: " + ", ".join(config["enabled_arms"]), flush=True)
    began = time.monotonic()
    if args.selector == "hierarchical":
        options = dict(native_diagnostics=native_diagnostics) if native_diagnostics is not None else {}
        if config.get("grasp_mode", "auto") != "auto":
            options["grasp_mode"] = config["grasp_mode"]
        selection, batch, audit = select_live_hierarchical(
            robot,
            sim,
            rows,
            config["enabled_arms"],
            state["full_positions_rad"],
            config["hierarchical_selector"],
            **options,
        )
        write_json(root / "canonical_candidates.json", batch)
        write_json(root / "hierarchical_selection.json", audit)
        if audit["plan"] is not None:
            write_json(root / "grasp_plan.json", audit["plan"])
        report.update(selector="hierarchical", selection_status=audit["status"], selected_mode=audit["selected_mode"])
    else:
        selection = select_candidates(robot, sim, rows, config["enabled_arms"], state["full_positions_rad"])
    timings["selection_s"] = time.monotonic() - began
    selection["geometry_note"] = (
        "Measured table and both robots; partial object cloud is not a verified collision solid"
    )
    write_json(root / "selection.json", selection)
    if selection["selected"] is None:
        report.update(status="no_feasible_candidate")
        if args.selector == "hierarchical":
            print("NO_FEASIBLE_GRASP: see hierarchical_selection.json for the mode trace", flush=True)
        return
    selected = selection["selected"]
    report.update(
        selected_side=selected["selected_side"], candidate_id=selected["candidate_id"], effort=selected["effort"]
    )
    report.update(
        selected_source_mode=selected["candidate"]["provenance"].get("source_model_mode", "right_full"),
        execution_mode=selected["candidate"].get("execution_mode", selected["selected_side"] + "_full"),
        source_model_mode=selected["candidate"]["provenance"].get("source_model_mode", "right_full"),
        required_arms=selected["required_arms"],
    )
    began = time.monotonic()
    packet_path = export_dual_selection(
        robot, sim, selection, state_path, obj, config, root / "export", root / "candidates.json"
    )
    packet = load_packet(packet_path)
    timings["export_s"] = time.monotonic() - began
    report.update(status="exported", packet_path=str(packet_path))
    if "schedule" in packet:
        report.update(shared_timing=packet["schedule"], physical_execution_verified=False)
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
    if displayed != preview_identity(packet, packet_path):
        raise ValueError("Preview differs from selected packet")
    fresh = read_json(snapshot("before_execute"))
    for side in ("left", "right"):
        check_start(fresh["sides"][side]["positions_rad"], state["sides"][side]["positions_rad"])
    if load_packet(packet_path) != packet:
        raise ValueError("Reviewed packet changed")
    if len(selected["required_arms"]) == 2:
        module = "shared_hardware" if config["environment"] == "hardware" else "shared_executor"
        command = [
            config["ros_python"],
            "-m",
            "ada_grasp_ctrl.ros_execution." + module,
            "--targets",
            str(packet_path),
            "--output-root",
            str(root / "shared_executor"),
            "--ros-domain-id",
            str(config["ros_domain_id"]),
        ]
        if config["environment"] == "hardware":
            command += ["--reviewed-sha256", displayed["packet_sha256"]]
        report.update(status="executing", execution_started=True)
        write_json(root / "run_report.json", report)
        processes.run(command, "shared_executor", ros=True, timeout=3000)
        report.update(status="completed", execution=read_json(root / "shared_executor/run_report.json"))
        return
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
    parser.add_argument("--environment", choices=("hardware", "mock", "simulation"))
    parser.add_argument("--ros-domain-id", type=int)
    parser.add_argument("--arms", choices=("auto", "left", "right", "both_only"), default="auto")
    parser.add_argument("--selector", choices=("legacy", "hierarchical"), default="legacy")
    parser.add_argument("--grasp-model", choices=("right_full", "native_multimode"), default="right_full")
    parser.add_argument("--grasp-mode", choices=("auto", "both_only"), default="auto")
    parser.add_argument("--roi", nargs=4, type=int)
    parser.add_argument("--cluster", type=int)
    parser.add_argument(
        "--multi-camera", action="store_true", help="Opt-in stationary D415 + two D405 observation (hardware)"
    )
    parser.add_argument("--multi-camera-config", help="Override multi-camera capture/fusion settings")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-only", action="store_true")
    mode.add_argument("--preview-only", action="store_true")
    mode.add_argument("--placement-only", action="store_true", help="Select placement in 8089; no grasp or execution")
    mode.add_argument("--placement-aware", action="store_true", help="Simulation-only grasp/place endpoint IK preview")
    parser.add_argument(
        "--placement-execute", action="store_true", help="Enable reviewed simulation grasp-to-place execution"
    )
    parser.add_argument("--placement-planner", choices=("rrt_connect",), help="Opt-in sampled goal-based placement")
    parser.add_argument("--object-root", help="Placement only: reuse an explicitly selected captured object")
    parser.add_argument("--uop-root", help="Existing UOP-Net checkout (or UOP_NET_ROOT)")
    parser.add_argument("--uop-python", help="Existing UOP Python (defaults to checkout/.venv/bin/python)")
    args = parser.parse_args(argv)
    if args.arms == "both_only":
        args.arms, args.grasp_mode = "auto", "both_only"
    root, processes, created = resolve_from_root(args.output_root, "output"), None, False
    report = dict(schema_version=1, status="preflight", object_lifted=None, object_held=None)
    try:
        config = configuration(args.config)
        if args.multi_camera_config and not args.multi_camera:
            raise ValueError("--multi-camera-config requires --multi-camera")
        if args.multi_camera:
            if (args.environment or config["environment"]) != "hardware":
                raise ValueError("--multi-camera supports hardware only; simulation/mock remain unchanged")
            if args.placement_only:
                raise ValueError("--multi-camera --placement-only is not supported")
            config["multi_camera"] = True
            config["multi_camera_config"] = (
                args.multi_camera_config or "src/ada_grasp_ctrl/config/experiments/multi_camera.yaml"
            )
        if args.placement_planner and not (args.placement_execute and args.placement_aware):
            raise ValueError(
                "--placement-planner requires --placement-aware --placement-execute --environment simulation"
            )
        if args.placement_execute and not args.placement_aware:
            raise ValueError("--placement-execute requires --placement-aware --environment simulation")
        if args.placement_aware:
            if (args.environment or config["environment"]) != "simulation":
                raise ValueError("--placement-aware requires --environment simulation")
            if args.grasp_mode == "both_only" and args.placement_planner != "rrt_connect":
                raise ValueError("Paired placement requires --placement-planner rrt_connect")
        if (args.environment or config["environment"]) == "simulation":
            from .simulation_ui import run_simulation

            if args.placement_planner:
                config.setdefault("simulation", {})["placement_planner"] = args.placement_planner

            if args.ros_domain_id is not None or args.roi is not None or args.cluster is not None or args.object_root:
                raise ValueError("Simulation does not accept ROS domain, camera ROI/cluster, or captured --object-root")
            if not (args.placement_only or args.placement_aware) and (args.uop_root or args.uop_python):
                raise ValueError("Placement options require --placement-only or --placement-aware")
            if args.arms != "auto":
                config["enabled_arms"] = [args.arms]
            enabled_set(config["enabled_arms"])
            config["grasp_mode"] = args.grasp_mode
            if not args.placement_only:
                configure_model(config, args.grasp_model, args.selector)
                if args.grasp_mode == "both_only" and (
                    args.grasp_model != "native_multimode" or set(config["enabled_arms"]) != {"left", "right"}
                ):
                    raise ValueError("--grasp-mode both_only requires native_multimode and both enabled arms")
                inward_offset_limit(config.get("endpoint_inward_offset_m", 0.0))
                if args.selector == "hierarchical" and args.grasp_model == "right_full":
                    selector_settings(config.get("hierarchical_selector"))
            return run_simulation(args, config, root)
        if args.placement_only:
            from ada_grasp_ctrl.placement.workflow import run_placement

            return run_placement(args, config, root)
        if args.object_root or args.uop_root or args.uop_python:
            raise ValueError("Placement options require --placement-only")
        configure_model(config, args.grasp_model, args.selector)
        report["grasp_model"] = args.grasp_model
        if args.environment:
            config["environment"] = args.environment
        config["ros_domain_id"] = validate_domain(
            config["environment"], args.ros_domain_id if args.ros_domain_id is not None else os.getenv("ROS_DOMAIN_ID")
        )
        if args.arms != "auto":
            config["enabled_arms"] = [args.arms]
        enabled_set(config["enabled_arms"])
        config["grasp_mode"] = args.grasp_mode
        report["grasp_mode"] = args.grasp_mode
        if args.grasp_mode == "both_only" and (
            args.grasp_model != "native_multimode" or set(config["enabled_arms"]) != {"left", "right"}
        ):
            raise ValueError("--grasp-mode both_only requires native_multimode and both enabled arms")
        inward_offset_limit(config.get("endpoint_inward_offset_m", 0.0))
        if args.selector == "hierarchical" and args.grasp_model == "right_full":
            selector_settings(config.get("hierarchical_selector"))
        root.mkdir(parents=True, exist_ok=False)
        created = True
        if args.multi_camera:
            from ada_grasp_ctrl.experiments.multi_camera import load_config as load_multi_config
            from ada_grasp_ctrl.experiments.multi_camera_calibration import load_cameras

            calibration_copy = root / "calibration.json"
            shutil.copyfile(resolve_from_root(config.get("extrinsics") or DEFAULT_CALIBRATION), calibration_copy)
            load_cameras(calibration_copy, load_multi_config(resolve_from_root(config["multi_camera_config"])))
            config["extrinsics"] = str(calibration_copy)
        config["robot_snapshot"] = str(
            calibrated_snapshot(
                config["robot_snapshot"], root / "robot_snapshot", config.get("extrinsics") or DEFAULT_CALIBRATION
            )
        )
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
