"""Operator entrypoint for a selected pipeline packet; read-only readiness by default."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.ros_execution.settings import bridge_config, validate_domain


def readiness(packet, args, output: Path) -> dict:
    """Read existing drivers and run one actuator-free proposal, never publish a target."""
    import rclpy
    from ada_grasp_ctrl.experiments.right_ros_contract import check_kinematics, kinematic_contract
    from ada_grasp_ctrl.ros_execution.runtime import RunLog, Runtime, ARM_COMMAND
    from .live_execution import LiveFeed, create_worker

    config = bridge_config(
        {k: v for k, v in packet["reviewed_plan"]["receiver_settings"].items() if k not in ("mode", "path_verified")}
    )
    config["operator_reviewed_joint_goal"] = True
    log = rt = worker = None
    report = dict(
        schema_version=1,
        status="NOT_READY",
        block_reasons=[],
        robot_command_publishers=0,
        synchronization="receipt_aligned_not_acquisition_synchronized",
        physical_active_verified=False,
        selected_candidate=packet["reviewed_plan"]["candidate_id"],
        environment=packet["environment"],
    )
    rclpy.init(args=[])
    try:
        log = RunLog(str(output / "diagnostics"), config, args)
        rt = Runtime(config, log)
        feed = LiveFeed(rt)
        rt.preflight(True)  # Inspects ownership/safety only; publisher creation is separate.
        check_kinematics(packet["kinematics"], kinematic_contract(rt.description))
        worker = create_worker(packet, args, output)
        rt.wait(lambda: len(feed.frames) == 5 and rt.fresh(), "five tactile streams and fresh joints")
        feedback, tactile = feed.sample()
        now = time.monotonic()
        worker.submit_tactile(now, feedback, feedback.positions_rad, tactile, "grasp")
        proposal = None
        while proposal is None:
            rt.spin(0.001)
            proposal = worker.poll()
        report.update(
            proposal=asdict(proposal),
            joint_names=list(feedback.joint_names),
            feedback_age_s=now - feedback.source_timestamp,
            command_ownership={t: rt.node.count_publishers(t) for t in (ARM_COMMAND, rt.hand_command)},
            solver_period_s=0.1,
            guarded_stream_period_s=0.01,
            maximum_proposal_delta_rad=max(abs(a - b) for a, b in zip(proposal.positions_rad, feedback.positions_rad))
            if proposal.positions_rad
            else None,
            safety=log.result.get("driver_interfaces"),
            q_cmd_source="measured_hold_probe_not_a_published_command",
            geometry_and_force_calibration="provisional; operator commissioning required",
        )
        if proposal.valid:
            report["status"] = "READY_FOR_OPERATOR_REVIEW"
        else:
            report["block_reasons"].append(proposal.diagnostics.get("reason", "Native proposal unavailable"))
    except Exception as error:
        report["block_reasons"].append(str(error))
    finally:
        if worker is not None:
            worker.close()
        if rt is not None:
            rt.node.destroy_node()
        if log is not None:
            log.finish()
        rclpy.try_shutdown()
        (output / "readiness.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="Fresh original pipeline export/targets.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--controller-mode", choices=("linear", "mpc_passive", "mpc_active"), default="linear")
    parser.add_argument("--execute", action="store_true", help="Run the reviewed pipeline; default is read-only")
    parser.add_argument("--hardware-enable", action="store_true", help="Explicit active-hardware opt-in")
    parser.add_argument("--mpc-python", default=str(resolve_from_root(".conda/bin/python")))
    parser.add_argument(
        "--mpc-model-report",
        default=str(
            resolve_from_root("output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json")
        ),
    )
    parser.add_argument(
        "--mpc-package-share", default="/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave"
    )
    args = parser.parse_args()
    args.targets = str(resolve_from_root(args.targets))
    packet = load_packet(Path(args.targets))
    validate_domain(packet["environment"], os.environ.get("ROS_DOMAIN_ID"))
    output = resolve_from_root(args.output_root, "output")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=1,
        arguments=vars(args),
        selected_reference=packet["source"],
        default_linear="unchanged successful receiver path",
        hardware_operator_confirmed=False,
    )
    (output / "run_manifest.yaml").write_text(json.dumps(manifest, indent=2))
    if not args.execute:
        report = readiness(packet, args, output)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "READY_FOR_OPERATOR_REVIEW" else 2
    if packet["environment"] == "hardware":
        if args.controller_mode == "mpc_active" and not args.hardware_enable:
            raise ValueError("Active hardware requires --hardware-enable")
        answer = input(
            f"Operator present, stop accessible, selected candidate {packet['reviewed_plan']['candidate_id']}. "
            f"Type {args.controller_mode} to execute this hardware run: "
        )
        if answer != args.controller_mode:
            raise ValueError("Hardware run not confirmed")
        manifest["hardware_operator_confirmed"] = True
        (output / "run_manifest.yaml").write_text(json.dumps(manifest, indent=2))
    # Retain the existing receiver probe: never run alongside another action receiver.
    subprocess.run([sys.executable, "-m", "ada_grasp_ctrl.pipeline.ros_probe"], check=True)
    config = {
        k: v for k, v in packet["reviewed_plan"]["receiver_settings"].items() if k not in ("mode", "path_verified")
    }
    config_path = output / "receiver.yaml"
    config_path.write_text(yaml.safe_dump(config))
    options = ["--controller-mode", args.controller_mode]
    receiver = [
        sys.executable,
        "-m",
        "ada_grasp_ctrl.ros_execution.bridge",
        "--targets",
        args.targets,
        "--config",
        str(config_path),
        "--output-dir",
        str(output / "receiver"),
        *options,
        "--dynamic-lift-plan",
        str(output / "sender/dynamic-lift-result.json"),
        "--mpc-python",
        args.mpc_python,
        "--mpc-model-report",
        args.mpc_model_report,
        "--mpc-package-share",
        args.mpc_package_share,
    ]
    if args.hardware_enable:
        receiver += ["--hardware-enable"]
    if manifest["hardware_operator_confirmed"]:
        receiver += ["--operator-confirmed"]
    with (output / "receiver.log").open("w") as log:
        process = subprocess.Popen(receiver, stdout=log, stderr=subprocess.STDOUT)
        try:
            command = [
                sys.executable,
                "-m",
                "ada_grasp_ctrl.experiments.right_ros_sender",
                "--targets",
                args.targets,
                "--output-root",
                str(output / "sender"),
                "--execute",
                "--environment",
                packet["environment"],
                *options,
                "--mpc-python",
                args.mpc_python,
            ]
            for key in ("arm_speed", "hand_speed", "arm_acceleration", "hand_acceleration"):
                command += ["--" + key.replace("_", "-"), str(config[key])]
            return subprocess.run(command, check=False).returncode
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
