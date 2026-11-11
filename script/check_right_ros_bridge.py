"""Audit Ada sender/receiver transport using owned bimanual mock drivers only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from ada_grasp_ctrl.experiments.right_ros_contract import ACTION, JOINT_NAMES
from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.paths import resolve_external_root


def main() -> int:
    """Require an unused domain 88, then validate idleness, three poses and cancellation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "88":
        raise ValueError("This mock audit requires ROS_DOMAIN_ID=88")
    root = resolve_external_root("output", args.output_root)
    path = resolve_external_root("data", args.targets)
    packet = load_packet(path)
    root.mkdir(parents=True, exist_ok=False)
    (root / "run_manifest.yaml").write_text(
        json.dumps(
            {"mode": "owned_mock_bridge_audit", "domain": 88, "targets": str(path), "output_root": str(root)}, indent=2
        )
        + "\n"
    )
    rclpy.init()
    node = rclpy.create_node("ada_bridge_audit")
    client = ActionClient(node, FollowJointTrajectory, ACTION)
    counts = {"arm": 0, "hand": 0, "left_arm": 0, "left_hand": 0}
    wrong_names = []

    def receive(key: str, message: Any) -> None:
        counts[key] += 1
        if key in ("arm", "hand") and any(not n.startswith("right_") for n in message.name):
            wrong_names.append(message.name)

    for key, cls, topic in (
        ("arm", JointState, "/crx5ia/joint_targets"),
        ("hand", JointState, "/sharpa/right_hand/joint_command"),
        ("left_arm", Float64MultiArray, "/crx5ia/left/forward_position_controller/commands"),
        ("left_hand", JointState, "/sharpa/left_hand/joint_command"),
    ):
        node.create_subscription(cls, topic, lambda m, k=key: receive(k, m), qos_profile_sensor_data)

    def spin_until(predicate: Any, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.monotonic() > deadline:
                raise TimeoutError("Audit timed out")

    discovery = time.monotonic() + 1
    spin_until(lambda: time.monotonic() > discovery, 2)
    if node.count_publishers("/crx5ia/right/joint_states") or client.server_is_ready():
        raise RuntimeError("Domain 88 is already occupied; refusing to start duplicate drivers/bridge")
    processes = []
    outputs = []
    report: dict[str, Any] = {"status": "running", "domain": 88}

    def start(command: list[str], label: str) -> subprocess.Popen:
        output = (root / (label + ".log")).open("w")
        outputs.append(output)
        environment = os.environ.copy()
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=environment
        )
        processes.append(process)
        return process

    try:
        start(
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
        )
        start(
            [
                sys.executable,
                "-m",
                "ada_grasp_ctrl.ros_execution.bridge",
                "--output-dir",
                str(root / "receiver"),
            ],
            "bridge",
        )
        spin_until(lambda: client.server_is_ready() and node.count_publishers("/crx5ia/right/joint_states") > 0, 30)
        idle = time.monotonic() + 2
        spin_until(lambda: time.monotonic() > idle, 3)
        assert not any(counts.values()), counts
        assert node.count_publishers("/crx5ia/joint_targets") == 0
        assert node.count_publishers("/sharpa/right_hand/joint_command") == 0
        report["idle_command_counts"] = counts.copy()
        bad_names = ["left_J1"] + list(JOINT_NAMES[1:])
        bad_goal = FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(
                joint_names=bad_names,
                points=[JointTrajectoryPoint(positions=[0.0] * 28, time_from_start=Duration(sec=2))],
            )
        )
        bad = client.send_goal_async(bad_goal)
        spin_until(bad.done, 5)
        assert not bad.result().accepted
        report["invalid_names_rejected"] = True
        command = [
            sys.executable,
            "-m",
            "ada_grasp_ctrl.experiments.right_ros_sender",
            "--targets",
            str(path),
            "--execute",
        ]
        sender = start(command + ["--output-root", str(root / "sender")], "sender")
        spin_until(lambda: sender.poll() is not None, 180)
        sender_report = json.loads((root / "sender/run_report.json").read_text())
        assert sender.returncode == 0, sender_report
        assert sender_report["task_completed"], sender_report
        assert [r["phase"] for r in sender_report["phases"]] == ["pregrasp", "grasp", "squeeze"]
        sent = [
            e
            for e in map(json.loads, (root / "sender/events.jsonl").read_text().splitlines())
            if e["kind"] == "goal_sent"
        ]
        assert len(sent) == 3
        report["planned_duration_seconds"] = {event["phase"]: event["duration"] for event in sent}
        for actual, expected in zip(sent, packet["targets"]):
            assert actual["positions"] == expected["positions"]
        report["completed_sender"] = sender_report
        report["exact_source_targets_preserved"] = True
        report["complete_command_counts"] = counts.copy()
        baseline = counts["arm"]
        sender = start(command + ["--output-root", str(root / "cancel_sender")], "cancel_sender")
        spin_until(lambda: counts["arm"] >= baseline + 30 or sender.poll() is not None, 40)
        assert sender.poll() is None, (root / "cancel_sender.log").read_text()
        sender.send_signal(signal.SIGINT)
        spin_until(lambda: sender.poll() is not None, 10)
        canceled = json.loads((root / "cancel_sender/run_report.json").read_text())
        assert sender.returncode == 130 and canceled["cancel_result_received"], canceled
        assert not canceled["phases"], canceled
        report["cancel_sender"] = canceled
        spin_until(lambda: len(list((root / "receiver").glob("*/result.json"))) == 4, 5)
        outcomes = [json.loads(p.read_text()) for p in sorted((root / "receiver").glob("*/result.json"))]
        assert [result["final_state"] for result in outcomes] == ["DONE", "DONE", "DONE", "ABORTED"], outcomes
        assert outcomes[-1]["stop"]["controller_deactivated"], outcomes[-1]
        report["receiver_states"] = [result["final_state"] for result in outcomes]
        report["cancel_controller_deactivated"] = True
        assert not wrong_names and counts["left_arm"] == counts["left_hand"] == 0
        report.update(status="passed", final_counts=counts.copy())
    except BaseException as error:
        report.update(status="failed", reason=str(error))
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
        for output in outputs:
            output.close()
        (root / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
        client.destroy()
        node.destroy_node()
        rclpy.try_shutdown()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
