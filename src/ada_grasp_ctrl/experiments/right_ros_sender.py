"""Ada-owned ROS action client for the reviewed grasp, lift and return sequence.

Run this module under sourced ROS Jazzy system Python. It imports no MuJoCo,
Torch, or Ada controller runtime. JSON bridges the Python 3.10/3.12 environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

from ada_grasp_ctrl.paths import resolve_external_root
from .right_ros_contract import ACTION, ARM_NAMES, HAND_NAMES, JOINT_NAMES
from .right_ros_contract import check_kinematics, kinematic_contract, validate_packet


def digest(path: Path) -> str:
    """Hash a recorded input without loading pickled scientific data in ROS Python."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_packet(path: Path) -> dict[str, Any]:
    """Verify the current export manifest and source artifacts before transport."""
    manifest = json.loads((path.parent / "run_manifest.yaml").read_text())
    if manifest["mode"] != "ada_ros_target_export" or digest(path) != manifest["targets_sha256"]:
        raise ValueError("Export packet/manifest mismatch")
    packet = json.loads(path.read_text())
    validate_packet(packet)
    if packet["source"] != manifest["source"] or packet["sample_id"] != manifest["sample_id"]:
        raise ValueError("Export provenance mismatch")
    if packet["source"].get("kind") == "grasp_selection":
        from ada_grasp_ctrl.pipeline.plan import validate_plan

        source = packet["source"]
        for name in ("model_path", "selection_path", "snapshot_path"):
            if source[name] not in source["references"]:
                raise ValueError("Missing selection source reference: " + name)
        for name, expected in source["references"].items():
            if digest(Path(name)) != expected:
                raise ValueError("Selection source changed since preview: " + name)
        if digest(Path(source["plan_path"])) != source["plan_sha256"]:
            raise ValueError("Reviewed plan changed")
        plan = json.loads(Path(source["plan_path"]).read_text())
        validate_plan(plan)
        if plan != packet["reviewed_plan"]:
            raise ValueError("Packet differs from reviewed plan")
        selected = json.loads(Path(source["selection_path"]).read_text())
        snapshot = json.loads(Path(source["snapshot_path"]).read_text())
        if selected["selected_candidate_id"] != plan["candidate_id"]:
            raise ValueError("Reviewed candidate identity differs")
        if snapshot["full_positions_rad"] != plan["keyframes"][0]:
            raise ValueError("Reviewed start differs from measured snapshot")
        if packet["environment"] != plan["environment"] or snapshot.get("environment", "mock") != plan["environment"]:
            raise ValueError("Snapshot, packet and reviewed environment differ")
        if [target["phase"] for target in packet["targets"]] != plan["phases"]:
            raise ValueError("Packet phase sequence differs from preview")
        for index, target in enumerate(packet["targets"]):
            if target["phase"] == "above_object":
                original = plan["above_object"]["positions"]
                if target["flange_target_wxyz"] != plan["above_object"]["target_flange_wxyz"]:
                    raise ValueError("Above-object pose differs from preview")
            elif target["phase"] == "lift":
                original = plan["lift"]["positions"]
                if target["flange_target_wxyz"] != plan["lift"]["target_flange_wxyz"]:
                    raise ValueError("Lift pose differs from preview")
            elif target["phase"] in ("hold", "release", "return"):
                original = plan["keyframes"][index + 1]
                if target["flange_target_wxyz"] != plan["cycle"]["flange_targets_wxyz"][target["phase"]]:
                    raise ValueError("Cycle pose differs from preview")
            else:
                original = selected["selected"]["stage_checks"][target["phase"]]["q_rad"]
            if target["positions"] != original or target["positions"] != plan["keyframes"][index + 1]:
                raise ValueError("Target differs from selected/reviewed joints")
            if target["duration_s"] != plan["durations_s"][index]:
                raise ValueError("Target duration differs from preview")
        return packet
    for key in ("preview", "prepared", "model"):
        field = "preview_report" if key == "preview" else key + "_path"
        if digest(Path(packet["source"][field])) != packet["source"][key + "_sha256"]:
            raise ValueError(f"Source {key} changed since export")
    return packet


def motion_duration(start: list[float], target: list[float], args: argparse.Namespace) -> float:
    """Calculate a conservative duration from measured state; never replace targets."""
    times = [args.minimum_duration]
    for sl, speed, acceleration in (
        (slice(0, 6), args.arm_speed, args.arm_acceleration),
        (slice(6, 28), args.hand_speed, args.hand_acceleration),
    ):
        delta = max(abs(a - b) for a, b in zip(start[sl], target[sl]))
        times.extend([1.875 * delta / speed, math.sqrt(5.774 * delta / acceleration)])
    return max(times) * 1.05 + 0.1  # Margin for fresh-state sampling and duration rounding.


def execute(packet: dict[str, Any], args: argparse.Namespace, root: Path, report: dict[str, Any]) -> None:
    """Send the declared exported phases and wait for each actual ROS result.

    Raises:
        RuntimeError: If feedback/model/control status or an action result is invalid.
        TimeoutError: If discovery or an action does not complete within its bound.
    """
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from control_msgs.action import FollowJointTrajectory
    from action_msgs.msg import GoalStatus
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration

    from ada_grasp_ctrl.ros_execution.settings import validate_domain

    if args.environment != packet["environment"]:
        raise ValueError("Sender environment differs from reviewed packet")
    validate_domain(args.environment, os.environ.get("ROS_DOMAIN_ID"))
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("ada_right_grasp_sender")
    client = ActionClient(node, FollowJointTrajectory, ACTION)
    states: dict[str, tuple[list[float], float]] = {}
    stamps: dict[str, float] = {}
    description: list[str] = []
    faults: list[str] = []
    active = None
    pending_goal = None
    events = (root / "events.jsonl").open("w", buffering=1)
    old_handlers = {}
    interrupt_requested = False

    def event(kind: str, **data: Any) -> None:
        events.write(json.dumps({"time": time.monotonic(), "kind": kind, **data}, allow_nan=False) + "\n")

    def receive(group: str, expected: tuple[str, ...], msg: Any) -> None:
        if (
            len(msg.name) != len(msg.position)
            or len(set(msg.name)) != len(msg.name)
            or set(msg.name) != set(expected)
            or not all(math.isfinite(v) for v in msg.position)
        ):
            faults.append(f"{group}: malformed feedback")
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = node.get_clock().now().nanoseconds * 1e-9 - stamp
        if stamp <= 0 or abs(age) > args.feedback_timeout:
            event("feedback_rejected", group=group, stamp=stamp, age_seconds=age, timeout=args.feedback_timeout)
            faults.append(f"{group}: stale/invalid timestamp")
            return
        if stamp <= stamps.get(group, 0):
            return
        stamps[group] = stamp
        mapping = dict(zip(msg.name, msg.position))
        states[group] = ([mapping[n] for n in expected], time.monotonic())

    def model_received(msg: Any) -> None:
        try:
            if description and description[0] != msg.data:
                raise ValueError("ROS model changed during execution")
            check_kinematics(packet["kinematics"], kinematic_contract(msg.data))
            if not description:
                description.append(msg.data)
        except ValueError as error:
            faults.append(str(error))

    for group, topic, names in (
        ("arm", "/crx5ia/right/joint_states", ARM_NAMES),
        ("hand", "/sharpa/right_hand/joint_states", HAND_NAMES),
    ):
        node.create_subscription(
            JointState, topic, lambda m, g=group, n=names: receive(g, n, m), qos_profile_sensor_data
        )
    node.create_subscription(
        String,
        "/bimanual/robot_description",
        model_received,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )

    def fresh() -> bool:
        return all(g in states and time.monotonic() - states[g][1] <= args.feedback_timeout for g in ("arm", "hand"))

    def wait(predicate: Any, timeout: float, require_feedback: bool = False) -> None:
        if interrupt_requested:
            raise KeyboardInterrupt
        deadline = time.monotonic() + timeout
        while not predicate():
            rclpy.spin_once(node, timeout_sec=0.01)
            if interrupt_requested:
                raise KeyboardInterrupt
            if faults:
                raise RuntimeError(faults[0])
            if require_feedback and not fresh():
                raise RuntimeError("Feedback expired while awaiting ROS result")
            if time.monotonic() > deadline:
                raise TimeoutError("ROS discovery/result timeout")

    def interrupted(signum: int, frame: Any) -> None:
        nonlocal interrupt_requested
        interrupt_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, interrupted)
    try:
        wait(lambda: bool(description) and fresh() and client.server_is_ready(), args.startup_timeout)
        (root / "live_model.urdf").write_text(description[0])
        event("model_verified", source_model_sha256=packet["source"]["model_sha256"])
        for index, target in enumerate(packet["targets"]):
            if interrupt_requested:
                raise KeyboardInterrupt
            if not fresh() or faults:
                raise RuntimeError("Feedback/model invalid before next phase")
            phase = target["phase"]
            start = states["arm"][0] + states["hand"][0]
            duration = motion_duration(start, target["positions"], args)
            measured_duration = duration
            if "reviewed_plan" in packet:
                from ada_grasp_ctrl.pipeline.plan import check_start

                plan = packet["reviewed_plan"]
                check_start(
                    start,
                    plan["keyframes"][index],
                    hand_tolerance=plan["receiver_settings"]["hand_tolerance"] if index > 0 else 0.02,
                    ignore_hand_error=index > 0 and not plan["receiver_settings"].get("hand_arrival_required", True),
                )
                for key in ("arm_speed", "hand_speed", "arm_acceleration", "hand_acceleration"):
                    if getattr(args, key) != plan["receiver_settings"][key]:
                        raise ValueError("Sender limits differ from reviewed plan")
                duration = plan["durations_s"][index]
            secs = int(duration)
            point = JointTrajectoryPoint(
                positions=target["positions"], time_from_start=Duration(sec=secs, nanosec=int((duration - secs) * 1e9))
            )
            goal = FollowJointTrajectory.Goal(trajectory=JointTrajectory(joint_names=list(JOINT_NAMES), points=[point]))
            event("goal_sent", phase=phase, positions=target["positions"], duration=duration)
            print(f"Ada sending {phase}: 28 named joints, {duration:.2f} s", flush=True)

            def feedback(message: Any, phase: str = phase) -> None:
                value = message.feedback
                event(
                    "action_feedback",
                    phase=phase,
                    actual=list(value.actual.positions),
                    desired=list(value.desired.positions),
                    error=list(value.error.positions),
                )

            future = client.send_goal_async(goal, feedback_callback=feedback)
            pending_goal = future
            wait(future.done, args.startup_timeout, True)
            active = future.result()
            pending_goal = None
            if not active.accepted:
                active = None
                raise RuntimeError(f"{phase}: receiver rejected the goal")
            outcome = active.get_result_async()
            wait_duration = duration
            if not packet.get("reviewed_plan", {}).get("receiver_settings", {}).get("hand_arrival_required", True):
                wait_duration = max(duration, measured_duration)
            wait(outcome.done, wait_duration + args.startup_timeout + args.result_margin, True)
            response = outcome.result()
            active = None
            entry = {
                "phase": phase,
                "status": response.status,
                "error_code": response.result.error_code,
                "message": response.result.error_string,
            }
            report["phases"].append(entry)
            event("result", **entry)
            if response.status != GoalStatus.STATUS_SUCCEEDED or response.result.error_code != 0:
                raise RuntimeError(f"{phase}: {response.result.error_string}")
            if phase == "lift":
                report["lift_motion_completed"] = True
            if phase in ("hold", "release", "return"):
                report[phase + "_motion_completed"] = True
        report["status"] = "completed"
        report["task_completed"] = True
    finally:
        # A signal can arrive after the server accepts but before its reply reaches us.
        if active is None and pending_goal is not None:
            deadline = time.monotonic() + 3
            while not pending_goal.done() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
            if pending_goal.done():
                active = pending_goal.result()
            else:
                report["goal_acceptance_unknown"] = True
        if active is not None and active.accepted:
            cancellation = active.cancel_goal_async()
            deadline = time.monotonic() + 5
            while not cancellation.done() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
            report["cancel_acknowledged"] = bool(cancellation.done() and cancellation.result().goals_canceling)
            result_future = active.get_result_async()
            while not result_future.done() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
            report["cancel_result_received"] = result_future.done()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        events.close()
        client.destroy()
        node.destroy_node()
        rclpy.try_shutdown()


def main() -> int:
    """Preview by default; --execute sends the declared phases through the passive bridge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--environment", choices=("mock", "hardware"), default="mock")
    for name, value in (
        ("minimum-duration", 2.0),
        ("arm-speed", 0.15),
        ("hand-speed", 0.4),
        ("arm-acceleration", 0.3),
        ("hand-acceleration", 0.8),
        ("feedback-timeout", 0.5),
        ("startup-timeout", 25.0),
        ("result-margin", 10.0),
    ):
        parser.add_argument("--" + name, type=float, default=value)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "status": "preflight",
        "phases": [],
        "task_completed": False,
        "object_lifted": None,
        "object_held": None,
        "ros_domain_id": os.getenv("ROS_DOMAIN_ID"),
    }
    root = None
    output_created = False
    try:
        for key, value in vars(args).items():
            if isinstance(value, float) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")
        path = resolve_external_root("data", args.targets)
        packet = load_packet(path)
        root = resolve_external_root("output", args.output_root)
        root.mkdir(parents=True, exist_ok=False)
        output_created = True
        (root / "run_manifest.yaml").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": "ada_ros_sender",
                    "input_path": str(path),
                    "input_sha256": digest(path),
                    "output_root": str(root),
                    "execute": args.execute,
                    "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                },
                indent=2,
            )
            + "\n"
        )
        (root / "targets.json").write_text(json.dumps(packet, indent=2) + "\n")
        if not args.execute:
            print(
                json.dumps(
                    {
                        "sample_id": packet["sample_id"],
                        "joint_names": packet["joint_names"],
                        "targets": packet["targets"],
                    },
                    indent=2,
                )
            )
            report["status"] = "preview"
            return 0
        execute(packet, args, root, report)
        return 0
    except KeyboardInterrupt:
        report.update(status="aborted", reason="User interruption; no next phase")
        return 130
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        report.update(status="execution_error" if args.execute and root else "preflight_error", reason=str(error))
        print(str(error), file=sys.stderr)
        return 1 if report["status"] == "execution_error" else 2
    finally:
        if output_created:
            (root / "run_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
