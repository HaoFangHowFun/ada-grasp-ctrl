"""Opt-in domain-88 DDS audit with synthetic driver services; never launch hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["both", "right", "left", "late_hand_error", "stale_right", "limit_left", "owner", "legacy_right"],
    )
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "88" or os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
        parser.error("Requires ROS_DOMAIN_ID=88 and ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST before any ROS startup")
    checkout = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(checkout / "src"), str(checkout / "tests")]
    from ada_grasp_ctrl.paths import resolve_from_root
    from ada_grasp_ctrl.experiments.right_ros_contract import side_names
    from ada_grasp_ctrl.ros_execution.shared_packet import SIDES
    from shared_executor_fixtures import packet, description
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String, Float64MultiArray
    from controller_manager_msgs.srv import ListControllers, SwitchController
    from controller_manager_msgs.msg import ControllerState
    from dual_sharpa_wave.qos import hand_qos

    root = resolve_from_root(args.output_root, "output")
    root.mkdir(parents=True, exist_ok=False)
    rclpy.init(args=[])
    executor = SingleThreadedExecutor()
    anchor = Node("ada_shared_mock_fixture")
    executor.add_node(anchor)
    nodes = [anchor]
    processes = []

    def spin_until(predicate, seconds=30):
        deadline = time.monotonic() + seconds
        while not predicate():
            executor.spin_once(timeout_sec=0.002)
            if time.monotonic() > deadline:
                raise TimeoutError("Mock fixture wait expired")

    def drain(seconds):
        deadline = time.monotonic() + seconds
        spin_until(lambda: time.monotonic() >= deadline, seconds + 1)

    def write(path, data):
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    def node(name, namespace):
        value = Node(name, namespace=namespace)
        executor.add_node(value)
        nodes.append(value)
        return value

    reports = {}
    try:
        drain(1.0)
        others = [
            (name, namespace) for name, namespace in anchor.get_node_names_and_namespaces() if name != anchor.get_name()
        ]
        if others:
            raise RuntimeError(f"Domain 88 is not empty; refusing to create mock driver identities: {others}")
        interp = node("joint_interpolation", "/crx5ia")
        interp.declare_parameter("active_arms", list(SIDES))
        interp.declare_parameter("input_rate_hz", 100.0)
        interp.declare_parameter("method", "quintic")
        transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        model_pub = anchor.create_publisher(String, "/bimanual/robot_description", transient)
        model_pub.publish(String(data=description()))
        limiter = anchor.create_publisher(String, "/crx5ia/collision_force_limiter/state", transient)
        limiter.publish(String(data="ARMED"))
        state = dict(
            case=None, q={s: [0.0] * 28 for s in SIDES}, history=[], stops=[], first_command=None, competing=None
        )
        feedback_pubs = {}
        controllers = dict.fromkeys(SIDES, "active")

        def received(kind, msg, side=None):
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            now = time.monotonic()
            if state["first_command"] is None:
                state["first_command"] = now
            topic = "/crx5ia/joint_targets" if kind == "arm" else f"/sharpa/{side}_hand/joint_command"
            owners = [
                dict(name=i.node_name, namespace=i.node_namespace) for i in anchor.get_publishers_info_by_topic(topic)
            ]
            state["history"].append(
                dict(
                    kind=kind,
                    side=side,
                    ros_stamp_ns=stamp,
                    received=now,
                    joint_names=list(msg.name),
                    positions_rad=list(msg.position),
                    owners=owners,
                )
            )
            values = dict(zip(msg.name, msg.position))
            for branch in SIDES:
                names = side_names(branch)[0 if kind == "arm" else 1]
                if all(name in values for name in names):
                    vector = [values[name] for name in names]
                    if kind == "arm":
                        if (
                            state["case"] == "late_hand_error"
                            and branch == "right"
                            and now - state["first_command"] < 6.05
                        ):
                            vector = [q - 0.03 for q in vector]
                        state["q"][branch][:6] = vector
                    elif state["case"] != "late_hand_error":
                        state["q"][branch][6:] = vector

        interp.create_subscription(JointState, "/crx5ia/joint_targets", lambda msg: received("arm", msg), 1)
        for side in SIDES:
            manager = node("controller_manager", f"/crx5ia/{side}")
            hand = node("hand_node", f"/sharpa/{side}_hand")
            for key, value in dict(side=side, backend="mock", read_only=False, control_mode="position").items():
                hand.declare_parameter(key, value)
            hand.create_subscription(
                JointState,
                f"/sharpa/{side}_hand/joint_command",
                lambda msg, s=side: received("hand", msg, s),
                hand_qos(),
            )
            hand_state = hand.create_publisher(String, f"/sharpa/{side}_hand/control_state", transient)
            hand_state.publish(String(data='{"state":"normal"}'))
            driver_model = manager.create_publisher(String, f"/crx5ia/{side}/robot_description", transient)
            driver_model.publish(
                String(
                    data="<robot><ros2_control><hardware><plugin>mock_components/GenericSystem</plugin></hardware></ros2_control></robot>"
                )
            )
            interp.create_publisher(Float64MultiArray, f"/crx5ia/{side}/forward_position_controller/commands", 1)
            for group, publisher_node, topic in (
                ("arm", manager, f"/crx5ia/{side}/joint_states"),
                ("hand", hand, f"/sharpa/{side}_hand/joint_states"),
            ):
                feedback_pubs[side, group] = publisher_node.create_publisher(JointState, topic, qos_profile_sensor_data)

            def listing(request, response, s=side):
                response.controller = [
                    ControllerState(name="forward_position_controller", state=controllers[s]),
                    ControllerState(name="joint_state_broadcaster", state="active"),
                ]
                return response

            def switching(request, response, s=side):
                state["stops"].append(
                    dict(side=s, deactivate=list(request.deactivate_controllers), time=time.monotonic())
                )
                controllers[s] = "inactive"
                response.ok = True
                return response

            manager.create_service(ListControllers, f"/crx5ia/{side}/controller_manager/list_controllers", listing)
            manager.create_service(SwitchController, f"/crx5ia/{side}/controller_manager/switch_controller", switching)

        def publish_feedback():
            now = time.monotonic()
            elapsed = now - state["first_command"] if state["first_command"] is not None else 0
            if state["case"] == "owner" and elapsed > 0.2 and state["competing"] is None:
                state["competing"] = anchor.create_publisher(JointState, "/crx5ia/joint_targets", 1)
            stamp = anchor.get_clock().now().to_msg()
            for side in SIDES:
                if state["case"] == "stale_right" and side == "right" and elapsed > 0.2:
                    continue
                for group, index, sl in (("arm", 0, slice(0, 6)), ("hand", 1, slice(6, 28))):
                    q = state["q"][side][sl].copy()
                    if state["case"] == "limit_left" and side == "left" and group == "arm" and elapsed > 0.2:
                        q[0] = 2.0
                    message = JointState(name=list(side_names(side)[index]), position=q)
                    message.header.stamp = stamp
                    feedback_pubs[side, group].publish(message)

        anchor.create_timer(0.01, publish_feedback)

        def reset(case):
            state.update(case=case, q={s: [0.0] * 28 for s in SIDES}, history=[], stops=[], first_command=None)
            controllers.update(dict.fromkeys(SIDES, "active"))
            if state["competing"] is not None:
                anchor.destroy_publisher(state["competing"])
                state["competing"] = None
            drain(0.5)

        def start_child(command, path):
            log = path.open("w")
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append((child, log))
            return child

        def stop_child(child):
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
                spin_until(lambda: child.poll() is not None, 10)

        def audit(case, active, case_root, failed):
            history = state["history"]
            write(case_root / "wire_recording.json", history)
            assert history, "No DDS commands observed"
            arm_rows = [r for r in history if r["kind"] == "arm"]
            hands = {s: [r for r in history if r["kind"] == "hand" and r["side"] == s] for s in SIDES}
            for row in history:
                if case == "legacy_right":
                    # Legacy action runtimes destroy publishers between goals;
                    # their final in-flight samples can outlive graph endpoints.
                    assert len(row["owners"]) <= 1, "Competing legacy command publisher"
                elif case != "owner" or row["kind"] != "arm":
                    assert len(row["owners"]) == 1, "Unexpected command publisher count"
            for side in SIDES:
                if side not in active:
                    assert not hands[side], "Passive Sharpa received commands"
                    for r in arm_rows:
                        values = dict(zip(r["joint_names"], r["positions_rad"]))
                        assert [values[n] for n in side_names(side)[0]] == [0.0] * 6
            result = dict(
                environment="mock",
                ros_domain_id=88,
                synthetic_drivers=True,
                hardware_used=False,
                arm_messages=len(arm_rows),
                hand_messages={s: len(v) for s, v in hands.items()},
                active_arms=list(active),
                stops=state["stops"],
            )
            if case != "legacy_right":
                events = [
                    json.loads(line)
                    for path in (case_root / "run").glob("*/events.jsonl")
                    for line in path.read_text().splitlines()
                ]
                published = {e["ros_stamp_ns"]: e for e in events if e["kind"] == "paired_published"}
                wire = {r["ros_stamp_ns"]: r for r in arm_rows}
                assert len(wire) == len(arm_rows), "Duplicate arm command timestamp"
                assert wire.keys() == published.keys(), "Published/audited arm stamps differ"
                for side in active:
                    assert {r["ros_stamp_ns"] for r in hands[side]} == wire.keys(), "Incomplete paired timestamps"
                    for r in hands[side]:
                        assert r["positions_rad"] == published[r["ros_stamp_ns"]]["hands"][side]["positions_rad"]
                        assert r["joint_names"] == published[r["ros_stamp_ns"]]["hands"][side]["joint_names"]
                for stamp, row in wire.items():
                    assert row["positions_rad"] == published[stamp]["arm"]["positions_rad"]
                    assert row["joint_names"] == published[stamp]["arm"]["joint_names"]
                    if case != "owner":
                        assert len(row["owners"]) == 1, "Unexpected arm publisher count before a command"
                completed = [e for e in events if e["kind"] == "stage_completed"]
                started = [e for e in events if e["kind"] == "stage_started"]
                report = json.loads((case_root / "run/run_report.json").read_text())
                if failed:
                    assert report["final_state"] == "FAULT" and not report["task_completed"]
                    assert not completed and len(started) == 1
                    assert {s["side"] for s in state["stops"]} == set(SIDES), "Abort did not stop both CRX controllers"
                else:
                    assert report["final_state"] == "DONE" and report["task_completed"]
                    assert len(completed) == len(started)
                    for before, after in zip(completed, started[1:]):
                        assert after["stage_start_time"] >= before["monotonic_time"]
                if case == "late_hand_error":
                    assert completed[0]["monotonic_time"] - started[0]["stage_start_time"] > 6.05
                    assert max(abs(v) for v in completed[-1]["target_error_rad"]["right"][6:]) > 0.02
                result.update(
                    paired_stamps_verified=True,
                    published_frames=len(published),
                    completed_stages=len(completed),
                    abort_reason=report["abort_reason"],
                    shared_progress_verified=all(
                        e["progress"] == min(1.0, (e["monotonic_time"] - e["stage_start_time"]) / e["duration_s"])
                        for e in published.values()
                    ),
                )
                assert result["shared_progress_verified"]
            write(case_root / "audit.json", result)
            return result

        for case in args.cases:
            if case not in {
                "both",
                "right",
                "left",
                "late_hand_error",
                "stale_right",
                "limit_left",
                "owner",
                "legacy_right",
            }:
                raise ValueError(f"Unknown mock case: {case}")
            reset(case)
            case_root = root / case
            case_root.mkdir()
            active = (case,) if case in SIDES else ("right",) if case == "legacy_right" else SIDES
            if case == "legacy_right":
                from shared_legacy_mock import run_legacy

                run_legacy(case_root, description(), anchor, spin_until, start_child, stop_child)
                failed = False
            else:
                path = case_root / "targets.json"
                write(path, packet(active, full=case == "both"))
                child = start_child(
                    [
                        sys.executable,
                        "-m",
                        "ada_grasp_ctrl.ros_execution.shared_executor",
                        "--targets",
                        str(path),
                        "--output-root",
                        str(case_root / "run"),
                        "--ros-domain-id",
                        "88",
                    ],
                    case_root / "executor.log",
                )
                spin_until(lambda: child.poll() is not None, 65)
                failed = case in ("stale_right", "limit_left", "owner")
                assert child.returncode == (1 if failed else 0), (
                    case,
                    child.returncode,
                    (case_root / "executor.log").read_text()[-4000:],
                )
            drain(0.2)
            reports[case] = audit(case, active, case_root, failed)
            print(json.dumps(dict(case=case, **reports[case])), flush=True)
        write(
            root / "audit.json",
            dict(
                schema_version=1,
                status="passed",
                domain=88,
                cases=reports,
                fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
        )
        return 0
    finally:
        for child, log in processes:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            log.close()
        for value in reversed(nodes):
            executor.remove_node(value)
            value.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
