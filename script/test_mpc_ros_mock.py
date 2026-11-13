"""Opt-in isolated ROS mock integration with native MPC in a separate Python ABI."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


def main() -> int:
    """Exercise real DDS/driver topics after independently verifying mock backends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--mpc-python", required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "199" or os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
        raise ValueError("Requires isolated localhost test domain 199")
    args.output_root.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from rclpy.qos import qos_profile_sensor_data
    from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, PHASES
    from ada_grasp_ctrl.mpc.ipc import MPCWorker
    from ada_grasp_ctrl.mpc.contracts import MPCProposal
    from ada_grasp_ctrl.ros_execution.mpc_stream import ROSMockStream
    from ada_grasp_ctrl.ros_execution.runtime import RunLog, ARM_COMMAND, HAND_COMMAND
    from ada_grasp_ctrl.ros_execution.settings import bridge_config

    config = bridge_config()
    report = json.loads(args.model_report.read_text())
    plan = {"phases": list(PHASES), "durations_s": [2.0, 2.0, 2.0], "keyframes": [[0.0] * 28 for _ in range(4)]}
    worker = MPCWorker(
        args.mpc_python,
        {
            "schema_version": 1,
            "environment": "mock",
            "plan": plan,
            "model_report": report,
            "output": str(args.output_root),
        },
        args.output_root / "worker.log",
    )
    rclpy.init(args=[])
    probe = rclpy.create_node("ada_mpc_mock_test_observer")
    received = {"arm": [], "hand": [], "lower": [], "left_arm": [], "left_hand": []}

    def receive(key, message):
        values = list(message.position) if hasattr(message, "position") else list(message.data)
        received[key].append(
            {
                "time": time.monotonic(),
                "positions": values,
                "names": list(message.name) if hasattr(message, "name") else [],
            }
        )

    for key, topic, kind in (
        ("arm", ARM_COMMAND, JointState),
        ("hand", HAND_COMMAND, JointState),
        ("lower", "/crx5ia/right/forward_position_controller/commands", Float64MultiArray),
        ("left_arm", "/crx5ia/left/forward_position_controller/commands", Float64MultiArray),
        ("left_hand", "/sharpa/left_hand/joint_command", JointState),
    ):
        probe.create_subscription(kind, topic, lambda m, k=key: receive(k, m), qos_profile_sensor_data)
    log = RunLog(str(args.output_root / "runtime"), config, argparse.Namespace(test="mpc_mock_only"))
    transport = ROSMockStream({}, log, domain=199)
    requests, proposals, commands, measurements = [], [], [], []
    result = {"hardware_used": False, "domain": 199, "environment": "mock", "fault_tests": {}}
    try:
        initial = transport.open()
        began = time.monotonic()
        transport.guard.accept(
            MPCProposal(tuple(JOINT_NAMES), initial.positions_rad, True, began, 0, "startup_hold"), began
        )
        next_mpc = began
        result["publishers"] = transport.inventory()
        while time.monotonic() - began < args.duration:
            rclpy.spin_once(probe, timeout_sec=0)
            candidate = worker.poll()
            if candidate:
                proposals.append(asdict(candidate))
            command = transport.tick(candidate)
            now = time.monotonic()
            if command is not None:
                commands.append({"time": transport.guard.last_tick, "positions": command})
            fb = transport.feedback()
            measurements.append({"time": now, "source_timestamp": fb.source_timestamp, "positions": fb.positions_rad})
            if now >= next_mpc:
                reference = list(initial.positions_rad)
                reference[0] += 0.002
                observation = {
                    "timestamp": fb.source_timestamp,
                    "contacts": [],
                    "source": "synthetic_valid_empty",
                    "valid": True,
                    "errors": [],
                    "environment_contacts": [],
                }
                worker.submit(now, fb, transport.guard.q, tuple(reference), observation, "grasp")
                requests.append(now)
                next_mpc = now + 0.1
            time.sleep(0.0005)
        # A second owner must be rejected before it can publish, including the
        # legacy receiver (which uses exactly this Runtime.preflight(True) gate).
        from ada_grasp_ctrl.ros_execution.runtime import Runtime

        competitor = Runtime(config, log)
        try:
            try:
                competitor.preflight(True)
                raise AssertionError("Competing owner unexpectedly passed preflight")
            except RuntimeError as error:
                if "Another sender owns" not in str(error):
                    raise
                result["fault_tests"]["second_owner"] = str(error)
        finally:
            competitor.node.destroy_node()
        # Discovery can now finish faster than 50 ms. Inject the intended stall
        # explicitly instead of relying on incidental DDS discovery latency.
        time.sleep(0.06)
        before = transport.published
        try:
            transport.tick()
            raise AssertionError("Command stall unexpectedly published")
        except ValueError as error:
            result["fault_tests"]["command_stall"] = str(error)
        assert transport.published == before
        silence_start = time.monotonic()
        while time.monotonic() - silence_start < 0.2:
            rclpy.spin_once(probe, timeout_sec=0.005)
        result["publisher_counts_after_shutdown"] = {t: probe.count_publishers(t) for t in (ARM_COMMAND, HAND_COMMAND)}
        assert all(n == 0 for n in result["publisher_counts_after_shutdown"].values())
        assert not received["left_arm"] and not received["left_hand"]
        assert all(len(r["names"]) == 6 and r["names"] == list(JOINT_NAMES[:6]) for r in received["arm"])
        assert all(len(r["names"]) == 22 and r["names"] == list(JOINT_NAMES[6:]) for r in received["hand"])
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        transport.close("test shutdown")
        worker.close()
        log.finish()
        probe.destroy_node()
        rclpy.try_shutdown()

        def stats(values):
            return (
                None
                if not values
                else dict(zip(("p50", "p95", "p99", "max"), map(float, np.percentile(values, [50, 95, 99, 100]))))
            )

        result.update(
            mpc_requests=len(requests),
            proposals=len(proposals),
            commands=len(commands),
            received_counts={k: len(v) for k, v in received.items()},
            mpc_period_s=stats(np.diff(requests).tolist()),
            publish_period_s=stats(np.diff([c["time"] for c in commands]).tolist()),
            lower_period_s=stats(np.diff([c["time"] for c in received["lower"]]).tolist()),
            solve_s=stats([p["solve_duration_s"] for p in proposals]),
            feedback_age_s=stats([r["time"] - r["source_timestamp"] for r in measurements]),
        )
        (args.output_root / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        (args.output_root / "trace.json").write_text(
            json.dumps(
                {
                    "requests": requests,
                    "proposals": proposals,
                    "commands": commands,
                    "received": received,
                    "measurements": measurements,
                },
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
