"""Read a verified full right-arm/hand state without creating command publishers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Callable

from ada_grasp_ctrl.experiments.right_ros_contract import ARM_NAMES, JOINT_NAMES, check_kinematics, kinematic_contract
from ada_grasp_ctrl.grasp_selection.adapters.state import world_base_from_description
from ada_grasp_ctrl.paths import resolve_external_root


def snapshot_receiver(runtime: Any) -> Callable:
    """Discard queued stale samples while a command-free snapshot waits for fresh feedback.

    The execution runtime remains strict. Discarded messages never update its
    feedback or freshness clock, and the bounded startup timeout still applies.
    """
    original_receive = runtime.receive

    def receive(group: str, names: list[str], message: Any) -> None:
        if group == "arm" and message.header.frame_id != "base_link":
            runtime.fault = "Unexpected right-arm feedback frame"
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        age = runtime.node.get_clock().now().nanoseconds * 1e-9 - stamp
        if stamp <= 0 or abs(age) > runtime.c["feedback_timeout"]:
            runtime.log.event("snapshot_feedback_discarded", group=group, stamp=stamp, age_seconds=age)
            return
        original_receive(group, names, message)

    return receive


def main(argv: list[str] | None = None) -> int:
    """Capture both groups without commands after checking the requested backends."""
    import rclpy

    from .model import Model
    from .runtime import RunLog, Runtime
    from .settings import bridge_config, validate_domain

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-urdf", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--environment", choices=("mock", "hardware"), default="mock")
    args = parser.parse_args(argv)
    validate_domain(args.environment, os.environ.get("ROS_DOMAIN_ID"))
    root = resolve_external_root("output", args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    expected = resolve_external_root("data", args.model_urdf).read_text()
    config = bridge_config({"environment": args.environment})
    log = RunLog(str(root / "diagnostics"), config, args)
    rclpy.init(args=[])
    runtime = Runtime(config, log)
    runtime.receive = snapshot_receiver(runtime)
    try:
        runtime.preflight(False)
        check_kinematics(kinematic_contract(expected), kinematic_contract(runtime.description))
        model = Model(runtime.description, config)
        runtime.spin(0.01)
        if not runtime.fresh():
            raise ValueError("Feedback expired while validating the snapshot")
        full = [v for group in ("arm", "hand") for v in model.check(group, runtime.feedback[group].q).tolist()]
        now = runtime.node.get_clock().now().nanoseconds * 1e-9
        result = {
            "schema_version": 1,
            "unit": "rad",
            "joint_names": list(ARM_NAMES),
            "positions_rad": full[:6],
            "full_joint_names": list(JOINT_NAMES),
            "full_positions_rad": full,
            "stamp_s": runtime.stamps["arm"],
            "hand_stamp_s": runtime.stamps["hand"],
            "received_s": now,
            "captured_wall_s": time.time(),
            "model_id": hashlib.sha256(expected.encode()).hexdigest(),
            "frame_id": "world",
            "source_frame_id": "base_link",
            "expected_source_frame_id": "base_link",
            "clock": "ros_system",
            "environment": args.environment,
            "environment_evidence": log.result["driver_interfaces"],
            "source_topic": "/crx5ia/right/joint_states",
            "hand_source_topic": "/sharpa/right_hand/joint_states",
            "expected_description": expected,
            "observed_description": runtime.description,
            "world_T_right_base": world_base_from_description(runtime.description),
            "transform_source": "/bimanual/robot_description: fixed world->right_base_link",
            "base_independently_verified": False,
            "motion_commands": 0,
        }
        (root / "robot_state.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        log.result.update(final_state="SNAPSHOT", task_completed=True)
        return 0
    except Exception as error:
        log.result.update(final_state="FAULT", abort_reason=str(error))
        raise
    finally:
        runtime.node.destroy_node()
        rclpy.try_shutdown()
        log.finish()


if __name__ == "__main__":
    raise SystemExit(main())
