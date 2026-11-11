"""Optional ROS Jazzy subscriber-only snapshot process; no Ada numeric imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from .adapters.state import check_description, named_arm, world_base_from_description


def capture(args: argparse.Namespace) -> dict:
    """Observe a right-only JointState and model description; never create command endpoints."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    expected = Path(args.model_urdf).read_text()
    rclpy.init(args=[])
    node = Node("grasp_selector_snapshot", enable_rosout=False, start_parameter_services=False)
    received = {}

    def on_state(message: JointState) -> None:
        received["state"] = message
        received["received_s"] = node.get_clock().now().nanoseconds * 1e-9

    def on_description(message: String) -> None:
        received["description"] = message.data

    node.create_subscription(JointState, args.topic, on_state, 10)
    node.create_subscription(
        String,
        args.description_topic,
        on_description,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    try:
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if "state" not in received or "description" not in received:
                continue
            message = received["state"]
            named_arm(list(message.name), list(message.position))
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            now = node.get_clock().now().nanoseconds * 1e-9
            if stamp <= 0 or not 0 <= now - stamp <= args.max_age_s:
                continue
            check_description(expected, received["description"])
            if message.header.frame_id != args.expected_header_frame:
                raise ValueError("Unexpected joint state frame: " + message.header.frame_id)
            return {
                "schema_version": 1,
                "unit": "rad",
                "joint_names": list(message.name),
                "positions_rad": list(message.position),
                "stamp_s": stamp,
                "received_s": received["received_s"],
                "model_id": hashlib.sha256(expected.encode()).hexdigest(),
                "frame_id": "world",
                "source_frame_id": message.header.frame_id,
                "clock": "ros_system",
                "expected_source_frame_id": args.expected_header_frame,
                "environment": args.environment,
                "environment_evidence": "operator-selected launch mode",
                "source_topic": args.topic,
                "description_topic": args.description_topic,
                "expected_description": expected,
                "observed_description": received["description"],
                "observed_description_sha256": hashlib.sha256(received["description"].encode()).hexdigest(),
                "world_T_right_base": world_base_from_description(received["description"]),
                "transform_source": args.description_topic + ": fixed world->right_base_link",
                "base_independently_verified": False,
                "motion_commands": 0,
            }
        raise ValueError("Timed out waiting for fresh right-only state and matching robot description")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    """Capture into a fresh directory with explicit model and environment inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-urdf", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--environment", required=True, choices=("mock", "hardware"))
    parser.add_argument("--topic", default="/crx5ia/right/joint_states")
    parser.add_argument("--description-topic", default="/bimanual/robot_description")
    parser.add_argument(
        "--expected-header-frame",
        default="base_link",
        help="Installed joint_state_broadcaster default; joint positions use the named URDF axes",
    )
    parser.add_argument("--timeout-s", type=float, default=15)
    parser.add_argument("--max-age-s", type=float, default=0.5)
    args = parser.parse_args(argv)
    output = Path(args.output_root)
    try:
        if not output.is_absolute() or not Path(args.model_urdf).is_absolute():
            raise ValueError("ROS boundary requires absolute model/output paths")
        if not all(math.isfinite(x) and x > 0 for x in (args.timeout_s, args.max_age_s)):
            raise ValueError("Timeout and maximum age must be finite and positive")
        output.mkdir(parents=True, exist_ok=False)
        result = capture(args)
        (output / "robot_state.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "captured", "snapshot": str(output / "robot_state.json"), "motion_commands": 0}))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "preflight_error", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
