"""Inspect a ROS domain before starting owned mock drivers or a passive receiver."""

import argparse
import os
import time
from typing import Any, Callable

from ada_grasp_ctrl.experiments.right_ros_contract import action_name


def wait_for_tactile_publishers(node: Any, spin_once: Callable, timeout_s: float = 8.0) -> dict[str, int]:
    """Allow bounded DDS discovery after the initial probe; require exactly one source per finger."""
    from ada_grasp_ctrl.tactile_preview.data import TOPICS

    deadline = time.monotonic() + timeout_s
    while True:
        counts = {topic: node.count_publishers(topic) for topic in TOPICS.values()}
        if all(count == 1 for count in counts.values()):
            return counts
        if time.monotonic() >= deadline:
            detail = "; ".join(f"{topic}={count}" for topic, count in counts.items())
            raise RuntimeError(
                "Active tactile discovery timed out; require exactly 1 publisher per topic "
                f"(ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}): {detail}"
            )
        spin_once(node, timeout_sec=0.05)


def main() -> int:
    """Refuse duplicate drivers or action servers without creating motion endpoints."""
    import rclpy
    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty-domain", action="store_true")
    parser.add_argument(
        "--active-readiness", action="store_true", help="Require existing right hardware tactile services"
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    args = parser.parse_args()
    rclpy.init(args=[])
    node = rclpy.create_node("ada_pipeline_probe", enable_rosout=False, start_parameter_services=False)
    client = ActionClient(node, FollowJointTrajectory, action_name(args.side))
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if args.active_readiness:
            counts = wait_for_tactile_publishers(node, rclpy.spin_once)
            print("Active tactile ready: " + ", ".join(f"{t}={n}" for t, n in counts.items()), flush=True)
        if client.server_is_ready():
            raise RuntimeError("A receiver already owns the action; refusing to start another")
        if args.empty_domain and any(name != "ada_pipeline_probe" for name in node.get_node_names()):
            raise RuntimeError("ROS domain is occupied; use an empty mock domain or attach without --start-mock")
        return 0
    finally:
        client.destroy()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
