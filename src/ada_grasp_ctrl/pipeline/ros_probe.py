"""Inspect a ROS domain before starting owned mock drivers or a passive receiver."""

import argparse
import time

from ada_grasp_ctrl.experiments.right_ros_contract import action_name


def main() -> int:
    """Refuse duplicate drivers or action servers without creating motion endpoints."""
    import rclpy
    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty-domain", action="store_true")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    args = parser.parse_args()
    rclpy.init(args=[])
    node = rclpy.create_node("ada_pipeline_probe", enable_rosout=False, start_parameter_services=False)
    client = ActionClient(node, FollowJointTrajectory, action_name(args.side))
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
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
