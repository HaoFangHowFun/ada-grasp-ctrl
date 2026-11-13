"""Synthetic tactile/joint telemetry only, gated to the test-only localhost domain."""

import os
import signal
import threading
import time


def main() -> None:
    """Publish generated messages; never create a command publisher or connect an SDK."""
    if os.environ.get("ROS_DOMAIN_ID") != "198" or os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
        raise RuntimeError("Synthetic fixture requires isolated localhost test domain 198")
    import rclpy
    from dual_sharpa_wave.msg import TactileFrame
    from sensor_msgs.msg import JointState
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from ada_grasp_ctrl.experiments.right_ros_contract import HAND_NAMES
    from ada_grasp_ctrl.tactile_preview.data import FINGERS, TOPICS

    rclpy.init(args=[])
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    node = rclpy.create_node("ada_tactile_synthetic_fixture", enable_rosout=False, start_parameter_services=False)
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    publishers = {f: node.create_publisher(TactileFrame, TOPICS[f], qos) for f in FINGERS}
    joints = node.create_publisher(JointState, "/sharpa/right_hand/joint_states", qos)
    try:
        deadline, sequence = time.monotonic() + 25, 0
        while time.monotonic() < deadline and not stopped.is_set():
            sequence += 1
            stamp = node.get_clock().now().to_msg()
            for finger, publisher in publishers.items():
                msg = TactileFrame()
                msg.schema_version, msg.source, msg.side = 1, "sharpa_sdk", "right"
                msg.finger, msg.channel = finger, FINGERS.index(finger)
                msg.header.stamp, msg.publish_stamp = stamp, stamp
                msg.session_id, msg.receipt_sequence = "synthetic-only", sequence
                msg.sdk_frame_id_valid, msg.sdk_frame_id = True, sequence
                msg.sdk_frame_id_raw = str(sequence)
                msg.sdk_timestamp_unit = msg.sdk_timestamp_epoch = "unknown"
                msg.f6.status, msg.f6.shape = 1, [1, 1, 6]
                msg.f6.raw_values = [1.0, -2.0, 3.0, 4.0, 5.0, 6.0]
                msg.contact_points.status, msg.contact_points.shape = 1, [1, 3]
                msg.contact_points.raw_values = [124.8046875, 134.8828125, 11.578125]
                publisher.publish(msg)
            joint = JointState()
            joint.header.stamp = stamp
            joint.name, joint.position = list(HAND_NAMES), [0.0] * 22
            joint.position[joint.name.index("right_index_MCP_FE")] = 0.2
            joints.publish(joint)
            rclpy.spin_once(node, timeout_sec=0.01)
            time.sleep(0.02)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
