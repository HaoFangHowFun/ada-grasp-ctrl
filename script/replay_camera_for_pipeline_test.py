"""Replay recorded real camera messages with fresh stamps on isolated mock domain 88.

This is a sensor transport test source, not a live physical-camera validation.
It publishes no robot command and must not be used on a hardware ROS domain.
"""

import argparse
import json
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation
import numpy as np


def main() -> int:
    """Publish original recorded cloud/RGB/intrinsics and the recorded internal transform."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "88":
        parser.error("This recorded-camera test source requires isolated ROS_DOMAIN_ID=88")
    metadata = json.loads((args.snapshot / "snapshot.json").read_text())
    rclpy.init(args=[])
    node = Node("fixed", namespace="/camera")
    node.declare_parameter("serial_no", metadata["serial"])
    node.declare_parameter("recorded_camera_test_source", True)
    rows = []
    for key, cls in (("cloud", PointCloud2), ("rgb", Image), ("color_info", CameraInfo), ("depth_info", CameraInfo)):
        message = deserialize_message((args.snapshot / (key + ".cdr")).read_bytes(), cls)
        publisher = node.create_publisher(cls, metadata["topics"][key + "_topic"], qos_profile_sensor_data)
        rows.append((publisher, message))
    transform = TransformStamped()
    transform.header.frame_id = metadata["color_frame"]
    transform.child_frame_id = metadata["source_frame"]
    transform.header.stamp = node.get_clock().now().to_msg()
    matrix = np.asarray(metadata["T_color_source"])
    translation, quaternion = transform.transform.translation, transform.transform.rotation
    translation.x, translation.y, translation.z = map(float, matrix[:3, 3])
    quaternion.x, quaternion.y, quaternion.z, quaternion.w = map(float, Rotation.from_matrix(matrix[:3, :3]).as_quat())
    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(transform)

    def publish() -> None:
        stamp = node.get_clock().now().to_msg()
        for publisher, message in rows:
            message.header.stamp = stamp
            publisher.publish(message)

    node.create_timer(0.1, publish)
    print("Recorded-camera ROS transport source ready; not a physical camera", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
