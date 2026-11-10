"""Optional system-Python ROS boundary; only subscriptions and visualization publishers."""

from __future__ import annotations

from collections import deque
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .fixed_camera_geometry import rigid, transform, validate_timing, valid_depth_mask


def capture(config: dict, world_color: np.ndarray, provenance: dict, output: Path) -> tuple[dict, dict]:
    """Capture a fresh synchronized cloud/RGB pair and live intrinsics without controlling devices.

    Args:
        config: Topics, expected serial/frame, freshness and depth thresholds.
        world_color: Verified-as-rigid supplied calibration, applied exactly once.
        provenance: Calibration path/hash; physical verification remains pending.
        output: Newly created run directory for serialized original ROS messages.

    Returns:
        Numeric snapshot arrays and JSON-serializable capture metadata.

    Raises:
        RuntimeError: Missing ROS/TF/device metadata or timeout waiting for valid data.
        ValueError: Wrong serial/frame, invalid calibration or insufficient valid points.
    """
    import rclpy
    from rclpy.parameter_client import AsyncParameterClient
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.convert import message_to_ordereddict
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from sensor_msgs_py import point_cloud2
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = rclpy.create_node("ada_fixed_camera_capture")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    queues = {key: deque(maxlen=12) for key in ("cloud", "rgb", "color_info", "depth_info")}
    subscriptions = []
    for key, cls in [("cloud", PointCloud2), ("rgb", Image), ("color_info", CameraInfo), ("depth_info", CameraInfo)]:
        subscriptions.append(
            node.create_subscription(
                cls, config[key + "_topic"], lambda msg, key=key: queues[key].append(msg), qos_profile_sensor_data
            )
        )

    def stamp(msg: object) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    client = AsyncParameterClient(node, config["camera_node"])
    deadline = time.monotonic() + config["capture_timeout_s"]
    future = None
    serial = None
    reason = "waiting for camera metadata and synchronized messages"
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if future is None and client.services_are_ready():
                future = client.get_parameters(["serial_no"])
            if future is not None and future.done() and serial is None:
                serial = future.result().values[0].string_value.lstrip("_")
                if serial != config["serial"]:
                    raise ValueError(f"Camera serial parameter {serial!r} != expected {config['serial']!r}")
            if serial is None or not all(queues.values()):
                continue
            cloud = queues["cloud"][-1]
            selected = {"cloud": cloud}
            for key in ("rgb", "color_info", "depth_info"):
                selected[key] = min(queues[key], key=lambda msg: abs(stamp(msg) - stamp(cloud)))
            now = node.get_clock().now().nanoseconds / 1e9
            try:
                ages, offsets = validate_timing(
                    {key: stamp(msg) for key, msg in selected.items()},
                    now,
                    config["max_age_s"],
                    config["sync_tolerance_s"],
                )
            except ValueError as exc:
                reason = str(exc)
                continue
            rgb, info, depth_info = (selected[key] for key in ("rgb", "color_info", "depth_info"))
            if rgb.header.frame_id != config["color_frame"] or info.header.frame_id != config["color_frame"]:
                raise ValueError("RGB/CameraInfo must identify the configured color optical frame")
            if (rgb.width, rgb.height) != (info.width, info.height) or rgb.encoding not in ("rgb8", "bgr8"):
                raise ValueError("RGB dimensions/encoding do not match live CameraInfo")
            if cloud.header.frame_id not in (config["color_frame"], depth_info.header.frame_id):
                raise ValueError(f"Unexpected pointcloud frame: {cloud.header.frame_id}")
            if cloud.header.frame_id == config["color_frame"]:
                color_source = np.eye(4)
            else:
                try:
                    tf = buffer.lookup_transform(
                        config["color_frame"], cloud.header.frame_id, rclpy.time.Time.from_msg(cloud.header.stamp)
                    )
                except Exception as exc:
                    reason = f"depth-to-color TF unavailable: {exc}"
                    continue
                q, t = tf.transform.rotation, tf.transform.translation
                color_source = np.eye(4)
                color_source[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                color_source[:3, 3] = [t.x, t.y, t.z]
            source = point_cloud2.read_points_numpy(cloud, field_names=("x", "y", "z"), skip_nans=False).reshape(-1, 3)
            try:
                valid, valid_fraction, pixel_coverage = valid_depth_mask(
                    source, depth_info.width * depth_info.height, config
                )
            except ValueError as exc:
                reason = str(exc)
                continue
            color = transform(source[valid], color_source)
            world = transform(color, world_color)
            world_source = rigid(world_color @ color_source)
            roundtrip = float(np.max(np.abs(transform(world, np.linalg.inv(world_source)) - source[valid])))
            image = np.ndarray(
                (rgb.height, rgb.width, 3), dtype=np.uint8, buffer=rgb.data, strides=(rgb.step, 3, 1)
            ).copy()
            if rgb.encoding == "bgr8":
                image = image[:, :, ::-1].copy()
            for key, msg in selected.items():
                (output / f"{key}.cdr").write_bytes(serialize_message(msg))
            arrays = dict(
                points_source_m=source,
                valid_source_indices=np.flatnonzero(valid),
                points_color_m=color,
                points_world_m=world,
                rgb=image,
                T_color_source=color_source,
                T_world_color=world_color,
                T_world_source=world_source,
            )
            metadata = dict(
                schema_version=1,
                unit="m",
                frame_id="world",
                serial=serial,
                serial_evidence="live driver serial_no parameter; explicit serial selection",
                source_frame=cloud.header.frame_id,
                color_frame=config["color_frame"],
                capture_stamp_s=stamp(cloud),
                ages_s=ages,
                offsets_s=offsets,
                cloud_dimensions=[cloud.width, cloud.height],
                valid_fraction=valid_fraction,
                valid_depth_pixel_fraction=pixel_coverage,
                color_info=message_to_ordereddict(info),
                depth_info=message_to_ordereddict(depth_info),
                transform_mode="direct calibration JSON @ internal TF; world TF not applied",
                extrinsics=provenance,
                camera_roundtrip_max_m=roundtrip,
                topics={key: config[key] for key in config if key.endswith("_topic")},
                T_world_source=world_source.tolist(),
                T_world_color=world_color.tolist(),
                T_color_source=color_source.tolist(),
            )
            return arrays, metadata
        raise RuntimeError(f"Capture timed out: {reason}")
    finally:
        del listener
        node.destroy_node()
        rclpy.shutdown()


def publish_preview(points: np.ndarray, plane: dict, duration_s: float = 0, topic: str = "/ada/table/points") -> None:
    """Publish only world PointCloud2 and a finite observed plane patch for RViz."""
    import rclpy
    from geometry_msgs.msg import Point
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    from visualization_msgs.msg import Marker

    rclpy.init()
    node = rclpy.create_node("ada_table_preview")
    cloud_pub = node.create_publisher(PointCloud2, topic, 1)
    marker_pub = node.create_publisher(Marker, "/ada/table/plane", 1)
    center = np.asarray(plane["centroid_world_m"])
    axes = np.asarray(plane["observed_tangent_axes"])
    half = np.asarray(plane["observed_span_m"]) / 2
    corners = [
        center + sx * half[0] * axes[0] + sy * half[1] * axes[1] for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ]
    marker = Marker()
    marker.header.frame_id = "world"
    marker.ns = "observed_table_support"
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.2, 0.8, 0.4, 0.35)
    marker.points = [
        Point(x=float(corners[i][0]), y=float(corners[i][1]), z=float(corners[i][2])) for i in [0, 1, 2, 0, 2, 3]
    ]
    start = time.monotonic()
    try:
        while rclpy.ok() and (duration_s <= 0 or time.monotonic() - start < duration_s):
            header = Header(frame_id="world", stamp=node.get_clock().now().to_msg())
            cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, points[:: max(1, len(points) // 30000)]))
            marker.header = header
            marker_pub.publish(marker)
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
