"""Concurrent passive RGB-D capture through the existing RealSense ROS drivers."""

from __future__ import annotations

from collections import deque
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .fixed_camera_geometry import rigid
from .multi_camera_geometry import ROLES, backproject, select_bundle


def image_array(message: object) -> np.ndarray:
    """Decode ROS image row padding and endianness without assuming packed rows."""
    formats = {"16UC1": ("u2", 1), "32FC1": ("f4", 1), "rgb8": ("u1", 3), "bgr8": ("u1", 3)}
    if message.encoding not in formats:
        raise ValueError(f"Unsupported image encoding {message.encoding}")
    code, channels = formats[message.encoding]
    dtype = np.dtype((">" if message.is_bigendian else "<") + code)
    row_bytes = message.width * dtype.itemsize * channels
    if message.step < row_bytes or len(message.data) < message.step * message.height:
        raise ValueError("Invalid image buffer or row stride")
    shape = (message.height, message.width) + ((channels,) if channels > 1 else ())
    strides = (message.step, dtype.itemsize * channels) + ((dtype.itemsize,) if channels > 1 else ())
    result = np.ndarray(shape, dtype=dtype, buffer=message.data, strides=strides).copy()
    return result[:, :, ::-1].copy() if message.encoding == "bgr8" else result


def capture(config: dict, diagnostics: dict, observe_s: float = 0.0) -> tuple[dict, dict]:
    """Subscribe to every camera before spinning; capture a bounded fresh common window.

    Args:
        config: Validated camera roles and freshness limits.
        diagnostics: Mutable per-camera diagnostics, retained even on timeout.
        observe_s: Optional sampling interval to measure live rates before selection.

    Returns:
        Per-camera arrays (depth-optical XYZ, RGB, depth and internal TF) and metadata.

    Raises:
        ValueError: On wrong device/frame/encoding or invalid intrinsics.
        RuntimeError: On bounded timeout with per-stream diagnostics.
    """
    import rclpy
    from rclpy.parameter_client import AsyncParameterClient
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.convert import message_to_ordereddict
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = rclpy.create_node("ada_multi_camera_capture")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    queues, stamps, subscriptions, clients, futures, serials = {}, {}, [], {}, {}, {}
    began = time.monotonic()
    after = node.get_clock().now().nanoseconds * 1e-9
    suffixes = dict(
        rgb="color/image_raw",
        depth="depth/image_rect_raw",
        color_info="color/camera_info",
        depth_info="depth/camera_info",
    )
    diagnostics.update(status="capturing", cameras={}, capture_request_s=after)

    def stamp(msg: object) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def receive(msg: object, key: str) -> None:
        timestamp = stamp(msg)
        queues[key].append((timestamp, msg))
        stamps[key].append(timestamp)

    for role in ROLES:
        clients[role] = AsyncParameterClient(node, f"/camera/{role}")
        diagnostics["cameras"][role] = dict(expected_serial=config["cameras"][role]["serial"], streams={})
        for stream, suffix in suffixes.items():
            key = role + "/" + stream
            queues[key], stamps[key] = deque(maxlen=20), deque(maxlen=1200)
            subscriptions.append(
                node.create_subscription(
                    CameraInfo if stream.endswith("info") else Image,
                    f"/camera/{role}/{suffix}",
                    lambda msg, key=key: receive(msg, key),
                    qos_profile_sensor_data,
                )
            )

    def update_diagnostics() -> None:
        now = node.get_clock().now().nanoseconds * 1e-9
        for role in ROLES:
            for stream in suffixes:
                times = stamps[role + "/" + stream]
                elapsed = times[-1] - times[0] if len(times) > 1 else 0
                diagnostics["cameras"][role]["streams"][stream] = dict(
                    received=len(times),
                    last_stamp_s=times[-1] if times else None,
                    age_s=now - times[-1] if times else None,
                    received_rate_hz=(len(times) - 1) / elapsed if elapsed > 0 else None,
                )
        diagnostics["capture_s"] = time.monotonic() - began

    reason = "Waiting for serial metadata and all RGB-D streams"
    try:
        while time.monotonic() - began < config["capture_timeout_s"]:
            rclpy.spin_once(node, timeout_sec=0.01)
            for role, client in clients.items():
                if role not in futures and client.services_are_ready():
                    futures[role] = client.get_parameters(["serial_no"])
                if role not in serials and role in futures and futures[role].done():
                    serials[role] = futures[role].result().values[0].string_value.lstrip("_")
                    diagnostics["cameras"][role]["serial"] = serials[role]
                    if serials[role] != config["cameras"][role]["serial"]:
                        raise ValueError(f"{role}: wrong device serial {serials[role]}")
            if len(serials) != len(ROLES) or time.monotonic() - began < observe_s:
                continue
            try:
                selected = select_bundle(
                    queues,
                    node.get_clock().now().nanoseconds * 1e-9,
                    after,
                    config["max_age_s"],
                    config["sync_tolerance_s"],
                )
            except ValueError as error:
                reason = str(error)
                continue
            transforms = {}
            for role in ROLES:
                depth, rgb, info, color_info = (
                    selected[role + "/" + key] for key in ("depth", "rgb", "depth_info", "color_info")
                )
                if (
                    depth.header.frame_id != role + "_depth_optical_frame"
                    or info.header.frame_id != depth.header.frame_id
                ):
                    raise ValueError(f"{role}: depth and depth CameraInfo frame mismatch")
                if (
                    rgb.header.frame_id != role + "_color_optical_frame"
                    or color_info.header.frame_id != rgb.header.frame_id
                ):
                    raise ValueError(f"{role}: RGB and color CameraInfo frame mismatch")
                if (rgb.width, rgb.height) != (color_info.width, color_info.height) or rgb.encoding not in (
                    "rgb8",
                    "bgr8",
                ):
                    raise ValueError(f"{role}: invalid color dimensions/encoding")
                try:
                    tf = buffer.lookup_transform(
                        rgb.header.frame_id, depth.header.frame_id, rclpy.time.Time.from_msg(depth.header.stamp)
                    )
                except Exception as error:
                    reason = f"{role}: internal depth-to-color TF unavailable: {error}"
                    break
                q, t = tf.transform.rotation, tf.transform.translation
                matrix = np.eye(4)
                matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                matrix[:3, 3] = [t.x, t.y, t.z]
                transforms[role] = rigid(matrix)
            if len(transforms) != len(ROLES):
                continue
            arrays, metadata = {}, {}
            for role in ROLES:
                messages = {key: selected[role + "/" + key] for key in suffixes}
                depth = image_array(messages["depth"])
                info = message_to_ordereddict(messages["depth_info"])
                points, indices = backproject(depth, messages["depth"].encoding, info, config["cameras"][role])
                limits = config["cameras"][role]
                if len(points) < limits["min_points"] or len(points) / depth.size < limits["min_valid_fraction"]:
                    raise ValueError(
                        f"{role}: insufficient valid depth: {len(points)} points, "
                        f"coverage={len(points) / depth.size:.4f}"
                    )
                arrays[role] = dict(
                    points_source_m=points,
                    depth_pixel_indices=indices,
                    depth=depth,
                    rgb=image_array(messages["rgb"]),
                    T_color_source=transforms[role],
                )
                metadata[role] = dict(
                    serial=serials[role],
                    source_frame=messages["depth"].header.frame_id,
                    color_frame=messages["rgb"].header.frame_id,
                    depth_encoding=messages["depth"].encoding,
                    depth_unit_m=0.001 if messages["depth"].encoding == "16UC1" else 1.0,
                    depth_info=info,
                    color_info=message_to_ordereddict(messages["color_info"]),
                    timestamps_s={key: stamp(msg) for key, msg in messages.items()},
                    topics={key: f"/camera/{role}/{suffix}" for key, suffix in suffixes.items()},
                    valid_points=len(points),
                    valid_depth_fraction=len(points) / depth.size,
                    T_color_source=transforms[role].tolist(),
                )
            times = [stamp(msg) for msg in selected.values()]
            update_diagnostics()
            diagnostics.update(status="captured", timestamp_skew_s=max(times) - min(times))
            return arrays, metadata
        update_diagnostics()
        raise RuntimeError(f"Three-camera capture timed out: {reason}; cameras={diagnostics['cameras']}")
    finally:
        update_diagnostics()
        del listener
        node.destroy_node()
        rclpy.shutdown()
