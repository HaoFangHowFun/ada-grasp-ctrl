"""Attach calibrated stationary RealSense roots without duplicating optical TF."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ada_grasp_ctrl.calibration import DEFAULT_CALIBRATION
from ada_grasp_ctrl.paths import resolve_from_root
from .fixed_camera_geometry import rigid
from .multi_camera import DEFAULT_CONFIG, load_config
from .multi_camera_calibration import load_cameras


def root_transform(base_optical: np.ndarray, root_optical: np.ndarray) -> np.ndarray:
    """Convert a calibrated optical pose to the driver's root attachment exactly once."""
    return rigid(rigid(base_optical) @ np.linalg.inv(rigid(root_optical)))


def matrix_from_tf(message: object) -> np.ndarray:
    """Convert a ROS transform to a rigid matrix in its parent-to-child direction."""
    t, q = message.transform.translation, message.transform.rotation
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    result[:3, 3] = [t.x, t.y, t.z]
    return rigid(result)


def require_camera_drivers_stopped(roles: list[str]) -> None:
    """Reject visible same-domain drivers before launching duplicate serial owners."""
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor

    context = Context()
    rclpy.init(args=[], context=context)
    node = rclpy.create_node(
        "calibrated_camera_start_check", context=context, enable_rosout=False, start_parameter_services=False
    )
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            if any(
                name in roles and namespace == "/camera" for name, namespace in node.get_node_names_and_namespaces()
            ):
                raise ValueError("Camera drivers already running; use start_cameras:=false to reuse them")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)


def publish(cameras: dict, config: dict, timeout: float) -> None:
    """Verify live serials/internal TF, then publish two persistent static root edges.

    Raises:
        ValueError: For wrong serials or an existing external parent on either root.
        TimeoutError: When the cameras have not supplied their internal transforms.
    """
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from rclpy.parameter_client import AsyncParameterClient
    from rclpy.time import Time
    from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener

    rclpy.init(args=[])
    node = rclpy.create_node("stationary_camera_extrinsics")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    broadcaster = StaticTransformBroadcaster(node)
    roles = ("wrist_left", "wrist_right")
    clients = {role: AsyncParameterClient(node, f"/camera/{role}") for role in roles}
    futures, serials, internals = {}, {}, {}
    began = time.monotonic()
    try:
        while time.monotonic() - began < timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
            for role, client in clients.items():
                if role not in futures and client.services_are_ready():
                    futures[role] = client.get_parameters(["serial_no"])
                if role not in serials and role in futures and futures[role].done():
                    serials[role] = futures[role].result().values[0].string_value.lstrip("_")
                    if serials[role] != config["cameras"][role]["serial"]:
                        raise ValueError(f"{role}: wrong live camera serial {serials[role]}")
                try:
                    internals[role] = matrix_from_tf(
                        buffer.lookup_transform(f"{role}_link", cameras[role]["calibrated_optical_frame"], Time())
                    )
                except TransformException:
                    pass
            if len(serials) == 2 and len(internals) == 2 and time.monotonic() - began >= 2:
                break
        else:
            raise TimeoutError("Waiting for D405 serial parameters and internal root-to-optical TF; enable publish_tf")
        frames = yaml.safe_load(buffer.all_frames_as_yaml()) or {}
        for role in roles:
            parent = frames.get(f"{role}_link", {}).get("parent")
            if parent and parent != "NO_PARENT":
                raise ValueError(f"{role}_link already has TF parent {parent}; stop the old extrinsics publisher")
        messages = []
        for role in roles:
            matrix = root_transform(cameras[role]["matrix"], internals[role])
            message = TransformStamped()
            message.header.frame_id, message.child_frame_id = "left_base_link", f"{role}_link"
            message.header.stamp = node.get_clock().now().to_msg()
            t, q = message.transform.translation, message.transform.rotation
            t.x, t.y, t.z = map(float, matrix[:3, 3])
            q.x, q.y, q.z, q.w = map(float, Rotation.from_matrix(matrix[:3, :3]).as_quat())
            messages.append(message)
        broadcaster.sendTransform(messages)
        for role in roles:
            node.get_logger().info(
                f"Loaded left_base_link -> {role}_link; calibration status: {cameras[role]['status']}"
            )
        rclpy.spin(node)
    finally:
        listener.unregister()
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv: list[str] | None = None) -> int:
    """Load the same calibrated transforms used by the existing fusion path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not np.isfinite(args.timeout) or args.timeout <= 2:
        parser.error("--timeout must be greater than two seconds")
    config = load_config(resolve_from_root(args.config))
    cameras = load_cameras(Path(resolve_from_root(args.calibration)), config)
    try:
        publish(cameras, config, args.timeout)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
