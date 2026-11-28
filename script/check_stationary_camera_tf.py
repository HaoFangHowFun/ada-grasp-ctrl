#!/usr/bin/env python3
"""Synthetic ROS test for camera root attachments; never start camera/robot drivers."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener

from ada_grasp_ctrl.calibration import DEFAULT_CALIBRATION
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.experiments.multi_camera import DEFAULT_CONFIG, load_config
from ada_grasp_ctrl.experiments.multi_camera_calibration import load_cameras
from ada_grasp_ctrl.experiments.stationary_camera_tf import matrix_from_tf


def main() -> None:
    """Verify actual TF propagation, duplicate-owner rejection, and wrong-serial rejection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("ROS_DOMAIN_ID") not in (None, "", "0", "87")
    assert os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") == "LOCALHOST"
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(resolve_from_root(DEFAULT_CONFIG))
    cameras = load_cameras(resolve_from_root(DEFAULT_CALIBRATION), config)
    rclpy.init()
    executor = SingleThreadedExecutor()
    node = rclpy.create_node("stationary_camera_tf_synthetic_check")
    executor.add_node(node)
    fake_cameras = []
    for role in ("wrist_left", "wrist_right"):
        camera = rclpy.create_node(
            role,
            namespace="/camera",
            parameter_overrides=[Parameter("serial_no", value="_" + config["cameras"][role]["serial"])],
        )
        camera.declare_parameter("serial_no", "")
        fake_cameras.append(camera)
        executor.add_node(camera)
    listener_buffer = Buffer()
    listener = TransformListener(listener_buffer, node)
    broadcaster = StaticTransformBroadcaster(node)
    messages = []
    for index, role in enumerate(("wrist_left", "wrist_right")):
        message = TransformStamped()
        message.header.frame_id = role + "_link"
        message.child_frame_id = role + "_color_optical_frame"
        message.header.stamp = node.get_clock().now().to_msg()
        message.transform.translation.x = 0.012 + index * 0.005
        message.transform.translation.y = -0.003
        q = Rotation.from_euler("xyz", [0.6, -0.2, 0.4 + index]).as_quat()
        rotation = message.transform.rotation
        rotation.x, rotation.y, rotation.z, rotation.w = map(float, q)
        messages.append(message)
    broadcaster.sendTransform(messages)
    processes = []
    logs = []

    def start(name: str) -> subprocess.Popen:
        log = (args.output / (name + ".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(
            [sys.executable, "-m", "ada_grasp_ctrl.experiments.stationary_camera_tf", "--timeout", "6"],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append(process)
        return process

    def wait(predicate: object, timeout: float = 10) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
            if predicate():
                return
        raise AssertionError("Timed out; inspect test process logs")

    def ready() -> bool:
        try:
            for role in ("wrist_left", "wrist_right"):
                value = listener_buffer.lookup_transform("left_base_link", role + "_color_optical_frame", Time())
                np.testing.assert_allclose(matrix_from_tf(value), cameras[role]["matrix"], atol=1e-10)
            return True
        except TransformException:
            return False

    try:
        first = start("normal")
        wait(ready)
        assert first.poll() is None
        duplicate = start("duplicate")
        wait(lambda: duplicate.poll() is not None)
        assert duplicate.returncode != 0
        assert "already has TF parent" in (args.output / "duplicate.log").read_text()
        first.terminate()
        first.wait(timeout=5)
        fake_cameras[0].set_parameters([Parameter("serial_no", value="_WRONG")])
        wrong = start("wrong_serial")
        wait(lambda: wrong.poll() is not None)
        assert wrong.returncode != 0
        assert "wrong live camera serial" in (args.output / "wrong_serial.log").read_text()
        report = dict(
            synthetic=True,
            domain=os.environ["ROS_DOMAIN_ID"],
            left_optical_matches_solver=True,
            right_optical_matches_solver=True,
            duplicate_root_rejected=True,
            wrong_serial_rejected=True,
            robot_commands=False,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for log in logs:
            log.close()
        executor.shutdown()
        listener.unregister()
        for camera in fake_cameras:
            camera.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
