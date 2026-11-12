"""Exercise the actual Viser WebSocket, controls and rendered object transforms."""

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Callable, Iterator
import unittest
from urllib.request import urlopen

import msgspec
import numpy as np
import viser
from websockets.sync.client import connect, ClientConnection
import yaml
import zstandard

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.robot_environment.model import JOINT_NAMES


def receive(ws: ClientConnection) -> list[dict]:
    """Decode Viser's installed hybrid wire format, including geometry/pose arrays."""
    packet = ws.recv(timeout=15)
    compressed = int.from_bytes(packet[8:16], "little")
    payload = msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(packet[16 : 16 + compressed]))
    offset = (16 + compressed + 7) // 8 * 8
    buffers = []
    for length in payload["binaryBufferLengths"]:
        buffers.append(packet[offset : offset + length])
        offset = (offset + length + 7) // 8 * 8

    def decode(value: object) -> object:
        if isinstance(value, dict):
            if "__binary_index" in value:
                return np.frombuffer(buffers[value["__binary_index"]], dtype=value["dtype"]).copy()
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(payload["messages"])


@unittest.skipUnless(os.environ.get("ADA_DUAL_ARM_SNAPSHOT"), "Set ADA_DUAL_ARM_SNAPSHOT to real exported assets")
class WebPlacementTests(unittest.TestCase):
    @contextmanager
    def viewer(self, args: list[str]) -> Iterator[ClientConnection]:
        """Start the real CLI on a temporary port; forbid ROS and inference imports."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        bootstrap = """
import sys, runpy
class OfflineOnly:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rclpy', 'rospy') or fullname in (
            'ada_grasp_ctrl.experiments.mirrored_left_grasp',
            'ada_grasp_ctrl.experiments.dual_arm_grasp_pipeline',
            'ada_grasp_ctrl.experiments.dual_arm_grasp_episode',
        ):
            raise AssertionError('Viewer imported forbidden runtime: ' + fullname)
sys.meta_path.insert(0, OfflineOnly())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    bootstrap,
                    str(project_root() / "script/preview_dual_arm_web.py"),
                    *args,
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=log,
            )
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    log.seek(0)
                    output = log.read()
                    if "SOURCE_DAE_MATERIALS=" in output:
                        break
                    self.assertIsNone(process.poll(), output)
                    time.sleep(0.05)
                else:
                    self.fail("Viewer startup timed out: " + output)
                with urlopen(f"http://127.0.0.1:{port}", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                with connect(
                    f"ws://127.0.0.1:{port}", subprotocols=[f"viser-v{viser.__version__}"], max_size=None
                ) as ws:
                    ws.send(
                        msgspec.msgpack.encode(
                            dict(
                                type="ViewerCameraMessage",
                                wxyz=[1, 0, 0, 0],
                                position=[2, -2, 2],
                                fov=1.0,
                                near=0.01,
                                far=100.0,
                                image_height=600,
                                image_width=800,
                                look_at=[0, 0, 0],
                                up_direction=[0, 0, 1],
                            )
                        )
                    )
                    self.gui, self.nodes = {}, {}
                    self.ws = ws
                    self.wait_for(lambda: set(JOINT_NAMES) <= self.gui.keys())
                    yield ws
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)

    def wait_for(self, predicate: Callable[[], bool]) -> None:
        """Consume actual server messages until a checked UI/scene state is observed."""
        deadline = time.monotonic() + 20
        while not predicate():
            self.assertLess(time.monotonic(), deadline, "Expected viewer update did not arrive")
            for message in receive(self.ws):
                label = message.get("props", {}).get("label")
                if label is not None and "uuid" in message:
                    self.gui[label] = message
                name = message.get("name")
                if name:
                    node = self.nodes.setdefault(name, {})
                    node.update(message.get("props", {}))
                    node.update(message.get("updates", {}))
                    for key in ("position", "wxyz"):
                        if key in message:
                            node[key] = message[key]
                if message["type"] == "GuiUpdateMessage":
                    for control in self.gui.values():
                        if control["uuid"] == message["uuid"]:
                            control.setdefault("props", {}).update(message["updates"])

    def update(self, label: str, value: object) -> None:
        """Send the same value-change event as the browser frontend."""
        self.ws.send(
            msgspec.msgpack.encode(
                dict(type="GuiUpdateMessage", uuid=self.gui[label]["uuid"], updates={"value": value})
            )
        )

    def test_scene_sliders_render_save_and_optional_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = yaml.safe_load((project_root() / "config/simulation_scene.yaml").read_text())
            config["scene"]["robot_snapshot"] = os.environ["ADA_DUAL_ARM_SNAPSHOT"]
            source, saved = root / "scene.yaml", root / "saved.yaml"
            source.write_text(yaml.safe_dump(config))
            original = source.read_bytes()
            with self.viewer(["--scene", str(source), "--save-scene", str(saved), "--camera-preview"]):
                self.wait_for(lambda: "Save Scene" in self.gui and "/bodies/object/group2" in self.nodes)
                node = self.nodes["/bodies/object/group2"]
                self.wait_for(lambda: np.allclose(node.get("batched_positions", [0, 0, 0]), [0.08, -0.08, 0.072]))
                robot_transforms = {
                    name: (data["batched_positions"].copy(), data["batched_wxyzs"].copy())
                    for name, data in self.nodes.items()
                    if name.startswith(("/bodies/left_", "/bodies/right_")) and "batched_positions" in data
                }
                self.assertTrue(robot_transforms)
                self.assertTrue(any("calibration_table_surface" in name for name in self.nodes))
                self.assertEqual(
                    sum(
                        name.startswith("/source_visuals/") and "vertices" in data for name, data in self.nodes.items()
                    ),
                    34,
                )
                for label in ("Position Z (m)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)", "Settle Object"):
                    self.assertNotIn(label, self.gui)
                for label, value in (("Position X (m)", 0.1), ("Position Y (m)", -0.06)):
                    self.update(label, value)
                expected_quat = config["object"]["pose"]["quaternion_wxyz"]
                self.wait_for(
                    lambda: (
                        np.allclose(node["batched_positions"], [0.1, -0.06, 0.072])
                        and np.allclose(node["batched_wxyzs"], expected_quat)
                    )
                )
                for name, (position, quaternion) in robot_transforms.items():
                    np.testing.assert_array_equal(self.nodes[name]["batched_positions"], position)
                    np.testing.assert_array_equal(self.nodes[name]["batched_wxyzs"], quaternion)
                self.update("Save Scene", True)
                self.wait_for(lambda: saved.is_file())
                result = yaml.safe_load(saved.read_text())
                np.testing.assert_allclose(result["object"]["pose"]["position_m"], [0.1, -0.06, 0.072])
                np.testing.assert_allclose(result["object"]["pose"]["quaternion_wxyz"], expected_quat)
                result["object"]["pose"] = config["object"]["pose"]
                self.assertEqual(result, config)
                self.assertEqual(source.read_bytes(), original)
                self.update("Refresh camera / point cloud", True)
                self.wait_for(lambda: "Simulated camera" in self.gui and "/object_cloud" in self.nodes)
                self.assertEqual(self.nodes["/object_cloud"]["points"].size, 2048 * 3)

    def test_fixed_object_render_transform_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = yaml.safe_load((project_root() / "config/simulation_scene.yaml").read_text())
            config["scene"]["robot_snapshot"] = os.environ["ADA_DUAL_ARM_SNAPSHOT"]
            del config["object"]["physics"]
            source = Path(directory) / "fixed.yaml"
            source.write_text(yaml.safe_dump(config))
            with self.viewer(["--scene", str(source)]):
                name = "/fixed_bodies/object/group2"
                self.wait_for(lambda: name in self.nodes and "Position Y (m)" in self.gui)
                self.update("Position X (m)", 0.12)
                self.update("Position Y (m)", -0.04)
                expected = config["object"]["pose"]["quaternion_wxyz"]
                self.wait_for(
                    lambda: (
                        np.allclose(self.nodes[name].get("position", [0, 0, 0]), [0.12, -0.04, 0.072])
                        and np.allclose(self.nodes[name].get("wxyz", [1, 0, 0, 0]), expected)
                    )
                )

    def test_snapshot_still_has_56_joint_sliders_and_original_reset(self) -> None:
        snapshot = os.environ["ADA_DUAL_ARM_SNAPSHOT"]
        with self.viewer(["--snapshot", snapshot]):
            self.assertNotIn("Save Scene", self.gui)
            self.assertNotIn("Position X (m)", self.gui)
            initial = yaml.safe_load((Path(snapshot) / "initial_pose.yaml").read_text())
            for side in ("left", "right"):
                for index in range(6):
                    self.assertAlmostEqual(self.gui[f"{side}_J{index + 1}"]["value"], np.deg2rad(initial[side][index]))
            self.update("left_J1", 0.2)
            self.update("Reset to bimanual initial pose", True)
            self.wait_for(lambda: self.gui["left_J1"]["props"].get("value") == 0.0)


if __name__ == "__main__":
    unittest.main()
