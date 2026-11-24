"""Real 8089 WebSocket review and optional installed-UOP integration smoke test."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen

import msgspec
import numpy as np
import viser
from websockets.sync.client import connect

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.placement.session import PlacementSession, sha256
from test_object_placement_web import receive
from test_placement_selection import candidate


class PlacementWebTests(unittest.TestCase):
    @contextmanager
    def viewer(self, arguments: list[str]):
        """Exercise the existing server port; never stop an unrelated listener."""
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8089)) == 0:
                self.skipTest("User's 8089 is occupied")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project_root() / "src")
        environment.pop("LD_LIBRARY_PATH", None)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        # ROS imports in this offline review are a regression, even before publishing.
        bootstrap = """
import sys, runpy
class OfflineOnly:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rclpy', 'rospy'):
            raise AssertionError('Placement imported ROS: ' + fullname)
sys.meta_path.insert(0, OfflineOnly())
module = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_module(module, run_name='__main__')
"""
        with tempfile.TemporaryFile(mode="w+") as log:
            child = subprocess.Popen(
                [sys.executable, "-c", bootstrap, *arguments],
                env=environment,
                stdout=log,
                stderr=log,
                cwd=project_root(),
            )
            try:
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    log.seek(0)
                    output = log.read()
                    if "PREVIEW_URL=" in output:
                        break
                    self.assertIsNone(child.poll(), output)
                    time.sleep(0.05)
                else:
                    self.fail("Placement viewer startup timed out: " + output)
                with urlopen("http://127.0.0.1:8089", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                with connect(
                    "ws://127.0.0.1:8089",
                    subprotocols=[f"viser-v{viser.__version__}"],
                    max_size=None,
                ) as ws:
                    self.ws, self.gui, self.nodes = ws, {}, {}
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
                    self.wait_for(lambda: "Placement candidate" in self.gui and "/object" in self.nodes)
                    yield
            finally:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 45
        while not predicate():
            self.assertLess(time.monotonic(), deadline, "Expected placement UI update")
            try:
                messages = receive(self.ws)
            except TimeoutError:
                continue
            for message in messages:
                props = message.get("props", {})
                if "label" in props and "uuid" in message:
                    if "value" in message:
                        props["value"] = message["value"]
                    self.gui[props["label"]] = message
                if "name" in message:
                    node = self.nodes.setdefault(message["name"], {})
                    node.update(props)
                    node.update(message.get("updates", {}))
                if message["type"] == "GuiUpdateMessage":
                    for control in self.gui.values():
                        if control["uuid"] == message["uuid"]:
                            control["props"].update(message["updates"])

    def update(self, label, value):
        self.ws.send(
            msgspec.msgpack.encode(
                dict(
                    type="GuiUpdateMessage",
                    uuid=self.gui[label]["uuid"],
                    updates={"value": value},
                )
            )
        )

    def review(self, session):
        options = self.gui["Placement candidate"]["props"]["options"]
        self.assertEqual(self.gui["Placement candidate"]["props"]["value"], options[0])
        self.assertTrue(options[0].startswith("P00"))
        original_environment = self.nodes["/environment"]["points"].copy()
        for option in list(options[1:]) + [options[0]]:
            expected = session.select(option.split(" — ")[0])
            self.update("Placement candidate", option)
            self.wait_for(
                lambda: np.allclose(
                    np.asarray(self.nodes["/object"]["points"]).reshape(-1, 3),
                    expected,
                    atol=2e-5,
                )
            )
        np.testing.assert_array_equal(self.nodes["/environment"]["points"], original_environment)
        self.update("Confirm Placement", True)
        self.wait_for(lambda: (session.root / "selection.json").exists())
        saved = json.loads((session.root / "selection.json").read_text())
        self.assertEqual(saved["candidate_id"], "P00")
        self.assertEqual(saved["relative_rotation"], np.eye(3).tolist())
        self.assertEqual(saved["snapshot_sha256"], sha256(session.root / "snapshot.npz"))
        self.assertFalse((session.root / "export/targets.json").exists())
        self.assertFalse((session.root / "sender").exists())

    def test_cached_candidates_in_existing_8089_viewport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = np.random.default_rng(3).normal(size=(150, 3)) * 0.01 + [0.3, 0.1, 0.2]
            np.savez(
                root / "snapshot.npz",
                object_points_world_m=points,
                environment_points_world_m=np.r_[points, [[0, 0, 0], [1, 1, 0]]],
                object_source_indices=np.arange(len(points)),
            )
            session = PlacementSession(root, points, "web-fixture", dict(snapshot_sha256=sha256(root / "snapshot.npz")))
            session.infer_once(lambda: [candidate(1, [1, 0, 0]), candidate(2, [0, 1, 0])])
            with self.viewer(["ada_grasp_ctrl.pipeline.dual_viewer", "--run", str(root), "--placement-only"]):
                self.assertTrue(self.gui["Run UOP-Net once"]["props"]["disabled"])
                self.review(session)
            self.assertFalse((root / "uop_invocation.json").exists())

    @unittest.skipUnless(
        os.environ.get("ADA_PLACEMENT_OBJECT_ROOT"), "Set ADA_PLACEMENT_OBJECT_ROOT for local UOP smoke"
    )
    def test_real_selected_capture_cli_uop_once_switch_and_confirm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            args = [
                "ada_grasp_ctrl.pipeline.dual_runner",
                "--placement-only",
                "--object-root",
                os.environ["ADA_PLACEMENT_OBJECT_ROOT"],
                "--output-root",
                str(root),
                "--environment",
                "mock",
                "--ros-domain-id",
                "88",
            ]
            if os.environ.get("UOP_NET_ROOT"):
                args += ["--uop-root", os.environ["UOP_NET_ROOT"]]
            with self.viewer(args):
                self.assertFalse(self.gui["Run UOP-Net once"]["props"]["disabled"])
                self.update("Run UOP-Net once", True)
                self.wait_for(
                    lambda: (
                        (root / "uop_result.json").exists()
                        and len(self.gui["Placement candidate"]["props"]["options"]) > 1
                    )
                )
                saved_result = (root / "uop_result.json").read_bytes()
                session = PlacementSession.load(root)
                self.assertEqual(session.inference_status, "completed")
                self.review(session)
                self.assertEqual((root / "uop_result.json").read_bytes(), saved_result)
                self.assertEqual(json.loads(saved_result)["inference_calls"], 1)
            report = json.loads((root / "run_report.json").read_text())
            self.assertEqual(report["status"], "confirmed")
            self.assertFalse(report["execution_started"])


if __name__ == "__main__":
    unittest.main()
