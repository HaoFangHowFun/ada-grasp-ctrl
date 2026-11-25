"""Real 8089 WebSocket lifecycle with explicitly synthetic model results and real dynamics."""

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

import msgspec
import viser
from websockets.sync.client import connect

from ada_grasp_ctrl.paths import project_root
import test_placement_web


class SimulationWebTests(unittest.TestCase):
    def wait_for(self, predicate):
        """Receive actual scene transforms as well as GUI properties from Viser."""
        from test_object_placement_web import receive

        deadline = time.monotonic() + 45
        while not predicate():
            self.assertLess(time.monotonic(), deadline, "Expected simulation UI update")
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
                    for key in ("position", "wxyz"):
                        if key in message:
                            node[key] = message[key]
                if message["type"] == "GuiUpdateMessage":
                    for control in self.gui.values():
                        if control["uuid"] == message["uuid"]:
                            control["props"].update(message["updates"])

    update = test_placement_web.PlacementWebTests.update

    @contextmanager
    def viewer(self, root, *options):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8089)) == 0:
                self.skipTest("User port 8089 occupied")
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        env["PYTHONPATH"] = str(project_root() / "src") + os.pathsep + str(project_root() / "tests")
        bootstrap = """
import sys
from pathlib import Path
class NoROS:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rclpy','rospy'):
            raise AssertionError('Simulation imported ROS')
sys.meta_path.insert(0,NoROS())
from ada_grasp_ctrl.pipeline import dual_runner,simulation_ui
from ada_grasp_ctrl.placement.session import write_json
from test_dual_mujoco_simulation import idle_reference
from test_placement_selection import candidate
from unittest.mock import patch

def fake_plan(config, selector, root, processes, state):
    import time
    time.sleep(1.0)
    # Test fixture only: use a physically compiled stationary reference to prove
    # the UI review/execute/hold/reset mechanism independently of learned success.
    from types import SimpleNamespace
    import numpy as np
    scene = SimpleNamespace(sim=SimpleNamespace(configuration=lambda: np.array(state['full_positions_rad'])))
    plan=idle_reference(scene,['left','right'])
    write_json(root/'simulation_plan.json',plan)
    write_json(root/'test_fixture.json',dict(source='synthetic stationary failure; no model inference'))
    return plan
simulation_ui.prepare_plan=fake_plan
simulation_ui.run_uop=lambda *args: [candidate(1,[1,0,0])]
with patch.object(dual_runner,'validate_domain',side_effect=AssertionError('ROS domain')),
     patch.object(dual_runner,'Processes',side_effect=AssertionError('ROS transport')):
    raise SystemExit(dual_runner.main(sys.argv[1:]))
""".replace(
            "with patch.object(dual_runner,'validate_domain',side_effect=AssertionError('ROS domain')),\n",
            "with patch.object(dual_runner,'validate_domain',side_effect=AssertionError('ROS domain')), \\\n",
        )
        with tempfile.TemporaryFile(mode="w+") as log:
            child = subprocess.Popen(
                [sys.executable, "-c", bootstrap, "--environment", "simulation", "--output-root", str(root), *options],
                env=env,
                stdout=log,
                stderr=log,
                cwd=project_root(),
            )
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    log.seek(0)
                    text = log.read()
                    self.assertIsNone(child.poll(), text)
                    if "PREVIEW_URL=" in text:
                        break
                    time.sleep(0.05)
                else:
                    self.fail("Viewer startup timeout: " + text)
                with connect("ws://127.0.0.1:8089", subprotocols=[f"viser-v{viser.__version__}"], max_size=None) as ws:
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
                    self.wait_for(lambda: "Confirm Scene" in self.gui)
                    yield
            finally:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)

    def state(self, root):
        return json.loads((root / "viewer_state.json").read_text())

    def confirm_scene(self, root, expected):
        self.update("Confirm Scene", True)
        self.wait_for(lambda: self.state(root)["state"] in (expected, "ERROR"))
        state = self.state(root)
        self.assertEqual(state["state"], expected, state)
        return Path(state["attempt"])

    def reset_scene(self, root):
        self.update("Reset Scene", True)
        self.wait_for(lambda: self.state(root)["state"] == "EDITING")
        self.wait_for(lambda: not self.gui["Confirm Scene"]["props"].get("disabled", False))

    def test_preview_execute_hold_reset_same_server_and_passive_editor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            with self.viewer(root):
                self.assertIn("Yaw (deg)", self.gui)
                self.assertIn("Scale", self.gui)
                for removed in ("Roll (deg)", "Pitch (deg)"):
                    self.assertNotIn(removed, self.gui)
                self.update("X (m)", 0.46)
                self.wait_for(lambda: abs(self.nodes.get("/editable_object", {}).get("position", [0])[0] - 0.46) < 1e-5)
                self.assertFalse(list(root.glob("attempt_*")))
                attempt = self.confirm_scene(root, "PREVIEW_READY")
                self.assertEqual(json.loads((attempt / "requested_scene.json").read_text())["xy_m"][0], 0.46)
                self.assertFalse((attempt / "physics.jsonl").exists())
                self.update("Playback (%)", 80)
                time.sleep(0.1)
                self.assertFalse((attempt / "physics.jsonl").exists())
                self.update("Execute Grasp", True)
                self.wait_for(lambda: self.state(root)["state"] in ("HOLDING", "ERROR"))
                self.assertEqual(self.state(root)["state"], "HOLDING", self.state(root))
                report = json.loads((attempt / "execution_report.json").read_text())
                self.assertTrue(report["pipeline_completed"])
                self.assertFalse(report["scheduled_hold_success"])
                self.reset_scene(root)
                second = self.confirm_scene(root, "PREVIEW_READY")
                self.assertNotEqual(attempt, second)
                self.update("Execute Grasp", True)
                self.wait_for(lambda: (second / "physics.jsonl").exists())
                self.reset_scene(root)
                self.assertTrue((second / "cancellation.json").exists())
                self.assertEqual(self.state(root)["port"], 8089)

    def test_placement_p00_once_reset_and_new_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            with self.viewer(root, "--placement-only"):
                attempt = self.confirm_scene(root, "PLACEMENT_READY")
                self.wait_for(lambda: "Confirm Placement" in self.gui)
                self.assertTrue(self.gui["Placement candidate"]["props"]["value"].startswith("P00"))
                self.update("Run UOP-Net once", True)
                self.wait_for(lambda: len(self.gui["Placement candidate"]["props"]["options"]) == 2)
                self.assertTrue(self.gui["Run UOP-Net once"]["props"]["disabled"])
                self.update("Confirm Placement", True)
                self.wait_for(lambda: (attempt / "placement/selection.json").exists())
                self.assertFalse((attempt / "simulation_plan.json").exists())
                self.reset_scene(root)
                second = self.confirm_scene(root, "PLACEMENT_READY")
                self.wait_for(lambda: not self.gui["Run UOP-Net once"]["props"]["disabled"])
                self.assertEqual(len(json.loads((second / "placement/candidates.json").read_text())["candidates"]), 1)
                self.assertEqual(len(list(root.glob("attempt_*"))), 2)
                self.reset_scene(root)

    def test_reset_during_planning_discards_old_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            with self.viewer(root):
                self.update("Confirm Scene", True)
                self.wait_for(lambda: self.state(root)["state"] == "INFERENCE/PLANNING")
                first = Path(self.state(root)["attempt"])
                self.reset_scene(root)
                second = self.confirm_scene(root, "PREVIEW_READY")
                time.sleep(1.1)
                self.assertEqual(Path(self.state(root)["attempt"]), second)
                self.assertNotEqual(first, second)
                self.assertTrue((first / "cancellation.json").exists())
                self.assertFalse((first / "physics.jsonl").exists())
                self.reset_scene(root)

    def test_preview_only_disables_execution(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "run"
            with self.viewer(root, "--preview-only"):
                attempt = self.confirm_scene(root, "PREVIEW_READY")
                self.update("Execute Grasp", True)
                self.wait_for(lambda: self.gui["Execute Grasp"]["props"].get("disabled", False))
                self.assertFalse((attempt / "physics.jsonl").exists())
                self.reset_scene(root)


if __name__ == "__main__":
    unittest.main()
