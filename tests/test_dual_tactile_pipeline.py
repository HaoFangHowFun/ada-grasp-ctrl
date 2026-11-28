"""Original-viewer passive bridge: measured pose, side routing and fail-closed display."""

from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ada_grasp_ctrl.contact.force_display import ForceOverlay, ForceSnapshot, normal_load_totals
from ada_grasp_ctrl.pipeline.dual_tactile import DualTactile, display_frame, measured_dual, start_reader
from ada_grasp_ctrl.pipeline.telemetry import full_feedback
import test_dual_tactile_adapter as adapter_tests
from test_fingertip_forces import NS
from test_pipeline_dashboard import fixture


class DualBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        adapter_tests.DualTactileTests.setUpClass()

    def setUp(self):
        self.data = adapter_tests.DualTactileTests()
        self.data.setUp()
        self.packets, self.readers = {}, {}
        for side in ("left", "right"):
            packet = fixture()
            names = self.data.robot.joint_names[:28] if side == "left" else self.data.robot.joint_names[28:]
            packet["arm"]["names"] = names[:6]
            packet["joints"] = self.data.joints[side]
            packet["frames"] = {f: asdict(row) for f, row in self.data.frames[side].items()}
            self.packets[side] = packet
            self.readers[side] = Mock()
            self.readers[side].snapshot.return_value = (packet, True, "")
        self.bridge = DualTactile(self.data.robot, self.data.adapters, self.readers, clock=lambda: self.data.now)

    def test_both_hands_measured_force_and_reader_loss(self):
        q, forces, report = self.bridge.sample()
        self.assertEqual(q.shape, (56,))
        self.assertEqual({r.status for r in forces.fingers}, {"VALID"})
        self.assertEqual(normal_load_totals(forces), (None, None))
        self.assertEqual(set(report["fingers"]), {"left", "right"})
        self.readers["left"].snapshot.return_value = (self.packets["left"], False, "exited")
        q, forces, report = self.bridge.sample()
        self.assertIsNone(q)
        self.assertTrue(all(r.force is None for r in forces.fingers))
        self.assertIn("left", report["errors"])
        self.bridge.close()
        for reader in self.readers.values():
            reader.close.assert_called_once()

    def test_ambiguous_model_wrong_side_stale_and_cross_side_skew(self):
        for mutation in (
            lambda p: p["left"].update(model_valid=False),
            lambda p: p["left"]["joint_sources"].update(hand=2),
            lambda p: p["left"].update(joints=p["right"]["joints"]),
            lambda p: p["left"]["arm"].update(received=9.0),
            lambda p: [p["left"][g].update(received=9.8) for g in ("arm", "joints")],
        ):
            packets = deepcopy(self.packets)
            mutation(packets)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                measured_dual(packets, dict(left=True, right=True), self.data.robot, 10.0)
        self.assertEqual(full_feedback(self.packets["right"], 10.0), [0] * 28)
        self.data.now = 10.6
        q, forces, report = self.bridge.sample()
        self.assertIsNone(q)
        self.assertTrue(all(r.force is None for r in forces.fingers))

    def test_playback_and_dropout_cannot_display_live_forces(self):
        measured, forces, _ = self.bridge.sample()
        planned, previous = np.ones(56), np.ones(56) * 2
        q, visible = display_frame(planned, previous, measured, forces, live=True)
        np.testing.assert_array_equal(q, measured)
        self.assertIs(visible, forces)
        q, visible = display_frame(planned, previous, measured, forces, live=False)
        np.testing.assert_array_equal(q, planned)
        self.assertIsNone(visible)
        q, visible = display_frame(planned, previous, None, forces, live=True)
        np.testing.assert_array_equal(q, previous)
        self.assertIsNone(visible)

    def test_readers_use_existing_side_telemetry_and_explicit_domain(self):
        config = dict(
            ros_setup="/ros/setup",
            workspace_setup="/ws/setup",
            ros_python="/usr/bin/python3",
            robot_snapshot="output/test",
            ros_domain_id=87,
        )
        with patch("ada_grasp_ctrl.pipeline.dual_tactile.RosProcess") as process:
            for side in ("left", "right"):
                start_reader(config, side)
                kw = process.call_args.kwargs
                command = kw["command"]
                self.assertEqual(command[command.index("--side") + 1], side)
                self.assertIn("ada_grasp_ctrl.pipeline.telemetry", command)
                self.assertEqual(kw["environment"]["ROS_DOMAIN_ID"], "87")
                self.assertNotIn("LD_LIBRARY_PATH", kw["environment"])

    def test_hardware_clock_in_existing_force_hud(self):
        # Exercise the shared display update without constructing a browser server.
        overlay = ForceOverlay.__new__(ForceOverlay)
        overlay.root = NS(visible=False)
        overlay.hud = NS(visible=False, content="")
        overlay.readout = NS(content="")
        overlay.enabled = NS(value=True)
        overlay.groups = {}
        snapshot = ForceSnapshot(10, (), timestamp_clock="Hardware receipt monotonic")
        overlay.update(snapshot)
        self.assertIn("Hardware receipt monotonic", overlay.hud.content)
        self.assertNotIn("Simulation", overlay.hud.content)


if __name__ == "__main__":
    unittest.main()
