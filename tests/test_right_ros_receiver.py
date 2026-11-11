"""ROS receiver contracts; run with sourced Jazzy system Python for full coverage."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("rclpy"), "Requires sourced ROS Python")
class RosReceiverTest(unittest.TestCase):
    def setUp(self) -> None:
        from ada_grasp_ctrl.ros_execution import bridge, runtime
        from ada_grasp_ctrl.ros_execution.settings import bridge_config

        self.bridge = bridge
        self.runtime = runtime
        self.config = bridge_config()

    def goal(self):
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        return FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(
                joint_names=list(self.bridge.NAMES),
                points=[JointTrajectoryPoint(positions=[0.0] * 28, time_from_start=Duration(sec=2))],
            )
        )

    def test_reorders_all_28_names_without_losing_targets(self) -> None:
        goal = self.goal()
        goal.trajectory.joint_names = self.bridge.NAMES[::-1]
        goal.trajectory.points[0].positions = list(np.arange(28, dtype=float))
        q, duration, extra = self.bridge.unpack_goal(goal)
        np.testing.assert_array_equal(q, np.arange(28)[::-1])
        self.assertEqual((duration, extra), (2, 0))

    def test_rejects_unsupported_goals(self) -> None:
        for fault in ("left", "duplicate", "nan", "multiple", "velocity", "scheduled", "zero", "long", "extra"):
            goal = self.goal()
            point = goal.trajectory.points[0]
            if fault == "left":
                goal.trajectory.joint_names[0] = "left_J1"
            elif fault == "duplicate":
                goal.trajectory.joint_names[0] = goal.trajectory.joint_names[1]
            elif fault == "nan":
                point.positions[0] = float("nan")
            elif fault == "multiple":
                goal.trajectory.points.append(point)
            elif fault == "velocity":
                point.velocities = [0.0] * 28
            elif fault == "scheduled":
                goal.trajectory.header.stamp.sec = 1
            elif fault == "zero":
                point.time_from_start.sec = 0
            elif fault == "long":
                point.time_from_start.sec = 301
            else:
                goal.goal_time_tolerance.sec = 61
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.bridge.unpack_goal(goal)

    def test_duration_respects_model_velocity_and_configured_acceleration(self) -> None:
        model = SimpleNamespace(limits={"arm": (None, None, np.ones(6)), "hand": (None, None, np.ones(22))})
        start, target = np.zeros(28), np.zeros(28)
        for index, group in ((0, "arm"), (6, "hand")):
            target[:] = 0
            target[index] = 1
            self.assertAlmostEqual(
                self.bridge.minimum_duration(start, target, self.config, model), 1.875 / self.config[group + "_speed"]
            )
        model.limits["hand"][2][:] = 0.05
        self.assertAlmostEqual(self.bridge.minimum_duration(start, target, self.config, model), 1.875 / 0.05)
        self.assertEqual(self.bridge.minimum_duration(start, target, self.config, model, include_hand=False), 0)
        target[6] = 0.001
        self.assertAlmostEqual(
            self.bridge.minimum_duration(start, target, self.config, model), np.sqrt(5.774 * 0.001 / 0.8)
        )

    def test_replayed_feedback_does_not_refresh_watchdog(self) -> None:
        from sensor_msgs.msg import JointState

        rt = self.runtime.Runtime.__new__(self.runtime.Runtime)
        rt.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10**10)))
        rt.c, rt.feedback, rt.stamps, rt.fault = self.config, {}, {}, None
        names = self.bridge.ARM_NAMES
        message = JointState(name=names, position=[0.0] * 6)
        message.header.stamp.sec = 10
        with patch.object(self.runtime.time, "monotonic", return_value=100.0):
            rt.receive("arm", names, message)
        with patch.object(self.runtime.time, "monotonic", return_value=101.0):
            rt.receive("arm", names, message)
        self.assertEqual(rt.feedback["arm"].received, 100.0)
        message.header.stamp.sec = 1
        rt.receive("arm", names, message)
        self.assertIn("timestamp", rt.fault)

    def test_receiver_import_does_not_depend_on_bimanual_application(self) -> None:
        program = """
import importlib.abc
import sys
class NoBimanual(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'bimanual_manipulation':
            raise RuntimeError('Receiver depends on the bimanual application')
sys.meta_path.insert(0, NoBimanual())
from ada_grasp_ctrl.ros_execution.bridge import Bridge
"""
        subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)

    def test_unreviewed_hardware_goals_retain_the_path_guard(self) -> None:
        from ada_grasp_ctrl.ros_execution.settings import bridge_config

        rt = self.runtime.Runtime.__new__(self.runtime.Runtime)
        rt.c = bridge_config({"environment": "hardware"})
        rt.description = "fixture"
        rt.arm_description = (
            "<robot><ros2_control><hardware><plugin>fanuc_robot_driver/FanucHardwareInterface</plugin>"
            "</hardware></ros2_control></robot>"
        )
        rt.log = SimpleNamespace(result={})
        rt.fresh = lambda: True
        rt.wait = lambda *args: None
        hand = {"side": "right", "backend": "sharpa_sdk", "read_only": False, "control_mode": "mit"}

        def parameters(node, names):
            if "hand_node" in node:
                return hand
            raise RuntimeError("Reached the remaining controller checks")

        rt.parameters = parameters
        with self.assertRaisesRegex(RuntimeError, "reviewed path"):
            rt.preflight(True)
        rt.c["operator_reviewed_joint_goal"] = True
        with self.assertRaisesRegex(RuntimeError, "remaining controller checks"):
            rt.preflight(True)
        hand["read_only"] = True
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            rt.preflight(True)
        hand["read_only"] = False
        hand["control_mode"] = "position"
        with self.assertRaisesRegex(RuntimeError, "requires MIT"):
            rt.preflight(True)
        rt.arm_description = "<robot/>"
        with self.assertRaisesRegex(RuntimeError, "hardware plugin"):
            rt.preflight(False)

    def test_default_receiver_settings_match_sender_cli(self) -> None:
        import tempfile

        from ada_grasp_ctrl.experiments.right_ros_sender import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "targets.json"
            target.write_text("{}")
            output = root / "preview"
            with (
                patch("sys.argv", ["sender", "--targets", str(target), "--output-root", str(output)]),
                patch(
                    "ada_grasp_ctrl.experiments.right_ros_sender.load_packet",
                    return_value={"sample_id": "test", "joint_names": self.bridge.NAMES, "targets": []},
                ),
                patch("builtins.print"),
            ):
                self.assertEqual(main(), 0)
            args = json.loads((output / "run_manifest.yaml").read_text())["arguments"]
        for key in ("arm_speed", "hand_speed", "arm_acceleration", "hand_acceleration", "feedback_timeout"):
            self.assertEqual(args[key], self.config[key])
        self.assertEqual(args["minimum_duration"], 2.0)


if __name__ == "__main__":
    unittest.main()
