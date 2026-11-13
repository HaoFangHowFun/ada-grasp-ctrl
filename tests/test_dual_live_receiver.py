"""ROS-only side routing and passive state regressions."""

import importlib.util
from types import SimpleNamespace
import unittest
import numpy as np
from ada_grasp_ctrl.experiments.right_ros_contract import side_names


@unittest.skipUnless(importlib.util.find_spec("rclpy"), "Requires sourced ROS Python")
class DualLiveReceiver(unittest.TestCase):
    def test_left_receiver_reorders_names_and_rejects_right_goal(self):
        from ada_grasp_ctrl.ros_execution.bridge import unpack_goal
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration

        names = list(sum(side_names("left"), ()))
        goal = FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(
                joint_names=names[::-1],
                points=[
                    JointTrajectoryPoint(positions=list(np.arange(28, dtype=float)), time_from_start=Duration(sec=2))
                ],
            )
        )
        target, _, _ = unpack_goal(goal, "left")
        np.testing.assert_array_equal(target, np.arange(28)[::-1])
        with self.assertRaises(ValueError):
            unpack_goal(goal, "right")

    def test_passive_feedback_drift_is_detected(self):
        from ada_grasp_ctrl.ros_execution.runtime import Runtime

        runtime = Runtime.__new__(Runtime)
        runtime.c = dict(side="left", passive_positions_rad=[0.0] * 28)
        runtime.fresh = lambda: True
        runtime.feedback = dict(
            passive_arm=SimpleNamespace(q=np.zeros(6)), passive_hand=SimpleNamespace(q=np.zeros(22))
        )
        runtime.check_passive()
        self.assertEqual(runtime.hand_command, "/sharpa/left_hand/joint_command")
        self.assertEqual(runtime.manager, "/crx5ia/left/controller_manager")
        runtime.feedback["passive_arm"].q[0] = 0.011
        with self.assertRaisesRegex(ValueError, "moved"):
            runtime.check_passive()
        runtime.fresh = lambda: False
        with self.assertRaisesRegex(RuntimeError, "expired"):
            runtime.check_passive()


if __name__ == "__main__":
    unittest.main()
