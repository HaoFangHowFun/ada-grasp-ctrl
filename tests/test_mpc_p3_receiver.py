"""Synthetic P3 handoff with the real receiver/stream loop and fake I/O; no ROS init."""

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("rclpy"), "Requires sourced ROS Python")
class P3ReceiverTests(unittest.TestCase):
    def run_receiver(self, mode, fault=None):
        from builtin_interfaces.msg import Duration, Time
        from control_msgs.action import FollowJointTrajectory
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from rclpy.serialization import serialize_message
        from ada_grasp_ctrl.ros_execution import bridge
        from ada_grasp_ctrl.ros_execution.settings import bridge_config
        from ada_grasp_ctrl.mpc import live_execution
        from ada_grasp_ctrl.mpc.contracts import JointFeedback, MPCProposal

        clock, sent, proposals, records = [10.0], [], [], []
        config = bridge_config()
        config["settle_time"] = 0.02
        names = tuple(bridge.NAMES)
        model = NS(limits={g: ([-3.0] * n, [3.0] * n, [1.0] * n) for g, n in (("arm", 6), ("hand", 22))}, check=Mock())
        handle = Mock(is_cancel_requested=False)
        target = [0.005] * 28
        handle.request = FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(
                joint_names=list(names),
                points=[JointTrajectoryPoint(positions=target, time_from_start=Duration(sec=1))],
            )
        )
        rt = NS(
            c=config,
            description="fixture",
            pubs={},
            JointState=JointState,
            hand_command="/sharpa/right_hand/joint_command",
            abort=Mock(),
            preflight=Mock(),
            watchdog=Mock(),
        )
        rt.feedback = {g: NS(q=np.zeros(n), received=clock[0]) for g, n in (("arm", 6), ("hand", 22))}
        rt.fresh = lambda: fault != "stale"

        def spin(seconds):
            if fault == "cancel":
                handle.is_cancel_requested = True
            clock[0] += 0.01
            for row in rt.feedback.values():
                row.received = clock[0]

        rt.spin = spin
        rt.wait = lambda predicate, label: self.assertTrue(predicate(), label)

        def publisher(topic):
            group = "arm" if "crx" in topic else "hand"

            def publish(msg):
                sent.append((group, serialize_message(msg)))
                rt.feedback[group].q = np.asarray(msg.position)

            return NS(publish=publish, get_subscription_count=lambda: 1)

        rt.node = NS(
            create_publisher=lambda cls, topic, qos: publisher(topic),
            destroy_node=Mock(),
            count_publishers=lambda topic: int(fault == "duplicate"),
            get_clock=lambda: NS(
                now=lambda: NS(to_msg=lambda: Time(sec=int(clock[0]), nanosec=int((clock[0] % 1) * 1e9)))
            ),
        )

        def sample():
            q = tuple(rt.feedback["arm"].q) + tuple(rt.feedback["hand"].q)
            return JointFeedback(names, q, clock[0], clock[0], "synthetic"), {}

        feed = NS(frames={str(i): {} for i in range(5)}, sample=sample)

        class Worker:
            pending = None
            count = 0
            close = Mock()

            def submit_tactile(self, now, feedback, command, tactile, stage):
                self.pending = now

            def poll(self):
                if self.pending is None:
                    return None
                now, self.pending = self.pending, None
                self.count += 1
                if self.count == 1:
                    result = MPCProposal(names, tuple(target), True, now, 0.001, "synthetic")
                else:
                    reason = "native_timeout" if fault == "timeout" else "native_complete"
                    if fault == "solver":
                        reason = "solver_rejected"
                    result = MPCProposal(
                        names, None, False, now, 0.001, "synthetic", dict(termination=reason, reason=reason)
                    )
                proposals.append(result)
                return result

        worker = Worker()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            log = NS(path=Path(tmp), result={}, finish=Mock(), event=lambda kind, **kw: records.append((kind, kw)))
            passive = (
                NS(observe=Mock(side_effect=RuntimeError("synthetic passive failure")))
                if mode == "mpc_passive"
                else None
            )
            instance = NS(
                side="right",
                names=list(names),
                config=config,
                args=NS(output_dir=tmp, hardware_enable=False),
                packet=dict(
                    environment="mock",
                    kinematics={},
                    targets=[dict(phase=p) for p in ("grasp", "squeeze", "lift")],
                    reviewed_plan=dict(keyframes=[[0.0] * 28] * 4),
                ),
                mpc_mode=mode,
                phase_index=0,
                passive_observer=passive,
                stopping=threading.Event(),
                lock=threading.Lock(),
                busy=True,
                get_logger=lambda: Mock(),
            )
            for module, name, value in (
                (bridge, "Runtime", lambda *a: rt),
                (bridge, "Model", lambda *a: model),
                (bridge, "RunLog", lambda *a: log),
                (bridge, "check_reviewed_goal", Mock()),
                (bridge, "check_kinematics", Mock()),
                (bridge, "kinematic_contract", Mock()),
                (live_execution, "LiveFeed", lambda *a: feed),
                (live_execution, "create_worker", lambda *a: worker),
            ):
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.object(bridge.time, "monotonic", lambda: clock[0]))
            result = bridge.Bridge.execute_goal(instance, handle)
        return result, sent, proposals, instance, rt, handle, records

    def test_passive_failure_cannot_change_baseline_command_bytes(self):
        baseline = self.run_receiver("linear")
        passive = self.run_receiver("mpc_passive")
        self.assertEqual(baseline[0].error_code, 0)
        self.assertEqual(passive[0].error_code, 0)
        self.assertGreater(len(baseline[1]), 0)
        self.assertEqual(baseline[1], passive[1])
        self.assertEqual(passive[2], [])
        passive[3].passive_observer.observe.assert_called()

    def test_synthetic_native_complete_skips_combined_episode_only(self):
        result, sent, proposals, instance, rt, handle, records = self.run_receiver("mpc_active")
        self.assertEqual(result.error_code, 0, result.error_string)
        self.assertTrue(result.error_string.startswith("Ours native_complete:"))
        self.assertEqual(instance.phase_index, 2)
        self.assertGreater(len(sent), 0)
        self.assertEqual(len(proposals), 2)
        self.assertFalse(instance.stopping.is_set())
        rt.abort.assert_not_called()
        commands = [r[1]["q_100Hz"] for r in records if r[0] == "mpc_command"]
        for a, b in zip(commands, commands[1:]):
            self.assertLessEqual(max(abs(x - y) for x, y in zip(a, b)), 0.004 + 1e-9)

    def test_faults_never_advance_to_lift(self):
        for fault in ("timeout", "solver", "duplicate", "cancel", "stale"):
            with self.subTest(fault=fault):
                result, sent, _, instance, rt, handle, _ = self.run_receiver("mpc_active", fault)
                self.assertNotEqual(result.error_code, 0)
                self.assertEqual(instance.phase_index, 0)
                self.assertTrue(instance.stopping.is_set())
                rt.abort.assert_called_once()
                handle.succeed.assert_not_called()
                if fault in ("duplicate", "cancel", "stale"):
                    self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
