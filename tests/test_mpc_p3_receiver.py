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
                if fault == "publish_" + group and len(sent) >= 2:
                    raise RuntimeError("synthetic " + group + " publish failure")
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

    def test_passive_failure_cannot_change_baseline_command_fields(self):
        baseline = self.run_receiver("linear")
        passive = self.run_receiver("mpc_passive")
        self.assertEqual(baseline[0].error_code, 0)
        self.assertEqual(passive[0].error_code, 0)
        self.assertGreater(len(baseline[1]), 0)
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import JointState

        self.assertEqual(len(baseline[1]), len(passive[1]))
        for (group_a, raw_a), (group_b, raw_b) in zip(baseline[1], passive[1]):
            self.assertEqual(group_a, group_b)
            a, b = deserialize_message(raw_a, JointState), deserialize_message(raw_b, JointState)
            # CDR padding is unspecified; compare every field and exact numeric bytes.
            self.assertEqual(a, b)
            for field in ("position", "velocity", "effort"):
                self.assertEqual(getattr(a, field).tobytes(), getattr(b, field).tobytes())
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
        for kind, row in records:
            if kind == "mpc_command":
                alpha = min(1.0, max(0.0, (row["sampled_at"] - row["segment_started"]) / 0.1))
                np.testing.assert_allclose(
                    row["q_100Hz"],
                    (1 - alpha) * np.asarray(row["q_segment_start"]) + alpha * np.asarray(row["q_mpc"]),
                    atol=1e-14,
                )
                self.assertEqual(row["q_100Hz"], row["q_last_published"])
        self.assertTrue(commands)
        from sensor_msgs.msg import JointState
        from rclpy.serialization import deserialize_message

        for arm, hand in zip(sent[::2], sent[1::2]):
            a, h = deserialize_message(arm[1], JointState), deserialize_message(hand[1], JointState)
            self.assertEqual(a.header.stamp, h.header.stamp)
            self.assertEqual(a.name + h.name, list(instance.names))

    def test_faults_never_advance_to_lift(self):
        for fault in ("timeout", "solver", "duplicate", "cancel", "stale", "publish_arm", "publish_hand"):
            with self.subTest(fault=fault):
                result, sent, _, instance, rt, handle, records = self.run_receiver("mpc_active", fault)
                self.assertNotEqual(result.error_code, 0)
                self.assertEqual(instance.phase_index, 0)
                self.assertTrue(instance.stopping.is_set())
                rt.abort.assert_called_once()
                handle.succeed.assert_not_called()
                if fault in ("duplicate", "cancel", "stale"):
                    self.assertEqual(sent, [])
                if fault.startswith("publish_"):
                    failures = [row for kind, row in records if kind == "mpc_publish_failed"]
                    self.assertEqual(len(failures), 1)
                    good = [row for kind, row in records if kind == "mpc_command"]
                    self.assertEqual(failures[0]["q_last_published"], good[-1]["q_last_published"])
                    self.assertEqual(failures[0]["published_groups"], ["arm"] if fault == "publish_hand" else [])
                    self.assertEqual(len(sent), 3 if fault == "publish_hand" else 2)

    def test_receiver_fault_reaches_sender_without_squeeze_or_lift(self):
        import rclpy
        import rclpy.action
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from ada_grasp_ctrl.experiments import right_ros_sender as sender
        from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES
        from ada_grasp_ctrl.ros_execution.settings import bridge_config

        for fault in ("publish_hand", "timeout", "solver", "cancel"):
            result = self.run_receiver("mpc_active", fault)[0]
            config = bridge_config()
            packet = dict(
                kind="ada_right_joint_targets",
                environment="mock",
                side="right",
                kinematics={},
                source=dict(model_sha256="fixture"),
                targets=[dict(phase=p, positions=[0.005] * 28) for p in ("grasp", "squeeze", "lift")],
                reviewed_plan=dict(
                    phases=["grasp", "squeeze", "lift"],
                    durations_s=[1.0, 1.0, 1.0],
                    keyframes=[[0.0] * 28] * 4,
                    receiver_settings=config,
                ),
            )
            args = NS(
                environment="mock",
                controller_mode="mpc_active",
                minimum_duration=2.0,
                startup_timeout=1.0,
                result_margin=1.0,
                **{
                    k: config[k]
                    for k in ("arm_speed", "hand_speed", "arm_acceleration", "hand_acceleration", "feedback_timeout")
                },
            )

            def future(value):
                return NS(done=lambda: True, result=lambda: value)

            response = NS(status=6, result=result)
            client = Mock()
            client.server_is_ready.return_value = True
            client.send_goal_async.return_value = future(NS(accepted=True, get_result_async=lambda: future(response)))
            node = Mock()
            node.get_clock.return_value.now.return_value.nanoseconds = 10**10

            def subscribe(cls, topic, callback, qos):
                if cls is String:
                    callback(String(data="fixture"))
                else:
                    names = JOINT_NAMES[:6] if "crx" in topic else JOINT_NAMES[6:]
                    msg = JointState(name=list(names), position=[0.0] * len(names))
                    msg.header.stamp.sec = 10
                    callback(msg)

            node.create_subscription.side_effect = subscribe
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                stack.enter_context(patch.dict("os.environ", ROS_DOMAIN_ID="199"))
                stack.enter_context(patch.object(rclpy, "init"))
                stack.enter_context(patch.object(rclpy, "try_shutdown"))
                stack.enter_context(patch.object(rclpy, "create_node", return_value=node))
                stack.enter_context(patch.object(rclpy.action, "ActionClient", return_value=client))
                stack.enter_context(patch.object(sender, "check_kinematics"))
                stack.enter_context(patch.object(sender, "kinematic_contract"))
                stack.enter_context(patch("builtins.print"))
                report = dict(phases=[])
                with self.assertRaisesRegex(RuntimeError, "grasp:"):
                    sender.execute(packet, args, Path(tmp), report)
                self.assertEqual(report["phases"][0]["message"], result.error_string)
            self.assertEqual(client.send_goal_async.call_count, 1)


if __name__ == "__main__":
    unittest.main()
