"""Synthetic sender/receiver dynamic handoff; mocked I/O, no ROS initialization."""

from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("rclpy") and os.environ.get("ADA_DYNAMIC_LIFT_REVIEW"),
    "Requires sourced ROS and offline recorded-state review",
)
class DynamicROSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ada_grasp_ctrl.experiments.right_ros_sender import load_packet

        cls.path = Path(os.environ["ADA_DYNAMIC_LIFT_PACKET"])
        cls.packet = load_packet(cls.path)
        cls.handoff = json.loads(Path(os.environ["ADA_DYNAMIC_LIFT_REVIEW"]).read_text())
        from ada_grasp_ctrl.pipeline import dynamic_lift

        # Rebase only synthetic receipt clocks; all recorded measured joint values stay intact.
        cls.handoff["snapshot"].update(captured_at=100.0, received_at=dict(arm=99.99, hand=99.98))
        cls.handoff["review_id"] = dynamic_lift.fingerprint({k: v for k, v in cls.handoff.items() if k != "review_id"})

    def request(self, handoff=None):
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from ada_grasp_ctrl.pipeline import dynamic_lift

        h = handoff or self.handoff
        s = h["segments"]["lift"]
        seconds = int(s["duration_s"])
        request = FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(
                joint_names=self.packet["joint_names"],
                points=[
                    JointTrajectoryPoint(
                        positions=s["positions"],
                        time_from_start=Duration(sec=seconds, nanosec=int((s["duration_s"] - seconds) * 1e9)),
                    )
                ],
            )
        )
        request.trajectory.header.frame_id = dynamic_lift.PREFIX + json.dumps(h)
        return request

    def receiver(self, fault=None):
        from sensor_msgs.msg import JointState
        from builtin_interfaces.msg import Time
        from ada_grasp_ctrl.ros_execution import bridge

        config = self.packet["reviewed_plan"]["receiver_settings"]
        clock, sent, records = [101.0], [], []
        actual = self.handoff["snapshot"]["positions"].copy()
        if fault == "start":
            actual[0] += 0.02
        instance = NS(
            side="right",
            names=self.packet["joint_names"],
            config=config,
            packet=self.packet,
            packet_path=self.path,
            mpc_mode="mpc_active",
            native_completed_at=99.0,
            dynamic_handoff=None,
            phase_index=self.packet["reviewed_plan"]["phases"].index("lift"),
            passive_observer=None,
            stopping=threading.Event(),
            lock=threading.Lock(),
            busy=False,
            get_logger=lambda: Mock(),
        )
        if fault == "native":
            instance.native_completed_at = 0.0
        request = self.request()
        if fault == "expired":
            clock[0] = 121.0
        if fault == "phase":
            instance.phase_index += 1
        if fault == "tampered":
            from ada_grasp_ctrl.pipeline import dynamic_lift

            changed = deepcopy(self.handoff)
            changed["segments"]["lift"]["positions"][0] += 0.02
            changed["review_id"] = dynamic_lift.fingerprint({k: v for k, v in changed.items() if k != "review_id"})
            request = self.request(changed)
        if fault == "review":
            request.trajectory.header.frame_id = ""
        handle = Mock(request=request, is_cancel_requested=fault == "cancel")
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
        rt.feedback = {
            g: NS(q=np.array(actual[sl]), received=clock[0]) for g, sl in (("arm", slice(0, 6)), ("hand", slice(6, 28)))
        }
        rt.fresh = lambda: fault != "stale"
        rt.wait = lambda predicate, label: self.assertTrue(predicate(), label)

        def spin(seconds):
            clock[0] += 0.01
            for sample in rt.feedback.values():
                sample.received = clock[0]

        rt.spin = spin

        def publisher(topic):
            group = "arm" if "crx" in topic else "hand"

            def publish(message):
                sent.append((group, list(message.position)))
                rt.feedback[group].q = np.array(message.position)

            return NS(publish=publish, get_subscription_count=lambda: 1)

        rt.node = NS(
            create_publisher=lambda cls, topic, qos: publisher(topic),
            destroy_node=Mock(),
            get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: Time(sec=int(clock[0])))),
        )
        model = NS(limits={g: ([-5.0] * n, [5.0] * n, [1.0] * n) for g, n in (("arm", 6), ("hand", 22))}, check=Mock())
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            plan_path = Path(tmp) / "trusted-plan.json"
            trusted = deepcopy(self.handoff)
            if fault == "path":
                trusted["validation"]["sampled_known_geometry"] = False
            plan_path.write_text(json.dumps(dict(ok=True, handoff=trusted)))
            instance.args = NS(output_dir=tmp, mpc_python="unused", dynamic_lift_plan=str(plan_path))
            log = NS(path=Path(tmp), result={}, finish=Mock(), event=lambda kind, **kw: records.append((kind, kw)))

            for name, value in (
                ("Runtime", lambda *a: rt),
                ("Model", lambda *a: model),
                ("RunLog", lambda *a: log),
                ("load_packet", lambda *a: self.packet),
                ("check_kinematics", Mock()),
                ("kinematic_contract", Mock()),
            ):
                stack.enter_context(patch.object(bridge, name, value))
            planner = stack.enter_context(
                patch.object(bridge.dynamic_lift, "compute", side_effect=AssertionError("Receiver must not plan again"))
            )
            stack.enter_context(patch.object(bridge.time, "monotonic", lambda: clock[0]))
            accepted = bridge.Bridge.accept_goal(instance, request)
            result = bridge.Bridge.execute_goal(instance, handle) if accepted == bridge.GoalResponse.ACCEPT else None
            planner.assert_not_called()
        return accepted, result, sent, records, instance

    def test_receiver_accepts_new_start_and_holds_hand(self):
        from ada_grasp_ctrl.ros_execution.bridge import GoalResponse

        accepted, result, sent, records, instance = self.receiver()
        self.assertEqual(accepted, GoalResponse.ACCEPT)
        self.assertEqual(result.error_code, 0, result.error_string)
        self.assertEqual(instance.dynamic_handoff, self.handoff)
        self.assertTrue(sent)
        for group, q in sent:
            if group == "hand":
                np.testing.assert_allclose(q, self.handoff["snapshot"]["positions"][6:], atol=1e-14)
        self.assertTrue(any(kind == "dynamic_start_check" for kind, _ in records))

    def test_receiver_rejections_never_publish(self):
        for fault in ("native", "review", "start", "path", "cancel", "stale", "expired", "phase", "tampered"):
            with self.subTest(fault=fault):
                _, result, sent, _, _ = self.receiver(fault)
                self.assertEqual(sent, [])
                if result is not None:
                    self.assertNotEqual(result.error_code, 0)

    def sender(self, mode="mpc_active", fault=None):
        import rclpy
        import rclpy.action
        from control_msgs.action import FollowJointTrajectory
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from ada_grasp_ctrl.experiments import right_ros_sender as sender
        from ada_grasp_ctrl.pipeline import dynamic_lift

        config = self.packet["reviewed_plan"]["receiver_settings"]
        args = NS(
            environment="hardware",
            controller_mode=mode,
            minimum_duration=2.0,
            startup_timeout=2.0,
            result_margin=1.0,
            mpc_python="unused",
            targets=str(self.path),
            **{
                k: config[k]
                for k in ("arm_speed", "hand_speed", "arm_acceleration", "hand_acceleration", "feedback_timeout")
            },
        )
        clock, q, sent, callbacks = [100.0], self.packet["reviewed_plan"]["keyframes"][0].copy(), [], {}
        node = Mock()
        node.count_publishers.return_value = 1
        node.get_clock.return_value.now.side_effect = lambda: NS(nanoseconds=int(clock[0] * 1e9))

        def emit():
            for group, (names, callback) in callbacks.items():
                values = q[:6] if group == "arm" else q[6:]
                msg = JointState(name=names, position=values)
                msg.header.stamp.sec = int(clock[0])
                msg.header.stamp.nanosec = int((clock[0] % 1) * 1e9)
                callback(msg)

        def subscribe(cls, topic, callback, qos):
            if cls is String:
                callback(String(data="fixture"))
            else:
                group = "arm" if "crx" in topic else "hand"
                callbacks[group] = (
                    self.packet["joint_names"][:6] if group == "arm" else self.packet["joint_names"][6:],
                    callback,
                )
                emit()

        node.create_subscription.side_effect = subscribe

        def spin(*args, **kw):
            clock[0] += 0.01
            emit()

        def future(value):
            return NS(done=lambda: True, result=lambda: value)

        def send(goal, **kw):
            sent.append(goal)
            position = list(goal.trajectory.points[0].positions)
            phase = self.packet["targets"][len(sent) - 1]["phase"]
            # Active combined Grasp skips the independent Squeeze target.
            if mode == "mpc_active" and len(sent) >= 4:
                phase = self.packet["targets"][len(sent)]["phase"]
            q[:] = self.handoff["snapshot"]["positions"] if mode == "mpc_active" and phase == "grasp" else position
            spin()
            if phase == "lift" and fault == "reject":
                return future(NS(accepted=False))
            message = "Ours native_complete: synthetic" if phase == "grasp" and mode == "mpc_active" else "success"
            code = 0
            if phase == "grasp" and fault in ("timeout", "abort", "invalid_outcome"):
                message = fault
                code = -1 if fault != "invalid_outcome" else 0
            result = FollowJointTrajectory.Result(error_code=code, error_string=message)
            response = NS(status=4 if code == 0 else 6, result=result)
            return future(NS(accepted=True, get_result_async=lambda: future(response)))

        client = Mock()
        client.server_is_ready.return_value = True
        client.send_goal_async.side_effect = send

        def compute(python, packet, snapshot, root, spin):
            if fault == "path":
                raise ValueError("invalid path")
            h = deepcopy(self.handoff)
            h["snapshot"] = snapshot
            h["review_id"] = dynamic_lift.fingerprint({k: v for k, v in h.items() if k != "review_id"})
            if fault == "start":
                q[0] += 0.02
            spin()
            return h

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, ROS_DOMAIN_ID="87"))
            for name, value in (
                ("init", Mock()),
                ("try_shutdown", Mock()),
                ("create_node", lambda *a: node),
                ("spin_once", spin),
            ):
                stack.enter_context(patch.object(rclpy, name, value))
            stack.enter_context(patch.object(rclpy.action, "ActionClient", return_value=client))
            stack.enter_context(patch.object(sender, "check_kinematics", Mock()))
            stack.enter_context(patch.object(sender, "kinematic_contract", Mock()))
            stack.enter_context(patch.object(sender.time, "monotonic", lambda: clock[0]))
            stack.enter_context(patch.object(dynamic_lift, "compute", compute))
            stack.enter_context(patch("builtins.print"))
            report = dict(phases=[])
            if fault:
                with self.assertRaises((ValueError, RuntimeError)):
                    sender.execute(self.packet, args, Path(tmp), report)
            else:
                sender.execute(self.packet, args, Path(tmp), report)
                self.assertTrue(report["task_completed"])
        return sent

    def test_sender_dynamic_sequence_and_legacy(self):
        from ada_grasp_ctrl.pipeline import dynamic_lift

        active = self.sender()
        self.assertEqual(len(active), len(self.packet["targets"]) - 1)
        h = dynamic_lift.decode(active[3].trajectory.header.frame_id)
        self.assertEqual(
            active[3].trajectory.points[0].positions.tolist()
            if hasattr(active[3].trajectory.points[0].positions, "tolist")
            else list(active[3].trajectory.points[0].positions),
            h["segments"]["lift"]["positions"],
        )

        baseline, passive = self.sender("linear"), self.sender("mpc_passive")
        self.assertEqual(baseline, passive)
        self.assertTrue(all(not g.trajectory.header.frame_id for g in baseline))

    def test_sender_failures_no_lift(self):
        for fault in ("timeout", "abort", "invalid_outcome", "path", "start"):
            with self.subTest(fault=fault):
                self.assertEqual(len(self.sender(fault=fault)), 3)
        self.assertEqual(len(self.sender(fault="reject")), 4)


if __name__ == "__main__":
    unittest.main()
