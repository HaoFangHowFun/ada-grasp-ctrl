"""Offline mode, named-state, rate, ownership and fault contracts."""

from dataclasses import replace
import unittest

import numpy as np

from ada_grasp_ctrl.contact import ContactObservation
from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, PHASES
from ada_grasp_ctrl.mpc.contracts import JointFeedback, MPCProposal, controller_mode
from ada_grasp_ctrl.mpc.orchestration import StageOrchestrator
from ada_grasp_ctrl.mpc.reference import LinearReference, linear_reference
from ada_grasp_ctrl.mpc.baseline_stream import BaselineStream
from ada_grasp_ctrl.mpc.streaming import CommandOwnership, GuardedStream
from ada_grasp_ctrl.pipeline.plan import sample_plan, validate_plan
from ada_grasp_ctrl.ros_execution.settings import bridge_config
from test_grasp_pipeline import plan_fixture


def feedback(now: float, q: tuple = (0.0,) * 28) -> JointFeedback:
    return JointFeedback(tuple(JOINT_NAMES), q, now, now, "synthetic_mock")


def proposal(now: float, q: tuple = (0.1,) * 28) -> MPCProposal:
    return MPCProposal(tuple(JOINT_NAMES), q, True, now, 0, "valid_empty")


def stream() -> GuardedStream:
    return GuardedStream()


class ReferenceAndModeTests(unittest.TestCase):
    def test_baseline_default_and_hardware_gate(self):
        self.assertEqual(controller_mode({"environment": "hardware"}), "baseline")
        for mode in ("linear", "mpc_passive", "mpc_active"):
            self.assertEqual(controller_mode({"environment": "mock"}, mode), mode)
            with self.assertRaises(ValueError):
                controller_mode({"environment": "hardware"}, mode)
        with self.assertRaises(ValueError):
            controller_mode({"controller": {"hardware_enable": True}})

    def test_exact_linear_all_joints_and_preview_parity(self):
        a, b = np.linspace(-0.3, 0.3, 28), np.linspace(0.3, -0.3, 28)
        ref = LinearReference(a, b, 1.0, 2.0, (-1.0,) * 28, (1.0,) * 28)
        plan = plan_fixture()
        plan.update(
            controller_mode="linear",
            interpolation="linear",
            keyframes=[a.tolist(), b.tolist(), b.tolist(), b.tolist()],
            receiver_settings=bridge_config({"interpolation": "linear"}),
        )
        plan["durations_s"][0] = 2.0
        validate_plan(plan)
        total = sum(plan["durations_s"]) + 3 * plan["settle_time_s"]
        for elapsed in (0.0, 0.25, 1.0, 1.75, 2.0):
            expected = a + elapsed / 2 * (b - a)
            np.testing.assert_array_equal(ref.sample(1 + elapsed), expected)
            np.testing.assert_allclose(sample_plan(plan, elapsed / total * 100), expected, atol=1e-15)
        np.testing.assert_array_equal(ref.sample(100), b)
        with self.assertRaisesRegex(ValueError, "backward"):
            ref.sample(2)

    def test_bad_reference_and_limits_rejected(self):
        for args in (([0], [1], 0, 0), ([0], [float("nan")], 0, 1), ([0], [1, 2], 0, 1)):
            with self.assertRaises(ValueError):
                linear_reference(*args)
        with self.assertRaises(ValueError):
            LinearReference([0], [2], 0, 1, [-1], [1])

    def test_named_boundaries_reject_missing_duplicate_reordered(self):
        for names in (JOINT_NAMES[:-1], JOINT_NAMES[::-1], (JOINT_NAMES[0],) * 28):
            with self.assertRaises(ValueError):
                JointFeedback(tuple(names), (0.0,) * 28, 0, 0, "mock")
        with self.assertRaises(ValueError):
            proposal(0, (float("nan"),) * 28)

    def test_legacy_plan_targets_and_samples_unchanged(self):
        plan = plan_fixture()
        validate_plan(plan)
        total = sum(plan["durations_s"]) + 3 * plan["settle_time_s"]
        q = sample_plan(plan, 0.25 * plan["durations_s"][0] / total * 100)
        np.testing.assert_allclose(q, [0.0103515625] * 28, atol=1e-16)
        self.assertEqual(tuple(plan["phases"]), PHASES)
        self.assertNotIn("interpolation", plan["receiver_settings"])

    def test_no_linear_hardware_or_preview_execution_mismatch(self):
        with self.assertRaises(ValueError):
            bridge_config({"environment": "hardware", "interpolation": "linear"})
        plan = plan_fixture()
        plan["interpolation"] = "linear"
        with self.assertRaises(ValueError):
            validate_plan(plan)


class StageTests(unittest.TestCase):
    def test_passive_only_grasp_squeeze_without_publisher_capability(self):
        calls, logs = [], []

        def solve(now, fb, cmd, ref, observation, *, stage):
            calls.append(stage)
            return proposal(now)

        runner = StageOrchestrator("mpc_passive", PHASES, solve, logs.append)
        now = 0
        for i, phase in enumerate(PHASES):
            runner.enter(phase, previous_arrived=i > 0)
            for _ in range(30):
                now = round(now + 0.01, 6)
                runner.tick(now, feedback(now), (0.0,) * 28, (0.2,) * 28, ContactObservation(now, (), "mock"))
        self.assertEqual(calls, ["grasp"] * 3 + ["squeeze"] * 3)
        self.assertTrue(all(row["owner"] == "baseline" and not row["proposal_executed"] for row in logs))
        self.assertFalse(hasattr(runner, "publish"))

    def test_passive_computation_exception_does_not_abort_baseline(self):
        def fail(*args, **kwargs):
            raise RuntimeError("injected solver crash")

        runner = StageOrchestrator("mpc_passive", PHASES, fail, lambda row: None)
        runner.enter("pregrasp")
        runner.enter("grasp", previous_arrived=True)
        result = runner.tick(0, feedback(0), (0.0,) * 28, (0.0,) * 28, None)
        self.assertFalse(result.valid)
        self.assertFalse(runner.failed)
        runner.enter("squeeze", previous_arrived=True)

    def test_sequence_arrival_cancel_stale_fail_closed(self):
        for fault in ("sequence", "arrival", "cancel", "stale", "clock"):
            logs = []
            runner = StageOrchestrator("mpc_passive", PHASES, lambda *a, **k: proposal(0), logs.append)
            runner.enter("pregrasp")
            with self.assertRaises(ValueError):
                if fault == "sequence":
                    runner.enter("squeeze", previous_arrived=True)
                elif fault == "arrival":
                    runner.enter("grasp")
                else:
                    if fault == "cancel":
                        runner.cancel()
                    if fault == "clock":
                        runner.tick(0, feedback(0), (0.0,) * 28, (0.0,) * 28, None)
                    runner.tick(1 if fault == "stale" else 0, feedback(0), (0.0,) * 28, (0.0,) * 28, None)
            self.assertTrue(runner.failed)


class BaselineStreamTests(unittest.TestCase):
    def test_rate_and_all_limits_under_target_reversals(self):
        guard = BaselineStream((-1.0,) * 28, (1.0,) * 28, (2.0,) * 28)
        guard.start(feedback(0), 0)
        guard.accept(proposal(0), 0)
        previous, old_velocity = np.zeros(28), np.zeros(28)
        count = 0
        for i in range(1, 401):
            now = i * 0.005
            if i % 20 == 0:
                guard.accept(proposal(now, ((0.2 if i < 200 else -0.1),) * 28), now)
            output = guard.tick(now, feedback(now, tuple(previous)))
            if output is not None:
                current = np.asarray(output)
                velocity = (current - previous) / 0.01
                self.assertTrue(np.all(np.abs(velocity) <= np.asarray(guard.speed) + 1e-8))
                self.assertTrue(np.all(np.abs(velocity - old_velocity) / 0.01 <= np.asarray(guard.acceleration) + 1e-7))
                self.assertTrue(np.all(np.abs(current - previous) <= np.asarray(guard.step) + 1e-9))
                previous, old_velocity = current, velocity
                count += 1
        self.assertEqual(count, 200)


class GuardedStreamTests(unittest.TestCase):
    def test_duplicate_target_does_not_refresh_age(self):
        guard = stream()
        guard.start(feedback(0), 0)
        guard.accept(proposal(0), 0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            guard.accept(proposal(0), 0.1)
        self.assertFalse(guard.running)

    def test_faults_silence_commands(self):
        for fault in ("stale", "contact", "cancel", "dds", "stall", "solver", "reorder"):
            guard = stream()
            guard.start(feedback(0), 0)
            guard.accept(proposal(0), 0)
            with self.assertRaises(ValueError):
                if fault == "contact":
                    guard.accept(replace(proposal(0.01), valid=False), 0.01)
                elif fault == "solver":
                    guard.accept(replace(proposal(0.01), solve_duration_s=0.11), 0.01)
                else:
                    if fault == "cancel":
                        guard.stop("cancel")
                    fb = feedback(0.01)
                    if fault == "stale":
                        fb = feedback(-1)
                    if fault == "reorder":
                        guard.last_source = 0.02
                    guard.tick(0.06 if fault == "stall" else 0.01, fb, transport_ok=fault != "dds")
            self.assertFalse(guard.running)
            with self.assertRaises(ValueError):
                guard.tick(0.02, feedback(0.02))

    def test_single_writer_and_hardware_rejection(self):
        owner = CommandOwnership()
        owner.claim("baseline")
        guard = GuardedStream(ownership=owner)
        with self.assertRaises(ValueError):
            guard.start(feedback(0), 0)
        with self.assertRaises(ValueError):
            GuardedStream(config={"environment": "hardware"})


if __name__ == "__main__":
    unittest.main()
