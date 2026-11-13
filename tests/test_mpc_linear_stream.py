"""Linear MPC transport, publication acknowledgment and recorded-boundary regressions."""

from dataclasses import replace
import unittest

import numpy as np

from ada_grasp_ctrl.mpc.streaming import GuardedStream
from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES
from test_mpc_phase2 import feedback, proposal


class LinearStreamTests(unittest.TestCase):
    def start(self, q=(0.0,) * 28, goal=(0.1,) * 28, now=0.0):
        stream = GuardedStream()
        stream.start(feedback(now, q), now)
        stream.accept(proposal(now, goal), now)
        return stream

    def send(self, stream, now, measured=(0.0,) * 28):
        command = stream.tick(now, feedback(now, measured))
        self.assertIsNotNone(command)
        stream.published(command, now)
        return command

    def test_constant_target_all_axes_exact_endpoint_and_hold(self):
        a, b = tuple(np.linspace(-0.3, 0.3, 28)), tuple(np.linspace(0.5, -0.5, 28))
        stream = self.start(a, b)
        self.assertEqual(stream.q, a)
        self.assertIsNone(stream.q_last_published)
        self.assertIsNone(stream.tick(0, feedback(0)))
        for i in range(1, 26):
            now = i / 100
            actual = self.send(stream, now)
            np.testing.assert_allclose(
                actual, (1 - min(now / 0.1, 1)) * np.array(a) + min(now / 0.1, 1) * np.array(b), atol=1e-15
            )
            if i >= 10:
                self.assertEqual(actual, b)
        with self.assertRaisesRegex(ValueError, "Stale"):
            stream.tick(0.26, feedback(0.26))
        self.assertFalse(stream.running)

    def test_retarget_anchors_to_last_published_not_unsent_or_measured(self):
        for switch in (0.03, 0.05, 0.07):
            with self.subTest(switch=switch):
                stream = self.start()
                for i in range(1, round(switch * 100) - 1):
                    anchor = self.send(stream, i / 100)
                unsent_at = switch - 0.01
                unsent = stream.tick(unsent_at, feedback(unsent_at))
                self.assertNotEqual(unsent, anchor)
                self.assertEqual(stream.q, anchor)
                goal = (-0.2,) * 28
                stream.accept(proposal(switch, goal), switch)
                self.assertEqual(stream.q_start, anchor)
                self.assertEqual(self.send(stream, switch, measured=(0.9,) * 28), anchor)
                for i in range(1, 11):
                    now = switch + i / 100
                    result = self.send(stream, now, measured=(0.9,) * 28)
                    np.testing.assert_allclose(
                        result, np.array(anchor) + (np.array(goal) - anchor) * i / 10, atol=1e-15
                    )
                self.assertEqual(result, goal)

    def test_delay_burst_free_and_missing_publish_latches(self):
        stream = self.start()
        out = self.send(stream, 0.03)
        np.testing.assert_allclose(out, 0.03)
        self.assertIsNone(stream.tick(0.031, feedback(0.031)))
        self.assertEqual(stream.dropped_ticks, 2)
        stream.tick(0.04, feedback(0.04))
        with self.assertRaisesRegex(ValueError, "not been published"):
            stream.tick(0.05, feedback(0.05))
        self.assertEqual(stream.q, out)
        self.assertFalse(stream.running)

    def test_bad_commit_missing_proposal_and_owner_loss_latch(self):
        for case in ("commit", "missing", "owner", "backward"):
            stream = self.start()
            if case == "missing":
                stream.target = None
            elif case == "owner":
                stream.ownership.owner = "competitor"
            with self.assertRaises(ValueError):
                if case == "commit":
                    stream.tick(0.01, feedback(0.01))
                    stream.published((0.9,) * 28, 0.01)
                else:
                    stream.tick(-0.01 if case == "backward" else 0.01, feedback(0))
            self.assertFalse(stream.running)
            self.assertIsNone(stream.q_last_published)
            if case == "owner":
                self.assertEqual(stream.ownership.owner, "competitor")

    def test_no_stream_motion_constraints_or_silent_clipping(self):
        stream = self.start(goal=(5.0,) * 28)
        self.assertFalse(any(hasattr(stream, n) for n in ("velocity", "speed", "acceleration", "step", "check_limits")))
        for i in range(1, 11):
            result = self.send(stream, i / 100)
        self.assertEqual(result, (5.0,) * 28)

    def test_recorded_ring_boundary_and_upper_boundary(self):
        # Actual saved command history; no fabricated rejected sample. 2026-11-12 trial.
        prior, last, lower = -0.34883730710778654, -0.3490208210570351, -0.3491
        self.assertGreater(prior, last)
        axis = JOINT_NAMES.index("right_ring_MCP_AA")
        for anchor, target in ((last, lower), (-last, -lower)):
            q, goal = [0.0] * 28, [0.0] * 28
            q[axis], goal[axis] = anchor, target
            stream = self.start(tuple(q), tuple(goal))
            values = [anchor]
            for i in range(1, 11):
                values.append(self.send(stream, i / 100)[axis])
            self.assertTrue(all(min(anchor, target) <= q <= max(anchor, target) for q in values))
            self.assertEqual(values[-1], target)

    def test_finite_extremes_and_unchanged_equal_endpoints(self):
        q = (1e308,) * 28
        stream = self.start(q, tuple(-v for v in q))
        for i in range(1, 11):
            self.assertTrue(np.isfinite(self.send(stream, i / 100)).all())
        stream = self.start((-0.3491,) * 28, (-0.3491,) * 28, now=10.0)
        for i in range(1, 11):
            self.assertEqual(self.send(stream, 10 + i / 100), (-0.3491,) * 28)

    def test_invalid_named_or_nonfinite_data_never_publishes(self):
        for names in (JOINT_NAMES[::-1], JOINT_NAMES[:-1], (JOINT_NAMES[0],) * 28):
            with self.assertRaises(ValueError):
                replace(feedback(0), joint_names=names)
        for values in ((float("nan"),) * 28, (float("inf"),) * 28, (0.0,) * 27):
            with self.assertRaises(ValueError):
                proposal(0, values)
        stream = self.start()
        with self.assertRaisesRegex(ValueError, "deadline"):
            stream.accept(replace(proposal(0.01), solve_duration_s=0.101), 0.01)
        self.assertFalse(stream.running)


if __name__ == "__main__":
    unittest.main()
