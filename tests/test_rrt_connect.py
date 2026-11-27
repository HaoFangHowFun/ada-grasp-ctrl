"""Directed graph, sampled collision, reproducibility and fail-closed planner tests."""

import time
import unittest

import numpy as np

from ada_grasp_ctrl.motion_planning.rrt_connect import plan


def sampled(a, b):
    points = np.linspace(a, b, max(2, int(np.ceil(np.max(np.abs(b - a)) / 0.01)) + 1))
    return not any(abs(q[0]) < 0.15 and abs(q[1]) < 0.65 for q in points)


class RRTConnectTests(unittest.TestCase):
    limits = np.array([[-1, 1], [-1, 1]])
    start, goal = np.array([-0.8, 0.0]), np.array([0.8, 0.0])

    def test_direct_and_identical_endpoints(self):
        for end in (self.goal, self.start):
            result = plan(self.start, end, self.limits, lambda a, b: True)
            self.assertEqual(result.status, "success")
            self.assertEqual(result.method, "direct")
            self.assertEqual(result.iterations, 0)
            self.assertEqual(len(result.path), 1 if np.array_equal(end, self.start) else 2)
            np.testing.assert_array_equal(result.path[-1], end)

    def test_detour_ordering_interior_checks_and_shortcut_legality(self):
        results = [plan(self.start, self.goal, self.limits, sampled, seed=13, shortcut_s=0) for _ in range(2)]
        for result in results:
            self.assertEqual(result.status, "success")
            self.assertEqual(result.method, "rrt_connect")
            np.testing.assert_array_equal(result.path[0], self.start)
            np.testing.assert_array_equal(result.path[-1], self.goal)
            self.assertTrue(all(sampled(a, b) for a, b in zip(result.path, result.path[1:])))
            self.assertTrue(all(not np.array_equal(a, b) for a, b in zip(result.path, result.path[1:])))
        np.testing.assert_array_equal(results[0].path, results[1].path)
        short = plan(self.start, self.goal, self.limits, sampled, seed=13)
        self.assertEqual(short.status, "success")
        self.assertLessEqual(len(short.path), short.original_waypoints)
        self.assertTrue(all(sampled(a, b) for a, b in zip(short.path, short.path[1:])))

    def test_invalid_endpoints_no_solution_timeout_cancellation(self):
        for start, goal, status in (
            ([2, 0], self.goal, "invalid_start"),
            (self.start, [np.nan, 0], "invalid_goal"),
            ([0, 0], self.goal, "invalid_start"),
            (self.start, [0, 0], "invalid_goal"),
        ):
            result = plan(start, goal, self.limits, sampled)
            self.assertEqual(result.status, status)
            self.assertEqual(result.path, [])
        result = plan(self.start, self.goal, self.limits, lambda a, b: np.array_equal(a, b), max_iterations=5)
        self.assertEqual(result.status, "no_solution")
        result = plan(self.start, self.goal, self.limits, sampled, deadline=time.monotonic() - 1)
        self.assertEqual(result.status, "timeout")
        result = plan(self.start, self.goal, self.limits, sampled, canceled=lambda: True)
        self.assertEqual(result.status, "canceled")

    def test_reverse_tree_obeys_forward_execution_direction(self):
        # Every useful edge must move forward in x; a backward goal-tree check
        # would be trapped and fail this deterministic case.
        def directed(a, b):
            return b[0] >= a[0] and sampled(a, b)

        result = plan(self.start, self.goal, self.limits, directed, seed=7, max_iterations=5000)
        self.assertEqual(result.status, "success")
        self.assertTrue(all(directed(a, b) for a, b in zip(result.path, result.path[1:])))

    def test_final_check_and_callback_overrun_fail_closed(self):
        count = 0

        def changing(a, b):
            nonlocal count
            count += 1
            return count < 4

        result = plan(self.start, self.goal, self.limits, changing)
        self.assertEqual(result.status, "final_validation_failed")
        self.assertEqual(result.path, [])

        def slow(a, b):
            time.sleep(0.005)
            return True

        result = plan(self.start, self.goal, self.limits, slow, timeout_s=0.001)
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.path, [])


if __name__ == "__main__":
    unittest.main()
