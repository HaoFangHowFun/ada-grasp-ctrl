"""Bounded ROS graph discovery tests; no ROS initialization or commands."""

import unittest
from unittest.mock import Mock, patch

from ada_grasp_ctrl.pipeline.ros_probe import wait_for_tactile_publishers
from ada_grasp_ctrl.tactile_preview.data import TOPICS


class DiscoveryTests(unittest.TestCase):
    def test_delayed_discovery_waits_until_all_fingers_are_unique(self):
        now = [0.0]
        node = Mock()
        node.count_publishers.side_effect = lambda topic: int(now[0] >= 3.0)

        def spin_once(node, timeout_sec):
            now[0] += timeout_sec

        with patch("ada_grasp_ctrl.pipeline.ros_probe.time.monotonic", side_effect=lambda: now[0]):
            counts = wait_for_tactile_publishers(node, spin_once)
        self.assertEqual(counts, dict.fromkeys(TOPICS.values(), 1))
        self.assertGreaterEqual(now[0], 3.0)
        self.assertLess(now[0], 3.1)

    def test_missing_and_duplicate_sources_still_fail_with_actual_counts(self):
        for bad_count in (0, 2):
            with self.subTest(count=bad_count):
                node = Mock()
                node.count_publishers.side_effect = lambda topic: bad_count if "pinky" in topic else 1
                spin = Mock()
                with patch("ada_grasp_ctrl.pipeline.ros_probe.time.monotonic", side_effect=[0.0, 8.0]):
                    with self.assertRaisesRegex(RuntimeError, f"/pinky/frame={bad_count}"):
                        wait_for_tactile_publishers(node, spin)
                spin.assert_not_called()

    def test_ready_sources_do_not_add_delay(self):
        node, spin = Mock(), Mock()
        node.count_publishers.return_value = 1
        self.assertEqual(wait_for_tactile_publishers(node, spin), dict.fromkeys(TOPICS.values(), 1))
        spin.assert_not_called()
