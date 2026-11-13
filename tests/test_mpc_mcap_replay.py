"""Recorded joint ordering and acquisition/status gates without ROS or hardware."""

import unittest

from ada_grasp_ctrl.mpc.mcap_replay import contact_blockers, ordered


class MCAPReplayTests(unittest.TestCase):
    def test_joint_mapping_and_invalid_names(self):
        self.assertEqual(ordered(dict(names=["b", "a"], positions=[2, 1]), ("a", "b")), (1, 2))
        for row in (
            dict(names=["a", "a"], positions=[1, 2]),
            dict(names=["a"], positions=[1]),
            dict(names=["a", "b"], positions=[1, float("nan")]),
        ):
            with self.assertRaises(ValueError):
                ordered(row, ("a", "b"))

    def test_missing_no_longer_blocks_but_acquisition_is_not_receipt(self):
        frame = dict(
            contact_status="VALID",
            status="VALID",
            contact_count=1,
            received=10,
            sdk_timestamp_unit="unknown",
            sdk_timestamp_epoch="unknown",
            sdk_timestamp=10,
        )
        self.assertEqual(contact_blockers(frame, 10), ["acquisition_clock_unmapped"])
        for status in ("MISSING", "INVALID", "EMPTY", "VALID"):
            reasons = contact_blockers(dict(frame, contact_status=status), 10)
            self.assertEqual("contact_" + status.lower() in reasons, status == "INVALID")
        self.assertIn("multi_contact_resultant", contact_blockers(dict(frame, contact_count=2), 10))
        self.assertIn("tactile_stale", contact_blockers(frame, 11))
        self.assertIn("future_receipt", contact_blockers(frame, 9))
        known = dict(frame, sdk_timestamp_unit="s", sdk_timestamp_epoch="unix")
        self.assertEqual(contact_blockers(known, 10), [])
        self.assertIn("acquisition_stale_or_future", contact_blockers(known, 11))


if __name__ == "__main__":
    unittest.main()
