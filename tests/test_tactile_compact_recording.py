"""ROS Jazzy CDR/MCAP regression checks for explicit distribution omission."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    import rosbag2_py
    from dual_sharpa_wave.msg import TactileBlock, TactileFrame
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import JointState

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from ada_grasp_ctrl.tactile_preview.record_compact import CompactBag, JOINT_TOPICS, TACTILE_TOPICS, omit_distribution


@unittest.skipUnless(ROS_AVAILABLE, "Requires sourced ROS Jazzy and dual_sharpa_wave messages")
class CompactRecordingTests(unittest.TestCase):
    def frame(self, contact_status: int) -> "TactileFrame":
        """Build a frame with identities, force, contacts, and a full distribution."""
        frame = TactileFrame(schema_version=1, session_id="test-session", receipt_sequence=42, finger="index")
        frame.header.stamp.sec = 123
        frame.receive_monotonic_ns = 999
        frame.f6 = TactileBlock(status=1, shape=[1, 1, 6], raw_values=[-1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        frame.contact_points = TactileBlock(status=contact_status)
        if contact_status == TactileBlock.VALID:
            frame.contact_points.shape = [1, 3]
            frame.contact_points.raw_values = [1.0, 2.0, 3.0]
        frame.dist_force = TactileBlock(status=1, shape=[60, 60, 3], raw_values=[0.5] * 10800, dtype="float64")
        return frame

    def test_omission_preserves_other_fields_and_source(self) -> None:
        for status in (TactileBlock.MISSING, TactileBlock.EMPTY, TactileBlock.VALID, TactileBlock.INVALID):
            with self.subTest(contact_status=status):
                original = self.frame(status)
                before = serialize_message(original)
                compact = omit_distribution(original)
                # CDR alignment padding is not guaranteed deterministic across serializations.
                self.assertEqual(original, deserialize_message(before, TactileFrame))
                for field in original.get_fields_and_field_types():
                    if field != "dist_force":
                        self.assertEqual(getattr(compact, field), getattr(original, field), field)
                self.assertEqual(compact.dist_force.status, TactileBlock.MISSING)
                self.assertEqual(len(compact.dist_force.raw_values), 0)
                marker = json.loads(compact.dist_force.error)
                self.assertEqual(marker["reason"], "omitted_by_compact_recorder")
                self.assertEqual(marker["original_status"], TactileBlock.VALID)
                self.assertEqual(marker["original_shape"], [60, 60, 3])
                self.assertLess(len(serialize_message(compact)), len(before) / 10)

    def test_mcap_roundtrip_preserves_joint_bytes_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bag"
            bag = CompactBag(path)
            joint_bytes = serialize_message(JointState(name=["right_J1"], position=[0.123]))
            frame = self.frame(TactileBlock.VALID)
            try:
                for topic in JOINT_TOPICS:
                    bag.write(topic, joint_bytes, 100)
                for topic in TACTILE_TOPICS:
                    bag.write(topic, serialize_message(frame), 200)
                with self.assertRaises(ValueError):
                    bag.write("/unexpected", joint_bytes, 300)
            finally:
                bag.close()
            reader = rosbag2_py.SequentialReader()
            reader.open(
                rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"), rosbag2_py.ConverterOptions("", "")
            )
            seen = set()
            while reader.has_next():
                topic, payload, stamp = reader.read_next()
                seen.add(topic)
                if topic in JOINT_TOPICS:
                    self.assertEqual(payload, joint_bytes)
                    self.assertEqual(stamp, 100)
                else:
                    restored = deserialize_message(payload, TactileFrame)
                    self.assertEqual(restored, omit_distribution(frame))
                    self.assertEqual(stamp, 200)
            self.assertEqual(seen, set(JOINT_TOPICS + TACTILE_TOPICS))
            with self.assertRaises(FileExistsError):
                CompactBag(path)


if __name__ == "__main__":
    unittest.main()
