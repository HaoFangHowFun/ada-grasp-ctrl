"""Official left/right FK and offline world-display conversion; no ROS or SDK owner."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.contact.force_display import normal_load_totals
from ada_grasp_ctrl.experiments.right_ros_contract import side_names
from ada_grasp_ctrl.tactile_preview.adapter import TactileAdapter
from ada_grasp_ctrl.tactile_preview.data import FINGERS, TOPICS, LatestFrames, normalize_frame, topics
from ada_grasp_ctrl.tactile_preview.kinematics import load_hand
from ada_grasp_ctrl.tactile_preview.world_forces import world_forces
from test_tactile_preview import SHARE, message, synthetic_geometry


class DualTactileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for side in ("left", "right"):
            if not (SHARE / f"urdf/{side}_sharpa_wave_with_flange.urdf").exists():
                raise unittest.SkipTest("Official local left/right hand URDF assets unavailable")

    def setUp(self):
        self.now = 10.0
        self.adapters, self.frames, self.joints = {}, {}, {}
        for side in ("left", "right"):
            hand = load_hand(SHARE / f"urdf/{side}_sharpa_wave_with_flange.urdf", SHARE, side=side, load_meshes=False)
            self.adapters[side] = TactileAdapter(hand, synthetic_geometry(), side=side, clock=lambda: self.now)
            self.frames[side] = {}
            for finger in FINGERS:
                payload = message(finger)
                payload.side = side
                payload.channel += 5 if side == "left" else 0
                self.frames[side][finger] = normalize_frame(payload, finger, 10, 100.01, side=side)
            self.joints[side] = dict(
                names=list(side_names(side)[1]),
                positions=[0.0] * 22,
                received=10.0,
                receipt_ros=100.0,
                subscriber_ros=100.01,
            )
            self.adapters[side].update(self.frames[side], self.joints[side])
        names = list(sum(side_names("left"), ()) + sum(side_names("right"), ()))
        self.poses = {}
        for side, angle in (("left", 0.4), ("right", -0.7)):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler("xyz", [angle, 0.3, -0.2]).as_matrix()
            pose[:3, 3] = [0.4, angle, 0.6]
            self.poses[side] = pose
        self.robot = SimpleNamespace(
            joint_names=names, configuration=lambda q: np.asarray(q), fk=lambda q, side: self.poses[side]
        )
        self.measured = dict(joint_names=names, positions_rad=[0.0] * 56, received=10.0)

    def test_official_left_channels_reject_right_channel_identity(self):
        for finger in FINGERS:
            payload = message(finger)
            payload.side = "left"
            with self.assertRaises(ValueError):
                normalize_frame(payload, finger, 10, 100.01, side="left")
            payload.channel += 5
            self.assertEqual(normalize_frame(payload, finger, 10, 100.01, side="left").side, "left")
            payload.side = "right"
            with self.assertRaises(ValueError):
                normalize_frame(payload, finger, 10, 100.01, side="right")

    def test_official_side_fk_and_world_vector_position_semantics(self):
        snapshot = world_forces(self.adapters, self.robot, self.measured, now=self.now)
        self.assertEqual(len(snapshot.fingers), 10)
        self.assertEqual(snapshot.timestamp_clock, "Hardware receipt monotonic")
        for row in snapshot.fingers:
            adapter = self.adapters[row.side]
            contact = adapter.get_contacts()[row.finger]
            pose = self.poses[row.side]
            self.assertEqual(row.status, "VALID")
            np.testing.assert_allclose(row.position, pose[:3, :3] @ contact.position + pose[:3, 3])
            np.testing.assert_allclose(row.force, -(pose[:3, :3] @ contact.force))
            np.testing.assert_allclose(row.normal, pose[:3, :3] @ contact.normal)
            self.assertIsNone(row.normal_load)
            self.assertEqual(contact.frame_id, f"{row.side}_hand_flange")
            if row.side == "left":
                self.assertIn("pending hardware validation", contact.force_convention_quality)
        self.assertEqual(normal_load_totals(snapshot), (None, None))
        self.assertEqual(TOPICS, topics("right"))
        self.assertEqual(topics("left")["thumb"], "/sharpa/left_hand/tactile/thumb/frame")
        with self.assertRaises(ValueError):
            topics("both")
        # These actual URDF transforms differ; no mirrored-right force mapping is used.
        left = self.adapters["left"].hand.get_transform("left_thumb_elastomer")
        right = self.adapters["right"].hand.get_transform("right_thumb_elastomer")
        self.assertFalse(np.allclose(left, right))

    def test_empty_missing_stale_and_multi_contact_load_unavailable(self):
        for side in ("left", "right"):
            empty = {
                f: replace(frame, contact_status="EMPTY", contact_count=0, contact_row=(), contact_shape=())
                for f, frame in self.frames[side].items()
            }
            self.adapters[side].update(empty, self.joints[side])
        snapshot = world_forces(self.adapters, self.robot, self.measured, now=self.now)
        self.assertEqual(normal_load_totals(snapshot), (0, 0))
        self.assertEqual({r.status for r in snapshot.fingers}, {"NO_CONTACT"})
        self.adapters["left"].update({}, self.joints["left"])
        self.assertEqual(
            normal_load_totals(world_forces(self.adapters, self.robot, self.measured, now=self.now)), (None, 0)
        )
        self.now += 0.6
        snapshot = world_forces(self.adapters, self.robot, self.measured, now=self.now)
        self.assertTrue(all(r.position is None and r.normal_load is None for r in snapshot.fingers))
        self.assertEqual(normal_load_totals(snapshot), (None, None))

    def test_wrong_side_or_joint_order_is_unavailable(self):
        self.measured["joint_names"] = list(reversed(self.measured["joint_names"]))
        snapshot = world_forces(self.adapters, self.robot, self.measured, now=self.now)
        self.assertTrue(all(r.force is None for r in snapshot.fingers))
        self.assertTrue(snapshot.errors)
        with self.assertRaises(ValueError):
            TactileAdapter(self.adapters["left"].hand, synthetic_geometry())
        cache = LatestFrames(side="left")
        payload = message("index")
        cache.receive(payload, "index", 100.0)
        self.assertNotIn("index", cache.snapshot()["frames"])
        payload.side = "left"
        payload.channel += 5
        cache.receive(payload, "index", 100.0)
        self.assertEqual(cache.snapshot()["frames"]["index"]["side"], "left")
        frames = dict(self.frames["left"], index=self.frames["right"]["index"])
        result = dict(self.adapters["left"].update(frames, self.joints["left"]).fingers)
        self.assertEqual(result["index"].status, "INVALID")


if __name__ == "__main__":
    unittest.main()
