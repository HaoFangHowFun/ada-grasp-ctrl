"""Regression checks for the isolated Sharpa mirror API."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.sharpa_mirror import SharpaHandMirror


class SharpaMirrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mirror = SharpaHandMirror()

    def test_full_ranges_and_named_order(self) -> None:
        limits = np.array(self.mirror.config["right_limits"])
        for q in (limits[:, 0], limits[:, 1], limits.mean(axis=1)):
            actual = self.mirror.right_to_left(q)
            for i, name in enumerate(self.mirror.left_joint_names):
                index = self.mirror.right_joint_names.index(name.replace("left_", "right_", 1))
                self.assertEqual(actual[i], q[index])

    def test_invalid_joint_inputs(self) -> None:
        for q in (np.zeros(21), np.full(22, np.nan), np.full(22, np.inf), np.full(22, 50)):
            with self.assertRaises(ValueError):
                self.mirror.right_to_left(q)

    def test_world_geometry_and_offset_plane(self) -> None:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.5, 0.9]).as_matrix()
        pose[:3, 3] = [0.2, -0.1, 0.7]
        n = np.array([1, 2, 3]) / np.sqrt(14)
        c = np.array([0.4, 0.2, -0.1])
        actual = self.mirror.mirror_wrist_pose(pose, n, c)
        local_right = np.array([0.02, 0.03, 0.1])
        local_left = local_right * [1, -1, 1]
        world_right = pose[:3, :3] @ local_right + pose[:3, 3]
        expected = world_right - 2 * n * np.dot(n, world_right - c)
        np.testing.assert_allclose(actual[:3, :3] @ local_left + actual[:3, 3], expected, atol=1e-12)
        np.testing.assert_allclose(self.mirror.mirror_wrist_pose(actual, n, c), pose, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(actual[:3, :3]), 1)

    def test_invalid_poses_and_planes(self) -> None:
        invalid = np.eye(4)
        invalid[0, 0] = -1
        for pose, normal in (
            (invalid, [1, 0, 0]),
            (np.eye(3), [1, 0, 0]),
            (np.eye(4), [0, 0, 0]),
            (np.eye(4), [np.nan, 0, 0]),
        ):
            with self.assertRaises(ValueError):
                self.mirror.mirror_wrist_pose(pose, normal)


if __name__ == "__main__":
    unittest.main()
