"""Geometry and artifact regressions for passive fixed-camera calibration."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.fixed_camera import load_config, load_plane, accept_plane, write_json
from ada_grasp_ctrl.experiments.fixed_camera_geometry import (
    clusters,
    fit_plane,
    load_extrinsics,
    rigid,
    roi_mask,
    signed_height,
    transform,
)
from ada_grasp_ctrl.paths import project_root


class FixedCameraTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(project_root() / "src/ada_grasp_ctrl/config/experiments/fixed_camera.yaml")
        rng = np.random.default_rng(5)
        xy = rng.uniform(-0.4, 0.4, (2000, 2))
        self.points = np.column_stack([xy, 0.23 + 0.15 * xy[:, 0] - 0.07 * xy[:, 1]])
        self.points[:, 2] += rng.normal(0, 0.0005, len(xy))

    def test_tilted_plane_sign_outliers_and_reload(self):
        rng = np.random.default_rng(11)
        points = np.r_[self.points, rng.uniform(-0.5, 0.5, (300, 3))]
        plane, mask = fit_plane(points, self.config)
        expected = np.array([-0.15, 0.07, 1.0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(plane["normal_xyz"], expected, atol=0.001)
        self.assertTrue(plane["quality_passed"])
        self.assertGreater(mask.sum(), 1950)
        self.assertNotIn("z_table", plane)
        self.assertGreater(signed_height(np.array([[0, 0, 0.33]]), plane)[0], 0)
        self.assertLess(signed_height(np.array([[0, 0, 0.13]]), plane)[0], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plane.json"
            write_json(path, plane)
            np.testing.assert_array_equal(signed_height(points, plane), signed_height(points, load_plane(path)))

    def test_horizontal_approximation(self):
        points = self.points.copy()
        points[:, 2] = 0.42
        plane, _ = fit_plane(points, self.config)
        self.assertAlmostEqual(plane["z_table"], 0.42)
        self.assertAlmostEqual(plane["d"], -0.42)

    def test_degenerate_nonfinite_and_insufficient(self):
        for points in (self.points[:10], np.full((500, 3), np.nan), np.zeros((500, 3))):
            with self.assertRaises(ValueError):
                fit_plane(points, self.config)

    def test_quality_failure_not_forced_horizontal(self):
        config = dict(self.config, max_tilt_deg=2.0)
        plane, _ = fit_plane(self.points, config)
        self.assertFalse(plane["quality_passed"])
        self.assertTrue(plane["quality_failures"])
        self.assertGreater(plane["quality"]["tilt_deg"], 8)

    def test_rigid_direction_and_roundtrip(self):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.5]).as_matrix()
        matrix[:3, 3] = [0.3, -0.5, 0.1]
        np.testing.assert_allclose(
            transform(transform(self.points, matrix), np.linalg.inv(matrix)), self.points, atol=1e-12
        )
        for bad in (np.diag([1, 1, -1, 1]), np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                rigid(bad)

    def test_documented_calibration_direction(self):
        # Frozen numerical reference from the task, independent of file availability.
        matrix = np.array(
            [
                [0.0314127607, -0.7278113145, 0.68505761, 0.017245259],
                [-0.9989448841, -0.0458341917, -0.0028888471, -0.382160736],
                [0.0335015975, -0.6842440481, -0.7284831677, 0.367672239],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_allclose(
            transform([[0, 0, 0.5]], matrix), [[0.359774064, -0.38360515955, 0.00343065515]], atol=1e-8
        )

    def test_bad_extrinsics_and_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"T_world_cam": np.eye(4).tolist(), "frame": "depth optical to world"}))
            with self.assertRaisesRegex(ValueError, "color"):
                load_extrinsics(path)
        with self.assertRaises(ValueError):
            signed_height(self.points, dict(normal_xyz=[0, 0, -1], d=0))

    def test_roi_half_open_and_invalid(self):
        pixels = np.array([[10, 20], [19, 29], [20, 29], [np.nan, 22]])
        np.testing.assert_array_equal(roi_mask(pixels, [10, 20, 20, 30], 640, 480), [True, True, False, False])
        with self.assertRaises(ValueError):
            roi_mask(pixels, [-1, 0, 100, 100], 640, 480)

    def test_cluster_separation(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.003, (200, 3))
        b = rng.normal(0, 0.003, (150, 3)) + [0.2, 0, 0]
        groups = clusters(np.r_[a, b], 0.004, 0.02, 100)
        self.assertEqual([len(g) for g in groups], [200, 150])
        self.assertTrue(np.all(groups[0] < 200))

    def test_bad_depth_and_stale_data_rejected(self):
        from ada_grasp_ctrl.experiments.fixed_camera_geometry import valid_depth_mask, validate_timing

        points = np.tile([0.0, 0.0, 0.5], (1000, 1))
        valid, _, coverage = valid_depth_mask(points, 1000, self.config)
        self.assertTrue(valid.all())
        self.assertEqual(coverage, 1)
        for bad, resolution in [(points * 0, 1000), (points * np.nan, 1000), (points, 100000)]:
            with self.assertRaisesRegex(ValueError, "insufficient"):
                valid_depth_mask(bad, resolution, self.config)
        for stamps in [{"cloud": 0.0, "rgb": 0.0}, {"cloud": 2.0, "rgb": 2.0}, {"cloud": 0.9, "rgb": 0.7}]:
            with self.assertRaises(ValueError):
                validate_timing(stamps, 1.0, 0.5, 0.05)
        ages, offsets = validate_timing({"cloud": 0.9, "rgb": 0.91}, 1.0, 0.5, 0.05)
        self.assertAlmostEqual(ages["cloud"], 0.1)
        self.assertAlmostEqual(offsets["rgb"], 0.01)

    def test_accept_rejects_missing_independent_evidence(self):
        plane, _ = fit_plane(self.points, self.config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "plane.json", plane)
            write_json(root / "evidence.json", {})
            with self.assertRaisesRegex(ValueError, "reviewed"):
                accept_plane(root / "plane.json", root / "evidence.json", root, False)
            with self.assertRaisesRegex(ValueError, "camera/base"):
                accept_plane(root / "plane.json", root / "evidence.json", root, True)


if __name__ == "__main__":
    unittest.main()
