"""Measured-surface topology and export checks without camera hardware."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from ada_grasp_ctrl.experiments.fixed_camera_asset import triangulate_depth, write_ply


class MeasuredSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.info = dict(
            width=4,
            height=4,
            distortion_model="plumb_bob",
            d=[0.0, 0.0, 0.0, 0.0, 0.0],
            k=[500.0, 0.0, 1.5, 0.0, 500.0, 1.5, 0.0, 0.0, 1.0],
        )
        y, x = np.mgrid[0:4, 0:4]
        self.pixels = np.c_[x.ravel(), y.ravel()]
        self.points = np.c_[(self.pixels - 1.5) * 0.001, np.full(16, 0.5)]

    def test_plane_has_only_grid_neighbors_and_faces_camera(self):
        faces, report = triangulate_depth(self.points, self.info, 0.008, 0.008)
        self.assertEqual(len(faces), 18)
        self.assertEqual(report["open_boundary_edges"], 12)
        self.assertEqual(report["nonmanifold_edges"], 0)
        triangles = self.points[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        self.assertTrue(np.all(normals[:, 2] < 0))
        self.assertTrue(np.all(np.ptp(self.pixels[faces], axis=1) <= 1))
        self.assertFalse(report["watertight"])

    def test_missing_pixels_are_not_filled(self):
        keep = np.arange(16) != 5
        points, pixels = self.points[keep], self.pixels[keep]
        faces, _ = triangulate_depth(points, self.info, 0.008, 0.008)
        self.assertLess(len(faces), 18)
        # A projected triangle can use only neighboring retained pixels, not a long hole bridge.
        self.assertTrue(np.all(np.ptp(pixels[faces], axis=1) <= 1))

    def test_depth_discontinuity_is_not_bridged(self):
        points = self.points.copy()
        far = self.pixels[:, 0] >= 2
        points[far] *= 1.2
        faces, report = triangulate_depth(points, self.info, 0.008, 0.008)
        self.assertGreater(report["rejected_triangles"], 0)
        self.assertTrue(np.all(np.ptp(points[faces, 2], axis=1) < 0.008))
        self.assertTrue(np.all(np.ptp(far[faces].astype(int), axis=1) == 0))

    def test_wrong_grid_duplicate_and_nonfinite_rejected(self):
        cases = [self.points.copy(), np.r_[self.points, self.points[:1]], self.points * np.nan]
        cases[0][0, 0] += 0.0004
        for points in cases:
            with self.assertRaises(ValueError):
                triangulate_depth(points, self.info, 0.008, 0.008)
        with self.assertRaisesRegex(ValueError, "zero-distortion"):
            triangulate_depth(self.points, dict(self.info, d=[0.1, 0, 0, 0, 0]), 0.008, 0.008)

    def test_metric_ply_roundtrip_and_colors(self):
        import trimesh

        faces, _ = triangulate_depth(self.points, self.info, 0.008, 0.008)
        colors = np.tile([20, 170, 30], (len(self.points), 1)).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "surface.ply"
            write_ply(path, self.points, colors, faces)
            mesh = trimesh.load(path, process=False)
            np.testing.assert_allclose(mesh.vertices, self.points, atol=1e-12)
            np.testing.assert_array_equal(mesh.faces, faces)
            np.testing.assert_array_equal(mesh.visual.vertex_colors[:, :3], colors)
            self.assertFalse(mesh.is_watertight)


if __name__ == "__main__":
    unittest.main()
