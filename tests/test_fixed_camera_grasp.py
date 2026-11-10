"""Checkpoint adapter math, acceptance and artifact-integrity regression checks."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.fixed_camera import accept_plane, load_config, load_snapshot, write_json
from ada_grasp_ctrl.experiments.fixed_camera_geometry import file_hash, fit_plane
from ada_grasp_ctrl.experiments.fixed_camera_grasp import STAGES, decode_candidates, preprocess
from ada_grasp_ctrl.paths import project_root


class GraspMathTests(unittest.TestCase):
    def test_preprocess_roundtrip_and_seed(self):
        points = np.random.default_rng(1).normal(size=(200, 3)) * 0.02 + [0.4, -0.5, -0.1]
        normalized, meta = preprocess(points, 1024, 42)
        again, again_meta = preprocess(points, 1024, 42)
        np.testing.assert_array_equal(normalized, again)
        self.assertEqual(meta, again_meta)
        restored = normalized + meta["center_world_m"]
        np.testing.assert_allclose(restored, points[meta["indices"]], atol=1e-7, rtol=0)
        self.assertEqual(meta["scale"], 1)

    def test_all_stages_preserved_without_double_transform(self):
        raw = np.zeros((2, 3, 58))
        raw[:, :, 3] = 1
        raw[:, :, 32] = 1
        for i in range(3):
            raw[:, i, :3] = [0.01 * i, -0.02 * i, 0.03 * i]
            raw[:, i, 7:29] = np.arange(22) * 0.01 + i * 0.1
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.5, -0.3]).as_matrix()
        matrix[:3, 3] = [0.3, -0.2, 0.1]
        names = [f"right_joint_{i}" for i in range(22)]
        meta = dict(frame_id="world", scale=1, center_world_m=[0.4, -0.6, -0.1])
        decoded = decode_candidates(raw, np.array([1.0, 0.0]), names, meta, matrix)
        for i, stage in enumerate(STAGES):
            row = decoded[0]["stages"][stage]
            np.testing.assert_allclose(row["position_m"], raw[0, i, :3] + meta["center_world_m"])
            np.testing.assert_array_equal(list(row["joint_positions_rad"].values()), raw[0, i, 7:29])
            np.testing.assert_allclose(matrix @ row["T_color_hand"], row["T_world_hand"], atol=1e-12)
            np.testing.assert_array_equal(np.asarray(row["T_world_hand"])[:3, :3], np.eye(3))
        raw[0, 0, 3] = 0
        with self.assertRaisesRegex(ValueError, "quaternion"):
            decode_candidates(raw, np.array([1.0, 0.0]), names, meta, matrix)

    def test_snapshot_tamper_and_world_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points = np.arange(30, dtype=float).reshape(-1, 3) / 100
            arrays = dict(
                points_source_m=points,
                valid_source_indices=np.arange(10),
                points_color_m=points,
                points_world_m=points,
                T_world_source=np.eye(4),
                T_world_color=np.eye(4),
                T_color_source=np.eye(4),
            )
            np.savez(root / "snapshot.npz", **arrays)
            meta = dict(schema_version=1, unit="m", frame_id="world", snapshot_sha256=file_hash(root / "snapshot.npz"))
            write_json(root / "snapshot.json", meta)
            load_snapshot(root)
            arrays["points_world_m"] = points + 0.1
            np.savez(root / "snapshot.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_snapshot(root)
            meta["snapshot_sha256"] = file_hash(root / "snapshot.npz")
            write_json(root / "snapshot.json", meta)
            with self.assertRaisesRegex(ValueError, "round-trip"):
                load_snapshot(root)

    def test_accept_and_reject_wrong_xy_calibration(self):
        config = load_config(project_root() / "src/ada_grasp_ctrl/config/experiments/fixed_camera.yaml")
        xy = np.random.default_rng(3).uniform(-0.4, 0.4, (500, 2))
        plane, _ = fit_plane(np.c_[xy, np.full(500, 0.2)], config)
        plane.update(
            config=config,
            snapshot_sha256="capture0",
            extrinsics={"sha256": "calibration"},
            T_world_color=np.eye(4).tolist(),
        )
        refs = np.array([[0, 0, 0.2], [0.2, 0, 0.2], [0, 0.2, 0.2]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "plane.json", plane)
            repeats = []
            for i in (1, 2):
                repeat = deepcopy(plane)
                repeat["snapshot_sha256"] = f"capture{i}"
                repeat["centroid_world_m"][0] += i * 0.1
                path = root / f"repeat{i}.json"
                write_json(path, repeat)
                repeats.append(str(path))
            evidence = dict(
                camera_base_unchanged=True,
                world_definition_confirmed=True,
                reference_points_world_m=refs.tolist(),
                reference_points_color_m=refs.tolist(),
                repeat_plane_files=repeats,
            )
            write_json(root / "evidence.json", evidence)
            accepted = accept_plane(root / "plane.json", root / "evidence.json", root, True)
            self.assertEqual(accepted["status"], "accepted")
            self.assertTrue(accepted["extrinsics"]["independently_verified"])
            # Coplanar reference points alone cannot catch a wrong world XY origin.
            evidence["reference_points_color_m"] = (refs + [0.04, 0, 0]).tolist()
            write_json(root / "evidence.json", evidence)
            with self.assertRaisesRegex(ValueError, "XYZ"):
                accept_plane(root / "plane.json", root / "evidence.json", root, True)
            evidence["reference_points_color_m"] = refs.tolist()
            evidence["repeat_plane_files"] = [repeats[0], repeats[0]]
            write_json(root / "evidence.json", evidence)
            with self.assertRaisesRegex(ValueError, "independently captured"):
                accept_plane(root / "plane.json", root / "evidence.json", root, True)

    def test_project_distortion_and_behind_camera(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("Projection boundary uses system ROS Python OpenCV")
        from ada_grasp_ctrl.experiments.fixed_camera_geometry import project_color

        info = dict(distortion_model="plumb_bob", k=[100, 0, 320, 0, 100, 240, 0, 0, 1], d=[0.1, 0, 0, 0, 0])
        pixels = project_color(np.array([[0.5, 0, 1], [0, 0, -1]]), info)
        self.assertAlmostEqual(pixels[0, 0], 371.25)
        self.assertTrue(np.isnan(pixels[1]).all())


if __name__ == "__main__":
    unittest.main()
