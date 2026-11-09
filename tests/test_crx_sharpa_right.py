"""Regression contracts for world registration and static right-arm preparation."""

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.crx_sharpa_right_model import JOINTS, matrix_pose, pose_matrix
from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import (
    json_write,
    load_initial_configuration,
    load_preview,
    object_bounds,
    register_record,
    set_state,
)
from ada_grasp_ctrl.experiments.phase_comparison import sha256


class RegistrationContractTest(unittest.TestCase):
    def setUp(self):
        self.source = {
            "source_frame": "right_hand_flange",
            "pregrasp_extrapolation": 2,
            "obj_pose": matrix_pose(self.transform([0.8, -0.4, 0.3], [0.3, -0.4, 0.7])),
            "obj_scale": 0.133,
            "flange_targets_wxyz": np.array(
                [matrix_pose(self.transform([0.7, -0.3, 0.6 + i * 0.02], [0.4, 0.1, -0.2])) for i in range(3)]
            ),
        }
        self.registration = self.transform([0.2, 0.5, -0.1], [0.4, -0.2, 1.2])

    @staticmethod
    def transform(position, rotation):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
        result[:3, 3] = position
        return result

    def test_initial_config_uses_radians_and_named_open_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            task, reference = Path(directory) / "task.yaml", Path(directory) / "initial.yaml"
            degrees = [-90, -30, 210, 0, -60, 0]
            hand_names = list(JOINTS[6:])
            hand = {name: i / 100 for i, name in enumerate(hand_names)}
            task.write_text(yaml.safe_dump({"offline_seed": np.deg2rad(degrees).tolist(), "open_hand": hand}))
            reference.write_text(yaml.safe_dump({"right": degrees}))
            q = load_initial_configuration(task, reference)
            np.testing.assert_allclose(q[:6], np.deg2rad(degrees), atol=1e-12)
            np.testing.assert_allclose(q[6:], np.arange(22) / 100)
            reference.write_text(yaml.safe_dump({"right": [0] * 6}))
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_initial_configuration(task, reference)

    def test_world_targets_transform_once_and_preserve_relative_pose(self):
        actual, checks = register_record(self.source, self.registration)
        obj = pose_matrix(actual["obj_pose"])
        for i, target in enumerate(actual["flange_targets_wxyz"]):
            source_target = pose_matrix(self.source["flange_targets_wxyz"][i])
            np.testing.assert_allclose(pose_matrix(target), self.registration @ source_target, atol=1e-12)
            np.testing.assert_allclose(
                np.linalg.solve(obj, pose_matrix(target)),
                np.linalg.solve(pose_matrix(self.source["obj_pose"]), source_target),
                atol=1e-12,
            )
        self.assertTrue(all(c["passed"] for c in checks))
        self.assertEqual(actual["joint_names"], list(JOINTS))
        self.assertEqual(actual["quaternion_order"], "WXYZ")

    def test_scale_does_not_change_world_translation(self):
        small = deepcopy(self.source)
        small["obj_scale"] = 0.06
        a, _ = register_record(small, self.registration)
        b, _ = register_record(self.source, self.registration)
        np.testing.assert_array_equal(a["obj_pose"], b["obj_pose"])
        np.testing.assert_array_equal(a["flange_targets_wxyz"], b["flange_targets_wxyz"])

    def test_source_is_not_mutated_or_pregrasp_extrapolated_again(self):
        original = deepcopy(self.source)
        actual, _ = register_record(self.source, np.eye(4))
        np.testing.assert_allclose(actual["flange_targets_wxyz"], original["flange_targets_wxyz"], atol=1e-12)
        np.testing.assert_array_equal(self.source["flange_targets_wxyz"], original["flange_targets_wxyz"])
        np.testing.assert_array_equal(self.source["obj_pose"], original["obj_pose"])

    def test_reject_nonrigid_reflected_and_nonfinite_registration(self):
        for matrix in (np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                register_record(self.source, matrix)

    def test_reject_unverified_frame_and_extrapolation(self):
        for key, value in (("source_frame", "left_hand_flange"), ("pregrasp_extrapolation", 1)):
            source = {**self.source, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                register_record(source, np.eye(4))

    def test_reject_invalid_quaternion(self):
        for value in (np.zeros(7), np.ones(6), np.full(7, np.nan)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pose_matrix(value)

    def test_mesh_scale_is_applied_once_before_world_pose(self):
        import trimesh

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "urdf/meshes").mkdir(parents=True)
            trimesh.creation.box(extents=[1, 2, 3]).export(root / "urdf/meshes/convex_piece_0.obj")
            record = {"obj_path": str(root), "obj_scale": 0.1, "obj_pose": np.array([4, 5, 6, 1, 0, 0, 0])}
            bounds = object_bounds(record)
            np.testing.assert_allclose(bounds["world_dimensions_m"], [0.1, 0.2, 0.3])
            np.testing.assert_allclose(bounds["world_aabb_m"], [[3.95, 4.9, 5.85], [4.05, 5.1, 6.15]])

    def test_static_state_uses_joint_names_even_when_xml_order_is_reversed(self):
        bodies = "".join(
            f'<body name="b{i}"><joint name="{name}" type="hinge"/><geom type="sphere" size="0.01" mass="0.1"/></body>'
            for i, name in enumerate(reversed(JOINTS))
        )
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>" + bodies + '<body name="object"><freejoint name="object_freejoint"/>'
            '<geom type="sphere" size="0.01" mass="0.1"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        q = np.linspace(-0.5, 0.5, 28)
        record = {"pregrasp_qpos": q, "obj_pose": np.array([0.2, 0.3, 0.4, 1, 0, 0, 0])}
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("Preview must never step physics")):
            set_state(model, data, record, "pregrasp_qpos")
        for name, expected in zip(JOINTS, q):
            self.assertEqual(data.qpos[model.joint(name).qposadr[0]], expected)
        self.assertEqual(data.time, 0)

    def test_reopen_rejects_model_and_report_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.xml"
            model.write_text("model-v1")
            suite = root / "suite.json"
            suite.write_text("{}")
            report = {
                "suite_manifest": str(suite),
                "suite_sha256": sha256(suite),
                "dependencies": [{"path": str(model), "sha256": sha256(model)}],
            }
            path = root / "preview_report.json"
            json_write(path, report)
            path.with_suffix(".sha256").write_text(sha256(path))
            with patch("ada_grasp_ctrl.experiments.crx_sharpa_right_preparation.load_suite", return_value={}):
                self.assertEqual(load_preview(root), report)
                model.write_text("model-v2")
                with self.assertRaisesRegex(ValueError, "dependency changed"):
                    load_preview(root)
                path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "report changed"):
                    load_preview(root)


if __name__ == "__main__":
    unittest.main()
