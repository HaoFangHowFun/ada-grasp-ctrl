"""Sharpa frame, immutable-input, and native-statistics contracts."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.hugs_sharpa_input import FIELDS, adapt_prediction
from ada_grasp_ctrl.experiments.phase_comparison import sha256, write_json
from ada_grasp_ctrl.experiments.sharpa_model import FLANGE, HAND_JOINTS, JOINTS, ROBOT
from ada_grasp_ctrl.experiments.sharpa_regression import PROTOCOL, load_suite, phase_config, run_suite
from ada_grasp_ctrl.tasks.control_eval import SUPPORTED_HANDS, _validate_control_preflight
from ada_grasp_ctrl.tasks.control_stat import _build_grasp_controller
from ada_grasp_ctrl.errors import PreflightError


class SharpaInputTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        obj = self.root / "object"
        for folder in ("mesh", "info", "urdf/meshes"):
            (obj / folder).mkdir(parents=True, exist_ok=True)
        (obj / "mesh/simplified.obj").write_text("v 0 0 0\n")
        (obj / "urdf/meshes/convex_piece_0.obj").write_text("v 0 0 0\n")
        write_json(obj / "info/simplified.json", {"mass": 1, "density": 1, "scale": 1})
        self.scene = self.root / "tabletop_ur10e/scene.npy"
        self.scene.parent.mkdir()
        self.scene_value = {
            "task": {"obj_name": "object"},
            "scene": {
                "object": {
                    "file_path": "../object/mesh/simplified.obj",
                    "scale": np.array([0.081] * 3),
                    "pose": np.array([0, 0, 0.05, 1, 0, 0, 0]),
                }
            },
        }
        np.save(self.scene, self.scene_value)
        self.metadata = {"joint_names": list(HAND_JOINTS), "wrist_body_names": [FLANGE]}
        q = Rotation.from_euler("XYZ", [0.4, -0.7, 1.1]).as_quat()
        pose = np.r_[[0.2, -0.1, 0.3], q[[3, 0, 1, 2]]]
        self.raw = {field: np.r_[pose, np.linspace(0, 0.2, 22), np.zeros(29)] for field in FIELDS}
        self.raw.update(scene_path=str(self.scene), pred_grasp_type_id=3)
        self.raw["pregrasp_qpos"][:3] += [0.01, 0.02, 0.03]
        self.raw["pregrasp_qpos"][7:29] -= 0.05

    def test_world_flange_and_intrinsic_rotation_without_shadow_offset(self):
        result = adapt_prediction(self.raw, self.metadata, self.root)
        np.testing.assert_array_equal(result["grasp_qpos"][:3], self.raw["grasp_qpos"][:3])
        np.testing.assert_allclose(result["grasp_qpos"][3:6], [0.4, -0.7, 1.1], atol=1e-12)
        self.assertEqual(result["joint_names"], list(JOINTS))
        self.assertEqual(result["source_frame"], FLANGE)
        self.assertEqual(result["grasp_qpos"].shape, (28,))

    def test_extrapolation_once_leaves_grasp_squeeze_and_raw_unchanged(self):
        saved = deepcopy(self.raw)
        result = adapt_prediction(self.raw, self.metadata, self.root)
        np.testing.assert_allclose(result["pregrasp_qpos"][:3], [0.22, -0.06, 0.36])
        np.testing.assert_allclose(result["pregrasp_qpos"][6:], saved["grasp_qpos"][7:29] - 0.1)
        for field in FIELDS:
            np.testing.assert_array_equal(self.raw[field], saved[field])
        for field in FIELDS[1:]:
            np.testing.assert_array_equal(result[field][6:], self.raw[field][7:29])

    def test_named_joint_permutation_not_dictionary_order(self):
        canonical = adapt_prediction(self.raw, self.metadata, self.root)
        self.metadata["joint_names"].reverse()
        for field in FIELDS:
            self.raw[field][7:29] = self.raw[field][7:29][::-1]
        permuted = adapt_prediction(self.raw, self.metadata, self.root)
        for field in FIELDS:
            np.testing.assert_array_equal(canonical[field], permuted[field])

    def test_wrong_hand_frame_and_type_are_rejected(self):
        for metadata in (
            {**self.metadata, "wrist_body_names": ["rh_palm"]},
            {**self.metadata, "joint_names": [n.replace("right_", "left_") for n in HAND_JOINTS]},
        ):
            with self.assertRaises(ValueError):
                adapt_prediction(self.raw, metadata, self.root)
        self.raw["pred_grasp_type_id"] = 5
        with self.assertRaises(ValueError):
            adapt_prediction(self.raw, self.metadata, self.root)

    def test_bad_dimensions_quaternion_and_inactive_slot_nan_rejected(self):
        for value in (np.zeros(29), np.zeros(58), np.r_[self.raw["grasp_qpos"][:57], np.nan]):
            raw = dict(self.raw, grasp_qpos=value)
            with self.assertRaises(ValueError):
                adapt_prediction(raw, self.metadata, self.root)

    def test_floating_and_nonuniform_scale_rejected(self):
        floating = self.root / "floating/scene.npy"
        floating.parent.mkdir()
        np.save(floating, self.scene_value)
        with self.assertRaisesRegex(ValueError, "tabletop"):
            adapt_prediction(dict(self.raw, scene_path=str(floating)), self.metadata, self.root)
        self.scene_value["scene"]["object"]["scale"] = [0.081, 0.081, 0.1]
        np.save(self.scene, self.scene_value)
        with self.assertRaisesRegex(ValueError, "uniform"):
            adapt_prediction(self.raw, self.metadata, self.root)


class SharpaFrozenSuiteTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dependency = self.root / "asset"
        dependency.write_text("frozen")
        rows = []
        for i, scale in enumerate(PROTOCOL["scales"]):
            row = {"sample_id": f"g{i:04d}", "sample_index": i * 10, "scale": scale, "status": "prepared"}
            for field in ("source", "scene", "prepared"):
                row["source" if field == "source" else f"{field}_path"] = str(dependency)
                row[f"{field}_sha256"] = sha256(dependency)
            rows.append(row)
        self.suite = {
            "protocol": PROTOCOL,
            "robot": ROBOT,
            "smoke": True,
            "samples": rows,
            "dependencies": [{"path": str(dependency), "sha256": sha256(dependency)}],
        }
        self.path = self.root / "suite_manifest.json"
        self.save()

    def save(self):
        write_json(self.path, self.suite)
        self.path.with_suffix(".sha256").write_text(sha256(self.path))

    def test_corrupt_manifest_and_dependency_fail_closed(self):
        self.assertEqual(len(load_suite(self.path)["samples"]), 4)
        self.path.write_text(self.path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "manifest"):
            load_suite(self.path)
        self.save()
        (self.root / "asset").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_suite(self.path)

    def test_duplicate_identity_is_rejected(self):
        self.suite["samples"][1]["sample_index"] = 0
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_suite(self.path)

    def test_smoke_cannot_be_used_as_full_suite(self):
        with self.assertRaisesRegex(ValueError, "smoke-only"):
            run_suite(self.path, self.root / "not-created", False, 1)
        self.assertFalse((self.root / "not-created").exists())

    def test_incompatible_baseline_fails_before_execution(self):
        old = self.root / "old.json"
        write_json(old, {"suite_sha256": "other"})
        with self.assertRaisesRegex(ValueError, "Baseline mismatch"):
            run_suite(self.path, self.root / "not-created", True, 1, old)
        self.assertFalse((self.root / "not-created").exists())

    def test_private_config_keeps_public_support_and_no_double_extrapolation(self):
        suite = {"data_root": str(self.root), "asset_root": str(self.root), "prepared_root": str(self.root)}
        config = phase_config(suite, "phase2", self.root / "run", 1)
        self.assertFalse(config.task.arm_pregrasp_is_grasp)
        self.assertEqual(config.task.graspdata.pregrasp_t, 0)
        self.assertEqual(config.task.graspdata.squeeze_t, 0)
        self.assertNotIn(ROBOT, SUPPORTED_HANDS)
        with self.assertRaises(PreflightError):
            _validate_control_preflight(config)


class StatisticsKinematicsTest(unittest.TestCase):
    def test_statistics_respects_explicit_model_and_keeps_legacy_defaults(self):
        for hand, options, prefix, kind in (
            ("dummy_arm_shadow", {}, "rh_", "mjcf"),
            (ROBOT, {"robot_prefix": "", "kinematics_file_type": "urdf"}, "", "urdf"),
        ):
            with (
                self.subTest(hand=hand),
                patch("ada_grasp_ctrl.tasks.control_stat.RobotFactory.create_robot") as create,
                patch("ada_grasp_ctrl.utils.pin_helper.PinocchioHelper") as helper,
                patch("ada_grasp_ctrl.utils.robot_adaptor.RobotAdaptor"),
                patch("ada_grasp_ctrl.tasks.control_stat.GraspController"),
            ):
                _build_grasp_controller(OmegaConf.create({"hand_name": hand, "hand": options}))
                create.assert_called_once_with(robot_type=hand, prefix=prefix)
                create.return_value.get_file_path.assert_called_once_with(kind)
                self.assertEqual(helper.call_args.kwargs["robot_file_type"], kind)


if __name__ == "__main__":
    unittest.main()
