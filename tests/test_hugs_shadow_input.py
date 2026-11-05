"""Frame and joint-order contracts for the explicit HUGS Shadow bridge."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.hugs_shadow_input import ADA_JOINTS, FIELDS, adapt_prediction


class HugsShadowInputTest(unittest.TestCase):
    def test_rotated_palm_offset_and_named_joint_targets(self):
        """A rotated offset is removed once; all three named targets survive."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scene.npy").touch()
            metadata = {"joint_names": ADA_JOINTS[::-1], "wrist_body_names": ["rh_palm"]}
            rotation = Rotation.from_euler("xyz", [0.3, 1.1, -0.4])
            quat = rotation.as_quat()[[3, 0, 1, 2]]
            raw = {"scene_path": "scene.npy", "pred_grasp_type_id": 3}
            for index, field in enumerate(FIELDS):
                raw[field] = np.r_[[1, 2, 3], quat, np.arange(22) + index, np.zeros(29)]
            output = adapt_prediction(raw, metadata, root)
            for index, field in enumerate(FIELDS):
                np.testing.assert_allclose(output[field][:3] + rotation.apply([0, 0, 0.034]), [1, 2, 3])
                np.testing.assert_array_equal(output[field][7:], np.arange(22)[::-1] + index)
            raw["squeeze_qpos"][0] = np.nan
            with self.assertRaisesRegex(ValueError, "squeeze_qpos"):
                adapt_prediction(raw, metadata, root)
            raw["pred_grasp_type_id"] = 2
            with self.assertRaisesRegex(ValueError, "right_full"):
                adapt_prediction(raw, metadata, root)

    def test_rejects_wrong_wrist_and_duplicate_joints(self):
        metadata = {"joint_names": ADA_JOINTS, "wrist_body_names": ["base"]}
        with self.assertRaisesRegex(ValueError, "rh_palm"):
            adapt_prediction({}, metadata, Path("/tmp"))
        metadata["joint_names"] = [ADA_JOINTS[0]] * 22
        with self.assertRaisesRegex(ValueError, "22 named"):
            adapt_prediction({}, metadata, Path("/tmp"))
