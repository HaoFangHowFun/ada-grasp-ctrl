"""Saved target previews must not alter the object pose, physics, or source artifacts."""

import hashlib
import os
from pathlib import Path
import unittest

import numpy as np

from ada_grasp_ctrl.experiments.grasp_pose_preview import GraspPosePreview


@unittest.skipUnless(os.environ.get("ADA_GRASP_PREVIEW_ROOT"), "Requires saved thin-box and cheese-box runs")
class GraspPoseTests(unittest.TestCase):
    def test_saved_targets_preserve_object_and_passive_state_for_all_stages(self) -> None:
        for name in ("thin_box_original", "cheese_box_original"):
            root = Path(os.environ["ADA_GRASP_PREVIEW_ROOT"]) / name
            hashes = {
                p: hashlib.sha256((root / p).read_bytes()).digest()
                for p in ("scene.mjb", "selection.json", "initial_state.npz")
            }
            preview = GraspPosePreview(root)
            obj = preview.data.joint("object_free").qpos.copy()
            for stage in ("pregrasp", "grasp", "squeeze"):
                preview.set_stage(stage)
                np.testing.assert_array_equal(
                    preview.data.qpos[preview.indices], preview.selected["stages"][stage]["q56"]
                )
                np.testing.assert_array_equal(preview.data.joint("object_free").qpos, obj)
                np.testing.assert_array_equal(preview.data.qvel, 0)
                self.assertEqual(preview.data.time, 0)
            with self.assertRaises(ValueError):
                preview.set_stage("lift")
            for p, value in hashes.items():
                self.assertEqual(hashlib.sha256((root / p).read_bytes()).digest(), value)

    def test_context_group_does_not_modify_collision_or_motor_properties(self) -> None:
        preview = GraspPosePreview(Path(os.environ["ADA_GRASP_PREVIEW_ROOT"]) / "thin_box_original")
        fields = (
            "geom_contype",
            "geom_conaffinity",
            "geom_friction",
            "geom_solref",
            "geom_solimp",
            "body_mass",
            "actuator_gainprm",
        )
        values = {k: getattr(preview.model, k).copy() for k in fields}
        q = preview.data.qpos.copy()
        preview.assign_context_group()
        for k, value in values.items():
            np.testing.assert_array_equal(getattr(preview.model, k), value)
        np.testing.assert_array_equal(preview.data.qpos, q)
        collisions = np.isin(preview.original_groups, [0, 3])
        np.testing.assert_array_equal(preview.model.geom_group[collisions], preview.original_groups[collisions])
        self.assertTrue(np.any(preview.model.geom_group == 4))
        keep = preview.hand_ids | {preview.object_id}
        self.assertTrue(
            all(preview.model.geom_bodyid[g] not in keep for g in np.flatnonzero(preview.model.geom_group == 4))
        )


if __name__ == "__main__":
    unittest.main()
