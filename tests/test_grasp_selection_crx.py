"""Actual CRX/Sharpa FK, static collision and endpoint selection tests."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.grasp_selection.adapters.crx_sharpa import CrxSharpaAdapter
from ada_grasp_ctrl.grasp_selection.adapters.state import check_description, world_base_from_description
from ada_grasp_ctrl.grasp_selection.selector import GraspSelector
from ada_grasp_ctrl.grasp_selection.types import (
    ARM_NAMES,
    HAND_NAMES,
    STAGES,
    GraspCandidate,
    RobotState,
    SelectionContext,
    StageTarget,
)
from ada_grasp_ctrl.experiments.crx_sharpa_right_model import matrix_pose


class RealCrxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = (
            project_root() / "output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json"
        )
        if not cls.report.exists():
            raise unittest.SkipTest("Existing CRX preparation assets unavailable; see selector integration docs")
        info = json.loads(cls.report.read_text())["model"]
        cls.world_base = info["world_T_base"]
        cls.adapter = CrxSharpaAdapter(
            cls.report, {"frame_id": "world", "unit": "m", "normal_xyz": [0, 0, 1], "d": 2}, cls.world_base
        )
        cls.q = np.r_[np.deg2rad([-90, -30, 210, 0, -60, 0]), np.zeros(22)]

    def test_named_real_model_and_independent_fk(self):
        self.assertEqual(self.adapter.model.nq, 28)
        self.assertEqual(tuple(self.adapter.model.joint(i).name for i in range(28)), ARM_NAMES + HAND_NAMES)
        self.assertLess(max(self.adapter.kin.audit(self.q).values()), 1e-6)

    def test_selected_fk_matches_original_and_no_physics(self):
        a = self.adapter
        target = a.kin.forward(self.q).copy()
        pose = matrix_pose(target)
        stages = {s: StageTarget(pose[:3], pose[3:], self.q[6:]) for s in STAGES}
        c = GraspCandidate("fk_roundtrip", stages, "FK_fixture", "fixture")
        state = RobotState(self.q[:6], 10.0, 10.1, a.model_id, "FK_fixture", "fixture")
        ctx = SelectionContext(state, self.world_base, "model URDF", "fixture", 10.1)
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("No physics permitted")):
            result = GraspSelector(a).select([c], ctx)
        self.assertEqual(result.selected_candidate_id, "fk_roundtrip")
        self.assertLess(result.cost, 1e-15)
        for stage in STAGES:
            check = result.selected["stage_checks"][stage]
            self.assertLessEqual(check["residual"]["position_m"], 1e-3)
            self.assertLessEqual(check["residual"]["orientation_rad"], 1e-2)
            self.assertEqual(check["joint_limit_violations"], [])
        self.assertFalse(result.execution_ready)

    def test_real_table_collision_rejected_and_clear_case_passes(self):
        self.assertTrue(self.adapter.static_check(self.q)["known_geometry_passed"])
        collision = CrxSharpaAdapter(
            self.report, {"frame_id": "world", "unit": "m", "normal_xyz": [0, 0, 1], "d": -2}, self.world_base
        )
        check = collision.static_check(self.q)
        self.assertFalse(check["known_geometry_passed"])
        self.assertGreater(check["arm_penetration_m"], 1e-5)

    def test_source_urdf_base_and_mount_mismatch(self):
        description = Path(self.adapter.info["urdf"]).read_text()
        np.testing.assert_allclose(world_base_from_description(description), self.world_base, atol=1e-12)
        check_description(description, description)
        changed = description.replace('xyz="0.03058936 -0.91156264 0.00873476"', 'xyz="0 0 0"')
        self.assertNotEqual(changed, description)
        with self.assertRaises(ValueError):
            check_description(description, changed)

    def test_context_mismatch_and_current_limits_rejected(self):
        state = RobotState([20] * 6, 10.0, 10.1, self.adapter.model_id, "fixture", "fixture")
        ctx = SelectionContext(state, self.world_base, "model", "cal", 10.1)
        with self.assertRaisesRegex(ValueError, "limits"):
            self.adapter.validate_context(ctx)


if __name__ == "__main__":
    unittest.main()
