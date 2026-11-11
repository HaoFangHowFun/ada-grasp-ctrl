"""Overhead endpoint geometry and exact reviewed phase sequence regressions."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import test_grasp_pipeline as fixtures
from ada_grasp_ctrl.pipeline.approach import AboveObjectTargetError, above_object_target
from ada_grasp_ctrl.pipeline.plan import validate_plan, sample_plan, phase_progress, check_start
from ada_grasp_ctrl.experiments.right_ros_sender import digest, load_packet
from ada_grasp_ctrl.ros_execution.review import check_reviewed_goal
from ada_grasp_ctrl.ros_execution.settings import bridge_config
from ada_grasp_ctrl.ros_execution.motion import arrival_reached


class CartesianKinematics:
    bounds = np.array([[-2.0] * 6, [2.0] * 6])

    def forward(self, q: np.ndarray) -> np.ndarray:
        result = np.eye(4)
        result[:3, 3] = q[:3]
        result[:3, :3] = Rotation.from_rotvec(q[3:6].copy()).as_matrix()
        return result


class ApproachTest(unittest.TestCase):
    def test_nonblocking_finger_errors_keep_arm_and_finite_feedback_gates(self) -> None:
        config = bridge_config({"hand_arrival_required": False, "hand_tolerance": 0.06})
        error = [0.0] * 28
        error[6:9] = [0.0924, 0.08675, 0.0784]
        self.assertTrue(arrival_reached(error, config))
        self.assertFalse(arrival_reached(error, {**config, "hand_arrival_required": True}))
        check_start(error, [0.0] * 28, ignore_hand_error=True)
        with self.assertRaises(ValueError):
            check_start(error, [0.0] * 28)
        error[0] = 0.0101
        self.assertFalse(arrival_reached(error, config))
        with self.assertRaises(ValueError):
            check_start(error, [0.0] * 28, ignore_hand_error=True)
        error[0], error[6] = 0.0, float("nan")
        with self.assertRaises(ValueError):
            arrival_reached(error, config)
        with self.assertRaises(ValueError):
            check_start(error, [0.0] * 28, ignore_hand_error=True)
        with self.assertRaises(ValueError):
            bridge_config({"hand_arrival_required": "false"})

    def test_extrapolated_fingers_recompute_clearance_without_mutating_grasp(self) -> None:
        adapter = SimpleNamespace(
            kin=CartesianKinematics(), model_T_world=np.eye(4), limits=np.array([[-2, 2]] * 28, dtype=float)
        )
        pre = np.r_[[0.4, -0.4, 0.5, 0, 0, 0], [0.2] * 22]
        grasp = np.r_[pre[:6], [0.3] * 22]
        original = pre.copy()
        geometry = SimpleNamespace(measure=lambda q: {"gap_m": float(q[2] - 0.1 - q[6])})
        with patch("ada_grasp_ctrl.pipeline.approach.HandGeometry", return_value=geometry):
            result = above_object_target(adapter, pre, [[0, 0, 0.4]], 0.1, grasp=grasp, hand_extrapolation_factor=1.5)
            np.testing.assert_allclose(result["positions"][6:], [0.15] * 22)
            self.assertAlmostEqual(result["pregrasp_lowest_world_z_m"], 0.2)
            self.assertAlmostEqual(result["approach_hand_lowest_world_z_m"], 0.25)
            self.assertAlmostEqual(result["lowest_world_z_m"], 0.5)
            np.testing.assert_array_equal(pre, original)
            adapter.limits[6, 0] = 0.18
            with self.assertRaisesRegex(AboveObjectTargetError, "joint limits"):
                above_object_target(adapter, pre, [[0, 0, 0.4]], 0.1, grasp=grasp, hand_extrapolation_factor=1.5)

    def test_arrival_tolerance_does_not_relax_initial_start_or_arm(self) -> None:
        expected = [0.0] * 28
        actual = expected.copy()
        actual[6] = 0.0482
        with self.assertRaises(ValueError):
            check_start(actual, expected)
        check_start(actual, expected, hand_tolerance=0.06)
        actual[0] = 0.0101
        with self.assertRaises(ValueError):
            check_start(actual, expected, hand_tolerance=0.06)
        actual[0], actual[6] = 0.0, 0.0601
        with self.assertRaises(ValueError):
            check_start(actual, expected, hand_tolerance=0.06)

    def target(self, top: float) -> tuple:
        adapter = SimpleNamespace(kin=CartesianKinematics(), model_T_world=np.eye(4), limits=np.array([[-2, 2]] * 28))
        q = np.r_[[0.4, -0.4, 0.3, 0, 0, 0], np.linspace(0, 0.1, 22)]
        geometry = SimpleNamespace(measure=lambda joints: {"gap_m": float(joints[2] - 0.1)})
        with patch("ada_grasp_ctrl.pipeline.approach.HandGeometry", return_value=geometry):
            result = above_object_target(adapter, q, [[0, 0, top - 0.05], [0, 0, top]], 0.1)
        return q, result

    def test_lowest_hand_clears_object_top_and_retains_pregrasp(self) -> None:
        original, result = self.target(0.25)
        self.assertAlmostEqual(result["lowest_world_z_m"] - result["object_top_world_z_m"], 0.1)
        np.testing.assert_allclose(result["positions"][:2], original[:2], atol=1e-8)
        np.testing.assert_array_equal(result["positions"][6:], original[6:])
        np.testing.assert_allclose(result["positions"][3:6], original[3:6], atol=1e-8)

    def test_already_high_pregrasp_is_not_lowered(self) -> None:
        original, result = self.target(0.0)
        self.assertEqual(result["height_m"], 0)
        np.testing.assert_array_equal(result["positions"], original)

    def test_unreachable_above_target_is_not_replaced_by_another_grasp(self) -> None:
        with patch("ada_grasp_ctrl.pipeline.approach.lift_target", side_effect=ValueError("unreachable")):
            with self.assertRaisesRegex(AboveObjectTargetError, "IK failed"):
                self.target(0.25)

    def test_eight_phase_hardware_plan_is_bound_to_preview(self) -> None:
        fixture = fixtures.PipelineTransportTest()
        with tempfile.TemporaryDirectory() as directory:
            path, packet = fixture.write_packet(Path(directory), lift=True, complete=True, environment="hardware")
            plan = packet["reviewed_plan"]
            pregrasp = plan["keyframes"][1]
            q = [v + 0.01 for v in pregrasp[:6]] + pregrasp[6:]
            above = dict(
                positions=q,
                clearance_m=0.1,
                object_top_world_z_m=0.2,
                pregrasp_lowest_world_z_m=0.2,
                lowest_world_z_m=0.3,
                height_m=0.1,
                start_flange_wxyz=[0, 0, 0, 1, 0, 0, 0],
                target_flange_wxyz=[0, 0, 0.1, 1, 0, 0, 0],
                frame_id="world",
                axis="+Z",
                hand_policy="retain_selected_pregrasp",
                solver_success=True,
                used_for_candidate_selection=False,
                path_checked=False,
                collision_checked=False,
            )
            plan["above_object"] = above
            plan["phases"].insert(0, "above_object")
            plan["keyframes"].insert(1, q)
            plan["durations_s"].insert(0, 2.2)
            packet["targets"].insert(
                0,
                dict(phase="above_object", positions=q, duration_s=2.2, flange_target_wxyz=above["target_flange_wxyz"]),
            )
            plan_path = Path(plan_source := packet["source"]["plan_path"])
            plan_path.write_text(json.dumps(plan))
            packet["source"]["plan_sha256"] = digest(Path(plan_source))
            fixture.save_packet(path, packet)
            self.assertEqual(load_packet(path), packet)
            for i, target in enumerate(packet["targets"]):
                check_reviewed_goal(
                    packet, i, target["positions"], target["duration_s"], bridge_config({"environment": "hardware"})
                )
                np.testing.assert_allclose(
                    sample_plan(plan, phase_progress(plan, i + 1)), target["positions"], atol=1e-9
                )
            changed = deepcopy(plan)
            changed["keyframes"][1][6] += 0.01
            with self.assertRaisesRegex(ValueError, "pregrasp fingers"):
                validate_plan(changed)
            packet["targets"][0]["flange_target_wxyz"] = [0, 0, 0.2, 1, 0, 0, 0]
            fixture.save_packet(path, packet)
            with self.assertRaisesRegex(ValueError, "Above-object pose differs"):
                load_packet(path)


if __name__ == "__main__":
    unittest.main()
