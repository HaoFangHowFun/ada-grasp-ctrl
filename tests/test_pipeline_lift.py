"""Post-selection lift generation must not call candidate or collision gates."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.pipeline.lift import lift_target
from ada_grasp_ctrl.pipeline.plan import sample_plan, validate_plan
from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, LIFT_PHASES, validate_packet


class CartesianKinematics:
    bounds = np.array([[-10.0] * 6, [10.0] * 6])

    def forward(self, q: np.ndarray) -> np.ndarray:
        result = np.eye(4)
        result[:3, 3] = q[:3]
        result[:3, :3] = Rotation.from_rotvec(q[3:6].copy()).as_matrix()
        return result


class PipelineLiftTest(unittest.TestCase):
    def adapter(self) -> SimpleNamespace:
        model_world = np.eye(4)
        model_world[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        return SimpleNamespace(
            kin=CartesianKinematics(),
            model_T_world=model_world,
            static_check=lambda q: self.fail("Lift must not invoke collision checks"),
            evaluate=lambda *args: self.fail("Lift must not evaluate/reselect a candidate"),
        )

    def test_world_vertical_lift_preserves_hand_and_orientation_without_collision_checks(self) -> None:
        squeeze = np.r_[[0.2, 0.3, 0.4, 0.1, 0.2, 0.3], np.linspace(0, 1, 22)]
        original = squeeze.copy()
        result = lift_target(self.adapter(), squeeze, 0.05)
        np.testing.assert_array_equal(squeeze, original)
        np.testing.assert_array_equal(result["positions"][6:], squeeze[6:])
        delta = np.asarray(result["actual_flange_wxyz"][:3]) - result["start_flange_wxyz"][:3]
        np.testing.assert_allclose(delta, [0, 0, 0.05], atol=1e-7)
        np.testing.assert_allclose(result["actual_flange_wxyz"][3:], result["start_flange_wxyz"][3:], atol=1e-7)
        self.assertFalse(result["used_for_candidate_selection"])
        self.assertFalse(result["path_checked"])
        self.assertFalse(result["collision_checked"])

    def test_failed_conversion_cannot_return_a_command(self) -> None:
        fit = SimpleNamespace(x=np.zeros(6), success=False, message="did not converge")
        with patch("ada_grasp_ctrl.pipeline.lift.least_squares", return_value=fit):
            with self.assertRaisesRegex(ValueError, "Selected candidate retained"):
                lift_target(self.adapter(), np.zeros(28), 0.05)

    def test_nonfinite_residual_cannot_return_a_command(self) -> None:
        with patch(
            "ada_grasp_ctrl.pipeline.lift.residual",
            return_value={"position_m": float("nan"), "orientation_rad": 0.0},
        ):
            with self.assertRaisesRegex(ValueError, "Selected candidate retained"):
                lift_target(self.adapter(), np.zeros(28), 0.05)

    def test_four_phase_plan_and_packet_preserve_squeeze_hand(self) -> None:
        squeeze = np.zeros(28)
        lift = lift_target(self.adapter(), squeeze, 0.05)
        plan = {
            "schema_version": 1,
            "environment": "mock",
            "joint_names": list(JOINT_NAMES),
            "phases": list(LIFT_PHASES),
            "keyframes": [squeeze.tolist()] * 4 + [lift["positions"]],
            "durations_s": [2.2] * 4,
            "settle_time_s": 0.3,
            "interpolation": "quintic_smoothstep",
            "lift": lift,
        }
        validate_plan(plan)
        np.testing.assert_array_equal(sample_plan(plan, 100), lift["positions"])
        packet = {
            "schema_version": 1,
            "kind": "ada_right_joint_targets",
            "environment": "mock",
            "units": "rad",
            "joint_names": list(JOINT_NAMES),
            "reference_frame": "world",
            "pose_frame": "right_hand_flange",
            "execution_phases": list(LIFT_PHASES),
            "kinematics": {"fixture": True},
            "targets": [
                {"phase": phase, "positions": q, "flange_target_wxyz": [0, 0, 0, 1, 0, 0, 0]}
                for phase, q in zip(LIFT_PHASES, plan["keyframes"][1:])
            ],
        }
        validate_packet(packet)
        changed = deepcopy(plan)
        changed["keyframes"][-1][6] += 0.01
        with self.assertRaisesRegex(ValueError, "preserve"):
            validate_plan(changed)
        changed = deepcopy(packet)
        changed.pop("execution_phases")
        with self.assertRaises(ValueError):
            validate_packet(changed)


if __name__ == "__main__":
    unittest.main()
