"""Contracts for the isolated right-arm smoke lifecycle and failure accounting."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from ada_grasp_ctrl.experiments.crx_sharpa_right_approach import overhead_pose, interpolate_pose
from ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation import (
    ApproachInvalidError,
    LiftIKError,
    RightEvaluator,
    classify,
    run_episode,
    sample_preflight,
    tracking_diagnostics,
    verify_pairs,
)
from ada_grasp_ctrl.experiments.crx_sharpa_right_model import ACTUATORS, JOINTS
from ada_grasp_ctrl.experiments.crx_sharpa_right_simulation import ApprovedScene, exclude_base_joint_contact
from ada_grasp_ctrl.experiments.sharpa_model import FLANGE
from ada_grasp_ctrl.utils.hand_util import MjHO


class RightSmokeTest(unittest.TestCase):
    def test_overhead_is_palm_30cm_above_object_center(self):
        palm = np.eye(4)
        palm[:3, 3] = [0.2, -0.9, 0.1]
        obj = np.array([0.8, -0.7, 0.02, 1, 0, 0, 0])
        target = overhead_pose(palm, obj)
        np.testing.assert_allclose(target[:3, 3], [0.8, -0.7, 0.32])
        np.testing.assert_array_equal(target[:3, :3], palm[:3, :3])
        np.testing.assert_array_equal(palm[:3, 3], [0.2, -0.9, 0.1])

    def test_cartesian_interpolation_preserves_height_during_overhead_rotation(self):
        from scipy.spatial.transform import Rotation

        start = np.eye(4)
        start[:3, 3] = [0.8, -0.7, 0.32]
        end = start.copy()
        end[:3, :3] = Rotation.from_euler("xyz", [0.3, 0.8, -0.2]).as_matrix()
        path = interpolate_pose(start, end, 20)
        self.assertEqual(len(path), 20)
        for pose in path:
            np.testing.assert_allclose(pose[:3, 3], start[:3, 3])
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(path[-1], end, atol=1e-12)

    @staticmethod
    def approach_evaluator():
        evaluator = object.__new__(RightEvaluator)
        evaluator.approach_diagnostics = []
        evaluator.motion = Mock()
        evaluator.mj_ho = SimpleNamespace(
            collision_history=[],
            get_curr_contact_info=lambda: [],
            contact_diagnostics=lambda: dict(
                arm_penetration_m=0, hand_object_penetration_m=0, hand_self_penetration_m=0, hand_table_penetration_m=0
            ),
        )
        return evaluator

    def test_approach_waits_for_actual_overhead_arrival_before_descent(self):
        evaluator = self.approach_evaluator()
        plan = {
            "segments": [
                dict(name=name, path=[np.zeros(28).tolist()], target_palm_wxyz=[0] * 7)
                for name in ("orient_above_object", "descend_to_pregrasp")
            ]
        }
        errors = [{"position_m": value, "orientation_rad": 0} for value in (0.02, 0.0005, 0.0005)]
        with patch("ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation.actual_palm_error", side_effect=errors):
            evaluator.overhead_approach(plan)
        self.assertEqual(
            [c.args[1] for c in evaluator.motion.call_args_list],
            ["orient_above_object", "settle_orient_above_object", "descend_to_pregrasp"],
        )

    def test_approach_never_descends_after_arrival_timeout(self):
        evaluator = self.approach_evaluator()
        plan = {
            "segments": [
                dict(name=name, path=[np.zeros(28).tolist()], target_palm_wxyz=[0] * 7)
                for name in ("orient_above_object", "descend_to_pregrasp")
            ]
        }
        with patch(
            "ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation.actual_palm_error",
            return_value={"position_m": 0.02, "orientation_rad": 0},
        ):
            with self.assertRaisesRegex(ApproachInvalidError, "Did not reach"):
                evaluator.overhead_approach(plan)
        self.assertEqual(evaluator.motion.call_count, 31)
        self.assertNotIn("descend_to_pregrasp", [c.args[1] for c in evaluator.motion.call_args_list])

    def test_approach_rejects_object_contact_before_overhead(self):
        evaluator = self.approach_evaluator()
        evaluator.mj_ho.get_curr_contact_info = lambda: [{"body1_name": "finger"}]
        plan = {"segments": [dict(name="raise_clearance", path=[np.zeros(28).tolist()], target_palm_wxyz=[0] * 7)]}
        with self.assertRaisesRegex(ApproachInvalidError, "Unexpected contact"):
            evaluator.overhead_approach(plan)

    def test_classification_keeps_degradation_out_of_success_denominator(self):
        args = dict(
            error=None,
            invalid=False,
            degraded=False,
            lift_error=False,
            lift_completed=True,
            initial_z=0.02,
            terminal_z=0.22,
        )
        self.assertEqual(classify(**args), "success")
        for updates, expected in [
            ({"terminal_z": 0.02}, "physical_failure"),
            ({"lift_error": True}, "ik_invalid"),
            ({"degraded": True, "lift_error": True}, "solver_degraded"),
            ({"invalid": True, "degraded": True}, "initialization_invalid"),
            ({"error": "failed", "invalid": True}, "execution_error"),
            ({"lift_completed": False}, "execution_error"),
        ]:
            self.assertEqual(classify(**(args | updates)), expected)

    def test_invalid_targets_keep_both_ids_without_physics(self):
        row = dict(sample_id="g0020", sample_index=20, scale=0.106, object_id="object", all_targets_reachable=False)
        preflight = sample_preflight(row, {}, {})
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation.RightEvaluator") as constructor,
        ):
            results = [run_episode(row, {}, Path(directory), method, preflight) for method in ("op", "ours")]
        constructor.assert_not_called()
        self.assertEqual([r["status"] for r in results], ["ik_invalid"] * 2)
        self.assertEqual([r["physics_steps"] for r in results], [0, 0])
        self.assertEqual(verify_pairs(results)["pairs_rejected_before_physics"], ["g0020"])
        results[1]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed/preflight mismatch"):
            verify_pairs(results)

    def test_named_reset_with_reversed_xml_joints_and_actuators(self):
        # Deliberately scramble both simulator orders independently of canonical data.
        bodies = "".join(
            f'<body name="body{i}" pos="0 0 {i + 1}"><joint name="{name}"/>'
            '<geom type="sphere" size="0.01" mass="0.1"/></body>'
            for i, name in enumerate(reversed(JOINTS))
        )
        actuators = "".join(
            f'<position name="{act}" joint="{joint}" kp="1"/>' for act, joint in reversed(list(zip(ACTUATORS, JOINTS)))
        )
        xml = f'''<mujoco><worldbody>
        <body name="right_base_link"><body name="right_J1_link"/></body>
        <body name="{FLANGE}">{bodies}</body>
        <body name="object"><freejoint name="object_freejoint"/>
        <geom type="sphere" size="0.01" mass="0.1"/></body></worldbody>
        <actuator>{actuators}</actuator></mujoco>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.xml"
            path.write_text(xml)
            sim = ApprovedScene(path)
            q = np.arange(28) * 0.01
            pose = np.array([3, 4, 5, 1, 0, 0, 0.0])
            sim.reset_pose_qpos(q, pose)
            np.testing.assert_array_equal(sim.get_qpos_f(JOINTS), q)
            np.testing.assert_array_equal(sim.data.ctrl[sim.a_addresses], q)
            np.testing.assert_array_equal(sim.get_obj_pose(), pose)
            self.assertNotEqual(sim.q_addresses, list(range(28)))

    def test_base_joint_exclusion_preserves_fk_and_environment_contacts(self):
        spec = mujoco.MjSpec.from_string("""<mujoco><worldbody>
          <body name="table"><geom type="plane" size="2 2 0.1"/></body>
          <body name="right_base_link" pos="0 0 0.05">
            <geom type="sphere" size="0.1"/>
            <body name="right_J1_link" pos="0.15 0 0">
              <joint name="right_J1" type="hinge"/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </body></worldbody></mujoco>""")

        def state(model):
            data = mujoco.MjData(model)
            data.qpos[0] = 0.2
            mujoco.mj_forward(model, data)
            pairs = {frozenset(model.body(model.geom_bodyid[g]).name for g in c.geom) for c in data.contact}
            return data, pairs

        original = spec.compile()
        before, pairs_before = state(original)
        adjacent = frozenset(("right_base_link", "right_J1_link"))
        environment = frozenset(("table", "right_J1_link"))
        self.assertIn(adjacent, pairs_before)
        self.assertIn(environment, pairs_before)
        exclude_base_joint_contact(spec)
        exclude_base_joint_contact(spec)
        self.assertEqual(len(list(spec.excludes)), 1)
        corrected = spec.compile()
        after, pairs_after = state(corrected)
        self.assertNotIn(adjacent, pairs_after)
        self.assertEqual(pairs_after, pairs_before - {adjacent})
        np.testing.assert_array_equal(before.xpos, after.xpos)
        np.testing.assert_array_equal(before.xmat, after.xmat)
        np.testing.assert_array_equal(original.geom_contype, corrected.geom_contype)
        np.testing.assert_array_equal(original.geom_conaffinity, corrected.geom_conaffinity)

    def test_base_joint_exclusion_rejects_unrelated_bodies(self):
        spec = mujoco.MjSpec.from_string("""<mujoco><worldbody>
          <body name="right_base_link"/><body name="right_J1_link"/>
        </worldbody></mujoco>""")
        with self.assertRaisesRegex(ValueError, "Expected adjacent"):
            exclude_base_joint_contact(spec)

    def test_contact_filter_excludes_arm_and_table(self):
        sim = object.__new__(ApprovedScene)
        sim.hand_bodies = {"finger"}
        contacts = [{"body1_name": name, "body2_name": "object"} for name in ("finger", "right_J1_link", "table")]
        with patch.object(MjHO, "get_curr_contact_info", return_value=contacts):
            self.assertEqual(sim.get_curr_contact_info(), contacts[:1])
        with patch.object(MjHO, "get_contact_info", return_value=(contacts, [])):
            self.assertEqual(sim.get_contact_info(np.zeros(28), np.zeros(7)), (contacts[:1], []))

    def test_cartesian_lift_preserves_fingers_and_never_adds_height_to_j3(self):
        evaluator = object.__new__(RightEvaluator)
        actual = np.arange(28) * 0.01
        command = actual + 0.1
        evaluator.mj_ho = SimpleNamespace(
            get_qpos_f=lambda names: actual.copy(),
            get_qpos_a=lambda: command.copy(),
            get_obj_pose=lambda: np.array([0, 0, 0, 1, 0, 0, 0]),
        )
        evaluator.grasp_ctrl = SimpleNamespace(r_data={"obj_pose": []})
        evaluator.lift_diagnostics = []
        evaluator.motion = Mock()
        targets = []

        def solve(target, hand, seed):
            targets.append(target.copy())
            np.testing.assert_array_equal(hand, command[6:])
            q = np.r_[seed, hand]
            q[0] += 0.01
            return q, {"reachable": True, "arm_table_clear": True}

        kin = SimpleNamespace(forward=lambda q: np.eye(4), solve=solve)
        evaluator.cartesian_lift(kin)
        np.testing.assert_allclose([t[2, 3] for t in targets], np.linspace(0.01, 0.2, 20))
        for call in evaluator.motion.call_args_list:
            self.assertEqual(call.args[0][0, 2], actual[2])
            np.testing.assert_array_equal(call.args[0][0, 6:], command[6:])
        evaluator.motion.reset_mock()
        kin.solve = lambda target, hand, seed: (np.r_[seed, hand], {"reachable": False, "arm_table_clear": True})
        with self.assertRaises(LiftIKError):
            evaluator.cartesian_lift(kin)
        evaluator.motion.assert_not_called()

    def test_tracking_uses_named_addresses(self):
        frames = [dict(segment="squeeze", command=np.zeros(28), qpos=np.arange(35.0))]
        result = tracking_diagnostics(frames, list(range(20, 26)))
        self.assertEqual(result["squeeze"]["terminal_error_rad"], 25.0)


if __name__ == "__main__":
    unittest.main()
