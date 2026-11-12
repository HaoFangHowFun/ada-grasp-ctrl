"""Historical CRX protocol extension: world frames, passive state and native failures."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import pinocchio as pin
import yaml

from ada_grasp_ctrl.errors import ControlSolveEpisodeAbort
from ada_grasp_ctrl.experiments.dual_arm_crx_approach import prepare_overhead
from ada_grasp_ctrl.experiments.dual_arm_crx_episode import execute_native
from ada_grasp_ctrl.experiments.dual_arm_native_control import ActiveKinematics, ActiveRobot, ActiveSimulation
from ada_grasp_ctrl.experiments.simulated_object_cloud import ObjectScene
from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.utils.robot_adaptor import RobotAdaptor


class OverheadTests(unittest.TestCase):
    def scene(self, side: str) -> tuple:
        """Use Cartesian analytical IK to verify route semantics independently of solver behavior."""
        arm, block = side_slice(side, True), side_slice(side)
        q = np.zeros(56)
        q[arm][:3] = [0, 0, 0.5]
        target = q.copy()
        target[arm][:3] = [0.4, 0.1, 0.05]
        target[block][6:] = 0.4

        def fk(q: np.ndarray, *_: object) -> np.ndarray:
            pose = np.eye(4)
            pose[:3, 3] = q[arm][:3]
            return pose

        def solve(seed: np.ndarray, side: str, pose: np.ndarray) -> tuple:
            solved = seed.copy()
            solved[arm][:3] = pose[:3, 3]
            return solved, dict(accepted=True)

        robot = SimpleNamespace(
            configuration=lambda q: q.copy(),
            fk=fk,
            solve_arm_ik=solve,
            pin_model=SimpleNamespace(velocityLimit=np.ones(56) * 10),
            pin_v=np.arange(56),
        )
        sim = SimpleNamespace(configuration=lambda: q.copy(), check_static_contacts=lambda q: [])
        chosen = dict(selected_side=side, stages=dict(pregrasp=dict(q56=target.tolist())))
        return SimpleNamespace(robot=robot, sim=sim), chosen

    def test_both_sides_overhead_and_open_fingers_until_descent(self) -> None:
        for side in ("left", "right"):
            scene, chosen = self.scene(side)
            obj = np.eye(4)
            obj[:3, 3] = [0.4, 0.1, 0]
            before = scene.sim.configuration()
            with patch("ada_grasp_ctrl.experiments.dual_arm_crx_approach.body_pose", return_value=obj):
                plan = prepare_overhead(scene, chosen)
            self.assertEqual(plan["status"], "ready")
            segments = plan["segments"]
            self.assertEqual(
                [s["name"] for s in segments],
                ["raise_clearance", "move_above_object", "orient_above_object", "descend_to_pregrasp"],
            )
            np.testing.assert_allclose(np.array(segments[1]["target_palm"])[:3, 3], [0.4, 0.1, 0.3])
            for segment in segments:
                for q in np.array(segment["path"]):
                    passive = side_slice("right" if side == "left" else "left")
                    np.testing.assert_array_equal(q[passive], before[passive])
                    if segment["name"] != "descend_to_pregrasp":
                        np.testing.assert_array_equal(q[side_slice(side)][6:], 0)
            np.testing.assert_allclose(segments[-1]["path"][-1], chosen["stages"]["pregrasp"]["q56"])
            np.testing.assert_array_equal(scene.sim.configuration(), before)

    def test_failed_ik_is_reported_without_fallback_candidate(self) -> None:
        scene, chosen = self.scene("right")
        scene.robot.solve_arm_ik = lambda q, *_: (q, dict(accepted=False))
        with patch("ada_grasp_ctrl.experiments.dual_arm_crx_approach.body_pose", return_value=np.eye(4)):
            result = prepare_overhead(scene, chosen)
        self.assertEqual(result["reason"], "overhead_ik_failed")
        self.assertFalse(result["execution_ready"])


@unittest.skipUnless(os.environ.get("ADA_DUAL_ARM_SNAPSHOT"), "Requires explicit real Phase 1 snapshot")
class NativeSceneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = yaml.safe_load((project_root() / "config/simulation_scene.yaml").read_text())
        config["scene"]["robot_snapshot"] = os.environ["ADA_DUAL_ARM_SNAPSHOT"]
        self.scene = ObjectScene(config, self.root)

    def test_reduced_fk_and_jacobians_match_full_world_model_on_both_sides(self) -> None:
        scene = self.scene
        q = scene.sim.targets
        for side in ("left", "right"):
            robot = ActiveRobot(scene, side)
            helper = ActiveKinematics(scene.robot, side, q)
            adaptor = RobotAdaptor(helper, robot.dof_names, robot.doa_names, robot.doa2dof_matrix)
            self.assertEqual(set(helper.dof_joint_names), set(scene.robot.joint_names[side_slice(side)]))
            adaptor.compute_jaco_a(q[side_slice(side)])
            full_q = pin.neutral(scene.robot.pin_model)
            full_q[scene.robot.pin_q] = q
            for frame in (f"{side}_hand_flange", f"{side}_hand_C_MC", f"{side}_thumb_DP"):
                np.testing.assert_allclose(adaptor.get_frame_pose(frame), scene.robot.fk(q, side, frame), atol=1e-12)
                jac = pin.computeFrameJacobian(
                    scene.robot.pin_model,
                    scene.robot.pin_data,
                    full_q,
                    scene.robot.pin_model.getFrameId(frame),
                    pin.LOCAL_WORLD_ALIGNED,
                )
                np.testing.assert_allclose(
                    adaptor.get_frame_space_jaco(frame), jac[:, scene.robot.pin_v[side_slice(side)]], atol=1e-12
                )

    def test_native_adapter_preserves_live_object_gravity_and_passive_commands(self) -> None:
        scene = self.scene
        for side in ("left", "right"):
            before_q = scene.sim.data.qpos.copy()
            commands = scene.sim.targets.copy()
            gravity = scene.sim.model.opt.gravity.copy()
            count = []
            adapter = ActiveSimulation(scene, side, lambda: count.append(1), compensate=True)
            np.testing.assert_array_equal(scene.sim.data.qpos, before_q)
            np.testing.assert_array_equal(
                adapter.get_obj_pose(), before_q[adapter.object_address : adapter.object_address + 7]
            )
            with self.assertRaisesRegex(RuntimeError, "Reset is forbidden"):
                adapter.reset_pose_qpos(None, None)
            for bad in (np.full(28, np.nan), np.zeros(27)):
                with self.assertRaises(ValueError):
                    adapter.ctrl_qpos_a(adapter.actuator_names, bad)
            with self.assertRaisesRegex(ValueError, "order"):
                adapter.ctrl_qpos_a(adapter.actuator_names[::-1], commands[side_slice(side)])
            np.testing.assert_array_equal(scene.sim.targets, commands)
            adapter.ctrl_qpos_a_with_interp(
                commands[side_slice(side)], commands[side_slice(side)], adapter.actuator_names, 2, 5
            )
            self.assertEqual(len(count), 10)
            passive = side_slice("right" if side == "left" else "left")
            np.testing.assert_array_equal(scene.sim.targets[passive], commands[passive])
            np.testing.assert_array_equal(scene.sim.model.opt.gravity, gravity)
            self.assertFalse(scene.sim.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))

    def test_initial_measured_command_projection_keeps_state_and_regular_commands_strict(self) -> None:
        scene = self.scene
        adapter = ActiveSimulation(scene, "right", lambda: None, compensate=True)
        index = 28 + 7
        lower = scene.robot.limits[index, 0]
        scene.sim.data.qpos[scene.sim.qpos[index]] = lower - 0.0008
        measured = scene.sim.configuration()[side_slice("right")]
        before = scene.sim.data.qpos.copy()
        with self.assertRaises(ValueError):
            adapter.ctrl_qpos_a(adapter.actuator_names, measured)
        adapter.initializing_control = True
        adapter.ctrl_qpos_a(adapter.actuator_names, measured)
        self.assertFalse(adapter.initializing_control)
        self.assertAlmostEqual(adapter.target_qpos_a[7], lower)
        self.assertAlmostEqual(adapter.initial_command_projection["maximum_projection_rad"], 0.0008)
        np.testing.assert_array_equal(scene.sim.data.qpos, before)
        with self.assertRaises(ValueError):
            adapter.ctrl_qpos_a(adapter.actuator_names, measured)
        scene.sim.data.qpos[scene.sim.qpos[index]] = lower - 0.02
        adapter.initializing_control = True
        with self.assertRaisesRegex(ValueError, "soft-limit"):
            adapter.ctrl_qpos_a(adapter.actuator_names, scene.sim.configuration()[side_slice("right")])

    def test_roundoff_at_command_boundary_is_clipped_but_real_violation_is_rejected(self) -> None:
        scene = self.scene
        adapter = ActiveSimulation(scene, "left", lambda: None, compensate=True)
        command = scene.sim.targets[side_slice("left")]
        lower = scene.robot.limits[7, 0]
        before = scene.sim.data.qpos.copy()
        command[7] = lower - 2e-15
        adapter.ctrl_qpos_a(adapter.actuator_names, command)
        self.assertEqual(adapter.target_qpos_a[7], lower)
        self.assertEqual(len(adapter.command_roundoff_projections), 1)
        np.testing.assert_array_equal(scene.sim.data.qpos, before)
        command[7] = lower - 1e-9
        with self.assertRaises(ValueError):
            adapter.ctrl_qpos_a(adapter.actuator_names, command)

    def test_solver_abort_saves_partial_record_and_never_lifts(self) -> None:
        def abort(evaluator: object, chosen: dict) -> None:
            evaluator.grasp_ctrl.solver_degraded = True
            raise ControlSolveEpisodeAbort("test rejected command")

        with (
            patch(
                "ada_grasp_ctrl.experiments.dual_arm_crx_episode.prepare_overhead",
                return_value=dict(status="ready", segments=[]),
            ),
            patch("ada_grasp_ctrl.experiments.dual_arm_crx_episode.NativeEvaluator.grasp", abort),
        ):
            result = execute_native(
                self.scene,
                dict(selected=dict(selected_side="right")),
                deepcopy(self.scene.config["execution"]),
                self.root,
            )
        self.assertEqual(result["status"], "solver_degraded")
        self.assertFalse(result.get("lift_completed", False))
        self.assertEqual(result["physics_steps"], 0)
        self.assertTrue(result["passive_target_unchanged"])
        record = np.load(self.root / "control.npy", allow_pickle=True).item()
        self.assertEqual(record["episode_status"], "solver_degraded")
        self.assertTrue((self.root / "trajectory.npz").is_file())


if __name__ == "__main__":
    unittest.main()
