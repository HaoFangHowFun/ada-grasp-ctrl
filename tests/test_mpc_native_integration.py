"""Real 28-DoF model regression, native optimization parity and offline stage replay."""

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from ada_grasp_ctrl.contact import ContactObservation
from ada_grasp_ctrl.contact.compatibility import contact_frame
from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES
from ada_grasp_ctrl.mpc.contracts import JointFeedback
from ada_grasp_ctrl.mpc.native import NativeMPC
from ada_grasp_ctrl.mpc.offline import make_runner
from ada_grasp_ctrl.tasks.control_eval_func.episode_runner import EpisodeStep
from test_grasp_pipeline import plan_fixture
from test_mpc_tactile_compatibility import snapshot_from_contacts, estimator


@unittest.skipUnless(os.environ.get("ADA_RIGHT_FULL_PREVIEW"), "Requires explicit verified right-full fixture")
class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import load_preview

        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name)
        cls.report = load_preview(Path(os.environ["ADA_RIGHT_FULL_PREVIEW"]).resolve())
        cls.row = next(r for r in cls.report["samples"] if r["sample_id"] == "g0000")
        cls.record = np.load(cls.row["prepared_path"], allow_pickle=True).item()
        cls.plan = plan_fixture()
        cls.plan["keyframes"] = [
            cls.record[k].tolist() for k in ("pregrasp_qpos", "pregrasp_qpos", "grasp_qpos", "squeeze_qpos")
        ]
        cls.plan["durations_s"] = [2.0, 2.0, 2.0]

    def runner(self):
        return make_runner(self.report, self.plan, self.output)

    def test_native_absolute_command_parity_and_measured_commanded_separation(self):
        from ada_grasp_ctrl.experiments.right_full_control import Controller

        runner, reference_runner = self.runner(), self.runner()
        native = NativeMPC(runner, clock=lambda: 0.0)
        direct = Controller("ours")
        direct.reset(reference_runner)
        q = np.asarray(self.plan["keyframes"][0])
        q_cmd = q + 0.0001
        obs = ContactObservation(10.0, (), "mujoco")
        fb = JointFeedback(tuple(JOINT_NAMES), tuple(q), 10, 10, "mock")
        target = runner.qpos_path[0]
        with patch.object(runner.grasp_ctrl, "ctrl_opt", wraps=runner.grasp_ctrl.ctrl_opt) as call:
            result = native.tick(10, fb, tuple(q_cmd), tuple(target), obs, stage="grasp")
        self.assertTrue(result.valid, result.diagnostics)
        np.testing.assert_array_equal(call.call_args.kwargs["curr_q_f"], q)
        np.testing.assert_array_equal(call.call_args.kwargs["curr_q_a"], q_cmd)
        state = EpisodeStep(0, 0, target, q, q_cmd, [], np.zeros(7), np.empty((0, 3)), 0.0)
        expected = direct.step(state, obs, 0.1, timestamp=10)
        np.testing.assert_array_equal(result.positions_rad, expected.target_qpos_a)

    def test_native_failure_and_solve_deadline_return_no_candidate(self):
        runner = self.runner()
        q = tuple(self.plan["keyframes"][0])
        fb = JointFeedback(tuple(JOINT_NAMES), q, 0, 0, "mock")
        for failure in ("solver", "deadline", "invalid_contact"):
            clock_values = iter([0.0, 0.2, 0.2, 0.2])
            native = NativeMPC(runner, clock=(lambda: next(clock_values)) if failure == "deadline" else (lambda: 0.0))
            obs = ContactObservation(0, (), "mujoco", failure != "invalid_contact")
            if failure == "solver":
                with patch.object(runner.grasp_ctrl, "ctrl_opt", side_effect=ValueError("injected infeasible")):
                    result = native.tick(0, fb, q, q, obs, stage="grasp")
            else:
                result = native.tick(0, fb, q, q, obs, stage="grasp")
            self.assertFalse(result.valid, failure)
            self.assertIsNone(result.positions_rad)
            self.assertIn("reason", result.diagnostics)

    def test_native_completion_timeout_and_safety_authority(self):
        q = tuple(self.plan["keyframes"][0])
        for case in ("complete", "timeout", "stale"):
            native = NativeMPC(self.runner(), clock=lambda: 0.0)
            native.index = native.controller.max_steps()
            fb = JointFeedback(tuple(JOINT_NAMES), q, -1 if case == "stale" else 0, 0, "mock")
            with patch.object(native.controller, "should_stop", return_value=case != "timeout"):
                proposal = native.tick(
                    0, fb, q, tuple(self.plan["keyframes"][2]), ContactObservation(0, (), "synthetic"), stage="squeeze"
                )
            self.assertFalse(proposal.valid)
            self.assertIsNone(proposal.positions_rad)
            if case == "complete":
                self.assertEqual(proposal.diagnostics["termination"], "native_complete")
                self.assertIsNone(native.fault)
            elif case == "timeout":
                self.assertEqual(proposal.diagnostics["termination"], "native_timeout")
                self.assertIn("native termination not achieved", native.fault)
            else:
                self.assertNotIn("termination", proposal.diagnostics)
                self.assertIn("Stale", native.fault)

    def test_real_contact_jacobian_mujoco_finite_difference_and_Ks_equivalence(self):
        from ada_grasp_ctrl.experiments.crx_sharpa_right_simulation import ApprovedScene

        runner = self.runner()
        q = np.asarray(self.plan["keyframes"][2])
        sim = ApprovedScene(Path(self.row["scene_path"]))
        self.addCleanup(sim.close_view_and_render)
        sim.reset_pose_qpos(q, self.record["obj_pose"])
        kin = runner.robot_adaptor
        kin.compute_jaco_a(q)
        contacts = []
        for finger in ("index", "thumb"):
            body = f"right_{finger}_elastomer"
            pose = kin.get_frame_pose(body)
            point_local = np.array([0.003, -0.005, 0.007])
            rotation = contact_frame(pose[:3, 0])
            contacts.append(
                {
                    "body1_name": body,
                    "body2_name": "object",
                    "contact_pos": pose[:3, :3] @ point_local + pose[:3, 3],
                    "contact_pos_local": point_local.reshape(3, 1),
                    "contact_frame": rotation,
                    "contact_frame_local": pose[:3, :3].T @ rotation,
                    "contact_force": np.array([2.0, 0.1, -0.2, 55.0, -66.0, 77.0]),
                    "normal_force": 2.0,
                    "timestamp": 10.0,
                    "source": "mujoco",
                    "hand_side": "right",
                    "valid": True,
                }
            )
        snapshot = snapshot_from_contacts(contacts, kin)
        snapshot = replace(snapshot, configuration=tuple(q[6:]))
        fb = JointFeedback(tuple(JOINT_NAMES), tuple(q), 10, 10, "mock")
        converted = estimator(kin).estimate(snapshot, fb, 10)
        self.assertEqual(converted.status, "eligible", converted.reasons)
        expected = {c["body1_name"]: c for c in contacts}
        native_contacts = [expected[c["body1_name"]] for c in converted.observation.contacts]
        a, stack_a = runner.grasp_ctrl.Ks(q + 1e-5, q, deepcopy(native_contacts))
        b, stack_b = runner.grasp_ctrl.Ks(q + 1e-5, q, deepcopy(list(converted.observation.contacts)))
        for key in stack_a:
            np.testing.assert_allclose(stack_a[key], stack_b[key], rtol=1e-5, atol=1e-6)
        # Match contact order, centroid and native cm scaling in the full solver.
        from ada_grasp_ctrl.experiments.right_full_control import Controller

        outputs, outcomes = [], []
        for sample in (native_contacts, list(converted.observation.contacts)):
            comparison_runner = self.runner()
            controller = Controller("ours")
            controller.reset(comparison_runner)
            observation = ContactObservation(10.0, tuple(sample), sample[0]["source"])
            force = np.array([c["contact_force"][:3] for c in sample])
            state = EpisodeStep(0, 0, q, q, q + 1e-5, sample, self.record["obj_pose"], force, float(force[:, 0].sum()))
            outputs.append(controller.step(state, observation, 0.1, timestamp=10).target_qpos_a)
            outcomes.append([d["accepted"] for d in comparison_runner.grasp_ctrl.r_data["solver_diagnostics"]])
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
        self.assertEqual(outcomes[0], outcomes[1])
        for c in b:
            body, local = c["body1_name"], c["contact_pos_local"].ravel()
            numeric = np.zeros((3, 28))
            for j in range(28):
                world = []
                for sign in (1, -1):
                    shifted = q.copy()
                    shifted[j] += sign * 1e-7
                    kin.compute_fk_a(shifted)
                    pose = kin.get_frame_pose(body)
                    world.append(pose[:3, :3] @ local + pose[:3, 3])
                numeric[:, j] = c["contact_frame"].T @ (world[0] - world[1]) / 2e-7
            np.testing.assert_allclose(c["jaco_f"], numeric, rtol=1e-5, atol=1e-6)
            jp, jr = np.zeros((3, sim.model.nv)), np.zeros((3, sim.model.nv))
            mujoco.mj_jac(sim.model, sim.data, jp, jr, c["contact_pos"], sim.model.body(body).id)
            indices = [sim.model.joint(n).dofadr[0] for n in JOINT_NAMES]
            np.testing.assert_allclose(c["jaco_f"], c["contact_frame"].T @ jp[:, indices], rtol=1e-5, atol=1e-6)

    def test_ipc_worker_named_roundtrip_without_ros_import(self):
        import sys
        import time
        from dataclasses import asdict
        from ada_grasp_ctrl.mpc.ipc import MPCWorker

        worker = MPCWorker(
            sys.executable,
            {
                "schema_version": 1,
                "environment": "mock",
                "plan": self.plan,
                "model_report": self.report,
                "output": str(self.output),
            },
            self.output / "worker.log",
        )
        self.addCleanup(worker.close)
        now = time.monotonic()
        q = tuple(self.plan["keyframes"][0])
        fb = JointFeedback(tuple(JOINT_NAMES), q, now, now, "mock")
        worker.submit(now, fb, q, q, asdict(ContactObservation(now, (), "synthetic_valid_empty")), "grasp")
        with self.assertRaisesRegex(ValueError, "backlog"):
            worker.submit(now, fb, q, q, {}, "grasp")
        result = None
        while result is None:
            result = worker.poll()
            time.sleep(0.001)
        self.assertTrue(result.valid, result.diagnostics)
        self.assertEqual(result.joint_names, tuple(JOINT_NAMES))
        self.assertLess(result.solve_duration_s, 0.1)


if __name__ == "__main__":
    unittest.main()
