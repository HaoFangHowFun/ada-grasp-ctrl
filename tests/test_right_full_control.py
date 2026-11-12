"""Deterministic contracts and optional real-CRX paired execution/parity tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from omegaconf import OmegaConf

from ada_grasp_ctrl.contact import ContactObservation
from ada_grasp_ctrl.contact.mujoco_adapter import MuJoCoContactAdapter
from ada_grasp_ctrl.experiments.right_full_control import (
    Controller,
    linear_reference,
    physics_substeps,
    validate_command,
)
from ada_grasp_ctrl.tasks.control_eval_func.episode_runner import EpisodeStep
from ada_grasp_ctrl.utils.hand_util import MjHO


class RightFullContractTests(unittest.TestCase):
    def test_linear_endpoints_order_and_continuity(self):
        start, middle, end = np.arange(28.0), np.arange(28.0) + 2, np.arange(28.0) - 1
        for elapsed in (-1, 0, 0.5, 1, 2, 3):
            np.testing.assert_array_equal(
                linear_reference(start, middle, elapsed, 2), start + np.clip(elapsed / 2, 0, 1) * (middle - start)
            )
        np.testing.assert_array_equal(linear_reference(start, middle, 2, 2), linear_reference(middle, end, 0, 2))
        with self.assertRaises(ValueError):
            linear_reference(start, end, 0, 0)

    def test_modes_and_timing(self):
        for mode in ("mpc", "op", "", None):
            with self.assertRaises(ValueError):
                Controller(mode)
        self.assertEqual(physics_substeps(0.004), 25)
        self.assertEqual(physics_substeps(0.002), 50)
        for dt, hz in ((0.003, 10), (0.004, 20), (0.004, 100), (float("nan"), 10), (0.1, 10), (0, 10)):
            with self.assertRaises(ValueError):
                physics_substeps(dt, hz)

    def test_invalid_commands(self):
        q, limits, velocity = np.zeros(28), np.tile([[-1.0], [1.0]], (1, 28)), np.ones(28)
        validate_command(q + 0.1, q, limits, velocity)
        for bad in (np.zeros(27), q + np.nan, q + np.inf, q + 2, q + 0.101):
            with self.assertRaises(ValueError):
                validate_command(bad, q, limits, velocity)

    def test_linear_never_uses_contact_or_optimizer(self):
        target = np.arange(28.0) / 100
        runner = SimpleNamespace(
            qpos_path=np.zeros((40, 28)), robot_adaptor=SimpleNamespace(_dof2doa=lambda q: q.copy())
        )
        state = EpisodeStep(0, 0, target, np.zeros(28), np.zeros(28), [], np.zeros(7), np.empty((0, 3)), 0.0)
        controller = Controller("linear")
        controller.reset(runner)
        from ada_grasp_ctrl.utils.grasp_controller import GraspController

        with (
            patch.object(GraspController, "ctrl_opt", side_effect=AssertionError("optimization called")),
            patch.object(GraspController, "check_wrench_balance", side_effect=AssertionError("balance called")),
        ):
            # Unknown/unavailable tactile data must never alter the open-loop command.
            command = controller.step(state, ContactObservation(0.0, (), "unavailable", False), 0.1, timestamp=0.0)
        np.testing.assert_array_equal(command.target_qpos_a, target)
        self.assertFalse(controller.should_stop(state))
        self.assertEqual(controller.max_steps(), 40)
        self.assertEqual(controller.diagnostics(), {})

    def test_python_module_cli(self):
        module = [sys.executable, "-m", "ada_grasp_ctrl.experiments.right_full_pipeline"]
        result = subprocess.run([*module, "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--controller {linear,ours}", result.stdout)
        self.assertIn("--compare", result.stdout)
        invalid = subprocess.run([*module, "--controller", "mpc"], capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid choice", invalid.stderr)
        missing = subprocess.run(module, capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("required", missing.stderr)

    def test_ros_imports_blocked(self):
        code = """
import sys
class BlockROS:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('rclpy', 'rospy', 'sensor_msgs', 'geometry_msgs', 'launch', 'sharpa_sdk'):
   raise AssertionError('offline import attempted: ' + fullname)
sys.meta_path.insert(0, BlockROS())
from ada_grasp_ctrl.experiments.right_full_pipeline import main
main(['--help'])
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--controller {linear,ours}", result.stdout)


class ContactObservationTests(unittest.TestCase):
    @staticmethod
    def scene(object_first=False, multiple=False):
        hand = '<body name="hand"><geom type="sphere" size=".05"/></body>'
        if multiple:
            hand += '<body name="finger" pos=".09 .09 0"><geom type="sphere" size=".05"/></body>'
        obj = '<body name="object" pos=".09 0 0"><freejoint/><geom type="sphere" size=".05"/></body>'
        xml = '<mujoco><option gravity="0 0 0"/><worldbody>' + (obj + hand if object_first else hand + obj)
        xml += "</worldbody></mujoco>"
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qvel[:] = [0, 0.5, 0.2, 3, 0, 0]
        mujoco.mj_forward(model, data)
        adapter = MuJoCoContactAdapter(
            model, data, hand_bodies={"hand", "finger"} if multiple else {"hand"}, object_bodies={"object"}
        )
        return model, data, adapter

    def test_native_extraction_parity_and_contact_loss(self):
        for multiple in (False, True):
            model, data, adapter = self.scene(multiple=multiple)
            native = SimpleNamespace(model=model, data=data, hand_mocap=False, hand_prefix="", hand_contact_prefixes=())
            direct = MjHO.get_curr_contact_info(native)
            observation = adapter.observe()
            self.assertEqual(len(direct), len(observation.contacts))
            self.assertEqual(len(direct), 2 if multiple else 1)
            for a, b in zip(direct, observation.contacts):
                for key in a:
                    if isinstance(a[key], str):
                        self.assertEqual(a[key], b[key])
                    else:
                        np.testing.assert_array_equal(a[key], b[key])
                hand = data.body(b["body1_name"])
                np.testing.assert_allclose(
                    hand.xmat.reshape(3, 3) @ b["contact_pos_local"].ravel() + hand.xpos, b["contact_pos"], atol=1e-12
                )
            disposable = observation.controller_contacts(data.time)
            disposable[0]["contact_force"][:] = 999
            self.assertNotEqual(observation.contacts[0]["contact_force"][0], 999)
            data.qpos[:3] = [10, 10, 10]
            mujoco.mj_forward(model, data)
            self.assertEqual(adapter.observe().contacts, ())

    def test_geom_order_and_world_force(self):
        observations = []
        for object_first in (False, True):
            _, data, adapter = self.scene(object_first)
            contact = adapter.observe().contacts[0]
            self.assertGreaterEqual(contact["normal_force"], 0)
            world_force = contact["contact_frame"] @ contact["contact_force"][:3]
            np.testing.assert_allclose(world_force, data.qfrc_constraint[:3], atol=1e-10)
            observations.append(contact)
        for field in ("contact_frame", "contact_frame_local", "contact_force", "contact_pos", "contact_pos_local"):
            np.testing.assert_allclose(observations[0][field], observations[1][field], atol=1e-10)

    def test_multiple_contacts_on_one_body_and_changing_count(self):
        xml = """<mujoco><option gravity="0 0 0"/><worldbody><body name="hand">
        <geom type="sphere" size=".05" pos="0 -.03 0"/><geom type="sphere" size=".05" pos="0 .03 0"/>
        </body><body name="object" pos=".09 0 0"><freejoint/><geom type="sphere" size=".06"/>
        </body></worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        adapter = MuJoCoContactAdapter(model, data, hand_bodies={"hand"}, object_bodies={"object"})
        mujoco.mj_forward(model, data)
        self.assertEqual([c["body1_name"] for c in adapter.observe().contacts], ["hand", "hand"])
        data.qpos[1] = 0.07
        mujoco.mj_forward(model, data)
        self.assertEqual(len(adapter.observe().contacts), 1)
        data.qpos[1] = 1.0
        mujoco.mj_forward(model, data)
        self.assertEqual(len(adapter.observe().contacts), 0)

    def test_nonfinite_mujoco_reading_is_rejected(self):
        _, data, adapter = self.scene()

        def bad_force(model, data, index, output):
            output[:] = np.nan

        with patch("mujoco.mj_contactForce", side_effect=bad_force):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                adapter.observe()
        self.assertEqual(data.time, 0.0)

    def test_bad_frame_force_unavailable_and_stale(self):
        _, _, adapter = self.scene()
        good = adapter.observe()
        for field, value in (
            ("contact_frame", np.ones((3, 3))),
            ("contact_frame_local", np.diag([-1.0, 1, 1])),
            ("contact_force", np.full(6, np.nan)),
            ("contact_pos_local", None),
        ):
            contacts = deepcopy(good.contacts)
            contacts[0][field] = value
            with self.assertRaises(ValueError):
                replace(good, contacts=contacts).controller_contacts(0)
        with self.assertRaises(ValueError):
            good.controller_contacts(0.1)
        with self.assertRaises(ValueError):
            replace(good, valid=False, errors=("tactile frame unavailable",)).controller_contacts(0)

    def test_object_table_is_not_hand_contact(self):
        xml = """<mujoco><worldbody><body name="table"><geom type="plane" size="1 1 .1"/></body>
        <body name="hand" pos="2 2 2"><geom type="sphere" size=".05"/></body>
        <body name="object" pos="0 0 .04"><freejoint/><geom type="sphere" size=".05"/></body>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)
        adapter = MuJoCoContactAdapter(model, data, hand_bodies={"hand"}, object_bodies={"object"})
        self.assertEqual(adapter.observe().contacts, ())
        self.assertEqual(len(adapter.observe().environment_contacts), data.ncon)


@unittest.skipUnless(
    os.environ.get("ADA_RIGHT_FULL_PREVIEW"), "Set ADA_RIGHT_FULL_PREVIEW to a verified frozen preview"
)
class RealCrxIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ada_grasp_ctrl.experiments.right_full_pipeline import run
        from ada_grasp_ctrl.paths import project_root

        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name) / "comparison"
        cls.preview = Path(os.environ["ADA_RIGHT_FULL_PREVIEW"])
        config = OmegaConf.load(project_root() / "src/ada_grasp_ctrl/config/right_full.yaml")
        cls.summary = run(cls.preview, cls.output, ["g0000"], config, compare=True)

    def test_complete_identical_scene_comparison(self):
        self.assertTrue(self.summary["identical_control_initial_state"])
        for result in self.summary["results"]:
            self.assertTrue(result["lift_completed"])
            self.assertTrue(result["lift_success"])
            self.assertEqual(result["status"], "success")
            self.assertGreater(result["contact_steps"], 0)
            mode = result["controller_mode"]
            ticks = json.loads((self.output / "g0000" / mode / "diagnostics.json").read_text())
            np.testing.assert_allclose(np.diff([t["timestamp"] for t in ticks]), 0.1, atol=1e-12, rtol=0)
            self.assertTrue(all(abs(t["state_timestamp"] - t["timestamp"] - 0.1) < 1e-12 for t in ticks))
            if mode == "linear":
                self.assertTrue(all(t["solver_time_ms"] is None and t["wrench_balance"] is None for t in ticks))
            else:
                self.assertEqual(sum(t["solver_time_ms"] is not None for t in ticks), result["control_steps"])

    def test_repeat_modes_deterministic(self):
        from ada_grasp_ctrl.experiments.right_full_pipeline import run
        from ada_grasp_ctrl.paths import project_root
        from ada_grasp_ctrl.experiments.episode_recording import load_episode

        config = OmegaConf.load(project_root() / "src/ada_grasp_ctrl/config/right_full.yaml")
        repeat = run(self.preview, Path(self.temporary.name) / "repeat", ["g0000"], config, compare=True)
        for first, second in zip(self.summary["results"], repeat["results"]):
            self.assertEqual(first["status"], second["status"])
            a, b = load_episode(Path(first["replay_path"])), load_episode(Path(second["replay_path"]))
            self.assertEqual(len(a["frames"]), len(b["frames"]))
            for x, y in zip(a["frames"], b["frames"]):
                for key in ("qpos", "qvel", "ctrl", "act", "object_pose", "time"):
                    np.testing.assert_array_equal(x[key], y[key])

    def test_native_mpc_trajectory_and_stage_parity(self):
        from ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation import run_episode
        from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import load_preview

        report = load_preview(self.preview)
        row = deepcopy(next(r for r in report["samples"] if r["sample_id"] == "g0000"))
        row["prepared_path"] = str(self.output / "selected_grasp.npy")
        preflight = json.loads((self.output / "preflight.json").read_text())
        direct = run_episode(row, report, Path(self.temporary.name) / "native", "ours", preflight)
        adapted = next(r for r in self.summary["results"] if r["controller_mode"] == "ours")
        self.assertEqual(direct["status"], adapted["status"])
        a = np.load(direct["control_path"], allow_pickle=True).item()
        b = np.load(adapted["control_path"], allow_pickle=True).item()
        for key in ("stage", "dof", "doa", "obj_pose", "planned_dof", "balance_metric", "desired_sum_force"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-6, err_msg=key)
        for x, y in zip(a["opt_res"], b["opt_res"]):
            for key in ("q_a", "dq_a", "cf"):
                np.testing.assert_allclose(x[key], y[key], rtol=1e-5, atol=1e-6)
        self.assertEqual([len(c) for c in a["contacts"]], [len(c) for c in b["contacts"]])

    def prepared_evaluator(self, mode="ours"):
        from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import load_preview
        from ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation import make_config
        from ada_grasp_ctrl.experiments.right_full_episode import RightFullEvaluator

        report = load_preview(self.preview)
        row = deepcopy(next(r for r in report["samples"] if r["sample_id"] == "g0000"))
        row["prepared_path"] = str(self.output / "selected_grasp.npy")
        config = make_config(report, row, self.output, "ours")
        config = OmegaConf.merge(
            OmegaConf.to_container(config, resolve=True), {"controller": {"mode": mode, "ours": {"frequency_hz": 10}}}
        )
        evaluator = RightFullEvaluator(row, report, config, mode)
        self.addCleanup(evaluator.mj_ho.close_view_and_render)
        evaluator.mj_ho.reset_pose_qpos(evaluator.grasp_data["pregrasp_qpos"], evaluator.grasp_data["obj_pose"])
        return evaluator

    def test_invalid_optimized_command_never_steps_physics(self):
        from ada_grasp_ctrl.utils.grasp_controller import GraspController

        for value in (float("nan"), 100.0):
            evaluator = self.prepared_evaluator()
            with (
                patch.object(
                    GraspController,
                    "ctrl_opt",
                    return_value={"q_a": np.full(28, value), "dq_a": np.zeros(28), "cf": np.zeros(0)},
                ),
                patch.object(evaluator.mj_ho, "ctrl_qpos_a_with_interp") as apply,
            ):
                with self.assertRaises(ValueError):
                    evaluator.native_grasp()
                apply.assert_not_called()
                self.assertEqual(evaluator.mj_ho.data.time, 0.0)
                self.assertFalse(evaluator.ticks[-1]["command_applied"])
                self.assertEqual(evaluator.ticks[-1]["solver_status"], "failed")

    def test_solver_abort_does_not_fall_back_to_linear(self):
        from ada_grasp_ctrl.errors import ControlSolveEpisodeAbort
        from ada_grasp_ctrl.utils.grasp_controller import GraspController

        evaluator = self.prepared_evaluator()
        with patch.object(GraspController, "ctrl_opt", side_effect=ControlSolveEpisodeAbort("injected rejection")):
            with self.assertRaises(ControlSolveEpisodeAbort):
                evaluator.native_grasp()
        self.assertEqual(evaluator.mj_ho.data.time, 0.0)
        self.assertFalse(evaluator.ticks[-1]["command_applied"])

    def test_object_motion_before_pregrasp_rejected(self):
        from ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation import ApproachInvalidError

        evaluator = self.prepared_evaluator()
        evaluator.mj_ho.data.qpos[evaluator.mj_ho.object_address] += 0.01
        mujoco.mj_forward(evaluator.mj_ho.model, evaluator.mj_ho.data)
        with self.assertRaisesRegex(ApproachInvalidError, "Object moved"):
            evaluator.native_grasp()
        self.assertEqual(evaluator.mj_ho.data.time, 0.0)

    def test_no_contact_timeout_is_explicit(self):
        evaluator = self.prepared_evaluator()
        evaluator.grasp_data["obj_pose"][:3] += [5, 5, 5]
        evaluator.mj_ho.reset_pose_qpos(evaluator.grasp_data["pregrasp_qpos"], evaluator.grasp_data["obj_pose"])
        evaluator.native_grasp()
        self.assertEqual(evaluator.completion["termination"], "timeout")
        self.assertEqual(len(evaluator.ticks), 80)
        self.assertTrue(all(t["contact_count"] == 0 for t in evaluator.ticks))
        self.assertNotIn("grasp_success", evaluator.completion)

    def test_real_contact_point_jacobian(self):
        from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import load_preview
        from ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation import RightEvaluator, make_config
        from mr_utils.utils_calc import skew

        report = load_preview(self.preview)
        row = next(r for r in report["samples"] if r["sample_id"] == "g0000")
        config = make_config(report, row, self.output, "ours")
        evaluator = RightEvaluator(row, report, config, "ours")
        self.addCleanup(evaluator.mj_ho.close_view_and_render)
        ticks = json.loads((self.output / "g0000/ours/diagnostics.json").read_text())
        tick = next(t for t in ticks if t["contacts"] and t["mpc_stage"] is not None)
        # Contact metadata is pre-action, so restore the matching native pre-action state.
        record = np.load(self.output / "g0000/ours/control.npy", allow_pickle=True).item()
        index = next(i for i, c in enumerate(record["contacts"]) if c)
        q, pose = record["dof"][index], record["obj_pose"][index]
        sim = evaluator.mj_ho
        sim.reset_pose_qpos(q, pose)
        evaluator.robot_adaptor.compute_jaco_a(q)
        adapter = MuJoCoContactAdapter(sim.model, sim.data, hand_bodies=sim.hand_bodies, object_bodies={"object"})
        self.assertTrue(adapter.observe().contacts)
        for contact in adapter.observe().contacts:
            body = contact["body1_name"]
            jac = evaluator.robot_adaptor.get_frame_jaco(body, type="body")
            local = np.asarray(contact["contact_pos_local"]).ravel()
            expected = contact["contact_frame_local"].T @ np.block([np.eye(3), -skew(local)]) @ jac
            jp, jr = np.zeros((3, sim.model.nv)), np.zeros((3, sim.model.nv))
            mujoco.mj_jac(sim.model, sim.data, jp, jr, contact["contact_pos"], sim.model.body(body).id)
            dofs = [sim.model.joint(n).dofadr[0] for n in evaluator.robot.dof_names]
            np.testing.assert_allclose(expected, contact["contact_frame"].T @ jp[:, dofs], atol=1e-6, rtol=1e-5)
        self.assertTrue(tick["contacts"])


if __name__ == "__main__":
    unittest.main()
