"""Paired protocol, status accounting, and passive recording contracts."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from ada_grasp_ctrl.experiments.episode_recording import EpisodeRecorder, load_episode, replay_episode
from ada_grasp_ctrl.experiments.phase_comparison import build_phase_config, compare_phase_reports
from ada_grasp_ctrl.paths import source_checkout_root
from ada_grasp_ctrl.tasks.control_eval import safe_eval_one
from ada_grasp_ctrl.utils.grasp_controller import GraspController


class PhaseComparisonTest(unittest.TestCase):
    def test_status_accounting_retains_all_selected_ids(self):
        samples = [{"sample_id": str(i), "status": "prepared", "scene_path": "scene"} for i in range(5)]
        samples[-1]["status"] = "preparation_error"
        result = compare_phase_reports(
            samples,
            {
                "phase1": [
                    {"sample_id": "0", "status": "success"},
                    {"sample_id": "1", "status": "failure"},
                    {"sample_id": "2", "status": "solver_degraded"},
                ],
                "phase2": [
                    {"sample_id": "0", "status": "execution_error"},
                    {"sample_id": "1", "status": "invalid_initialization"},
                ],
            },
        )
        self.assertEqual(result["selected_count"], 5)
        self.assertEqual(result["summary"]["phase1"]["success_over_valid"], 0.5)
        self.assertEqual(result["summary"]["phase1"]["success_over_selected"], 0.2)
        self.assertEqual(result["pairs"][-1]["phase2"], "preparation_error")
        self.assertEqual(result["pairs"][3]["phase1"], "missing_result")

    def test_recorder_copies_without_mutating_state(self):
        data = SimpleNamespace(time=0.0, qpos=np.arange(3.0), qvel=np.zeros(3), ctrl=np.zeros(2), act=np.zeros(0))
        sim = SimpleNamespace(data=data, get_obj_pose=lambda: np.zeros(7))
        recorder = EpisodeRecorder(sim, {})
        recorder.capture("init")
        np.testing.assert_array_equal(data.qpos, np.arange(3.0))
        data.qpos[:] = 4
        np.testing.assert_array_equal(recorder.frames[0]["qpos"], np.arange(3.0))

    def test_phase_protocol_preserves_targets_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = {"robot": "dummy_arm_shadow", "data_root": str(root), "prepared_root": str(root)}
            configs = [build_phase_config(suite, phase, root / phase, 37) for phase in ("phase1", "phase2")]
            for config in configs:
                self.assertFalse(config.task.arm_pregrasp_is_grasp)
                self.assertEqual(config.task.graspdata.squeeze_t, 0)
                self.assertEqual(config.task.graspdata.pregrasp_t, 0)
                self.assertEqual(list(config.task.offsets), [0.0])
                self.assertEqual(config.seed, 37)
            self.assertEqual(configs[0].task.control, configs[1].task.control)

    def test_real_op_has_no_optimizer_and_recorder_is_passive(self):
        """Exercise real MuJoCo, native policy, and independently reloadable lift."""
        checkout = source_checkout_root()
        candidates = sorted((checkout / "examples/data/shadow").rglob("*.npy"))
        prepared = next(p for p in candidates if "dummy_arm" in str(p))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = {
                "robot": "dummy_arm_shadow",
                "data_root": str(checkout),
                "prepared_root": str(prepared.parent),
            }
            outputs = []
            with (
                patch.object(GraspController, "ctrl_opt", side_effect=AssertionError("op optimized")),
                patch.object(GraspController, "check_wrench_balance", side_effect=AssertionError("op balanced")),
            ):
                for enabled in (False, True):
                    config = build_phase_config(suite, "phase1", root / str(enabled))
                    config.task.record_episode = enabled
                    result = safe_eval_one((str(prepared), config, None, 0))
                    self.assertEqual(result.status.value, "completed", result.message)
                    outputs.append(Path(result.output_paths[0]))
            records = [np.load(path, allow_pickle=True).item() for path in outputs]
            for field in ("dof", "doa", "planned_dof", "obj_pose"):
                np.testing.assert_array_equal(records[0][field], records[1][field])
            replay = load_episode(outputs[1].with_suffix(".replay.npz"))
            self.assertEqual([f["segment"] for f in replay["frames"]].count("lift"), 20)
            self.assertEqual(replay["frames"][-1]["segment"], "terminal")
            model = mujoco.MjModel.from_xml_path(replay["scene_path"])
            data = mujoco.MjData(model)
            final = replay["frames"][-1]
            data.qpos[:] = final["qpos"]
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(data.qpos, final["qpos"])
            with (
                patch("ada_grasp_ctrl.utils.viewer_session.DebugViewerSession") as viewer,
                patch("ada_grasp_ctrl.experiments.episode_recording.time.sleep"),
            ):
                replay_episode(outputs[1].with_suffix(".replay.npz"), hold=False)
                replay_data = viewer.return_value.attach.call_args.args[1]
                np.testing.assert_array_equal(replay_data.qpos, final["qpos"])
                viewer.return_value.close.assert_called_once()
            # A sidecar cannot enter the native legacy *.npy control inventory.
            self.assertEqual(len(list(outputs[1].parent.glob("*.npy"))), 1)
            calls = {"control": 0, "balance": 0}
            original_control = GraspController.ctrl_opt
            original_balance = GraspController.check_wrench_balance

            def control_spy(controller, *args, **kwargs):
                calls["control"] += 1
                return original_control(controller, *args, **kwargs)

            def balance_spy(controller, *args, **kwargs):
                calls["balance"] += 1
                return original_balance(controller, *args, **kwargs)

            config = build_phase_config(suite, "phase2", root / "adaptive")
            with (
                patch.object(GraspController, "ctrl_opt", control_spy),
                patch.object(GraspController, "check_wrench_balance", balance_spy),
            ):
                result = safe_eval_one((str(prepared), config, None, 0))
            self.assertIn(result.status.value, ("completed", "solver_degraded"), result.message)
            self.assertGreater(calls["control"], 0)
            self.assertGreater(calls["balance"], 0)
