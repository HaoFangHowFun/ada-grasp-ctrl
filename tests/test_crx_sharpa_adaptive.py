"""Integration coverage for experimental CRX-Sharpa contact adaptation."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from ada_grasp_ctrl.crx_sharpa_adaptive import replay_phase2, run_phase2, run_phase2_example
from ada_grasp_ctrl.crx_sharpa_contract import JOINTS
from ada_grasp_ctrl.crx_sharpa_demo import create_simulation, run_phase1
from ada_grasp_ctrl.schema import load_npy_record, validate_control_record


class CrxSharpaAdaptiveTest(unittest.TestCase):
    def test_ours_optimizes_commands_while_fingers_touch_the_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_phase1(run_dir)
            summary = run_phase2(run_dir)
            record = validate_control_record(load_npy_record(summary["control_output"]), summary["control_output"])

            self.assertEqual(summary["statuses"], ["completed"])
            self.assertGreater(summary["contact_steps"], 0)
            self.assertGreater(summary["optimized_contact_steps"], 0)
            self.assertIn(1, summary["stage_values"])
            self.assertIn(2, summary["stage_values"])
            self.assertFalse(summary["solver_degraded"])
            self.assertEqual(len(record["dof"]), summary["control_steps"])
            self.assertGreater(len(record["obj_pose"]), summary["control_steps"])
            self.assertTrue(
                all(
                    contact["body1_name"].startswith(
                        ("left_thumb_", "left_index_", "left_middle_", "left_ring_", "left_pinky_")
                    )
                    for step in record["contacts"]
                    for contact in step
                )
            )
            self.assertTrue(np.isfinite(summary["object_lift_delta_m"]))

    def test_replay_uses_saved_states_without_rerunning_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_phase1(run_dir)
            phase2 = run_phase2(run_dir)
            record = load_npy_record(phase2["control_output"])
            session = mock.MagicMock()
            session.native_is_running.side_effect = [True] * (phase2["control_steps"] + 2) + [False]
            scenes = []

            def create_scene(control_model, grasp, *, viewer):
                scene = create_simulation(control_model, grasp, viewer=viewer)
                scenes.append(scene)
                return scene

            with mock.patch("ada_grasp_ctrl.utils.hand_util.DebugViewerSession", return_value=session):
                with mock.patch("ada_grasp_ctrl.crx_sharpa_adaptive.create_simulation", side_effect=create_scene):
                    with mock.patch("ada_grasp_ctrl.crx_sharpa_adaptive.time.sleep"):
                        replay = replay_phase2(run_dir)

            self.assertEqual(replay["frames_replayed"], phase2["control_steps"])
            self.assertFalse(replay["lift_replayed"])
            np.testing.assert_allclose(scenes[0].get_qpos_f(JOINTS), record["dof"][-1])
            np.testing.assert_allclose(scenes[0].get_obj_pose(), record["obj_pose"][phase2["control_steps"] - 1])
            session.set_native_geom_group.assert_called_once_with(0, False)
            session.close.assert_called_once()

    def test_replay_rejects_invalid_playback_rate_before_opening_viewer(self) -> None:
        with self.assertRaisesRegex(ValueError, "fps"):
            replay_phase2(Path("missing"), fps=0)

    def test_live_viewer_starts_before_phase2_record_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "adaptive"
            control_path = run_dir / "control" / "ours_default" / "grasp_dist_0_pos_0.npy"
            session = mock.MagicMock()
            session.native_is_running.side_effect = [True, False]
            output_exists_at_sync = []
            session.sync.side_effect = lambda: output_exists_at_sync.append(control_path.exists())

            with mock.patch("ada_grasp_ctrl.utils.hand_util.DebugViewerSession", return_value=session):
                with mock.patch("ada_grasp_ctrl.crx_sharpa_adaptive.time.sleep"):
                    summary = run_phase2_example(run_dir, viewer=True)

            self.assertEqual(summary["statuses"], ["completed"])
            self.assertFalse(output_exists_at_sync[0])
            self.assertTrue(control_path.exists())
            session.attach.assert_called_once()
            session.set_native_geom_group.assert_called_once_with(0, False)
            session.detach.assert_called_once()
            session.close.assert_called_once()

    def test_three_times_bottle_scale_runs_both_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            phase1 = run_phase1(run_dir, object_scale=0.18)
            phase2 = run_phase2(run_dir)

            self.assertEqual(phase1["object_scale"], 0.18)
            self.assertEqual(phase2["object_scale"], 0.18)
            self.assertEqual(phase2["statuses"], ["completed"])
            self.assertGreater(phase2["contact_steps"], 0)
            self.assertGreater(phase2["optimized_contact_steps"], 0)

    def test_phase2_can_prepare_the_same_inputs_without_running_phase1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase1_dir = root / "fixed"
            phase2_dir = root / "adaptive"
            run_phase1(phase1_dir, object_scale=0.18)
            phase2 = run_phase2_example(phase2_dir, object_scale=0.18)
            fixed_grasp = load_npy_record(phase1_dir / "grasp.npy")
            adaptive_grasp = load_npy_record(phase2_dir / "grasp.npy")

            self.assertEqual(phase2["statuses"], ["completed"])
            self.assertEqual(phase2["run_dir"], str(phase2_dir.resolve()))
            self.assertEqual(fixed_grasp["joint_names"], adaptive_grasp["joint_names"])
            self.assertEqual(fixed_grasp["obj_scale"], adaptive_grasp["obj_scale"])
            for field in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos", "obj_pose"):
                np.testing.assert_array_equal(fixed_grasp[field], adaptive_grasp[field])

    def test_invalid_initialization_has_no_lift_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_phase1(run_dir, object_scale=0.18)
            grasp_path = run_dir / "grasp.npy"
            grasp = load_npy_record(grasp_path)
            grasp["obj_pose"][:2] -= [0.025, 0.225]
            np.save(grasp_path, grasp)

            summary = run_phase2(run_dir)

            self.assertEqual(summary["statuses"], ["invalid_initialization"])
            self.assertEqual(summary["control_steps"], 0)
            self.assertIsNone(summary["object_lift_delta_m"])


if __name__ == "__main__":
    unittest.main()
