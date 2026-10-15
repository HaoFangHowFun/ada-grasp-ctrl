"""End-to-end checks for the experimental CRX-Sharpa fixed-target example."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from ada_grasp_ctrl.crx_sharpa_demo import JOINTS, build_grasp_record, run_phase1
from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record


class CrxSharpaDemoTest(unittest.TestCase):
    def test_phase1_generates_a_controllable_28_joint_grasp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            summary = run_phase1(run_dir)
            record = validate_grasp_record(
                load_npy_record(run_dir / "grasp.npy"), run_dir / "grasp.npy",
                expected_joint_dim=28, expected_joint_names=JOINTS, require_joint_names=True,
            )
            self.assertEqual(summary["actuator_count"], 28)
            self.assertEqual(record["joint_names"], list(JOINTS))
            self.assertEqual(record["pregrasp_qpos"].shape, (28,))
            self.assertTrue(any(count > 0 for count in summary["peak_contact_counts"]))
            self.assertTrue(np.isfinite(summary["final_joint_error_norm"]))

    def test_grasp_record_has_distinct_reach_and_close_targets(self) -> None:
        record = build_grasp_record()
        self.assertFalse(np.allclose(record["pregrasp_qpos"][:6], record["grasp_qpos"][:6]))
        self.assertTrue(np.allclose(record["grasp_qpos"][:6], record["squeeze_qpos"][:6]))
        self.assertTrue(np.any(record["squeeze_qpos"][6:] > record["grasp_qpos"][6:]))

    def test_viewer_stays_open_after_replay_with_scene_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            session = mock.MagicMock()
            window_states = iter([True, True, False])

            def window_is_running() -> bool:
                self.assertTrue((run_dir / "phase1_summary.json").is_file())
                return next(window_states)

            session.native_is_running.side_effect = window_is_running
            with mock.patch("ada_grasp_ctrl.utils.hand_util.DebugViewerSession", return_value=session):
                with mock.patch("ada_grasp_ctrl.crx_sharpa_demo.time.sleep"):
                    run_phase1(run_dir, viewer=True)

            record = load_npy_record(run_dir / "grasp.npy")
            camera = session.set_native_camera.call_args.kwargs
            np.testing.assert_allclose(camera["lookat"], [0.5 * record["obj_pose"][0], 0.5 * record["obj_pose"][1], 0.3])
            self.assertEqual(camera["distance"], 1.5)
            self.assertEqual(camera["azimuth"], 125)
            self.assertEqual(camera["elevation"], -22)
            self.assertEqual(session.native_is_running.call_count, 3)
            session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
