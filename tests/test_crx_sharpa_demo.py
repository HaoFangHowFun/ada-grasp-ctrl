"""End-to-end checks for the experimental CRX-Sharpa fixed-target example."""

from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
