"""Integration coverage for experimental CRX-Sharpa contact adaptation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from ada_grasp_ctrl.crx_sharpa_adaptive import run_phase2
from ada_grasp_ctrl.crx_sharpa_demo import run_phase1
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


if __name__ == "__main__":
    unittest.main()
