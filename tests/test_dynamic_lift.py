"""Offline dynamic Lift contracts and optional recorded real-state geometry regression."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES
from ada_grasp_ctrl.pipeline import dynamic_lift as lift


def snapshot(q=None):
    return dict(
        joint_names=list(JOINT_NAMES),
        positions=q or [0.0] * 28,
        captured_at=100.0,
        received_at=dict(arm=99.99, hand=99.98),
        header_stamps=dict(arm=1000.0, hand=1000.0),
        source="same_host_receipt_aligned",
        topics=dict(arm="/crx5ia/right/joint_states", hand="/sharpa/right_hand/joint_states"),
    )


class SnapshotTests(unittest.TestCase):
    def test_scope_and_provenance(self):
        for mode in ("linear", "mpc_passive", "mpc_active"):
            for environment in ("mock", "hardware"):
                for complete in (False, True):
                    self.assertEqual(
                        lift.eligible(mode, dict(environment=environment), complete),
                        mode == "mpc_active" and environment == "hardware" and complete,
                    )
        self.assertEqual(lift.validate_snapshot(snapshot(), 0.5, after=99), [0.0] * 28)
        for mutation in (
            lambda s: s["joint_names"].reverse(),
            lambda s: s["positions"].__setitem__(0, float("nan")),
            lambda s: s["received_at"].update(arm=99),
            lambda s: s["received_at"].update(hand=99.7),
            lambda s: s["received_at"].update(arm=100.1),
            lambda s: s["header_stamps"].update(hand=0),
            lambda s: s.update(source="commanded_target"),
            lambda s: s.update(topics={}),
        ):
            s = snapshot()
            mutation(s)
            with self.assertRaises(ValueError):
                lift.validate_snapshot(s, 0.5)
        with self.assertRaises(ValueError):
            lift.validate_snapshot(snapshot(), 0.5, after=100)

    def test_unchanged_start_tolerance_logs_delta(self):
        event = Mock()
        q = [0.0] * 28
        q[0] = 0.009
        lift.check_handoff_start(q, [0.0] * 28, event)
        q[0] = 0.011
        with self.assertRaisesRegex(ValueError, "right_J1"):
            lift.check_handoff_start(q, [0.0] * 28, event)
        self.assertEqual(event.call_args.kwargs["delta_rad"][0], 0.011)
        q[0] = 0
        q[6] = 0.021
        with self.assertRaises(ValueError):
            lift.check_handoff_start(q, [0.0] * 28, event)

    def test_metadata(self):
        self.assertIsNone(lift.decode(""))
        self.assertEqual(lift.decode(lift.PREFIX + '{"schema_version":1}'), dict(schema_version=1))
        for value in ("world", lift.PREFIX + "[]", lift.PREFIX + "x" * 200001):
            with self.assertRaises(ValueError):
                lift.decode(value)


@unittest.skipUnless(os.environ.get("ADA_DYNAMIC_LIFT_PACKET"), "Requires recorded selection packet")
class RecordedLiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ada_grasp_ctrl.experiments.right_ros_sender import load_packet

        cls.path = Path(os.environ["ADA_DYNAMIC_LIFT_PACKET"])
        cls.packet = load_packet(cls.path)
        cls.snapshot = json.loads(Path(os.environ["ADA_DYNAMIC_LIFT_SNAPSHOT"]).read_text())
        cls.handoff = lift.build_handoff(cls.packet, cls.snapshot)

    def test_recorded_start_fk_ik_and_hand(self):
        import numpy as np
        from ada_grasp_ctrl.pipeline.plan import check_start

        h = self.handoff
        index = self.packet["reviewed_plan"]["phases"].index("lift")
        with self.assertRaisesRegex(ValueError, "right_J1"):
            check_start(self.snapshot["positions"], self.packet["reviewed_plan"]["keyframes"][index])
        check_start(self.snapshot["positions"], h["segments"]["lift"]["start"])
        self.assertEqual(h["segments"]["lift"]["positions"][6:], self.snapshot["positions"][6:])
        start, target, actual = (
            np.array(h["lift"][key]) for key in ("start_flange_wxyz", "target_flange_wxyz", "actual_flange_wxyz")
        )
        np.testing.assert_allclose(target - start, [0, 0, 0.15, 0, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(actual, target, atol=1e-6)
        self.assertTrue(h["validation"]["sampled_known_geometry"])
        self.assertGreater(h["validation"]["path_samples"], 10)
        self.assertFalse(h["validation"]["continuous_path_verified"])
        self.assertEqual(h["segments"]["hold"]["positions"], h["segments"]["lift"]["positions"])
        self.assertEqual(h["segments"]["release"]["positions"][6:], [0.0] * 22)
        self.assertEqual(h["segments"]["return"]["positions"], self.packet["reviewed_plan"]["keyframes"][-1])

    def test_review_binding_and_no_mutation(self):
        before = deepcopy(self.packet)
        for phase, segment in self.handoff["segments"].items():
            index = self.packet["reviewed_plan"]["phases"].index(phase)
            lift.check_goal(
                self.packet,
                self.handoff,
                index,
                segment["positions"],
                segment["duration_s"],
                self.packet["reviewed_plan"]["receiver_settings"],
            )
            with self.assertRaises(ValueError):
                lift.check_goal(
                    self.packet,
                    self.handoff,
                    index,
                    [0.0] * 28,
                    segment["duration_s"],
                    self.packet["reviewed_plan"]["receiver_settings"],
                )
        for field in ("review_id", "packet_sha256"):
            bad = deepcopy(self.handoff)
            bad[field] = "changed"
            with self.assertRaises(ValueError):
                lift.check_goal(
                    self.packet,
                    bad,
                    index,
                    segment["positions"],
                    segment["duration_s"],
                    self.packet["reviewed_plan"]["receiver_settings"],
                )
        self.assertEqual(before, self.packet)

    def test_geometry_failures_never_return_goal(self):
        from ada_grasp_ctrl.grasp_selection.adapters.crx_sharpa import CrxSharpaAdapter

        cases = (
            patch("ada_grasp_ctrl.pipeline.lift.lift_target", side_effect=ValueError("unreachable IK")),
            patch.object(CrxSharpaAdapter, "static_check", return_value=dict(known_geometry_passed=False)),
            patch("ada_grasp_ctrl.experiments.table_clearance.HandGeometry.measure", return_value=dict(gap_m=-0.01)),
            patch.object(CrxSharpaAdapter, "__init__", side_effect=ValueError("missing model/frame")),
        )
        for case in cases:
            with case, self.assertRaises(ValueError):
                lift.build_handoff(self.packet, self.snapshot)
        bad = deepcopy(self.snapshot)
        bad["positions"][0] = 100
        with self.assertRaisesRegex(ValueError, "limits"):
            lift.build_handoff(self.packet, bad)

    def test_native_subprocess_reproduces_review(self):
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            actual = lift.compute(
                sys.executable, self.path, self.snapshot, Path(tmp), lambda: time.sleep(0.01), expected=self.handoff
            )
        self.assertEqual(actual, self.handoff)


if __name__ == "__main__":
    unittest.main()
