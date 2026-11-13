"""P3 compatibility policies and read-only selected-reference integration."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

import test_mpc_tactile_compatibility as fixtures
from test_mpc_tactile_compatibility import estimator
from ada_grasp_ctrl.mpc.streaming import GuardedStream


class P3CompatibilityTests(unittest.TestCase):
    def test_measured_cone_is_diagnostic_only_when_opted_in(self):
        snapshot, fb, kin = fixtures.MixedContactTests().fixture()
        idx = next(i for i, (_, row) in enumerate(snapshot.fingers) if row.observation is not None)
        finger, row = snapshot.fingers[idx]
        force = (-2.0, -3.0, 0.0)
        raw = force + row.frame.raw[3:]
        changed = replace(
            row, frame=replace(row.frame, raw=raw), observation=replace(row.observation, force=force, raw_f6=raw)
        )
        snapshot = replace(
            snapshot, fingers=tuple((f, changed if i == idx else r) for i, (f, r) in enumerate(snapshot.fingers))
        )
        self.assertIsNone(estimator(kin).estimate(snapshot, fb, 10).observation)
        result = estimator(kin, measured_force_policy="diagnostic").estimate(snapshot, fb, 10)
        self.assertEqual(result.status, "eligible", result.reasons)
        contact = next(c for c in result.observation.contacts if c["body1_name"] == f"right_{finger}_elastomer")
        self.assertFalse(contact["provenance"]["measured_cone_pass"])
        self.assertLess(contact["provenance"]["friction_slack_n"], 0)
        np.testing.assert_allclose(contact["contact_force"][:3], [2, 3, 0])
        bad = replace(changed, observation=replace(changed.observation, force=(2.0, 3.0, 0.0)))
        result = estimator(kin, measured_force_policy="diagnostic").estimate(
            replace(snapshot, fingers=tuple((f, bad if i == idx else r) for i, (f, r) in enumerate(snapshot.fingers))),
            fb,
            10,
        )
        self.assertIsNone(result.observation)
        self.assertIn("negative normal", result.reasons[0])

    def test_ambiguous_resultant_is_omitted_without_losing_other_contact(self):
        snapshot, fb, kin = fixtures.MixedContactTests().fixture()
        rows = list(snapshot.fingers)
        indices = [i for i, (_, row) in enumerate(rows) if row.observation is not None]
        for count, index in enumerate(indices):
            finger, row = rows[index]
            rows[index] = (
                finger,
                replace(
                    row,
                    frame=replace(row.frame, contact_count=2),
                    observation=replace(row.observation, contact_count=2),
                ),
            )
            result = estimator(kin, ambiguous_contact_policy="omit").estimate(
                replace(snapshot, fingers=tuple(rows)), fb, 10
            )
            self.assertEqual(result.provenance["finger_statuses"][finger]["interpretation"], "AMBIGUOUS")
            if count == 0:
                self.assertEqual(len(result.observation.contacts), 1)
                self.assertEqual(result.observation.contacts[0]["body1_name"], "right_thumb_elastomer")
            else:
                self.assertIsNone(result.observation)
                self.assertIn("All sensed contacts are ambiguous", result.reasons[0])

    def test_hardware_stream_requires_opt_in(self):
        with self.assertRaisesRegex(ValueError, "explicit enable"):
            GuardedStream(config={"environment": "hardware"})
        stream = GuardedStream(config={"environment": "hardware"}, hardware_enabled=True)
        self.assertFalse(stream.running)


@unittest.skipUnless(os.environ.get("ADA_P3_PACKET"), "Requires selected real packet fixture")
class SelectedReferenceTests(unittest.TestCase):
    def test_real_selected_reference_and_receipt_tactile_ipc(self):
        import sys
        import time
        from dataclasses import asdict
        from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
        from ada_grasp_ctrl.mpc.observation import selected_runner
        from ada_grasp_ctrl.mpc.ipc import MPCWorker
        from ada_grasp_ctrl.mpc.contracts import JointFeedback
        from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, HAND_NAMES

        packet = load_packet(Path(os.environ["ADA_P3_PACKET"]))
        report = json.loads((Path(os.environ["ADA_RIGHT_FULL_PREVIEW"]) / "preview_report.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = selected_runner(report, packet, root)
            plan = packet["reviewed_plan"]
            np.testing.assert_array_equal(runner.qpos_path[-1], plan["keyframes"][plan["phases"].index("squeeze") + 1])
            self.assertEqual(runner.reference_provenance["candidate_id"], plan["candidate_id"])
            worker = MPCWorker(
                sys.executable,
                dict(
                    schema_version=1,
                    environment="hardware",
                    packet=packet,
                    model_report=report,
                    output=temp,
                    package_share="/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave",
                ),
                root / "worker.log",
            )
            try:
                now = time.monotonic()
                q = tuple(plan["keyframes"][plan["phases"].index("grasp")])
                feedback = JointFeedback(tuple(JOINT_NAMES), q, now, now, "synthetic_receipt")
                snapshot, _, _ = fixtures.MixedContactTests().fixture(contacts=False)
                frames = {
                    f: asdict(replace(row.frame, received=now, receipt_ros=now, subscriber_ros=now))
                    for f, row in snapshot.fingers
                }
                worker.submit_tactile(
                    now,
                    feedback,
                    q,
                    dict(
                        frames=frames,
                        joints=dict(
                            names=list(HAND_NAMES),
                            positions=list(q[6:]),
                            received=now,
                            receipt_ros=now,
                            subscriber_ros=now,
                        ),
                    ),
                    "grasp",
                )
                proposal = None
                while proposal is None:
                    proposal = worker.poll()
                    time.sleep(0.001)
                self.assertTrue(proposal.valid, proposal.diagnostics)
                self.assertEqual(proposal.contact_status, "valid_empty")
                self.assertEqual(proposal.diagnostics["reference_provenance"]["candidate_id"], plan["candidate_id"])
                self.assertFalse(proposal.diagnostics["contact_provenance"]["acquisition_synchronized"])
            finally:
                worker.close()


if __name__ == "__main__":
    unittest.main()
