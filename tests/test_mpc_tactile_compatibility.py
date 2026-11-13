"""MuJoCo-to-tactile physical equivalence and strict provenance/time rejection."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.contact.compatibility import TactileContactEstimator, contact_frame
from ada_grasp_ctrl.contact.mujoco_adapter import MuJoCoContactAdapter
from ada_grasp_ctrl.contact.replay import TactileReplay, encode_snapshot
from ada_grasp_ctrl.contact.tactile_observation import GeometryProvenance, TactileContactObservation
from ada_grasp_ctrl.experiments.right_ros_contract import HAND_NAMES, JOINT_NAMES
from ada_grasp_ctrl.mpc.contracts import JointFeedback
from ada_grasp_ctrl.tactile_preview.adapter import FingerObservation, TactileSnapshot
from ada_grasp_ctrl.tactile_preview.data import FINGERS, Frame
from ada_grasp_ctrl.utils.grasp_controller import GraspController


class KnownKinematics:
    """Known nonidentity world/base and body transforms; no duplicated sensor FK."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.base = np.eye(4)
        self.base[:3, :3] = Rotation.from_rotvec([0.3, -0.5, 0.7]).as_matrix()
        self.base[:3, 3] = [0.4, -0.2, 0.3]

    def compute_jaco_a(self, q):
        self.q = q.copy()

    def get_frame_pose(self, frame_name):
        if frame_name == "right_hand_flange":
            return self.base.copy()
        body = self.data.body(frame_name)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = body.xmat.reshape(3, 3), body.xpos
        return pose

    def get_frame_jaco(self, frame_name, type="body"):
        return np.zeros((6, 28))


def equivalent_fixture(object_first=False, now=10.0, sequence=1):
    normals = [np.array([1.0, 0, 0]), np.array([0.2, 0.7, 0.3]), np.array([0.0, 0.0, 1.0])]
    hand = ""
    for finger, normal in zip(FINGERS[:3], normals):
        normal /= np.linalg.norm(normal)
        pos = " ".join(str(v) for v in -0.09 * normal)
        hand += f'<body name="right_{finger}_elastomer" pos="{pos}"><geom type="sphere" size=".05"/></body>'
    obj = '<body name="object"><freejoint/><geom type="sphere" size=".05"/></body>'
    xml = '<mujoco><option gravity="0 0 0"/><default><geom friction=".8 .02 .001" condim="3"/></default><worldbody>'
    xml += (obj + hand if object_first else hand + obj) + "</worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[:] = [0.03, -0.02, 0.01, 0.2, -0.1, 0.3]
    data.time = now
    mujoco.mj_forward(model, data)
    adapter = MuJoCoContactAdapter(
        model, data, hand_bodies={f"right_{f}_elastomer" for f in FINGERS[:3]}, object_bodies={"object"}
    )
    native = adapter.observe()
    kin = KnownKinematics(model, data)
    snapshot = snapshot_from_contacts(native.contacts, kin, now, sequence)
    fb = JointFeedback(tuple(JOINT_NAMES), (0.0,) * 28, now, now, "synthetic")
    return native, snapshot, fb, kin


def snapshot_from_contacts(contacts, kin, now=10.0, sequence=1):
    """Create equivalent sensed vectors; clearly label the generated fixture."""
    geometry = GeometryProvenance(
        "synthetic_candidate",
        "generated_fixture",
        "native",
        0.001,
        tuple(map(tuple, np.eye(4))),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "UNVERIFIED CANDIDATE",
        (("origin", "synthetic"),),
    )
    rows = []
    base = kin.get_frame_pose("right_hand_flange")
    for finger in FINGERS:
        contact = next((c for c in contacts if c["body1_name"] == f"right_{finger}_elastomer"), None)
        frame = Frame(
            finger,
            "VALID",
            (0.0,) * 6,
            (6,),
            "",
            now,
            now + 100,
            now + 100,
            "synthetic-fixture",
            sequence,
            sequence,
            str(sequence),
            None,
            "unknown",
            "unknown",
            "unknown",
            "synthetic",
            (),
            "EMPTY",
            (0, 3),
            (),
            0,
        )
        if contact is None:
            rows.append((finger, FingerObservation("EMPTY", frame=frame)))
            continue
        force_base = -(base[:3, :3].T @ contact["contact_frame"] @ contact["contact_force"][:3])
        body = kin.get_frame_pose(contact["body1_name"])
        # Raw F6 is body-frame reaction force; trailing channels intentionally are
        # large, nonzero, uncalibrated values, never interpreted as local torque.
        raw_force = body[:3, :3].T @ base[:3, :3] @ force_base
        frame = replace(
            frame,
            raw=tuple(raw_force) + (123.0, -456.0, 789.0),
            contact_status="VALID",
            contact_shape=(1, 3),
            contact_row=(1.0, 2.0, 3.0),
            contact_count=1,
        )
        obs = TactileContactObservation(
            finger,
            tuple(base[:3, :3].T @ (contact["contact_pos"] - base[:3, 3])),
            tuple(base[:3, :3].T @ contact["contact_frame"][:, 0]),
            tuple(force_base),
            "right_hand_flange",
            now,
            now + 100,
            now + 100,
            frame.session_id,
            sequence,
            sequence,
            str(sequence),
            None,
            "unknown",
            "unknown",
            "unknown",
            "synthetic",
            (1.0, 2.0),
            1,
            frame.contact_row,
            frame.contact_shape,
            frame.raw,
            (0, 1, 2),
            geometry,
            2,
            1,
            joint_received=now,
            joint_receipt_ros=now + 100,
        )
        rows.append((finger, FingerObservation("VALID", observation=obs, frame=frame)))
    return TactileSnapshot(now, tuple(rows), (0.0,) * 22, "VALID")


def estimator(kin, **kwargs):
    return TactileContactEstimator(
        kin,
        object_bodies=("object",),
        tactile_joint_names=tuple(HAND_NAMES),
        friction_mu=0.8,
        allow_candidate_geometry=True,
        **kwargs,
    )


class MixedContactTests(unittest.TestCase):
    """Operator-confirmed Sharpa no-contact semantics, with no simulation stepping."""

    def fixture(self, contacts=True):
        class Kinematics:
            def compute_jaco_a(self, q):
                pass

            def get_frame_pose(self, name):
                return np.eye(4)

            def get_frame_jaco(self, name, type="body"):
                return np.zeros((6, 28))

        kin = Kinematics()
        points = (
            [
                dict(
                    body1_name=f"right_{finger}_elastomer",
                    contact_pos=np.array([0.01, 0, 0]),
                    contact_frame=np.eye(3),
                    contact_force=np.array([2.0, 0, 0, 0, 0, 0]),
                )
                for finger in ("thumb", "index")
            ]
            if contacts
            else []
        )
        snapshot = snapshot_from_contacts(points, kin)
        rows = []
        for finger, result in snapshot.fingers:
            if result.status == "EMPTY":
                result = replace(
                    result,
                    status="MISSING",
                    transport_status="VALID",
                    frame=replace(result.frame, contact_status="MISSING", contact_shape=()),
                )
            rows.append((finger, result))
        snapshot = replace(snapshot, fingers=tuple(rows))
        fb = JointFeedback(tuple(JOINT_NAMES), (0.0,) * 28, 10, 10, "synthetic")
        return snapshot, fb, kin

    def test_thumb_index_valid_others_missing_preserves_raw_metadata(self):
        snapshot, fb, kin = self.fixture()
        before = asdict(snapshot)
        result = estimator(kin).estimate(snapshot, fb, 10)
        self.assertEqual(result.status, "eligible", result.reasons)
        self.assertEqual(
            {c["body1_name"] for c in result.observation.contacts}, {"right_thumb_elastomer", "right_index_elastomer"}
        )
        self.assertEqual(asdict(snapshot), before)
        for finger, row in snapshot.fingers:
            self.assertEqual(result.provenance["fingers"][finger], asdict(row.frame))
            expected = "CONTACT" if finger in ("thumb", "index") else "NO_CONTACT"
            self.assertEqual(result.provenance["finger_statuses"][finger]["interpretation"], expected)

    def test_missing_and_empty_all_no_contact_is_valid_empty(self):
        snapshot, fb, kin = self.fixture(contacts=False)
        rows = list(snapshot.fingers)
        finger, row = rows[0]
        rows[0] = (finger, replace(row, status="EMPTY", frame=replace(row.frame, contact_status="EMPTY")))
        result = estimator(kin).estimate(replace(snapshot, fingers=tuple(rows)), fb, 10)
        self.assertEqual(result.status, "valid_empty", result.reasons)
        self.assertEqual(result.observation.contacts, ())

    def test_missing_with_sensing_failure_is_unavailable_not_no_contact(self):
        snapshot, fb, kin = self.fixture()
        finger, row = snapshot.fingers[0]
        for fault in (
            "invalid",
            "stale",
            "disconnected",
            "no_topic",
            "transport",
            "f6",
            "nan",
            "source",
            "source_warning",
            "inconsistent",
        ):
            with self.subTest(fault=fault):
                bad = row
                if fault == "invalid":
                    bad = replace(row, status="INVALID", frame=replace(row.frame, contact_status="INVALID"))
                elif fault == "stale":
                    bad = replace(row, frame=replace(row.frame, received=9.7))
                elif fault == "disconnected":
                    bad = replace(row, status="DISCONNECTED")
                elif fault == "no_topic":
                    bad = replace(row, frame=None)
                elif fault == "transport":
                    bad = replace(row, transport_status="DISCONNECTED")
                elif fault == "f6":
                    bad = replace(row, frame=replace(row.frame, status="INVALID"))
                elif fault == "nan":
                    bad = replace(row, frame=replace(row.frame, raw=(float("nan"),) * 6))
                elif fault == "source":
                    bad = replace(row, frame=replace(row.frame, session_id=""))
                elif fault == "source_warning":
                    bad = replace(row, frame=replace(row.frame, warnings=("sdk_frame_id_regression",)))
                else:
                    bad = replace(row, frame=replace(row.frame, contact_count=1, contact_row=(1.0, 2.0, 3.0)))
                current = replace(snapshot, fingers=((finger, bad), *snapshot.fingers[1:]))
                result = estimator(kin).estimate(current, fb, 10)
                self.assertEqual(result.status, "insufficient_data")
                self.assertIsNone(result.observation)
                self.assertEqual(set(result.provenance["fingers"]), set(FINGERS))
                self.assertEqual(result.provenance["finger_statuses"][finger]["interpretation"], "UNAVAILABLE")

    def test_no_contact_source_regression_is_rejected(self):
        snapshot, fb, kin = self.fixture(contacts=False)
        est = estimator(kin)
        self.assertEqual(est.estimate(snapshot, fb, 10).status, "valid_empty")
        rows = tuple(
            (finger, replace(row, frame=replace(row.frame, received=10.1, sequence=0, sdk_frame_id=0)))
            for finger, row in snapshot.fingers
        )
        result = est.estimate(
            replace(snapshot, evaluated_at=10.1, fingers=rows),
            replace(fb, source_timestamp=10.1, receive_timestamp=10.1),
            10.1,
        )
        self.assertIsNone(result.observation)
        self.assertIn("reordered", result.reasons[0])


class CompatibilityTests(unittest.TestCase):
    def test_mujoco_equivalence_frames_forces_matrix_and_nonidentity_transforms(self):
        for reverse in (False, True):
            native, snapshot, fb, kin = equivalent_fixture(reverse)
            before = asdict(snapshot)
            result = estimator(kin).estimate(snapshot, fb, 10)
            self.assertEqual(result.status, "eligible", result.reasons)
            self.assertEqual(asdict(snapshot), before)
            contacts = result.observation.controller_contacts(10)
            native_by_body = {c["body1_name"]: c for c in native.contacts}
            ordered = [native_by_body[c["body1_name"]] for c in contacts]
            self.assertTrue(any(np.linalg.norm(c["contact_force"][1:3]) > 0 for c in ordered))
            for actual, expected in zip(contacts, ordered):
                for field in ("contact_pos", "contact_pos_local", "contact_frame", "contact_frame_local"):
                    np.testing.assert_allclose(actual[field], expected[field], atol=1e-12)
                np.testing.assert_allclose(actual["contact_force"][:3], expected["contact_force"][:3], atol=1e-12)
                np.testing.assert_array_equal(actual["contact_force"][3:], 0)
                self.assertIn("not_measured", actual["provenance"]["moment"])
                self.assertFalse(actual["provenance"]["hardware_eligible"])
            g_actual = GraspController.compute_grasp_matrix(None, contacts)
            g_expected = GraspController.compute_grasp_matrix(None, ordered)
            np.testing.assert_allclose(g_actual, g_expected, atol=1e-12)
            forces = np.concatenate([c["contact_force"][:3] for c in contacts])
            centroid = np.mean([c["contact_pos"] for c in contacts], axis=0)
            wrench = sum(
                (
                    np.r_[
                        c["contact_frame"] @ c["contact_force"][:3],
                        np.cross((c["contact_pos"] - centroid) * 100, c["contact_frame"] @ c["contact_force"][:3]),
                    ]
                    for c in contacts
                )
            )
            np.testing.assert_allclose(g_actual @ forces, wrench, atol=1e-10)

    def test_negative_projection_and_friction_violation_are_rejected_without_repair(self):
        _, snapshot, fb, kin = equivalent_fixture()
        for fault in ("sign", "friction"):
            rows = list(snapshot.fingers)
            finger, result = rows[0]
            obs = result.observation
            force = (
                -np.asarray(obs.force)
                if fault == "sign"
                else -(np.asarray(obs.normal) + 100 * contact_frame(np.asarray(obs.normal))[:, 1])
            )
            rows[0] = (finger, replace(result, observation=replace(obs, force=tuple(force))))
            estimate = estimator(kin).estimate(replace(snapshot, fingers=tuple(rows)), fb, 10)
            self.assertEqual(estimate.status, "insufficient_data")
            self.assertIn("negative normal" if fault == "sign" else "friction cone", estimate.reasons[0])
            self.assertIsNone(estimate.observation)

    def test_unknown_torque_channels_do_not_block_or_change_force(self):
        _, snapshot, fb, kin = equivalent_fixture()
        estimate = estimator(kin).estimate(snapshot, fb, 10)
        self.assertEqual(estimate.status, "eligible")
        for c in estimate.observation.contacts:
            self.assertEqual(c["provenance"]["observation"]["raw_f6"][3:], (123.0, -456.0, 789.0))

    def test_invalid_empty_missing_stale_partial_skew_and_object_ambiguity(self):
        _, snapshot, fb, kin = equivalent_fixture()
        for fault in ("invalid", "missing", "stale", "skew", "multiple", "pose", "channels"):
            rows = list(snapshot.fingers)
            finger, result = rows[0]
            now = 10
            if fault == "invalid":
                rows[0] = (finger, replace(result, status="INVALID"))
            elif fault == "missing":
                rows.pop()
            elif fault == "stale":
                now = 10.3
            elif fault == "skew":
                rows[0] = (finger, replace(result, observation=replace(result.observation, joint_received=9)))
            elif fault == "multiple":
                rows[0] = (
                    finger,
                    replace(
                        result,
                        frame=replace(result.frame, contact_count=2),
                        observation=replace(result.observation, contact_count=2),
                    ),
                )
            elif fault == "pose":
                fb = replace(fb, positions_rad=(0.01,) * 28)
            else:
                rows[0] = (finger, replace(result, observation=replace(result.observation, force_indices=(3, 4, 5))))
            estimate = estimator(kin).estimate(replace(snapshot, fingers=tuple(rows)), fb, now)
            self.assertEqual(estimate.status, "insufficient_data", fault)
            self.assertIsNone(estimate.observation)
        with self.assertRaises(ValueError):
            TactileContactEstimator(kin, object_bodies=("a", "b"), tactile_joint_names=HAND_NAMES, friction_mu=0.8)
        empty = snapshot_from_contacts([], kin)
        result = estimator(kin).estimate(empty, replace(fb, positions_rad=(0.0,) * 28), 10)
        self.assertEqual(result.status, "valid_empty")
        self.assertEqual(result.observation.contacts, ())

    def test_repeated_reads_age_and_reordered_sdk_frame_is_invalid(self):
        _, snapshot, fb, kin = equivalent_fixture()
        est = estimator(kin)
        self.assertEqual(est.estimate(snapshot, fb, 10).status, "eligible")
        self.assertEqual(est.estimate(snapshot, fb, 10.1).observation.timestamp, 10)
        self.assertEqual(est.estimate(snapshot, fb, 10.3).status, "insufficient_data")
        _, older, fb2, _ = equivalent_fixture(now=10.4, sequence=0)
        result = est.estimate(older, fb2, 10.4)
        self.assertIn("reordered", result.reasons[0])

    def test_geometry_candidate_requires_opt_in_and_stays_uncalibrated(self):
        _, snapshot, fb, kin = equivalent_fixture()
        est = TactileContactEstimator(kin, object_bodies=("object",), tactile_joint_names=HAND_NAMES, friction_mu=0.8)
        self.assertEqual(est.estimate(snapshot, fb, 10).status, "insufficient_data")

    def test_replay_roundtrip_determinism_and_no_timestamp_refresh(self):
        _, snapshot, fb, kin = equivalent_fixture()
        payload = {
            "schema_version": 1,
            "kind": "ada_tactile_adapter_replay",
            "origin_s": 10,
            "provenance": {"kind": "synthetic_adapter_output"},
            "tactile_joint_names": list(HAND_NAMES),
            "events": [{"delivered_at": 10, "snapshot": encode_snapshot(snapshot), "feedback": asdict(fb)}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(payload))
            a, b = TactileReplay(path), TactileReplay(path)
            for now in (10, 10.1, 10.4):
                sa, qa = a.sample(now)
                sb, qb = b.sample(now)
                self.assertEqual(asdict(sa), asdict(sb))
                self.assertEqual(qa, qb)
                result = estimator(kin).estimate(sa, qa, now)
                self.assertEqual(result.status, "eligible" if now <= 10.1 else "insufficient_data")
            with self.assertRaises(ValueError):
                a.sample(10)

    def test_portable_fixture_contact_on_off_invalid_reorder_and_gap(self):
        replay = TactileReplay(Path(__file__).parent / "fixtures/mpc_tactile_replay_v1.json")
        _, _, _, kin = equivalent_fixture()
        est = estimator(kin)
        statuses = []
        for now in (10.0, 10.1, 10.2, 10.3, 10.4, 10.8):
            snapshot, fb = replay.sample(now)
            statuses.append(est.estimate(snapshot, fb, now).status)
        self.assertEqual(
            statuses,
            ["eligible", "valid_empty", "eligible", "insufficient_data", "insufficient_data", "insufficient_data"],
        )
        self.assertFalse(replay.provenance["hardware_recording"])


if __name__ == "__main__":
    unittest.main()
