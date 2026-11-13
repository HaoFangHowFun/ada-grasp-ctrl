"""Read-only observation contracts with deterministic clocks and independently known FK."""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import math
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import HAND_NAMES
from ada_grasp_ctrl.tactile_preview import TactileAdapter, TactileContactObservation
from ada_grasp_ctrl.tactile_preview.data import FINGERS, normalize_frame
from ada_grasp_ctrl.tactile_preview.geometry import GeometryCatalog
from ada_grasp_ctrl.tactile_preview.geometry import SDK_LIBRARY
from test_tactile_contact_geometry import profile
from test_tactile_preview import message, synthetic_geometry


class KnownHand:
    """Exact test transform: rotate about X by 90 degrees, then joint-controlled Z."""

    base_link = "right_hand_flange"
    actuated_joint_names = list(HAND_NAMES)

    def __init__(self) -> None:
        self.cfg = np.zeros(22)
        self.joint_map = {n: NS(limit=NS(lower=-3.2, upper=3.2)) for n in HAND_NAMES}

    def update_cfg(self, q: object) -> None:
        self.cfg = np.array(q, dtype=float)

    def get_transform(self, name: str) -> np.ndarray:
        finger = name.removeprefix("right_").removesuffix("_elastomer")
        c, s = math.cos(self.cfg[0]), math.sin(self.cfg[0])
        pose = np.eye(4)
        pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        pose[:3, 3] = [0.1 * FINGERS.index(finger), 0.2, 0.3]
        return pose


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 10.0
        self.hand = KnownHand()
        self.geometry = synthetic_geometry()
        self.adapter = TactileAdapter(self.hand, self.geometry, clock=lambda: self.now)
        self.frames = {f: asdict(normalize_frame(message(f), f, 10, 100.01)) for f in FINGERS}
        self.joints = dict(
            names=list(HAND_NAMES),
            positions=[0.0] * 22,
            received=10.0,
            receipt_ros=100.0,
            subscriber_ros=100.01,
            error="",
        )

    def update(self, **options: object) -> dict:
        return dict(self.adapter.update(self.frames, self.joints, **options).fingers)

    def test_five_fingers_immutable_vectors_frames_and_provenance(self) -> None:
        results = self.update()
        for finger, result in results.items():
            obs = result.observation
            self.assertIsInstance(obs, TactileContactObservation)
            self.assertEqual(result.status, "VALID")
            self.assertEqual(obs.frame_id, "right_hand_flange")
            self.assertEqual(obs.force_convention, "object_on_finger")
            self.assertEqual(obs.force_indices, (0, 1, 2))
            self.assertEqual(obs.force, (1, -3, -2))
            self.assertEqual(result.local.force, (1, -2, 3))
            self.assertAlmostEqual(np.linalg.norm(obs.force), math.sqrt(14))
            self.assertAlmostEqual(np.linalg.norm(obs.normal), 1)
            self.assertEqual(obs.raw_f6, (1, -2, 3, 4, 5, 6))
            self.assertEqual(obs.contact_uv, (124.8046875, 134.8828125))
            self.assertEqual((obs.sdk_frame_id, obs.sequence, obs.session_id), (-3, 7, "test-session"))
            self.assertEqual(obs.timestamp, 10)
            self.assertEqual(obs.joint_received, 10)
            self.assertEqual(obs.geometry.verification, "VERIFIED BY CONFIG EVIDENCE")
            with self.assertRaises(FrozenInstanceError):
                obs.force = (0, 0, 0)
            with self.assertRaises(TypeError):
                obs.position[0] = 42
        self.frames["index"]["raw"] = (999,) * 6
        self.joints["positions"][0] = 2
        self.assertEqual(self.adapter.get_contacts()["index"].force, (1, -3, -2))
        self.assertEqual(set(self.adapter.get_contacts()), set(FINGERS))

    def test_nonidentity_map_and_fk_transform(self) -> None:
        transform = np.array([[0, -1, 0, 0.01], [1, 0, 0, 0.02], [0, 0, 1, 0.03], [0, 0, 0, 1]])
        self.adapter.geometry = GeometryCatalog({f: profile(T_E_S=transform) for f in FINGERS})
        self.frames["index"].update(contact_row=(2, 1, 99), contact_count=1)
        self.joints["positions"][0] = math.pi / 2
        result = self.update()["index"]
        # Raw point [2,10,2] mm -> S-to-E -> E [0,.022,.032].
        np.testing.assert_allclose(result.local.position, [0, 0.022, 0.032], atol=1e-12)
        # R_B_E maps [x,y,z] -> [z,x,y], plus [.3,.2,.3].
        np.testing.assert_allclose(result.observation.position, [0.332, 0.2, 0.322], atol=1e-12)
        np.testing.assert_allclose(result.observation.normal, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(result.observation.force, [3, 1, -2], atol=1e-12)

    def test_multiple_contacts_keep_row_zero_and_atomic_identity(self) -> None:
        msg = message()
        msg.contact_points.shape = [2, 3]
        msg.contact_points.raw_values = [20, 30, 11, 200, 210, 99]
        msg.sdk_frame_id, msg.receipt_sequence = 41024, 8
        msg.f6.raw_values = [4, 5, 6, 7, 8, 9]
        self.frames["index"] = normalize_frame(msg, "index", 10, 100.01)
        obs = self.update()["index"].observation
        self.assertEqual((obs.contact_count, obs.pixel_row, obs.pixel_col), (2, 30, 20))
        self.assertEqual((obs.contact_row, obs.raw_f6, obs.sdk_frame_id), ((20, 30, 11), (4, 5, 6, 7, 8, 9), 41024))

    def test_release_and_invalid_blocks_never_reuse_previous_contact(self) -> None:
        for status in ("MISSING", "EMPTY", "INVALID"):
            self.frames["index"] = asdict(normalize_frame(message(), "index", 10, 100.01))
            self.assertIsNotNone(self.update()["index"].observation)
            self.frames["index"].update(contact_status=status, contact_row=(), contact_count=0, sequence=8)
            result = self.update()["index"]
            self.assertEqual(result.status, status)
            self.assertIsNone(result.observation)
            self.assertIsNone(result.local)
        self.frames.pop("index")
        self.assertEqual(self.update()["index"].status, "NO_DATA")
        self.assertEqual(self.update(errors={"index": "wrong source"})["index"].status, "INVALID")

    def test_expiration_without_update_and_repeated_snapshot_does_not_refresh_age(self) -> None:
        self.update()
        self.now += 0.6
        self.assertTrue(all(c is None for c in self.adapter.get_contacts().values()))
        self.assertEqual(set(self.adapter.get_statuses().values()), {"STALE"})
        self.assertEqual(self.update()["index"].status, "STALE")

    def test_missing_stale_invalid_and_skewed_joints_keep_only_explicit_local_diagnostic(self) -> None:
        cases = [
            ({}, "POSE_MISSING"),
            ({"received": 9}, "POSE_STALE"),
            ({"received": 9.8}, "POSE_SKEWED"),
            ({"positions": [float("nan")] * 22}, "POSE_INVALID"),
            ({"receipt_ros": float("nan")}, "POSE_INVALID"),
        ]
        original = dict(self.joints)
        for changes, status in cases:
            self.joints = dict(original, **changes) if changes else {}
            result = self.update()["index"]
            self.assertEqual(result.status, status)
            self.assertIsNone(result.observation)
            self.assertEqual(result.local.frame_id, "right_index_elastomer")

    def test_clock_domains_and_transport_fail_closed(self) -> None:
        self.frames["index"]["sdk_timestamp"] = -1e30
        self.assertEqual(self.update()["index"].status, "VALID")
        self.frames["index"]["received"] = 100
        self.assertEqual(self.update()["index"].status, "CLOCK_ERROR")
        self.frames["index"]["received"] = 10
        self.assertEqual(self.update(connected=False)["index"].status, "DISCONNECTED")
        self.assertIsNone(self.update(endpoints={})["index"].observation)
        self.adapter.synchronized = True
        self.frames["index"]["receipt_ros"] = 90
        self.assertEqual(self.update()["index"].status, "STALE")

    def test_unverified_missing_hole_bounds_and_unit_errors(self) -> None:
        p = profile(verified=False)
        self.adapter.geometry = GeometryCatalog({"index": p}, profile_names={"index": "known-test"})
        self.frames["index"]["contact_row"] = (2, 1, 99)
        self.assertEqual(self.update()["index"].status, "GEOMETRY_UNVERIFIED")
        result = self.update(allow_unverified_geometry=True)["index"]
        self.assertEqual(result.observation.geometry.profile, "known-test")
        self.assertEqual(result.observation.geometry.verification, "UNVERIFIED DIAGNOSTIC")
        self.assertEqual(self.update()["thumb"].status, "GEOMETRY_UNAVAILABLE")
        for uv in ((-1, 0, 1), (4, 0, 1)):
            self.frames["index"]["contact_row"] = uv
            self.assertEqual(self.update(allow_unverified_geometry=True)["index"].status, "GEOMETRY_INVALID")
        for units, scale in (("unknown", 1), ("inch", 0.0254), ("mm", 1)):
            with self.assertRaises(ValueError):
                profile(units=units, meters_per_unit=scale)
        normals = np.zeros((3, 4, 3))
        self.adapter.geometry = GeometryCatalog({"index": profile(normals=normals)})
        self.frames["index"]["contact_row"] = (2, 1, 99)
        self.assertEqual(self.update()["index"].status, "GEOMETRY_INVALID")

    def test_invalid_f6_mapping_and_normalized_payload(self) -> None:
        self.assertIn(
            "not operator-validated", self.update(force_indices=(2, 0, 1))["index"].observation.force_convention_quality
        )
        self.assertEqual(self.update(force_indices=(0, 0, 2))["index"].status, "INVALID_FORCE")
        self.frames["index"]["raw"] = (float("nan"), 0, 0, 0, 0, 0)
        self.assertEqual(self.update()["index"].status, "INVALID_FORCE")
        self.frames["index"] = asdict(normalize_frame(message(), "index", 10, 100.01))
        self.frames["index"]["contact_count"] = 99
        self.assertEqual(self.update()["index"].status, "INVALID")
        self.frames["index"]["finger"] = "thumb"
        self.assertEqual(self.update()["index"].status, "INVALID")
        with self.assertRaises(ValueError):
            self.adapter.update({"left_index": self.frames["index"]}, self.joints)
        self.assertTrue(all(c is None for c in self.adapter.get_contacts().values()))

    def test_duplicate_and_regressed_sdk_ids_preserve_whole_arrival_record(self) -> None:
        for sdk_id, seq, raw in ((4, 8, 2), (4, 8, 2), (-10, 9, 7)):
            self.frames["index"].update(sdk_frame_id=sdk_id, sequence=seq, raw=(raw,) * 6)
            obs = self.update()["index"].observation
            self.assertEqual((obs.sdk_frame_id, obs.sequence, obs.raw_f6), (sdk_id, seq, (raw,) * 6))
            self.assertEqual(obs.timestamp, 10)

    def test_observation_constructor_rejects_bad_vectors_and_owns_arrays(self) -> None:
        obs = self.update()["index"].observation
        for updates in ({"normal": [0, 0, 0]}, {"force": [1, 2]}, {"position": [1, math.inf, 3]}):
            with self.assertRaises(ValueError):
                replace(obs, **updates)
        position = np.array([1, 2, 3])
        copied = replace(obs, position=position)
        position[0] = 999
        self.assertEqual(copied.position, (1, 2, 3))

    @unittest.skipUnless(SDK_LIBRARY.exists(), "Audited installed SDK static maps unavailable")
    def test_sanitized_capture_matches_pre_refactor_geometry_math(self) -> None:
        self.adapter.geometry = GeometryCatalog.installed_candidate()
        self.frames["index"].update(
            contact_row=(146.8359375, 134.1796875, 14.9921875),
            sdk_frame_id=41024,
            raw=(-0.02685546875, 0.90478515625, 10.84375, -0.179931640625, -0.017791748046875, -0.0017414093017578125),
        )
        result = self.update(allow_unverified_geometry=True)["index"]
        self.assertIsNotNone(result.observation, result.detail)
        obs = result.observation
        # Frozen numeric evidence from the previous preview, independent of adapter math.
        p_e = np.array([0.0015695039, -0.0166018314, -0.0060232024])
        n_e = np.array([0.1525959183, -0.0953974431, -0.9836736316])
        pose = self.hand.get_transform("right_index_elastomer")
        np.testing.assert_allclose(obs.position, pose[:3, 3] + pose[:3, :3] @ p_e, atol=1e-9)
        np.testing.assert_allclose(obs.normal, pose[:3, :3] @ n_e, atol=1e-9)
        np.testing.assert_allclose(obs.force, pose[:3, :3] @ np.array(obs.raw_f6[:3]), atol=1e-12)
        self.assertEqual((obs.pixel_row, obs.pixel_col, obs.sdk_frame_id), (134, 147, 41024))
        self.assertIn("sha256=", obs.geometry.source)
        self.assertEqual(obs.geometry.verification, "UNVERIFIED DIAGNOSTIC")

    def test_public_import_has_no_renderer_ros_or_simulator(self) -> None:
        code = (
            "from ada_grasp_ctrl.tactile_preview import TactileAdapter, TactileContactObservation; "
            "import sys,json; print(json.dumps(sorted(set(sys.modules) & {'viser','rclpy','mujoco'})))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
