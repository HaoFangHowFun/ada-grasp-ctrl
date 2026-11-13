"""Force/frame contracts and actual Viser/URDF preview smoke tests."""

from dataclasses import asdict, replace
import importlib.util
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.tactile_preview.data import (
    FINGERS,
    Frame,
    LatestFrames,
    force_to_world,
    frame_status,
    normalize_frame,
    select_force,
)

SHARE = Path("/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave")
HAS_VIEWER = importlib.util.find_spec("viser") is not None and SHARE.exists()


def message(finger: str = "index") -> NS:
    """Create a coherent atomic F6/contact callback; the unused grid is not accessible."""
    return NS(
        schema_version=1,
        source="sharpa_sdk",
        side="right",
        finger=finger,
        channel=FINGERS.index(finger),
        f6=NS(status=1, shape=[1, 1, 6], raw_values=[1.0, -2.0, 3.0, 4.0, 5.0, 6.0], error=""),
        header=NS(stamp=NS(sec=100, nanosec=0), frame_id=""),
        session_id="test-session",
        receipt_sequence=7,
        sdk_frame_id_valid=True,
        sdk_frame_id=-3,
        sdk_frame_id_raw="-3",
        sdk_timestamp_valid=True,
        sdk_timestamp_raw=999999999.0,
        sdk_timestamp_repr="999999999",
        sdk_timestamp_unit="unknown",
        sdk_timestamp_epoch="unknown",
        identity_warnings=[],
        validation_errors=[],
        contact_points=NS(status=1, shape=[1, 3], raw_values=[124.8046875, 134.8828125, 11.578125], error=""),
        dist_force=object(),
    )


def synthetic_geometry() -> object:
    """A deterministic test-only map, with no claims about real SDK coordinates."""
    from ada_grasp_ctrl.tactile_preview.geometry import GeometryCatalog, GeometryMap, CHECKS

    rows, cols = np.indices((240, 240))
    xyz = np.stack((cols * 0.0001, rows * 0.0002, np.full_like(rows, 0.003, dtype=float)), axis=-1)
    normals = np.zeros_like(xyz)
    normals[:, :, 2] = 1
    profile = GeometryMap(
        xyz,
        normals,
        np.ones((240, 240), dtype=bool),
        meters_per_unit=1,
        T_E_S=np.eye(4),
        uv_to_col_row=[[1, 0, 0], [0, 1, 0]],
        verified=True,
        evidence={k: "Synthetic test fixture only" for k in CHECKS},
        source="synthetic test fixture",
        units="m (synthetic)",
    )
    return GeometryCatalog({f: profile for f in FINGERS})


class FrameTests(unittest.TestCase):
    def test_all_finger_identity_and_optional_blocks(self) -> None:
        for finger in FINGERS:
            frame = normalize_frame(message(finger), finger, 5.0, 100.02)
            self.assertEqual(frame.status, "VALID")
            self.assertEqual(frame.sdk_frame_id, -3)
            self.assertEqual(frame.shape, (1, 1, 6))
            self.assertEqual(frame.raw, (1, -2, 3, 4, 5, 6))

    def test_reject_schema_identity(self) -> None:
        for field, value in (
            ("schema_version", 2),
            ("source", "other"),
            ("side", "left"),
            ("finger", "toe"),
            ("channel", 8),
        ):
            with self.subTest(field=field):
                msg = message()
                setattr(msg, field, value)
                with self.assertRaises(ValueError):
                    normalize_frame(msg, "index", 5.0, 100.0)

    def test_block_states_and_malformed_payloads(self) -> None:
        for status, expected in ((0, "MISSING"), (2, "EMPTY"), (3, "INVALID"), (99, "INVALID")):
            msg = message()
            msg.f6.status = status
            frame = normalize_frame(msg, "index", 5.0, 100.0)
            self.assertEqual((frame.status, frame.raw), (expected, ()))
        for shape, raw in (
            ([], [1.0] * 6),
            ([1, 1, 5], [1.0] * 6),
            ([6], [1.0] * 5),
            ([6], [math.nan] * 6),
            ([6], [math.inf] * 6),
            ([-1, -6], [1.0] * 6),
        ):
            msg = message()
            msg.f6.shape, msg.f6.raw_values = shape, raw
            self.assertEqual(normalize_frame(msg, "index", 5.0, 100.0).status, "INVALID")

    def test_freshness_uses_ros_receipt_not_sdk_epoch(self) -> None:
        frame = normalize_frame(message(), "index", 5.0, 100.02)
        self.assertEqual(frame_status(frame, 5.1, 0.5, True)[0], "VALID")
        self.assertAlmostEqual(frame_status(frame, 5.1, 0.5, True)[2], 0.12)
        self.assertEqual(frame_status(frame, 6.0, 0.5, False)[0], "STALE")
        old = replace(frame, receipt_ros=98.0)
        self.assertEqual(frame_status(old, 5.1, 0.5, True)[0], "STALE")
        self.assertIsNone(frame_status(old, 5.1, 0.5, False)[2])
        future = replace(frame, receipt_ros=102.0)
        self.assertEqual(frame_status(future, 5.1, 0.5, True)[0], "CLOCK_ERROR")

    def test_invalid_sdk_identity_preserves_raw(self) -> None:
        msg = message()
        msg.sdk_frame_id_valid = msg.sdk_timestamp_valid = False
        frame = normalize_frame(msg, "index", 5.0, 100.0)
        self.assertIsNone(frame.sdk_frame_id)
        self.assertIsNone(frame.sdk_timestamp)
        self.assertEqual(frame.sdk_frame_id_raw, "-3")
        self.assertEqual(frame.sdk_timestamp_repr, "999999999")

    def test_selection_zero_norm_sign_and_rotations(self) -> None:
        raw = (1.0, -2.0, 3.0, 4.0, 5.0, 6.0)
        self.assertEqual(select_force(raw, (2, 0, 1)), (3, 1, -2))
        self.assertEqual(select_force(raw, (0, 1, 2), True), (-1, 2, -3))
        self.assertEqual(np.linalg.norm(select_force((0,) * 6, (0, 1, 2))), 0)
        for axis, rotation, expected in (
            ("x", [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [1, -3, -2]),
            ("y", [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [3, -2, -1]),
            ("z", [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [2, 1, 3]),
        ):
            with self.subTest(axis=axis):
                result = force_to_world(rotation, raw[:3])
                np.testing.assert_allclose(result, expected)
                self.assertAlmostEqual(np.linalg.norm(result), math.sqrt(14))
        for indices in ((0, 0, 1), (0, 1, 6), (0, 1), (True, 1, 2)):
            with self.assertRaises(ValueError):
                select_force(raw, indices)
        with self.assertRaises(ValueError):
            force_to_world(np.eye(4), [1, 2, 3])
        with self.assertRaises(ValueError):
            force_to_world(np.eye(3) * 2, [1, 2, 3])

    def test_cache_is_bounded_owned_and_rejects_old_arrow(self) -> None:
        cache = LatestFrames()
        msg = message()
        for _ in range(200):
            cache.receive(msg, "index", 100.0)
        msg.f6.raw_values[0] = 1234.0
        snapshot = cache.snapshot()
        self.assertEqual(snapshot["frames"]["index"]["raw"][0], 1)
        self.assertEqual(len(cache.times["index"]), 120)
        self.assertGreater(snapshot["rates"]["index"], 0)
        msg.side = "left"
        cache.receive(msg, "index", 100.0)
        self.assertNotIn("index", cache.snapshot()["frames"])
        self.assertIn("index", cache.snapshot()["errors"])


@unittest.skipUnless(HAS_VIEWER, "Requires existing Viser environment and installed official hand URDF")
class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import viser
        from scipy.spatial.transform import Rotation
        from ada_grasp_ctrl.tactile_preview.__main__ import parse_args
        from ada_grasp_ctrl.tactile_preview.viewer import Preview, load_hand

        cls.args = parse_args(["--offline", "--receipt-clock-synchronized"])
        cls.hand = load_hand(cls.args.urdf, SHARE)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        cls.server = viser.ViserServer(host="127.0.0.1", port=cls.port, verbose=False)
        cls.geometry = synthetic_geometry()
        cls.preview = Preview(cls.server, cls.hand, cls.args, geometry=cls.geometry)
        # Capture the initial browser scene before any test can deliver joint feedback.
        cls.initial_joint_transforms = []
        for joint, handle in zip(cls.hand.joint_map.values(), cls.preview.robot._joint_frames):
            actual = np.eye(4)
            actual[:3, 3] = handle.position
            actual[:3, :3] = Rotation.from_quat(np.asarray(handle.wxyz)[[1, 2, 3, 0]]).as_matrix()
            expected = cls.hand.get_transform(joint.child, joint.parent).copy()
            cls.initial_joint_transforms.append((joint.name, actual, expected))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def packet(self) -> dict:
        now = time.monotonic()
        return dict(
            frames={f: asdict(normalize_frame(message(f), f, now, 100.01)) for f in FINGERS},
            rates={f: 28.0 for f in FINGERS},
            joints=dict(
                names=self.hand.actuated_joint_names,
                positions=[0.0] * 22,
                received=now,
                receipt_ros=100.0,
                subscriber_ros=100.01,
                error="",
            ),
            endpoints={
                f: [
                    dict(
                        node="/synthetic",
                        type="dual_sharpa_wave/msg/TactileFrame",
                        reliability="BEST_EFFORT",
                        durability="VOLATILE",
                        depth=1,
                    )
                ]
                for f in FINGERS
            },
        )

    def setUp(self) -> None:
        self.preview.geometry = self.geometry
        self.preview.allow_geometry.value = False
        self.preview.mode.value = "World f_W"
        self.preview.flip.value = False
        self.preview.selected.value = "index"
        for widget, value in zip(self.preview.indices, (0, 1, 2)):
            widget.value = str(value)

    def test_http_and_no_data(self) -> None:
        with urlopen(f"http://127.0.0.1:{self.port}", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"<html", response.read().lower())
        state = self.preview.update({}, False)
        self.assertTrue(all(v["status"] == "NO DATA" and not v["arrow"] for v in state["fingers"].values()))

    def test_initial_mesh_transforms_without_joint_feedback(self) -> None:
        for name, actual, expected in self.initial_joint_transforms:
            np.testing.assert_allclose(actual, expected, atol=1e-7, err_msg=f"Initial displayed joint: {name}")

    def test_five_vectors_follow_current_fk(self) -> None:
        packet = self.packet()
        state = self.preview.update(packet, True)
        for finger in FINGERS:
            self.assertTrue(state["fingers"][finger]["arrow"])
            pose = self.hand.get_transform(f"right_{finger}_elastomer")
            points = self.preview.arrows[finger].points[0]
            contact = self.geometry.lookup(finger, packet["frames"][finger]["contact_row"][:2], False)
            np.testing.assert_allclose(points[0], pose[:3, 3] + pose[:3, :3] @ contact.p_E, atol=1e-7)
            np.testing.assert_allclose(self.preview.normals[finger].points[0, 0], points[0], atol=1e-7)
            np.testing.assert_allclose(self.preview.markers[finger].position, points[0], atol=1e-7)
            np.testing.assert_allclose(points[1] - points[0], 0.02 * pose[:3, :3] @ [1, -2, 3], atol=1e-7)
        before = np.array(state["fingers"]["index"]["origin"])
        packet["joints"]["positions"][packet["joints"]["names"].index("right_index_MCP_FE")] = 0.3
        state = self.preview.update(packet, True)
        self.assertGreater(np.linalg.norm(before - state["fingers"]["index"]["origin"]), 0.001)

    def test_renderer_uses_exact_adapter_observation_and_flip_is_display_only(self) -> None:
        packet = self.packet()
        self.preview.flip.value = True
        self.preview.update(packet, True)
        for finger, result in self.preview.last_snapshot.fingers:
            observation = result.observation
            self.assertIsNotNone(observation)
            points = self.preview.arrows[finger].points[0]
            np.testing.assert_allclose(points[0], observation.position, atol=1e-7)
            np.testing.assert_allclose(points[1] - points[0], -0.02 * np.asarray(observation.force), atol=1e-7)
            np.testing.assert_allclose(self.preview.markers[finger].position, observation.position, atol=1e-7)
            normal = self.preview.normals[finger].points[0]
            np.testing.assert_allclose(normal[1] - normal[0], 0.025 * np.asarray(observation.normal), atol=1e-7)
            self.assertEqual(result.local.force, (1, -2, 3))
            self.assertEqual(observation.force_convention, "object_on_finger")

    def test_missing_stale_disconnect_and_reconnect_hide_arrows(self) -> None:
        for condition in ("missing", "invalid", "stale", "disconnect", "pose", "skew", "duplicate_publisher"):
            packet = self.packet()
            if condition in ("missing", "invalid"):
                packet["frames"]["index"]["status"] = condition.upper()
                packet["frames"]["index"]["raw"] = ()
            if condition == "stale":
                packet["frames"]["index"]["receipt_ros"] -= 10
            if condition == "pose":
                packet["joints"]["received"] -= 10
            if condition == "skew":
                packet["joints"]["receipt_ros"] -= 0.25
            if condition == "duplicate_publisher":
                packet["endpoints"]["index"] *= 2
            with self.subTest(condition=condition):
                state = self.preview.update(packet, condition != "disconnect")
                self.assertFalse(state["fingers"]["index"]["arrow"])
                self.assertFalse(state["fingers"]["index"]["contact_visible"])
                self.assertFalse(self.preview.normals["index"].visible)
        self.assertTrue(self.preview.update(self.packet(), True)["fingers"]["index"]["arrow"])

    def test_local_mode_sign_and_zero(self) -> None:
        packet = self.packet()
        packet["joints"] = {}
        self.preview.mode.value = "Local f_E"
        self.preview.flip.value = True
        self.preview.update(packet, True)
        self.assertTrue(self.preview.local_arrow.visible)
        points = self.preview.local_arrow.points[0]
        np.testing.assert_allclose(points[1] - points[0], 0.02 * np.array([-1, 2, -3]), atol=1e-7)
        packet["frames"]["index"]["raw"] = (0.0,) * 6
        self.preview.update(packet, True)
        self.assertFalse(self.preview.local_arrow.visible)
        self.assertTrue(self.preview.local_marker.visible)
        self.assertTrue(self.preview.local_normal.visible)

    def test_contact_disappears_without_reusing_previous_frame(self) -> None:
        for status in ("MISSING", "EMPTY", "INVALID"):
            packet = self.packet()
            self.assertTrue(self.preview.update(packet, True)["fingers"]["index"]["contact_visible"])
            packet["frames"]["index"].update(
                contact_status=status, contact_row=(), contact_count=0, sdk_frame_id=999, sequence=8
            )
            state = self.preview.update(packet, True)
            self.assertEqual(state["fingers"]["index"]["contact_status"], status)
            self.assertFalse(self.preview.markers["index"].visible)
            self.assertFalse(self.preview.normals["index"].visible)
            self.assertFalse(self.preview.arrows["index"].visible)
        self.assertTrue(self.preview.update(self.packet(), True)["fingers"]["index"]["contact_visible"])

    def test_contact_geometry_error_and_unverified_opt_in(self) -> None:
        from ada_grasp_ctrl.tactile_preview.geometry import GeometryCatalog

        packet = self.packet()
        packet["frames"]["index"]["contact_row"] = (-0.01, 120, 11.0)
        self.assertFalse(self.preview.update(packet, True)["fingers"]["index"]["contact_visible"])
        self.preview.geometry = GeometryCatalog(errors={f: "missing map" for f in FINGERS})
        self.assertFalse(self.preview.update(self.packet(), True)["fingers"]["index"]["contact_visible"])
        profile = synthetic_geometry().maps["index"]
        profile.verified = False
        self.preview.geometry = GeometryCatalog({f: profile for f in FINGERS})
        self.assertFalse(self.preview.update(self.packet(), True)["fingers"]["index"]["contact_visible"])
        self.preview.allow_geometry.value = True
        self.assertTrue(self.preview.update(self.packet(), True)["fingers"]["index"]["contact_visible"])
        self.assertTrue(self.preview.geometry_labels["index"].visible)

    def test_multiple_contacts_use_only_first_and_force_stays_resultant(self) -> None:
        packet = self.packet()
        msg = message()
        msg.contact_points.shape = [2, 4]
        msg.contact_points.raw_values = [20.0, 30.0, 11.0, 42.0, 200.0, 210.0, 100.0, 900.0]
        packet["frames"]["index"] = asdict(normalize_frame(msg, "index", time.monotonic(), 100.01))
        state = self.preview.update(packet, True)["fingers"]["index"]
        self.assertEqual(state["contact_count"], 2)
        self.assertEqual((state["pixel_row"], state["pixel_col"]), (30, 20))
        self.assertEqual(state["f_E"], (1.0, -2.0, 3.0))
        self.assertAlmostEqual(state["norm"], math.sqrt(14))

    def test_joint_identity_finite_and_limits(self) -> None:
        from ada_grasp_ctrl.tactile_preview.viewer import joint_configuration

        for invalid in ("missing", "duplicate", "nan", "limit"):
            packet = self.packet()["joints"]
            packet["names"] = list(packet["names"])
            if invalid == "missing":
                packet["names"].pop()
            elif invalid == "duplicate":
                packet["names"][0] = packet["names"][1]
            elif invalid == "nan":
                packet["positions"][0] = math.nan
            else:
                packet["positions"][0] = 100
            self.assertIsNone(joint_configuration(packet, self.hand, time.monotonic(), 0.5, True)[0])

    def test_entrypoint_port_8089_start_stop(self) -> None:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", 8089))
            except OSError:
                self.skipTest("Port 8089 already in use; preserving existing server")
        process = subprocess.Popen(
            [sys.executable, "-m", "ada_grasp_ctrl.tactile_preview", "--offline", "--run-seconds", "3"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    with urlopen("http://127.0.0.1:8089", timeout=0.3) as response:
                        self.assertEqual(response.status, 200)
                    break
                except OSError:
                    if process.poll() is not None:
                        self.fail(process.communicate()[0])
                    time.sleep(0.1)
            else:
                self.fail("Viewer HTTP startup timed out")
            output, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, output)
            self.assertIn("PREVIEW_URL=http://127.0.0.1:8089", output)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        with socket.socket() as probe:
            self.assertNotEqual(probe.connect_ex(("127.0.0.1", 8089)), 0)

    @unittest.skipUnless(os.environ.get("ADA_TACTILE_DDS_TEST") == "1", "Opt-in isolated synthetic ROS DDS integration")
    def test_real_dds_to_viewer_synthetic_five_finger_and_pose(self) -> None:
        from ada_grasp_ctrl.tactile_preview.__main__ import RosProcess

        # Only these test children use this isolated localhost domain. No hardware driver is started.
        with patch.dict(os.environ, ROS_DOMAIN_ID="198", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST"):
            publisher = subprocess.Popen(["/usr/bin/python3", str(project_root() / "tests/tactile_ros_fixture.py")])
            reader = RosProcess("/usr/bin/python3", "/sharpa/right_hand/joint_states")
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                packet, connected, error = reader.snapshot()
                state = self.preview.update(packet, connected, error)
                if all(v["arrow"] for v in state["fingers"].values()):
                    break
                time.sleep(0.05)
            else:
                self.fail(str(state))
            self.assertEqual(set(packet["frames"]), set(FINGERS))
            self.assertTrue(all(Frame(**f).status == "VALID" for f in packet["frames"].values()))
            publisher.terminate()
            publisher.wait(timeout=5)
            time.sleep(0.7)
            state = self.preview.update(*reader.snapshot())
            self.assertTrue(all(not v["arrow"] for v in state["fingers"].values()))
        finally:
            reader.close()
            if publisher.poll() is None:
                publisher.terminate()
                publisher.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
