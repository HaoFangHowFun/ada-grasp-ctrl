"""Read-only force signs, native cone diagnostics, receipt replay and Viser overlay."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import tempfile
import unittest

import numpy as np

from ada_grasp_ctrl.tactile_preview.adapter import TactileAdapter
from ada_grasp_ctrl.tactile_preview.data import FINGERS, normalize_frame
from ada_grasp_ctrl.tactile_preview.force_viewer import BLUE, GREEN, ORANGE, ForcePreview, contact_diagnostic
from ada_grasp_ctrl.tactile_preview.replay import ReceiptReplay
from test_tactile_adapter import KnownHand
import test_tactile_preview as preview_tests
from test_tactile_preview import HAS_VIEWER, SHARE, message, synthetic_geometry


class ForceDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.hand = KnownHand()
        self.adapter = TactileAdapter(self.hand, synthetic_geometry(), clock=lambda: 10)
        self.joints = dict(
            names=self.hand.actuated_joint_names,
            positions=[0.0] * 22,
            received=10,
            receipt_ros=100,
            subscriber_ros=100.01,
        )

    def result(self, force, status="VALID", count=1):
        frame = normalize_frame(message("index"), "index", 10, 100.01)
        frame = replace(
            frame,
            raw=tuple(force) + (9.0, 8.0, 7.0),
            contact_status=status,
            contact_count=count,
            contact_shape=(count, 3) if count else (),
            contact_row=frame.contact_row if count else (),
        )
        snapshot = self.adapter.update({"index": frame}, self.joints)
        return dict(snapshot.fingers)["index"]

    def test_force_sign_rotation_and_native_cone(self):
        # Synthetic normal is local +Z, and KnownHand rotates both force and normal.
        for force, expected in (
            ([0, 0, -2], "PASS"),
            ([1, 0, -2], "FAIL"),
            ([0, 0, 2], "NEGATIVE_NORMAL_FORCE"),
            ([1, 0, 0], "NEGATIVE_NORMAL_FORCE"),
        ):
            row = contact_diagnostic(self.result(force), 0.3)
            self.assertEqual(row["cone"], expected)
            np.testing.assert_array_equal(row["f_object"], -np.array(row["f_sharpa"]))
            normal = self.hand.get_transform("right_index_elastomer")[:3, :3] @ [0, 0, 1]
            np.testing.assert_allclose(row["normal"], normal)
            self.assertAlmostEqual(row["fn_n"], -force[2])
            self.assertAlmostEqual(row["ft_n"], abs(force[0]))
            if row["fn_n"] <= 0:
                self.assertIsNone(row["mu_required"])
            else:
                self.assertAlmostEqual(row["mu_required"], abs(force[0]) / -force[2])
                self.assertAlmostEqual(row["cone_slack_n"], 0.3 * row["fn_n"] - row["ft_n"])

    def test_no_contact_ambiguity_and_unavailability(self):
        for status in ("MISSING", "EMPTY"):
            self.assertEqual(contact_diagnostic(self.result([0, 0, -2], status, 0), 0.3)["status"], "NO_CONTACT")
        self.assertEqual(contact_diagnostic(self.result([0, 0, -2], count=2), 0.3)["status"], "AMBIGUOUS")
        result = self.result([0, 0, -2])
        for bad in (
            replace(result, status="INVALID"),
            replace(result, transport_status="STALE"),
            replace(result, transport_status="DISCONNECTED"),
        ):
            self.assertEqual(contact_diagnostic(bad, 0.3)["status"], "UNAVAILABLE")
        self.assertEqual(contact_diagnostic(result, 0.3, "source regression")["status"], "UNAVAILABLE")

    def test_receipt_order_pause_seek_without_future_samples(self):
        frames = []
        for t, seq in ((1.0, 1), (2.0, 2)):
            frame = replace(normalize_frame(message("index"), "index", t, 100 + t), sequence=seq, sdk_frame_id=seq)
            frames.append(
                dict(topic="/sharpa/right_hand/tactile/index/frame", bag_ns=int(t * 1e9), frame=asdict(frame))
            )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decoded.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in frames))
            replay = ReceiptReplay(path)
            _, first = replay.sample(0.5)
            self.assertEqual(first["frames"]["index"]["sequence"], 1)
            self.assertEqual(replay.sample(0.5)[1], first)
            self.assertEqual(replay.sample(1)[1]["frames"]["index"]["sequence"], 2)
            self.assertEqual(replay.sample(0)[1]["frames"]["index"]["received"], 1)


@unittest.skipUnless(HAS_VIEWER, "Requires installed Viser and hand URDF")
class ForceRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import viser
        from ada_grasp_ctrl.tactile_preview.__main__ import parse_args
        from ada_grasp_ctrl.tactile_preview.kinematics import load_hand

        cls.args = parse_args(["--offline", "--receipt-clock-synchronized"])
        cls.args.force_replay = True
        cls.hand = load_hand(cls.args.urdf, SHARE)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        cls.server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
        cls.preview = ForcePreview(cls.server, cls.hand, cls.args, 0.3, synthetic_geometry())
        cls.addClassCleanup(cls.server.stop)

    def test_three_arrows_share_origin_opposite_forces_and_unchanged_normal(self):
        state = self.preview.update(preview_tests.ViewerTests.packet(self), True)
        for finger in FINGERS:
            row = state["force_diagnostics"][finger]
            sharpa, obj, normal = (
                handle[finger].points[0]
                for handle in (self.preview.arrows, self.preview.object_arrows, self.preview.normals)
            )
            for start in (sharpa[0], obj[0], normal[0]):
                np.testing.assert_allclose(start, row["position"], atol=1e-7)
            np.testing.assert_allclose(sharpa[1] - sharpa[0], -(obj[1] - obj[0]), atol=1e-7)
            np.testing.assert_allclose(normal[1] - normal[0], 0.025 * np.array(row["normal"]), atol=1e-7)
            for handles, color in (
                (self.preview.arrows, ORANGE),
                (self.preview.object_arrows, BLUE),
                (self.preview.normals, GREEN),
            ):
                np.testing.assert_array_equal(handles[finger].colors.reshape(-1, 3)[0], color)
                self.assertTrue(handles[finger].visible)

    def test_no_arrows_for_missing_ambiguous_stale_and_hand_fk_animates(self):
        packet = preview_tests.ViewerTests.packet(self)
        self.preview.update(packet, True)
        before = self.hand.get_transform("right_index_elastomer").copy()
        packet["joints"]["positions"][packet["joints"]["names"].index("right_index_MCP_FE")] = 0.3
        self.preview.update(packet, True)
        self.assertFalse(np.allclose(before, self.hand.get_transform("right_index_elastomer")))
        for status, count, expected in (
            ("MISSING", 0, "NO_CONTACT"),
            ("EMPTY", 0, "NO_CONTACT"),
            ("VALID", 2, "AMBIGUOUS"),
            ("INVALID", 0, "UNAVAILABLE"),
        ):
            packet = preview_tests.ViewerTests.packet(self)
            packet["frames"]["index"].update(
                contact_status=status,
                contact_count=count,
                contact_shape=(count, 3) if count else (),
                contact_row=(1.0, 2.0, 3.0) if count else (),
            )
            state = self.preview.update(packet, True)
            self.assertEqual(state["force_diagnostics"]["index"]["status"], expected)
            for handles in (self.preview.arrows, self.preview.object_arrows, self.preview.normals):
                self.assertFalse(handles["index"].visible)
        packet = preview_tests.ViewerTests.packet(self)
        packet["frames"]["index"]["received"] -= 1
        self.assertEqual(self.preview.update(packet, True)["force_diagnostics"]["index"]["status"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
