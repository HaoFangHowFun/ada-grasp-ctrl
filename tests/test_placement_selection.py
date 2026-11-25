"""Orientation selection contracts and the grasp/hardware execution boundary."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ada_grasp_ctrl.pipeline import dual_runner
from ada_grasp_ctrl.placement.session import PlacementCandidate, PlacementSession, sha256
from ada_grasp_ctrl.placement.uop_adapter import align_vectors, fit_candidates
from ada_grasp_ctrl.placement.ui import PlacementPanel


def candidate(instance: int, normal: list) -> dict:
    """Create explicit plane fixtures with independently checkable rotations."""
    return dict(
        instance_id=instance,
        relative_rotation=align_vectors(normal, [0, 0, -1]).tolist(),
        support_normal=normal,
        plane_parameters=[*normal, -0.1],
        inlier_count=50,
        inlier_ratio=0.5,
    )


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.points = np.random.default_rng(2).normal(size=(150, 3)) * 0.02 + [0.3, -0.4, 0.2]
        np.savez(
            self.root / "snapshot.npz",
            object_points_world_m=self.points,
            environment_points_world_m=self.points,
            object_source_indices=np.arange(150),
        )
        self.session = PlacementSession(
            self.root,
            self.points,
            "captured-object",
            dict(
                snapshot_sha256=sha256(self.root / "snapshot.npz"),
                frame_id="world",
                rotation_convention="active column; rows @ R.T",
                calibration={"sha256": "calibration"},
            ),
        )
        self.session.save_candidates()

    def test_original_default_and_no_plane(self):
        self.assertEqual(self.session.selected_id, "P00")
        self.session.infer_once(lambda: [])
        self.assertEqual([c.candidate_id for c in self.session.candidates], ["P00"])
        self.assertEqual(self.session.inference_status, "no_plane")
        self.assertTrue(self.session.confirm().is_file())

    def test_failed_inference_keeps_original_and_attempts_only_once(self):
        infer = Mock(side_effect=RuntimeError("GPU unavailable"))
        self.session.infer_once(infer)
        self.session.infer_once(infer)
        infer.assert_called_once()
        self.assertEqual(self.session.selected.source, "original")
        self.assertIn("GPU unavailable", self.session.inference_error)
        self.session.confirm()

    def test_single_candidate_and_identity_not_deduplicated(self):
        self.session.infer_once(lambda: [candidate(1, [0, 0, -1])])
        self.assertEqual(len(self.session.candidates), 2)
        self.assertEqual(self.session.selected_id, "P00")
        np.testing.assert_array_equal(self.session.candidates[1].relative_rotation, np.eye(3))

    def test_switches_restore_original_without_inference_or_accumulation(self):
        infer = Mock(return_value=[candidate(8, [1, 0, 0]), candidate(2, [0, 1, 0])])
        self.session.infer_once(infer)
        self.assertEqual([c.instance_id for c in self.session.candidates], [None, 2, 8])
        self.session.select("P01")
        second = self.session.select("P02")
        expected = (self.points - self.points.mean(0)) @ np.array(self.session.selected.relative_rotation).T
        np.testing.assert_allclose(second, expected + self.points.mean(0), atol=1e-12)
        np.testing.assert_array_equal(self.session.select("P00"), self.points)
        np.testing.assert_array_equal(self.session.points, self.points)
        infer.assert_called_once()

    def test_simulation_preview_grounds_rotated_cloud_and_round_trips_without_drift(self):
        for normal in ([0, 0, 1], [0, 0.6, 0.8]):
            for offset in (0.4, -0.4):
                with self.subTest(normal=normal, offset=offset):
                    metadata = {**self.session.metadata, "preview_table_plane_world": [*normal, offset]}
                    session = PlacementSession(self.root, self.points, "sim-object", metadata)
                    infer = Mock(return_value=[candidate(1, [1, 0, 0]), candidate(2, [0, 1, 0])])
                    session.infer_once(infer)
                    first = session.select("P01")
                    for key in ("P02", "P01", "P02", "P01"):
                        actual = session.select(key)
                        distances = actual @ normal + offset
                        self.assertGreaterEqual(distances.min(), -1e-12)
                        self.assertAlmostEqual(distances.min(), 0, places=12)
                        rotated = (self.points - session.center) @ np.array(session.selected.relative_rotation).T
                        rotated += session.center
                        np.testing.assert_array_equal(actual[:, :2], rotated[:, :2])
                        _, shift = session.preview_geometry()
                        if offset == 0.4:
                            self.assertLess(shift[2], 0)
                        else:
                            self.assertGreater(shift[2], 0)
                    np.testing.assert_array_equal(actual, first)
                    saved = json.loads(session.confirm().read_text())
                    np.testing.assert_array_equal(saved["preview_translation_world_m"], shift)
                    self.assertEqual(saved["translation_status"], "undetermined")
                    self.assertFalse(saved["stability_certified"])
                    restored = PlacementSession.load(self.root)
                    np.testing.assert_array_equal(restored.select("P01"), first)
                    np.testing.assert_array_equal(session.select("P00"), self.points)
                    np.testing.assert_array_equal(session.points, self.points)
                    infer.assert_called_once()

    def test_invalid_preview_table_plane_is_rejected(self):
        for plane in ([0, 0, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 1, float("nan")], [0, 1]):
            with self.subTest(plane=plane), self.assertRaisesRegex(ValueError, "Preview table plane"):
                PlacementSession(self.root, self.points, "id", {"preview_table_plane_world": plane})

    def test_serialization_and_confirmation_artifact(self):
        self.session.infer_once(lambda: [candidate(2, [0, 1, 0])])
        self.session.select("P01")
        path = self.session.confirm()
        record = json.loads(path.read_text())
        self.assertEqual(record["candidate_id"], "P01")
        self.assertEqual(record["relative_rotation"], self.session.selected.relative_rotation)
        self.assertEqual(record["frame_convention"], self.session.metadata)
        self.assertEqual(record["object_snapshot_id"], "captured-object")
        self.assertEqual(record["source"], "uop_net")
        self.assertEqual(record["translation_status"], "undetermined")
        self.assertNotIn("pose", record)
        restored = PlacementSession.load(self.root)
        self.assertEqual(restored.confirmed_id, "P01")
        self.assertEqual(restored.selected_id, "P00")
        self.assertEqual(asdict(restored.candidates[1]), asdict(self.session.candidates[1]))
        self.assertEqual(sha256(self.root / record["candidates_file"]), record["candidates_sha256"])

    def test_p00_confirmation_remains_reproducible_after_inference(self):
        record = json.loads(self.session.confirm().read_text())
        self.session.infer_once(lambda: [candidate(1, [1, 0, 0])])
        self.assertEqual(sha256(self.root / record["candidates_file"]), record["candidates_sha256"])
        self.assertEqual(self.session.confirmed_id, "P00")

    def test_bad_rotation_and_snapshot_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            PlacementCandidate("P00", "original", (-np.eye(3)).tolist(), "id")
        (self.root / "snapshot.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            PlacementSession.load(self.root)

    def test_real_plane_instance_adapter_filters_only_official_eligibility(self):
        # Positive eligible groups 2 and 8; group 3 has only 49 points.
        labels = np.r_[np.zeros(50), np.full(50, 2), np.full(49, 3), np.full(60, 8)]
        points = np.random.default_rng(7).normal(size=(len(labels), 3))
        fit = Mock(return_value=(np.array([0, 0, -1, -0.2]), np.arange(12)))
        rows, errors = fit_candidates(points, labels, fit, Mock(), np.array([0.2, 0.3, 0.4]), 2, [0, 0, 1])
        self.assertEqual([r["instance_id"] for r in rows], [2, 8])
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["inlier_count"], 12)
        self.assertAlmostEqual(rows[0]["inlier_ratio"], 12 / 50)
        self.assertAlmostEqual(rows[0]["plane_parameters"][3], 0)

    def test_plane_fit_failure_and_single_instance(self):
        points = np.tile([0.2, 0.1, -0.4], (100, 1))
        labels = np.r_[np.ones(50), np.full(50, 2)]
        fit = Mock(side_effect=[RuntimeError("fit failed"), (np.array([0, 0, -1, -0.4]), np.arange(50))])
        rows, errors = fit_candidates(points, labels, fit, Mock(), np.zeros(3), 1, [0, 0, 1])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["instance_id"], 2)
        self.assertEqual(errors[0]["instance_id"], 1)

    def test_rotation_maps_normal_to_tilted_table_and_plane_to_world(self):
        normal = np.array([1.0, 0, 0])
        table = np.array([0.0, 0.6, 0.8])
        points = np.tile(normal, (60, 1))
        raw = align_vectors(normal, [0, 0, -1])
        rows, _ = fit_candidates(
            points,
            np.ones(60),
            lambda p: ([1, 0, 0, -1], np.arange(60)),
            lambda *args: raw,
            np.array([0.3, 0.4, 0.5]),
            2,
            table,
        )
        np.testing.assert_allclose(np.array(rows[0]["relative_rotation"]) @ normal, -table, atol=1e-12)
        plane = np.asarray(rows[0]["plane_parameters"])
        self.assertAlmostEqual(np.dot(plane[:3], [2.3, 0.4, 0.5]) + plane[3], 0)
        for n in ([0, 0, 1], [0, 0, -1], normal):
            r = align_vectors(n, [0, 0, -1])
            np.testing.assert_allclose(r @ n, [0, 0, -1], atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r), 1)

    def test_ui_switch_and_confirm_have_no_inference_callback(self):
        class GUI:
            def control(self, *args, **kwargs):
                return SimpleNamespace(
                    value=kwargs.get("initial_value"),
                    on_update=Mock(),
                    on_click=Mock(),
                    disabled=False,
                )

            add_markdown = add_button = add_dropdown = control

        server = SimpleNamespace(
            gui=GUI(), scene=SimpleNamespace(add_point_cloud=Mock()), initial_camera=SimpleNamespace()
        )
        infer = Mock(return_value=[candidate(1, [1, 0, 0]), candidate(2, [0, 1, 0])])
        panel = PlacementPanel(server, self.session, infer)
        panel.run()
        panel.run()
        for label in panel.options()[1:] + panel.options()[:1]:
            panel.choice.value = label
            panel.switch()
        np.testing.assert_array_equal(panel.cloud.points, self.points)
        panel.confirm()
        infer.assert_called_once()
        self.assertEqual(json.loads((self.root / "selection.json").read_text())["candidate_id"], "P00")
        # Confirm must use the current GUI value even if its update callback is still queued.
        panel.choice.value = panel.options()[1]
        panel.confirm()
        self.assertEqual(json.loads((self.root / "selection.json").read_text())["candidate_id"], "P01")
        infer.assert_called_once()

    def test_cli_branches_before_grasp_and_robot_setup(self):
        with (
            patch.object(dual_runner, "configure_model", side_effect=AssertionError("grasp setup")),
            patch.object(dual_runner, "calibrated_snapshot", side_effect=AssertionError("robot setup")),
            patch.object(dual_runner, "run", side_effect=AssertionError("execution path")),
            patch("ada_grasp_ctrl.placement.workflow.run_placement", return_value=0) as run,
        ):
            self.assertEqual(dual_runner.main(["--placement-only", "--output-root", str(self.root / "new")]), 0)
            run.assert_called_once()

    def test_live_placement_reuses_selection_without_asset_or_grasp_inference(self):
        from ada_grasp_ctrl.placement.workflow import run_placement

        args = argparse.Namespace(
            object_root=None,
            uop_root="../uop-net",
            uop_python=None,
            environment="mock",
            ros_domain_id=88,
            roi=[0, 0, 10, 10],
            cluster=0,
        )
        with (
            patch("ada_grasp_ctrl.placement.workflow.socket.socket") as socket,
            patch("ada_grasp_ctrl.pipeline.runner.Processes") as processes,
            patch("ada_grasp_ctrl.pipeline.runner.perceive", return_value=self.root) as perceive,
            patch("ada_grasp_ctrl.placement.workflow.prepare_session", return_value=self.session),
            patch("ada_grasp_ctrl.placement.ui.serve") as serve,
        ):
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            self.assertEqual(run_placement(args, {"environment": "mock"}, self.root / "live"), 0)
            self.assertEqual(perceive.call_args.kwargs, dict(infer=False, build_asset=False))
            processes.return_value.run.assert_not_called()
            serve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
