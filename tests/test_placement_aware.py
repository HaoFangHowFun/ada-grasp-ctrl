"""Placement rigid transforms, support geometry, feasibility ordering and CLI boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# Load canonical robotics dependencies before scipy numerical work (local Pinocchio ABI).
from ada_grasp_ctrl.pipeline import dual_runner
from ada_grasp_ctrl.pipeline.simulation_plan import adjust_squeeze_candidates
from ada_grasp_ctrl.pipeline.simulation_scene import table_frame
from ada_grasp_ctrl.placement.target import TargetGeometry, transform_points, digest
from ada_grasp_ctrl.placement.feasibility import PlacementGate
from ada_grasp_ctrl.placement.session import PlacementSession
from ada_grasp_ctrl.grasp_selection.multi_arm import select_candidates
from ada_grasp_ctrl.robot_environment.model import side_slice
from test_phase2_pipeline import FakeRobot, FakeSim, candidate
from test_hierarchical_selection import row, run
from test_placement_selection import candidate as orientation
from ada_grasp_ctrl.pipeline.dual_selection import select_live_hierarchical
from ada_grasp_ctrl.pipeline.grasp_models import SOURCE_MODES
from ada_grasp_ctrl.pipeline.placement_preview import PlacementPreview
from ada_grasp_ctrl.placement.session import write_json

import numpy as np
from scipy.spatial.transform import Rotation


class TargetTests(unittest.TestCase):
    def fixture(self, normal=(0, 0, 1)):
        points = np.array([[0.3, -0.4, 0.1], [0.34, -0.38, 0.2], [0.28, -0.42, 0.14]])
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        session = PlacementSession(Path(directory.name), points, "snapshot", {"snapshot_sha256": "hash"})
        geometry = TargetGeometry.__new__(TargetGeometry)
        geometry.center = session.center.copy()
        geometry.support = points.copy()
        geometry.footprint = points.copy()
        n = np.array(normal, dtype=float)
        n /= np.linalg.norm(n)
        geometry.plane = np.r_[n, 0.1]
        geometry.table = dict(center_xy_m=[0.3, -0.4], half_extents_xy_m=[0.6, 0.7])
        geometry.table_center, geometry.table_rotation, geometry.half_extents = table_frame(
            dict(normal_xyz=n, d=0.1), geometry.table
        )
        geometry.clearance = 0.0
        geometry.source = dict(kind="analytical_test_fixture")
        return session, geometry

    def test_p00_translation_and_tilted_support_no_accumulation(self):
        for normal in ((0, 0, 1), (0.2, -0.1, 1)):
            session, geometry = self.fixture(normal)
            original = session.points.copy()
            a = geometry.resolve(session, [0.4, -0.3], 1)
            transformed = transform_points(original, a)
            np.testing.assert_allclose(transformed - transformed.mean(0), original - original.mean(0), atol=1e-14)
            self.assertAlmostEqual(np.min(transformed @ geometry.plane[:3] + geometry.plane[3]), 0, places=12)
            marker = geometry.marker([0.4, -0.3])
            self.assertAlmostEqual(marker @ geometry.plane[:3] + geometry.plane[3], 0, places=12)
            self.assertNotEqual(marker[2], a["target_centroid_world_m"][2])
            geometry.resolve(session, [0.5, -0.2], 2)
            self.assertEqual(a, geometry.resolve(session, [0.4, -0.3], 1))
            np.testing.assert_array_equal(session.select("P00"), original)
            self.assertEqual(json.loads(json.dumps(a)), a)
            self.assertEqual(a["target_sha256"], digest({k: v for k, v in a.items() if k != "target_sha256"}))

    def test_nonzero_pivot_rotation_and_held_object_reference_both_arms(self):
        session, geometry = self.fixture((0.2, 0.1, 1))
        session.infer_once(lambda: [orientation(1, [1, 0, 0])])
        session.select("P01")
        target = geometry.resolve(session, [0.4, -0.3], 3)
        R = np.asarray(session.selected.relative_rotation)
        expected = (session.points - session.center) @ R.T + target["target_centroid_world_m"]
        np.testing.assert_allclose(transform_points(session.points, target), expected)
        for side in ("left", "right"):
            raw = candidate(side, 0.2)
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.3, 0.4]).as_matrix()
            pose[:3, 3] = session.center + [0.02, -0.03, 0.04]
            raw["targets"][side]["squeeze"]["T_world_hand"] = pose.tolist()
            held = np.linspace(0, 0.05, 56)
            result = PlacementGate(FakeRobot(), FakeSim(), target)(raw, side, held)
            destination = np.asarray(result["T_world_hand_place"])
            source_object = np.eye(4)
            source_object[:3, 3] = session.center
            target_object = np.asarray(target["Delta_T"]) @ source_object
            np.testing.assert_allclose(
                np.linalg.inv(destination) @ target_object, np.linalg.inv(pose) @ source_object, atol=1e-14
            )
            mask = np.ones(56, dtype=bool)
            mask[side_slice(side, True)] = False
            np.testing.assert_array_equal(np.asarray(result["q56"])[mask], held[mask])

    def test_off_table_goal_rejected_not_silently_moved(self):
        session, geometry = self.fixture()
        with self.assertRaisesRegex(ValueError, "footprint"):
            geometry.resolve(session, [0.9, -0.4], 1)
        marker = geometry.marker([99, -99])
        np.testing.assert_allclose(
            np.abs(((marker - geometry.table_center) @ geometry.table_rotation)[:2]), geometry.half_extents
        )


class GateTests(unittest.TestCase):
    def test_collision_filter_selects_clear_branch_and_skips_failed_ik(self):
        target = dict(Delta_T=np.eye(4).tolist(), target_revision=1, target_sha256="test")
        for side in ("left", "right"):
            robot, sim = FakeRobot(), FakeSim()
            held = np.zeros(56)
            arm = side_slice(side, True)
            collision = dict(
                category="robot_table", sides=[side], bodies=["live_table", side + "_J4_link"], penetration_m=0.1
            )

            def solve(q, side, pose):
                q = q.copy()
                q[arm] = 1 if np.array_equal(q[arm], held[arm]) else 2
                return q, dict(accepted=True)

            with (
                patch.object(robot, "solve_arm_ik", side_effect=solve),
                patch.object(sim, "check_static_contacts", side_effect=lambda q: [collision] if q[arm][0] == 1 else []),
            ):
                result = PlacementGate(robot, sim, target)(candidate(side, 0.1), side, held)
            self.assertTrue(result["accepted"])
            self.assertTrue(result["collision_checked"])
            np.testing.assert_array_equal(np.asarray(result["q56"])[arm], [2] * 6)
            self.assertEqual(result["attempts"][0]["reason"], "place_endpoint_collision")
            self.assertEqual(result["attempts"][0]["forbidden_contacts"], [collision])
            with patch.object(sim, "check_static_contacts", return_value=[collision]):
                result = PlacementGate(robot, sim, target)(candidate(side, 0.1), side, held)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "place_endpoint_collision")
            self.assertNotIn("q56", result)
            with patch.object(sim, "check_static_contacts", side_effect=AssertionError("IK failed before collision")):
                result = PlacementGate(robot, sim, target)(candidate(side, 6), side, held)
            self.assertFalse(result["accepted"])
            self.assertFalse(result["collision_checked"])
            self.assertEqual(result["reason"], "place_endpoint_ik_rejected")

    def test_collision_rejects_low_effort_candidate_and_allows_mode_fallback(self):
        robot, sim = FakeRobot(), FakeSim()
        delta = np.eye(4)
        delta[0, 3] = 2
        gate = PlacementGate(robot, sim, dict(Delta_T=delta.tolist(), target_revision=1, target_sha256="test"))

        def contacts(q):
            return [
                dict(category="robot_table", sides=[side], bodies=["live_table", side + "_J4_link"], penetration_m=0.1)
                for side in ("left", "right")
                if 2.05 < q[side_slice(side, True)][0] < 2.15
            ]

        with patch.object(sim, "check_static_contacts", side_effect=contacts):
            selected = select_candidates(
                robot,
                sim,
                [candidate("right", 0.1, "bad"), candidate("right", 0.2, "good")],
                ["right"],
                np.zeros(56),
                endpoint_gate=gate,
            )
            self.assertEqual(selected["selected"]["candidate_id"], "right:good")
            self.assertAlmostEqual(selected["selected"]["effort"], 6 * 0.2**2)
            self.assertEqual(selected["evaluations"][0]["attempts"][0]["reasons"], ["place_endpoint_collision"])
            groups = {"left_full": [row("left_full", "left", 0.1)], "right_full": [row("right_full", "right", 0.2)]}
            selected = run(groups, dict(left_full=2, right_full=1), robot=robot, sim=sim, endpoint_gate=gate)
            self.assertEqual(selected["selected_mode"], "right_full")
            self.assertEqual(selected["mode_trace"][0]["feasible_candidate_count"], 0)
            selected = run(
                {"left_full": groups["left_full"]}, dict(left_full=1), robot=robot, sim=sim, endpoint_gate=gate
            )
            self.assertEqual(selected["status"], "NO_FEASIBLE_GRASP")

    def test_target_changes_reuse_raw_inference_but_recompute_ik(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "object").mkdir()
            np.savez(root / "object/object.npz", points_world_m=np.ones((3, 3)))
            write_json(root / "table.json", {})
            robot = FakeRobot()
            config = dict(
                robot_snapshot=str(root),
                table_plane=str(root / "table.json"),
                enabled_arms=["right"],
                grasp_model="right_full",
            )
            raw = candidate("right", 0.2)
            original = deepcopy(raw)

            def infer(*args):
                write_json(root / "candidates.json", dict(candidates=[raw]))
                return [raw]

            with (
                patch("ada_grasp_ctrl.pipeline.placement_preview.live_scene", return_value=(robot, FakeSim())),
                patch("ada_grasp_ctrl.pipeline.placement_preview.infer_live_sides", side_effect=infer) as inference,
            ):
                engine = PlacementPreview(config, "legacy", root, dict(full_positions_rad=np.zeros(56).tolist()), None)
                for revision, x in enumerate((8, 9), 1):
                    delta = np.eye(4)
                    delta[0, 3] = x
                    target = dict(Delta_T=delta.tolist(), target_revision=revision, target_sha256=str(revision))
                    output = root / str(revision)
                    output.mkdir()
                    self.assertIsNone(engine.prepare(target, output, lambda: False))
                    report = json.loads((output / "placement_selection_report.json").read_text())
                    attempt = report["evaluations"][0]["attempts"][0]
                    self.assertEqual(attempt["placement"]["target_revision"], revision)
                    self.assertAlmostEqual(attempt["placement"]["T_world_hand_place"][0][3], x + 0.2)
                    self.assertEqual(report["counts"]["grasp_feasible"], 1)
                    self.assertEqual(report["counts"]["place_feasible"], 0)
                self.assertEqual(inference.call_count, 1)
                self.assertEqual(raw, original)

    def test_native_auto_left_right_exclude_paired_modes_before_ik(self):
        for available in (["left", "right"], ["left"], ["right"]):
            rows = []
            for side in available:
                item = candidate(side, 0.2, "right_full:c0")
                item.update(mode="right_full", execution_mode=side + "_full")
                item["provenance"].update(
                    generator="mirrored_right_full" if side == "left" else "right_full",
                    source_model_mode="right_full",
                    checkpoint={"path": "test"},
                    metadata={"path": "test"},
                    reflection_plane={"normal_world": [0, 1, 0]},
                )
                rows.append(item)
            paired = candidate("left", 0.1, "pair")
            paired.update(
                candidate_id="both_full:pair",
                mode="both_full",
                execution_mode="both_full",
                required_arms=["left", "right"],
            )
            paired["targets"].update(candidate("right", 0.1)["targets"])
            paired["provenance"].update(
                generator="both_full",
                source_model_mode="both_full",
                checkpoint={"path": "test"},
                metadata={"path": "test"},
            )
            rows.append(paired)
            with patch(
                "ada_grasp_ctrl.grasp_selection.hierarchical.select_paired_candidates",
                side_effect=AssertionError("paired IK"),
            ):
                selection, _, audit = select_live_hierarchical(
                    FakeRobot(),
                    FakeSim(),
                    rows,
                    available,
                    np.zeros(56),
                    {},
                    native_diagnostics=dict(
                        availability_scores=dict.fromkeys(SOURCE_MODES, 1), ranked_available_modes=list(SOURCE_MODES)
                    ),
                    single_only=True,
                    endpoint_gate=lambda *args: dict(accepted=True),
                )
            self.assertIn(selection["selected"]["selected_side"], available)
            paired_trace = next(entry for entry in audit["mode_trace"] if entry["mode"] == "both_full")
            self.assertEqual(paired_trace["evaluated_candidate_count"], 0)
            self.assertIn("disabled_mode", paired_trace["reasons"])

    def test_grasp_failure_never_calls_gate_and_place_failure_cannot_win(self):
        calls = []

        def gate(row, side, held):
            calls.append(row["candidate_id"])
            return dict(accepted=row["candidate_id"].endswith("pass"))

        rows = [
            candidate("right", 6, "grasp_fail"),
            candidate("right", 0.1, "place_fail"),
            candidate("right", 0.2, "pass"),
        ]
        result = select_candidates(FakeRobot(), FakeSim(), rows, ["right"], np.zeros(56), endpoint_gate=gate)
        self.assertNotIn("right:grasp_fail", calls)
        self.assertEqual(result["selected"]["candidate_id"], "right:pass")
        self.assertAlmostEqual(result["selected"]["effort"], 6 * 0.2**2)
        self.assertTrue(all(a["reasons"] == ["grasp:ik_rejected"] for a in result["evaluations"][0]["attempts"]))
        self.assertTrue(
            all(a["reasons"] == ["place_endpoint_ik_rejected"] for a in result["evaluations"][1]["attempts"])
        )

    def test_mode_fallback_and_disabled_pair(self):
        groups = {"left_full": [row("left_full", "left", 0.1)], "right_full": [row("right_full", "right", 0.3)]}
        result = run(
            groups,
            dict(left_full=2, right_full=1),
            endpoint_gate=lambda row, side, held: dict(accepted=side == "right"),
        )
        self.assertEqual(result["selected_mode"], "right_full")
        self.assertEqual(result["mode_trace"][0]["feasible_candidate_count"], 0)
        result = run(groups, dict(left_full=2, right_full=1), endpoint_gate=lambda *args: dict(accepted=False))
        self.assertEqual(result["status"], "NO_FEASIBLE_GRASP")

    def test_adjusted_squeeze_stage_used_and_raw_candidate_preserved(self):
        raw = candidate("left", 0.2)
        raw["targets"]["left"]["squeeze"]["T_world_hand"][0][3] = 0.3
        for name in raw["targets"]["left"]["squeeze"]["joint_positions_rad"]:
            raw["targets"]["left"]["squeeze"]["joint_positions_rad"][name] = 0.1
        before = deepcopy(raw)
        target = dict(Delta_T=np.eye(4).tolist(), target_revision=1, target_sha256="hash")
        result = select_candidates(
            FakeRobot(),
            FakeSim(),
            adjust_squeeze_candidates([raw], 2),
            ["left"],
            np.zeros(56),
            endpoint_gate=PlacementGate(FakeRobot(), FakeSim(), target),
        )
        place = result["selected"]["placement"]
        self.assertAlmostEqual(place["T_world_hand_source_held"][0][3], 0.4)
        np.testing.assert_allclose(place["q56"][6:28], 0.2)
        self.assertEqual(raw, before)

    def test_cli_rejects_unsupported_before_server_or_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in (
                ("--environment", "hardware"),
                ("--environment", "mock"),
                ("--environment", "simulation", "--grasp-mode", "both_only"),
            ):
                with patch(
                    "ada_grasp_ctrl.pipeline.simulation_ui.run_simulation",
                    side_effect=AssertionError("server launched"),
                ):
                    self.assertEqual(
                        dual_runner.main(["--placement-aware", "--output-root", directory + "/run", *options]), 2
                    )
            self.assertFalse((Path(directory) / "run").exists())


if __name__ == "__main__":
    unittest.main()
