"""Dual-live opt-in selection, existing export/auth gates and model provenance, without hardware."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import validate_packet
from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.grasp_selection.multi_arm import select_candidates
from ada_grasp_ctrl.pipeline import dual_runner
from ada_grasp_ctrl.pipeline.dual_selection import (
    canonical_batch,
    legacy_selection,
    select_live_hierarchical,
    selector_settings,
)
from dual_selector_fixtures import LiveFixture
from test_phase2_pipeline import FakeRobot, FakeSim


class DualHierarchicalTests(unittest.TestCase):
    def test_default_and_explicit_legacy_retain_selection_export_and_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = []
            for index, options in enumerate(([], ["--selector", "legacy"])):
                fixture = LiveFixture(Path(directory) / str(index))
                # A YAML value alone must never activate hierarchical selection.
                fixture.config["selector"] = "hierarchical"
                fixture.config["hierarchical_selector"] = {"invalid": True}
                with patch.object(
                    dual_runner, "select_live_hierarchical", side_effect=AssertionError("unexpected selector")
                ):
                    self.assertEqual(fixture.invoke(options), 0)
                rows = fixture.artifact("candidates.json")["candidates"]
                expected = select_candidates(fixture.robot, FakeSim(), rows, ["left", "right"], fixture.q)
                selection = fixture.artifact("selection.json")
                selection.pop("geometry_note")
                self.assertEqual(selection, expected)
                self.assertFalse((fixture.root / "run/grasp_plan.json").exists())
                packet = load_packet(fixture.root / "run/export/targets.json")
                runs.append((fixture, packet))
            a, b = runs
            # Path-bound hashes necessarily differ between fresh output roots.
            self.assertEqual(
                {k: v for k, v in a[1].items() if k != "source"}, {k: v for k, v in b[1].items() if k != "source"}
            )
            self.assertEqual(
                json.dumps(a[0].calls).replace(str(a[0].root), "ROOT"),
                json.dumps(b[0].calls).replace(str(b[0].root), "ROOT"),
            )

    def test_rank_order_wins_over_cross_mode_effort_and_exports_both_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            for side in ("right", "left"):
                fixture = LiveFixture(Path(directory) / side)
                other = "left" if side == "right" else "right"
                fixture.config["hierarchical_selector"]["prior"]["ranking"] = [side + "_full", other + "_full"]
                self.assertEqual(fixture.invoke(["--selector", "hierarchical"]), 0)
                audit = fixture.artifact("hierarchical_selection.json")
                self.assertEqual(audit["selected_mode"], side + "_full")
                self.assertEqual(audit["mode_trace"][1]["reasons"], ["not_evaluated_after_selection"])
                packet = load_packet(fixture.root / "run/export/targets.json")
                self.assertEqual(packet["side"], side)
                self.assertEqual(packet["sample_id"], side + ":c0000")
                self.assertEqual(packet["kind"], "ada_side_joint_targets")
                self.assertEqual(packet["targets"][2]["positions"][:6], [fixture.distances[side]] * 6)
                self.assertEqual(packet["reviewed_plan"]["passive_positions_rad"], [0.0] * 28)
                self.assertEqual(audit["effort"], 6 * fixture.distances[side] ** 2)
                self.assertFalse(audit["plan"]["execution_ready"])
                self.assertFalse(audit["plan"]["hardware_ready"])
                with self.assertRaises(ValueError):
                    validate_packet(audit["plan"])

    def test_fallback_all_infeasible_and_empty_emit_no_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, distances in enumerate(
                (dict(right=6, left=0.2), dict(right=6, left=6), dict(right=[], left=[]))
            ):
                fixture = LiveFixture(Path(directory) / str(index), distances=distances)
                expected = 0 if index == 0 else 1
                self.assertEqual(fixture.invoke(["--selector", "hierarchical"]), expected)
                audit = fixture.artifact("hierarchical_selection.json")
                if index == 0:
                    self.assertEqual(audit["selected_mode"], "left_full")
                    self.assertEqual(audit["mode_trace"][0]["reasons"], ["no_feasible_candidate"])
                else:
                    self.assertEqual(audit["status"], "NO_FEASIBLE_GRASP")
                    self.assertIsNone(audit["plan"])
                    self.assertFalse((fixture.root / "run/export").exists())
                    self.assertFalse((fixture.root / "run/grasp_plan.json").exists())
                self.assertTrue(
                    all(
                        label in ("initial_state", "state", "inference_left", "inference_right")
                        for label, _ in fixture.calls
                    )
                )

    def test_arms_restrict_actual_inference_and_ik(self):
        with tempfile.TemporaryDirectory() as directory:
            for side in ("left", "right"):
                fixture = LiveFixture(Path(directory) / side)
                with patch.object(fixture.robot, "solve_arm_ik", wraps=fixture.robot.solve_arm_ik) as ik:
                    self.assertEqual(fixture.invoke(["--selector", "hierarchical", "--arms", side]), 0)
                self.assertEqual(
                    [label for label, _ in fixture.calls if label.startswith("inference_")], ["inference_" + side]
                )
                self.assertTrue(all(call.args[1] == side for call in ik.call_args_list))
                batch = fixture.artifact("canonical_candidates.json")
                self.assertEqual(list(batch["candidates_by_mode"]), [side + "_full"])
                self.assertEqual(fixture.artifact("grasp_plan.json")["active_arms"], [side])
                self.assertEqual(fixture.artifact("grasp_plan.json")["available_arms"], [side])

    def test_provenance_and_original_rows_survive_adaptation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = LiveFixture(Path(directory) / "data")
            self.assertEqual(fixture.invoke(["--selector", "hierarchical"]), 0)
            rows = fixture.artifact("candidates.json")["candidates"]
            before = deepcopy(rows)
            batch = canonical_batch(rows, ["left", "right"])
            self.assertEqual(rows, before)
            for row in rows:
                side = row["required_arms"][0]
                adapted = batch["candidates_by_mode"][side + "_full"][0]
                self.assertEqual(adapted["candidate_id"], row["candidate_id"])
                self.assertEqual(adapted["targets"], row["targets"])
                for key, value in row["provenance"].items():
                    self.assertEqual(adapted["provenance"][key], value)
                self.assertEqual(adapted["provenance"]["source_model_mode"], "right_full")
                self.assertEqual(
                    adapted["provenance"]["generation_method"],
                    "mirrored_right_full" if side == "left" else "right_full",
                )
                self.assertNotIn("source_model_mode", row["provenance"])
            self.assertEqual(fixture.artifact("selection.json")["selected"]["candidate"], rows[1])

    def test_malformed_config_fails_before_processes_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = LiveFixture(Path(directory) / "base").config
            invalid = [None, {}, {"prior": {}}, {**base["hierarchical_selector"], "unexpected": True}]
            for labels in ([], ["right_full", "right_full"], ["right_three"], ["both_full"], ["unknown"], [None]):
                value = deepcopy(base["hierarchical_selector"])
                value["prior"]["ranking"] = labels
                invalid.append(value)
            for key, value in (("schema_version", True), ("kind", "scores")):
                setting = deepcopy(base["hierarchical_selector"])
                setting["prior"][key] = value
                invalid.append(setting)
            for index, value in enumerate(invalid):
                config = {**base, "hierarchical_selector": value}
                root = Path(directory) / f"bad{index}"
                with (
                    patch.object(dual_runner, "configuration", return_value=config),
                    patch.object(dual_runner, "Processes") as processes,
                ):
                    self.assertEqual(
                        dual_runner.main(
                            ["--selector", "hierarchical", "--output-root", str(root), "--ros-domain-id", "88"]
                        ),
                        2,
                    )
                    processes.assert_not_called()
                self.assertFalse(root.exists())
            with self.assertRaises(ValueError):
                selector_settings(
                    dict(
                        prior=dict(schema_version=1, kind="ranking", ranking=["right_full"]),
                        supported_modes=["left_full"],
                    )
                )

    def test_identity_conflicts_and_readiness_cannot_reach_export(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = LiveFixture(Path(directory) / "data")
            self.assertEqual(fixture.invoke(["--selector", "hierarchical"]), 0)
            rows = fixture.artifact("candidates.json")["candidates"]
            batch = fixture.artifact("canonical_candidates.json")
            audit = fixture.artifact("hierarchical_selection.json")
            for fault in ("side", "id", "mode", "arms", "plan", "ready"):
                bad = deepcopy(audit)
                if fault == "side":
                    bad["selected"]["selected_side"] = "left"
                elif fault == "id":
                    bad["selected"]["candidate_id"] = "right:missing"
                elif fault == "mode":
                    bad["selected_mode"] = "both_full"
                elif fault == "arms":
                    bad["selected"]["required_arms"] = ["left", "right"]
                elif fault == "plan":
                    bad["plan"]["candidate_id"] = "right:other"
                else:
                    bad["plan"]["execution_ready"] = True
                with self.subTest(fault=fault), self.assertRaises(ValueError):
                    legacy_selection(bad, batch, rows, ["left", "right"])
            for fault in ("duplicate", "generator", "mode", "prefix", "reflection"):
                bad = deepcopy(rows)
                if fault == "duplicate":
                    bad.append(deepcopy(bad[0]))
                elif fault == "generator":
                    bad[0]["provenance"]["generator"] = "trained_left_full"
                elif fault == "mode":
                    bad[0]["mode"] = "both_full"
                elif fault == "prefix":
                    bad[0]["candidate_id"] = "right:wrong"
                else:
                    bad[0]["provenance"].pop("reflection_plane")
                with self.subTest(fault=fault), self.assertRaises(ValueError):
                    canonical_batch(bad, ["left", "right"])

    def test_ties_and_skip_trace_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = LiveFixture(Path(directory) / "data", distances=dict(left=0.1, right=[0.2, 0.2]))
            self.assertEqual(fixture.invoke(["--selector", "hierarchical"]), 0)
            rows = fixture.artifact("candidates.json")["candidates"]
            results = [
                select_live_hierarchical(
                    FakeRobot(),
                    FakeSim(),
                    order,
                    ["left", "right"],
                    np.zeros(56),
                    fixture.config["hierarchical_selector"],
                )[2]
                for order in (rows, rows[::-1])
            ]
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0]["selected"]["candidate_id"], "right:c0000")
            self.assertEqual(results[0]["selected"]["seed_index"], 0)

    def test_canonical_selection_still_requires_existing_enter_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = LiveFixture(Path(directory) / "gate")
            self.assertEqual(fixture.invoke(["--selector", "hierarchical"], export_only=False), 0)
            self.assertEqual(fixture.artifact("run_report.json")["status"], "canceled_before_execution")
            self.assertEqual(fixture.calls[-1][0], "viewer")
            self.assertFalse(
                any(label in ("receiver", "sender", "receiver_probe", "before_execute") for label, _ in fixture.calls)
            )

    def test_hash_bound_selection_rejects_tampered_hierarchical_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = LiveFixture(Path(directory) / "tamper")
            self.assertEqual(fixture.invoke(["--selector", "hierarchical"]), 0)
            path = fixture.root / "run/selection.json"
            selection = json.loads(path.read_text())
            selection["hierarchical_report"]["selected_mode"] = "left_full"
            path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "source changed"):
                load_packet(fixture.root / "run/export/targets.json")


if __name__ == "__main__":
    unittest.main()
