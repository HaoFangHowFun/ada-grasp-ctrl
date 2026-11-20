"""Deterministic mode-priority, fallback and unchanged feasibility-policy fixtures."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from ada_grasp_ctrl.grasp_selection.contracts import candidate_from_schema1, validate_grasp_plan
from ada_grasp_ctrl.grasp_selection.hierarchical import select_hierarchical
from ada_grasp_ctrl.grasp_selection.multi_arm import select_candidates
from test_phase2_pipeline import FakeRobot, FakeSim, candidate


def row(mode, side="right", distance=0.1, name="c0"):
    return candidate_from_schema1(candidate(side, distance, name), mode=mode)


def run(groups, scores, **kwargs):
    return select_hierarchical(
        kwargs.pop("robot", FakeRobot()),
        kwargs.pop("sim", FakeSim()),
        dict(schema_version=1, candidates_by_mode=groups),
        dict(schema_version=1, kind="scores", scores=scores),
        kwargs.pop("available_arms", ["left", "right"]),
        kwargs.pop("q_current", np.zeros(56)),
        **kwargs,
    )


class HierarchicalSelectionTests(unittest.TestCase):
    def test_higher_prior_wins_despite_lower_effort_elsewhere(self):
        groups = {"left_full": [row("left_full", "left", 2)], "right_full": [row("right_full", distance=0.1)]}
        before = deepcopy(groups)
        with patch.object(FakeRobot, "solve_arm_ik", autospec=True, side_effect=FakeRobot.solve_arm_ik) as ik:
            result = run(groups, {"left_full": 0.9, "right_full": 0.1})
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected_mode"], "left_full")
        self.assertEqual(result["effort"], 24)
        self.assertTrue(all(call.args[2] == "left" for call in ik.call_args_list))
        self.assertEqual(result["mode_trace"][1]["reasons"], ["not_evaluated_after_selection"])
        self.assertEqual(result["mode_trace"][1]["evaluated_candidate_count"], 0)
        self.assertEqual(result["plan"]["active_arms"], ["left"])
        self.assertEqual(result["plan"]["available_arms"], ["left", "right"])
        self.assertEqual(validate_grasp_plan(result["plan"]), result["plan"])
        self.assertEqual(groups, before)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_infeasible_high_mode_falls_back_with_original_diagnostics(self):
        result = run(
            {"left_full": [row("left_full", "left", 6)], "right_full": [row("right_full", distance=0.2)]},
            {"left_full": 5, "right_full": 2},
        )
        self.assertEqual(result["selected_mode"], "right_full")
        trace = result["mode_trace"]
        self.assertEqual([t["rank"] for t in trace], [1, 2])
        self.assertEqual([t["score"] for t in trace], [5, 2])
        self.assertEqual(trace[0]["status"], "no_feasible_candidate")
        self.assertEqual(trace[0]["candidates"][0]["reasons"], ["no_feasible_ik_branch"])
        self.assertTrue(all(a["reasons"] == ["grasp:ik_rejected"] for a in trace[0]["candidates"][0]["attempts"]))
        self.assertAlmostEqual(result["effort"], 6 * 0.2**2)

    def test_mode_candidate_and_seed_ties_do_not_depend_on_input_order(self):
        groups = {
            "right_full": [row("right_full", distance=0.1)],
            "left_full": [row("left_full", "left", 0.2, "z"), row("left_full", "left", 0.2, "a")],
        }
        first = run(groups, {"right_full": 1, "left_full": 1})
        reverse = {m: list(reversed(rows)) for m, rows in reversed(list(groups.items()))}
        second = run(reverse, {"left_full": 1, "right_full": 1})
        self.assertEqual(first, second)
        self.assertEqual(first["selected"]["candidate_id"], "left:a")
        self.assertEqual(first["selected"]["seed_index"], 0)
        self.assertEqual(first["mode_trace"][0]["feasible_candidate_count"], 2)

    def test_intra_mode_cost_uses_current_state_and_per_arm_weights(self):
        current = np.zeros(56)
        current[:6], current[28:34] = 0.9, 0.5
        rows = [row("choice", "left", 1.0), row("choice", "right", 0.7)]
        result = run(
            {"choice": rows},
            {"choice": 10},
            supported_modes=["choice"],
            q_current=current,
            weights_by_arm={"left": [10] * 6, "right": [1] * 6},
        )
        self.assertEqual(result["selected_side"], "right")
        self.assertAlmostEqual(result["effort"], 6 * 0.2**2)
        self.assertEqual(result["selected"]["delta_rad"], [0.7 - 0.5] * 6)

    def test_single_mode_matches_legacy_for_both_sides(self):
        for side in ("left", "right"):
            mode = side + "_full"
            rows = [row(mode, side, 0.7, "z"), row(mode, side, 0.2, "a")]
            expected = select_candidates(FakeRobot(), FakeSim(), rows, ["left", "right"], np.zeros(56))
            result = run({mode: rows}, {mode: 1})
            self.assertEqual(result["selected"], expected["selected"])
            self.assertEqual(
                result["mode_trace"][0]["candidates"], sorted(expected["evaluations"], key=lambda x: x["candidate_id"])
            )

    def test_unsupported_disabled_unranked_and_dual_candidates_are_explicit(self):
        dual = row("both_full", "left", name="both")
        dual["required_arms"].append("right")
        dual["targets"].update(row("both_full")["targets"])
        groups = {
            "both_full": [dual],
            "unknown": [row("unknown", name="unknown")],
            "left_full": [row("left_full", "left")],
            "right_full": [row("right_full")],
            "unranked": [row("unranked", name="unranked")],
        }
        with patch.object(FakeRobot, "solve_arm_ik", side_effect=AssertionError("unexpected IK")):
            result = run(
                groups,
                {"both_full": 10, "unknown": 9, "left_full": 8, "right_full": 7},
                supported_modes=["both_full", "left_full", "right_full", "unranked"],
                enabled_modes=["both_full", "left_full", "unranked"],
                available_arms=["right"],
            )
        trace = {t["mode"]: t for t in result["mode_trace"]}
        self.assertEqual(trace["both_full"]["candidates"][0]["reasons"], ["unsupported_multi_arm_candidate"])
        self.assertEqual(trace["left_full"]["candidates"][0]["reasons"], ["disabled_arm"])
        self.assertEqual(trace["unknown"]["reasons"], ["unsupported_mode"])
        self.assertEqual(trace["right_full"]["reasons"], ["disabled_mode"])
        self.assertEqual(trace["unranked"]["reasons"], ["missing_priority"])
        self.assertEqual(result["status"], "NO_FEASIBLE_GRASP")
        self.assertIsNone(result["plan"])
        self.assertIsNone(result["selected"])
        self.assertNotIn("targets", result)

    def test_empty_and_all_infeasible_emit_no_plan(self):
        for groups, scores in (
            ({}, {}),
            ({"right_full": []}, {"right_full": 1}),
            ({"right_full": [row("right_full", distance=6)]}, {"right_full": 1}),
        ):
            with self.subTest(groups=groups):
                result = run(groups, scores)
                self.assertEqual(result["status"], "NO_FEASIBLE_GRASP")
                self.assertIsNone(result["plan"])
                self.assertIsNone(result["effort"])
                self.assertFalse(result["execution_ready"])
                self.assertFalse(result["hardware_ready"])

    def test_bad_candidates_and_original_hand_limits_allow_fallback(self):
        bad = row("left_full", "left", name="bad")
        bad["joint_unit"] = "deg"
        limits = row("left_full", "left", name="limits")
        joints = limits["targets"]["left"]["grasp"]["joint_positions_rad"]
        joints[next(iter(joints))] = 11
        mismatch = row("right_full", "left", name="mismatch")
        result = run(
            {"left_full": [bad, limits, mismatch], "right_full": [row("right_full")]},
            {"left_full": 2, "right_full": 1},
        )
        self.assertEqual(result["selected_mode"], "right_full")
        diagnostics = result["mode_trace"][0]["candidates"]
        self.assertEqual(
            [d["reasons"] for d in diagnostics], [["invalid_candidate"], ["hand_joint_limits"], ["invalid_candidate"]]
        )

    def test_collision_reasons_and_thresholds_are_backend_owned(self):
        class Collision(FakeSim):
            def check_static_contacts(self, q):
                return [
                    dict(bodies=["left_J2_link", "table"], category="robot_table", sides=["left"], penetration_m=0.001)
                ]

        rows = [row("left_full", "left")]
        legacy = select_candidates(FakeRobot(), Collision(), rows, ["left"], np.zeros(56))
        result = run({"left_full": rows}, {"left_full": 1}, sim=Collision(), available_arms=["left"])
        self.assertEqual(result["mode_trace"][0]["candidates"], legacy["evaluations"])
        self.assertEqual(result["status"], "NO_FEASIBLE_GRASP")

    def test_explicit_ranking_preserves_user_order_without_score_conversion(self):
        result = select_hierarchical(
            FakeRobot(),
            FakeSim(),
            dict(
                schema_version=1,
                candidates_by_mode={
                    "left_full": [row("left_full", "left")],
                    "right_full": [row("right_full", distance=2)],
                },
            ),
            dict(schema_version=1, kind="ranking", ranking=["right_full", "left_full"]),
            ["left", "right"],
            np.zeros(56),
        )
        self.assertEqual(result["selected_mode"], "right_full")
        self.assertIsNone(result["mode_trace"][0]["score"])

    def test_invalid_priority_run_input_and_duplicate_ids_fail_before_ik(self):
        batch = dict(schema_version=1, candidates_by_mode={"right_full": [row("right_full")]})
        priors = [
            {"schema_version": 1, "kind": "counts", "K_t": {"right_full": 8}},
            {"schema_version": True, "kind": "scores", "scores": {}},
            {"schema_version": 1, "kind": "ranking", "ranking": ["right_full", "right_full"]},
            *[
                dict(schema_version=1, kind="scores", scores={"right_full": v})
                for v in (True, float("nan"), float("inf"), "1")
            ],
        ]
        with patch.object(FakeRobot, "solve_arm_ik", side_effect=AssertionError("unexpected IK")):
            for prior in priors:
                with self.subTest(prior=prior), self.assertRaises(ValueError):
                    select_hierarchical(FakeRobot(), FakeSim(), batch, prior, ["right"], np.zeros(56))
            for overrides in (
                {"enabled_modes": ["unknown"]},
                {"available_arms": []},
                {"available_arms": ["right", "right"]},
                {"max_nfev": True},
                {"weights_by_arm": {"right": [1] * 6}},
                {"q_current": [0] * 55},
            ):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    run(batch["candidates_by_mode"], {"right_full": 1}, **overrides)
            duplicate = deepcopy(batch["candidates_by_mode"])
            duplicate["left_full"] = [row("left_full")]
            with self.assertRaisesRegex(ValueError, "globally unique"):
                run(duplicate, {"right_full": 1, "left_full": 2})


if __name__ == "__main__":
    unittest.main()
