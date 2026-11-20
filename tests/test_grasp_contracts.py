"""Canonical grasp data, legacy adapters and retained execution capability gates."""

from copy import deepcopy
import json
import unittest

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import validate_packet
from ada_grasp_ctrl.grasp_selection.contracts import (
    candidate_from_schema1,
    candidate_to_schema1,
    plan_from_candidate,
    plan_from_selection,
    validate_grasp_candidate,
    validate_grasp_plan,
)
from ada_grasp_ctrl.grasp_selection.multi_arm import from_right_record, select_candidates, validate_candidate
from ada_grasp_ctrl.pipeline.plan import validate_plan
from test_phase2_pipeline import FakeRobot, FakeSim, candidate


def canonical(arms=("right",), mode="right_full"):
    row = candidate(arms[0], 0.12)
    row["required_arms"] = list(arms)
    row["targets"] = {side: candidate(side, 0.12)["targets"][side] for side in arms}
    row["provenance"].update(source_record={"raw_index": 4, "likelihood": 0.25}, calibration="fixture")
    return candidate_from_schema1(row, mode=mode)


class CanonicalContractTests(unittest.TestCase):
    def test_round_trip_left_right_both_and_future_mode_labels(self):
        for arms in (("left",), ("right",), ("left", "right")):
            for mode in ("left_full", "right_full", "both_full", "future_mode_4", "future_mode_5"):
                with self.subTest(arms=arms, mode=mode):
                    row = canonical(arms, mode)
                    self.assertEqual(validate_grasp_candidate(json.loads(json.dumps(row, allow_nan=False))), row)
                    plan = plan_from_candidate(row, available_arms=("left", "right"))
                    self.assertEqual(validate_grasp_plan(json.loads(json.dumps(plan, allow_nan=False))), plan)
                    self.assertEqual(plan["active_arms"], list(arms))
                    self.assertEqual(plan["available_arms"], ["left", "right"])
                    self.assertEqual(plan["mode"], mode)
                    self.assertEqual(plan["candidate_id"], row["candidate_id"])
                    self.assertEqual(plan["provenance"], row["provenance"])
                    self.assertEqual(plan["targets"], row["targets"])
                    self.assertFalse(plan["execution_ready"])
                    self.assertFalse(plan["hardware_ready"])
                    self.assertNotIn("required_arms", plan)

    def test_copies_do_not_mutate_inputs_or_share_provenance(self):
        row = canonical()
        original = deepcopy(row)
        plan = plan_from_candidate(row)
        self.assertNotIn("available_arms", plan)
        plan["targets"]["right"]["grasp"]["T_world_hand"][0][3] = 42
        plan["provenance"]["source_record"]["raw_index"] = 99
        self.assertEqual(row, original)
        validated = validate_grasp_candidate(row)
        validated["required_arms"].append("left")
        self.assertEqual(row, original)

    def test_arm_sets_and_selected_targets_must_match(self):
        for field in ("active_arms", "available_arms"):
            for value in ([], ["left", "left"], ["unknown"], "right", None, [None], [[]]):
                with self.subTest(field=field, value=value):
                    plan = plan_from_candidate(canonical())
                    plan[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        validate_grasp_plan(plan)
        for value in ([], ["right", "right"], ["both"], "right", None, [[]]):
            row = canonical()
            row["required_arms"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "required_arms"):
                validate_grasp_candidate(row)
        with self.assertRaisesRegex(ValueError, "subset"):
            plan_from_candidate(canonical(), available_arms=["left"])
        plan = plan_from_candidate(canonical())
        plan["targets"]["left"] = candidate("left", 0.2)["targets"]["left"]
        with self.assertRaisesRegex(ValueError, "targets"):
            validate_grasp_plan(plan)
        plan.pop("targets")
        with self.assertRaisesRegex(ValueError, "targets"):
            validate_grasp_plan(plan)

    def test_bad_headers_and_metadata_fail_with_candidate_context(self):
        for field, value in (
            ("schema_version", 2),
            ("schema_version", True),
            ("schema_version", 1.0),
            ("frame_id", "camera"),
            ("unit", "mm"),
            ("joint_unit", "deg"),
            ("candidate_id", ""),
            ("candidate_id", "  "),
            ("mode", ""),
            ("mode", None),
            ("provenance", []),
            ("provenance", {"score": float("nan")}),
            ("provenance", {1: "not a string key"}),
            ("targets", {"left": {}}),
        ):
            row = canonical()
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "grasp\\["):
                validate_grasp_candidate(row)

    def test_malformed_stage_targets_fail(self):
        for field, value in (
            ("frame_id", "camera"),
            ("pose_frame", "right_palm"),
            ("pose_frame", "left_hand_flange"),
            ("unit", "mm"),
            ("joint_unit", "deg"),
            ("quaternion_order", "xyzw"),
            ("T_world_hand", None),
            ("T_world_hand", [[1] * 4] * 3),
            ("T_world_hand", [[True] * 4] * 4),
            ("T_world_hand", [["0"] * 4] * 4),
            ("T_world_hand", np.diag([-1, 1, 1, 1]).tolist()),
            ("T_world_hand", np.diag([2, 1, 1, 1]).tolist()),
            ("T_world_hand", np.diag([1, 1, 1, 2]).tolist()),
            ("T_world_hand", np.full((4, 4), float("inf")).tolist()),
            ("joint_positions_rad", {}),
            ("joint_positions_rad", []),
            ("position_m", [0, 0, 0]),
            ("quaternion_wxyz", [0, 0, 0, 0]),
            ("quaternion_wxyz", [0, 1, 0, 0]),
        ):
            row = canonical()
            row["targets"]["right"]["grasp"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_grasp_candidate(row)
        for value in (float("nan"), float("inf"), True, "0", None):
            row = canonical()
            joints = row["targets"]["right"]["grasp"]["joint_positions_rad"]
            joints[next(iter(joints))] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_grasp_candidate(row)
        for stages in ({}, {"grasp": {}}, {"pregrasp": {}, "grasp": {}, "squeeze": {}, "lift": {}}, None):
            row = canonical(("left", "right"), "both_full")
            row["targets"]["left"] = stages
            with self.subTest(stages=stages), self.assertRaisesRegex(ValueError, "targets.left"):
                validate_grasp_candidate(row)

    def test_consistent_xyz_wxyz_are_preserved_without_transforming(self):
        row = canonical()
        for target in row["targets"]["right"].values():
            target.update(position_m=[0.12, 0, 0], quaternion_wxyz=[-1, 0, 0, 0])
        self.assertEqual(validate_grasp_candidate(row), row)

    def test_plans_cannot_claim_readiness_or_be_used_as_execution_packets(self):
        for arms in (("left",), ("right",), ("left", "right")):
            plan = plan_from_candidate(canonical(arms))
            for field in ("execution_ready", "hardware_ready"):
                for value in (True, 0, None, "false"):
                    altered = {**plan, field: value}
                    with self.subTest(arms=arms, field=field, value=value), self.assertRaises(ValueError):
                        validate_grasp_plan(altered)
                altered = deepcopy(plan)
                altered.pop(field)
                with self.assertRaises(ValueError):
                    validate_grasp_plan(altered)
            with self.assertRaises(ValueError):
                validate_packet(plan)
            with self.assertRaises(ValueError):
                validate_plan(plan)


class LegacyAdapterTests(unittest.TestCase):
    def test_schema1_round_trip_preserves_id_and_original_mirrored_targets(self):
        raw = dict(candidate_id="c0004", stages=candidate("right", 0.2)["targets"]["right"])
        plane = dict(normal_world=[1, 0, 0], point_world_m=[0, 0, 0])
        for side in ("left", "right"):
            legacy = from_right_record(raw, {"checkpoint": "fixture"}, side=side, plane=plane)
            before = deepcopy(legacy)
            row = candidate_from_schema1(legacy, mode=side + "_full")
            self.assertEqual(candidate_to_schema1(row), before)
            self.assertEqual(legacy, before)
            self.assertEqual(row["candidate_id"], side + ":c0004")
            self.assertEqual(validate_candidate(row), row)
        with self.assertRaisesRegex(ValueError, "mode"):
            candidate_from_schema1(canonical(), mode="conflicting_mode")

    def test_actual_selector_result_adapts_only_the_winner_without_mutation(self):
        for side in ("left", "right"):
            other = "left" if side == "right" else "right"
            result = select_candidates(
                FakeRobot(), FakeSim(), [candidate(side, 0.1), candidate(other, 0.3)], ["left", "right"], np.zeros(56)
            )
            before = deepcopy(result)
            plan = plan_from_selection(result, mode=side + "_full")
            self.assertEqual(plan["active_arms"], [side])
            self.assertEqual(plan["available_arms"], ["left", "right"])
            self.assertEqual(plan["targets"], result["selected"]["candidate"]["targets"])
            self.assertEqual(plan["provenance"], result["selected"]["candidate"]["provenance"])
            self.assertEqual(result, before)
            self.assertEqual(result["selected"]["selected_side"], side)

    def test_inconsistent_or_empty_selected_results_fail(self):
        result = select_candidates(FakeRobot(), FakeSim(), [candidate("right", 0.1)], ["right"], np.zeros(56))
        for field, value in (("candidate_id", "other"), ("selected_side", "left"), ("required_arms", ["left"])):
            altered = deepcopy(result)
            altered["selected"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                plan_from_selection(altered, mode="right_full")
        for field, value in (
            ("status", "no_feasible_candidate"),
            ("schema_version", True),
            ("selected", None),
            ("enabled_arms", ["left"]),
            ("enabled_arms", []),
            ("enabled_arms", ["right", "right"]),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                plan_from_selection({**result, field: value}, mode="right_full")

    def test_both_arm_data_does_not_enable_selector_ik_or_execution(self):
        row = canonical(("left", "right"), "both_full")
        sim = FakeSim()
        result = select_candidates(FakeRobot(), sim, [row], ["left", "right"], np.zeros(56))
        self.assertEqual(result["evaluations"][0]["reasons"], ["unsupported_multi_arm_candidate"])
        self.assertIsNone(result["selected"])
        self.assertEqual(sim.checked, [])
        self.assertFalse(result["execution_ready"])
        with self.assertRaisesRegex(ValueError, "no selected"):
            plan_from_selection(result, mode="both_full")


if __name__ == "__main__":
    unittest.main()
