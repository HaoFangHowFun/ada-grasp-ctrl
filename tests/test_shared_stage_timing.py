"""Offline named-joint timing fixtures; no IK, GPU, ROS, network or actuation."""

import argparse
from copy import deepcopy
import json
import math
import subprocess
import sys
import unittest
from unittest.mock import patch

from ada_grasp_ctrl.experiments.right_ros_contract import CYCLE_PHASES, PHASES, side_names
from ada_grasp_ctrl.experiments.right_ros_sender import motion_duration
from ada_grasp_ctrl.grasp_selection.contracts import plan_from_candidate
from ada_grasp_ctrl.mpc.reference import linear_reference
from ada_grasp_ctrl.pipeline.shared_timing import build_stage_schedule, sample_stage, validate_stage_schedule
from ada_grasp_ctrl.ros_execution.settings import DEFAULTS
from test_grasp_contracts import canonical
from test_hierarchical_selection import row, run


def named(side, arm=0.0, finger=0.0):
    arm_names, hand_names = side_names(side)
    return {**dict.fromkeys(arm_names, arm), **dict.fromkeys(hand_names, finger)}


def fixture(arms=("left", "right"), *, full=False):
    candidate = canonical(arms, "both_full" if len(arms) == 2 else arms[0] + "_full")
    starts = {side: named(side) for side in arms}
    targets = {}
    phases = ("above_object", *CYCLE_PHASES) if full else PHASES
    for phase in phases:
        targets[phase] = {}
        for side in arms:
            distance = 0.1 if side == "left" else 0.4
            fingers = 0.0 if phase in ("above_object", "release", "return") else 0.2
            targets[phase][side] = named(side, arm=0.0 if phase == "return" else distance, finger=fingers)
            if phase in PHASES:
                candidate["targets"][side][phase]["joint_positions_rad"] = dict.fromkeys(side_names(side)[1], fingers)
    return plan_from_candidate(candidate, available_arms=["left", "right"]), starts, targets


class SharedTimingTests(unittest.TestCase):
    def test_selected_candidate_connects_to_schedule_using_existing_ik_endpoints(self):
        result = run({"right_full": [row("right_full", distance=0.2)]}, {"right_full": 1})
        names = sum(side_names("right"), ())
        endpoints = {
            phase: {"right": dict(zip(names, result["selected"]["stages"][phase]["q56"][28:]))} for phase in PHASES
        }
        schedule = build_stage_schedule(result["plan"], {"right": named("right")}, endpoints)
        self.assertEqual(schedule["plan"]["candidate_id"], result["selected"]["candidate_id"])
        for stage in schedule["stages"]:
            sample = sample_stage(schedule, stage["phase"], stage["stage_end_time"])
            self.assertEqual(
                sample["arms"]["right"]["positions_rad"], result["selected"]["stages"][stage["phase"]]["q56"][28:]
            )

    def test_shared_timestamps_progress_and_endpoints_with_asymmetric_limits(self):
        plan, starts, targets = fixture()
        policies = {"left": {"arm_speed": 0.05}, "right": {"hand_speed": 0.2}}
        schedule = build_stage_schedule(plan, starts, targets, policies_by_arm=policies, stage_start_time=10.0)
        stage = schedule["stages"][0]
        self.assertNotEqual(stage["required_durations_s"]["left"], stage["required_durations_s"]["right"])
        self.assertEqual(stage["duration_s"], max(stage["required_durations_s"].values()))
        for phase_stage in schedule["stages"]:
            phase = phase_stage["phase"]
            t0, tf = phase_stage["stage_start_time"], phase_stage["stage_end_time"]
            for time, fraction in (
                (t0 - 1, 0),
                (t0, 0),
                (t0 + phase_stage["duration_s"] * 0.5, 0.5),
                (tf, 1),
                (tf + 1, 1),
            ):
                result = sample_stage(schedule, phase, time)
                self.assertAlmostEqual(result["progress"], fraction)
                self.assertEqual((result["stage_start_time"], result["stage_end_time"]), (t0, tf))
                for side in plan["active_arms"]:
                    expected = phase_stage["arms"][side]
                    sampled = result["arms"][side]
                    self.assertEqual(sampled["joint_names"], list(sum(side_names(side), ())))
                    for a, b, q in zip(
                        expected["start_positions_rad"], expected["positions_rad"], sampled["positions_rad"]
                    ):
                        self.assertAlmostEqual(q, a + fraction * (b - a))
                    if fraction in (0, 1):
                        self.assertEqual(
                            sampled["positions_rad"], expected["positions_rad" if fraction else "start_positions_rad"]
                        )

    def test_hand_motion_including_last_joint_can_dominate_duration(self):
        plan, starts, targets = fixture(("right",))
        last = side_names("right")[1][-1]
        for phase in PHASES:
            targets[phase]["right"][last] = 3.0
            plan["targets"]["right"][phase]["joint_positions_rad"][last] = 3.0
        stage = build_stage_schedule(plan, starts, targets)["stages"][0]
        expected = (
            max(2.0, 1.875 * 0.4 / 0.15, math.sqrt(5.774 * 0.4 / 0.3), 1.875 * 3 / 0.4, math.sqrt(5.774 * 3 / 0.8))
            * 1.05
            + 0.1
        )
        self.assertEqual(stage["duration_s"], expected)
        self.assertGreater(stage["duration_s"], 3 / DEFAULTS["hand_speed"])

    def test_single_arm_matches_existing_duration_and_linear_reference(self):
        for side in ("left", "right"):
            plan, starts, targets = fixture((side,))
            before = deepcopy((plan, starts, targets))
            schedule = build_stage_schedule(plan, starts, targets)
            self.assertEqual((plan, starts, targets), before)
            args = argparse.Namespace(minimum_duration=2.0, **DEFAULTS)
            for stage in schedule["stages"]:
                data = stage["arms"][side]
                expected = motion_duration(data["start_positions_rad"], data["positions_rad"], args)
                self.assertEqual(stage["duration_s"], expected)
                elapsed = expected * 0.3
                result = sample_stage(schedule, stage["phase"], stage["stage_start_time"] + elapsed)
                reference = linear_reference(data["start_positions_rad"], data["positions_rad"], elapsed, expected)
                for a, b in zip(result["arms"][side]["positions_rad"], reference):
                    self.assertAlmostEqual(a, b, places=14)
            self.assertFalse(schedule["execution_ready"])
            self.assertFalse(schedule["hardware_ready"])

    def test_passive_measured_side_does_not_change_schedule(self):
        plan, starts, targets = fixture(("right",))
        baseline = build_stage_schedule(plan, starts, targets)
        starts["left"] = named("left", arm=10000, finger=10000)
        self.assertEqual(build_stage_schedule(plan, starts, targets), baseline)
        self.assertEqual(baseline["active_arms"], ["right"])
        self.assertTrue(all(set(s["arms"]) == {"right"} for s in baseline["stages"]))

    def test_cycle_schedule_preserves_settle_hold_release_return_policies(self):
        plan, starts, targets = fixture(full=True)
        with (
            patch("socket.socket", side_effect=AssertionError("network")),
            patch("ada_grasp_ctrl.robot_environment.kinematics.solve_arm_ik", side_effect=AssertionError("IK")),
        ):
            schedule = build_stage_schedule(plan, starts, targets, stage_start_time=3.0, hold_s=0.5)
        self.assertEqual([s["phase"] for s in schedule["stages"]], ["above_object", *CYCLE_PHASES])
        for previous, following in zip(schedule["stages"], schedule["stages"][1:]):
            self.assertEqual(previous["settled_time"], following["stage_start_time"])
            self.assertEqual(previous["stage_end_time"] + DEFAULTS["settle_time"], following["stage_start_time"])
        hold = next(s for s in schedule["stages"] if s["phase"] == "hold")
        self.assertEqual(hold["duration_s"], 0.5)
        self.assertEqual(hold["required_durations_s"], {"left": 0.5, "right": 0.5})
        self.assertEqual(schedule["end_time"], schedule["stages"][-1]["settled_time"])
        self.assertEqual(validate_stage_schedule(json.loads(json.dumps(schedule, allow_nan=False))), schedule)

    def test_joint_order_is_explicit_not_dictionary_insertion_order(self):
        plan, starts, targets = fixture()
        expected = build_stage_schedule(plan, starts, targets)
        starts = {side: dict(reversed(list(values.items()))) for side, values in starts.items()}
        targets = {
            phase: {side: dict(reversed(list(values.items()))) for side, values in data.items()}
            for phase, data in reversed(list(targets.items()))
        }
        self.assertEqual(build_stage_schedule(plan, starts, targets), expected)

    def test_actual_start_rebases_all_references_after_arrival(self):
        schedule = build_stage_schedule(*fixture())
        stage = schedule["stages"][1]
        actual = 123.0
        sample = sample_stage(schedule, "grasp", actual + stage["duration_s"] / 2, stage_start_time=actual)
        self.assertEqual(sample["stage_start_time"], actual)
        self.assertEqual(sample["stage_end_time"], actual + stage["duration_s"])
        self.assertAlmostEqual(sample["progress"], 0.5)
        self.assertEqual(set(sample["arms"]), {"left", "right"})

    def test_speed_bounds_and_linear_acceleration_limitation_are_explicit(self):
        schedule = build_stage_schedule(*fixture())
        self.assertFalse(schedule["acceleration_continuous"])
        self.assertFalse(schedule["continuous_path_verified"])
        for stage in schedule["stages"]:
            for side, data in stage["arms"].items():
                for index, (a, b) in enumerate(zip(data["start_positions_rad"], data["positions_rad"])):
                    limit = schedule["policies_by_arm"][side]["arm_speed" if index < 6 else "hand_speed"]
                    self.assertLessEqual(abs(b - a) / stage["duration_s"], limit)
        stage = schedule["stages"][0]
        target = stage["arms"]["right"]
        self.assertGreater((target["positions_rad"][0] - target["start_positions_rad"][0]) / stage["duration_s"], 0)

    def test_malformed_targets_and_selected_finger_mismatch_are_rejected(self):
        for change in (
            "missing_joint",
            "unknown_joint",
            "wrong_side",
            "nan",
            "bool",
            "finger_mismatch",
            "missing_stage",
            "unknown_stage",
            "passive_target",
        ):
            plan, starts, targets = fixture(("right",))
            values = targets["grasp"]["right"]
            if change == "missing_joint":
                values.pop("right_J1")
            elif change == "unknown_joint":
                values["wrong_name"] = values.pop("right_J1")
            elif change == "wrong_side":
                targets["grasp"] = {"left": named("left")}
            elif change in ("nan", "bool"):
                values["right_J1"] = float("nan") if change == "nan" else True
            elif change == "finger_mismatch":
                values[side_names("right")[1][0]] += 0.1
            elif change == "missing_stage":
                targets.pop("pregrasp")
            elif change == "unknown_stage":
                targets["place"] = {"right": named("right")}
            else:
                targets["grasp"]["left"] = named("left")
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_stage_schedule(plan, starts, targets)

    def test_policy_floor_limits_and_oversized_durations_fail_closed(self):
        plan, starts, targets = fixture(("right",))
        for policy in (
            {"minimum_duration": 1},
            {"arm_speed": 0.16},
            {"hand_speed": 0},
            {"arm_acceleration": 0.31},
            {"hand_acceleration": float("nan")},
            {"arm_speed": 1e-9},
            {"unknown": 2},
        ):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                build_stage_schedule(plan, starts, targets, policies_by_arm={"right": policy})
        for kwargs in (
            {"stage_start_time": -1},
            {"stage_start_time": 1e100},
            {"settle_time_s": 0},
            {"hold_s": 301},
            {"hold_s": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_stage_schedule(plan, starts, targets, **kwargs)

    def test_cycle_endpoint_policy_violations_are_rejected(self):
        for phase, name in (
            ("hold", "right_J1"),
            ("release", "right_J1"),
            ("return", "right_J1"),
            ("lift", side_names("right")[1][0]),
        ):
            plan, starts, targets = fixture(("right",), full=True)
            targets[phase]["right"][name] += 0.01
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, phase):
                build_stage_schedule(plan, starts, targets)

    def test_serialized_schedule_tampering_is_rejected(self):
        baseline = build_stage_schedule(*fixture())
        for change in (
            "duration",
            "start",
            "joint_order",
            "continuity",
            "readiness",
            "schema",
            "duplicate",
            "extra_arm",
        ):
            schedule = deepcopy(baseline)
            stage = schedule["stages"][0]
            if change == "duration":
                stage["duration_s"] /= 2
            elif change == "start":
                stage["stage_start_time"] += 1
            elif change == "joint_order":
                stage["arms"]["right"]["joint_names"].reverse()
            elif change == "continuity":
                schedule["stages"][1]["arms"]["right"]["start_positions_rad"][0] += 1
            elif change == "readiness":
                schedule["execution_ready"] = True
            elif change == "schema":
                schedule["schema_version"] = True
            elif change == "duplicate":
                schedule["stages"].append(deepcopy(stage))
            else:
                stage["arms"]["third"] = deepcopy(stage["arms"]["right"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_stage_schedule(schedule)

    def test_import_has_no_ros_torch_or_network_dependency(self):
        code = (
            "import sys, socket; "
            "socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network')); "
            "import ada_grasp_ctrl.pipeline.shared_timing; "
            "assert 'rclpy' not in sys.modules; assert 'torch' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
