"""Selection contracts, malformed inputs and command-free import boundaries."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from ada_grasp_ctrl.grasp_selection import GraspSelector
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import decode, load_candidates
from ada_grasp_ctrl.grasp_selection.adapters.state import load_state, named_arm
from ada_grasp_ctrl.grasp_selection.criteria.joint_effort import JointDisplacement
from ada_grasp_ctrl.grasp_selection.types import (
    ARM_NAMES,
    HAND_NAMES,
    STAGES,
    GraspCandidate,
    IKOption,
    RobotState,
    SelectionContext,
    StageTarget,
    rigid,
)
from ada_grasp_ctrl.paths import project_root


def candidate(name="a"):
    target = StageTarget([0.4, -0.4, 0.2], [1, 0, 0, 0], np.zeros(22))
    return GraspCandidate(name, {s: target for s in STAGES}, "fixture", "calibration")


def context(q=None, **kwargs):
    state = RobotState(np.zeros(6) if q is None else q, 10.0, 10.1, "model", "fixture", "fixture")
    return SelectionContext(state, np.eye(4), "fixture", "calibration", 10.1, **kwargs)


def payload():
    c = candidate()
    return {
        "schema_version": 1,
        "frame_id": "world",
        "unit": "m",
        "joint_unit": "rad",
        "quaternion_order": "wxyz",
        "pose_frame": "right_hand_flange",
        "stages": list(STAGES),
        "metadata": {"joint_names": list(HAND_NAMES)},
        "calibration": {"sha256": "calibration"},
        "candidates": [
            {
                "candidate_id": "a",
                "stages": {
                    s: {
                        "position_m": t.position_m.tolist(),
                        "quaternion_wxyz": t.quaternion_wxyz.tolist(),
                        "joint_positions_rad": dict(zip(HAND_NAMES, t.joint_positions_rad.tolist())),
                        "frame_id": t.frame_id,
                        "pose_frame": t.pose_frame,
                        "T_world_hand": t.transform().tolist(),
                    }
                    for s, t in c.stages.items()
                },
            }
        ],
    }


class FakeRobot:
    provenance = {"kind": "test_double"}

    def __init__(self, branches):
        self.branches = branches
        self.calls = 0

    def validate_context(self, ctx):
        if ctx.robot_state.model_id != "model":
            raise ValueError("model mismatch")

    def evaluate(self, c, ctx, ev):
        self.calls += 1
        ev.options = [IKOption(i, q, {}) for i, q in enumerate(self.branches[c.candidate_id])]


class SelectionTests(unittest.TestCase):
    def test_weighted_squared_cost_and_zero_delta(self):
        q = np.arange(6, dtype=float)
        score = JointDisplacement([1, 2, 3, 4, 5, 6])
        cost, detail = score.score(IKOption(0, q, {}), context())
        self.assertEqual(cost, float(q @ np.diag([1, 2, 3, 4, 5, 6]) @ q))
        np.testing.assert_array_equal(detail["delta_rad"], q)
        self.assertEqual(score.score(IKOption(0, q, {}), context(q))[0], 0)

    def test_limited_angles_are_not_wrapped(self):
        cost, details = JointDisplacement().score(IKOption(0, [3.1, 0, 0, 0, 0, 0], {}), context([-3.1, 0, 0, 0, 0, 0]))
        self.assertAlmostEqual(cost, 6.2**2)
        self.assertAlmostEqual(details["delta_rad"][0], 6.2)

    def test_bad_weights(self):
        for weights in ([1] * 5, np.eye(6), [-1] * 6, [float("nan")] * 6, [float("inf")] * 6):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                JointDisplacement(weights)

    def test_multibranch_and_candidate_order(self):
        robot = FakeRobot({"a": [np.ones(6) * 3, np.ones(6)], "b": [np.ones(6) * 2]})
        for order in ([candidate("a"), candidate("b")], [candidate("b"), candidate("a")]):
            result = GraspSelector(robot).select(order, context())
            self.assertEqual(result.selected_candidate_id, "a")
            self.assertEqual(result.selected["solution_index"], 1)
            self.assertEqual(result.cost, 6)
        self.assertEqual(robot.calls, 4)

    def test_exact_tie_break(self):
        robot = FakeRobot({"z": [np.ones(6)], "a": [np.ones(6), np.ones(6)]})
        result = GraspSelector(robot).select([candidate("z"), candidate("a")], context())
        self.assertEqual((result.selected_candidate_id, result.selected["solution_index"]), ("a", 0))

    def test_nonnegative_diagonal_changes_choice(self):
        robot = FakeRobot({"a": [[1, 0, 0, 0, 0, 0]], "b": [[0, 2, 0, 0, 0, 0]]})
        score = JointDisplacement([10, 1, 1, 1, 1, 1])
        result = GraspSelector(robot, scoring_criteria=[score]).select([candidate("a"), candidate("b")], context())
        self.assertEqual(result.selected_candidate_id, "b")

    def test_empty_and_no_feasible(self):
        for cs in ([], [candidate()]):
            result = GraspSelector(FakeRobot({"a": []})).select(cs, context())
            self.assertEqual(result.status, "no_feasible_candidate")
            self.assertIsNone(result.selected_q_rad)

    def test_hard_constraint_cannot_be_outscored(self):
        from ada_grasp_ctrl.grasp_selection.criteria.reachability import Reachability

        class Reject:
            def evaluate(self, c, ctx, ev):
                ev.options[0].reasons.append("forbidden_collision")

        robot = FakeRobot({"a": [np.zeros(6)]})
        result = GraspSelector(robot, [Reachability(robot), Reject()]).select([candidate()], context())
        self.assertIsNone(result.selected_candidate_id)
        self.assertIn("forbidden_collision", result.evaluations[0].options[0].reasons)

    def test_nonfinite_score_rejected(self):
        class BadScore:
            name = "bad"

            def score(self, option, ctx):
                return float("nan"), {}

        result = GraspSelector(FakeRobot({"a": [np.zeros(6)]}), scoring_criteria=[BadScore()]).select(
            [candidate()], context()
        )
        self.assertIsNone(result.selected_candidate_id)

    def test_duplicate_and_calibration_mismatch(self):
        selector = GraspSelector(FakeRobot({"a": [np.zeros(6)]}))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            selector.select([candidate(), candidate()], context())
        result = selector.select([replace(candidate(), calibration_id="wrong")], context())
        self.assertEqual(result.evaluations[0].reasons, ["calibration_mismatch"])

    def test_readiness_never_claims_path_or_hardware_safety(self):
        for gates in (
            {},
            dict.fromkeys(["camera_world", "world_right_base", "hand_mount", "table", "environment"], True),
        ):
            result = GraspSelector(FakeRobot({"a": [np.zeros(6)]})).select([candidate()], context(acceptance=gates))
            self.assertFalse(result.execution_ready)
            self.assertIn("continuous_path_not_checked", result.readiness_reasons)

    def test_input_arrays_cannot_be_mutated(self):
        with self.assertRaises(ValueError):
            candidate().stages["grasp"].position_m[0] = 0


class CandidateContractTests(unittest.TestCase):
    def test_saved_named_joints_permuted_without_double_transform(self):
        p = payload()
        p["metadata"]["joint_names"].reverse()
        original = deepcopy(p)
        batch = decode(p, "fixture")
        self.assertFalse(batch.rejected)
        for s in STAGES:
            np.testing.assert_allclose(
                batch.candidates[0].stages[s].transform(), original["candidates"][0]["stages"][s]["T_world_hand"]
            )
        self.assertEqual(p, original)

    def test_frame_units_quaternion_conventions(self):
        for key, wrong in (
            ("frame_id", "camera"),
            ("pose_frame", "palm"),
            ("unit", "mm"),
            ("joint_unit", "deg"),
            ("quaternion_order", "xyzw"),
        ):
            p = payload()
            p[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(p, "fixture")

    def test_bad_stage_reports_candidate_id(self):
        for key, bad in (
            ("quaternion_wxyz", [0, 0, 0, 0]),
            ("quaternion_wxyz", [2, 0, 0, 0]),
            ("position_m", [float("nan"), 0, 0]),
            ("position_m", [float("inf"), 0, 0]),
            ("pose_frame", "right_hand_C_MC"),
            ("joint_positions_rad", {}),
        ):
            p = payload()
            p["candidates"][0]["stages"]["grasp"][key] = bad
            with self.subTest(key=key):
                batch = decode(p, "fixture")
                self.assertFalse(batch.candidates)
                self.assertEqual(batch.rejected[0].candidate_id, "a")

    def test_missing_phase_and_inconsistent_matrix(self):
        p = payload()
        del p["candidates"][0]["stages"]["pregrasp"]
        self.assertEqual(len(decode(p, "fixture").rejected), 1)
        p = payload()
        p["candidates"][0]["stages"]["grasp"]["T_world_hand"][0][3] += 1
        self.assertEqual(len(decode(p, "fixture").rejected), 1)

    def test_duplicate_ids_fail_batch(self):
        p = payload()
        p["candidates"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            decode(p, "fixture")

    def test_rigid_transform_roundtrip_and_invalid_frames(self):
        from scipy.spatial.transform import Rotation

        world_base = np.eye(4)
        world_base[:3, :3] = Rotation.from_euler("z", 0.7).as_matrix()
        world_base[:3, 3] = [0.1, -0.9, 0.01]
        world_flange = candidate().stages["grasp"].transform()
        base_flange = np.linalg.solve(rigid(world_base, "base"), world_flange)
        np.testing.assert_allclose(world_base @ base_flange, world_flange, atol=1e-12)
        for bad in (np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                rigid(bad, "base")


class StateTests(unittest.TestCase):
    def setUp(self):
        self.p = dict(
            schema_version=1,
            unit="rad",
            joint_names=list(ARM_NAMES),
            positions_rad=list(range(6)),
            stamp_s=10.0,
            received_s=10.1,
            model_id="model",
            source_topic="fixture",
            environment="fixture",
            frame_id="world",
            clock="ros_system",
        )

    def test_permutation(self):
        self.assertEqual(named_arm(list(reversed(ARM_NAMES)), list(reversed(range(6)))), list(range(6)))

    def test_absent_incomplete_duplicate_wrong_side_nonfinite(self):
        for names, values in (
            ([], []),
            (list(ARM_NAMES)[:-1], [0] * 5),
            (["left_J1", *ARM_NAMES[1:]], [0] * 6),
            ([*ARM_NAMES, "left_J1"], [0] * 7),
            ([*ARM_NAMES, "right_J1"], [0] * 7),
            (list(ARM_NAMES), [float("nan")] * 6),
        ):
            with self.subTest(names=names), self.assertRaises(ValueError):
                named_arm(names, values)

    def test_snapshot_replay_and_stale_transport(self):
        state, when = load_state(self.p, "model", replay=True)
        self.assertEqual(when, 10.1)
        for field, value in (("stamp_s", 0), ("stamp_s", 9), ("stamp_s", 11), ("received_s", float("inf"))):
            p = {**self.p, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                load_state(p, "model", replay=True)
        with self.assertRaises(ValueError):
            state.check_fresh(15.0, 0.5)

    def test_bad_units_model_and_live_mock(self):
        for field, value in (("unit", "deg"), ("model_id", "wrong"), ("frame_id", "camera"), ("clock", "unknown")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_state({**self.p, field: value}, "model", replay=True)
        with self.assertRaises(ValueError):
            load_state(self.p, "model", replay=False, now_s=10.1)


class BoundaryTests(unittest.TestCase):
    def test_preview_interpolation_preserves_endpoints_and_limited_angles(self):
        from ada_grasp_ctrl.grasp_selection.preview import interpolate

        frames = np.array([[0.0] * 28, [1.0] * 28, [2.0] * 28, [3.0] * 28])
        for index in range(4):
            np.testing.assert_allclose(interpolate(frames, index * 100 / 3), frames[index], atol=1e-15)
        np.testing.assert_allclose(interpolate(frames, 50), np.full(28, 1.5))
        for bad in (-1, 101, float("nan")):
            with self.assertRaises(ValueError):
                interpolate(frames, bad)

    def test_core_import_does_not_load_ros_torch_or_mujoco(self):
        code = (
            "from ada_grasp_ctrl.grasp_selection import GraspSelector; import sys; "
            "assert not any(n in sys.modules for n in ['rclpy','torch','mujoco'])"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_ros_boundary_import_has_no_numeric_dependencies(self):
        code = (
            "import ada_grasp_ctrl.grasp_selection.capture_state; import sys; "
            "assert not any(n in sys.modules for n in ['rclpy','torch','mujoco','numpy','scipy'])"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_no_command_endpoints_in_package(self):
        import ast

        root = project_root() / "src/ada_grasp_ctrl/grasp_selection"
        forbidden = {"create_publisher", "create_client", "create_service", "ActionClient", "ActionServer", "mj_step"}
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                    self.assertNotIn(name, forbidden, str(path))

    def test_cli_does_not_overwrite_existing_output(self):
        from ada_grasp_ctrl.grasp_selection.cli import main

        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "manifest.json"
            sentinel.write_text("preserve")
            result = main(
                [
                    "--candidates",
                    "missing",
                    "--state",
                    "missing",
                    "--model-report",
                    "missing",
                    "--output-root",
                    directory,
                    "--replay",
                ]
            )
            self.assertEqual(result, 2)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_actual_five_captures_decode(self):
        root = project_root() / "output/fixed_camera"
        manifest = root / "five_objects_gallery/manifest.json"
        if not manifest.exists():
            self.skipTest("Local real captures unavailable; run documented integration command on capture host")
        groups = json.loads(manifest.read_text())["groups"]
        self.assertEqual(len(groups), 5)
        for group in groups:
            batch = load_candidates(Path(group["candidates_path"]))
            self.assertEqual(len(batch.candidates), 8)
            self.assertFalse(batch.rejected)
            self.assertTrue((Path(group["snapshot"]) / "snapshot.npz").exists())


if __name__ == "__main__":
    unittest.main()
