"""Contact boundary policies and real compiled MuJoCo planning isolation."""

from pathlib import Path
from copy import deepcopy
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ada_grasp_ctrl.motion_planning.placement_collision import PlanningScene, EdgePolicy, Observation, contacts
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.pipeline.rrt_placement import (
    COMPLETION_POLICY,
    prepare_execution,
    PlacementPlanner,
    arm_path,
    settings,
    transition,
)
from ada_grasp_ctrl.pipeline.simulation_executor import SimulationExecutor
from ada_grasp_ctrl.pipeline.simulation_plan import verify_plan, preview
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash
from ada_grasp_ctrl.pipeline import dual_runner
import test_dual_mujoco_simulation as simulation_tests


class BoundaryPolicyTests(unittest.TestCase):
    def test_completion_fallback_preserves_failed_search_and_rejects_invalid_commands(self):
        def configuration(q):
            q = np.asarray(q)
            if q.shape != (56,) or not np.all(np.isfinite(q)) or np.max(np.abs(q)) > 2:
                raise ValueError("Invalid command")
            return q.copy()

        scene = SimpleNamespace(
            fixed=np.zeros(56),
            arm=slice(0, 6),
            mode="carry",
            queries=0,
            robot=SimpleNamespace(configuration=configuration),
            query=lambda q: Observation({}, {}, [dict(category="robot_table")]),
        )
        scene.full = lambda arm: np.r_[arm, np.zeros(50)]
        goal = np.zeros(56)
        goal[0] = 1
        path, report = arm_path(
            scene, goal, settings({}), lambda: False, time.monotonic(), direct_only=True, allow_fallback=True
        )
        self.assertEqual(report["status"], "invalid_descent")
        self.assertTrue(report["fallback_applied"])
        np.testing.assert_array_equal(path[-1], goal)
        for index, value in ((0, np.nan), (0, 3), (6, 0.1), (28, 0.1)):
            invalid = goal.copy()
            invalid[index] = value
            with self.assertRaises(ValueError):
                arm_path(
                    scene, invalid, settings({}), lambda: False, time.monotonic(), direct_only=True, allow_fallback=True
                )
        path, report = arm_path(
            scene, goal, settings({}), lambda: True, time.monotonic(), direct_only=True, allow_fallback=True
        )
        self.assertFalse(path)
        self.assertEqual(report["status"], "canceled")

    def test_completion_transition_and_low_clearance_warn_without_stopping(self):
        scene = SimpleNamespace(
            mode="carry",
            robot=SimpleNamespace(configuration=lambda q: q),
            query=lambda q: Observation({}, {}, [dict(category="robot_table")]),
        )
        warnings = []
        transition(scene, np.zeros(56), np.zeros(56), settings({}), lambda: False, warnings=warnings)
        self.assertEqual(len(warnings), 1)
        planner = PlacementPlanner.__new__(PlacementPlanner)
        planner.plan = dict(execution_policy=COMPLETION_POLICY, target=dict(table_plane_world=[0, 0, 1, 0]))
        planner.scene = SimpleNamespace(
            observe=scene.query,
            clearance=lambda *_: 0.0015,
        )
        planner.executor = SimpleNamespace(sim=SimpleNamespace(data=None))
        planner.warn = Mock()
        self.assertEqual(planner.validate_waypoint_support(), 0.0015)
        self.assertEqual(planner.warn.call_count, 2)

    def test_drop_requires_sustained_absence_and_displacement_not_collision(self):
        planner = PlacementPlanner.__new__(PlacementPlanner)
        planner.plan = dict(execution_policy=COMPLETION_POLICY)
        planner.last_attachment = np.eye(4)
        planner.contact_absent_since = None
        planner.grasp_evidence = dict(drop_confirmed=False, cleanup_required=False, maximum_contact_absence_s=0.0)
        data = SimpleNamespace(time=0.0)
        pose = np.eye(4)
        observed = Observation({}, {}, [dict(category="object_table")])
        planner.executor = SimpleNamespace(sim=SimpleNamespace(data=data))
        planner.scene = SimpleNamespace(
            mode="carry",
            wrist_body=0,
            object_body=1,
            observe=lambda _: observed,
            body_pose=lambda _, body: np.eye(4) if body == 0 else pose,
        )
        planner.monitor = dict(hand={})
        planner.warn = Mock()
        for timestamp in (0.0, 0.3, 1.0):
            data.time = timestamp
            planner.check_live(np.zeros(56), np.zeros(56), timestamp)
            self.assertFalse(planner.cleanup_requested())
        observed.hand = {(1, 2): -0.001}
        planner.check_live(np.zeros(56), np.zeros(56), 1.0)
        observed.hand = {}
        pose[0, 3] = 0.03
        for timestamp in (1.1, 1.4):
            data.time = timestamp
            planner.check_live(np.zeros(56), np.zeros(56), timestamp)
            self.assertFalse(planner.cleanup_requested())
        data.time = 1.7
        planner.check_live(np.zeros(56), np.zeros(56), 1.7)
        self.assertTrue(planner.cleanup_requested())
        self.assertEqual(planner.grasp_evidence["outcome"], "dropped")

    def test_carry_ik_uses_actual_attachment_retains_command_and_rejects_bad_results(self):
        planner = PlacementPlanner.__new__(PlacementPlanner)
        planner.side, planner.block, planner.arm = "right", slice(28, 56), slice(28, 34)
        measured = np.zeros(56)
        held = measured.copy()
        held[34:] = 0.236
        attachment = np.eye(4)
        attachment[:3, 3] = [0.03, -0.02, 0.1]
        pose, target = np.eye(4), np.eye(4)
        pose[2, 3], target[0, 3] = 0.2, 0.3
        nominal = np.zeros(56)
        nominal[planner.arm] = 0.4
        planner.plan = dict(accepted_place_q56=nominal.tolist(), target=dict(table_plane_world=[0, 0, 1, 0]))
        planner.options = dict(timeout_s=3)
        planner.executor = SimpleNamespace(cancel=threading.Event(), sim=SimpleNamespace(data=object()))
        planner.held_state = Mock(return_value=(held, attachment, {(1, 2)}))
        goal = held.copy()
        goal[planner.arm] = 0.1
        planner.scene = SimpleNamespace(
            robot=SimpleNamespace(solve_arm_ik=Mock(return_value=(goal, dict(accepted=True)))),
            clearance_pose=Mock(return_value=pose),
        )
        planner.planned = Mock(return_value=[held, goal])
        for segment, destination in (("lift", None), ("carry", target), ("place", target)):
            planner.carry_path(measured, pose, destination, segment, time.monotonic())
            seed, side, wrist_goal = planner.scene.robot.solve_arm_ik.call_args.args
            np.testing.assert_array_equal(seed[34:], held[34:])
            np.testing.assert_array_equal(seed[:28], measured[:28])
            np.testing.assert_array_equal(
                seed[planner.arm], nominal[planner.arm] if segment == "carry" else measured[planner.arm]
            )
            np.testing.assert_allclose(wrist_goal @ attachment, pose if segment == "lift" else target, atol=1e-12)
            self.assertEqual(side, "right")
            self.assertEqual(
                planner.held_state.call_args.kwargs["table_clearance"], 0.005 if segment == "carry" else 0.0
            )
        planner.planned.reset_mock()
        planner.executor.cancel.set()
        with self.assertRaises(InterruptedError):
            planner.carry_path(measured, pose, target, "carry", time.monotonic())
        planner.executor.cancel.clear()
        planner.scene.robot.solve_arm_ik.return_value = (goal, dict(accepted=False))
        with self.assertRaisesRegex(ValueError, "IK rejected"):
            planner.carry_path(measured, pose, target, "carry", time.monotonic())
        mutated = goal.copy()
        mutated[34] = 0.0
        planner.scene.robot.solve_arm_ik.return_value = (mutated, dict(accepted=True))
        with self.assertRaisesRegex(ValueError, "changed held fingers"):
            planner.carry_path(measured, pose, target, "carry", time.monotonic())
        planner.planned.assert_not_called()

    def test_rejected_ik_reuses_valid_saved_endpoint_without_applying_rejected_vector(self):
        planner = PlacementPlanner.__new__(PlacementPlanner)
        planner.side, planner.block, planner.arm = "left", slice(0, 28), slice(0, 6)
        held, nominal, rejected = np.zeros((3, 56))
        held[6:28] = 0.236
        nominal[:6] = 0.4
        rejected[:] = np.nan
        planner.plan = dict(execution_policy=COMPLETION_POLICY, accepted_place_q56=nominal.tolist())
        planner.options = dict(timeout_s=3)
        planner.executor = SimpleNamespace(cancel=threading.Event(), sim=SimpleNamespace(data=None))
        planner.held_state = Mock(return_value=(held, np.eye(4), {(1, 2)}))
        planner.warn = Mock()
        planner.scene = SimpleNamespace(
            robot=SimpleNamespace(
                solve_arm_ik=Mock(return_value=(rejected, dict(accepted=False))),
                configuration=lambda q: q.copy(),
            )
        )
        planner.planned = Mock(return_value=[])
        planner.carry_path(held, np.eye(4), np.eye(4), "place", time.monotonic())
        commanded = planner.planned.call_args.args[0]
        np.testing.assert_array_equal(commanded[:6], nominal[:6])
        np.testing.assert_array_equal(commanded[6:], held[6:])
        diagnostic = planner.planned.call_args.kwargs["ik"]
        self.assertFalse(diagnostic["accepted"])
        self.assertEqual(diagnostic["command_source"], "accepted_place_endpoint")

    def test_transport_ik_tries_clear_branch_before_collision_fallback(self):
        planner = PlacementPlanner.__new__(PlacementPlanner)
        planner.side, planner.arm = "left", slice(0, 6)
        planner.options = dict(timeout_s=3)
        planner.executor = SimpleNamespace(cancel=threading.Event())
        held, nominal = np.zeros((2, 56))
        held[6:28] = nominal[6:28] = 0.236
        nominal[:6] = 0.4
        planner.scene = SimpleNamespace(
            robot=SimpleNamespace(
                configuration=lambda q: q.copy(),
                solve_arm_ik=Mock(side_effect=[(held, dict(accepted=True)), (nominal, dict(accepted=True))]),
            ),
            query=Mock(side_effect=[Observation({}, {}, [dict(category="robot_table")]), Observation({}, {}, [])]),
        )
        goal, diagnostic = planner.transport_ik(held, nominal, np.eye(4), time.monotonic())
        np.testing.assert_array_equal(goal, nominal)
        self.assertEqual(diagnostic["seed_index"], 1)
        self.assertEqual(len(diagnostic["attempts"]), 2)

    def test_squeeze_command_and_measured_to_commanded_collision_validation(self):
        for side in ("left", "right"):
            planner = PlacementPlanner.__new__(PlacementPlanner)
            planner.side, planner.block, planner.arm = side, side_slice(side), side_slice(side, True)
            measured = np.zeros(56)
            measured[side_slice("right" if side == "left" else "left")] = 0.4
            squeeze = np.ones(56) * 0.8
            squeeze[planner.block][6:] = 0.236
            planner.plan = dict(nominal_endpoints=dict(squeeze=squeeze.tolist()))
            planner.options = dict(resolution=0.02, timeout_s=3)
            planner.executor = SimpleNamespace(sim=SimpleNamespace(data=object()), cancel=threading.Event())
            sampled = []

            def query(q):
                sampled.append(q.copy())
                pairs = {(1, 2): -0.001}
                if q[planner.block][6] > 0.1:
                    pairs[(1, 3)] = -0.001
                return Observation({}, pairs, [])

            planner.scene = SimpleNamespace(
                configure=Mock(),
                observe=lambda _: Observation({}, {(1, 2): -0.001}, []),
                query=query,
                body_pose=lambda *_: np.eye(4),
                wrist_body=0,
                mode="squeeze",
            )
            pose = np.eye(4)
            pose[0, 3] = 0.2
            held, attachment, pairs = planner.held_state(measured, pose)
            np.testing.assert_array_equal(held[planner.block][6:], squeeze[planner.block][6:])
            unchanged = np.ones(56, dtype=bool)
            unchanged[planner.block][6:] = False
            np.testing.assert_array_equal(held[unchanged], measured[unchanged])
            np.testing.assert_array_equal(attachment, pose)
            self.assertEqual(pairs, {(1, 2), (1, 3)})
            self.assertGreater(len(sampled), 2)
            self.assertLessEqual(np.max(np.abs(np.diff(sampled, axis=0))), 0.02)
            np.testing.assert_array_equal(planner.scene.configure.call_args.kwargs["measured_hand"], measured)
            # A collision between the measured and commanded hand shapes must reject carry.
            planner.scene.query = lambda q: Observation(
                {}, {(1, 2): -0.001}, [dict(category="robot_table")] if 0.08 < q[planner.block][6] < 0.16 else []
            )
            with self.assertRaisesRegex(ValueError, "transition collision"):
                planner.held_state(measured, pose)

    def test_both_held_hand_shapes_are_checked_at_arm_samples(self):
        scene = PlanningScene.__new__(PlanningScene)
        scene.side = "left"
        scene.measured_hand = np.zeros(56)
        commanded = np.ones(56)
        scene._query = Mock(side_effect=[Observation({}, {}, []), Observation({}, {}, [dict(category="robot_table")])])
        observed = scene.query(commanded)
        self.assertTrue(observed.forbidden)
        first, second = [call.args[0] for call in scene._query.call_args_list]
        np.testing.assert_array_equal(first, commanded)
        np.testing.assert_array_equal(second[:6], commanded[:6])
        np.testing.assert_array_equal(second[6:28], np.zeros(22))
        np.testing.assert_array_equal(second[28:], commanded[28:])

    def test_carry_stages_keep_squeeze_and_gate_transport_on_actual_five_mm(self):
        for clearance, missed_hover in (
            (0.006, False),
            (0.005, False),
            (0.004, False),
            (-0.000057, False),
            (0.006, True),
        ):
            planner = PlacementPlanner.__new__(PlacementPlanner)
            planner.side, planner.block, planner.arm = "left", slice(0, 28), slice(0, 6)
            planner.executor = SimpleNamespace(
                sim=SimpleNamespace(data=object()), cancel=threading.Event(), lock=threading.Lock(), result={}
            )
            planner.cursor, planner.accepted, planner.pending = 0.0, [], None
            planner.reports = [{}]
            planner.save = Mock()
            plane = [0, 0, 1, 0]
            target = np.eye(4)
            target[0, 3] = 0.1
            planner.plan = dict(
                target=dict(table_plane_world=plane),
                target_object_pose=target.tolist(),
                placement_parameters=dict(settle_s=0.3),
            )
            q0, q1, q2, q3 = np.zeros((4, 56))
            q1[0], q2[0], q3[0] = 0.01, 0.1, 0.11
            poses = [np.eye(4) for _ in range(4)]
            poses[1][2, 3] = clearance
            planner.snapshot = Mock(side_effect=list(zip((q0, q1, q2, q3), poses)))
            held = [q.copy() for q in (q0, q1, q2, q3)]
            if missed_hover:
                q2[0] += 0.02
            for q in held:
                q[6:28] = 0.236
            planner.carry_path = Mock(
                side_effect=[
                    ([held[0], held[1]], held[1], {(1, 2)}),
                    ([held[1], held[2]], held[2], {(1, 2)}),
                    ([held[2], held[3]], held[3], {(1, 2)}),
                ]
            )
            observed = Observation({(3, 4): clearance}, {(1, 2): -0.001}, [])
            preplace = target.copy()
            preplace[2, 3] = 0.015
            planner.scene = SimpleNamespace(
                observe=lambda _: observed,
                query=lambda _: observed,
                configure=Mock(),
                clearance=Mock(return_value=clearance),
                clearance_pose=Mock(return_value=preplace),
            )
            stages = planner.carry_stages()
            lift = next(stages)
            self.assertEqual(lift["phase"], "lift")
            self.assertEqual(planner.carry_path.call_count, 1)
            if clearance < 0.005:
                with self.assertRaisesRegex(ValueError, "below required 0.005"):
                    next(stages)
                self.assertEqual(planner.carry_path.call_count, 1)
                continue
            transport = next(stages)
            self.assertEqual(transport["phase"], "transport")
            transport_args = planner.carry_path.call_args.args
            np.testing.assert_array_equal(transport_args[0], q1)
            np.testing.assert_array_equal(transport_args[1], poses[1])
            np.testing.assert_array_equal(transport_args[2], preplace)
            self.assertEqual(planner.carry_path.call_count, 2)
            if missed_hover:
                with self.assertRaisesRegex(ValueError, "has not reached preplace"):
                    next(stages)
                self.assertEqual(planner.carry_path.call_count, 2)
                continue
            place, settle = next(stages), next(stages)
            for stage in (lift, transport, place, settle):
                for row in [stage["arms"]["left"]["start_positions_rad"], *[w["left"] for w in stage["waypoints_rad"]]]:
                    np.testing.assert_array_equal(row[6:], held[0][6:28])
            args = planner.carry_path.call_args.args
            np.testing.assert_array_equal(args[0], q2)
            np.testing.assert_array_equal(args[1], poses[2])
            np.testing.assert_array_equal(args[2], target)
            self.assertEqual(planner.reports[-1]["actual_lift_clearance_m"], clearance)
            np.testing.assert_array_equal(planner.scene.configure.call_args.args[0], held[3])
            with self.assertRaises(StopIteration):
                next(stages)

    def test_final_descent_checks_interior_and_never_falls_back_to_rrt(self):
        scene = SimpleNamespace(fixed=np.zeros(56), arm=slice(0, 6), mode="carry", queries=0)

        def full(arm):
            q = scene.fixed.copy()
            q[scene.arm] = arm
            return q

        def query(q):
            scene.queries += 1
            return Observation({}, {}, [dict(category="robot_table")] if 0.4 < q[0] < 0.6 else [])

        scene.full, scene.query = full, query
        goal = np.zeros(56)
        goal[0] = 1
        with patch("ada_grasp_ctrl.pipeline.rrt_placement.search") as search:
            path, report = arm_path(scene, goal, settings({}), lambda: False, time.monotonic(), direct_only=True)
            search.assert_not_called()
        self.assertFalse(path)
        self.assertEqual(report["status"], "invalid_descent")
        self.assertEqual(report["last_collision"][0]["category"], "robot_table")
        for canceled, began, expected in (
            (lambda: True, time.monotonic(), "canceled"),
            (lambda: False, time.monotonic() - 4, "timeout"),
        ):
            path, report = arm_path(scene, goal, settings({}), canceled, began, direct_only=True)
            self.assertFalse(path)
            self.assertEqual(report["status"], expected)

    def test_live_support_allows_persistent_contact_and_recontact_within_cap(self):
        planner = PlacementPlanner.__new__(PlacementPlanner)
        observed = Observation({(1, 2): -0.000113}, {(3, 4): -0.001}, [])
        planner.scene = SimpleNamespace(mode="carry", observe=lambda _: observed)
        planner.executor = SimpleNamespace(sim=SimpleNamespace(data=None))
        planner.arm = slice(0, 6)
        planner.options = dict(extension=0.2)
        planner.monitor = dict(hand=observed.hand.copy())
        q = np.zeros(56)
        observed.support = {(1, 2): -0.000253, (1, 5): -0.000236}
        planner.check_live(q, q, 0.04)
        # A contact checker violation must still abort before accepting support.
        observed.forbidden = [dict(category="object_table", distance_m=-0.0021)]
        with self.assertRaisesRegex(ValueError, "live_object_collision"):
            planner.check_live(q, q, 0.08)
        observed.forbidden = []
        outside = q.copy()
        outside[0] = 0.201
        planner.check_live(outside, outside, 0.08)
        observed.support = {}
        planner.check_live(q, q, 0.12)
        observed.support = {(1, 5): -0.002}
        planner.check_live(outside, outside, 10.0)
        observed.hand = {}
        with self.assertRaisesRegex(ValueError, "live_grasp_lost"):
            planner.check_live(outside, outside, 10.04)

    def test_legacy_grasp_and_release_policy_preserves_carry_and_retreat_gates(self):
        checker = PlanningScene.__new__(PlanningScene)
        checker.side, checker.allowed_pairs, checker.table_pairs = "left", set(), []
        checker.model = SimpleNamespace(geom_bodyid=np.array([1, 2]))
        checker.sim = SimpleNamespace(body_labels={1: "left", 2: "object"})
        checker.proxy = SimpleNamespace()
        # Contact extraction is stubbed; phase acceptance still uses the production policy.
        for side, body, is_hand in (
            ("left", "left_middle_DP", True),
            ("right", "right_middle_DP", True),
            ("left", "left_J6_link", False),
        ):
            checker.sim.body_labels[1] = side
            for penetration in (0.000113, 0.001, 0.01, 0.010001):
                row = dict(
                    category="robot_object",
                    sides=[side],
                    bodies=[body, "object"],
                    hand_bodies=[body] if is_hand else [],
                    penetration_m=penetration,
                    distance_m=-penetration,
                )
                checker.data = SimpleNamespace(contact=[SimpleNamespace(geom=(0, 1), dist=-penetration)])
                live_data = SimpleNamespace(contact=checker.data.contact)
                with patch("ada_grasp_ctrl.motion_planning.placement_collision.live_contacts", return_value=[row]):
                    for mode in ("approach", "pregrasp", "grasp", "squeeze", "release", "carry", "settle", "retreat"):
                        checker.mode = mode
                        expected_rejection = not (
                            side == "left"
                            and is_hand
                            and (
                                mode in ("grasp", "squeeze", "release") or (mode == "pregrasp" and penetration <= 0.01)
                            )
                        )
                        for data in (checker.data, live_data):
                            with self.subTest(side=side, body=body, mode=mode, penetration=penetration):
                                self.assertEqual(bool(checker.observe(data).forbidden), expected_rejection)

    def test_cli_opt_in_requires_simulation_placement_and_preserves_default(self):
        base = [
            "--environment",
            "simulation",
            "--placement-aware",
            "--placement-execute",
            "--output-root",
            "/tmp/unused-rrt-cli-test",
        ]
        with patch("ada_grasp_ctrl.pipeline.simulation_ui.run_simulation", return_value=0) as run:
            self.assertEqual(dual_runner.main(base), 0)
            self.assertNotIn("placement_planner", run.call_args.args[1]["simulation"])
            self.assertEqual(dual_runner.main([*base, "--placement-planner", "rrt_connect"]), 0)
            self.assertEqual(run.call_args.args[1]["simulation"]["placement_planner"], "rrt_connect")
            count = run.call_count
            for options in (
                ["--placement-planner", "rrt_connect", "--output-root", "/tmp/unused-rrt-cli-test"],
                [*base, "--environment", "hardware", "--placement-planner", "rrt_connect"],
            ):
                self.assertEqual(dual_runner.main(options), 2)
            self.assertEqual(run.call_count, count)

    def policy(self, values, mode="carry", goal=1.0):
        def query(q):
            distance = values(float(q[0]))
            touching = {} if distance is None else {(1, 2): distance}
            bad = (
                [dict(category="object_table")]
                if mode == "carry" and distance is not None and distance < -0.002
                else []
            )
            return Observation(touching if mode == "carry" else {}, touching if mode != "carry" else {}, bad)

        scene = SimpleNamespace(query=query, full=lambda q: q, mode=mode)
        return EdgePolicy(scene, np.array([0.0]), np.array([goal]), resolution=0.02)

    def test_support_is_allowed_at_all_samples_but_penetration_cap_is_enforced(self):
        for values in (
            lambda x: -0.002,
            lambda x: -0.001 if 0.4 < x < 0.6 else None,
            lambda x: -0.001 - x * 0.01 if x < 0.1 else None,
            lambda x: -0.001 if x < 0.3 else None,
            lambda x: -0.001 if x < 0.05 or 0.1 < x < 0.15 else None,
        ):
            policy = self.policy(values)
            for a, b in ((0.0, 1.0), (1.0, 0.0), (0.04, 0.5), (0.5, 0.5)):
                self.assertTrue(policy(np.array([a]), np.array([b])))
        for location in (0.0, 0.5, 1.0):
            policy = self.policy(lambda x: -0.0021 if abs(x - location) < 0.01 else -0.001)
            self.assertFalse(policy(np.array([0.0]), np.array([1.0])))

    def test_release_allows_active_hand_contact_changes_but_retreat_still_separates(self):
        for mode in ("release", "retreat"):
            policy = self.policy(lambda x: -0.002 * (1 - x / 0.1) if x < 0.1 else None, mode)
            self.assertTrue(policy(np.array([0.0]), np.array([1.0])))
            for values in (
                lambda x: -0.002 - x if x < 0.1 else None,
                lambda x: -0.001 if x < 0.05 or x > 0.95 else None,
            ):
                self.assertEqual(self.policy(values, mode)(np.array([0.0]), np.array([1.0])), mode == "release")
        self.assertTrue(self.policy(lambda x: -0.001, "release")(np.array([0.0]), np.array([1.0])))
        self.assertFalse(self.policy(lambda x: -0.001, "retreat")(np.array([0.0]), np.array([1.0])))
        self.assertFalse(self.policy(lambda x: -0.001, "retreat", goal=0.0)(np.array([0.0]), np.array([0.0])))

    def test_release_live_and_sampled_checks_allow_new_pairs_without_weakening_other_collisions(self):
        def observation(q):
            return Observation({}, {(1, 2) if q[0] < 0.1 else (1, 3): -0.000113}, [])

        for mode in ("release", "retreat"):
            scene = SimpleNamespace(mode=mode, query=observation, full=lambda q: q)
            policy = EdgePolicy(scene, np.array([0.0]), np.array([1.0]))
            self.assertEqual(policy(np.array([0.0]), np.array([1.0])), mode == "release")
        planner = PlacementPlanner.__new__(PlacementPlanner)
        observed = observation(np.array([1.0]))
        planner.scene = SimpleNamespace(mode="release", observe=lambda _: observed)
        planner.executor = SimpleNamespace(sim=SimpleNamespace(data=None))
        planner.monitor = dict(hand={(1, 2): -0.0001})
        q = np.zeros(56)
        planner.check_live(q, q, 0.04)
        observed.hand = {(1, 3): -0.001}
        planner.check_live(q, q, 0.08)
        observed.hand = {}
        planner.check_live(q, q, 0.12)
        observed.hand = {(1, 2): -0.0002}
        planner.check_live(q, q, 0.16)
        for category in ("robot_table", "robot_object", "object_table", "object_environment"):
            observed.forbidden = [dict(category=category)]
            with self.assertRaisesRegex(ValueError, "live_object_collision"):
                planner.check_live(q, q, 0.2)
        observed.forbidden = []
        planner.scene.mode = "retreat"
        with self.assertRaisesRegex(ValueError, "live_separation"):
            planner.check_live(q, q, 0.2)


class MuJoCoPlanningTests(unittest.TestCase):
    setUpClass = classmethod(simulation_tests.SceneContracts.setUpClass.__func__)
    tearDownClass = classmethod(simulation_tests.SceneContracts.tearDownClass.__func__)
    scene = simulation_tests.SceneContracts.scene

    def test_fifteen_mm_waypoint_uses_rotated_collision_mesh_and_tilted_plane(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _ = self.scene(Path(directory))
            checker = PlanningScene(scene.sim, "left")
            q = scene.sim.configuration()
            pose = checker.body_pose(scene.sim.data, checker.object_body)
            angle = 0.4
            pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
            normal = np.array([0.3, -0.2, 1.0])
            normal /= np.linalg.norm(normal)
            plane = np.r_[normal, 0.0]
            checker.configure(q, "squeeze", pose)
            checker.query(q)
            # Place the calibrated plane at the rotated mesh's lowest vertex.
            plane[3] = -checker.clearance(checker.data, plane)
            before = scene.sim.data.qpos.copy()
            target = checker.clearance_pose(checker.data, plane, 0.015)
            np.testing.assert_array_equal(target[:3, :3], pose[:3, :3])
            np.testing.assert_allclose(target[:3, 3] - pose[:3, 3], normal * 0.015, atol=1e-12)
            checker.configure(q, "squeeze", target)
            checker.query(q)
            self.assertAlmostEqual(checker.clearance(checker.data, plane), 0.015, places=12)
            # Destination orientation and lateral position are retained at preplace.
            requested = pose.copy()
            requested[:3, :3] = np.eye(3)
            requested[:3, 3] += np.array([0.02, -0.01, 0.0])
            requested[:3, 3] -= normal * 0.1
            preplace = checker.clearance_pose(checker.data, plane, 0.015, pose=requested)
            np.testing.assert_array_equal(preplace[:3, :3], requested[:3, :3])
            delta = preplace[:3, 3] - requested[:3, 3]
            np.testing.assert_allclose(delta - normal * (normal @ delta), 0, atol=1e-12)
            checker.configure(q, "squeeze", preplace)
            checker.query(q)
            self.assertAlmostEqual(checker.clearance(checker.data, plane), 0.015, places=12)
            np.testing.assert_array_equal(scene.sim.data.qpos, before)

    def test_transport_margin_covers_robot_hand_object_without_moving_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _ = self.scene(Path(directory))
            sim = scene.sim
            checker = PlanningScene(sim, "left")
            q = sim.configuration()
            plane = np.r_[self.builder.plane["normal_xyz"], self.builder.plane["d"]]
            before = {name: getattr(sim.data, name).copy() for name in ("qpos", "qvel", "ctrl", "geom_xpos")}
            table_positions = sim.model.geom_pos.copy()
            wrist = checker.body_pose(sim.data, checker.wrist_body)
            for gap in (0.003, 0.0049, 0.005, 0.006):
                pose = checker.clearance_pose(sim.data, plane, gap)
                checker.configure(
                    q, "carry", pose, attachment=np.linalg.solve(wrist, pose), table_plane=plane, table_clearance=0.005
                )
                observation = checker.query(q)
                violations = [r for r in observation.forbidden if r["category"] == "planning_table_clearance"]
                self.assertEqual(bool(violations), gap < 0.005)
                # The virtual margin is planning-only; physical support has its own 2 mm cap.
                self.assertFalse(checker.observe(checker.data).forbidden)
            normal = np.array([0.2, -0.1, 1.0])
            normal /= np.linalg.norm(normal)
            for body in ("left_J6_link", "left_thumb_elastomer", "right_J6_link"):
                geom = next(
                    g for g in checker.clearance_geoms if sim.model.body(int(sim.model.geom_bodyid[g])).name == body
                )
                base_distance = float(
                    np.min(checker.collision_meshes[geom] @ (checker.data.geom_xmat[geom].reshape(3, 3).T @ normal))
                    + checker.data.geom_xpos[geom] @ normal
                )
                checker.table_plane = np.r_[normal, 0.004 - base_distance]
                violations = checker.table_clearance_violations(checker.data)
                row = next(r for r in violations if r["geom"] == sim.model.geom(geom).name)
                self.assertAlmostEqual(row["distance_m"], 0.004)
                self.assertEqual(row["required_clearance_m"], 0.005)
            checker.configure(q, "carry", pose, attachment=np.linalg.solve(wrist, pose))
            self.assertFalse(checker.table_clearance_violations(checker.data))
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(sim.data, name), value)
            np.testing.assert_array_equal(sim.model.geom_pos, table_positions)

    def test_actual_support_depth_retains_two_mm_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _ = self.scene(Path(directory))
            checker = PlanningScene(scene.sim, "left")
            q = scene.sim.configuration()
            pose = checker.body_pose(scene.sim.data, checker.object_body)
            checker.configure(q, "grasp", pose)
            initial_distance = min(checker.query(q).support.values())
            normal = np.asarray(self.builder.plane["normal_xyz"])
            for distance, rejected in ((-0.0019, False), (-0.0021, True)):
                shifted = pose.copy()
                shifted[:3, 3] += (distance - initial_distance) * normal
                checker.configure(q, "grasp", shifted)
                observation = checker.query(q)
                self.assertAlmostEqual(min(observation.support.values()), distance, places=6)
                self.assertEqual(any(c["category"] == "object_table" for c in observation.forbidden), rejected)

    def test_zero_distance_does_not_fabricate_support_above_actual_table_box(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _ = self.scene(Path(directory))
            checker = PlanningScene(scene.sim, "left")
            q = scene.sim.configuration()
            plane = np.r_[self.builder.plane["normal_xyz"], self.builder.plane["d"]]
            pose = checker.clearance_pose(scene.sim.data, plane, 0.005)
            checker.configure(q, "squeeze", pose)
            checker.query(q)
            with patch("ada_grasp_ctrl.motion_planning.placement_collision.mujoco.mj_geomDistance", return_value=0.0):
                support = checker.observe(checker.data).support
                self.assertFalse(contacts(support))
                self.assertAlmostEqual(min(support.values()), 0.005, places=6)
            pose[:3, 3] -= np.asarray(plane[:3]) * 0.006
            checker.configure(q, "squeeze", pose)
            observation = checker.query(q)
            self.assertTrue(contacts(observation.support))
            self.assertAlmostEqual(min(observation.support.values()), -0.001, places=6)

    def stationary_plan(self, root, side="right"):
        """Synthetic no-grasp selection; used to prove absence of false attachment."""
        scene, _, state = self.scene(root)
        q = scene.sim.configuration().tolist()
        target = dict(Delta_T=np.eye(4).tolist(), target_revision=1, target_sha256="synthetic_target")
        chosen = dict(
            candidate_id="no_grasp_fixture",
            required_arms=[side],
            effort=0.0,
            candidate=dict(execution_mode=side + "_full"),
            stages={p: dict(q56=q) for p in ("pregrasp", "grasp", "squeeze")},
        )
        endpoint = dict(
            arm=side,
            candidate_id=chosen["candidate_id"],
            mode=side + "_full",
            place_q56=q,
            target=target,
            target_revision=1,
            target_sha256="synthetic_target",
            source_sha256="fixture",
            references={
                str(root / p): file_hash(root / p) for p in ("robot_state.json", "scene.mjb", "settled_state.npz")
            },
        )
        write_json(root / "placement_selection_report.json", dict(selected=chosen))
        write_json(root / "placement_endpoint_preview.json", endpoint)
        engine = SimpleNamespace(
            root=root, robot=scene.robot, static=self.builder.static, state=state, config=self.config
        )
        return scene, prepare_execution(engine, endpoint, root, lambda: False, scene.sim)

    def test_pending_preview_target_identity_missing_grasp_failure_and_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configure = PlanningScene.configure
            with patch.object(PlanningScene, "configure", autospec=True, side_effect=configure) as calls:
                scene, plan = self.stationary_plan(root)
            self.assertEqual(
                [call.args[2] for call in calls.call_args_list], ["approach", "pregrasp", "grasp", "squeeze"]
            )
            initial_time, initial_qpos = scene.sim.data.time, scene.sim.data.qpos.copy()
            verify_plan(plan)
            self.assertEqual(plan["pending_segments"], ["lift", "carry", "place", "retreat"])
            self.assertNotIn("place", plan["execution_phases"])
            for progress in (0, 20, 80, 100):
                preview(plan, progress)
            np.testing.assert_array_equal(scene.sim.data.qpos, initial_qpos)
            self.assertEqual(scene.sim.data.time, initial_time)
            changed = deepcopy(plan)
            changed["target_object_pose"][0][3] += 0.01
            with self.assertRaisesRegex(ValueError, "object target"):
                verify_plan(changed)
            changed = deepcopy(plan)
            changed["schedule"]["stages"][0]["segment_durations_s"][0] += 0.1
            with self.assertRaisesRegex(ValueError, "timing"):
                verify_plan(changed)
            executor = SimulationExecutor(scene, plan, self.config, root)
            with patch.object(PlanningScene, "configure", autospec=True, side_effect=configure) as calls:
                result = executor.run(realtime=False, continue_hold=False)
            self.assertEqual(
                [call.args[2] for call in calls.call_args_list][:4], ["approach", "pregrasp", "grasp", "squeeze"]
            )
            self.assertFalse(result["pipeline_completed"])
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["cleanup_completed"])
            self.assertEqual(result["grasp_evidence"]["outcome"], "not_acquired")
            self.assertEqual(
                [s["phase"] for s in executor.planner.accepted],
                plan["execution_phases"] + ["release_1", "release_2", "retreat"],
            )
            self.assertTrue((root / "dynamic_paths.json").exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, plan = self.stationary_plan(root)
            plan.pop("execution_policy")
            executor = SimulationExecutor(scene, plan, self.config, root)
            result = executor.run(realtime=False, continue_hold=False)
            self.assertFalse(result["pipeline_completed"])
            self.assertIn("attachment refused", result["error"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, plan = self.stationary_plan(root, "left")
            executor = SimulationExecutor(scene, plan, self.config, root)
            executor.cancel.set()
            clock = scene.sim.data.time
            result = executor.run(realtime=False, continue_hold=False)
            self.assertEqual(result["status"], "canceled")
            self.assertEqual(scene.sim.data.time, clock)
            self.assertFalse(executor.planner.accepted)

    def test_actual_geometry_attachment_passive_fingers_and_live_state_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _ = self.scene(Path(directory))
            sim = scene.sim
            before = {
                name: getattr(sim.data, name).copy()
                for name in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
            }
            clock, targets = sim.data.time, sim.targets
            for side in ("left", "right"):
                checker = PlanningScene(sim, side, self.builder.static)
                q = sim.configuration()
                pose = checker.body_pose(sim.data, checker.object_body)
                checker.configure(q, "approach", pose)
                observation = checker.query(q)
                self.assertFalse(observation.forbidden)
                self.assertTrue(observation.support)
                np.testing.assert_allclose(checker.body_pose(checker.data, checker.object_body), pose, atol=1e-14)
                attachment = np.linalg.solve(checker.body_pose(sim.data, checker.wrist_body), pose)
                checker.configure(q, "carry", pose, attachment=attachment)
                candidate = q[checker.arm].copy()
                candidate[0] += 0.03
                full = checker.full(candidate)
                checker.query(full)
                np.testing.assert_allclose(
                    checker.body_pose(checker.data, checker.object_body),
                    checker.body_pose(checker.data, checker.wrist_body) @ attachment,
                    atol=1e-12,
                )
                passive = side_slice("left" if side == "right" else "right")
                np.testing.assert_array_equal(checker.data.qpos[sim.qpos][passive], q[passive])
                np.testing.assert_array_equal(
                    checker.data.qpos[sim.qpos][side_slice(side)][6:], q[side_slice(side)][6:]
                )
                below = pose.copy()
                below[2, 3] -= 0.02
                checker.configure(q, "approach", below)
                self.assertTrue(any(r["category"] == "object_table" for r in checker.query(q).forbidden))
                wrist = checker.body_pose(sim.data, checker.wrist_body)
                checker.configure(q, "carry", pose, attachment=np.linalg.solve(wrist, below))
                self.assertTrue(any(r["category"] == "object_table" for r in checker.query(q).forbidden))
                other = "left" if side == "right" else "right"
                at_passive = checker.body_pose(sim.data, sim.model.body(other + "_hand_flange").id)
                checker.configure(q, "approach", at_passive)
                self.assertTrue(any(r["category"] == "robot_object" for r in checker.query(q).forbidden))
                checker.configure(q, "carry", pose, attachment=np.linalg.solve(wrist, at_passive))
                self.assertTrue(any(r["category"] == "robot_object" for r in checker.query(q).forbidden))
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(sim.data, name), value)
            self.assertEqual(sim.data.time, clock)
            np.testing.assert_array_equal(sim.targets, targets)


if __name__ == "__main__":
    unittest.main()
