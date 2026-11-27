"""Paired placement contracts; analytical tests are not physical grasp evidence."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.grasp_selection.multi_arm import select_paired_candidates
from ada_grasp_ctrl.motion_planning.placement_collision import Observation, PlanningScene
from ada_grasp_ctrl.pipeline.bimanual_placement import carry_path, interpolate_pose, solve_waypoint
from ada_grasp_ctrl.pipeline.placement_schedule import planned_stage, validate_rrt_schedule
from ada_grasp_ctrl.pipeline.rrt_placement import PlacementPlanner, COMPLETION_POLICY, arm_path, settings
from ada_grasp_ctrl.pipeline.simulation_plan import reference
from ada_grasp_ctrl.placement.feasibility import PlacementGate
from ada_grasp_ctrl.robot_environment.model import side_slice
from test_native_paired import paired
from test_phase2_pipeline import FakeRobot, FakeSim
import test_dual_mujoco_simulation as simulation_tests


class CartesianRobot(FakeRobot):
    """Analytical six-coordinate wrist used only for object-path invariants."""

    def solve_arm_ik(self, q, side, pose, **kwargs):
        q = q.copy()
        q[side_slice(side, True)] = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec()]
        return q, dict(accepted=True)

    def fk(self, q, side):
        values = q[side_slice(side, True)]
        pose = np.eye(4)
        pose[:3, 3] = values[:3]
        pose[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
        return pose


class PairedPlacementTests(unittest.TestCase):
    def test_cli_both_only_alias_and_auto_reach_paired_capable_simulation(self):
        from ada_grasp_ctrl.pipeline import dual_runner

        base = [
            "--environment",
            "simulation",
            "--placement-aware",
            "--placement-execute",
            "--placement-planner",
            "rrt_connect",
            "--selector",
            "hierarchical",
            "--grasp-model",
            "native_multimode",
            "--output-root",
            "/tmp/unused-paired-cli-test",
        ]
        with patch("ada_grasp_ctrl.pipeline.simulation_ui.run_simulation", return_value=0) as run:
            for mode in ("auto", "both_only"):
                self.assertEqual(dual_runner.main([*base, "--arms", mode]), 0)
                config = run.call_args.args[1]
                self.assertEqual(config["grasp_mode"], mode)
                self.assertEqual(config["enabled_arms"], ["left", "right"])

    def test_both_modes_gate_placement_without_changing_grasp_effort(self):
        robot, static = FakeRobot(), FakeSim()
        target = dict(Delta_T=np.eye(4).tolist(), target_revision=1, target_sha256="fixture")
        for mode in ("both_three", "both_full"):
            gate = PlacementGate(robot, static, target)
            gate.face_coverage = SimpleNamespace(
                check=Mock(side_effect=AssertionError("Paired coverage must be skipped"))
            )
            result = select_paired_candidates(
                robot, static, [paired(mode=mode)], ["left", "right"], np.zeros(56), endpoint_gate=gate
            )
            self.assertIsNotNone(result["selected"])
            self.assertAlmostEqual(result["selected"]["effort"], 6 * (0.2**2 + 0.3**2))
            self.assertTrue(result["selected"]["placement"]["face_coverage"]["skipped"])
            gate.face_coverage.check.assert_not_called()
            rejected = Mock(return_value=dict(accepted=False, reason="place_endpoint_ik_rejected"))
            result = select_paired_candidates(
                robot, static, [paired(mode=mode)], ["left", "right"], np.zeros(56), endpoint_gate=rejected
            )
            self.assertIsNone(result["selected"])
            self.assertTrue(any(a.get("grasp_feasible") for a in result["evaluations"][0]["attempts"]))

    def test_either_ik_failure_or_combined_interarm_collision_rejects_endpoint(self):
        robot, static = FakeRobot(), FakeSim()
        target = dict(Delta_T=np.eye(4).tolist(), target_revision=1, target_sha256="fixture")
        gate = PlacementGate(robot, static, target)
        for side in ("left", "right"):
            row = paired()
            row["targets"][side]["squeeze"]["T_world_hand"][0][3] = 6
            result = gate(row, "both", np.zeros(56))
            self.assertFalse(result["accepted"])
            self.assertEqual(result["reason"], "place_endpoint_ik_rejected")
        static.check_static_contacts = Mock(
            return_value=[
                dict(
                    category="inter_robot",
                    bodies=["left_J6_link", "right_J6_link"],
                    sides=["left", "right"],
                    hand_bodies=[],
                    penetration_m=0.001,
                )
            ]
        )
        result = gate(paired(), "both", np.zeros(56))
        self.assertEqual(result["reason"], "place_endpoint_collision")
        self.assertTrue(result["collision_checked"])

    def test_shared_timeline_and_review_rejects_either_arm_mutation(self):
        q, goal = np.zeros(56), np.zeros(56)
        goal[0], goal[28] = 0.1, 0.7
        sides = ["left", "right"]
        stage = planned_stage(sides, "pregrasp", [q, goal], 0, dwell=0.3)
        self.assertEqual(stage["required_durations_s"]["left"], stage["required_durations_s"]["right"])
        plan = dict(active_arms=sides, initial_q56=q.tolist())
        half = reference(plan, stage, stage["duration_s"] / 2)
        np.testing.assert_allclose(half, goal / 2)
        stages = [stage]
        for phase in ("pregrasp_fingers", "grasp", "squeeze"):
            stages.append(planned_stage(sides, phase, [goal, goal], stages[-1]["settled_time"], dwell=0.3))
        plan.update(
            schedule=dict(active_arms=sides, stages=stages, settle_time_s=0.3, end_time=stages[-1]["settled_time"]),
            pending_segments=["lift", "carry", "place", "retreat"],
            execution_phases=[s["phase"] for s in stages],
            stage_completion="timed",
            target=dict(Delta_T=np.eye(4), target_revision=1, target_sha256="fixture"),
            target_revision=1,
            target_sha256="fixture",
            source_object_pose=np.eye(4),
            target_object_pose=np.eye(4),
            nominal_endpoints={p: goal.tolist() for p in ("pregrasp", "grasp", "squeeze")},
        )
        validate_rrt_schedule(plan)
        changed = deepcopy(plan)
        changed["schedule"]["stages"][0]["waypoints_rad"][0]["right"][0] += 0.1
        with self.assertRaises(ValueError):
            validate_rrt_schedule(changed)

    def planner(self):
        robot = CartesianRobot()
        q = np.zeros(56)
        attachments = {}
        for side, y in (("left", -0.1), ("right", 0.1)):
            pose = np.eye(4)
            pose[1, 3] = y
            attachments[side] = pose
            q, _ = robot.solve_arm_ik(q, side, pose)
        squeeze = q.copy()
        fingers = np.r_[6:28, 34:56]
        squeeze[fingers] = 0.3
        scene = SimpleNamespace(
            robot=robot,
            configure=Mock(),
            observe=lambda _: Observation({}, {(0, 1): 0}, []),
            queries=0,
            query=lambda _: Observation({}, {}, [dict(category="inter_robot")]),
        )
        planner = SimpleNamespace(
            scene=scene,
            executor=SimpleNamespace(cancel=threading.Event(), sim=SimpleNamespace(data=None)),
            sides=["left", "right"],
            attachments=attachments,
            last_attachment=np.linalg.inv(attachments["left"]),
            last_pairs=set(),
            fingers=fingers,
            arm=np.r_[0:6, 28:34],
            plan=dict(nominal_endpoints=dict(squeeze=squeeze.tolist()), target=dict(table_plane_world=[0, 0, 1, 0])),
            options=settings({}),
            validate_transition=Mock(return_value=set()),
            reports=[],
            pending=None,
            warn=Mock(),
            save=Mock(),
            grasp_evidence={},
        )
        return planner, q

    def test_rotated_carry_uses_common_object_keeps_preload_and_attachments(self):
        for angle in (0, 0.25):
            planner, q = self.planner()
            goal = np.eye(4)
            goal[:3, 3] = [0.04, -0.02, 0.015]
            goal[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
            attachments = deepcopy(planner.attachments)
            path, end, _ = carry_path(planner, q, np.eye(4), goal, "carry", time.monotonic())
            self.assertGreater(len(path), 2)
            for point in path:
                np.testing.assert_array_equal(point[planner.fingers], 0.3)
            for side in planner.sides:
                np.testing.assert_array_equal(planner.attachments[side], attachments[side])
                np.testing.assert_allclose(planner.scene.robot.fk(end, side), goal @ attachments[side], atol=1e-12)
            self.assertFalse(planner.reports[-1]["collision_certified"])
            self.assertTrue(planner.reports[-1]["collision_warnings"])
            stage = planned_stage(planner.sides, "transport", path, 0)
            self.assertLess(stage["duration_s"], 10)
            self.assertEqual(set(stage["waypoints_rad"][0]), {"left", "right"})

    def test_invalid_waypoint_never_returns_partial_path_or_fallback(self):
        planner, q = self.planner()
        goal = np.eye(4)
        goal[0, 3] = 0.03
        solve = planner.scene.robot.solve_arm_ik

        def reject_after_first(q, side, pose, **kwargs):
            candidate, diagnostic = solve(q, side, pose)
            diagnostic["accepted"] = pose[0, 3] < 0.01
            return candidate, diagnostic

        planner.scene.robot.solve_arm_ik = reject_after_first
        with self.assertRaisesRegex(ValueError, "hold current grasp"):
            carry_path(planner, q, np.eye(4), goal, "carry", time.monotonic())
        report = planner.reports[-1]
        self.assertFalse(report["partial_path_executed"])
        self.assertEqual(report["execution_method"], "hold_current_grasp")
        self.assertEqual(planner.grasp_evidence["outcome"], "planning_failed_hold")
        with self.assertRaises(TimeoutError):
            solve_waypoint(planner.scene.robot, q, {"left": goal}, lambda: False, deadline=time.monotonic() - 1)
        planner.executor.cancel.set()
        with self.assertRaises(InterruptedError):
            solve_waypoint(planner.scene.robot, q, {"left": goal}, planner.executor.cancel.is_set)

    def test_shortest_arc_interpolation(self):
        start, end = np.eye(4), np.eye(4)
        start[:3, :3] = Rotation.from_euler("z", 170, degrees=True).as_matrix()
        end[:3, :3] = Rotation.from_euler("z", -170, degrees=True).as_matrix()
        midpoint = interpolate_pose(start, end, 0.5)
        self.assertAlmostEqual(Rotation.from_matrix(midpoint[:3, :3]).magnitude(), np.pi)

    def test_one_side_contact_resets_both_loss_timer(self):
        planner = PlacementPlanner.__new__(PlacementPlanner)
        planner.side = "left"
        planner.plan = dict(active_arms=["left", "right"], execution_policy=COMPLETION_POLICY)
        planner.attachments = {s: np.eye(4) for s in planner.sides}
        planner.contact_absent_since = None
        planner.grasp_evidence = dict(maximum_contact_absence_s=0.0, drop_confirmed=False, cleanup_required=False)
        planner.warn = Mock()
        data = SimpleNamespace(time=0.0)
        pose = np.eye(4)
        pose[0, 3] = 0.03
        planner.executor = SimpleNamespace(sim=SimpleNamespace(data=data))
        planner.scene = SimpleNamespace(
            object_body=2,
            body_pose=lambda _, b: pose if b == 2 else np.eye(4),
            model=SimpleNamespace(body=lambda name: SimpleNamespace(id=0), geom_bodyid=[0, 1, 2]),
            sim=SimpleNamespace(body_labels={0: "left", 1: "right", 2: "object"}),
        )
        for timestamp, pairs in ((0.0, {}), (0.4, {(1, 2): 0}), (1.0, {}), (1.4, {})):
            data.time = timestamp
            planner.monitor_grasp(Observation({}, pairs, []))
            self.assertFalse(planner.cleanup_requested())
        data.time = 1.6
        planner.monitor_grasp(Observation({}, {}, []))
        self.assertTrue(planner.cleanup_requested())
        self.assertEqual(planner.grasp_evidence["outcome"], "dropped")
        self.assertTrue(any(call.args[0] == "one_sided_contact_loss" for call in planner.warn.call_args_list))


class PairedGeometryTests(unittest.TestCase):
    setUpClass = classmethod(simulation_tests.SceneContracts.setUpClass.__func__)
    tearDownClass = classmethod(simulation_tests.SceneContracts.tearDownClass.__func__)
    scene = simulation_tests.SceneContracts.scene

    def test_paired_review_json_round_trip_cleanup_and_initial_retreat_goal(self):
        from test_rrt_placement import MuJoCoPlanningTests
        from ada_grasp_ctrl.pipeline.simulation_executor import SimulationExecutor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, plan = MuJoCoPlanningTests.stationary_plan(self, root, "both")
            self.assertEqual(plan, json.loads((root / "simulation_plan.json").read_text()))
            executor = SimulationExecutor(scene, plan, self.config, root)
            result = executor.run(realtime=False, continue_hold=False)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["cleanup_completed"])
            self.assertFalse(result["pipeline_completed"])
            self.assertEqual(result["grasp_evidence"]["outcome"], "not_acquired")
            retreat = executor.planner.accepted[-1]
            for side in ("left", "right"):
                np.testing.assert_allclose(
                    retreat["arms"][side]["positions_rad"][:6], np.asarray(plan["initial_q56"])[side_slice(side, True)]
                )
            self.assertTrue(all(set(s["arms"]) == {"left", "right"} for s in executor.planner.accepted))

    def test_twelve_dof_bounds_direct_path_and_isolated_common_object(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _ = self.scene(Path(directory))
            checker = PlanningScene(scene.sim, ["left", "right"])
            np.testing.assert_array_equal(checker.arm, np.r_[0:6, 28:34])
            q = scene.sim.configuration()
            pose = checker.body_pose(scene.sim.data, checker.object_body)
            before = scene.sim.data.qpos.copy()
            checker.configure(q, "approach", pose)
            goal = q.copy()
            goal[checker.arm] += 0.001
            path, report = arm_path(checker, goal, settings({}), lambda: False, time.monotonic(), allow_fallback=True)
            self.assertTrue(path)
            np.testing.assert_allclose(path[-1], goal)
            self.assertTrue(report["status"] == "success" or report["fallback_applied"])
            checker.configure(q, "carry", pose, attachment=np.eye(4))
            shared = pose.copy()
            shared[2, 3] += 0.02
            checker.object_path_pose = shared
            checker.query(q)
            np.testing.assert_allclose(checker.body_pose(checker.data, checker.object_body), shared, atol=1e-12)
            np.testing.assert_array_equal(scene.sim.data.qpos, before)


if __name__ == "__main__":
    unittest.main()
