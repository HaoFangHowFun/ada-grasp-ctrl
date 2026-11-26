"""Winner-only grasp-to-place geometry, reviewed timing and free-body execution."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ada_grasp_ctrl.pipeline import dual_runner
from ada_grasp_ctrl.pipeline.placement_execution import (
    endpoint_poses,
    prepare_execution,
    preview_points,
    solve_endpoint,
    check_reference,
)
from ada_grasp_ctrl.pipeline.placement_schedule import (
    PHASES,
    LEGACY_PHASES,
    PREVIOUS_PHASES,
    build_schedule,
    parameters,
    validate_schedule,
)
from ada_grasp_ctrl.pipeline.simulation_plan import preview, reference, verify_plan
from ada_grasp_ctrl.pipeline.simulation_executor import SimulationExecutor
from ada_grasp_ctrl.pipeline.shared_timing import build_stage_schedule
from ada_grasp_ctrl.pipeline.paired_artifacts import named_state
from ada_grasp_ctrl.grasp_selection.contracts import plan_from_candidate, candidate_to_schema1
from ada_grasp_ctrl.grasp_selection.multi_arm import select_candidates
from ada_grasp_ctrl.grasp_selection.types import jsonable
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.robot_environment.model import side_slice, JOINT_NAMES
from test_grasp_contracts import canonical
from test_phase2_pipeline import FakeRobot, FakeSim
import test_dual_mujoco_simulation as simulation_tests
from test_dual_mujoco_simulation import idle_reference

import numpy as np
from scipy.spatial.transform import Rotation


def schedule_fixture(side="right"):
    row = canonical([side], side + "_full")
    q = np.full(56, 0.001)
    targets = {}
    for phase, fingers in (("pregrasp", 0.02), ("grasp", 0.06), ("squeeze", 0.1)):
        row["targets"][side][phase]["joint_positions_rad"] = dict.fromkeys(
            row["targets"][side][phase]["joint_positions_rad"], fingers
        )
    for i, phase in enumerate(PHASES):
        full = q.copy()
        full[side_slice(side)] = [0.05 * min(i, 6)] * 6 + [0.1] * 22
        if phase in ("pregrasp", "grasp", "release_1", "release_2", "retreat_lift", "retreat"):
            source = {
                "release_1": "grasp",
                "release_2": "pregrasp",
                "retreat_lift": "pregrasp",
                "retreat": "pregrasp",
            }.get(phase, phase)
            full[side_slice(side)][6:] = list(row["targets"][side][source]["joint_positions_rad"].values())
        if phase in ("settle", "release_1", "release_2"):
            full[side_slice(side)][:6] = [0.3] * 6
        targets[phase] = {side: named_state(full)[side]}
    # Preserve Place exactly, including floating representation.
    for phase in ("settle", "release_1", "release_2"):
        for name in list(targets["place"][side])[:6]:
            targets[phase][side][name] = targets["place"][side][name]
    grasp = plan_from_candidate(row, available_arms=[side])
    schedule = build_schedule(grasp, named_state(q), targets, {})
    place = q.copy()
    place[side_slice(side)] = list(targets["place"][side].values())
    plan = dict(
        kind="ada_simulation_reference",
        environment="simulation",
        workflow="grasp_to_place",
        stage_completion="timed",
        reference_hz=25,
        schedule=schedule,
        execution_phases=list(PHASES),
        initial_q56=q.tolist(),
        accepted_place_q56=place.tolist(),
        active_arms=[side],
        references={},
    )
    return row, plan


class PlacementExecutionTests(unittest.TestCase):
    def test_geometry_rotates_approach_and_uses_tilted_normal(self):
        row, _ = schedule_fixture()
        row["targets"]["right"]["pregrasp"]["T_world_hand"][0][3] += 0.04
        place = np.eye(4)
        place[:3, 3] = [0.4, -0.2, 0.1]
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        normal = np.array([0.2, -0.1, 1.0])
        normal /= np.linalg.norm(normal)
        endpoint = dict(
            arm="right",
            placement=dict(T_world_hand_place=place.tolist()),
            target=dict(Delta_T=delta.tolist(), table_plane_world=[*normal, 0.1]),
        )
        poses = endpoint_poses(endpoint, row, parameters({}))
        np.testing.assert_allclose(poses["preplace"][:3, 3] - place[:3, 3], 0.15 * normal)
        np.testing.assert_allclose(poses["retreat_lift"][:3, 3] - place[:3, 3], 0.1 * normal)
        np.testing.assert_allclose(poses["retreat"][:3, 3] - poses["retreat_lift"][:3, 3], [0, 0.1, 0], atol=1e-15)
        for pose in poses.values():
            np.testing.assert_array_equal(pose[:3, :3], place[:3, :3])
        # A rotated withdrawal can point down; its origin must be the lifted wrist.
        delta[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix()
        endpoint["target"].update(Delta_T=delta.tolist(), table_plane_world=[0, 0, 1, 0])
        poses = endpoint_poses(endpoint, row, parameters({}))
        self.assertAlmostEqual(poses["retreat_lift"][2, 3] - place[2, 3], 0.1)
        self.assertAlmostEqual(poses["retreat"][2, 3] - place[2, 3], 0.05)
        row["targets"]["right"]["pregrasp"]["T_world_hand"] = row["targets"]["right"]["grasp"]["T_world_hand"]
        with self.assertRaisesRegex(ValueError, "degenerate"):
            endpoint_poses(endpoint, row, parameters({}))

    def test_schedule_exact_release_durations_passive_and_linear_preview(self):
        for side in ("left", "right"):
            _, plan = schedule_fixture(side)
            verify_plan(plan)
            stages = {s["phase"]: s for s in plan["schedule"]["stages"]}
            self.assertEqual(tuple(stages), PHASES)
            for phase, duration in (
                ("settle", 0.5),
                ("release_1", 0.5),
                ("release_2", 1),
                ("retreat_lift", 2),
                ("retreat", 2),
            ):
                stage = stages[phase]
                self.assertEqual(stage["duration_s"], duration)
                self.assertEqual(stage["settled_time"], stage["stage_end_time"])
            self.assertEqual(stages["place"]["stage_end_time"], stages["settle"]["stage_start_time"])
            targets = {
                phase: {side: dict(zip(stage["arms"][side]["joint_names"], stage["arms"][side]["positions_rad"]))}
                for phase, stage in stages.items()
            }
            grasp = plan["schedule"]["plan"]
            start = named_state(np.array(plan["initial_q56"]))
            original = build_stage_schedule(grasp, start, {p: targets[p] for p in PHASES[:5]})
            self.assertEqual(original["stages"], plan["schedule"]["stages"][:5])
            custom = build_schedule(
                grasp,
                start,
                targets,
                dict(settle_s=0.8, release_1_s=0.7, release_2_s=1.2, retreat_lift_s=2.5, retreat_s=3),
            )
            validate_schedule(custom)
            self.assertEqual([s["duration_s"] for s in custom["stages"][-5:]], [0.8, 0.7, 1.2, 2.5, 3])
            for stage in stages.values():
                q = reference(plan, stage, stage["duration_s"] / 2)
                data = stage["arms"][side]
                np.testing.assert_allclose(
                    q[side_slice(side)], (np.array(data["start_positions_rad"]) + data["positions_rad"]) / 2
                )
                progress = (stage["stage_start_time"] + stage["duration_s"] / 2) / plan["schedule"]["end_time"] * 100
                np.testing.assert_allclose(preview(plan, progress), q)
                passive = side_slice("right" if side == "left" else "left")
                np.testing.assert_array_equal(q[passive], np.array(plan["initial_q56"])[passive])
            changed = deepcopy(plan)
            changed["schedule"]["stages"][-1]["duration_s"] = 3
            with self.assertRaises(ValueError):
                verify_plan(changed)
            changed = deepcopy(plan)
            changed["accepted_place_q56"][side_slice(side).start] += 0.01
            with self.assertRaisesRegex(ValueError, "accepted Place"):
                verify_plan(changed)
            changed = deepcopy(plan["schedule"])
            changed["stages"][8]["arms"][side]["positions_rad"][0] += 0.01
            with self.assertRaisesRegex(ValueError, "Place arm"):
                validate_schedule(changed)

    def test_preview_cloud_immutable_attachment_and_release_fixed(self):
        row, plan = schedule_fixture()
        source = np.eye(4)
        source[:3, 3] = [0.4, 0.2, 0.1]
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_euler("x", 0.5).as_matrix()
        delta[:3, 3] = [0.1, -0.1, 0.1]
        points = np.arange(12).reshape(4, 3) / 100
        original = points.copy()
        plan.update(T_world_hand_source_held=source.tolist(), target=dict(Delta_T=delta.tolist()))
        robot = SimpleNamespace(fk=lambda *_: delta @ source)
        expected = points @ delta[:3, :3].T + delta[:3, 3]
        for _ in range(3):
            for phase in PHASES:
                result = preview_points(plan, robot, points, np.zeros(56), phase)
                np.testing.assert_allclose(result, points if phase in PHASES[:4] else expected)
        np.testing.assert_array_equal(points, original)

    def test_saved_schedules_without_new_waypoints_remain_readable(self):
        _, plan = schedule_fixture()
        schedule = plan["schedule"]
        for phases in (LEGACY_PHASES, PREVIOUS_PHASES):
            targets = {
                s["phase"]: {side: dict(zip(a["joint_names"], a["positions_rad"])) for side, a in s["arms"].items()}
                for s in schedule["stages"]
                if s["phase"] in phases
            }
            plan["schedule"] = build_schedule(schedule["plan"], named_state(np.array(plan["initial_q56"])), targets, {})
            plan["execution_phases"] = list(phases)
            self.assertNotIn("retreat_lift_height_m", plan["schedule"]["parameters"])
            verify_plan(plan)
            np.testing.assert_array_equal(preview(plan, 100), reference(plan, plan["schedule"]["stages"][-1], 2))
            plan["execution_phases"].insert(-1, "retreat_lift")
            with self.assertRaisesRegex(ValueError, "complete timed schedule"):
                verify_plan(plan)

    def test_new_ik_reuses_place_seed_and_rejects_colliding_branches(self):
        robot, static = FakeRobot(), FakeSim()
        held = np.zeros(56)
        held[28:34] = 0.2
        held[34:] = 0.1
        pose = np.eye(4)
        pose[0, 3] = 0.3
        checks = {}
        with patch.object(robot, "solve_arm_ik", wraps=robot.solve_arm_ik) as solve:
            q = solve_endpoint(robot, static, "right", pose, held, "preplace", lambda: False, checks)
        np.testing.assert_array_equal(solve.call_args_list[0].args[0], held)
        np.testing.assert_array_equal(q[:28], held[:28])
        np.testing.assert_array_equal(q[34:], held[34:])
        collision = dict(category="robot_table", sides=["right"], bodies=["right_J4_link", "table"], penetration_m=0.1)
        with patch.object(static, "check_static_contacts", return_value=[collision]):
            with self.assertRaisesRegex(ValueError, "retreat.*selected candidate retained"):
                solve_endpoint(robot, static, "right", pose, held, "retreat", lambda: False, {})

    def test_sampled_path_failure_and_cancellation_keep_stage_diagnostics(self):
        _, plan = schedule_fixture()
        robot, static, checks = FakeRobot(), FakeSim(), {}
        collision = dict(category="robot_table", sides=["right"], bodies=["right_J4_link", "table"], penetration_m=0.1)
        with patch.object(
            static, "check_static_contacts", side_effect=lambda q: [collision] if 0.26 < q[28] < 0.29 else []
        ):
            with self.assertRaisesRegex(ValueError, "transport: nominal path collision"):
                check_reference(robot, static, plan, lambda: False, checks)
        self.assertEqual(checks["phases"][-1]["phase"], "transport")
        self.assertEqual(checks["phases"][-1]["forbidden_contacts"], [collision])
        with self.assertRaisesRegex(RuntimeError, "canceled"):
            check_reference(robot, static, plan, lambda: True, {})

    def test_winner_only_preparation_exact_place_and_failure_no_reselection(self):
        row, _ = schedule_fixture()
        row["targets"]["right"]["pregrasp"]["T_world_hand"][0][3] += 0.04
        row = candidate_to_schema1(row)
        robot, static = FakeRobot(), FakeSim()
        lift_pose = np.eye(4)
        lift_pose[:3, :3] = Rotation.from_euler("z", 0.3).as_matrix()
        lift_pose[:3, 3] = [0.4, -0.2, 0.3]
        robot.fk = lambda q, side: lift_pose.copy()
        selection = jsonable(select_candidates(robot, static, [row], ["right"], np.zeros(56)))
        chosen = selection["selected"]
        self.assertIsNotNone(chosen)
        held = np.array(chosen["stages"]["squeeze"]["q56"])
        place = held.copy()
        place[28:34] = 0.25
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            robot.joint_names, robot.urdf_path, robot.directory = JOINT_NAMES, root / "robot.urdf", root
            for filename in ("robot.urdf", "manifest.json", "scene_config.yaml", "object/object.json", "table.json"):
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            write_json(root / "placement_selection_report.json", selection)
            endpoint = dict(
                arm="right",
                candidate_id=chosen["candidate_id"],
                target_revision=1,
                target_sha256="t",
                source_sha256="s",
                mode="right_full",
                references={},
                grasp_q56=held.tolist(),
                place_q56=place.tolist(),
                placement=dict(T_world_hand_place=np.eye(4).tolist(), T_world_hand_source_held=np.eye(4).tolist()),
                target=dict(Delta_T=np.eye(4).tolist(), table_plane_world=[0, 0, 1, 0]),
            )
            write_json(root / "placement_endpoint_preview.json", endpoint)
            engine = SimpleNamespace(
                root=root,
                robot=robot,
                static=static,
                config=dict(table_plane=str(root / "table.json"), lift_height_m=0.05),
            )
            frames = [np.zeros(28).tolist()] * 2 + [
                chosen["stages"][p]["q56"][28:] for p in ("pregrasp", "grasp", "squeeze")
            ]
            frames += [held[28:].tolist()]
            with (
                patch(
                    "ada_grasp_ctrl.pipeline.placement_execution.single_endpoints",
                    return_value=(np.zeros(56), None, None, None, {}, {}, PHASES[:5], frames),
                ) as initial_endpoints,
                patch("ada_grasp_ctrl.pipeline.placement_execution.solve_endpoint", wraps=solve_endpoint) as solve,
                patch(
                    "ada_grasp_ctrl.pipeline.placement_preview.select_candidates",
                    side_effect=AssertionError("reselection"),
                ),
            ):
                plan = prepare_execution(engine, endpoint, root, lambda: False)
                self.assertEqual(solve.call_count, 4)
                self.assertEqual(
                    [c.args[5] for c in solve.call_args_list], ["reorient", "preplace", "retreat_lift", "retreat"]
                )
                rotation_target = solve.call_args_list[0].args[3]
                np.testing.assert_array_equal(rotation_target[:3, 3], lift_pose[:3, 3])
                np.testing.assert_array_equal(rotation_target[:3, :3], np.eye(3))
                np.testing.assert_array_equal(solve.call_args_list[0].args[4], place)
                np.testing.assert_array_equal(plan["accepted_place_q56"], place)
                lift_stage = next(s for s in plan["schedule"]["stages"] if s["phase"] == "retreat_lift")
                np.testing.assert_array_equal(
                    solve.call_args_list[3].args[4], reference(plan, lift_stage, lift_stage["duration_s"])
                )
                verify_plan(plan)
                self.assertEqual(initial_endpoints.call_args.args[-1]["lift_height_m"], 0.15)
                self.assertEqual(plan["placement_lift_height_m"], 0.15)
                self.assertEqual(engine.config["lift_height_m"], 0.05)
                engine.config["simulation"] = dict(placement_lift_height_m=0.20)
                higher = prepare_execution(engine, endpoint, root, lambda: False)
                self.assertEqual(initial_endpoints.call_args.args[-1]["lift_height_m"], 0.20)
                self.assertEqual(higher["placement_lift_height_m"], 0.20)
                self.assertEqual(engine.config["lift_height_m"], 0.05)
                with patch.object(robot, "solve_arm_ik", return_value=(held, dict(accepted=False))):
                    with self.assertRaisesRegex(ValueError, "reorient"):
                        prepare_execution(engine, endpoint, root, lambda: False)

    def test_cli_requires_explicit_simulation_modifier(self):
        for options in ([], ["--placement-aware", "--environment", "mock"], ["--environment", "simulation"]):
            with patch("ada_grasp_ctrl.pipeline.simulation_ui.run_simulation", side_effect=AssertionError("startup")):
                self.assertEqual(
                    dual_runner.main(["--placement-execute", "--output-root", "/tmp/unused-phase3", *options]), 2
                )
        with patch("ada_grasp_ctrl.pipeline.simulation_ui.run_simulation", return_value=0) as run:
            self.assertEqual(
                dual_runner.main(
                    [
                        "--placement-aware",
                        "--placement-execute",
                        "--environment",
                        "simulation",
                        "--output-root",
                        "/tmp/unused-phase3",
                    ]
                ),
                0,
            )
            self.assertTrue(run.call_args.args[0].placement_execute)


class PlacementPhysicsTests(unittest.TestCase):
    setUpClass = classmethod(simulation_tests.SceneContracts.setUpClass.__func__)
    tearDownClass = classmethod(simulation_tests.SceneContracts.tearDownClass.__func__)
    scene = simulation_tests.SceneContracts.scene

    def test_real_executor_completes_retreat_continues_physics_and_never_evaluates_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _, _ = self.scene(root)
            plan = idle_reference(scene, ["right"])
            q = scene.sim.configuration()
            targets = {p: {"right": named_state(q)["right"]} for p in PHASES}
            plan.update(
                workflow="grasp_to_place",
                execution_phases=list(PHASES),
                accepted_place_q56=q.tolist(),
                schedule=build_schedule(plan["schedule"]["plan"], named_state(q), targets, {}),
            )
            executor = SimulationExecutor(scene, plan, self.config, root)
            initial_pose = scene.sim.data.joint("object_free").qpos.copy()
            scene.sim.data.joint("object_free").qvel[0] = 0.01
            tick, after = executor.tick, []

            def record(*args, **kwargs):
                result = tick(*args, **kwargs)
                if executor.result["pipeline_completed"]:
                    after.append(scene.sim.data.time)
                    if len(after) == 3:
                        executor.cancel.set()
                return result

            with patch.object(executor, "tick", side_effect=record):
                result = executor.run(realtime=False)
            self.assertEqual([e["phase"] for e in result["phase_events"]], list(PHASES))
            self.assertEqual(len(after), 3)
            self.assertTrue(result["pipeline_completed"])
            self.assertIsNone(result["scheduled_hold_success"])
            self.assertIsNone(result["post_hold_retained"])
            self.assertEqual(result["hold_evaluation"], "not_evaluated_no_hold_stage")
            self.assertNotIn("success_criterion", result)
            self.assertIsNone(result["last_observation"]["retained"])
            self.assertFalse(np.array_equal(initial_pose, scene.sim.data.joint("object_free").qpos))
            self.assertEqual(scene.sim.model.joint("object_free").type[0], 0)
            for event in result["phase_events"]:
                self.assertAlmostEqual(
                    event["end_s"] - event["start_s"], np.ceil((event["scheduled_total_s"] - 1e-9) * 25) / 25, places=7
                )
            # The executor samples the exact reference, including the final endpoint.
            np.testing.assert_array_equal(result["last_observation"]["reference_q56"], preview(plan, 100))
