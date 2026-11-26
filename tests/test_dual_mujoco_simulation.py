"""ROS-free scene, observation, shared reference and real MuJoCo executor contracts."""

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import mujoco
import numpy as np

from ada_grasp_ctrl.calibration import calibrated_snapshot
from ada_grasp_ctrl.experiments.simulated_object_cloud import ObjectScene, body_pose
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash
from ada_grasp_ctrl.grasp_selection.contracts import plan_from_candidate
from ada_grasp_ctrl.pipeline import dual_runner
from ada_grasp_ctrl.pipeline.observation import read_observation
from ada_grasp_ctrl.pipeline.paired_artifacts import named_state
from ada_grasp_ctrl.pipeline.shared_timing import build_stage_schedule
from ada_grasp_ctrl.pipeline.simulation_executor import SimulationExecutor
from ada_grasp_ctrl.pipeline.simulation_plan import (
    PHASES,
    adjust_squeeze_candidates,
    prepare_plan,
    reference,
    preview,
    verify_plan,
)
from ada_grasp_ctrl.pipeline.simulation_scene import SceneBuilder
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.placement.workflow import prepare_simulation_session
from ada_grasp_ctrl.robot_environment.model import side_slice
from test_grasp_contracts import canonical


def idle_reference(scene, arms):
    """Explicit unsuccessful fixture: stationary arms never lift the real object."""
    q = scene.sim.configuration()
    candidate = canonical(arms, "both_full" if len(arms) == 2 else arms[0] + "_full")
    named = named_state(q)
    for side in arms:
        for phase in ("pregrasp", "grasp", "squeeze"):
            candidate["targets"][side][phase]["joint_positions_rad"] = {
                key: value for key, value in named[side].items() if key not in [f"{side}_J{i}" for i in range(1, 7)]
            }
    endpoints = {phase: {side: dict(named[side]) for side in arms} for phase in (*PHASES, "release", "return")}
    for phase in ("release", "return"):
        for side in arms:
            for name in list(named[side])[6:]:
                endpoints[phase][side][name] = 0.0
    schedule = build_stage_schedule(
        plan_from_candidate(candidate, available_arms=["left", "right"]), named, endpoints, hold_s=0.1
    )
    return dict(
        reference_hz=25,
        stage_completion="timed",
        kind="ada_simulation_reference",
        environment="simulation",
        schedule=schedule,
        execution_phases=list(PHASES),
        initial_q56=q.tolist(),
        references={},
        source_mode=candidate["mode"],
        execution_mode=candidate["mode"],
        candidate_id="unsuccessful-fixture",
        active_arms=list(arms),
        effort=0.0,
    )


class SceneContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.config = dual_runner.configuration("src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml")
        if not Path(cls.config["robot_snapshot"]).is_dir():
            raise unittest.SkipTest("Local canonical snapshot unavailable")
        cls.config["robot_snapshot"] = str(calibrated_snapshot(cls.config["robot_snapshot"], cls.root / "robot"))
        cls.builder = SceneBuilder(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene(self, root):
        return self.builder.confirm(self.builder.defaults(), root)

    def test_object_only_gravity_matches_historical_force_and_free_fall(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, _, _ = self.scene(Path(directory))
            config = deepcopy(scene.config)
            np.testing.assert_allclose(
                scene.metadata["gravity_environment"]["object_force_world_n"], [0, 0, -1], atol=1e-12, rtol=0
            )
            self.assertFalse(scene.metadata["gravity_environment"]["arm_bias_compensation"])
            self.assertTrue(scene.sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
            for mass in (0.1, 0.2):
                with self.subTest(mass=mass), tempfile.TemporaryDirectory() as other:
                    config["object"]["physics"]["mass_kg"] = mass
                    falling = ObjectScene(config, Path(other))
                    sim = falling.sim
                    # Far from robot/table contacts, test physical acceleration and no double gravity.
                    sim.data.joint("object_free").qpos[:3] = [3, 3, 2]
                    sim.data.qvel[:] = 0
                    sim.data.qfrc_applied[sim.qvel[np.r_[0:6, 28:34]]] = 1
                    for _ in range(10):
                        falling.apply_forces()
                        sim.step()
                    np.testing.assert_allclose(
                        sim.data.joint("object_free").qvel[:3], [0, 0, -100 * sim.model.opt.timestep], atol=1e-10
                    )
                    expected = np.zeros_like(sim.data.xfrc_applied)
                    expected[sim.model.body("object").id, 2] = -mass * 10
                    np.testing.assert_allclose(sim.data.xfrc_applied, expected)
                    np.testing.assert_array_equal(sim.data.qfrc_applied[sim.qvel], 0)
                    sim.reset(sim.targets)
                    falling.apply_forces()
                    np.testing.assert_allclose(sim.data.xfrc_applied, expected)
            config["scene"].pop("gravity_mode")
            with tempfile.TemporaryDirectory() as other:
                legacy = ObjectScene(config, Path(other))
                self.assertFalse(legacy.sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
                legacy.apply_forces()
                np.testing.assert_array_equal(legacy.sim.data.xfrc_applied, 0)

    def test_execution_preserves_object_force_and_rejects_changed_force_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _, _ = self.scene(root)
            executor = SimulationExecutor(scene, idle_reference(scene, ["right"]), self.config, root)
            stage = executor.plan["schedule"]["stages"][0]
            expected = scene.sim.data.xfrc_applied.copy()
            scene.sim.data.xfrc_applied[:] = 0
            executor.tick(stage, 0, io.StringIO())
            np.testing.assert_array_equal(scene.sim.data.xfrc_applied, expected)
            np.testing.assert_array_equal(scene.sim.data.qfrc_applied[scene.sim.qvel], 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _, _ = self.scene(root)
            scene.sim.data.xfrc_applied[scene.sim.model.body("object").id, 2] = 0
            with self.assertRaisesRegex(ValueError, "Settled xfrc_applied changed"):
                SimulationExecutor(scene, idle_reference(scene, ["right"]), self.config, root)

    def test_reviewed_control_rate_and_legacy_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _, _ = self.scene(root)
            plan = idle_reference(scene, ["right"])
            executor = SimulationExecutor(scene, plan, self.config, root)
            self.assertEqual(executor.substeps, 10)
            bench_model = scene.sim.model
            self.assertEqual(scene.metadata["contact_profile"]["name"], "dexgraspbench")
            start = scene.sim.data.time
            executor.tick(plan["schedule"]["stages"][0], 0, io.StringIO())
            self.assertAlmostEqual(scene.sim.data.time - start, 0.04)
            for hz in (0, -1, float("nan"), float("inf")):
                with self.subTest(hz=hz), self.assertRaisesRegex(ValueError, "reference_hz"):
                    verify_plan({**plan, "reference_hz": hz})
            plan["reference_hz"] = 100
            with self.assertRaisesRegex(ValueError, "must divide"):
                SimulationExecutor(scene, plan, self.config, root)
        config = deepcopy(self.config)
        config["simulation"]["contact_profile"] = "source"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = SceneBuilder(config)
            scene, _, _ = builder.confirm(builder.defaults(), root)
            plan = idle_reference(scene, ["right"])
            plan.pop("reference_hz")
            executor = SimulationExecutor(scene, plan, config, root)
            self.assertEqual(executor.reference_hz, 100)
            self.assertEqual(executor.substeps, 5)
            source_model = scene.sim.model
            for field in (
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_forcerange",
                "actuator_ctrlrange",
                "dof_damping",
                "dof_armature",
                "body_mass",
                "body_inertia",
                "geom_size",
                "geom_pos",
            ):
                np.testing.assert_array_equal(getattr(bench_model, field), getattr(source_model, field), err_msg=field)
            for geom_id in range(bench_model.ngeom):
                body_name = bench_model.body(int(bench_model.geom_bodyid[geom_id])).name
                geom = bench_model.geom(geom_id)
                np.testing.assert_array_equal(geom.friction, [0.8, 0.02, 0.0001])
                self.assertEqual(geom.condim, 4)
                if body_name.startswith(("left_", "right_")):
                    np.testing.assert_array_equal(geom.solref, [0.005, 1])
                    np.testing.assert_array_equal(geom.solimp[:3], [0.5, 0.99, 0.0001])
                else:
                    np.testing.assert_array_equal(geom.solref, source_model.geom(geom_id).solref)
                    np.testing.assert_array_equal(geom.solimp, source_model.geom(geom_id).solimp)

    def test_scale_changes_support_height_and_stays_within_manifest_and_table(self):
        for key, entry in self.builder.entries.items():
            for scale in entry["scale_range"]:
                request = self.builder.tabletop_request(key, [10, -10], 47, 4, scale=scale)
                pose, vertices = self.builder.pose(request)
                self.builder.check_footprint(vertices)
                self.assertEqual(request["scale"], scale)
                heights = vertices @ np.array(self.builder.plane["normal_xyz"]) + self.builder.plane["d"]
                self.assertAlmostEqual(heights.min(), 0.001)
                local = (vertices - pose[:3, 3]) @ pose[:3, :3]
                np.testing.assert_allclose(local, self.builder.meshes[key].vertices * scale, atol=1e-12)
            for bad in (0, 100, float("nan")):
                with self.assertRaises(ValueError):
                    self.builder.tabletop_request(key, [0.45, -0.45], 0, 4, scale=bad)

    def test_settling_cancellation_never_generates_observation(self):
        import threading

        with tempfile.TemporaryDirectory() as directory:
            cancel = threading.Event()
            cancel.set()
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "settling canceled"):
                self.builder.confirm(self.builder.defaults(), root, cancel)
            self.assertFalse((root / "object/object.npz").exists())

    def test_settling_timeout_keeps_motion_diagnostics_without_generating_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = dict(timeout_s=0.2, window_s=0.05, position_span_m=1e-12, orientation_span_deg=1e-12)
            with patch.dict(self.builder.config, {"simulation": {"settling": policy}}):
                with self.assertRaisesRegex(ValueError, "position span.*rotation span.*table support"):
                    self.builder.confirm(self.builder.defaults(), root)
            report = json.loads((root / "settling_report.json").read_text())
            self.assertEqual(report["status"], "timeout")
            self.assertAlmostEqual(report["elapsed_s"], 0.2)
            self.assertIn("linear_speed_m_s", report)
            self.assertIn("angular_speed_rad_s", report)
            self.assertFalse((root / "object/object.npz").exists())

    def test_support_determinism_scale_footprint_and_local_manifest(self):
        b = self.builder
        self.assertEqual(len(b.entries), 10)
        n = np.array(b.plane["normal_xyz"])
        for index, key in enumerate(b.entries):
            req = b.defaults(key)
            req["rpy_deg"] = [31, -42, 17]
            pose, vertices = b.pose(req)
            self.assertAlmostEqual(np.min(vertices @ n) + b.plane["d"], 0.001, places=12)
            np.testing.assert_allclose(vertices, b.meshes[key].vertices * req["scale"] @ pose[:3, :3].T + pose[:3, 3])
            self.assertEqual(file_hash(Path(b.entries[key]["mesh"])), b.entries[key]["mesh_sha256"])
        req = b.defaults()
        req["xy_m"] = [100, 100]
        with self.assertRaisesRegex(ValueError, "beyond"):
            b.check_footprint(b.pose(req)[1])
        req["scale"] = float("nan")
        with self.assertRaises(ValueError):
            b.pose(req)

    def test_tabletop_yaw_preserves_support_and_fits_edges(self):
        from scipy.spatial.transform import Rotation

        b = self.builder
        n = np.array(b.plane["normal_xyz"])
        for key, entry in b.entries.items():
            baseline = b.pose(b.defaults(key))[0][:3, :3]
            for yaw in (-137, 0, 83, 180):
                for xy in ([0.5, -0.5], [-10, -10], [10, 10], [-10, 10], [10, -10]):
                    with self.subTest(key=key, yaw=yaw, xy=xy):
                        req = b.tabletop_request(key, xy, yaw, 4)
                        pose, vertices = b.pose(req)
                        b.check_footprint(vertices)
                        self.assertEqual(req["scale"], entry["default_scale"])
                        self.assertAlmostEqual(np.min(vertices @ n) + b.plane["d"], 0.001, places=12)
                        expected = Rotation.from_rotvec(n * np.deg2rad(yaw)).as_matrix() @ baseline
                        np.testing.assert_allclose(pose[:3, :3], expected, atol=1e-12)
                        if xy == [0.5, -0.5]:
                            self.assertEqual(req["xy_m"], xy)
        for seed in range(100):
            req = b.random_tabletop_request(seed)
            self.assertEqual(req, b.random_tabletop_request(seed))
            b.check_footprint(b.pose(req)[1])
            self.assertEqual(req["scale"], b.entries[req["object_id"]]["default_scale"])

    def test_all_presets_stay_supported_after_actual_settling(self):
        from scipy.spatial.transform import Rotation

        for key in self.builder.entries:
            for yaw, xy in ((0, [0.45, -0.45]), (-137, [0.5, -0.5])):
                with self.subTest(key=key, yaw=yaw), tempfile.TemporaryDirectory() as directory:
                    req = self.builder.tabletop_request(key, xy, yaw, 4)
                    expected = self.builder.pose(req)[0]
                    scene, _, _ = self.builder.confirm(req, Path(directory))
                    # A short settling window alone can miss slow rolling after confirmation.
                    for step in range(round(5 / scene.sim.model.opt.timestep)):
                        scene.apply_forces()
                        scene.sim.step()
                        if step % 100:
                            continue
                        actual = body_pose(scene.sim)
                        drift = Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude()
                        self.assertLess(np.rad2deg(drift), 3)
                        self.assertLess(np.linalg.norm(actual[:3, 3] - expected[:3, 3]), 0.01)
                        vertices = scene.mesh.vertices @ actual[:3, :3].T + actual[:3, 3]
                        self.builder.check_footprint(vertices)
                        heights = vertices @ np.array(self.builder.plane["normal_xyz"]) + self.builder.plane["d"]
                        self.assertGreaterEqual(heights.min(), -0.002)

    def test_editor_exposes_xy_yaw_scale_and_keeps_supported_requests(self):
        from ada_grasp_ctrl.pipeline.simulation_ui import SimulationApp

        controls = {}

        def control(label, **kwargs):
            handle = MagicMock()
            handle.value = kwargs.get("initial_value")
            controls[label] = handle
            return handle

        server = MagicMock()
        for method in ("add_slider", "add_dropdown", "add_number", "add_button"):
            getattr(server.gui, method).side_effect = control
        args = SimpleNamespace(placement_only=False)
        with (
            patch("ada_grasp_ctrl.pipeline.simulation_ui.ViserMujocoScene"),
            patch("ada_grasp_ctrl.pipeline.simulation_ui.threading.Thread"),
        ):
            app = SimulationApp(server, self.builder, args, self.root)
        self.assertTrue({"Object", "X (m)", "Y (m)", "Yaw (deg)", "Scale"}.issubset(controls))
        self.assertTrue({"Roll (deg)", "Pitch (deg)"}.isdisjoint(controls))
        app.choice.value = list(self.builder.entries)[8]  # Book regression.
        app.yaw.value = 83
        app.xy[0].value = 10
        app.edit()
        self.assertEqual(app.requested["yaw_deg"], 83)
        self.assertEqual(app.requested["xy_m"], [c.value for c in app.xy])
        self.builder.check_footprint(self.builder.pose(app.requested)[1])
        app.scale.value = 0.30
        app.edit()
        self.assertEqual(app.requested["scale"], 0.30)
        pose, vertices = self.builder.pose(app.requested)
        self.builder.check_footprint(vertices)
        heights = vertices @ np.array(self.builder.plane["normal_xyz"]) + self.builder.plane["d"]
        self.assertAlmostEqual(heights.min(), 0.001)
        app.choice.value = list(self.builder.entries)[0]
        app.edit()
        self.assertEqual(app.scale.value, self.builder.entries[app.choice.value]["default_scale"])
        app.random_scene()
        self.assertEqual(app.requested, self.builder.random_tabletop_request(app.seed.value))
        app.reset_controls()
        self.assertEqual(app.requested, self.builder.defaults(app.choice.value))

    def test_settled_complete_cloud_and_no_oracle_observation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scene, obj, state = self.scene(root)
            info, points, camera = read_observation(obj)
            self.assertGreater(scene.sim.data.time, 0)
            self.assertEqual(points.shape, (8192, 3))
            self.assertEqual(camera, {})
            self.assertEqual(state["full_joint_names"], list(scene.robot.joint_names))
            again, _ = scene.cloud()
            np.testing.assert_array_equal(points, again)
            self.assertNotIn("extrinsics", info)
            self.assertNotIn("T_world_object", info)
            self.assertEqual(scene.sim.model.ncam, 0)
            self.assertEqual(scene.sim.model.opt.timestep, 0.004)
            settling = json.loads((root / "settling_report.json").read_text())
            self.assertEqual(settling["status"], "settled")
            self.assertGreaterEqual(settling["observed_window_s"] + 1e-9, 0.5)
            self.assertLessEqual(settling["position_span_m"], 0.003)
            self.assertLessEqual(settling["orientation_span_deg"], 2.0)
            self.assertTrue(settling["supported_throughout_window"])
            placement = prepare_simulation_session(obj, root / "placement", state)
            np.testing.assert_array_equal(placement.points, points)
            np.testing.assert_array_equal(
                placement.preview_plane, [*self.builder.plane["normal_xyz"], self.builder.plane["d"]]
            )
            from test_placement_selection import candidate

            placement.infer_once(lambda: [candidate(1, [1, 0, 0])])
            grounded = placement.select("P01")
            self.assertAlmostEqual(np.min(grounded @ placement.preview_plane[:3] + placement.preview_plane[3]), 0)
            np.testing.assert_array_equal(placement.select("P00"), points)
            saved = json.loads(placement.confirm().read_text())
            self.assertEqual(saved["translation_status"], "undetermined")
            self.assertEqual(saved["candidate_id"], "P00")
            self.assertNotIn("T_world_color", saved["frame_convention"])
            with np.load(obj / "object.npz") as data:
                self.assertEqual(data.files, ["points_world_m"])
            write_json(obj / "object.json", {**info, "object_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_observation(obj)

    def test_static_scene_matches_robot_fk_without_mutating_physics_or_exposing_object(self):
        from ada_grasp_ctrl.pipeline.dual_scene import live_scene

        with tempfile.TemporaryDirectory() as d:
            scene, _, state = self.scene(Path(d))
            robot, static = live_scene(scene.robot.directory, state, self.builder.plane)
            before = scene.sim.data.qpos.copy()
            q = np.array(state["full_positions_rad"])
            static.check_static_contacts(q)
            np.testing.assert_array_equal(before, scene.sim.data.qpos)
            self.assertNotIn("object", static.body_labels.values())
            for side in ("left", "right"):
                expected = robot.fk(q, side)
                body = scene.sim.data.body(side + "_hand_flange")
                np.testing.assert_allclose(expected[:3, 3], body.xpos, atol=1e-10)
                np.testing.assert_allclose(expected[:3, :3], body.xmat.reshape(3, 3), atol=1e-10)
            plan = idle_reference(scene, ["left", "right"])
            scene.sim.data.joint("object_free").qpos[0] += 0.001
            with self.assertRaisesRegex(ValueError, "Settled qpos changed"):
                SimulationExecutor(scene, plan, self.config, Path(d))

    def test_real_physics_single_paired_failure_prefix_and_passive_targets(self):
        for arms in (["left"], ["right"], ["left", "right"]):
            with self.subTest(arms=arms), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                scene, _, _ = self.scene(root)
                sim = scene.sim
                plan = idle_reference(scene, arms)
                executor = SimulationExecutor(scene, plan, self.config, root)
                initial_targets = sim.targets.copy()
                initial_pose = body_pose(sim).copy()
                sample = executor.force_source.sample
                sampled_phases = set()
                coherent_samples = []

                def measured_sample():
                    snapshot = sample()
                    # Assert outside the display boundary, which intentionally catches failures.
                    coherent_samples.append(
                        not snapshot.errors
                        and snapshot.timestamp == executor.latest["time_s"]
                        and np.array_equal(sim.configuration(), executor.latest["q56"])
                        and len(snapshot.fingers) == 10
                    )
                    sampled_phases.add(executor.latest["phase"])
                    return snapshot

                with (
                    patch.object(sim, "reset", side_effect=AssertionError("execution reset/teleport")),
                    patch.object(executor.force_source, "sample", side_effect=measured_sample),
                ):
                    result = executor.run(realtime=False, continue_hold=False)
                self.assertTrue(result["pipeline_completed"], result)
                self.assertFalse(result["scheduled_hold_success"])
                self.assertEqual([e["phase"] for e in result["phase_events"]], list(PHASES))
                self.assertGreater(sim.data.time, 10)
                self.assertFalse(np.array_equal(body_pose(sim), initial_pose))
                for side in set(("left", "right")) - set(arms):
                    np.testing.assert_array_equal(sim.targets[side_slice(side)], initial_targets[side_slice(side)])
                self.assertTrue((root / "physics.jsonl").stat().st_size > 0)
                self.assertEqual(sampled_phases, set(PHASES))
                self.assertTrue(coherent_samples and all(coherent_samples))
                self.assertIsNone(executor.force_snapshot)
                self.assertNotIn("force_snapshot", executor.latest)

    def test_force_sampling_failure_does_not_stop_physics_or_hide_safety_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _, _ = self.scene(root)
            executor = SimulationExecutor(scene, idle_reference(scene, ["right"]), self.config, root)
            stage = executor.plan["schedule"]["stages"][0]
            before = scene.sim.data.time
            with patch.object(executor.force_source, "sample", side_effect=RuntimeError("force-only failure")):
                executor.tick(stage, 0, io.StringIO())
                self.assertGreater(scene.sim.data.time, before)
                self.assertTrue(all(r.status == "UNAVAILABLE" for r in executor.force_snapshot.fingers))
                self.assertIn("force-only failure", executor.force_snapshot.errors[0])
                with patch(
                    "ada_grasp_ctrl.pipeline.simulation_executor.forbidden_contacts", return_value=["collision"]
                ):
                    with self.assertRaisesRegex(ValueError, "live_collision"):
                        executor.tick(stage, 0, io.StringIO())
            executor.tick(stage, 0, io.StringIO())
            self.assertFalse(executor.force_snapshot.errors)

    def test_timed_progression_ignores_arrival_but_keeps_fixed_dwell_and_failure_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _, _ = self.scene(root)
            plan = idle_reference(scene, ["right"])
            executor = SimulationExecutor(scene, plan, self.config, root)
            with patch("ada_grasp_ctrl.pipeline.simulation_executor.arrival_reached", return_value=False):
                result = executor.run(realtime=False, continue_hold=False)
            self.assertTrue(result["pipeline_completed"], result)
            self.assertFalse(result["scheduled_hold_success"])
            self.assertEqual(result["stage_completion"], "timed")
            for event in result["phase_events"]:
                elapsed = event["end_s"] - event["start_s"]
                self.assertGreaterEqual(elapsed + 1e-9, event["scheduled_total_s"])
                self.assertLess(elapsed, event["scheduled_total_s"] + executor.reference_interval_s + 1e-6)
                self.assertFalse(event["arrival_reached"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _, _ = self.scene(root)
            executor = SimulationExecutor(scene, idle_reference(scene, ["right"]), self.config, root)
            with patch("ada_grasp_ctrl.pipeline.simulation_executor.forbidden_contacts", return_value=["collision"]):
                result = executor.run(realtime=False, continue_hold=False)
            self.assertEqual(result["error"], "live_collision")
            self.assertFalse(result["pipeline_completed"])

    def test_legacy_plan_preserves_arrival_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _, _ = self.scene(root)
            plan = idle_reference(scene, ["right"])
            plan.pop("stage_completion")
            executor = SimulationExecutor(scene, plan, self.config, root)
            with patch("ada_grasp_ctrl.pipeline.simulation_executor.arrival_reached", return_value=False):
                result = executor.run(realtime=False, continue_hold=False)
            self.assertEqual(result["error"], "stage_arrival_timeout")
            self.assertEqual(result["stage_completion"], "arrival")

    def test_both_targets_before_same_step_preview_is_pure_and_cancel_hold(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scene, _, _ = self.scene(root)
            sim = scene.sim
            plan = idle_reference(scene, ["left", "right"])
            executor = SimulationExecutor(scene, plan, self.config, root)
            original = sim.data.qpos.copy()
            preview(plan, 42)
            np.testing.assert_array_equal(original, sim.data.qpos)
            stage = plan["schedule"]["stages"][0]
            for side in ("left", "right"):
                stage["arms"][side]["positions_rad"][0] += 0.005
            expected = reference(plan, stage, stage["duration_s"])
            step = sim.step

            def check_step(*args):
                for side in ("left", "right"):
                    np.testing.assert_array_equal(sim.targets[side_slice(side)], expected[side_slice(side)])
                step(*args)

            with patch.object(sim, "step", side_effect=check_step):
                executor.tick(stage, stage["duration_s"], io.StringIO())
            stage["duration_s"] += 1
            # Invalid reference changes cannot pass review.
            with self.assertRaises(ValueError):
                verify_plan(plan)

    def test_hold_continues_until_cancellation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scene, _, _ = self.scene(root)
            executor = SimulationExecutor(scene, idle_reference(scene, ["left", "right"]), self.config, root)
            tick = executor.tick
            after = []

            def cancel_after_hold(*args, **kwargs):
                value = tick(*args, **kwargs)
                if kwargs.get("post_hold"):
                    self.assertIsNotNone(executor.force_snapshot)
                    self.assertEqual(executor.force_snapshot.timestamp, scene.sim.data.time)
                    after.append(scene.sim.data.time)
                    if len(after) == 3:
                        executor.cancel.set()
                return value

            with patch.object(executor, "tick", side_effect=cancel_after_hold):
                result = executor.run(realtime=False)
            self.assertEqual(len(after), 3)
            self.assertTrue(result["pipeline_completed"])
            self.assertFalse(result["scheduled_hold_success"])
            self.assertTrue(result["post_hold_retention_failure_latched"])
            self.assertFalse(result["post_hold_drop_latched"])
            self.assertEqual(result["status"], "canceled")
            self.assertIsNone(executor.force_snapshot)


class IsolationContracts(unittest.TestCase):
    def test_squeeze_multiplier_matches_bench_for_single_mirrored_and_paired_targets(self):
        from scipy.spatial.transform import Rotation

        for arms in (["right"], ["left"], ["left", "right"]):
            row = canonical(arms, "both_full" if len(arms) == 2 else arms[0] + "_full")
            for side in arms:
                target = row["targets"][side]
                target["grasp"]["T_world_hand"] = np.eye(4).tolist()
                squeeze = np.eye(4)
                squeeze[:3, :3] = Rotation.from_euler("z", 0.2).as_matrix()
                squeeze[:3, 3] = [0.01, -0.02, 0.03]
                target["squeeze"]["T_world_hand"] = squeeze.tolist()
                for name in target["grasp"]["joint_positions_rad"]:
                    target["grasp"]["joint_positions_rad"][name] = 0.1
                    target["squeeze"]["joint_positions_rad"][name] = 0.15
            original = deepcopy(row)
            result = adjust_squeeze_candidates([row], 2.0)[0]
            self.assertEqual(row, original)
            for side in arms:
                target = result["targets"][side]
                self.assertEqual(target["pregrasp"], row["targets"][side]["pregrasp"])
                self.assertEqual(target["grasp"], row["targets"][side]["grasp"])
                np.testing.assert_allclose(np.array(target["squeeze"]["T_world_hand"])[:3, 3], [0.02, -0.04, 0.06])
                np.testing.assert_array_equal(np.array(target["squeeze"]["T_world_hand"])[:3, :3], squeeze[:3, :3])
                np.testing.assert_allclose(list(target["squeeze"]["joint_positions_rad"].values()), 0.2)
            with self.assertRaisesRegex(ValueError, "already"):
                adjust_squeeze_candidates([result], 2.0)
            self.assertEqual(adjust_squeeze_candidates([row], 1.0)[0]["targets"], row["targets"])
        for bad in (0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                adjust_squeeze_candidates([], bad)

    def test_adjustment_precedes_ik_and_selection_and_retains_raw_predictions(self):
        from ada_grasp_ctrl.grasp_selection.contracts import candidate_to_schema1

        row = candidate_to_schema1(canonical())
        name = next(iter(row["targets"]["right"]["squeeze"]["joint_positions_rad"]))
        row["targets"]["right"]["grasp"]["joint_positions_rad"][name] = 0.1
        row["targets"]["right"]["squeeze"]["joint_positions_rad"][name] = 0.15
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "table.json", {})
            write_json(root / "candidates.json", {"candidates": [row]})
            config = dict(robot_snapshot=directory, table_plane=str(root / "table.json"), enabled_arms=["right"])
            with (
                patch("ada_grasp_ctrl.pipeline.simulation_plan.infer_live_sides", return_value=[row]),
                patch("ada_grasp_ctrl.pipeline.simulation_plan.live_scene", return_value=(None, None)),
                patch(
                    "ada_grasp_ctrl.pipeline.simulation_plan.select_candidates", return_value={"selected": None}
                ) as select,
            ):
                self.assertIsNone(prepare_plan(config, "minimum", root, None, {"full_positions_rad": [0] * 56}))
            adjusted = select.call_args.args[2][0]
            self.assertAlmostEqual(adjusted["targets"]["right"]["squeeze"]["joint_positions_rad"][name], 0.2)
            self.assertEqual(json.loads((root / "candidates.json").read_text())["candidates"], [row])
            self.assertEqual(json.loads((root / "simulation_candidates.json").read_text())["candidates"], [adjusted])

    def test_cli_branches_before_all_ros_and_capture_paths(self):
        with tempfile.TemporaryDirectory() as d:
            for extra in ([], ["--placement-only"]):
                with (
                    patch("ada_grasp_ctrl.pipeline.simulation_ui.run_simulation", return_value=0) as run,
                    patch.object(dual_runner, "validate_domain", side_effect=AssertionError("ROS domain")),
                    patch.object(dual_runner, "Processes", side_effect=AssertionError("ROS processes")),
                    patch.object(dual_runner, "perceive", side_effect=AssertionError("camera")),
                ):
                    result = dual_runner.main(["--environment", "simulation", "--output-root", d, *extra])
                    self.assertEqual(result, 0)
                    run.assert_called_once()
            for extra in (["--ros-domain-id", "88"], ["--roi", "0", "0", "20", "20"], ["--cluster", "0"]):
                self.assertEqual(dual_runner.main(["--environment", "simulation", "--output-root", d, *extra]), 2)

    def test_legacy_selected_candidate_gets_canonical_mode_only_at_schedule_boundary(self):
        from types import SimpleNamespace
        from ada_grasp_ctrl.pipeline.simulation_plan import build_plan
        from ada_grasp_ctrl.robot_environment.model import JOINT_NAMES

        for side in ("left", "right"):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "object").mkdir()
                row = canonical([side], side + "_full")
                row.pop("mode")
                selection = dict(
                    enabled_arms=["left", "right"],
                    selected=dict(
                        candidate=row,
                        candidate_id=row["candidate_id"],
                        required_arms=[side],
                        selected_side=side,
                        effort=0.0,
                    ),
                )
                q = np.zeros(56)
                names = list(named_state(q)[side])[6:]
                fingers = [row["targets"][side]["grasp"]["joint_positions_rad"][n] for n in names]
                target = [0.0] * 6 + fingers
                frames = [[0.0] * 28, [0.0] * 28, target, target, target, target, target, [0.0] * 28, [0.0] * 28]
                write_json(root / "robot_state.json", dict(full_positions_rad=q.tolist()))
                for name in (
                    "scene.mjb",
                    "settled_state.npz",
                    "selection.json",
                    "scene_config.yaml",
                    "model.urdf",
                    "manifest.json",
                    "table.json",
                    "object/object.npz",
                    "object/object.json",
                ):
                    (root / name).write_text("fixture")
                robot = SimpleNamespace(joint_names=JOINT_NAMES, urdf_path=root / "model.urdf", directory=root)
                with patch(
                    "ada_grasp_ctrl.pipeline.simulation_plan.single_endpoints",
                    return_value=(q, None, None, None, {}, {}, [*PHASES, "release", "return"], frames),
                ):
                    result = build_plan(
                        robot, None, selection, root, dict(hold_s=0.1, table_plane=str(root / "table.json"))
                    )
                self.assertEqual(result["schedule"]["plan"]["mode"], side + "_full")
                self.assertNotIn("mode", row)
                verify_plan(result)

    def test_offline_process_cancel_and_ros_rejection(self):
        from ada_grasp_ctrl.pipeline.simulation_processes import SimulationProcesses

        with tempfile.TemporaryDirectory() as d:
            p = SimulationProcesses(Path(d))
            with self.assertRaises(ValueError):
                p.run(["python", "-m", "rclpy"], "bad", ros=True)
            p.close()
            with self.assertRaisesRegex(RuntimeError, "canceled"):
                p.run(["unused"], "canceled")


if __name__ == "__main__":
    unittest.main()
