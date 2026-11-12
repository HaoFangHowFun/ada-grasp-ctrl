"""Real bottle placement, lossless scene settings, and the existing pipeline contract."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from ada_grasp_ctrl.paths import project_root, resolve_from_root
from ada_grasp_ctrl.experiments.object_placement_preview import ObjectPlacementPreview
from ada_grasp_ctrl.experiments.simulated_object_cloud import ObjectScene, body_pose


@unittest.skipUnless(os.environ.get("ADA_DUAL_ARM_SNAPSHOT"), "Set ADA_DUAL_ARM_SNAPSHOT to real exported assets")
class PlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = yaml.safe_load((project_root() / "config/simulation_scene.yaml").read_text())
        self.config["scene"]["robot_snapshot"] = os.environ["ADA_DUAL_ARM_SNAPSHOT"]
        self.source = self.root / "source.yaml"

    def editor(self, config: dict | None = None, destination: Path | None = None) -> ObjectPlacementPreview:
        """Use the actual Phase 2 loader and source assets in a fresh workspace."""
        self.source.write_text(yaml.safe_dump(config or self.config))
        workspace = self.root / f"workspace_{len(list(self.root.iterdir()))}"
        workspace.mkdir()
        return ObjectPlacementPreview(self.source, workspace, destination)

    def test_mesh_scale_origin_and_compiled_render_geometry(self) -> None:
        editor = self.editor()
        scene = editor.scene
        source = trimesh.load(resolve_from_root(self.config["object"]["mesh"]), force="mesh", process=False)
        expected = source.vertices * np.array(self.config["object"]["scale"])
        np.testing.assert_allclose(scene.mesh.vertices, expected, atol=1e-12)
        sim = scene.sim
        geom = sim.model.geom("object_visual")
        mesh_id = geom.dataid[0]
        start = sim.model.mesh_vertadr[mesh_id]
        count = sim.model.mesh_vertnum[mesh_id]
        vertices = sim.model.mesh_vert[start : start + count]
        world = vertices @ sim.data.geom("object_visual").xmat.reshape(3, 3).T + sim.data.geom("object_visual").xpos
        pose = body_pose(sim)
        recovered = (world - pose[:3, 3]) @ pose[:3, :3]
        np.testing.assert_allclose(np.min(recovered, axis=0), np.min(expected, axis=0), atol=1e-7)
        np.testing.assert_allclose(np.max(recovered, axis=0), np.max(expected, axis=0), atol=1e-7)
        np.testing.assert_allclose(pose[:3, 3], self.config["object"]["pose"]["position_m"])
        self.assertEqual(len(scene.robot.joint_names), 56)
        self.assertGreaterEqual(sim.model.body("calibration_table_surface").id, 0)
        for side in ("left", "right"):
            self.assertGreaterEqual(sim.model.body(side + "_hand_flange").id, 0)

    def test_pose_updates_preserve_robot_camera_geometry_and_time(self) -> None:
        editor = self.editor()
        sim = editor.scene.sim
        robot_q, targets = sim.configuration(), sim.targets
        robot_bodies = sim.data.xpos[: sim.model.body("object").id].copy()
        geometry = sim.model.mesh_vert.copy()
        camera = sim.data.camera(editor.scene.camera_name).xmat.copy()
        with (
            patch.object(sim, "step", side_effect=AssertionError("No physics")),
            patch.object(editor.scene, "settle", side_effect=AssertionError("No settling")),
        ):
            editor.set_pose([0.12, -0.04, 0.15], [65, -20, 35])
            expected = Rotation.from_euler("xyz", [65, -20, 35], degrees=True).as_matrix()
            np.testing.assert_allclose(body_pose(sim)[:3, :3], expected, atol=1e-12)
            np.testing.assert_allclose(body_pose(sim)[:3, 3], [0.12, -0.04, 0.15])
            np.testing.assert_array_equal(sim.configuration(), robot_q)
            np.testing.assert_array_equal(sim.targets, targets)
            np.testing.assert_array_equal(sim.data.xpos[: len(robot_bodies)], robot_bodies)
            np.testing.assert_array_equal(sim.model.mesh_vert, geometry)
            np.testing.assert_array_equal(sim.data.camera(editor.scene.camera_name).xmat, camera)
            self.assertEqual(sim.data.time, 0)
            # The viewer resets robot sliders on each update; object editing must survive it.
            changed = robot_q.copy()
            changed[0] += 0.1
            sim.reset(changed)
            editor.apply_pose()
            np.testing.assert_array_equal(sim.configuration(), changed)
            np.testing.assert_allclose(body_pose(sim)[:3, 3], editor.position)

    def test_save_reload_pipeline_and_point_cloud_follow_pose(self) -> None:
        editor = self.editor(destination=self.root / "edited.yaml")
        original = self.source.read_bytes()
        old_points, _ = editor.scene.cloud()
        old_pose = body_pose(editor.scene.sim)
        editor.set_pose([0.1, -0.07, 0.12], [70, 5, 15])
        saved = editor.save()
        config = yaml.safe_load(saved.read_text())
        without_pose = deepcopy(config)
        without_pose["object"]["pose"] = self.config["object"]["pose"]
        self.assertEqual(without_pose, self.config)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(editor.config, self.config)
        with self.assertRaises(FileExistsError):
            editor.save()
        reloaded_root = self.root / "reload"
        reloaded_root.mkdir()
        reloaded = ObjectScene(config, reloaded_root)
        np.testing.assert_allclose(body_pose(reloaded.sim), body_pose(editor.scene.sim), atol=1e-12)
        points, _ = reloaded.cloud()
        local = (old_points - old_pose[:3, 3]) @ old_pose[:3, :3]
        pose = body_pose(reloaded.sim)
        np.testing.assert_allclose(points, local @ pose[:3, :3].T + pose[:3, 3], atol=1e-12)
        result = subprocess.run(
            [
                sys.executable,
                str(project_root() / "script/run_dual_arm_grasp_pipeline.py"),
                "--environment",
                "simulation",
                "--scene",
                str(saved),
                "--perception-only",
                "--output-root",
                str(self.root / "pipeline"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with np.load(self.root / "pipeline/object_cloud.npz") as data:
            np.testing.assert_allclose(data["T_world_object"], pose, atol=1e-12)
            np.testing.assert_array_equal(data["points_world_m"], points)

    def test_unique_saves_and_source_aliases_are_rejected(self) -> None:
        editor = self.editor()
        original = self.source.read_bytes()
        with patch.dict(os.environ, {"ADA_GRASP_CTRL_OUTPUT_ROOT": str(self.root / "outputs")}):
            first, second = editor.save(), editor.save()
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file() and second.is_file())
        self.assertEqual(yaml.safe_load(first.read_text()), self.config)
        for path in (self.source, self.root / "alias.yaml"):
            if path != self.source:
                path.symlink_to(self.source)
            editor.destination = path
            with self.assertRaisesRegex(ValueError, "source YAML"):
                editor.save()
        self.assertEqual(self.source.read_bytes(), original)

    def test_fixed_object_and_settling_configuration_remain_kinematic(self) -> None:
        config = deepcopy(self.config)
        del config["object"]["physics"]
        config["object"]["placement"] = "settle_on_table"
        editor = self.editor(config, self.root / "fixed.yaml")
        q = editor.scene.sim.configuration()
        editor.set_pose([0.15, -0.02, 0.2], [25, 15, -30])
        np.testing.assert_allclose(body_pose(editor.scene.sim)[:3, 3], editor.position)
        np.testing.assert_array_equal(editor.scene.sim.configuration(), q)
        self.assertEqual(editor.scene.sim.data.time, 0)
        self.assertEqual(yaml.safe_load(editor.save().read_text())["object"]["placement"], "settle_on_table")

    def test_camera_preview_uses_existing_perception_and_restores_display_groups(self) -> None:
        if mujoco.GLContext.__module__ != "mujoco.egl":
            self.skipTest("Run with MUJOCO_GL=egl before importing MuJoCo")
        editor = self.editor()
        sim = editor.scene.sim
        sim.model.geom_group[0] = 5
        groups = sim.model.geom_group.copy()
        qpos = sim.data.qpos.copy()
        rgb, points = editor.perception_preview()
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(points.shape, (2048, 3))
        np.testing.assert_array_equal(sim.model.geom_group, groups)
        np.testing.assert_array_equal(sim.data.qpos, qpos)
        self.assertEqual(sim.data.time, 0)
        with patch.object(editor.scene, "depth_cloud", side_effect=ValueError("not visible")):
            with self.assertRaisesRegex(ValueError, "not visible"):
                editor.perception_preview()
        np.testing.assert_array_equal(sim.model.geom_group, groups)

    def test_invalid_pose_is_rejected_without_changing_state(self) -> None:
        editor = self.editor()
        before = editor.scene.sim.data.qpos.copy()
        for pos, rpy in (([float("nan"), 0, 0], [0, 0, 0]), ([0, 0, 0], [0, float("inf"), 0])):
            with self.assertRaises(ValueError):
                editor.set_pose(pos, rpy)
        np.testing.assert_array_equal(editor.scene.sim.data.qpos, before)


if __name__ == "__main__":
    unittest.main()
