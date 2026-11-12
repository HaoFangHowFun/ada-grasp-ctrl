"""Integration contracts using an explicitly exported real dual CRX/Sharpa snapshot."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ada_grasp_ctrl.robot_environment import DualArmModel, DualArmSimulation
from ada_grasp_ctrl.robot_environment.kinematics import residual
from ada_grasp_ctrl.robot_environment.model import JOINT_NAMES, side_slice


class InputContractTest(unittest.TestCase):
    def test_side_and_joint_layout(self) -> None:
        self.assertEqual(len(set(JOINT_NAMES)), 56)
        self.assertEqual(JOINT_NAMES[:6], tuple(f"left_J{i}" for i in range(1, 7)))
        self.assertEqual(JOINT_NAMES[28:34], tuple(f"right_J{i}" for i in range(1, 7)))
        with self.assertRaises(ValueError):
            side_slice("both")


@unittest.skipUnless(os.environ.get("ADA_DUAL_ARM_SNAPSHOT"), "Set ADA_DUAL_ARM_SNAPSHOT to a real exported snapshot")
class DualArmEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot = DualArmModel.load(os.environ["ADA_DUAL_ARM_SNAPSHOT"])

    def setUp(self) -> None:
        self.sim = DualArmSimulation(self.robot)
        self.home = self.robot.home_configuration()

    def test_source_geometry_and_56_mappings(self) -> None:
        model = self.sim.model
        self.assertEqual(model.nq, 56)
        self.assertEqual(model.nu, 56)
        self.assertEqual(model.nexclude, 0)
        source = ET.parse(self.robot.directory / "source.urdf").getroot()
        resolved = ET.parse(self.robot.urdf_path).getroot()
        for joint in source.findall("joint"):
            imported = resolved.find(f"joint[@name='{joint.attrib['name']}']")
            self.assertEqual(
                ET.canonicalize(ET.tostring(joint), strip_text=True),
                ET.canonicalize(ET.tostring(imported), strip_text=True),
            )
        collision_count = 0
        for link in source.findall("link"):
            for index, _ in enumerate(link.findall("collision")):
                geom = model.geom(f"{link.attrib['name']}_collision_{index}")
                self.assertTrue(geom.contype or geom.conaffinity)
                collision_count += 1
        self.assertEqual(
            collision_count, sum(bool(model.geom_contype[i] or model.geom_conaffinity[i]) for i in range(model.ngeom))
        )
        for name, row in self.sim.mapping.items():
            joint = model.joint(name)
            self.assertEqual(model.actuator_trnid[row["actuator"], 0], joint.id)
            self.assertEqual(row["qpos"], joint.qposadr[0])
            self.assertEqual(row["qvel"], joint.dofadr[0])

    def test_fk_for_every_body_zero_single_joint_and_random(self) -> None:
        rng = np.random.default_rng(2026)
        cases = [self.home]
        for index in range(56):
            q = self.home.copy()
            q[index] = 0.1
            cases.append(q)
        cases.extend(rng.uniform(self.robot.limits[:, 0], self.robot.limits[:, 1], size=(10, 56)))
        maximum = {"position_m": 0.0, "orientation_rad": 0.0}
        for q in cases:
            self.sim.data.qpos[self.sim.qpos] = q
            mujoco.mj_forward(self.sim.model, self.sim.data)
            for i in range(1, self.sim.model.nbody):
                name = self.sim.model.body(i).name
                if not name.startswith(("left_", "right_")):
                    continue
                actual = np.eye(4)
                actual[:3, :3] = self.sim.data.xmat[i].reshape(3, 3)
                actual[:3, 3] = self.sim.data.xpos[i]
                errors = residual(actual, self.robot.fk(q, name.split("_")[0], name))
                for key, value in errors.items():
                    maximum[key] = max(maximum[key], value)
        print("\nDual-arm all-body FK maximum:", maximum)
        self.assertLessEqual(max(maximum.values()), 1e-6)

    def test_ik_both_sides_holds_other_50_joints(self) -> None:
        current = np.random.default_rng(9).uniform(self.robot.limits[:, 0], self.robot.limits[:, 1])
        for side in ("left", "right"):
            arm = side_slice(side, True)
            current[arm] = [0.1, -0.3, 0.6, 0.2, -0.3, 0.1]
            target_q = current.copy()
            target_q[arm] += [0.04, -0.02, 0.03, 0.02, -0.04, 0.05]
            for frame in ("hand_flange", "crx_flange", "tcp"):
                candidate, diagnostic = self.robot.solve_arm_ik(
                    current, side, self.robot.fk(target_q, side, frame), frame=frame
                )
                print("\nIK", side, frame, diagnostic)
                self.assertTrue(diagnostic["accepted"])
                held = np.ones(56, dtype=bool)
                held[arm] = False
                np.testing.assert_array_equal(candidate[held], current[held])
                self.robot.configuration(candidate)

    def test_unreachable_and_budget_exhaustion_are_diagnosed(self) -> None:
        target = self.robot.fk(self.home, "left")
        target[:3, 3] += 100
        candidate, diagnostic = self.robot.solve_arm_ik(self.home, "left", target, max_nfev=1)
        self.robot.configuration(candidate)
        self.assertFalse(diagnostic["accepted"])
        self.assertFalse(diagnostic["solver_success"])

    def test_malformed_inputs_and_solver_output_are_rejected(self) -> None:
        for value in (np.zeros(55), np.full(56, np.nan), np.full(56, np.inf), self.robot.limits[:, 1] + 0.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.sim.check_static_contacts(value)
        for value in (np.eye(3), np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.robot.solve_arm_ik(self.home, "left", value)
        target = self.robot.fk(self.home, "left")
        for value in (np.zeros(5), np.full(6, np.nan), np.full(6, 100)):
            with patch("ada_grasp_ctrl.robot_environment.kinematics.least_squares") as solve:
                solve.return_value.x = value
                with self.assertRaises(ValueError):
                    self.robot.solve_arm_ik(self.home, "left", target)
        with self.assertRaises(ValueError):
            self.robot.fk(self.home, "left", "right_hand_flange")

    def test_independent_motion_and_targets(self) -> None:
        for side, other in (("left", "right"), ("right", "left")):
            q = self.home.copy()
            q[side_slice(side).start + 1] = 0.1
            self.assertGreater(np.linalg.norm(self.robot.fk(q, side) - self.robot.fk(self.home, side)), 0.01)
            np.testing.assert_array_equal(self.robot.fk(q, other), self.robot.fk(self.home, other))
            before = self.sim.targets
            self.sim.select_side(side)
            self.sim.set_target(q[side_slice(side)])
            np.testing.assert_array_equal(self.sim.targets[side_slice(other)], before[side_slice(other)])
            np.testing.assert_array_equal(self.sim.data.ctrl[self.sim.actuators], self.sim.targets)
        before = self.sim.targets
        for bad in (np.zeros(27), np.full(28, np.nan), np.full(28, 99)):
            with self.assertRaises(ValueError):
                self.sim.set_target(bad)
            np.testing.assert_array_equal(self.sim.targets, before)

    def test_contact_categories_and_live_state_preserved(self) -> None:
        self.sim.step(3)
        state = (self.sim.data.qpos.copy(), self.sim.data.qvel.copy(), self.sim.data.ctrl.copy(), self.sim.data.time)
        found = set()
        rng = np.random.default_rng(42)
        for _ in range(400):
            q = rng.uniform(self.robot.limits[:, 0], self.robot.limits[:, 1])
            for contact in self.sim.check_static_contacts(q):
                expected_hands = []
                for name in contact["bodies"]:
                    ancestor = name
                    while ancestor not in ("left_hand_flange", "right_hand_flange", "world"):
                        ancestor = self.robot.parent_links.get(ancestor, "world")
                    if ancestor != "world":
                        expected_hands.append(name)
                self.assertEqual(contact["hand_bodies"], expected_hands)
                if contact["penetration_m"] > 1e-5:
                    found.add(contact["category"])
            if {"left_right", "robot_table", "within_robot"} <= found:
                break
        self.assertTrue({"left_right", "robot_table", "within_robot"} <= found)
        np.testing.assert_array_equal(self.sim.data.qpos, state[0])
        np.testing.assert_array_equal(self.sim.data.qvel, state[1])
        np.testing.assert_array_equal(self.sim.data.ctrl, state[2])
        self.assertEqual(self.sim.data.time, state[3])
        self.assertTrue(any(c["neighboring_source_links"] for c in self.sim.check_static_contacts(self.home)))

    def test_free_object_mapping_and_intentional_contact(self) -> None:
        spec = self.robot.spec.copy()
        palm = self.robot.fk(self.home, "left", "palm")[:3, 3]
        body = spec.worldbody.add_body(name="test_object", pos=palm)
        body.add_freejoint(name="test_object_free")
        body.add_geom(name="test_object_sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.04, 0, 0], mass=0.1)
        sim = DualArmSimulation(self.robot, scene_spec=spec, object_bodies=("test_object",))
        self.assertEqual(sim.model.nq, 63)
        self.assertEqual(sim.model.nv, 62)
        self.assertEqual(len(sim.mapping), 56)
        q = self.home.copy()
        q[6] = 0.1
        contacts = sim.check_static_contacts(q)
        self.assertTrue(any(c["category"] == "robot_object" for c in contacts))
        np.testing.assert_array_equal(sim.static_data.qpos[sim.qpos], q)
        adr = sim.model.joint("test_object_free").qposadr[0]
        np.testing.assert_array_equal(sim.static_data.qpos[adr : adr + 7], sim.data.qpos[adr : adr + 7])
        sim.step(2)

    def test_headless_steps_are_finite_without_warnings(self) -> None:
        self.sim.step(10)
        self.assertAlmostEqual(self.sim.data.time, 0.02)
        self.assertTrue(np.isfinite(self.sim.data.qpos).all())
        self.assertTrue(np.isfinite(self.sim.data.qvel).all())
        self.assertFalse(any(w.number for w in self.sim.data.warning))

    def test_snapshot_is_portable_and_runtime_cannot_import_ros(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            moved = Path(temporary) / "moved_snapshot"
            shutil.copytree(self.robot.directory, moved)
            code = """
import importlib.abc, sys
class NoROS(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rclpy', 'ament_index_python', 'xacro', 'bimanual_manipulation', 'launch_ros'}:
            raise AssertionError('ROS runtime import: ' + fullname)
sys.meta_path.insert(0, NoROS())
from ada_grasp_ctrl.robot_environment import DualArmModel, DualArmSimulation
r = DualArmModel.load(sys.argv[1])
s = DualArmSimulation(r)
s.check_static_contacts(r.home_configuration())
s.step(2)
assert 'rclpy' not in sys.modules
"""
            environment = dict(os.environ)
            environment.pop("AMENT_PREFIX_PATH", None)
            environment.pop("ROS_DISTRO", None)
            result = subprocess.run(
                [sys.executable, "-c", code, str(moved)], env=environment, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((moved / "manifest.json").read_text())["mount_calibrated"], False)


if __name__ == "__main__":
    unittest.main()
