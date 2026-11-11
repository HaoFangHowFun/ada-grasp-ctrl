"""Geometry, motion constraints and malformed-plan regressions; no ROS commands."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.table_clearance import HandGeometry, make_plan, vertical_gaps
from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES
from ada_grasp_ctrl.ros_execution.table_clearance import check_command_step, validate
from ada_grasp_ctrl.ros_execution.settings import bridge_config


class CartesianKinematics:
    bounds = np.array([[-2.0] * 6, [2.0] * 6])

    def forward(self, q: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_rotvec(q[3:6].copy()).as_matrix()
        return pose


class FixtureHand:
    def __init__(self, adapter: object, plane: dict) -> None:
        self.adapter, self.plane = adapter, plane

    def measure(self, q: np.ndarray) -> dict:
        pose = np.linalg.inv(self.adapter.model_T_world) @ self.adapter.kin.forward(q)
        point = pose[:3, 3] + [0, 0, -0.1]
        return dict(gap_m=float(vertical_gaps(point[None], self.plane)[0]), point_world_m=point.tolist(), body="finger")


class TableClearanceTest(unittest.TestCase):
    def test_delayed_or_fast_commands_are_rejected(self) -> None:
        config = bridge_config()
        start = np.zeros(28)
        target = start.copy()
        target[0] = 0.001
        check_command_step(start, target, 0.01, config)
        target[0] = 0.002
        with self.assertRaisesRegex(RuntimeError, "arm"):
            check_command_step(start, target, 0.01, config)
        target[0] = 0.02
        with self.assertRaisesRegex(RuntimeError, "arm"):
            check_command_step(start, target, 0.3, config)

    def fixture(self) -> tuple:
        # Rotated model/world axes ensure vertical is not confused with model Z.
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix()
        transform[:3, 3] = [0.2, -0.1, 0.3]
        adapter = SimpleNamespace(
            kin=CartesianKinematics(),
            model_T_world=transform,
            model_id="fixture",
            limits=np.array([[-2, 2]] * 28),
            static_check=lambda q: {"known_geometry_passed": True},
        )
        q = np.r_[(transform @ [0.1, 0.1, 0.3, 1])[:3], [0, 0, 0], np.linspace(0, 0.1, 22)]
        state = dict(
            full_positions_rad=q.tolist(),
            full_joint_names=list(JOINT_NAMES),
            model_id="fixture",
            environment="mock",
            observed_description="fixture",
        )
        plane = dict(normal_xyz=[0, 0, 1], d=0, status="preview_unaccepted")
        return adapter, state, plane

    def plan(self, adapter: object, state: dict, plane: dict) -> dict:
        with (
            patch("ada_grasp_ctrl.experiments.table_clearance.HandGeometry", FixtureHand),
            patch("ada_grasp_ctrl.experiments.table_clearance.kinematic_contract", return_value={}),
        ):
            return make_plan(adapter, state, plane, 0.05, "mock")

    def test_tilted_plane_uses_vertical_gap(self) -> None:
        normal = np.array([-0.1, 0, 1.0])
        normal /= np.linalg.norm(normal)
        plane = dict(normal_xyz=normal.tolist(), d=0)
        np.testing.assert_allclose(vertical_gaps(np.array([[1, 0, 0.15], [2, 0, 0.25]]), plane), [0.05, 0.05])

    def test_world_vertical_preserves_xy_orientation_and_all_fingers(self) -> None:
        a, state, plane = self.fixture()
        plan = self.plan(a, state, plane)
        self.assertAlmostEqual(plan["target"]["gap_m"], 0.05, places=7)
        frames = np.array(plan["keyframes"])
        np.testing.assert_array_equal(frames[:, 6:], np.tile(frames[0, 6:], (len(frames), 1)))
        poses = [np.linalg.inv(a.model_T_world) @ a.kin.forward(q) for q in frames]
        for pose in poses:
            np.testing.assert_allclose(pose[:2, 3], poses[0][:2, 3], atol=1e-7)
            np.testing.assert_allclose(pose[:3, :3], poses[0][:3, :3], atol=1e-7)
        self.assertGreater(plan["path_samples"], 10)
        plan["file_sha256"] = {"fixture": "hash"}
        validate(plan)
        for mutation in ("finger", "speed", "policy", "nan"):
            changed = deepcopy(plan)
            if mutation == "finger":
                changed["keyframes"][-1][7] += 0.1
            if mutation == "speed":
                changed["durations_s"][0] = 0.01
            if mutation == "policy":
                changed["settings"]["force_limiter"] = "disabled"
            if mutation == "nan":
                changed["keyframes"][0][0] = float("nan")
            with self.assertRaises(ValueError):
                validate(changed)

    def test_collision_and_failed_ik_do_not_produce_plan(self) -> None:
        a, state, plane = self.fixture()
        a.static_check = lambda q: {"known_geometry_passed": False}
        with self.assertRaisesRegex(ValueError, "collides"):
            self.plan(a, state, plane)
        a, state, plane = self.fixture()
        fit = SimpleNamespace(x=np.array(state["full_positions_rad"][:6]), success=False)
        with patch("scipy.optimize.least_squares", return_value=fit):
            with self.assertRaisesRegex(ValueError, "IK"):
                self.plan(a, state, plane)

    def test_unsafe_initial_clearance_and_nonfinite_request_rejected(self) -> None:
        a, state, plane = self.fixture()
        plane["d"] = -0.19
        with self.assertRaisesRegex(ValueError, "recovery"):
            self.plan(a, state, plane)
        with self.assertRaises(ValueError):
            make_plan(a, state, plane, float("nan"), "mock")

    def test_mesh_descendants_and_transforms_define_lowest_point(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_string("""<mujoco><asset><mesh name="tetra"
            vertex="0 0 0  .02 0 0  0 .02 0  0 0 .04"/></asset><worldbody>
            <body name="hand" pos="0 0 .2"><geom type="mesh" mesh="tetra"/>
            <body name="finger" pos="0 0 -.1"><geom type="mesh" mesh="tetra"/></body></body>
            <body name="arm" pos="0 0 -.5"><geom type="mesh" mesh="tetra"/></body>
            </worldbody></mujoco>""")
        a = SimpleNamespace(
            model=model,
            data=mujoco.MjData(model),
            addresses=[],
            hand_bodies={"hand", "finger"},
            model_T_world=np.eye(4),
        )
        measurement = HandGeometry(a, dict(normal_xyz=[0, 0, 1], d=0)).measure(np.array([]))
        self.assertEqual(measurement["body"], "finger")
        self.assertAlmostEqual(measurement["gap_m"], 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
