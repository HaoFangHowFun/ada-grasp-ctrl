"""End-to-end inverse kinematics coverage for the bundled full example."""

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ada_grasp_ctrl.schema import load_npy_record
from ada_grasp_ctrl.tasks.dummy_arm_qpos import compute_dummy_arm_qpos
from ada_grasp_ctrl.utils.pk_helper import PytorchKinematicsHelper
from ada_grasp_ctrl.utils.robots import RobotFactory


class DummyArmIkTests(unittest.TestCase):
    def test_bundled_full_shadow_grasp_solves_on_cpu(self) -> None:
        """Solve each hand-base pose of the public one-record full fixture."""
        fixture = Path(__file__).resolve().parents[1] / "examples/data/shadow/formatted/grasp.npy"
        source = load_npy_record(fixture)
        robot = RobotFactory.create_robot(robot_type="dummy_arm_shadow", prefix="rh_")
        kin = PytorchKinematicsHelper(robot=robot, device="cpu")
        kin.create_serial_chain(ee_name=robot.arm.body_names[-1])

        with TemporaryDirectory() as directory:
            configs = SimpleNamespace(grasp_dir=str(fixture.parent), dummy_arm_grasp_dir=directory)
            output_path = compute_dummy_arm_qpos(([str(fixture)], configs), "cpu", robot, kin)[0]
            result = load_npy_record(output_path)

        for name in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos"):
            actual = kin.fk(
                torch.tensor(result[name], dtype=torch.float64).unsqueeze(0),
                link_name=robot.arm.body_names[-1],
            ).numpy()[0]
            expected = source[name][:7]
            self.assertLess(np.linalg.norm(actual[:3, 3] - expected[:3]), 1e-3)
            rotation_error = Rotation.from_matrix(actual[:3, :3]).inv() * Rotation.from_quat(
                expected[[4, 5, 6, 3]]
            )
            self.assertLess(rotation_error.magnitude(), 1e-2)


if __name__ == "__main__":
    unittest.main()
