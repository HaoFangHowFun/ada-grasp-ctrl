"""Authoritative calibration, absolute model geometry and retained capture contracts; no ROS."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ada_grasp_ctrl.calibration import DEFAULT_CALIBRATION, load_calibration, calibrated_urdf, calibrated_snapshot
from ada_grasp_ctrl.experiments import fixed_camera
from ada_grasp_ctrl.experiments.fixed_camera_geometry import load_extrinsics, transform, file_hash
from ada_grasp_ctrl.paths import resolve_from_root, reset_runtime_roots
from ada_grasp_ctrl.pipeline import dual_runner, runner
from ada_grasp_ctrl.robot_environment import DualArmModel
from ada_grasp_ctrl.robot_environment.model import JOINT_NAMES
from shared_executor_fixtures import description


def placement(value: dict) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", value["rpy_rad"]).as_matrix()
    matrix[:3, 3] = value["xyz_m"]
    return matrix


def source_snapshot(path: Path) -> Path:
    """Meshless 56-joint tree with the inspected physical model's fixed-frame names."""
    path.mkdir()
    root = ET.fromstring(description())
    for side in ("left", "right"):
        for name, parent, child, xyz, rpy in (
            (f"{side}_base_joint", "world", f"{side}_base_link", "0 0 0", "0 0 0"),
            (f"{side}_J6-flange", f"{side}_J6_link", f"{side}_flange", "0.1 0 0", "0 0 0"),
            (f"{side}_child_joint", f"{side}_flange", f"{side}_ee_mount", "0 0 0", "0 0 0"),
            (
                f"{side}_flange-fanuc_flange",
                f"{side}_flange",
                f"{side}_fanuc_flange",
                "0 0 0",
                "3.141592653589793 -1.5707963267948966 0",
            ),
            (f"{side}_ee_to_tcp", f"{side}_ee_mount", f"{side}_tcp", "0.035 0 0", "0 0 0"),
        ):
            ET.SubElement(root, "link", name=child)
            joint = ET.SubElement(root, "joint", name=name, type="fixed")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)
        root.find(f"joint[@name='{side}_J1']/parent").set("link", f"{side}_base_link")
        mount = root.find(f"joint[@name='{side}_hand_flange_mount']")
        mount.set("name", f"{side}_arm_to_hand")
        mount.find("parent").set("link", f"{side}_ee_mount")
    for link in root.findall("link"):
        if link.get("name") == "world":
            continue
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="0.1")
        ET.SubElement(inertial, "inertia", ixx="0.001", ixy="0", ixz="0", iyy="0.001", iyz="0", izz="0.001")
        geom = ET.SubElement(link, "collision", name=link.get("name") + "_collision")
        ET.SubElement(ET.SubElement(geom, "geometry"), "sphere", radius="0.01")
    ET.SubElement(ET.SubElement(root, "mujoco"), "compiler", fusestatic="false", discardvisual="false")
    (path / "model.urdf").write_text(ET.tostring(root, encoding="unicode"))
    (path / "manifest.json").write_text(
        json.dumps(dict(schema_version=1, status="complete", joint_names=list(JOINT_NAMES)))
    )
    return path


class UnifiedCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Other embedded-run tests can leave their temporary data root active.
        reset_runtime_roots()
        self.addCleanup(reset_runtime_roots)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = json.loads(resolve_from_root(DEFAULT_CALIBRATION).read_text())
        self.path = self.root / "calibration.json"
        self.write()

    def write(self) -> None:
        self.path.write_text(json.dumps(self.data))

    def test_authoritative_only_and_malformed_transforms(self):
        expected = load_calibration(self.path)
        self.data["derived"] = {"irrelevant": "not geometry"}
        self.data["non_authoritative_measurements"] = {"irrelevant": None}
        self.data["authoritative"]["dual_crx"]["T_leftbase_rightbase"]["matrix_4x4"] = None
        self.write()
        actual = load_calibration(self.path)
        self.assertEqual(actual["base"], expected["base"])
        self.assertEqual(actual["mounts"], expected["mounts"])
        np.testing.assert_array_equal(actual["camera"], expected["camera"])
        for bad in ([1, 2], [0, 0, float("nan")]):
            self.data["authoritative"]["hand_mounts"]["left"]["xyz_m"] = bad
            self.write()
            with self.assertRaises(ValueError):
                load_calibration(self.path)

    def test_camera_transform_changes_with_json_and_preserves_legacy_format(self):
        first, provenance = load_extrinsics(self.path)
        point = np.array([[0.1, 0.05, 1.0]])
        old = transform(point, first)
        shift = np.array([0.2, -0.1, 0.03])
        changed = first.copy()
        changed[:3, 3] += shift
        self.data["authoritative"]["camera"]["T_leftbase_color_optical"]["matrix_4x4"] = changed.tolist()
        self.data["authoritative"]["camera"]["color_intrinsics_at_calibrated_resolution"]["K"] = [[9999] * 3] * 3
        self.write()
        second, updated = load_extrinsics(self.path)
        np.testing.assert_allclose(transform(point, second), old + shift)
        np.testing.assert_allclose(transform(transform(point, second), np.linalg.inv(second)), point)
        self.assertNotEqual(provenance["sha256"], updated["sha256"])
        self.assertFalse(updated["independently_verified"])
        legacy = self.root / "legacy.json"
        legacy.write_text(
            json.dumps(dict(T_world_cam=first.tolist(), frame="world from color optical", converged=True))
        )
        np.testing.assert_array_equal(load_extrinsics(legacy)[0], first)

    def test_temporary_json_updates_fk_collision_and_preview_geometry(self):
        source = source_snapshot(self.root / "source")
        previous = None
        for index in range(2):
            if index:
                self.data["authoritative"]["dual_crx"]["T_leftbase_rightbase"].update(
                    xyz_m=[0.3, -0.8, 0.05], rpy_rad=[0.1, -0.2, -1.3]
                )
                for side in ("left", "right"):
                    self.data["authoritative"]["hand_mounts"][side].update(
                        xyz_m=[0.04, 0.02, 0.03], rpy_rad=[0.2, 0.3, 0.4]
                    )
                self.write()
            target = calibrated_snapshot(source, self.root / f"run{index}", self.path)
            robot = DualArmModel.load(target)
            q = robot.home_configuration()
            q[:6], q[28:34] = [0.1, -0.2, 0.3, 0.15, -0.1, 0.2], [-0.1, 0.2, -0.3, 0.1, 0.2, 0.15]
            expected = load_calibration(self.path)
            np.testing.assert_allclose(robot.fk(q, "left", "base"), np.eye(4), atol=1e-12)
            np.testing.assert_allclose(robot.fk(q, "right", "base"), placement(expected["base"]), atol=1e-12)
            data = mujoco.MjData(robot.mj_model)
            data.qpos[[robot.mapping[n]["qpos"] for n in JOINT_NAMES]] = q
            mujoco.mj_forward(robot.mj_model, data)
            current = {}
            for side in ("left", "right"):
                hand = robot.fk(q, side, "ee_mount") @ placement(expected["mounts"][side])
                np.testing.assert_allclose(robot.fk(q, side), hand, atol=1e-12)
                current[side] = hand
                for frame in ("base", "hand_flange", "palm"):
                    name = robot.frame_name(side, frame)
                    fk = robot.fk(q, side, frame)
                    body = robot.mj_model.body(name).id
                    geom = robot.mj_model.geom(name + "_collision").id
                    np.testing.assert_allclose(data.xpos[body], fk[:3, 3], atol=1e-10)
                    np.testing.assert_allclose(data.xmat[body].reshape(3, 3), fk[:3, :3], atol=1e-10)
                    np.testing.assert_allclose(data.geom_xpos[geom], fk[:3, 3], atol=1e-10)
                solved, diagnostics = robot.solve_arm_ik(q, side, hand)
                self.assertTrue(diagnostics["accepted"])
                np.testing.assert_allclose(robot.fk(solved, side), hand, atol=1e-6)
                if previous is not None:
                    self.assertFalse(np.allclose(previous[side], hand))
            previous = current

    def test_absolute_idempotent_deterministic_and_source_preserved(self):
        source = source_snapshot(self.root / "source")
        digest = file_hash(source / "model.urdf")
        a = calibrated_snapshot(source, self.root / "a", self.path)
        b = calibrated_snapshot(source, self.root / "b", self.path)
        repeated = calibrated_snapshot(a, self.root / "repeated", self.path)
        self.assertEqual((a / "model.urdf").read_bytes(), (b / "model.urdf").read_bytes())
        self.assertEqual((a / "manifest.json").read_bytes(), (b / "manifest.json").read_bytes())
        self.assertEqual((a / "model.urdf").read_bytes(), (repeated / "model.urdf").read_bytes())
        self.assertEqual(file_hash(source / "model.urdf"), digest)
        with self.assertRaises(FileExistsError):
            calibrated_snapshot(source, a, self.path)

    def test_wrong_mount_mapping_fails_without_guessing(self):
        source = source_snapshot(self.root / "source")
        xml = (source / "model.urdf").read_text()
        root = ET.fromstring(xml)
        root.find("joint[@name='left_arm_to_hand']/parent").set("link", "left_fanuc_flange")
        with self.assertRaisesRegex(ValueError, "Unresolved calibration frame mapping"):
            calibrated_urdf(ET.tostring(root, encoding="unicode"), load_calibration(self.path))

    def test_capture_default_and_explicit_historical_path(self):
        config = dual_runner.configuration("src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml")
        args = SimpleNamespace(mode="live", reuse_candidates=False, roi=None, cluster=None)
        for value in (None, str(self.path)):
            config["extrinsics"] = value
            process = Mock()
            process.run.side_effect = RuntimeError("capture intercepted before any process")
            with self.assertRaisesRegex(RuntimeError, "capture intercepted"):
                runner.perceive({}, args, config, process, self.root, infer=False)
            command = process.run.call_args.args[0]
            self.assertEqual(
                command[command.index("--extrinsics") + 1], str(resolve_from_root(value or DEFAULT_CALIBRATION))
            )

    def test_dual_entry_records_calibrated_snapshot_for_all_consumers(self):
        config = dual_runner.configuration("src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml")
        config.update(robot_snapshot=str(source_snapshot(self.root / "source")), extrinsics=str(self.path))
        output = self.root / "run"
        with (
            patch.object(dual_runner, "configuration", return_value=config),
            patch.object(dual_runner, "Processes"),
            patch.object(dual_runner, "run") as run,
        ):
            self.assertEqual(
                dual_runner.main(
                    ["--environment", "mock", "--ros-domain-id", "88", "--output-root", str(output), "--export-only"]
                ),
                0,
            )
        effective = yaml.safe_load((output / "effective_config.yaml").read_text())
        self.assertEqual(effective["robot_snapshot"], str(output / "robot_snapshot"))
        self.assertEqual(run.call_args.args[1]["robot_snapshot"], effective["robot_snapshot"])
        manifest = json.loads((output / "robot_snapshot/manifest.json").read_text())
        self.assertEqual(manifest["calibration"]["sha256"], file_hash(self.path))

    def test_old_plane_hash_remains_a_block_even_with_unaccepted_preview(self):
        # Exercise the existing extraction check; no new acceptance mechanism.
        _, provenance = load_extrinsics(self.path)
        metadata = dict(extrinsics=provenance, serial="synthetic")
        plane = dict(
            schema_version=1,
            frame_id="world",
            unit="m",
            normal_xyz=[0, 0, 1],
            d=0,
            extrinsics=dict(sha256="old"),
            serial="synthetic",
        )
        path = self.root / "plane.json"
        path.write_text(json.dumps(plane))
        before = path.read_bytes()
        with patch.object(fixed_camera, "load_snapshot", return_value=({}, metadata)):
            with self.assertRaisesRegex(ValueError, "share camera and calibration"):
                fixed_camera.extract_object(self.root, path, self.root / "object", {}, None, None, True)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
