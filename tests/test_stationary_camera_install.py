"""Explicit pending installation policy and RealSense root composition contracts."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ada_grasp_ctrl.calibration import DEFAULT_CALIBRATION
from ada_grasp_ctrl.paths import configure_runtime_roots, reset_runtime_roots, resolve_from_root
from ada_grasp_ctrl.experiments.fixed_camera_geometry import file_hash
from ada_grasp_ctrl.experiments.multi_camera import DEFAULT_CONFIG, load_config
from ada_grasp_ctrl.experiments.multi_camera_calibration import load_cameras
from ada_grasp_ctrl.experiments.stationary_camera_install import entry_digest, installed_master, pose_matrix
from ada_grasp_ctrl.experiments.stationary_camera_tf import root_transform


class StationaryCameraInstallTests(unittest.TestCase):
    def setUp(self):
        reset_runtime_roots()
        self.addCleanup(reset_runtime_roots)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.master = json.loads(resolve_from_root(DEFAULT_CALIBRATION).read_text())
        self.master["authoritative"].pop("stationary_cameras", None)
        self.config = load_config(resolve_from_root(DEFAULT_CONFIG))
        self.config.pop("accepted_pending_calibrations", None)
        self.pose = dict(
            zip(
                ("x", "y", "z", "qx", "qy", "qz", "qw"),
                [0.7, -0.3, 0.2, *Rotation.from_euler("xyz", [0.2, -0.6, 0.9]).as_quat()],
            )
        )
        # Convert numpy scalars for safe YAML serialization.
        self.pose = {key: float(value) for key, value in self.pose.items()}
        result = self.root / "result.yaml"
        result.write_text(yaml.safe_dump(dict(converged=True, camera_mount_to_camera=self.pose)))
        self.entries = {}
        for side in ("left", "right"):
            role = "wrist_" + side
            self.entries[role] = dict(
                serial=self.config["cameras"][role]["serial"],
                reference_frame="left_base_link",
                calibrated_optical_frame=role + "_color_optical_frame",
                transform_direction="optical_to_leftbase",
                unit="m",
                mounting="stationary",
                status="pending_independent_validation",
                T_leftbase_optical=pose_matrix(self.pose).tolist(),
                quality=dict(solver_converged=True, independent_validation_passed=False, sample_count=20),
                provenance=dict(solver_result=dict(path=str(result), sha256=file_hash(result))),
            )
        self.path = self.root / "master.json"
        self.path.write_text(json.dumps(installed_master(self.master, self.entries)))

    def test_pending_requires_exact_selection_and_keeps_status(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            load_cameras(self.path, self.config)
        self.config["accepted_pending_calibrations"] = {
            role: entry_digest(entry) for role, entry in self.entries.items()
        }
        cameras = load_cameras(self.path, self.config)
        for role in self.entries:
            self.assertFalse(cameras[role]["independent_validation_passed"])
            self.assertEqual(cameras[role]["status"], "pending_independent_validation")
            np.testing.assert_allclose(cameras[role]["matrix"], pose_matrix(self.pose))
        self.entries["wrist_left"]["T_leftbase_optical"][0][3] += 0.1
        self.path.write_text(json.dumps(installed_master(self.master, self.entries)))
        with self.assertRaisesRegex(ValueError, "independent"):
            load_cameras(self.path, self.config)

    def test_relative_provenance_uses_runtime_data_root(self):
        for entry in self.entries.values():
            entry["provenance"]["solver_result"]["path"] = "result.yaml"
        self.config["accepted_pending_calibrations"] = {
            role: entry_digest(entry) for role, entry in self.entries.items()
        }
        self.path.write_text(json.dumps(installed_master(self.master, self.entries)))
        configure_runtime_roots(asset_root=self.root, data_root=self.root, output_root=self.root)
        self.assertEqual(set(load_cameras(self.path, self.config)), {"fixed", "wrist_left", "wrist_right"})

    def test_pinned_entry_cannot_hide_changed_or_failed_solver(self):
        self.config["accepted_pending_calibrations"] = {
            role: entry_digest(entry) for role, entry in self.entries.items()
        }
        result = self.root / "result.yaml"
        result.write_text(yaml.safe_dump(dict(converged=False, camera_mount_to_camera=self.pose)))
        with self.assertRaisesRegex(ValueError, "changed calibration source"):
            load_cameras(self.path, self.config)
        for entry in self.entries.values():
            entry["provenance"]["solver_result"]["sha256"] = file_hash(result)
        self.config["accepted_pending_calibrations"] = {
            role: entry_digest(entry) for role, entry in self.entries.items()
        }
        self.path.write_text(json.dumps(installed_master(self.master, self.entries)))
        with self.assertRaisesRegex(ValueError, "converged solver"):
            load_cameras(self.path, self.config)

    def test_additive_install_preserves_all_legacy_geometry(self):
        before = deepcopy(self.master)
        installed = installed_master(self.master, self.entries)
        installed["authoritative"].pop("stationary_cameras")
        self.assertEqual(installed, before)
        self.assertEqual(self.master, before)

    def test_root_attachment_reconstructs_calibrated_optical_pose(self):
        base_optical = pose_matrix(self.pose)
        optical = dict(self.pose, x=0.012, y=-0.004, z=0.003)
        root_optical = pose_matrix(optical)
        base_root = root_transform(base_optical, root_optical)
        np.testing.assert_allclose(base_root @ root_optical, base_optical, atol=1e-12)
        point = np.array([0.1, 0.2, 0.4, 1.0])
        np.testing.assert_allclose(base_root @ root_optical @ point, base_optical @ point, atol=1e-12)

    def test_malformed_pose_and_policy_rejected(self):
        for pose in (dict(self.pose, qw=float("nan")), dict(self.pose, qw=2)):
            with self.assertRaises(ValueError):
                pose_matrix(pose)
        config = self.root / "bad.yaml"
        config.write_text(yaml.safe_dump({**self.config, "accepted_pending_calibrations": True}))
        with self.assertRaisesRegex(ValueError, "accepted_pending"):
            load_config(config)


if __name__ == "__main__":
    unittest.main()
