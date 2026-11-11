"""Ada-to-ROS packet and phase preservation, independent of ROS and controller dynamics."""

from copy import deepcopy
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import (
    JOINT_NAMES,
    PHASES,
    check_kinematics,
    validate_packet,
)
from ada_grasp_ctrl.experiments.right_ros_export import export_targets
from ada_grasp_ctrl.experiments.right_ros_sender import load_packet, motion_duration
from ada_grasp_ctrl.ros_execution.settings import bridge_config


class RosTransportContractTest(unittest.TestCase):
    def test_original_timing_is_preserved(self) -> None:
        config = bridge_config()
        self.assertEqual(
            [config[key] for key in ("arm_speed", "hand_speed", "arm_acceleration", "hand_acceleration")],
            [0.15, 0.4, 0.3, 0.8],
        )
        self.assertEqual(config["settle_time"], 0.3)
        self.assertEqual(config["settle_timeout"], 8.0)
        args = argparse.Namespace(minimum_duration=2.0, **config)
        start = [0.0] * 28
        self.assertAlmostEqual(motion_duration(start, start, args), 2.2)
        for index, speed in ((0, 0.15), (6, 0.4)):
            target = start.copy()
            target[index] = 1.0
            self.assertAlmostEqual(motion_duration(start, target, args), 1.875 / speed * 1.05 + 0.1)

    def test_receiver_config_requires_explicit_environment_and_valid_limits(self) -> None:
        hardware = bridge_config({"environment": "hardware"})
        self.assertFalse(hardware["path_verified"])
        self.assertNotIn("operator_reviewed_joint_goal", hardware)
        for overrides in (
            {"environment": "unknown"},
            {"operator_reviewed_joint_goal": True},
            {"arm_speed": float("nan")},
            {"hand_acceleration": 0},
            {"rate_hz": 101},
            {"settle_time": True},
            {"tip_frame": "left_hand_flange"},
            {"arm_speeed": 0.15},
            ["mock"],
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                bridge_config(overrides)

    def packet(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "ada_right_joint_targets",
            "environment": "mock",
            "units": "rad",
            "joint_names": list(JOINT_NAMES),
            "reference_frame": "world",
            "pose_frame": "right_hand_flange",
            "kinematics": {"link": {"xyz": [0.0, 0.0, 0.0]}},
            "targets": [
                {"phase": phase, "positions": [float(i) / 100] * 28, "flange_target_wxyz": [0, 0, 0, 1, 0, 0, 0]}
                for i, phase in enumerate(PHASES)
            ],
        }

    def test_existing_sender_output_is_never_overwritten(self) -> None:
        from ada_grasp_ctrl.experiments.right_ros_sender import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "run_report.json"
            report.write_text("existing evidence")
            arguments = ["sender", "--targets", str(root / "targets.json"), "--output-root", str(root)]
            with (
                patch("sys.argv", arguments),
                patch("ada_grasp_ctrl.experiments.right_ros_sender.load_packet", return_value=self.packet()),
            ):
                self.assertEqual(main(), 2)
            self.assertEqual(report.read_text(), "existing evidence")

    def test_sender_preview_does_not_import_ros(self) -> None:
        from ada_grasp_ctrl.experiments.right_ros_sender import main

        packet = self.packet()
        packet["sample_id"] = "sample"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "targets.json"
            target.write_text(json.dumps(packet))
            arguments = ["sender", "--targets", str(target), "--output-root", str(root / "preview")]
            with (
                patch("sys.argv", arguments),
                patch.dict("sys.modules", {"rclpy": None}),
                patch("builtins.print"),
                patch("ada_grasp_ctrl.experiments.right_ros_sender.load_packet", return_value=packet),
            ):
                self.assertEqual(main(), 0)
            self.assertEqual(json.loads((root / "preview/run_report.json").read_text())["status"], "preview")

    def test_packet_preserves_distinct_three_phase_hand_and_arm_targets(self):
        packet = self.packet()
        original = deepcopy(packet)
        validate_packet(packet)
        self.assertEqual(packet, original)
        self.assertEqual([t["phase"] for t in packet["targets"]], list(PHASES))

    def test_reject_wrong_names_units_frames_and_nonfinite(self):
        for key, value in (
            ("joint_names", ["left_J1"] + list(JOINT_NAMES[1:])),
            ("units", "deg"),
            ("pose_frame", "right_hand_C_MC"),
            ("environment", "hardware"),
        ):
            packet = self.packet()
            packet[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_packet(packet)
        packet = self.packet()
        packet["targets"][2]["positions"][5] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_packet(packet)

    def test_reject_stage_loss_or_reordering(self):
        for targets in (self.packet()["targets"][:2], self.packet()["targets"][::-1]):
            packet = self.packet()
            packet["targets"] = targets
            with self.assertRaises(ValueError):
                validate_packet(packet)

    def test_model_mount_drift_is_rejected(self):
        model = self.packet()["kinematics"]
        check_kinematics(model, deepcopy(model))
        changed = deepcopy(model)
        changed["link"]["xyz"][0] = 0.001
        with self.assertRaisesRegex(ValueError, "kinematics differ"):
            check_kinematics(model, changed)

    def test_invalid_preparation_is_rejected_before_export(self):
        report = {"samples": [{"sample_id": "bad", "all_targets_reachable": False, "targets": []}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch("ada_grasp_ctrl.experiments.right_ros_export.load_preview", return_value=report):
                with self.assertRaisesRegex(ValueError, "invalid IK"):
                    export_targets(Path(directory), "bad", output)
            self.assertFalse(output.exists())

    def test_packet_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "targets.json"
            path.write_text(json.dumps(self.packet()))
            (root / "run_manifest.yaml").write_text(
                json.dumps({"mode": "ada_ros_target_export", "targets_sha256": "bad"})
            )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_packet(path)

    def test_export_keeps_prepared_qpos_without_clipping_or_phase_substitution(self):
        packet = self.packet()
        record = {phase + "_qpos": np.arange(28, dtype=float) / 100 + i / 10 for i, phase in enumerate(PHASES)}
        record.update(
            sample_id="sample", quaternion_order="WXYZ", flange_targets_wxyz=np.array([[0, 0, 0, 1, 0, 0, 0]] * 3)
        )
        diagnostics = [
            {
                "solver_success": True,
                "reachable": True,
                "joint_limit_violations": [],
                "arm_table_clear": True,
                "static_contacts": {"arm_table_penetration": False},
                "position_m": 0.0,
                "orientation_rad": 0.0,
            }
        ] * 3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("preview_report.json", "model.urdf", "prepared.npy"):
                (root / name).write_text("test source")
            report = {
                "samples": [
                    {
                        "sample_id": "sample",
                        "prepared_path": str(root / "prepared.npy"),
                        "all_targets_reachable": True,
                        "targets": diagnostics,
                        "evaluation_flange_targets_wxyz": record["flange_targets_wxyz"].tolist(),
                    }
                ],
                "tolerances": {"ik_position_m": 0.001, "ik_angle_rad": 0.01},
                "model": {"urdf": str(root / "model.urdf")},
                "registration": np.eye(4).tolist(),
            }
            with (
                patch("ada_grasp_ctrl.experiments.right_ros_export.load_preview", return_value=report),
                patch("ada_grasp_ctrl.experiments.right_ros_export.load_npy_record", return_value=record),
                patch("ada_grasp_ctrl.experiments.right_ros_export.validate_grasp_record"),
                patch(
                    "ada_grasp_ctrl.experiments.right_ros_export.kinematic_contract", return_value=packet["kinematics"]
                ),
            ):
                path = export_targets(root, "sample", root / "export")
            actual = load_packet(path)
            for phase, target in zip(PHASES, actual["targets"]):
                np.testing.assert_array_equal(target["positions"], record[phase + "_qpos"])


if __name__ == "__main__":
    unittest.main()
