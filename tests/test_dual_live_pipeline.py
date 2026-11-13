"""Dual live contracts: explicit side routing, measured provenance and repeatable captures."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import test_grasp_pipeline as fixtures
from ada_grasp_ctrl.experiments.right_ros_contract import side_names, packet_side, validate_packet, action_name
from ada_grasp_ctrl.pipeline.plan import validate_plan
from ada_grasp_ctrl.pipeline.dual_runner import configuration
from ada_grasp_ctrl.pipeline.dual_inference import infer_live_sides
from ada_grasp_ctrl.ros_execution.settings import bridge_config
from ada_grasp_ctrl.ros_execution.review import check_reviewed_goal


class DualLiveContracts(unittest.TestCase):
    def test_explicit_left_plan_routes_only_to_left_and_retains_right_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            _, right = fixtures.PipelineTransportTest().write_packet(Path(directory), lift=True, complete=True)
        validate_packet(right)
        self.assertEqual(packet_side(right), "right")
        left = deepcopy(right)
        left.update(
            kind="ada_side_joint_targets",
            side="left",
            joint_names=list(sum(side_names("left"), ())),
            pose_frame="left_hand_flange",
        )
        plan = left["reviewed_plan"]
        plan.update(side="left", joint_names=left["joint_names"], passive_positions_rad=[0.0] * 28)
        settings = bridge_config(dict(side="left", active_arms=["left", "right"]))
        plan["receiver_settings"] = settings
        validate_packet(left)
        validate_plan(plan)
        goal = left["targets"][0]
        check_reviewed_goal(left, 0, goal["positions"], goal["duration_s"], settings)
        self.assertEqual(action_name("left"), "/bimanual/left_arm_hand/follow_joint_trajectory")
        for field, bad in (
            ("joint_names", right["joint_names"]),
            ("pose_frame", "right_hand_flange"),
            ("kind", "ada_right_joint_targets"),
            ("side", "both"),
        ):
            changed = deepcopy(left)
            changed[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_packet(changed)
        with self.assertRaisesRegex(ValueError, "side"):
            check_reviewed_goal(left, 0, goal["positions"], goal["duration_s"], bridge_config())
        changed = deepcopy(plan)
        changed["passive_positions_rad"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_plan(changed)

    def test_side_defaults_keep_legacy_settings_and_reject_mismatched_stack(self):
        right = bridge_config()
        self.assertNotIn("side", right)
        self.assertEqual(right["tip_frame"], "right_hand_flange")
        self.assertEqual(bridge_config({"side": "left"})["tip_frame"], "left_hand_flange")
        for overrides in (
            {"side": "left", "tip_frame": "right_hand_flange"},
            {"side": "left", "active_arms": ["right"]},
            {"side": "both"},
        ):
            with self.assertRaises(ValueError):
                bridge_config(overrides)

    def test_new_objects_are_inferred_fresh_and_reflection_preserves_camera_transform(self):
        from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = root / "reference.json"
            placeholder = root / "checkpoint"
            placeholder.write_text("fixture")
            reference = dict(path=str(placeholder), sha256=file_hash(placeholder))
            ref.write_text(json.dumps({key: reference for key in ("checkpoint", "training_config", "metadata")}))
            config = dict(
                inference_model=json.loads(ref.read_text()),
                enabled_arms=["left", "right"],
                dexlearn_python="python",
                dexlearn_root=str(root),
                samples=8,
                seed=4,
                device="cpu",
            )
            seen = []

            def run(command, label):
                source = Path(command[command.index("--object") + 1])
                destination = Path(command[command.index("--output-root") + 1])
                with np.load(source / "object.npz") as data:
                    cloud = data["points_world_m"].copy()
                    transform = data["T_world_color"]
                    np.testing.assert_allclose(data["points_color_m"] @ transform[:3, :3].T + transform[:3, 3], cloud)
                    seen.append((label, cloud))
                destination.mkdir()
                payload = dict(
                    candidates=[dict(candidate_id="c0000")], checkpoint=reference, metadata=reference, preprocessing={}
                )
                (destination / "candidates.json").write_text(json.dumps(payload))

            clouds = [
                np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.3], [0.3, 0.1, 0.4]]),
                np.array([[0.6, 0.1, 0.2], [0.7, 0.3, 0.4], [0.8, 0.5, 0.2]]),
            ]
            with patch(
                "ada_grasp_ctrl.pipeline.dual_inference.from_right_record",
                side_effect=lambda raw, provenance, side, plane: dict(side=side, provenance=provenance),
            ):
                for i, points in enumerate(clouds):
                    output = root / str(i)
                    obj = output / "object"
                    obj.mkdir(parents=True)
                    transform = np.eye(4)
                    transform[:3, 3] = [0.4, -0.2, 0.1]
                    np.savez(
                        obj / "object.npz",
                        points_world_m=points,
                        points_color_m=points - transform[:3, 3],
                        T_world_color=transform,
                    )
                    (obj / "object.json").write_text(
                        json.dumps(
                            dict(
                                schema_version=1,
                                table_status="accepted",
                                object_sha256=file_hash(obj / "object.npz"),
                                extrinsics=reference,
                            )
                        )
                    )
                    result = infer_live_sides(obj, config, SimpleNamespace(run=run), output)
                    self.assertEqual([r["side"] for r in result], ["left", "right"])
            self.assertEqual(len(seen), 4)
            np.testing.assert_array_equal(seen[1][1], clouds[0])
            np.testing.assert_array_equal(seen[3][1], clouds[1])
            self.assertFalse(np.array_equal(seen[0][1], clouds[0]))
            self.assertNotEqual(result[0]["provenance"]["input_point_source"], "simulation_diagnostic")

    def test_default_entry_is_object_independent_and_opens_hand(self):
        config = configuration("src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml")
        self.assertEqual(config["enabled_arms"], ["left", "right"])
        self.assertTrue(config["above_object_open_hand"])
        self.assertIn("inference_model", config)
        self.assertNotIn("inference_reference", config)
        for key in ("roi", "candidate_id", "object_pose", "selected_side"):
            self.assertNotIn(key, config)


if __name__ == "__main__":
    unittest.main()
