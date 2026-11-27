"""Synthetic geometry, calibration isolation and multi-camera observation contracts."""

from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.calibration import DEFAULT_CALIBRATION
from ada_grasp_ctrl.experiments.fixed_camera_geometry import file_hash, rigid, transform
from ada_grasp_ctrl.experiments.multi_camera import (
    DEFAULT_CONFIG,
    calibrated_arrays,
    load_config,
    main as observe_main,
    table_for_capture,
)
from ada_grasp_ctrl.experiments.multi_camera_calibration import load_cameras, validated_entry, main as calibration_main
from ada_grasp_ctrl.experiments.multi_camera_geometry import (
    ROLES,
    SERIALS,
    associate_seed,
    backproject,
    bounds_mask,
    fuse_clouds,
    select_bundle,
)
from ada_grasp_ctrl.experiments.multi_camera_ros import image_array
from ada_grasp_ctrl.paths import reset_runtime_roots, resolve_from_root
from ada_grasp_ctrl.pipeline.observation import read_observation


def matrix(xyz=(0.2, -0.3, 0.4), angles=(20, -30, 40)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    result[:3, 3] = xyz
    return result


def intrinsics(width=100, height=100):
    return dict(
        width=width,
        height=height,
        k=[100.0, 0, 50.0, 0, 100.0, 50.0, 0, 0, 1.0],
        distortion_model="plumb_bob",
        d=[0.0] * 5,
        header=dict(frame_id="fixed_depth_optical_frame"),
    )


class MultiCameraTests(unittest.TestCase):
    def setUp(self):
        reset_runtime_roots()
        self.addCleanup(reset_runtime_roots)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(resolve_from_root(DEFAULT_CONFIG))
        self.master = json.loads(resolve_from_root(DEFAULT_CALIBRATION).read_text())

    def test_bundle_searches_older_common_window_and_rejects_total_skew(self):
        queues = {
            "fixed/depth": [(10.02, "a"), (10.16, "late")],
            "left/depth": [(10.03, "b")],
            "right/rgb": [(10.04, "c")],
        }
        self.assertEqual(
            select_bundle(queues, 10.2, 10, 0.5, 0.05), {"fixed/depth": "a", "left/depth": "b", "right/rgb": "c"}
        )
        with self.assertRaisesRegex(ValueError, "global"):
            select_bundle({"a": [(10.01, 1)], "b": [(10.10, 2)], "c": [(10.19, 3)]}, 10.2, 10, 0.5, 0.1)

    def test_bundle_rejects_missing_stale_future_and_pre_request_frames(self):
        for queue in ([], [(9, 1)], [(12, 1)], [(10, 1)], [(float("nan"), 1)]):
            with self.subTest(queue=queue), self.assertRaisesRegex(ValueError, "Missing"):
                select_bundle({"camera/depth": queue}, 10.5, 10, 0.4, 0.1)

    def test_depth_uses_depth_intrinsics_and_ros_units(self):
        info = intrinsics(2, 2)
        info["k"] = [2.0, 0, 0, 0, 4.0, 0, 0, 0, 1]
        limits = dict(min_depth_m=0.01, max_depth_m=2)
        points, ids = backproject(np.array([[1000, 0], [500, 2000]], dtype=np.uint16), "16UC1", info, limits)
        np.testing.assert_array_equal(ids, [0, 2, 3])
        np.testing.assert_allclose(points, [[0, 0, 1], [0, 0.125, 0.5], [1, 0.5, 2]])
        floating, ids = backproject(np.array([[1.0, np.nan], [0, np.inf]]), "32FC1", info, limits)
        np.testing.assert_allclose(floating, [[0, 0, 1]])
        np.testing.assert_array_equal(ids, [0])
        for key, value in (("width", 3), ("d", [0.1] * 5), ("k", [0.0] * 9)):
            invalid = {**info, key: value}
            with self.assertRaises(ValueError):
                backproject(np.ones((2, 2)), "16UC1", invalid, limits)
        with self.assertRaises(ValueError):
            backproject(np.ones((2, 2)), "mono16", info, limits)

    def test_padded_big_endian_depth_and_bgr_image(self):
        msg = SimpleNamespace(
            encoding="16UC1",
            is_bigendian=True,
            width=2,
            height=2,
            step=6,
            data=bytes.fromhex("03e807d0000001f003e80000"),
        )
        np.testing.assert_array_equal(image_array(msg), [[1000, 2000], [496, 1000]])
        msg = SimpleNamespace(encoding="bgr8", is_bigendian=False, width=1, height=1, step=4, data=b"\x01\x02\x03\x00")
        np.testing.assert_array_equal(image_array(msg), [[[3, 2, 1]]])
        msg.step = 2
        with self.assertRaises(ValueError):
            image_array(msg)

    def test_nontrivial_optical_composition_applies_once(self):
        world_color, color_depth = matrix(), matrix((0.01, -0.02, 0.005), (2, 3, -4))
        points = np.array([[0.1, 0.2, 0.4], [-0.1, 0.1, 0.3]])
        raw = {
            role: dict(points_source_m=points.copy(), rgb=np.zeros((2, 2, 3)), T_color_source=color_depth)
            for role in ROLES
        }
        metadata = {
            role: dict(
                color_frame=role + "_color_optical_frame",
                source_frame=role + "_depth_optical_frame",
                timestamps_s=dict(depth=1.0),
            )
            for role in ROLES
        }
        cameras = {
            role: dict(matrix=world_color, calibrated_optical_frame=role + "_color_optical_frame") for role in ROLES
        }
        cameras["wrist_right"] = dict(
            matrix=world_color @ color_depth, calibrated_optical_frame="wrist_right_depth_optical_frame"
        )
        arrays, _, clouds = calibrated_arrays(raw, metadata, cameras, {})
        expected = transform(transform(points, color_depth), world_color)
        for cloud in clouds.values():
            np.testing.assert_allclose(cloud, expected, atol=1e-12)
        np.testing.assert_allclose(transform(arrays["points_color_m"], arrays["T_world_color"]), expected)

    def test_invalid_rigid_transforms_are_rejected(self):
        for broken in (np.full((4, 4), np.nan), np.diag([-1, 1, 1, 1]), np.diag([2, 1, 1, 1]), np.diag([1, 1, 1, 2])):
            with self.assertRaises(ValueError):
                rigid(broken)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "Run image tests with ROS system Python")
    def test_fusion_voxel_workspace_height_and_exclusion(self):
        points = np.array([[0, 0, 1.05], [0.0001, 0, 1.05], [0.02, 0, 1.05], [0.04, 0, 1.05], [0, 0, 1], [3, 0, 1.05]])
        clouds = dict(fixed=points, wrist_left=np.empty((0, 3)), wrist_right=np.empty((0, 3)))
        fixed = dict(T_world_color=np.eye(4), rgb=np.zeros((100, 100, 3)))
        camera = dict(min_depth_m=0.1, max_depth_m=2, object_min_height_m=0.015, object_max_height_m=0.5)
        config = {
            **self.config,
            "workspace_bounds_m": [[-0.1, -0.1, 0.9], [0.03, 0.1, 1.2]],
            "exclude_bounds_m": [[[0.01, -0.1, 0.9], [0.025, 0.1, 1.2]]],
        }
        fused, counts = fuse_clouds(
            clouds, fixed, dict(color_info=intrinsics()), dict(normal_xyz=[0, 0, 1], d=-1), config, camera
        )
        np.testing.assert_allclose(fused["points_world_m"], [[0, 0, 1.05]])
        self.assertEqual(counts["input_points"]["fixed"], 6)
        for bounds in ([[1, 0, 0], [0, 1, 1]], [[0, 0, 0], [1, np.nan, 1]]):
            with self.assertRaises(ValueError):
                bounds_mask(points, bounds)

    def test_seed_grows_backside_outside_roi_without_largest_fallback(self):
        target = np.column_stack((np.arange(0, 0.12, 0.003), np.zeros(40), np.ones(40)))
        unrelated = target + [0, 0.1, 0]
        points = np.concatenate([target, unrelated, unrelated + [0.15, 0, 0]])
        camera = dict(object_cluster_radius_m=0.006, object_min_points=3)
        selected, _ = associate_seed(points, target[:8], unrelated, self.config, camera)
        np.testing.assert_array_equal(selected, np.arange(40))
        self.assertGreater(points[selected, 0].max(), target[:8, 0].max() + 0.08)
        with self.assertRaises(ValueError):
            associate_seed(points, target + [0, 5, 0], unrelated, self.config, camera)
        with self.assertRaises(ValueError):
            associate_seed(points, np.concatenate([target[:8], unrelated[:8]]), np.empty((0, 3)), self.config, camera)
        with self.assertRaisesRegex(ValueError, "another"):
            associate_seed(points, target[:8], target[-6:], self.config, camera)

    def calibration_fixture(self):
        path = self.root / "measured_evidence.txt"
        path.write_text("Synthetic test only, never a real calibration")
        provenance = {
            key: dict(path=str(path), sha256=file_hash(path))
            for key in ("solver_result", "board_config", "mount_registration")
        }
        m = matrix()
        candidate = dict(
            role="wrist_left",
            serial=SERIALS["wrist_left"],
            calibrated_optical_frame="wrist_left_color_optical_frame",
            unit="m",
            reference_frame="left_base_link",
            transform_direction="optical_to_reference",
            converged=True,
            T_reference_optical=m.tolist(),
            fit_observation_ids=["fit1", "fit2", "fit3"],
            provenance=provenance,
        )
        evidence = dict(observations=[], overlap_distances_m=[0.001, 0.002, 0.003])
        for i in range(3):
            known = matrix((0.1 * i, 0.2, 0.3), (20 * i, -10 * i, 15 * i))
            evidence["observations"].append(
                dict(
                    id=f"heldout{i}",
                    T_leftbase_board=known.tolist(),
                    T_optical_board=(np.linalg.inv(m) @ known).tolist(),
                    reprojection_rms_px=0.3,
                )
            )
        limits = dict(translation_m=0.01, rotation_deg=2, reprojection_px=2, overlap_m=0.01)
        return candidate, evidence, limits

    def test_calibration_validation_and_rejection(self):
        candidate, evidence, limits = self.calibration_fixture()
        entry = validated_entry(candidate, evidence, self.master, limits)
        self.assertLess(entry["quality"]["translation_rms_m"], 1e-12)
        for field, value in (("serial", "wrong"), ("reference_frame", "world"), ("converged", False), ("unit", "mm")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validated_entry({**candidate, field: value}, evidence, self.master, limits)
        bad = deepcopy(evidence)
        bad["observations"][0]["id"] = "fit1"
        with self.assertRaisesRegex(ValueError, "excluded"):
            validated_entry(candidate, bad, self.master, limits)
        bad = deepcopy(evidence)
        bad["observations"][0]["T_leftbase_board"][0][3] += 0.1
        with self.assertRaisesRegex(ValueError, "failed"):
            validated_entry(candidate, bad, self.master, limits)

    def test_calibration_loader_missing_then_valid_and_wrong_frame(self):
        path = self.root / "calibration.json"
        path.write_text(json.dumps(self.master))
        with self.assertRaisesRegex(ValueError, "missing measured"):
            load_cameras(path, self.config)
        candidate, evidence, limits = self.calibration_fixture()
        entry = validated_entry(candidate, evidence, self.master, limits)
        self.master["authoritative"]["stationary_cameras"] = {
            role: {**entry, "serial": SERIALS[role], "calibrated_optical_frame": role + "_color_optical_frame"}
            for role in ROLES[1:]
        }
        path.write_text(json.dumps(self.master))
        self.assertEqual(set(load_cameras(path, self.config)), set(ROLES))
        self.master["authoritative"]["stationary_cameras"]["wrist_right"]["reference_frame"] = "world"
        path.write_text(json.dumps(self.master))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_cameras(path, self.config)

    def test_validation_only_preserves_master_bytes_and_save_preserves_existing_values(self):
        candidate, evidence, _ = self.calibration_fixture()
        master = self.root / "master.json"
        master.write_text(json.dumps(self.master))
        before = master.read_bytes()
        c, e = self.root / "candidate.json", self.root / "evidence.json"
        c.write_text(json.dumps(candidate))
        e.write_text(json.dumps(evidence))
        args = [
            "validate",
            "--role",
            "wrist_left",
            "--calibration",
            str(master),
            "--candidate",
            str(c),
            "--evidence",
            str(e),
        ]
        self.assertEqual(calibration_main([*args, "--output", str(self.root / "validated.json")]), 0)
        self.assertEqual(master.read_bytes(), before)
        self.assertEqual(calibration_main([*args, "--save", "--output", str(self.root / "saved.json")]), 0)
        saved = json.loads(master.read_text())
        saved["authoritative"].pop("stationary_cameras")
        self.assertEqual(saved, self.master)
        self.assertEqual((self.root / "saved.json.previous").read_bytes(), before)

    def test_table_rebinding_does_not_change_geometry_or_original(self):
        source = self.root / "table.json"
        plane = dict(
            schema_version=1,
            frame_id="world",
            unit="m",
            normal_xyz=[0, 0, 1],
            d=-0.2,
            serial=SERIALS["fixed"],
            T_world_color=matrix().tolist(),
            extrinsics=dict(sha256="old"),
        )
        source.write_text(json.dumps(plane))
        original = source.read_bytes()
        metadata = dict(serial=SERIALS["fixed"], T_world_color=matrix().tolist(), extrinsics=dict(sha256="new"))
        out = table_for_capture(source, self.root / "bound.json", metadata)
        self.assertEqual(source.read_bytes(), original)
        rebound = json.loads(out.read_text())
        self.assertEqual(rebound["normal_xyz"], plane["normal_xyz"])
        self.assertEqual(rebound["d"], plane["d"])
        metadata["T_world_color"] = np.eye(4).tolist()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            table_for_capture(source, self.root / "wrong.json", metadata)

    def test_observation_stops_before_ros_without_extrinsics(self):
        root = self.root / "missing"
        with patch("ada_grasp_ctrl.experiments.multi_camera_ros.capture") as capture:
            self.assertEqual(observe_main(["observe", "--output-root", str(root), "--plane", "unused"]), 2)
        capture.assert_not_called()
        self.assertIn("missing measured", json.loads((root / "run_report.json").read_text())["error"])

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "Run pipeline CLI tests with Ada Python")
    def test_cli_rejects_unsupported_combinations_before_runtime(self):
        from ada_grasp_ctrl.pipeline.dual_runner import main

        for extra in (
            ["--multi-camera", "--environment", "simulation"],
            ["--multi-camera", "--environment", "mock"],
            ["--multi-camera", "--placement-only"],
            ["--multi-camera-config", "unused"],
        ):
            with self.subTest(extra=extra):
                self.assertEqual(main(["--output-root", str(self.root / "unused"), *extra]), 2)
        self.assertFalse((self.root / "unused").exists())

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "Run image tests with ROS system Python")
    def test_synthetic_observation_grows_seed_and_preserves_inference_coordinates(self):
        # Synthetic scene: table z=1, object z=1.05. A side view extends +X beyond the seed ROI.
        x, y = np.meshgrid(np.arange(0, 0.08, 0.002), np.arange(0, 0.04, 0.002))
        front = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 1.05)))
        side = front + [0.075, 0, 0]
        raw = {
            role: dict(points_source_m=points, T_color_source=np.eye(4), rgb=np.zeros((100, 100, 3), dtype=np.uint8))
            for role, points in zip(ROLES, (front, side, side))
        }
        metadata = {
            role: dict(
                serial=SERIALS[role],
                source_frame=role + "_depth_optical_frame",
                color_frame=role + "_color_optical_frame",
                timestamps_s=dict(depth=10),
                color_info=intrinsics(),
                depth_info=intrinsics(),
            )
            for role in ROLES
        }
        candidate, evidence, limits = self.calibration_fixture()
        entry = validated_entry(candidate, evidence, self.master, limits)
        self.master["authoritative"]["camera"]["T_leftbase_color_optical"]["matrix_4x4"] = np.eye(4).tolist()
        self.master["authoritative"]["stationary_cameras"] = {
            role: {
                **entry,
                "serial": SERIALS[role],
                "calibrated_optical_frame": role + "_color_optical_frame",
                "T_leftbase_optical": np.eye(4).tolist(),
            }
            for role in ROLES[1:]
        }
        master = self.root / "synthetic.json"
        master.write_text(json.dumps(self.master))
        plane = self.root / "plane.json"
        plane.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    frame_id="world",
                    unit="m",
                    normal_xyz=[0, 0, 1],
                    d=-1,
                    serial=SERIALS["fixed"],
                    T_world_color=np.eye(4).tolist(),
                    extrinsics=dict(sha256=file_hash(master)),
                    status="synthetic",
                )
            )
        )

        def fake_capture(config, diagnostics, observe_s):
            diagnostics.update(capture_s=0.01)
            return deepcopy(raw), deepcopy(metadata)

        root = self.root / "synthetic_observation"
        with patch("ada_grasp_ctrl.experiments.multi_camera_ros.capture", side_effect=fake_capture):
            result = observe_main(
                [
                    "observe",
                    "--calibration",
                    str(master),
                    "--plane",
                    str(plane),
                    "--roi",
                    "50",
                    "50",
                    "58",
                    "55",
                    "--output-root",
                    str(root),
                ]
            )
        self.assertEqual(result, 0)
        info, points, _ = read_observation(root / "object")
        self.assertEqual(info["source_type"], "multi_camera_fused")
        self.assertGreater(points[:, 0].max(), 0.14)
        np.testing.assert_allclose(points[:, 2], 1.05)
        self.assertGreater(np.linalg.norm(points.mean(axis=0)), 1.0)  # No premature centering/normalization.
        with np.load(root / "d415_seed/object.npz") as seed:
            self.assertLess(seed["points_world_m"][:, 0].max(), 0.08)
        self.assertEqual(json.loads(master.read_text()), self.master)

    def test_right_base_calibration_uses_existing_authoritative_relation(self):
        candidate, evidence, limits = self.calibration_fixture()
        base = self.master["authoritative"]["dual_crx"]["T_leftbase_rightbase"]
        left_right = np.eye(4)
        left_right[:3, :3] = Rotation.from_euler("xyz", base["rpy_rad"]).as_matrix()
        left_right[:3, 3] = base["xyz_m"]
        expected = np.asarray(candidate["T_reference_optical"])
        candidate["reference_frame"] = "right_base_link"
        candidate["T_reference_optical"] = (np.linalg.inv(left_right) @ expected).tolist()
        entry = validated_entry(candidate, evidence, self.master, limits)
        np.testing.assert_allclose(entry["T_leftbase_optical"], expected, atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "Run pipeline tests with Ada Python")
    def test_pipeline_routes_only_fused_object_to_existing_inference_and_keeps_default(self):
        from ada_grasp_ctrl.pipeline import dual_runner

        args = SimpleNamespace(export_only=True, roi=[0, 0, 10, 10], cluster=0)
        for multi in (False, True):
            root = self.root / str(multi)
            (root / "multi_camera").mkdir(parents=True)
            (root / "multi_camera/run_report.json").write_text('{"status":"observed"}')
            config = dict(
                multi_camera=multi,
                robot_snapshot=str(self.root),
                ros_python="/usr/bin/python3",
                environment="hardware",
                camera_config="src/ada_grasp_ctrl/config/experiments/fixed_camera.yaml",
                multi_camera_config=DEFAULT_CONFIG,
                table_plane="unused",
            )
            processes = Mock()
            with (
                patch.object(dual_runner, "perceive", return_value=root / "object") as single,
                patch.object(
                    dual_runner, "infer_live_sides", side_effect=RuntimeError("inference boundary reached")
                ) as infer,
            ):
                with self.assertRaisesRegex(RuntimeError, "inference boundary"):
                    dual_runner.run(args, config, root, processes, {})
                expected = root / "multi_camera/object" if multi else root / "object"
                self.assertEqual(infer.call_args.args[0], expected)
                self.assertEqual(single.call_count, 0 if multi else 1)
                if multi:
                    command = processes.run.call_args.args[0]
                    self.assertIn("ada_grasp_ctrl.experiments.multi_camera", command)
                    self.assertIn("--roi", command)
                self.assertFalse(any("--execute" in call.args[0] for call in processes.run.call_args_list))
        processes = Mock()
        processes.run.side_effect = [None, RuntimeError("capture rejected")]
        with patch.object(dual_runner, "infer_live_sides") as infer:
            with self.assertRaisesRegex(RuntimeError, "capture rejected"):
                dual_runner.run(args, config, root, processes, {})
            infer.assert_not_called()

    def test_preview_layer_callbacks_control_independent_source_handles(self):
        from ada_grasp_ctrl.experiments.multi_camera_preview import add_layers

        root = self.root / "preview"
        for folder in ("capture", "d415_seed", "object"):
            (root / folder).mkdir(parents=True)
        for name in (
            "capture/snapshot",
            "capture/wrist_left",
            "capture/wrist_right",
            "d415_seed/object",
            "object/object",
        ):
            np.savez(root / (name + ".npz"), points_world_m=np.ones((4, 3)))
        np.savez(root / "capture/fused.npz", points_world_m=np.ones((4, 3)))
        (root / "capture/cameras.json").write_text("{}")
        evidence = {}
        for name, suffix in (("fused", "npz"), ("cameras", "json")):
            path = root / f"capture/{name}.{suffix}"
            evidence.update({name + "_path": str(path), name + "_sha256": file_hash(path)})
        (root / "object/object.json").write_text(json.dumps(dict(multi_camera=evidence)))
        server = MagicMock()
        handles, checkboxes = [Mock(visible=False) for _ in range(5)], [Mock(value=False) for _ in range(5)]
        server.scene.add_point_cloud.side_effect = handles
        server.gui.add_checkbox.side_effect = checkboxes
        add_layers(server, root)
        self.assertEqual(
            [call.kwargs["visible"] for call in server.scene.add_point_cloud.call_args_list],
            [False, False, False, False, True],
        )
        for index, checkbox in enumerate(checkboxes):
            checkbox.value = True
            checkbox.on_update.call_args.args[0](None)
            self.assertTrue(handles[index].visible)
            self.assertFalse(any(handle.visible for handle in handles[index + 1 :]))

    def test_d415_table_acceptance_survives_only_verified_additive_camera_changes(self):
        from ada_grasp_ctrl.experiments.fixed_camera import calibration_digest_matches, verify_calibration_acceptance
        from ada_grasp_ctrl.experiments.fixed_camera_geometry import load_extrinsics

        previous, current = self.root / "previous.json", self.root / "current.json"
        previous.write_text(json.dumps(self.master))
        digest = file_hash(previous)
        updated = deepcopy(self.master)
        updated["authoritative"]["stationary_cameras"] = dict(
            wrist_left=dict(provenance=dict(previous_calibration=dict(path=str(previous), sha256=digest)))
        )
        current.write_text(json.dumps(updated))
        provenance = load_extrinsics(current)[1]
        self.assertTrue(calibration_digest_matches(provenance, digest))
        table = self.root / "table.json"
        table.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    frame_id="world",
                    unit="m",
                    normal_xyz=[0, 0, 1],
                    d=0,
                    quality_passed=True,
                    extrinsics=dict(path=str(current), sha256=digest),
                )
            )
        )
        acceptance = self.root / "acceptance.json"
        acceptance.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    operator_confirmed=True,
                    basis="synthetic test only",
                    plane_sha256=file_hash(table),
                    extrinsics_sha256=digest,
                )
            )
        )
        self.assertTrue(verify_calibration_acceptance(table, acceptance)["operator_confirmed"])
        for section in ("camera", "dual_crx", "hand_mounts"):
            changed = deepcopy(updated)
            changed["authoritative"][section]["different_legacy_field"] = True
            current.write_text(json.dumps(changed))
            self.assertFalse(calibration_digest_matches(load_extrinsics(current)[1], digest))
            with self.assertRaises(ValueError):
                verify_calibration_acceptance(table, acceptance)
        current.write_text(json.dumps(updated))
        previous.write_text(json.dumps({**self.master, "tampered": True}))
        self.assertFalse(calibration_digest_matches(load_extrinsics(current)[1], digest))


if __name__ == "__main__":
    unittest.main()
