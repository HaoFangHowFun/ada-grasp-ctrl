"""Automatic active orchestration and extraction contracts; no ROS or hardware I/O."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from ada_grasp_ctrl.experiments import fixed_camera as camera
from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.pipeline import dynamic_lift, runner, viewer
from ada_grasp_ctrl.ros_execution.settings import bridge_config


class ActiveLauncherTests(unittest.TestCase):
    def config(self, root):
        config = yaml.safe_load(
            (project_root() / "src/ada_grasp_ctrl/config/experiments/grasp_pipeline.yaml").read_text()
        )
        model = root / "model.json"
        model.write_text(json.dumps(dict(model=dict(urdf="fixture"))))
        config["model_report"] = str(model)
        config["active_hardware"].update(
            mpc_python=sys.executable,
            mpc_package_share=str(root),
            ros_setup=str(model),
            workspace_setup=str(model),
            calibration_acceptance=None,
            active_output_root=str(root),
        )
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def test_explicit_one_command_routes_real_components_without_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self.config(root)
            calls = []
            packet = dict(
                reviewed_plan=dict(keyframes=[[0.0] * 28], receiver_settings=bridge_config({"environment": "hardware"}))
            )

            def start(processes, command, label, **kwargs):
                calls.append((label, command, kwargs))
                if label == "viewer":
                    (processes.root / "viewer_state.json").write_text("{}")
                return Mock()

            def run(processes, command, label, **kwargs):
                calls.append((label, command, kwargs))
                with processes.stage(label):
                    if "--model-urdf" in command:
                        dest = processes.root / label
                        dest.mkdir()
                        (dest / "robot_state.json").write_text(json.dumps(dict(full_positions_rad=[0.0] * 28)))
                    if label == "sender":
                        dest = processes.root / label
                        dest.mkdir()
                        (dest / "run_report.json").write_text('{"status":"completed"}')

            def perceive(*args):
                calls.append(("perception", [], {}))
                return root / "candidates.json"

            with (
                patch.object(runner.Processes, "start", start),
                patch.object(runner.Processes, "run", run),
                patch.object(camera, "load_plane"),
                patch.object(runner, "perceive", perceive),
                patch.object(
                    runner,
                    "select_candidates",
                    return_value=dict(feasible_candidates=1, candidates=1, selected_candidate_id="c1"),
                ),
                patch.object(runner, "load_entries", return_value=[dict(keyframes=[[]])]),
                patch.object(runner, "export_selection", return_value=root / "targets.json"),
                patch.object(runner, "load_packet", return_value=packet),
                patch.object(runner, "contact_sheets", side_effect=AssertionError("expensive preview")),
                patch.object(runner, "wait_for_enter", side_effect=AssertionError("gate")),
                patch("builtins.input", side_effect=AssertionError("prompt")),
                patch.object(camera, "choose_roi", side_effect=AssertionError("ROI prompt")),
            ):
                self.assertEqual(runner.main(["--config", str(config_path), "--controller-mode", "mpc_active"]), 0)
            self.assertEqual(calls[0][0], "viewer")
            receiver = next(c for label, c, _ in calls if label == "receiver")
            sender = next(c for label, c, _ in calls if label == "sender")
            for command in (receiver, sender):
                self.assertEqual(command[command.index("--controller-mode") + 1], "mpc_active")
                self.assertIn("--mpc-python", command)
            self.assertIn("--operator-confirmed", receiver)
            self.assertIn("--hardware-enable", receiver)
            self.assertIn("--dynamic-lift-plan", receiver)
            self.assertIn("--execute", sender)
            outputs = list(root.glob("active_*"))
            self.assertEqual(len(outputs), 1)
            manifest = json.loads((outputs[0] / "run_manifest.yaml").read_text())
            self.assertTrue(manifest["hardware_operator_confirmed"])
            self.assertEqual(manifest["config"]["environment"], "hardware")
            stages = [json.loads(line) for line in (outputs[0] / "timings.jsonl").read_text().splitlines()]
            self.assertTrue(any(s["stage"] == "cleanup" and "duration_s" in s for s in stages))

    def test_real_owned_subprocess_timing_and_failure_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processes = runner.Processes(dict(ros_domain_id=88), root)
            try:
                processes.run([sys.executable, "-c", "print('fixture')"], "fixture")
                with self.assertRaisesRegex(RuntimeError, "NO_OBJECT_DETECTED"):
                    processes.run([sys.executable, "-c", "raise SystemExit('NO_OBJECT_DETECTED')"], "failure")
            finally:
                processes.close()
            rows = [json.loads(line) for line in (root / "timings.jsonl").read_text().splitlines()]
            self.assertEqual([r["state"] for r in rows], ["running", "completed", "running", "failed"])
            self.assertGreater(rows[1]["duration_s"], 0)

    def test_scope_unique_outputs_and_saved_config_never_authorizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self.config(root)
            configs = []
            for options in (
                ["--controller-mode", "mpc_active"],
                ["--controller-mode", "mpc_active"],
                ["--controller-mode", "mpc_active", "--environment", "mock"],
                ["--controller-mode", "mpc_passive"],
                ["--controller-mode", "linear"],
                [],
            ):
                with patch.object(runner, "run") as run:
                    output = [] if len(configs) < 2 else ["--output-root", str(root / ("run" + str(len(configs))))]
                    self.assertEqual(runner.main(["--config", str(config_path), *options, *output]), 0)
                    configs.append(run.call_args.args)
            self.assertNotEqual(configs[0][2], configs[1][2])
            for args, config, *_ in configs[2:]:
                self.assertFalse(args.automatic_active)
                self.assertEqual(config["environment"], "mock")
                self.assertFalse(config["controller"]["hardware_enable"])
            config = yaml.safe_load(config_path.read_text())
            config.update(environment="hardware", controller=dict(mode="mpc_active"))
            config_path.write_text(yaml.safe_dump(config))
            with patch.object(runner, "run") as run:
                self.assertEqual(runner.main(["--config", str(config_path), "--output-root", str(root / "saved")]), 2)
                run.assert_not_called()

    def test_conflict_and_manual_override_fail_before_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(root)
            for flags in (
                ["--roi", "0", "0", "100", "100"],
                ["--cluster", "0"],
                ["--start-mock"],
                ["--tactile-replay", "old"],
            ):
                with patch.object(runner, "run") as run:
                    self.assertEqual(
                        runner.main(["--config", str(config), "--controller-mode", "mpc_active", *flags]), 2
                    )
                    run.assert_not_called()
            with patch.object(runner.socket, "socket") as sock, patch.object(runner.Processes, "start") as start:
                sock.return_value.__enter__.return_value.connect_ex.return_value = 0
                self.assertEqual(runner.main(["--config", str(config), "--controller-mode", "mpc_active"]), 2)
                start.assert_not_called()

    def test_perception_command_disables_manual_and_unaccepted_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = yaml.safe_load(self.config(root).read_text())
            config["calibration_acceptance"] = "acceptance.json"
            args = NS(mode="live", reuse_candidates=False, roi=None, cluster=None, automatic_active=True)
            processes = Mock()
            reference = NS(payload={k: {} for k in ("checkpoint", "training_config", "metadata")})
            with (
                patch.object(runner, "load_candidates", return_value=reference),
                patch.object(runner, "verify_reference", return_value="reference"),
            ):
                runner.perceive(dict(group="live", candidates_path="candidate"), args, config, processes, root)
            commands = [call.args[0] for call in processes.run.call_args_list]
            self.assertIn("--automatic-single-object", commands[1])
            self.assertIn("--calibration-acceptance", commands[1])
            self.assertTrue(
                all("--allow-unaccepted" not in c and "--roi" not in c and "--cluster" not in c for c in commands)
            )


class AutomaticObjectTests(unittest.TestCase):
    def extract(self, count, *, bad=None, automatic=True, elevations=None):
        config = camera.load_config(project_root() / "src/ada_grasp_ctrl/config/experiments/fixed_camera.yaml")
        rng = np.random.default_rng(4)
        points = np.zeros((400, 3))
        for i in range(count):
            points = np.r_[
                points, rng.normal(0, 0.002, (150 + 50 * i, 3)) + [i * 0.2, 0, elevations[i] if elevations else 0.04]
            ]
        if bad == "small":
            points = points[:10]
        if bad == "nan":
            points[0, 0] = np.nan
        arrays = dict(
            points_world_m=points,
            points_color_m=points + [0, 0, 1],
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            T_world_color=np.eye(4),
        )
        metadata = dict(
            extrinsics=dict(sha256="cal"),
            serial="camera",
            capture_stamp_s=2.0,
            snapshot_sha256="snap",
            T_world_color=np.eye(4).tolist(),
            color_info=dict(distortion_model="plumb_bob", k=[100, 0, 320, 0, 100, 240, 0, 0, 1], d=[0.0] * 5),
        )
        plane = dict(
            schema_version=1,
            frame_id="world",
            unit="m",
            normal_xyz=[0, 0, 1],
            d=0,
            status="accepted",
            quality_passed=True,
            extrinsics=dict(sha256="cal", independently_verified=True),
            serial="camera",
            capture_stamp_s=1.0,
            T_world_color=np.eye(4).tolist(),
        )
        if bad == "calibration":
            metadata["extrinsics"]["sha256"] = "changed"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "plane.json"
            camera.write_json(path, plane)
            with (
                patch.object(camera, "load_snapshot", return_value=(arrays, metadata)),
                patch.object(camera, "save_overlay"),
                patch.object(
                    camera, "project_color", side_effect=lambda points, info: points[:, :2] * 100 + [320, 240]
                ),
            ):
                return camera.extract_object(
                    root, path, root, config, None if automatic else [0, 0, 640, 480], None, False, automatic=automatic
                )

    def test_zero_one_multiple_and_manual_roi(self):
        with self.assertRaisesRegex(ValueError, "NO_OBJECT_DETECTED"):
            self.extract(0)
        self.assertEqual(self.extract(2)["point_count"], 200)
        self.assertEqual(self.extract(2, elevations=[0.04, 0.08])["point_count"], 150)
        self.assertEqual(self.extract(1, elevations=[0.052])["point_count"], 150)
        with self.assertRaisesRegex(ValueError, "NO_OBJECT_DETECTED"):
            self.extract(1, elevations=[0.06])
        with self.assertRaisesRegex(ValueError, "clusters found"):
            self.extract(2, automatic=False)
        self.assertEqual(self.extract(1)["point_count"], 150)
        self.assertEqual(self.extract(1, automatic=False)["roi"], [0, 0, 640, 480])

    def test_quality_and_calibration_fail_closed(self):
        for reason in ("small", "nan", "calibration"):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                self.extract(1, bad=reason)


class CalibrationAcceptanceTests(unittest.TestCase):
    def test_operator_attestation_is_bound_to_current_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extrinsics = root / "extrinsics.json"
            extrinsics.write_text("verified fixture")
            plane = root / "plane.json"
            camera.write_json(
                plane,
                dict(
                    schema_version=1,
                    frame_id="world",
                    unit="m",
                    normal_xyz=[0, 0, 1],
                    d=0,
                    quality_passed=True,
                    extrinsics=dict(path=str(extrinsics), sha256=camera.file_hash(extrinsics)),
                ),
            )
            record = root / "acceptance.json"
            camera.write_json(
                record,
                dict(
                    schema_version=1,
                    operator_confirmed=True,
                    basis="fixture attestation",
                    plane_sha256=camera.file_hash(plane),
                    extrinsics_sha256=camera.file_hash(extrinsics),
                ),
            )
            self.assertTrue(camera.verify_calibration_acceptance(plane, record)["operator_confirmed"])
            extrinsics.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                camera.verify_calibration_acceptance(plane, record)


class TrustedLiftTests(unittest.TestCase):
    def test_source_identity_freshness_and_feasibility(self):
        h = dict(snapshot=dict(captured_at=100.0), validation=dict(joint_limits=True, sampled_known_geometry=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(dict(ok=True, handoff=h)))
            dynamic_lift.check_trusted_plan(h, path, 101.0)
            for now in (99.0, 121.0, float("nan")):
                with self.assertRaises(ValueError):
                    dynamic_lift.check_trusted_plan(h, path, now)
            with self.assertRaises(ValueError):
                dynamic_lift.check_trusted_plan(h, None, 101.0)
            changed = deepcopy(h)
            changed["snapshot"]["captured_at"] = 100.1
            with self.assertRaises(ValueError):
                dynamic_lift.check_trusted_plan(changed, path, 101.0)
            changed["validation"]["sampled_known_geometry"] = False
            path.write_text(json.dumps(dict(ok=True, handoff=changed)))
            with self.assertRaises(ValueError):
                dynamic_lift.check_trusted_plan(changed, path, 101.0)


class ObservationalViewerTests(unittest.TestCase):
    def test_status_starts_without_summary_or_browser_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = root / "status.json"
            status.write_text('{"stage":"capture","state":"running"}')
            server = Mock()
            server.get_port.return_value = 8089
            with (
                patch("viser.ViserServer", return_value=server),
                patch.object(viewer.webbrowser, "open"),
                patch.object(viewer.time, "sleep", side_effect=KeyboardInterrupt),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    viewer.serve_status(status, root / "ready.json", root / "missing.json")
            self.assertTrue((root / "ready.json").exists())
            server.stop.assert_called_once()
            self.assertIn("capture", server.gui.add_markdown.return_value.content)

    def test_server_cannot_switch_to_another_port(self):
        server = Mock()
        server.get_port.return_value = 8090
        with patch("viser.ViserServer", return_value=server), self.assertRaisesRegex(RuntimeError, "occupied"):
            viewer.serve_status(Path("unused"), Path("unused"), Path("unused"))
        server.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
