"""Hardware orchestration and live display tests with no ROS or physical commands."""

from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from ada_grasp_ctrl.contact.force_display import ForceSnapshot
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.pipeline import dual_runner, dual_viewer, paired_artifacts
from ada_grasp_ctrl.pipeline.dual_packet import load_execution_packet, preview_identity
from ada_grasp_ctrl.pipeline.dual_tactile import execution_display_locked
from dual_selector_fixtures import AnalyticalSide
from test_native_multimode_pipeline import NativeFixture
from test_native_paired import PairedLiveFixture


class HardwareBoundary:
    """Replace every viewer/transport process; retain real CLI, selection and export."""

    viewer_alive = False
    bad_review = False

    def start(self, command, label, **kw):
        self.calls.append((label, command))
        if label == "viewer":
            self.viewer_alive = True
            path = self.root / "run/export/targets.json"
            self.reviewed_bytes = path.read_bytes()
            identity = preview_identity(load_execution_packet(path), path)
            if self.bad_review:
                identity["packet_sha256"] = "wrong"
            write_json(self.root / "run/viewer_state.json", identity)
        elif label != "receiver":
            raise AssertionError(f"Unexpected service: {label}")

    def run(self, command, label, **kw):
        if label == "receiver_probe":
            self.calls.append((label, command))
            return
        if label in ("sender", "shared_executor"):
            self.calls.append((label, command))
            assert self.viewer_alive and not self.closed
            assert self.artifact("run_report.json")["execution_started"] is True
            path = self.root / "run/export/targets.json"
            assert path.read_bytes() == self.reviewed_bytes
            if label == "shared_executor":
                assert "ada_grasp_ctrl.ros_execution.shared_hardware" in command
                assert (
                    command[command.index("--reviewed-sha256") + 1]
                    == preview_identity(load_execution_packet(path), path)["packet_sha256"]
                )
            else:
                assert "--execute" in command
                assert command[command.index("--environment") + 1] == "hardware"
            output = Path(command[command.index("--output-root") + 1])
            output.mkdir()
            write_json(output / "run_report.json", dict(synthetic_transport=True))
            return
        super().run(command, label, **kw)

    def close(self):
        self.viewer_alive = False
        super().close()


class SingleHardwareFixture(HardwareBoundary, NativeFixture):
    def __init__(self, root):
        super().__init__(root, available=["right_full"])


class PairedHardwareFixture(HardwareBoundary, PairedLiveFixture):
    pass


class TactileExecutionTests(unittest.TestCase):
    def test_single_and_paired_keep_viewer_and_review_gate_during_execution(self):
        options = [
            "--environment",
            "hardware",
            "--ros-domain-id",
            "87",
            "--grasp-model",
            "native_multimode",
            "--selector",
            "hierarchical",
            "--tactile",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for kind in (SingleHardwareFixture, PairedHardwareFixture):
                for confirm, bad_review in ((False, False), (True, True), (True, False)):
                    with self.subTest(kind=kind.__name__, confirm=confirm, bad_review=bad_review):
                        fixture = kind(Path(directory) / f"{kind.__name__}_{confirm}_{bad_review}")
                        fixture.bad_review = bad_review
                        with patch.object(paired_artifacts, "SideAdapter", AnalyticalSide):
                            code = fixture.invoke(options, export_only=False, confirm=confirm)
                        self.assertEqual(code, 2 if bad_review else 0)
                        config = yaml.safe_load((fixture.root / "run/effective_config.yaml").read_text())
                        self.assertTrue(config["tactile"])
                        labels = [label for label, _ in fixture.calls]
                        self.assertEqual(labels.count("viewer"), 1)
                        if confirm and not bad_review:
                            transport = "shared_executor" if kind is PairedHardwareFixture else "sender"
                            self.assertEqual(labels.count(transport), 1)
                            self.assertLess(labels.index("viewer"), labels.index("before_execute"))
                            self.assertLess(labels.index("before_execute"), labels.index(transport))
                            self.assertEqual(fixture.artifact("run_report.json")["status"], "completed")
                        else:
                            self.assertFalse(set(labels) & {"receiver", "sender", "shared_executor"})
                        self.assertTrue(fixture.closed)
                        self.assertFalse(fixture.viewer_alive)

    def test_unsupported_modes_fail_before_any_process(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(dual_runner, "Processes") as processes:
            for options in (
                ["--environment", "mock"],
                ["--environment", "simulation"],
                ["--environment", "hardware", "--export-only"],
                ["--environment", "hardware", "--placement-only"],
                ["--environment", "hardware", "--placement-aware"],
            ):
                with self.subTest(options=options):
                    self.assertEqual(
                        dual_runner.main(["--tactile", "--output-root", directory + "/unused", *options]), 2
                    )
            processes.assert_not_called()

    def test_execution_display_latches_across_partial_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "run_report.json"
            self.assertFalse(execution_display_locked(root, False))
            for status in ("exported", "preview_only", "awaiting_enter"):
                write_json(path, dict(status=status))
                self.assertFalse(execution_display_locked(root, False))
            write_json(path, dict(status="executing", execution_started=True))
            self.assertTrue(execution_display_locked(root, False))
            for contents in ('{"execution_started":', "[]", "{}"):
                path.write_text(contents)
                self.assertFalse(execution_display_locked(root, False))
                self.assertTrue(execution_display_locked(root, True))

    def test_viewer_switches_from_review_to_measured_and_clears_dropout(self):
        # Exercise the actual viewer loop with synthetic feedback and GUI handles.
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root / "export").mkdir()
            (root / "export/targets.json").write_text("{}")
            (root / "effective_config.yaml").write_text(
                yaml.safe_dump(
                    dict(
                        tactile=True, multi_camera=True, robot_snapshot=str(root), table_plane=str(root / "plane.json")
                    )
                )
            )
            (root / "state").mkdir()
            write_json(root / "state/robot_state.json", dict(full_positions_rad=[0.0] * 56))
            write_json(root / "plane.json", {})
            write_json(root / "run_report.json", dict(status="awaiting_enter"))
            packet = dict(
                side="left",
                sample_id="test",
                reviewed_plan=dict(phases=["grasp"], durations_s=[1.0], settle_time_s=0.3, candidate_id="test"),
            )
            gui, handles = Mock(), {}
            gui.gui.add_folder.return_value = nullcontext()

            def control(label, **kw):
                handle = Mock(value=kw.get("initial_value"), disabled=False)
                if label == "Live measured pose + tactile":
                    handle.value = False  # The operator was reviewing planned motion.
                handles[label] = handle
                return handle

            for name in ("add_slider", "add_checkbox", "add_dropdown"):
                getattr(gui.gui, name).side_effect = control
            gui.get_port.return_value = 8089
            sim = SimpleNamespace(model=object(), data=SimpleNamespace(qpos=np.zeros(56)), qpos=np.arange(56))
            scene, tactile, overlay = Mock(), Mock(), Mock()
            scene.geom_groups_visible = [True] * 4
            forces = ForceSnapshot(10, ())
            tactile.sample.side_effect = [
                (np.ones(56), forces, dict(pose_status="LIVE")),
                (np.ones(56) * 2, forces, dict(pose_status="LIVE")),
                (None, forces, dict(pose_status="stale")),
            ]
            observed = []

            def tick(seconds):
                observed.append((sim.data.qpos.copy(), deepcopy(overlay.update.call_args.args[0])))
                if len(observed) == 1:
                    self.assertFalse(handles["Live measured pose + tactile"].disabled)
                    write_json(root / "run_report.json", dict(status="executing", execution_started=True))
                else:
                    self.assertTrue(handles["Live measured pose + tactile"].disabled)
                    self.assertTrue(handles["Playback (%)"].disabled)
                if len(observed) == 3:
                    raise KeyboardInterrupt

            for name, value in (
                ("load_packet", lambda path: packet),
                ("live_scene", lambda *args: (object(), sim)),
                ("create_server", lambda: gui),
                ("add_measured_table", lambda *args: None),
                ("ViserMujocoScene", lambda *args, **kw: scene),
                ("sample_plan", lambda *args: np.ones(28) * 3),
                ("phase_at_progress", lambda *args: "grasp"),
            ):
                stack.enter_context(patch.object(dual_viewer, name, value))
            stack.enter_context(patch("sys.argv", ["viewer", "--run", str(root)]))
            stack.enter_context(patch.object(dual_viewer.mujoco, "mj_forward"))
            stack.enter_context(patch.object(dual_viewer.time, "sleep", side_effect=tick))
            stack.enter_context(patch("ada_grasp_ctrl.experiments.multi_camera_preview.add_layers"))
            stack.enter_context(patch("ada_grasp_ctrl.pipeline.dual_tactile.DualTactile.start", return_value=tactile))
            stack.enter_context(patch("ada_grasp_ctrl.pipeline.dual_tactile.diagnostic_text", return_value="forces"))
            stack.enter_context(patch("ada_grasp_ctrl.contact.force_display.ForceOverlay", return_value=overlay))
            dual_viewer.main()
            np.testing.assert_array_equal(observed[0][0][:28], 3)
            self.assertIsNone(observed[0][1])
            np.testing.assert_array_equal(observed[1][0], 2)
            self.assertEqual(observed[1][1], forces)
            np.testing.assert_array_equal(observed[2][0], 2)
            self.assertIsNone(observed[2][1])
            tactile.close.assert_called_once()
            gui.stop.assert_called_once()
            self.assertEqual(
                dual_runner.read_json(root / "viewer_state.json"),
                preview_identity(packet, root / "export/targets.json"),
            )


if __name__ == "__main__":
    unittest.main()
