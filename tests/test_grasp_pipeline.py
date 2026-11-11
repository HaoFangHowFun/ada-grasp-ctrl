"""Execution gate, reviewed-plan transport and timing regressions without ROS hardware."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, PHASES
from ada_grasp_ctrl.experiments.right_ros_sender import digest, load_packet
from ada_grasp_ctrl.pipeline.plan import check_start, phase_progress, sample_plan, validate_plan
from ada_grasp_ctrl.pipeline.runner import main, wait_for_enter
from ada_grasp_ctrl.pipeline.artifacts import contact_sheets, export_selection
from ada_grasp_ctrl.ros_execution.settings import bridge_config
from ada_grasp_ctrl.ros_execution.snapshot import snapshot_receiver


def plan_fixture() -> dict:
    return {
        "schema_version": 1,
        "environment": "mock",
        "candidate_id": "c0001",
        "joint_names": list(JOINT_NAMES),
        "phases": list(PHASES),
        "keyframes": [[value] * 28 for value in (0.0, 0.1, 0.2, 0.3)],
        "durations_s": [2.2, 3.2, 4.2],
        "settle_time_s": 0.3,
        "interpolation": "quintic_smoothstep",
        "receiver_settings": bridge_config(),
    }


class PipelinePlanTest(unittest.TestCase):
    def test_entry_defaults_to_live_and_forwards_explicit_environment(self) -> None:
        for environment in ("mock", "hardware"):
            with tempfile.TemporaryDirectory() as directory:
                output = str(Path(directory) / "new")
                with (
                    patch("ada_grasp_ctrl.pipeline.runner.run") as run,
                    patch("ada_grasp_ctrl.pipeline.runner.Processes"),
                ):
                    self.assertEqual(
                        main(
                            [
                                "--output-root",
                                output,
                                "--environment",
                                environment,
                                "--ros-domain-id",
                                "88",
                                "--preview-only",
                            ]
                        ),
                        0,
                    )
                args, config, *_ = run.call_args.args
                self.assertEqual(args.mode, "live")
                self.assertEqual(config["environment"], environment)
                self.assertTrue(config["complete_cycle"])
                self.assertEqual(config["hold_s"], 5.0)

    def test_domains_never_silently_mix_mock_and_hardware(self) -> None:
        from ada_grasp_ctrl.ros_execution.settings import validate_domain

        self.assertEqual(validate_domain("hardware", "87"), 87)
        self.assertEqual(validate_domain("mock", 88), 88)
        for environment, value in (("mock", 87), ("hardware", None), ("mock", True), ("hardware", -1), ("mock", 88.1)):
            with self.subTest(environment=environment, value=value), self.assertRaises(ValueError):
                validate_domain(environment, value)

    def test_snapshot_discards_stale_startup_samples_without_refreshing_state(self) -> None:
        received, discarded = [], []
        runtime = SimpleNamespace(
            c={"feedback_timeout": 0.5},
            fault=None,
            node=SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10**10))),
            log=SimpleNamespace(event=lambda *args, **kwargs: discarded.append(kwargs)),
            receive=lambda *args: received.append(args),
        )
        receive = snapshot_receiver(runtime)
        message = SimpleNamespace(header=SimpleNamespace(frame_id="base_link", stamp=SimpleNamespace(sec=9, nanosec=0)))
        receive("arm", [], message)
        self.assertEqual(received, [])
        self.assertEqual(len(discarded), 1)
        self.assertIsNone(runtime.fault)
        message.header.stamp.sec = 10
        receive("arm", [], message)
        self.assertEqual(len(received), 1)
        message.header.frame_id = "wrong_frame"
        receive("arm", [], message)
        self.assertIsNotNone(runtime.fault)
        self.assertEqual(len(received), 1)

    def test_quintic_preview_reaches_each_exact_target_and_holds(self) -> None:
        plan = plan_fixture()
        validate_plan(plan)
        total = sum(plan["durations_s"]) + 0.9
        self.assertEqual(sample_plan(plan, 0), plan["keyframes"][0])
        self.assertEqual(sample_plan(plan, 100), plan["keyframes"][-1])
        self.assertAlmostEqual(sample_plan(plan, 1.1 / total * 100)[0], 0.05)
        # A quarter-time quintic differs from the legacy linear preview.
        self.assertAlmostEqual(sample_plan(plan, 0.55 / total * 100)[0], 0.0103515625)
        for index in range(1, 4):
            for actual, expected in zip(sample_plan(plan, phase_progress(plan, index)), plan["keyframes"][index]):
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(sample_plan(plan, 2.35 / total * 100), plan["keyframes"][1])

    def test_start_drift_checks_arm_and_hand_without_wrapping(self) -> None:
        start = [0.0] * 28
        check_start(start, start)
        for index, value in ((0, 0.011), (6, 0.021), (0, 6.283185307)):
            changed = start.copy()
            changed[index] = value
            with self.subTest(index=index, value=value), self.assertRaisesRegex(ValueError, "moved"):
                check_start(changed, start)

    def test_bad_plan_is_never_previewed(self) -> None:
        for field, value in (
            ("environment", "hardware"),
            ("durations_s", [float("nan")] * 3),
            ("keyframes", [[0.0] * 28] * 3),
            ("phases", list(PHASES)[::-1]),
        ):
            plan = plan_fixture()
            plan[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_only_enter_approves_and_eof_never_approves(self) -> None:
        for content, expected in ((b"\n", True), (b"q\n", False), (b"", False), (b"yes\n", False)):
            read_fd, write_fd = os.pipe()
            os.write(write_fd, content)
            os.close(write_fd)
            with os.fdopen(read_fd) as stream, patch("builtins.print"):
                self.assertEqual(wait_for_enter(SimpleNamespace(check_services=lambda: None), stream), expected)

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            report = Path(directory) / "run_report.json"
            report.write_text("existing evidence")
            self.assertEqual(main(["--output-root", directory]), 2)
            self.assertEqual(report.read_text(), "existing evidence")


class PipelineTransportTest(unittest.TestCase):
    def write_packet(
        self, root: Path, *, lift: bool = False, complete: bool = False, environment: str = "mock"
    ) -> tuple[Path, dict]:
        plan = plan_fixture()
        plan["environment"] = environment
        plan["receiver_settings"] = bridge_config({"environment": environment})
        if lift:
            plan["phases"].append("lift")
            plan["keyframes"].append([0.35] * 6 + plan["keyframes"][-1][6:])
            plan["durations_s"].append(2.2)
            plan["lift"] = {
                "positions": plan["keyframes"][-1],
                "used_for_candidate_selection": False,
                "path_checked": False,
                "collision_checked": False,
                "solver_success": True,
                "frame_id": "world",
                "axis": "+Z",
                "height_m": 0.05,
                "start_flange_wxyz": [0, 0, 0, 1, 0, 0, 0],
                "target_flange_wxyz": [0, 0, 0.05, 1, 0, 0, 0],
            }
        if complete:
            plan["phases"].extend(["hold", "release", "return"])
            plan["keyframes"].extend(
                [
                    plan["keyframes"][-1].copy(),
                    plan["keyframes"][-1][:6] + [0.0] * 22,
                    plan["keyframes"][0][:6] + [0.0] * 22,
                ]
            )
            plan["durations_s"].extend([5.0, 2.2, 2.2])
            plan["cycle"] = {
                "hold_s": 5.0,
                "hold_starts_after": "lift_succeeded",
                "release_policy": "zero_joint_open",
                "return_policy": "measured_start_arm_open_hand",
                "flange_targets_wxyz": {p: [0, 0, 0, 1, 0, 0, 0] for p in ("hold", "release", "return")},
            }
        selection = {
            "selected_candidate_id": plan["candidate_id"],
            "selected": {
                "stage_checks": {phase: {"q_rad": plan["keyframes"][i + 1]} for i, phase in enumerate(PHASES)}
            },
        }
        files = {
            "selection.json": selection,
            "state.json": {"full_positions_rad": plan["keyframes"][0], "environment": environment},
            "plan.json": plan,
            "model.json": {"fixture": True},
        }
        for name, value in files.items():
            (root / name).write_text(json.dumps(value))
        source = {
            "kind": "grasp_selection",
            "model_path": str(root / "model.json"),
            "model_sha256": digest(root / "model.json"),
            "selection_path": str(root / "selection.json"),
            "snapshot_path": str(root / "state.json"),
            "plan_path": str(root / "plan.json"),
            "plan_sha256": digest(root / "plan.json"),
            "references": {str(root / name): digest(root / name) for name in files if name != "plan.json"},
        }
        packet = {
            "schema_version": 1,
            "kind": "ada_right_joint_targets",
            "environment": environment,
            "sample_id": "test",
            "units": "rad",
            "joint_names": list(JOINT_NAMES),
            "reference_frame": "world",
            "pose_frame": "right_hand_flange",
            "kinematics": {"fixture": True},
            "source": source,
            "reviewed_plan": plan,
            "targets": [
                {
                    "phase": phase,
                    "positions": plan["keyframes"][i + 1],
                    "duration_s": plan["durations_s"][i],
                    "flange_target_wxyz": [0, 0, 0, 1, 0, 0, 0],
                }
                for i, phase in enumerate(PHASES)
            ],
        }
        path = root / "targets.json"
        if lift:
            packet["execution_phases"] = plan["phases"]
            packet["targets"].append(
                {
                    "phase": "lift",
                    "positions": plan["keyframes"][4],
                    "duration_s": plan["durations_s"][3],
                    "flange_target_wxyz": plan["lift"]["target_flange_wxyz"],
                }
            )
        if complete:
            for i, phase in enumerate(plan["phases"][4:], start=4):
                packet["targets"].append(
                    {
                        "phase": phase,
                        "positions": plan["keyframes"][i + 1],
                        "duration_s": plan["durations_s"][i],
                        "flange_target_wxyz": plan["cycle"]["flange_targets_wxyz"][phase],
                    }
                )
        self.save_packet(path, packet)
        return path, packet

    def save_packet(self, path: Path, packet: dict) -> None:
        path.write_text(json.dumps(packet))
        (path.parent / "run_manifest.yaml").write_text(
            json.dumps(
                {
                    "mode": "ada_ros_target_export",
                    "source": packet["source"],
                    "sample_id": packet["sample_id"],
                    "targets_sha256": digest(path),
                }
            )
        )

    def test_selected_targets_and_reviewed_timing_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self.write_packet(Path(directory))
            self.assertEqual(load_packet(path), expected)

    def test_lift_is_bound_to_reviewed_plan_without_selection_lift_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self.write_packet(Path(directory), lift=True)
            self.assertEqual(load_packet(path), expected)
            expected["targets"][-1]["flange_target_wxyz"] = [0, 0, 0.1, 1, 0, 0, 0]
            self.save_packet(path, expected)
            with self.assertRaisesRegex(ValueError, "Lift pose differs"):
                load_packet(path)

    def test_cycle_holds_then_opens_and_returns_without_reclosing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, packet = self.write_packet(Path(directory), lift=True, complete=True)
            self.assertEqual(load_packet(path), packet)
            plan = packet["reviewed_plan"]
            total = sum(plan["durations_s"]) + len(plan["phases"]) * plan["settle_time_s"]
            hold_start = sum(plan["durations_s"][:4]) + 4 * plan["settle_time_s"]
            for offset in (0.1, 2.5, 4.9):
                self.assertEqual(sample_plan(plan, (hold_start + offset) / total * 100), plan["keyframes"][4])
            self.assertEqual(sample_plan(plan, 100), plan["keyframes"][0][:6] + [0.0] * 22)
            for index, joint in ((5, 0), (6, 0), (6, 6), (7, 0), (7, 6)):
                changed = deepcopy(plan)
                changed["keyframes"][index][joint] += 0.01
                with self.subTest(index=index, joint=joint), self.assertRaises(ValueError):
                    validate_plan(changed)

    def test_hardware_receiver_only_accepts_exact_next_reviewed_goal(self) -> None:
        from ada_grasp_ctrl.ros_execution.review import check_reviewed_goal

        with tempfile.TemporaryDirectory() as directory:
            path, packet = self.write_packet(Path(directory), lift=True, complete=True, environment="hardware")
            self.assertEqual(load_packet(path), packet)
            config = bridge_config({"environment": "hardware"})
            for i, target in enumerate(packet["targets"]):
                check_reviewed_goal(packet, i, target["positions"], target["duration_s"], config)
            target = packet["targets"][-1]
            for index, positions, duration, settings in (
                (0, target["positions"], target["duration_s"], config),
                (6, target["positions"], 1.0, config),
                (7, target["positions"], target["duration_s"], config),
                (6, target["positions"], target["duration_s"], bridge_config()),
                (6, target["positions"], target["duration_s"], {**config, "arm_speed": 0.3}),
            ):
                with self.assertRaises(ValueError):
                    check_reviewed_goal(packet, index, positions, duration, settings)

    def test_target_or_duration_change_is_rejected_even_with_new_packet_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, original = self.write_packet(Path(directory))
            for field, value in (("positions", [0.0] * 28), ("duration_s", 1.0)):
                packet = deepcopy(original)
                packet["targets"][1][field] = value
                self.save_packet(path, packet)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_packet(path)

    def test_selection_source_changed_after_preview_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.write_packet(root)
            (root / "selection.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed since preview"):
                load_packet(path)


class PipelineImagesTest(unittest.TestCase):
    def test_seven_feasible_grasps_produce_two_six_slot_pages(self) -> None:
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluations = [
                {
                    "candidate_id": f"c{i:04d}",
                    "best_solution_index": 0,
                    "options": [
                        {
                            "solution_index": 0,
                            "score": float(i),
                            "stages": {phase: {"q_rad": [0.0] * 28} for phase in PHASES},
                        }
                    ],
                }
                for i in range(7)
            ]
            (root / "candidate_evaluations.json").write_text(json.dumps(evaluations))
            entry = {
                "run": str(root),
                "group": "test",
                "keyframes": np.zeros((4, 28)),
                "result": {"selected_candidate_id": "c0006"},
            }

            def render(entries: list, output: Path) -> None:
                Image.new("RGB", (1000, 750)).save(output / f"group{entries[0]['group']}_closeup.png")

            with patch("ada_grasp_ctrl.pipeline.artifacts.render_images", side_effect=render):
                result = contact_sheets(entry, root / "images")
            self.assertEqual(result["feasible"], 7)
            self.assertEqual(len(result["pages"]), 2)
            self.assertEqual(result["selected"], "c0006")
            self.assertTrue(all(Path(page).is_file() for page in result["pages"]))

    def test_no_feasible_result_cannot_be_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "export"
            with self.assertRaisesRegex(ValueError, "No feasible"):
                export_selection({"keyframes": None}, Path(directory) / "state.json", destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
