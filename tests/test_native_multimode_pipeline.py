"""Native model contract, ranked live selection and export without physical devices."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ada_grasp_ctrl.experiments.fixed_camera_grasp import decode_candidates
from ada_grasp_ctrl.experiments.native_multimode_grasp import metadata_contract
from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.grasp_selection.contracts import validate_grasp_candidate
from ada_grasp_ctrl.grasp_selection.multi_arm import hand_names
from ada_grasp_ctrl.pipeline import dual_runner
from ada_grasp_ctrl.pipeline.dual_inference import from_native_record
from ada_grasp_ctrl.pipeline.dual_selection import canonical_batch, select_live_hierarchical
from ada_grasp_ctrl.pipeline.grasp_models import SOURCE_MODES, SINGLE_MODES, configure_model
from ada_grasp_ctrl.sharpa_mirror import SharpaHandMirror
from dual_selector_fixtures import LiveFixture
from test_phase2_pipeline import candidate, FakeRobot, FakeSim


class NativeFixture(LiveFixture):
    """Mock only model/scene/device boundaries; use real conversion, selection and packet export."""

    def __init__(self, root: Path, *, scores: dict | None = None, available: list | None = None) -> None:
        super().__init__(root)
        self.scores = scores or dict(zip(SOURCE_MODES, [0.94, 0.93, 0.92, 0.99, 0.98]))
        self.available = available if available is not None else list(SOURCE_MODES)
        self.costs = {m: dict(right=0.4, left=0.1) for m in SINGLE_MODES}
        refs = self.config["inference_model"]
        metadata = root / "metadata"
        for mode in SOURCE_MODES:
            (metadata / mode).mkdir(parents=True)
            write_json(metadata / mode / "metadata.json", dict(fixture=True))
        self.config["grasp_models"]["native_multimode"] = dict(
            checkpoint=refs["checkpoint"]["path"],
            training_config=refs["training_config"]["path"],
            metadata_root=str(metadata),
        )

    def run(self, command: list[str], label: str, **kw: object) -> None:
        if label != "inference_native":
            return super().run(command, label, **kw)
        self.calls.append((label, command))
        output = Path(command[command.index("--output-root") + 1])
        output.mkdir()
        import yaml

        config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
        records = []
        for mode in self.available:
            if mode not in SINGLE_MODES:
                continue
            for side in config["enabled_arms"]:
                source = candidate("right", self.costs[mode][side], "fixture")
                records.append(
                    dict(
                        candidate_id=mode + ":c0000",
                        side=side,
                        source_model_mode=mode,
                        raw_candidate_index=0,
                        stages=source["targets"]["right"],
                        provenance=dict(**self.config["inference_model"], model_score=self.scores[mode]),
                    )
                )
        ranked = sorted(self.available, key=lambda m: (-self.scores[m], m))
        write_json(
            output / "candidates.json",
            dict(
                schema_version=1,
                grasp_model="native_multimode",
                checkpoint=self.config["inference_model"]["checkpoint"],
                availability_scores=self.scores,
                ranked_available_modes=ranked,
                eligible_modes=[m for m in ranked if m in SINGLE_MODES],
                unsupported_modes={
                    m: "unsupported_simultaneous_hardware_execution" for m in ranked if m not in SINGLE_MODES
                },
                right_records=records,
                conversion_rejections=[],
                timings=dict(total_s=0),
            ),
        )

    def invoke_native(self, options: list[str] | None = None) -> int:
        return self.invoke(["--grasp-model", "native_multimode", "--selector", "hierarchical", *(options or [])])


class NativePipelineTests(unittest.TestCase):
    def test_viewer_displays_source_mode_execution_mode_and_five_scores(self):
        from ada_grasp_ctrl.pipeline.dual_viewer import model_summary

        text = model_summary(
            dict(
                grasp_model="native_multimode",
                checkpoint=dict(path="/model/step_200000.pth"),
                selected_source_mode="right_three",
                execution_mode="left_three",
                effort=0.12,
                availability_scores=dict.fromkeys(SOURCE_MODES, 0.95),
                ranked_available_modes=list(SOURCE_MODES),
                eligible_modes=list(SINGLE_MODES),
            )
        )
        for value in (*SOURCE_MODES, "left_three", "native_multimode", "step_200000.pth", "0.120000"):
            self.assertIn(value, text)

    def test_ranked_mode_compares_both_sides_then_stops_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = NativeFixture(Path(directory) / "run")
            fixture.costs["right_full"] = dict(left=0.01, right=0.01)
            with patch.object(fixture.robot, "solve_arm_ik", wraps=fixture.robot.solve_arm_ik) as ik:
                self.assertEqual(fixture.invoke_native(), 0)
            audit = fixture.artifact("hierarchical_selection.json")
            self.assertEqual(audit["selected_mode"], "right_two")
            self.assertEqual(audit["selected_side"], "left")
            self.assertEqual(audit["execution_mode"], "left_two")
            self.assertAlmostEqual(audit["effort"], 0.06)
            self.assertEqual({c.args[1] for c in ik.call_args_list}, {"left", "right"})
            for entry in audit["mode_trace"]:
                if entry["mode"].startswith("both_"):
                    self.assertEqual(entry["reasons"], ["unsupported_simultaneous_hardware_execution"])
                    self.assertEqual(entry["evaluated_candidate_count"], 0)
                elif entry["mode"] != "right_two":
                    self.assertEqual(entry["evaluated_candidate_count"], 0)
                    self.assertIn("not_evaluated_after_selection", entry["reasons"])
            packet = load_packet(fixture.root / "run/export/targets.json")
            self.assertEqual(packet["sample_id"], "left:right_two:c0000")
            self.assertEqual(packet["side"], "left")
            self.assertEqual(
                packet["execution_phases"],
                ["above_object", "pregrasp", "grasp", "squeeze", "lift", "hold", "release", "return"],
            )
            self.assertEqual(
                [label for label, _ in fixture.calls if label.startswith("inference")], ["inference_native"]
            )
            self.assertIn(str(fixture.root / "run/inference_native/candidates.json"), packet["source"]["references"])

    def test_infeasible_top_mode_falls_back_and_arms_restrict_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            for side in ("left", "right"):
                fixture = NativeFixture(Path(directory) / side)
                fixture.costs["right_two"] = dict(left=6, right=6)
                with patch.object(fixture.robot, "solve_arm_ik", wraps=fixture.robot.solve_arm_ik) as ik:
                    self.assertEqual(fixture.invoke_native(["--arms", side]), 0)
                audit = fixture.artifact("hierarchical_selection.json")
                self.assertEqual(audit["selected_mode"], "right_three")
                self.assertEqual(audit["selected_side"], side)
                self.assertEqual({c.args[1] for c in ik.call_args_list}, {side})
                self.assertTrue(
                    all(r["required_arms"] == [side] for r in fixture.artifact("candidates.json")["candidates"])
                )
                self.assertEqual((fixture.root / "run/native_mirrored_cloud.npy").exists(), side == "left")

    def test_only_unsupported_available_never_forces_single_mode_or_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = NativeFixture(Path(directory) / "run", available=["both_full"])
            with patch.object(fixture.robot, "solve_arm_ik", side_effect=AssertionError("IK must not run")):
                self.assertEqual(fixture.invoke_native(), 1)
            self.assertFalse((fixture.root / "run/export").exists())
            self.assertEqual(fixture.artifact("hierarchical_selection.json")["status"], "NO_FEASIBLE_GRASP")

    def test_model_legacy_selector_rejected_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = NativeFixture(Path(directory) / "run")
            self.assertEqual(fixture.invoke(["--grasp-model", "native_multimode"]), 2)
            self.assertEqual(fixture.calls, [])
            self.assertFalse((fixture.root / "run").exists())

    def test_legacy_model_default_preserves_exact_references_and_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = (LiveFixture(Path(directory) / label) for label in ("default", "explicit"))
            self.assertEqual(first.invoke(), 0)
            self.assertEqual(second.invoke(["--grasp-model", "right_full"]), 0)
            for fixture in (first, second):
                self.assertEqual(fixture.artifact("run_report.json")["grasp_model"], "right_full")
                self.assertEqual(
                    [name for name, _ in fixture.calls if name.startswith("inference")],
                    ["inference_left", "inference_right"],
                )
                self.assertEqual(
                    [r["candidate_id"] for r in fixture.artifact("candidates.json")["candidates"]],
                    ["left:c0000", "right:c0000"],
                )
        config = dual_runner.configuration("src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml")
        refs = deepcopy(config["inference_model"])
        configure_model(config, "right_full", "legacy")
        self.assertEqual(config["inference_model"], refs)
        self.assertEqual(
            refs["checkpoint"]["sha256"], "92fc78e580753a3659116e867e075e91d75a342176abd2434b9d98828b3083d6"
        )

    def test_ties_unique_ids_and_source_execution_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = NativeFixture(Path(directory) / "run", scores=dict.fromkeys(SOURCE_MODES, 0.95))
            self.assertEqual(fixture.invoke_native(), 0)
            rows = fixture.artifact("candidates.json")["candidates"]
            self.assertEqual(len({r["candidate_id"] for r in rows}), 6)
            diagnostics = fixture.artifact("candidates.json")
            reports = [
                select_live_hierarchical(
                    FakeRobot(), FakeSim(), order, ["left", "right"], np.zeros(56), {}, native_diagnostics=diagnostics
                )[2]
                for order in (rows, rows[::-1])
            ]
            self.assertEqual(reports[0], reports[1])
            self.assertEqual(reports[0]["selected_mode"], "right_full")
            bad = deepcopy(rows)
            bad[0]["execution_mode"] = "both_full"
            with self.assertRaises(ValueError):
                canonical_batch(bad, ["left", "right"], native=True)
            bad = deepcopy(rows)
            bad[0]["mode"] = "both_full"
            with self.assertRaisesRegex(ValueError, "unsupported_simultaneous"):
                canonical_batch(bad, ["left", "right"], native=True)


class NativeConversionTests(unittest.TestCase):
    def test_all_three_modes_preserve_stages_full_fingers_and_world_mirror(self):
        mirror = SharpaHandMirror()
        plane = dict(normal_world=[0, 1, 0], point_world_m=[0.2, 0.3, 0.1])
        prep = dict(frame_id="world", scale=1.0, center_world_m=[0.4, -0.2, 0.1])
        raw = np.zeros((1, 3, 58))
        raw[:, :, 3] = raw[:, :, 32] = 1
        limits = np.asarray(mirror.config["right_limits"])
        fingers = limits.mean(axis=1)
        for stage in range(3):
            raw[0, stage, :3] = [0.03, 0.04, stage * 0.01]
            raw[0, stage, 7:29] = fingers * (0.5 + stage * 0.1)
        world_color = np.eye(4)
        world_color[:3, 3] = [0.7, -0.3, 0.6]
        for mode in SINGLE_MODES:
            decoded = decode_candidates(raw, np.array([1.0]), list(hand_names("right")), prep, world_color)[0]
            for side in ("right", "left"):
                record = dict(
                    decoded,
                    candidate_id=mode + ":c0007",
                    raw_candidate_index=7,
                    source_model_mode=mode,
                    side=side,
                    provenance=dict(checkpoint={}, metadata={}),
                )
                row = validate_grasp_candidate(from_native_record(record, plane))
                self.assertEqual(row["mode"], mode)
                self.assertEqual(row["candidate_id"], f"{side}:{mode}:c0007")
                for i, stage in enumerate(("pregrasp", "grasp", "squeeze")):
                    target = row["targets"][side][stage]
                    expected = np.asarray(decoded["stages"][stage]["T_world_hand"])
                    q = raw[0, i, 7:29]
                    if side == "left":
                        expected = mirror.mirror_wrist_pose(expected, plane["normal_world"], plane["point_world_m"])
                        q = mirror.right_to_left(q)
                    np.testing.assert_allclose(target["T_world_hand"], expected)
                    np.testing.assert_allclose([target["joint_positions_rad"][n] for n in hand_names(side)], q)
                    self.assertEqual(len(target["joint_positions_rad"]), 22)
        with self.assertRaises(ValueError):
            decode_candidates(raw[:, :, :-1], np.array([1.0]), list(hand_names("right")), prep, world_color)
        record["source_model_mode"] = "both_three"
        with self.assertRaisesRegex(ValueError, "unsupported_simultaneous"):
            from_native_record(record, plane)

    def test_metadata_rejects_reordered_finger_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in SOURCE_MODES:
                (root / mode).mkdir()
                dual = mode.startswith("both_")
                names = list(hand_names("right"))
                if dual:
                    axes = ("TransX", "TransY", "TransZ", "RotX", "RotY", "RotZ")
                    names = [f"ra_{a}J" for a in axes] + names + [f"la_{a}J" for a in axes] + list(hand_names("left"))
                write_json(
                    root / mode / "metadata.json",
                    dict(
                        hand_family="sharpa_dual",
                        grasp_type=mode,
                        joint_names=names,
                        wrist_body_names=["right_hand_flange", "left_hand_flange"][: 2 if dual else 1],
                    ),
                )
            self.assertEqual(set(metadata_contract(root)), set(SOURCE_MODES))
            path = root / "right_two/metadata.json"
            import json

            data = json.loads(path.read_text())
            data["joint_names"].reverse()
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "metadata contract"):
                metadata_contract(root)


if __name__ == "__main__":
    unittest.main()
