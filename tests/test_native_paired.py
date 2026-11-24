"""Real native decoding fixtures plus deterministic paired IK/transport failure tests."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.native_multimode_grasp import decode_bimanual
from ada_grasp_ctrl.grasp_selection.contracts import validate_grasp_candidate
from ada_grasp_ctrl.grasp_selection.multi_arm import select_paired_candidates, solve_paired, forbidden_contacts
from ada_grasp_ctrl.pipeline.dual_inference import from_native_record
from ada_grasp_ctrl.pipeline.dual_selection import select_live_hierarchical
from ada_grasp_ctrl.pipeline.grasp_models import SOURCE_MODES
from ada_grasp_ctrl.pipeline.shared_artifacts import shared_hardware_packet, shared_mock_packet
from ada_grasp_ctrl.pipeline.shared_timing import build_stage_schedule, validate_stage_schedule
from ada_grasp_ctrl.ros_execution.shared_hardware import load_hardware_packet, validate_hardware_packet
from ada_grasp_ctrl.ros_execution.shared_packet import validate_shared_packet, stage_reference
from ada_grasp_ctrl.ros_execution.shared_executor import execute_shared, execute_hardware
from ada_grasp_ctrl.ros_execution.shared_stream import SharedStageStream
from ada_grasp_ctrl.robot_environment.model import side_slice
from shared_executor_fixtures import schedule, description, feedback, limits
from test_phase2_pipeline import FakeRobot, FakeSim, candidate
from test_native_multimode_pipeline import NativeFixture
from dual_selector_fixtures import AnalyticalSide


class PairedLiveFixture(NativeFixture):
    """Analytical reachable geometry for exercising the real CLI/export, never physical evidence."""

    def __init__(self, root):
        super().__init__(root, available=["both_full"])

        def fk(q, side):
            result = np.eye(4)
            result[:3, 3] = np.asarray(q)[side_slice(side)][:3]
            return result

        def solve(q, side, pose, **kw):
            q = np.asarray(q).copy()
            block = side_slice(side)
            q[block.start : block.start + 3] = np.asarray(pose)[:3, 3]
            return q, dict(accepted=True)

        self.robot.fk = fk
        self.robot.solve_arm_ik = solve

    def run(self, command, label, **kw):
        super().run(command, label, **kw)
        if label == "inference_native":
            path = Path(command[command.index("--output-root") + 1]) / "candidates.json"
            data = json.loads(path.read_text())
            row = paired()
            data["right_records"] = [
                dict(
                    source_model_mode="both_full",
                    side="right",
                    candidate_id="both_full:c0000",
                    raw_candidate_index=0,
                    targets=row["targets"],
                    log_probability=0.5,
                    provenance=dict(**self.config["inference_model"], model_score=0.99),
                )
            ]
            data.update(eligible_modes=["both_full"], unsupported_modes={})
            path.write_text(json.dumps(data))


class PairedLiveTests(unittest.TestCase):
    def test_cli_exports_full_paired_cycle_and_cancel_never_launches_sender(self):
        from ada_grasp_ctrl.pipeline import paired_artifacts
        from ada_grasp_ctrl.pipeline.dual_packet import load_execution_packet

        with tempfile.TemporaryDirectory() as directory:
            for export_only in (True, False):
                fixture = PairedLiveFixture(Path(directory) / str(export_only))
                with patch.object(paired_artifacts, "SideAdapter", AnalyticalSide):
                    code = fixture.invoke(
                        [
                            "--grasp-model",
                            "native_multimode",
                            "--selector",
                            "hierarchical",
                            "--grasp-mode",
                            "both_only",
                        ],
                        export_only=export_only,
                    )
                self.assertEqual(code, 0)
                packet = load_execution_packet(fixture.root / "run/export/targets.json")
                self.assertEqual(packet["plan_identity"]["active_arms"], ["left", "right"])
                self.assertEqual(
                    [s["phase"] for s in packet["schedule"]["stages"]],
                    ["above_object", "pregrasp", "grasp", "squeeze", "lift", "hold", "release", "return"],
                )
                lift = next(s for s in packet["schedule"]["stages"] if s["phase"] == "lift")
                self.assertEqual(len(lift["waypoints_rad"]), 5)
                self.assertFalse(any(label in ("sender", "receiver", "shared_executor") for label, _ in fixture.calls))
                report = fixture.artifact("run_report.json")
                self.assertEqual(report["required_arms"], ["left", "right"])
                self.assertEqual(report["status"], "exported" if export_only else "canceled_before_execution")

    def test_incompatible_both_only_fails_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            for args in (
                ["--grasp-mode", "both_only"],
                [
                    "--grasp-model",
                    "native_multimode",
                    "--selector",
                    "hierarchical",
                    "--arms",
                    "right",
                    "--grasp-mode",
                    "both_only",
                ],
            ):
                fixture = PairedLiveFixture(Path(directory) / str(len(args)))
                self.assertEqual(fixture.invoke(args), 2)
                self.assertEqual(fixture.calls, [])


def paired(left=0.2, right=0.3, mode="both_full"):
    row = candidate("left", left)
    row.update(candidate_id=mode + ":c0000", mode=mode, execution_mode=mode, required_arms=["left", "right"])
    row["targets"].update(candidate("right", right)["targets"])
    row["provenance"].update(
        source_model_mode=mode, generator=mode, checkpoint={"sha256": "fixture"}, metadata={"fixture": True}
    )
    return row


class PairedDecoderTests(unittest.TestCase):
    def test_real_200k_topk_pair_order_center_rotation_and_identity(self):
        root = Path(__file__).parent / "fixtures/native_paired_200k"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["checkpoint"]["iteration"], 200000)
        with np.load(root / "samples.npz") as data:
            for i in (1, 2):
                mode = SOURCE_MODES[int(data["type_ids"][i]) - 1]
                self.assertTrue(data["available_mask"][int(data["type_ids"][i]) - 1])
                prep = dict(frame_id="world", scale=1.0, center_world_m=data["center_world_m"].tolist())
                targets = decode_bimanual(data["poses"][i], manifest["metadata"][mode]["joint_names"], prep)
                index = int(data["raw_indices"][i])
                record = dict(
                    source_model_mode=mode,
                    side="right",
                    targets=targets,
                    raw_candidate_index=index,
                    candidate_id=f"{mode}:c{index:04d}",
                    log_probability=float(data["log_probabilities"][i]),
                    provenance=dict(
                        checkpoint=manifest["checkpoint"],
                        metadata=manifest["metadata"][mode],
                        model_score=float(data["availability_scores"][int(data["type_ids"][i]) - 1]),
                    ),
                )
                row = validate_grasp_candidate(from_native_record(record, {}))
                self.assertEqual(row, from_native_record(record, {"ignored_for_pair": True}))
                self.assertEqual(row["required_arms"], ["left", "right"])
                self.assertEqual(row["candidate_id"], f"{mode}:c{index:04d}")
                for side, offset, joint_offset in (("right", 0, 6), ("left", 29, 34)):
                    for j, phase in enumerate(("pregrasp", "grasp", "squeeze")):
                        raw = data["poses"][i, j, offset : offset + 29]
                        target = row["targets"][side][phase]
                        pose = np.array(target["T_world_hand"])
                        np.testing.assert_allclose(pose[:3, 3], raw[:3] + data["center_world_m"], atol=3e-8)
                        np.testing.assert_allclose(pose[:3, :3], Rotation.from_quat(raw[[4, 5, 6, 3]]).as_matrix())
                        names = manifest["metadata"][mode]["joint_names"][joint_offset : joint_offset + 22]
                        np.testing.assert_array_equal([target["joint_positions_rad"][n] for n in names], raw[7:29])
                bad = data["poses"][i].copy()
                bad[0, 32:36] = 0
                with self.assertRaisesRegex(ValueError, "quaternion"):
                    decode_bimanual(bad, manifest["metadata"][mode]["joint_names"], prep)
                with self.assertRaisesRegex(ValueError, "metadata"):
                    decode_bimanual(data["poses"][i], manifest["metadata"][mode]["joint_names"][::-1], prep)


class PairedSelectionTests(unittest.TestCase):
    def test_combined_state_effort_stages_and_seed_order(self):
        row = paired()
        row["targets"]["left"]["pregrasp"]["T_world_hand"][0][3] = 0.1
        row["targets"]["right"]["squeeze"]["T_world_hand"][0][3] = 0.4
        current = np.zeros(56)
        current[:6] = 0.05
        result = select_paired_candidates(FakeRobot(), FakeSim(), [row], ["left", "right"], current)
        chosen = result["selected"]
        self.assertAlmostEqual(chosen["effort"], 6 * (0.15**2 + 0.3**2))
        self.assertEqual(chosen["seed_index"], 0)
        for phase, left, right in (("pregrasp", 0.1, 0.3), ("grasp", 0.2, 0.3), ("squeeze", 0.2, 0.4)):
            np.testing.assert_allclose(np.array(chosen["stages"][phase]["q56"])[:6], left)
            np.testing.assert_allclose(np.array(chosen["stages"][phase]["q56"])[28:34], right)

    def test_second_ik_cannot_undo_first_and_interarm_collision_rejected(self):
        robot = FakeRobot()
        solve = robot.solve_arm_ik

        def corrupt(q, side, target, **kw):
            solved, diag = solve(q, side, target, **kw)
            if side == "right":
                solved[:6] = 0
            return solved, diag

        with patch.object(robot, "solve_arm_ik", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "outside"):
                solve_paired(
                    robot, np.zeros(56), {s: paired()["targets"][s]["grasp"]["T_world_hand"] for s in ("left", "right")}
                )
        collision = dict(
            category="inter_robot",
            bodies=["left_J6_link", "right_J6_link"],
            sides=["left", "right"],
            penetration_m=0.001,
            hand_bodies=[],
        )
        sim = FakeSim()
        with patch.object(sim, "check_static_contacts", return_value=[collision]):
            result = select_paired_candidates(robot, sim, [paired()], ["left", "right"], np.zeros(56))
        self.assertIsNone(result["selected"])
        self.assertTrue(all("collision" in a["reasons"][0] for a in result["evaluations"][0]["attempts"]))

    def test_both_active_object_contacts_retain_phase_policy(self):
        rows = [
            dict(
                category="robot_object",
                bodies=[f"{s}_finger", "object"],
                sides=[s],
                hand_bodies=[f"{s}_finger"],
                penetration_m=0.02,
            )
            for s in ("left", "right")
        ]
        self.assertEqual(forbidden_contacts(rows, ["left", "right"], phase="grasp"), [])
        self.assertEqual(len(forbidden_contacts(rows, ["left", "right"], phase="pregrasp")), 2)
        self.assertEqual(len(forbidden_contacts(rows, "left", phase="grasp")), 1)

    def test_ranking_filter_fallback_disabled_arm_and_no_manufactured_modes(self):
        single = candidate("right", 0.01, "right_full:c0000")
        single.update(mode="right_full", execution_mode="right_full")
        single["provenance"].update(
            generator="right_full",
            source_model_mode="right_full",
            checkpoint={"fixture": True},
            metadata={"fixture": True},
        )
        diagnostics = dict(
            availability_scores=dict(zip(SOURCE_MODES, [0.1, 0.2, 0.95, 0.1, 0.99])),
            ranked_available_modes=["both_full", "right_full"],
        )

        def run(rows, arms=("left", "right"), mode="auto"):
            return select_live_hierarchical(
                FakeRobot(),
                FakeSim(),
                rows,
                list(arms),
                np.zeros(56),
                {},
                native_diagnostics=diagnostics,
                grasp_mode=mode,
            )[2]

        self.assertEqual(run([single, paired()])["selected_mode"], "both_full")
        self.assertEqual(run([single, paired(left=6)])["selected_mode"], "right_full")
        self.assertEqual(run([single, paired(left=6)], mode="both_only")["status"], "NO_FEASIBLE_GRASP")
        self.assertEqual(run([single, paired()], arms=["right"])["selected_mode"], "right_full")
        diagnostics["ranked_available_modes"] = ["right_full"]
        self.assertEqual(run([single, paired()], mode="both_only")["status"], "NO_FEASIBLE_GRASP")
        diagnostics["ranked_available_modes"] = ["right_full", "both_full"]
        diagnostics["availability_scores"]["right_full"] = 1.0
        self.assertEqual(run([single, paired()])["selected_mode"], "right_full")
        self.assertEqual(run([single, paired()], mode="both_only")["selected_mode"], "both_full")


def waypoint_schedule():
    base = schedule(full=True)
    starts = {
        s: dict(zip(stage["joint_names"], stage["start_positions_rad"]))
        for s, stage in base["stages"][0]["arms"].items()
    }
    targets = {
        stage["phase"]: {s: dict(zip(v["joint_names"], v["positions_rad"])) for s, v in stage["arms"].items()}
        for stage in base["stages"]
    }
    end = next(s for s in base["stages"] if s["phase"] == "lift")
    points = [{s: v["positions_rad"].copy() for s, v in end["arms"].items()} for _ in range(2)]
    # Intermediate targets differ even when endpoints coincide, testing the actual waypoint path.
    points[0]["left"][0] += 0.01
    points[0]["right"][0] -= 0.02
    return build_stage_schedule(base["plan"], starts, targets, hold_s=0.5, lift_waypoints_rad=points), starts


class SharedHardwareTests(unittest.TestCase):
    def test_full_hardware_numeric_cycle_visits_lift_waypoints_on_one_clock(self):
        plan, starts = waypoint_schedule()
        hardware = shared_hardware_packet(
            plan, starts, description(), dict(plan_path="unused", references={"unused": "x"})
        )
        stream = SharedStageStream(hardware, limits())
        positions = {s: [0.0] * 28 for s in ("left", "right")}
        stream.start(feedback(0, positions), 0)
        phases, lift_left = [], []
        for tick in range(1, 10000):
            now = tick * 0.01
            frame = stream.tick(now, feedback(now, positions))
            if frame:
                positions = frame["commands"]
                if not phases or phases[-1] != frame["phase"]:
                    phases.append(frame["phase"])
                if frame["phase"] == "lift":
                    lift_left.append(frame["reference"]["left"][0])
            if stream.done:
                break
        self.assertTrue(stream.done)
        self.assertEqual(phases, [s["phase"] for s in plan["stages"]])
        self.assertGreater(max(lift_left), 0.055)
        self.assertEqual(positions, {s: [0.0] * 28 for s in ("left", "right")})

    def test_waypoint_schedule_transport_preview_and_hardware_numeric_stream(self):
        from ada_grasp_ctrl.pipeline.dual_viewer import sample_shared_preview

        plan, starts = waypoint_schedule()
        self.assertEqual(validate_stage_schedule(plan), plan)
        packet = shared_mock_packet(plan, starts, description())
        self.assertEqual(validate_shared_packet(packet), packet)
        stage = next(s for s in plan["stages"] if s["phase"] == "lift")
        t = stage["segment_durations_s"][0]
        for side in ("left", "right"):
            actual = stage_reference(stage, side, stage["arms"][side]["start_positions_rad"], t, stage["duration_s"])
            np.testing.assert_allclose(actual, stage["waypoints_rad"][0][side])
        q = sample_shared_preview(packet, (stage["stage_start_time"] + t) / plan["end_time"] * 100)
        np.testing.assert_allclose(q[:28], stage["waypoints_rad"][0]["left"])
        np.testing.assert_allclose(q[28:], stage["waypoints_rad"][0]["right"])
        hardware = shared_hardware_packet(
            plan, starts, description(), dict(plan_path="unused", references={"unused": "x"})
        )
        stream = SharedStageStream(hardware, limits())
        positions = {s: [0.0] * 28 for s in ("left", "right")}
        stream.start(feedback(0, positions), 0)
        frame = stream.tick(0.01, feedback(0.01, positions))
        self.assertEqual(len(frame["arm"]["positions_rad"]), 12)
        self.assertEqual({s: len(h["positions_rad"]) for s, h in frame["hands"].items()}, {"left": 22, "right": 22})

    def test_mock_hardware_scope_and_review_cannot_be_confused(self):
        plan, starts = waypoint_schedule()
        mock = shared_mock_packet(plan, starts, description())
        hardware = shared_hardware_packet(
            plan, starts, description(), dict(plan_path="unused", references={"unused": "x"})
        )
        with self.assertRaises(ValueError):
            validate_shared_packet(hardware)
        with self.assertRaises(ValueError):
            validate_hardware_packet(mock)
        with self.assertRaises(ValueError):
            execute_shared(hardware, None, threading.Event())
        with self.assertRaisesRegex(ValueError, "identity"):
            execute_hardware(hardware, None, threading.Event(), reviewed_sha256="wrong")
        with self.assertRaises(ValueError):
            execute_hardware(mock, None, threading.Event(), reviewed_sha256="wrong")

    def test_hardware_source_and_schedule_identity_are_bound(self):
        plan, starts = waypoint_schedule()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "plan.json"
            source.write_text(json.dumps(plan))
            packet = shared_hardware_packet(
                plan,
                starts,
                description(),
                dict(plan_path=str(source), references={str(source): hashlib.sha256(source.read_bytes()).hexdigest()}),
            )
            path = root / "targets.json"
            path.write_text(json.dumps(packet))
            self.assertEqual(load_hardware_packet(path), packet)
            changed = deepcopy(packet)
            changed["plan_identity"]["candidate_id"] = "another"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "identity"):
                load_hardware_packet(path)
            path.write_text(json.dumps(packet))
            source.write_text("{}")
            with self.assertRaisesRegex(ValueError, "source changed"):
                load_hardware_packet(path)


if __name__ == "__main__":
    unittest.main()
