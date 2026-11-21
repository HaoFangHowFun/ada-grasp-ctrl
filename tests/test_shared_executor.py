"""Atomic paired preparation, shared stage completion and explicit transport gates."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ada_grasp_ctrl.experiments.right_ros_contract import validate_packet, side_names
from ada_grasp_ctrl.ros_execution.shared_packet import SIDES, validate_shared_packet, load_shared_packet
from ada_grasp_ctrl.ros_execution.shared_stream import SharedStageStream
from shared_executor_fixtures import packet, limits, feedback, schedule, description


def simulate(active=SIDES, *, late=False, hand_error=False, full=False):
    events, frames = [], []
    stream = SharedStageStream(
        packet(active, full=full), limits(), lambda kind, **data: events.append(dict(kind=kind, **data))
    )
    measured = {s: [0.0] * 28 for s in SIDES}
    stream.start(feedback(0, measured), 0)
    for tick in range(1, 6001):
        now = tick / 100
        frame = stream.tick(now, feedback(now, measured))
        if frame:
            frames.append(frame)
            for side in active:
                values = frame["commands"][side].copy()
                if late and side == "right" and frame["index"] == 0 and now < frame["stage_end_time"] + 0.7:
                    values[:6] = [q - 0.03 for q in values[:6]]
                if hand_error:
                    values[6:] = [0.0] * 22
                measured[side] = values
        if stream.done:
            return stream, events, frames
    raise AssertionError("Fixture did not finish")


class SharedPacketTests(unittest.TestCase):
    @unittest.skipUnless(
        all(importlib.util.find_spec(name) for name in ("pinocchio", "mujoco")),
        "Ada exporter requires the maintained numeric environment",
    )
    def test_task3_export_matches_ros_numeric_projection(self):
        from ada_grasp_ctrl.pipeline.shared_artifacts import shared_mock_packet

        for active in (("left",), ("right",), SIDES):
            original = schedule(active, full=True)
            snapshot = {s: dict.fromkeys(sum(side_names(s), ()), 0.0) for s in SIDES}
            self.assertEqual(shared_mock_packet(original, snapshot, description()), packet(active, full=True))

    def test_packet_rejects_missing_side_wrong_names_duration_and_hardware(self):
        for fault in ("hardware", "ready", "missing", "order", "duration", "clock", "active", "snapshot", "limit"):
            data = packet()
            if fault == "hardware":
                data["environment"] = "hardware"
            elif fault == "ready":
                data["hardware_ready"] = True
            elif fault == "missing":
                data["schedule"]["stages"][0]["arms"].pop("left")
            elif fault == "order":
                data["schedule"]["stages"][0]["arms"]["right"]["joint_names"].reverse()
            elif fault == "duration":
                data["schedule"]["stages"][0]["duration_s"] /= 2
            elif fault == "clock":
                data["schedule"]["stages"][1]["stage_start_time"] += 1
            elif fault == "active":
                data["plan_identity"]["active_arms"] = ["right"]
            elif fault == "snapshot":
                data["snapshot"].pop("left")
            else:
                data["schedule"]["policies_by_arm"]["left"]["arm_speed"] = 1
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                validate_shared_packet(data)
        with self.assertRaises(ValueError):
            validate_packet(packet())
        with self.assertRaises(ValueError):
            validate_shared_packet(dict(schema_version=1, kind="ada_right_joint_targets", environment="mock"))

    def test_round_trip_and_domain87_rejected_before_ros_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.json"
            path.write_text(json.dumps(packet()))
            self.assertEqual(load_shared_packet(path), packet())
            command = [
                sys.executable,
                "-m",
                "ada_grasp_ctrl.ros_execution.shared_executor",
                "--targets",
                str(path),
                "--output-root",
                str(Path(directory) / "run"),
                "--ros-domain-id",
                "87",
            ]
            result = subprocess.run(command, env={**os.environ, "ROS_DOMAIN_ID": "87"}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("mock execution cannot use domain 87", result.stderr)
            self.assertFalse((Path(directory) / "run").exists())


class SharedStreamTests(unittest.TestCase):
    def test_single_active_side_keeps_passive_arm_stationary_without_passive_hand_commands(self):
        for active in ("left", "right"):
            stream, events, frames = simulate((active,))
            passive_slice = slice(6, 12) if active == "left" else slice(0, 6)
            self.assertTrue(stream.done)
            self.assertEqual(
                [e["phase"] for e in events if e["kind"] == "stage_completed"], ["pregrasp", "grasp", "squeeze"]
            )
            for frame in frames:
                self.assertEqual(set(frame["hands"]), {active})
                self.assertEqual(frame["arm"]["positions_rad"][passive_slice], [0.0] * 6)

    def test_both_active_use_one_clock_and_asymmetric_shared_duration(self):
        stream, events, frames = simulate(full=True)
        self.assertTrue(stream.done)
        self.assertEqual(stream.index, 8)
        first = frames[0]
        self.assertEqual(first["duration_s"], 5.35 if first["phase"] == "pregrasp" else 2.2)
        for frame in frames:
            self.assertEqual(set(frame["hands"]), set(SIDES))
            self.assertEqual(len(frame["arm"]["positions_rad"]), 12)
            self.assertAlmostEqual(
                frame["progress"], min(1, (frame["monotonic_time"] - frame["stage_start_time"]) / frame["duration_s"])
            )
        starts = [e for e in events if e["kind"] == "stage_started"]
        self.assertEqual(starts[1]["duration_s"], 5.35)
        for prior, following in zip(frames, frames[1:]):
            self.assertGreaterEqual(following["monotonic_time"] - prior["monotonic_time"], 0.01 - 1e-9)

    def test_late_crx_waits_for_shared_settle_and_finger_error_is_nonblocking(self):
        stream, events, frames = simulate(late=True, hand_error=True)
        self.assertTrue(stream.done)
        completed = [e for e in events if e["kind"] == "stage_completed"]
        self.assertGreaterEqual(completed[0]["monotonic_time"], frames[0]["stage_end_time"] + 0.7 + 0.3)
        self.assertGreater(max(abs(q) for q in completed[-1]["target_error_rad"]["right"][6:]), 0.02)
        next_phase = next(f for f in frames if f["index"] == 1)
        self.assertGreater(next_phase["monotonic_time"], completed[0]["monotonic_time"])

    def test_faults_on_either_side_latch_combined_silence(self):
        for side in SIDES:
            for fault in ("stale", "limit", "routing", "fault", "ownership", "rate"):
                measured = {s: [0.0] * 28 for s in SIDES}
                stream = SharedStageStream(packet(), limits())
                stream.start(feedback(0, measured), 0)
                sample = feedback(0.01, measured)
                if fault == "stale":
                    sample[side]["source_timestamp"] = -1
                elif fault == "limit":
                    sample[side]["positions_rad"][0] = 2
                elif fault == "routing":
                    sample[side]["joint_names"][0] = "wrong_J1"
                elif fault == "fault":
                    sample[side]["fault"] = "driver fault"
                with self.subTest(side=side, fault=fault), self.assertRaises(ValueError):
                    stream.tick(0.1 if fault == "rate" else 0.01, sample, transport_ok=fault != "ownership")
                self.assertIsNotNone(stream.fault)
                self.assertEqual(stream.index, 0)
                self.assertTrue(all(not g.running for g in stream.guards.values()))
                with self.assertRaises(ValueError):
                    stream.tick(0.02, feedback(0.02, measured))

    def test_passive_drift_is_not_ignored(self):
        measured = {s: [0.0] * 28 for s in SIDES}
        stream = SharedStageStream(packet(("right",)), limits())
        stream.start(feedback(0, measured), 0)
        measured["left"][0] = 0.011
        with self.assertRaisesRegex(ValueError, "moved"):
            stream.tick(0.01, feedback(0.01, measured))

    def test_one_side_unsafe_command_prevents_returning_any_paired_frame(self):
        for value in (0.5, 0.001):
            measured = {s: [0.0] * 28 for s in SIDES}
            stream = SharedStageStream(packet(), limits())
            stream.start(feedback(0, measured), 0)
            with patch.object(stream.guards["right"], "tick", return_value=(value,) * 28):
                with self.assertRaisesRegex(ValueError, "speed/acceleration"):
                    stream.tick(0.01, feedback(0.01, measured))
            self.assertFalse(stream.guards["left"].running)

    def test_arrival_timeout_aborts_whole_stage(self):
        stream = SharedStageStream(packet(), limits())
        measured = {s: [0.0] * 28 for s in SIDES}
        stream.start(feedback(0, measured), 0)
        with self.assertRaisesRegex(ValueError, "arrival timeout"):
            for tick in range(1, 1500):
                now = tick / 100
                stream.tick(now, feedback(now, measured))
        self.assertEqual(stream.index, 0)


if __name__ == "__main__":
    unittest.main()
