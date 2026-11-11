"""Audit the terminal gate and one mock execution through the complete Python entry point."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def main() -> int:
    """Start an owned pipeline, verify zero commands before Enter, then execute one group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cancel-during-hold", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    checkout = Path(__file__).resolve().parents[1]
    pipeline_root = root / "pipeline"
    output = (root / "pipeline.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            str(checkout / "script/run_grasp_pipeline.py"),
            "--mode",
            "replay",
            "--group",
            "01" if args.cancel_during_hold else "all",
            "--start-mock",
            "--reuse-candidates",
            "--output-root",
            str(pipeline_root),
        ],
        stdin=subprocess.PIPE,
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
    )
    node, ui = None, None
    report = {"status": "running", "hardware": False}
    counts = {"arm": 0, "hand": 0, "left_arm": 0, "left_hand": 0}
    wrong_names = []
    try:
        deadline = time.monotonic() + 60
        while not (pipeline_root / "initial_state/robot_state.json").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Pipeline failed before initial state; inspect pipeline.log")
            time.sleep(0.1)
        os.environ["ROS_DOMAIN_ID"] = "88"
        rclpy.init(args=[])
        node = rclpy.create_node("ada_pipeline_audit", enable_rosout=False, start_parameter_services=False)

        def receive(key: str, message: object) -> None:
            counts[key] += 1
            if key in ("arm", "hand") and any(not n.startswith("right_") for n in message.name):
                wrong_names.append(list(message.name))

        for key, cls, topic in (
            ("arm", JointState, "/crx5ia/joint_targets"),
            ("hand", JointState, "/sharpa/right_hand/joint_command"),
            ("left_arm", Float64MultiArray, "/crx5ia/left/forward_position_controller/commands"),
            ("left_hand", JointState, "/sharpa/left_hand/joint_command"),
        ):
            node.create_subscription(cls, topic, lambda m, k=key: receive(k, m), qos_profile_sensor_data)

        def wait(predicate, timeout: float) -> None:
            deadline = time.monotonic() + timeout
            while not predicate():
                rclpy.spin_once(node, timeout_sec=0.01)
                if process.poll() is not None:
                    raise RuntimeError("Pipeline exited early; inspect pipeline.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("Pipeline audit timed out")

        wait(lambda: (pipeline_root / "viewer_state.json").exists(), 300)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(checkout / "src")
        ui = subprocess.Popen(
            [
                str(checkout / ".conda/bin/python"),
                str(checkout / "script/check_grasp_pipeline_ui.py"),
                "--status",
                str(pipeline_root / "viewer_state.json"),
                "--output",
                str(root / "ui.json"),
            ],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        wait(lambda: ui.poll() is not None, 60)
        assert ui.returncode == 0, "UI audit failed"
        assert not any(counts.values()), counts
        assert node.count_publishers("/crx5ia/joint_targets") == 0
        assert node.count_publishers("/sharpa/right_hand/joint_command") == 0
        report["before_enter_counts"] = counts.copy()
        process.stdin.write("\n")
        process.stdin.flush()
        deadline = time.monotonic() + 180
        canceled = False
        while process.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.01)
            event_path = pipeline_root / "sender/events.jsonl"
            if args.cancel_during_hold and not canceled and event_path.exists():
                # Only parse newline-terminated records while the sender is writing.
                lines = event_path.read_text().split("\n")[:-1]
                if any(
                    json.loads(line).get("phase") == "hold" and json.loads(line)["kind"] == "action_feedback"
                    for line in lines
                ):
                    process.send_signal(signal.SIGINT)
                    canceled = True
            if time.monotonic() > deadline:
                raise TimeoutError("Mock execution timed out")
        result = json.loads((pipeline_root / "run_report.json").read_text())
        if args.cancel_during_hold:
            execution = json.loads((pipeline_root / "sender/run_report.json").read_text())
            events = [json.loads(line) for line in (pipeline_root / "sender/events.jsonl").read_text().splitlines()]
            sent = [e["phase"] for e in events if e["kind"] == "goal_sent"]
            assert canceled and process.returncode == 130 and result["status"] == "interrupted", result
            assert execution["status"] == "aborted" and execution["cancel_acknowledged"], execution
            assert sent == ["pregrasp", "grasp", "squeeze", "lift", "hold"], sent
            assert not counts["left_arm"] and not counts["left_hand"] and not wrong_names
            report.update(
                status="passed",
                canceled_during="hold",
                phases_sent=sent,
                no_release_or_return=True,
                cancel_acknowledged=True,
                final_counts=counts,
            )
            print(json.dumps(report, indent=2))
            return 0
        assert process.returncode == (1 if result.get("preparation_errors") else 0), result
        assert result["status"] == "completed", result
        assert result["approved_group"] == "01", result
        assert [phase["phase"] for phase in result["execution"]["phases"]] == [
            "pregrasp",
            "grasp",
            "squeeze",
            "lift",
            "hold",
            "release",
            "return",
        ], result
        assert result["execution"]["lift_motion_completed"] and result["execution"]["object_lifted"] is None
        assert all(result["execution"][p + "_motion_completed"] for p in ("hold", "release", "return"))
        assert not counts["left_arm"] and not counts["left_hand"] and not wrong_names
        assert counts["arm"] > 0 and counts["hand"] > 0
        events = [json.loads(line) for line in (pipeline_root / "sender/events.jsonl").read_text().splitlines()]
        goals = [e for e in events if e["kind"] == "goal_sent"]
        plan = json.loads((pipeline_root / "group01/export/motion_plan.json").read_text())
        assert [e["positions"] for e in goals] == plan["keyframes"][1:]
        assert [e["duration"] for e in goals] == plan["durations_s"]
        hold = next(e for e in goals if e["phase"] == "hold")
        lift_done = next(e for e in events if e["kind"] == "result" and e["phase"] == "lift")
        release = next(e for e in goals if e["phase"] == "release")
        assert hold["time"] >= lift_done["time"]
        assert release["time"] - hold["time"] >= 5.0
        assert plan["keyframes"][-1] == plan["keyframes"][0][:6] + [0.0] * 22
        report.update(
            status="passed",
            final_counts=counts,
            approved_group="01",
            exact_preview_targets_and_times=True,
            phases=plan["phases"],
            lift=plan["lift"],
            preparation_errors=result.get("preparation_errors", []),
            hold_before_release_s=release["time"] - hold["time"],
            returned_to_start_with_open_hand=True,
        )
    except BaseException as error:
        report.update(status="failed", reason=str(error))
        raise
    finally:
        if ui is not None and ui.poll() is None:
            ui.terminate()
            ui.wait(timeout=5)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=45)
        output.close()
        if node is not None:
            node.destroy_node()
            rclpy.try_shutdown()
        (root / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
