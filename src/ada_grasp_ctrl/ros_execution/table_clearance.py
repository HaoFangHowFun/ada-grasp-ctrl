"""Execute one exact tabletop evaluation plan after the owning preview confirms it.

This dedicated entry point does not change the grasp bridge's packet policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, check_kinematics, kinematic_contract
from ada_grasp_ctrl.pipeline.plan import check_start
from ada_grasp_ctrl.ros_execution.motion import smoothstep
from ada_grasp_ctrl.ros_execution.settings import bridge_config, validate_domain


def check_command_step(previous: np.ndarray, desired: np.ndarray, elapsed: float, config: dict) -> None:
    """Abort a delayed command instead of jumping or changing the reviewed path."""
    for group, sl in (("arm", slice(0, 6)), ("hand", slice(6, 28))):
        bound = min(config[group + "_step"], config[group + "_speed"] * elapsed)
        if np.max(np.abs(desired[sl] - previous[sl])) > bound + 1e-8:
            raise RuntimeError("Command step exceeds maintained limit: " + group)


def validate(plan: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reject foreign plans, moving fingers, invalid timing and changed policy."""
    if plan.get("schema_version") != 1 or plan.get("operation") != "table_clearance":
        raise ValueError("Expected a tabletop clearance plan")
    if plan.get("joint_names") != list(JOINT_NAMES) or plan.get("hand_policy") != "retain_measured_finger_angles":
        raise ValueError("Unexpected joints or hand policy")
    if plan.get("settings") != bridge_config({"environment": plan.get("environment")}):
        raise ValueError("Receiver settings differ from maintained defaults")
    if not plan.get("file_sha256") or not 0.02 <= plan["clearance_m"] <= 0.30:
        raise ValueError("Missing provenance or invalid clearance")
    frames = np.asarray(plan["keyframes"], dtype=float)
    durations = np.asarray(plan["durations_s"], dtype=float)
    if frames.ndim != 2 or frames.shape[1] != 28 or not 2 <= len(frames) <= 32:
        raise ValueError("Expected bounded 28-axis waypoints")
    if not np.isfinite(frames).all() or durations.shape != (len(frames) - 1,):
        raise ValueError("Invalid waypoints or durations")
    if not np.isfinite(durations).all() or np.any(durations < 0.5) or durations.sum() > 600:
        raise ValueError("Invalid motion duration")
    if not np.all(frames[:, 6:] == frames[0, 6:]):
        raise ValueError("Evaluation must preserve measured finger targets")
    deltas = np.max(np.abs(np.diff(frames[:, :6], axis=0)), axis=1)
    if np.any(deltas > 0.15 + 1e-10):
        raise ValueError("Waypoint branch jump")
    if np.any(1.875 * deltas / durations > 0.05 + 1e-8):
        raise ValueError("Plan exceeds evaluator speed")
    if np.any(5.774 * deltas / durations**2 > 0.1 + 1e-8):
        raise ValueError("Plan exceeds evaluator acceleration")
    return frames, durations


def main(argv: list[str] | None = None) -> int:
    """Recheck exact plan, fresh state and all live guards before publishing."""
    import rclpy
    from dual_sharpa_wave.qos import hand_qos
    from ada_grasp_ctrl.ros_execution.model import ARM_NAMES, HAND_NAMES, Model
    from ada_grasp_ctrl.ros_execution.runtime import ARM_COMMAND, HAND_COMMAND, RunLog, Runtime, save
    from ada_grasp_ctrl.ros_execution.bridge import minimum_duration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.plan.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError("Plan changed after preview")
    plan = json.loads(raw)
    frames, durations = validate(plan)
    domain = validate_domain(plan["environment"], os.getenv("ROS_DOMAIN_ID"))
    if domain != plan["ros_domain_id"]:
        raise ValueError("ROS domain differs from reviewed plan")
    for path, expected in plan["file_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Reviewed input changed: " + path)
    args.output_root.mkdir(parents=True, exist_ok=False)
    config = dict(plan["settings"])
    # Only this exact, terminal-confirmed evaluation receives the existing runtime
    # review marker. No path_verified or calibration claims are promoted.
    config["operator_reviewed_joint_goal"] = True
    log_args = argparse.Namespace(**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    log = RunLog(str(args.output_root / "diagnostics"), config, log_args)
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    rt = Runtime(config, log)
    result = {"status": "failed", "motion_commands": 0, "physical_clearance_independently_measured": False}
    try:
        rt.preflight(True)
        check_kinematics(plan["kinematics"], kinematic_contract(rt.description))
        model = Model(rt.description, config)
        for q in frames:
            model.check("arm", q[:6])
            model.check("hand", q[6:])
        for a, b, duration in zip(frames, frames[1:], durations):
            if duration + 1e-6 < minimum_duration(a, b, config, model):
                raise ValueError("Plan exceeds model velocity limits")
        if not rt.fresh():
            raise RuntimeError("Feedback expired during validation")
        actual = np.r_[rt.feedback["arm"].q, rt.feedback["hand"].q]
        check_start(actual.tolist(), frames[0].tolist())
        # Tighter than the grasp start check to avoid a jump to the first waypoint.
        if np.max(np.abs(actual - frames[0])) > 0.002:
            raise ValueError("Robot moved since preview by more than 0.002 rad; prepare again")
        rt.pubs = {
            "arm": rt.node.create_publisher(rt.JointState, ARM_COMMAND, 1),
            "hand": rt.node.create_publisher(rt.JointState, HAND_COMMAND, hand_qos()),
        }
        log.result["command_publishers_created"] = True
        rt.wait(lambda: all(p.get_subscription_count() >= 1 for p in rt.pubs.values()), "command discovery")
        actual = np.r_[rt.feedback["arm"].q, rt.feedback["hand"].q]
        if not rt.fresh() or np.max(np.abs(actual - frames[0])) > 0.002:
            raise ValueError("Robot moved or feedback expired before first command")
        previous_command = actual.copy()
        began = last = time.monotonic()
        total = float(durations.sum())
        settled = None
        while True:
            rt.spin(0.002)
            now = time.monotonic()
            if not rt.fresh() or now - last > config["feedback_timeout"]:
                raise RuntimeError("Stale feedback or stalled command loop")
            if now - last < 1 / config["rate_hz"]:
                continue
            rt.watchdog()
            elapsed = now - began
            progress = elapsed
            for i, duration in enumerate(durations):
                if progress <= duration or i == len(durations) - 1:
                    u = float(np.clip(progress / duration, 0, 1))
                    desired = frames[i] + smoothstep(u) * (frames[i + 1] - frames[i])
                    break
                progress -= duration
            actual = np.r_[rt.feedback["arm"].q, rt.feedback["hand"].q]
            model.check("arm", actual[:6])
            model.check("hand", actual[6:])
            if np.max(np.abs(actual[:6] - desired[:6])) > 0.03:
                raise RuntimeError("Arm tracking error exceeds 0.03 rad")
            if np.max(np.abs(actual[6:] - frames[0, 6:])) > config["hand_tolerance"]:
                raise RuntimeError("Hand posture changed during clearance evaluation")
            check_command_step(previous_command, desired, now - last, config)
            for group, names, sl in (("arm", ARM_NAMES, slice(0, 6)), ("hand", HAND_NAMES, slice(6, 28))):
                msg = rt.JointState(name=names, position=desired[sl].tolist())
                msg.header.stamp = rt.node.get_clock().now().to_msg()
                rt.pubs[group].publish(msg)
            result["motion_commands"] += 1
            previous_command = desired.copy()
            log.event("trajectory", command=desired.tolist(), measured=actual.tolist())
            last = now
            reached = np.max(np.abs(actual[:6] - frames[-1, :6])) <= config["arm_tolerance"]
            if elapsed >= total and reached:
                settled = now if settled is None else settled
                if now - settled >= config["settle_time"] and all(s.received > settled for s in rt.feedback.values()):
                    break
            else:
                settled = None
            if elapsed > total + config["settle_timeout"]:
                raise TimeoutError("Clearance target arrival timeout")
        result.update(status="completed", final_positions=actual.tolist(), automatic_return=False)
        log.result.update(final_state="DONE", task_completed=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        rt.abort()
        result["reason"] = str(error) or "Interrupted"
        log.result.update(final_state="FAULT", abort_reason=result["reason"])
        print(result["reason"], flush=True)
        return 1
    finally:
        result["stop"] = log.result.get("stop")
        save(args.output_root / "result.json", result)
        rt.node.destroy_node()
        rclpy.try_shutdown()
        log.finish()


if __name__ == "__main__":
    raise SystemExit(main())
