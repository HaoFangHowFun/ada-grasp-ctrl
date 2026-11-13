"""Passive right-only FollowJointTrajectory receiver, with no grasp-stage policy.

One goal contains one named 28-axis position target and a minimum motion time.
The upstream application owns sequencing; this server never invents a next goal.
"""

from __future__ import annotations

import argparse
from typing import Any
from pathlib import Path
import signal
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import yaml

from ada_grasp_ctrl.ros_execution.model import ARM_NAMES, HAND_NAMES, Model, named_positions
from ada_grasp_ctrl.ros_execution.runtime import ARM_COMMAND, RunLog, Runtime

from ada_grasp_ctrl.experiments.right_ros_contract import action_name, side_names, packet_side
from ada_grasp_ctrl.paths import resolve_external_root
from ada_grasp_ctrl.ros_execution.settings import bridge_config
from ada_grasp_ctrl.ros_execution.motion import smoothstep, arrival_reached
from ada_grasp_ctrl.ros_execution.review import check_reviewed_goal
from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.experiments.right_ros_contract import check_kinematics, kinematic_contract
from ada_grasp_ctrl.pipeline.plan import check_start
from ada_grasp_ctrl.mpc.reference import linear_reference

NAMES = ARM_NAMES + HAND_NAMES


class GoalCancelled(RuntimeError):
    pass


def unpack_goal(request: Any, side: str = "right") -> tuple[np.ndarray, float, float]:
    """Reject unsupported trajectory semantics instead of silently ignoring them."""
    trajectory = request.trajectory
    if len(trajectory.points) != 1:
        raise ValueError("Exactly one target per goal; the client owns sequencing")
    if trajectory.header.frame_id or trajectory.header.stamp.sec or trajectory.header.stamp.nanosec:
        raise ValueError("Joint goals must use an empty frame and immediate (zero) header stamp")
    if (
        request.multi_dof_trajectory.joint_names
        or request.multi_dof_trajectory.points
        or request.path_tolerance
        or request.component_path_tolerance
        or request.goal_tolerance
        or request.component_goal_tolerance
    ):
        raise ValueError("Only position targets and goal_time_tolerance are supported")
    point = trajectory.points[0]
    if point.velocities or point.accelerations or point.effort:
        raise ValueError("Velocity, acceleration and effort arrays must be empty")
    target = named_positions(trajectory.joint_names, point.positions, list(sum(side_names(side), ())))
    duration = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
    extra = request.goal_time_tolerance.sec + request.goal_time_tolerance.nanosec * 1e-9
    if not np.isfinite([duration, extra]).all() or not 0 < duration <= 300 or not 0 <= extra <= 60:
        raise ValueError("Invalid duration or goal_time_tolerance")
    return target, duration, extra


def minimum_duration(
    start: np.ndarray, target: np.ndarray, config: dict[str, Any], model: Model, *, include_hand: bool = True
) -> float:
    """Duration needed for a zero-end-velocity quintic under configured limits."""
    times = []
    for group, sl in (("arm", slice(0, 6)), ("hand", slice(6, 28))):
        if group == "hand" and not include_hand:
            continue
        delta = float(np.max(np.abs(target[sl] - start[sl])))
        speed = min(config[group + "_speed"], min(model.limits[group][2]))
        times.extend([1.875 * delta / speed, np.sqrt(5.774 * delta / config[group + "_acceleration"])])
    return max(times)


class Bridge(Node):
    def __init__(self, config: dict[str, Any], args: argparse.Namespace) -> None:
        super().__init__(f"ada_{config.get('side', 'right')}_arm_hand_bridge")
        self.side = config.get("side", "right")
        self.names = list(sum(side_names(self.side), ()))
        self.config, self.args = config, args
        self.packet_path = Path(args.targets) if getattr(args, "targets", None) else None
        self.packet = load_packet(self.packet_path) if self.packet_path else None
        if self.packet is not None and packet_side(self.packet) != self.side:
            raise ValueError("Packet side differs from receiver")
        self.mpc_mode = getattr(args, "controller_mode", "linear")
        if self.mpc_mode != "linear" and (self.packet is None or self.side != "right"):
            raise ValueError("MPC requires a reviewed right-hand pipeline packet")
        if (
            self.mpc_mode == "mpc_active"
            and config["environment"] == "hardware"
            and not (args.hardware_enable and args.operator_confirmed)
        ):
            raise ValueError("Hardware MPC needs explicit enable and operator confirmation")
        self.passive_observer = None
        if self.mpc_mode == "mpc_passive":
            from ada_grasp_ctrl.mpc.live_execution import PassiveObserver

            self.passive_observer = PassiveObserver(self.packet, args, Path(args.output_dir) / "mpc_passive")
        self.phase_index = 0
        if config["environment"] == "hardware" and self.packet is None:
            raise ValueError("Hardware receiver requires an explicitly reviewed pipeline packet")
        self.lock = threading.Lock()
        self.busy = False
        self.stopping = threading.Event()
        self.server = ActionServer(
            self,
            FollowJointTrajectory,
            action_name(self.side),
            execute_callback=self.execute_goal,
            goal_callback=self.accept_goal,
            cancel_callback=lambda handle: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info(
            f"Passive receiver ready: {action_name(self.side)}. No target or command publisher until a client goal."
        )

    def accept_goal(self, request: Any) -> GoalResponse:
        try:
            target, duration, extra = unpack_goal(request, self.side)
            if self.packet is not None:
                if load_packet(self.packet_path) != self.packet:
                    raise ValueError("Reviewed packet changed")
                if extra:
                    raise ValueError("Reviewed goals cannot override arrival timeout")
                check_reviewed_goal(self.packet, self.phase_index, target.tolist(), duration, self.config)
        except (ValueError, TypeError, AttributeError) as error:
            self.get_logger().warning(f"Rejecting invalid target: {error}")
            return GoalResponse.REJECT
        with self.lock:
            if self.busy or self.stopping.is_set():
                return GoalResponse.REJECT
            self.busy = True
        return GoalResponse.ACCEPT

    def execute_goal(self, handle: Any) -> FollowJointTrajectory.Result:
        """Run one caller-provided target, returning actual feedback and outcome."""
        log, rt = None, None
        result = FollowJointTrajectory.Result()
        try:
            target, duration, extra = unpack_goal(handle.request, self.side)
            runtime_config = dict(self.config)
            if self.packet is not None:
                check_reviewed_goal(self.packet, self.phase_index, target.tolist(), duration, self.config)
                runtime_config["operator_reviewed_joint_goal"] = True
                plan = self.packet["reviewed_plan"]
                if "passive_positions_rad" in plan:
                    runtime_config["passive_positions_rad"] = plan["passive_positions_rad"]
            log = RunLog(self.args.output_dir, self.config, self.args)
            log.result["operation"] = "external_joint_target"
            log.event("goal_received", joint_names=self.names, target=target.tolist(), duration=duration)
            rt = Runtime(runtime_config, log)
            original_spin = rt.spin

            def spin(seconds: float = 0.01) -> None:
                if handle.is_cancel_requested or self.stopping.is_set():
                    raise GoalCancelled("Client canceled or bridge shutdown requested")
                original_spin(seconds)

            rt.spin = spin
            rt.preflight(True)
            if self.packet is not None:
                check_kinematics(self.packet["kinematics"], kinematic_contract(rt.description, self.side))
                if "scene_kinematics" in self.packet:
                    for side, expected in self.packet["scene_kinematics"].items():
                        check_kinematics(expected, kinematic_contract(rt.description, side))
            model = Model(rt.description, self.config)
            (log.path / "model.urdf").write_text(rt.description)
            for group, sl in (("arm", slice(0, 6)), ("hand", slice(6, 28))):
                model.check(group, target[sl])
            if not rt.fresh():
                raise RuntimeError("Feedback expired during goal validation")
            start = np.r_[rt.feedback["arm"].q, rt.feedback["hand"].q]
            require_hand = self.config.get("hand_arrival_required", True)
            if self.packet is not None:
                check_start(
                    start.tolist(),
                    self.packet["reviewed_plan"]["keyframes"][self.phase_index],
                    hand_tolerance=self.config["hand_tolerance"] if self.phase_index > 0 else 0.02,
                    ignore_hand_error=self.phase_index > 0 and not require_hand,
                )
            required = minimum_duration(start, target, self.config, model, include_hand=require_hand)
            if duration + 1e-6 < required:
                raise ValueError(f"Target duration {duration:g}s is too short; need at least {required:g}s")
            phase = self.packet["targets"][self.phase_index]["phase"] if self.packet else None
            if self.mpc_mode == "mpc_active" and phase == "grasp":
                from ada_grasp_ctrl.mpc.live_execution import run_active

                outcome = run_active(rt, model, self.packet, self.args, log, handle)
                log.result.update(final_state="DONE", task_completed=True, **outcome)
                result.error_code = result.SUCCESSFUL
                result.error_string = "Ours native_complete: grasp+squeeze episode; no automatic next motion"
                self.phase_index += 2
                handle.succeed()
                return result
            passive_feed = None
            passive_error = None
            if self.passive_observer is not None and phase in ("grasp", "squeeze"):
                from ada_grasp_ctrl.mpc.live_execution import LiveFeed

                try:
                    passive_feed = LiveFeed(rt)
                except Exception as error:
                    log.event("mpc_passive_unavailable", reason=str(error))
            from dual_sharpa_wave.qos import hand_qos

            rt.pubs = {
                "arm": rt.node.create_publisher(rt.JointState, ARM_COMMAND, 1),
                "hand": rt.node.create_publisher(rt.JointState, rt.hand_command, hand_qos()),
            }
            log.result["command_publishers_created"] = True
            rt.wait(lambda: all(p.get_subscription_count() >= 1 for p in rt.pubs.values()), "command discovery")
            if not rt.fresh():
                raise RuntimeError("Feedback expired before first command")
            actual = np.r_[rt.feedback["arm"].q, rt.feedback["hand"].q]
            drift = actual - start if require_hand or self.phase_index == 0 else actual[:6] - start[:6]
            if np.max(np.abs(drift)) > min(self.config["arm_tolerance"], self.config["hand_tolerance"]):
                raise RuntimeError("Robot moved during goal validation; resend using current feedback")
            start = actual.copy()
            required = minimum_duration(start, target, self.config, model, include_hand=require_hand)
            if duration + 1e-6 < required:
                raise ValueError("Fresh start requires a longer target duration")
            if not require_hand:
                # Preserve joint speed/acceleration limits when contact left fingers
                # away from the previous target. The reviewed target is unchanged.
                reviewed_duration = duration
                duration = max(duration, minimum_duration(start, target, self.config, model))
                log.event(
                    "arrival_policy",
                    hand_arrival_required=False,
                    reviewed_duration_s=reviewed_duration,
                    execution_duration_s=duration,
                )
            began = last_tick = time.monotonic()
            settled = None
            command = start.copy()
            next_tick = began
            next_feedback = began
            linear_guard = None
            if self.config.get("interpolation") == "linear":
                from ada_grasp_ctrl.mpc.contracts import JointFeedback, MPCProposal
                from ada_grasp_ctrl.mpc.streaming import GuardedStream

                linear_guard = GuardedStream(
                    tuple(model.limits["arm"][0]) + tuple(model.limits["hand"][0]),
                    tuple(model.limits["arm"][1]) + tuple(model.limits["hand"][1]),
                    tuple(model.limits["arm"][2]) + tuple(model.limits["hand"][2]),
                    config={k: v for k, v in self.config.items() if k not in ("mode", "path_verified")},
                )
                linear_guard.start(
                    JointFeedback(tuple(self.names), tuple(start), began, began, "mock_reviewed_start"),
                    began,
                    owner="baseline",
                )
            log.event("transition", state="EXECUTING")
            while True:
                rt.spin(0.002)
                now = time.monotonic()
                if now < next_tick:
                    continue
                if not rt.fresh() or now - last_tick > self.config["feedback_timeout"]:
                    raise RuntimeError("Stale feedback or stalled command loop")
                rt.watchdog()
                actual = np.r_[rt.feedback["arm"].q, rt.feedback["hand"].q]
                for group, sl in (("arm", slice(0, 6)), ("hand", slice(6, 28))):
                    model.check(group, actual[sl])
                elapsed = now - began
                u = min(1.0, elapsed / duration)
                if linear_guard is not None:
                    desired = np.asarray(linear_reference(start, target, elapsed, duration))
                    source_time = min(rt.feedback[g].received for g in ("arm", "hand"))
                    received_time = max(rt.feedback[g].received for g in ("arm", "hand"))
                    linear_guard.accept(MPCProposal(tuple(self.names), tuple(desired), True, now, 0, "linear"), now)
                    bounded = linear_guard.tick(
                        now, JointFeedback(tuple(self.names), tuple(actual), source_time, received_time, "mock_ROS")
                    )
                    if bounded is None:
                        continue
                    command = np.asarray(bounded)
                    log.event("linear_reference", q_ref=desired.tolist(), q_100Hz=command.tolist())
                else:
                    desired = start + smoothstep(u) * (target - start)
                for group, sl in (("arm", slice(0, 6)), ("hand", slice(6, 28))):
                    if linear_guard is None:
                        step = min(self.config[group + "_step"], self.config[group + "_speed"] * (now - last_tick))
                        command[sl] += np.clip(desired[sl] - command[sl], -step, step)
                    msg = rt.JointState(name=self.names[sl], position=command[sl].tolist())
                    if group == "arm" and "passive_positions_rad" in rt.c:
                        # Replace any retained passive-arm trajectory with its reviewed stationary target.
                        msg.name.extend(side_names(rt.passive_side)[0])
                        msg.position.extend(rt.c["passive_positions_rad"][:6])
                    msg.header.stamp = rt.node.get_clock().now().to_msg()
                    rt.pubs[group].publish(msg)
                if passive_feed is not None:
                    try:
                        measured, tactile = passive_feed.sample()
                        self.passive_observer.observe(now, measured, command, tactile, phase)
                    except Exception as error:
                        # Report once per changed fault; passive failures never abort Linear.
                        if str(error) != passive_error:
                            log.event("mpc_passive_unavailable", reason=str(error))
                            passive_error = str(error)
                last_tick = now
                next_tick = now + 1 / self.config["rate_hz"]
                error = target - actual
                log.event("trajectory", command=command.tolist(), measured=actual.tolist(), target_error=error.tolist())
                if now >= next_feedback:
                    feedback = FollowJointTrajectory.Feedback(
                        joint_names=self.names,
                        desired=JointTrajectoryPoint(positions=command.tolist()),
                        actual=JointTrajectoryPoint(positions=actual.tolist()),
                        error=JointTrajectoryPoint(positions=(command - actual).tolist()),
                    )
                    handle.publish_feedback(feedback)
                    next_feedback = now + 0.1
                reached = arrival_reached(error.tolist(), self.config)
                if elapsed >= duration and np.max(np.abs(command - target)) < 1e-8 and reached:
                    settled = now if settled is None else settled
                    if now - settled >= self.config["settle_time"] and all(
                        v.received > settled for v in rt.feedback.values()
                    ):
                        break
                else:
                    settled = None
                if elapsed > duration + (extra or self.config["settle_timeout"]):
                    raise TimeoutError("Target arrival timeout")
            result.error_code = result.SUCCESSFUL
            result.error_string = (
                "Target reached; no automatic next motion"
                if require_hand
                else "Arm reached; finger errors recorded without gating arrival"
            )
            log.result.update(
                final_state="DONE",
                task_completed=True,
                final_positions=actual.tolist(),
                final_target_error_rad=error.tolist(),
                hand_arrival_required=require_hand,
                hand_within_tolerance=bool(np.max(np.abs(error[6:])) <= self.config["hand_tolerance"]),
            )
            self.phase_index += 1
            handle.succeed()
        except Exception as error:
            if self.packet is not None:
                self.stopping.set()
            if rt:
                rt.abort()
            if log:
                log.result.update(
                    final_state="ABORTED" if isinstance(error, GoalCancelled) else "FAULT", abort_reason=str(error)
                )
            result.error_code = result.INVALID_GOAL if isinstance(error, ValueError) else result.GOAL_TOLERANCE_VIOLATED
            result.error_string = str(error)
            if handle.is_cancel_requested:
                handle.canceled()
            else:
                handle.abort()
            self.get_logger().error(str(error))
        finally:
            if rt:
                rt.node.destroy_node()
            if log:
                log.finish()
            with self.lock:
                self.busy = False
        return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional YAML overrides for Ada receiver defaults")
    parser.add_argument("--output-dir", help="Log root; defaults to the Ada output root / ros_bridge/receiver")
    parser.add_argument("--targets", type=Path, help="Exact packet approved by the pipeline's terminal Enter")
    parser.add_argument("--controller-mode", choices=("linear", "mpc_passive", "mpc_active"), default="linear")
    parser.add_argument("--hardware-enable", action="store_true")
    parser.add_argument("--operator-confirmed", action="store_true")
    parser.add_argument("--mpc-python", default=str(resolve_external_root("data") / ".conda/bin/python"))
    parser.add_argument(
        "--mpc-model-report",
        default=str(
            resolve_external_root("data")
            / "output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json"
        ),
    )
    parser.add_argument(
        "--mpc-package-share", default="/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave"
    )
    args = parser.parse_args(argv)
    overrides = yaml.safe_load(resolve_external_root("data", args.config).read_text()) if args.config else None
    try:
        config = bridge_config(overrides)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    args.output_dir = str(
        resolve_external_root("output", args.output_dir)
        if args.output_dir
        else resolve_external_root("output") / "ros_bridge/receiver"
    )
    args.config = str(args.config) if args.config else None
    if args.targets:
        args.targets = str(resolve_external_root("data", args.targets))
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = Bridge(config, args)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, lambda signum, frame: node.stopping.set())
    try:
        while not node.stopping.is_set():
            executor.spin_once(timeout_sec=0.1)
        deadline = time.monotonic() + 5
        while node.busy and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
    finally:
        executor.shutdown(timeout_sec=5)
        if node.passive_observer is not None:
            node.passive_observer.close()
        node.server.destroy()
        node.destroy_node()
        rclpy.try_shutdown()
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
