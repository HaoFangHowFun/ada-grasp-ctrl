"""Existing action receiver's MPC branch; same publishers, watchdog and abort path."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import queue
import threading
import time

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES
from ada_grasp_ctrl.tactile_preview.data import TOPICS, normalize_frame
from .contracts import JointFeedback, fresh
from .ipc import MPCWorker
from .streaming import GuardedStream


class LiveFeed:
    """Subscribe to existing normalized ROS frames, never poll the SDK."""

    def __init__(self, rt) -> None:
        from dual_sharpa_wave.msg import TactileFrame
        from rclpy.qos import qos_profile_sensor_data

        self.rt, self.frames, self.errors = rt, {}, {}
        self.subscriptions = [
            rt.node.create_subscription(
                TactileFrame, topic, lambda msg, f=finger: self.receive(f, msg), qos_profile_sensor_data
            )
            for finger, topic in TOPICS.items()
        ]

    def receive(self, finger, message) -> None:
        try:
            self.frames[finger] = normalize_frame(
                message, finger, time.monotonic(), self.rt.node.get_clock().now().nanoseconds * 1e-9
            )
            self.errors.pop(finger, None)
        except (ValueError, TypeError, AttributeError) as error:
            self.frames.pop(finger, None)
            self.errors[finger] = str(error)

    def sample(self) -> tuple[JointFeedback, dict]:
        rt = self.rt
        if not rt.fresh():
            raise ValueError("Stale joint feedback")
        oldest = min(rt.feedback[g].received for g in ("arm", "hand"))
        newest = max(rt.feedback[g].received for g in ("arm", "hand"))
        if newest - oldest > 0.15:
            raise ValueError("Joint receipt skew")
        fb = JointFeedback(
            tuple(JOINT_NAMES),
            tuple(rt.feedback["arm"].q) + tuple(rt.feedback["hand"].q),
            oldest,
            newest,
            "same_host_receipt_aligned",
        )
        tactile = dict(
            frames={f: asdict(frame) for f, frame in self.frames.items()},
            errors=dict(self.errors),
            joints=dict(
                names=list(JOINT_NAMES[6:]),
                positions=list(fb.positions_rad[6:]),
                received=rt.feedback["hand"].received,
                receipt_ros=rt.stamps["hand"],
                subscriber_ros=rt.stamps["hand"],
            ),
            source_issues={
                f: "tactile publisher absent/ambiguous"
                for f, topic in TOPICS.items()
                if rt.node.count_publishers(topic) != 1
            },
        )
        return fb, tactile


def create_worker(packet: dict, args, output: Path) -> MPCWorker:
    return MPCWorker(
        args.mpc_python,
        dict(
            schema_version=1,
            environment=packet["environment"],
            packet=packet,
            model_report=json.loads(Path(args.mpc_model_report).read_text()),
            output=str(output),
            package_share=args.mpc_package_share,
        ),
        output / "native-worker.log",
    )


def run_active(rt, model, packet: dict, args, log, handle) -> dict:
    """Own grasp+squeeze within one receiver action; return only on native completion."""
    from dual_sharpa_wave.qos import hand_qos
    from ada_grasp_ctrl.ros_execution.runtime import ARM_COMMAND
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    if packet["environment"] == "hardware" and not (args.hardware_enable and args.operator_confirmed):
        raise ValueError("Active hardware requires explicit enable and operator confirmation")
    feed = LiveFeed(rt)
    worker = create_worker(packet, args, log.path)
    guard = None
    try:
        rt.wait(lambda: len(feed.frames) == 5 and rt.fresh(), "five tactile streams and fresh joints")
        fb, tactile = feed.sample()
        # Probe before acquiring publishers. It is a real first control step, not a discarded solve.
        now = time.monotonic()
        worker.submit_tactile(now, fb, fb.positions_rad, tactile, "grasp")
        proposal = None
        while proposal is None:
            rt.spin(0.001)
            proposal = worker.poll()
        log.event("mpc_proposal", **asdict(proposal))
        if not proposal.valid:
            raise ValueError("MPC not ready: " + proposal.diagnostics.get("reason", "no accepted first proposal"))
        # Recheck ownership after worker startup; preflight was read-only up to this point.
        if any(rt.node.count_publishers(t) for t in (ARM_COMMAND, rt.hand_command)):
            raise ValueError("Command owner conflict before MPC handoff")
        rt.pubs = {
            "arm": rt.node.create_publisher(rt.JointState, ARM_COMMAND, 1),
            "hand": rt.node.create_publisher(rt.JointState, rt.hand_command, hand_qos()),
        }
        log.result["command_publishers_created"] = True
        rt.wait(lambda: all(p.get_subscription_count() >= 1 for p in rt.pubs.values()), "MPC command discovery")
        config = {k: v for k, v in rt.c.items() if k not in ("mode", "path_verified", "operator_reviewed_joint_goal")}
        guard = GuardedStream(
            tuple(model.limits["arm"][0]) + tuple(model.limits["hand"][0]),
            tuple(model.limits["arm"][1]) + tuple(model.limits["hand"][1]),
            tuple(model.limits["arm"][2]) + tuple(model.limits["hand"][2]),
            config=config,
            hardware_enabled=args.hardware_enable,
        )
        fb, _ = feed.sample()
        now = time.monotonic()
        guard.start(fb, now, owner="mpc_active")
        guard.accept(proposal, now)
        next_solve, next_feedback = proposal.computed_at + 0.1, now
        log.event("mpc_owner_acquired", owner="existing_receiver", joint_names=list(JOINT_NAMES))
        while True:
            rt.spin(0.001)
            now = time.monotonic()
            rt.watchdog()
            fb, tactile = feed.sample()
            received = worker.poll()
            if received is not None:
                log.event("mpc_proposal", **asdict(received))
                if received.diagnostics.get("termination") == "native_complete":
                    # Completion carries no command but retains the same timing checks.
                    fresh(received.computed_at, received.computed_at, now, 0.25)
                    if received.solve_duration_s > 0.1:
                        raise ValueError("MPC completion deadline missed")
                    guard.tick(now, fb)
                    log.event("native_complete", diagnostics=received.diagnostics)
                    return dict(
                        termination="native_complete",
                        final_positions=list(fb.positions_rad),
                        reference_provenance=received.diagnostics.get("reference_provenance"),
                    )
                guard.accept(received, now)
            if now >= next_solve and worker.pending is None:
                worker.submit_tactile(now, fb, guard.q, tactile, "grasp")
                next_solve = now + 0.1
            command = guard.tick(now, fb)
            if command is not None:
                for group, sl in (("arm", slice(0, 6)), ("hand", slice(6, 28))):
                    msg = rt.JointState(name=list(JOINT_NAMES[sl]), position=list(command[sl]))
                    msg.header.stamp = rt.node.get_clock().now().to_msg()
                    rt.pubs[group].publish(msg)
                log.event(
                    "mpc_command",
                    joint_names=list(JOINT_NAMES),
                    q_100Hz=command,
                    measured_q=fb.positions_rad,
                    feedback_age_s=now - fb.source_timestamp,
                    owner="existing_receiver",
                    q_mpc=guard.target.positions_rad,
                )
            if now >= next_feedback:
                handle.publish_feedback(
                    FollowJointTrajectory.Feedback(
                        joint_names=list(JOINT_NAMES),
                        actual=JointTrajectoryPoint(positions=list(fb.positions_rad)),
                        desired=JointTrajectoryPoint(positions=list(guard.q)),
                    )
                )
                next_feedback = now + 0.1
    finally:
        if guard is not None and guard.running:
            guard.stop("MPC episode closed")
        worker.close()


class PassiveObserver:
    """Latest-only mailbox; worker startup, solving and file I/O never block Linear."""

    def __init__(self, packet: dict, args, output: Path) -> None:
        self.mailbox = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, args=(packet, args, output), daemon=True)
        self.thread.start()

    def observe(self, now, feedback, command, tactile, stage) -> None:
        try:
            self.mailbox.put_nowait((now, feedback, tuple(command), tactile, stage))
        except queue.Full:
            pass

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=36)

    def run(self, packet, args, output) -> None:
        worker = None
        output.mkdir(parents=True, exist_ok=True)
        with (output / "passive.jsonl").open("w", buffering=1) as stream:
            try:
                worker = create_worker(packet, args, output)
                last = -float("inf")
                while not self.stopped.is_set():
                    try:
                        request = self.mailbox.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if request[0] - last < 0.1 or time.monotonic() - request[0] > 0.2:
                        continue
                    worker.submit_tactile(*request)
                    last = request[0]
                    result = None
                    while result is None and not self.stopped.is_set():
                        result = worker.poll()
                        time.sleep(0.001)
                    if result is not None:
                        stream.write(json.dumps(asdict(result), allow_nan=False) + "\n")
            except Exception as error:
                stream.write(json.dumps(dict(status="unavailable", reason=str(error), baseline_affected=False)) + "\n")
            finally:
                if worker is not None:
                    worker.close()
