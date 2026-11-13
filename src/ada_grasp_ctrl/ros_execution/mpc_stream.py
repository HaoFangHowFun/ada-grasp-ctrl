"""Mock-only streaming ownership using the existing ROS runtime safety preflight."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES
from ada_grasp_ctrl.mpc.contracts import JointFeedback, MPCProposal, fresh
from ada_grasp_ctrl.mpc.streaming import GuardedStream
from .settings import bridge_config, validate_domain
from ada_grasp_ctrl.mpc.contracts import named_vector


def publish_pair(rt: Any, guard: GuardedStream, command: tuple[float, ...], log: Any) -> None:
    """Prepare one canonical pair, commit only after both local publishes return.

    ROS has no atomic two-topic send or physical-delivery acknowledgment. A
    partial send is fatal: preserve the previous complete pair, log the sent
    half, and let the existing receiver/transport abort without retries.
    """
    sent = []
    try:
        command = named_vector(JOINT_NAMES, command)
        if (
            not guard.running
            or guard.ownership.owner != guard.owner
            or guard.pending is None
            or command != guard.pending[1]
        ):
            raise ValueError("No matching pending command for publication")
        stamp = rt.node.get_clock().now().to_msg()
        messages = {}
        for group, sl in (("arm", slice(0, 6)), ("hand", slice(6, 28))):
            message = rt.JointState(name=list(JOINT_NAMES[sl]), position=list(command[sl]))
            message.header.stamp = stamp
            messages[group] = message
        for group in ("arm", "hand"):
            rt.pubs[group].publish(messages[group])
            sent.append(group)
        guard.published(command, time.monotonic())
    except Exception as error:
        guard.stop("Paired publication failed: " + str(error))
        log.event(
            "mpc_publish_failed",
            reason=guard.fault,
            published_groups=sent,
            q_attempted=command,
            q_last_published=guard.q_last_published,
        )
        raise RuntimeError(guard.fault) from error


class ROSMockStream:
    """A separate single-writer interface; never feed per-tick actions to Bridge.

    Construction rejects hardware before creating a ROS node. Opening reuses
    backend/plugin/model/controller/force-limiter checks. Every publication runs
    the existing watchdog and the MPC freshness and publication checks.
    """

    def __init__(self, config: dict | None, log: Any, *, domain: int) -> None:
        self.config = bridge_config(config)
        if (
            self.config["environment"] != "mock"
            or self.config.get("side", "right") != "right"
            or self.config.get("active_arms", ["right"]) != ["right"]
        ):
            raise ValueError("MPC ROS streaming is right-only mock; hardware cannot be enabled")
        validate_domain("mock", domain)
        import os

        if str(domain) != os.environ.get("ROS_DOMAIN_ID"):
            raise ValueError("ROS domain differs from the explicitly requested domain")
        self.log, self.runtime, self.guard = log, None, None
        self.published = 0
        self.completed = False

    def open(self) -> JointFeedback:
        """Complete read-only preflights before acquiring command-topic ownership."""
        from .runtime import Runtime, ARM_COMMAND
        from dual_sharpa_wave.qos import hand_qos

        self.runtime = rt = Runtime(self.config, self.log)
        try:
            rt.preflight(True)
            if any("bridge" in name for name, _ in rt.node.get_node_names_and_namespaces()):
                raise ValueError("Legacy action receiver must be stopped before MPC ownership")
            self.guard = GuardedStream(
                config={k: v for k, v in self.config.items() if k not in ("mode", "path_verified")},
            )
            rt.pubs = {
                "arm": rt.node.create_publisher(rt.JointState, ARM_COMMAND, 1),
                "hand": rt.node.create_publisher(rt.JointState, rt.hand_command, hand_qos()),
            }
            self.log.result["command_publishers_created"] = True
            rt.wait(lambda: all(p.get_subscription_count() >= 1 for p in rt.pubs.values()), "MPC command discovery")
            feedback = self.feedback()
            self.guard.start(feedback, time.monotonic())
            self.log.event("mpc_owner_acquired", owner="mpc_mock", publishers=self.inventory())
            return feedback
        except Exception:
            self.close("startup failure")
            raise

    def inventory(self) -> dict:
        from .runtime import ARM_COMMAND, HAND_COMMAND

        return {
            topic: [
                {"name": i.node_name, "namespace": i.node_namespace}
                for i in self.runtime.node.get_publishers_info_by_topic(topic)
            ]
            for topic in (
                ARM_COMMAND,
                HAND_COMMAND,
                "/sharpa/left_hand/joint_command",
                "/crx5ia/left/forward_position_controller/commands",
            )
        }

    def feedback(self) -> JointFeedback:
        """Use receipt times retained by Runtime's duplicate/source timestamp guard."""
        rt = self.runtime
        if not rt.fresh():
            raise ValueError("Stale or invalid ROS feedback")
        source = min(rt.feedback[g].received for g in ("arm", "hand"))
        received = max(rt.feedback[g].received for g in ("arm", "hand"))
        if received - source > 0.15:
            raise ValueError("Arm/hand feedback skew")
        return JointFeedback(
            tuple(JOINT_NAMES),
            tuple(rt.feedback["arm"].q) + tuple(rt.feedback["hand"].q),
            source,
            received,
            "ROS_header_guarded_host_monotonic_receipt",
        )

    def tick(self, proposal: MPCProposal | None = None) -> tuple[float, ...] | None:
        """Publish at most one command, failing closed on DDS/backlog/deadline faults."""
        rt = self.runtime
        try:
            rt.spin(0.0)
            now = time.monotonic()
            rt.watchdog()
            if any("bridge" in name for name, _ in rt.node.get_node_names_and_namespaces()):
                raise ValueError("Legacy receiver appeared while MPC owns command topics")
            if proposal is not None:
                self.log.event("mpc_proposal", **asdict(proposal))
                if proposal.diagnostics.get("termination") == "native_complete":
                    fresh(proposal.computed_at, proposal.computed_at, now, 0.25)
                    if proposal.solve_duration_s > 0.1:
                        raise ValueError("MPC completion deadline missed")
                    self.guard.check(now, self.feedback())
                    self.completed = True
                    self.close("native_complete")
                    return None
                self.guard.accept(proposal, now)
            feedback = self.feedback()
            command = self.guard.tick(now, feedback)
            if command is not None:
                publish_pair(rt, self.guard, command, self.log)
                self.published += 1
                self.log.event(
                    "mpc_publish",
                    owner="mpc_mock",
                    q_100Hz=command,
                    measured_q=feedback.positions_rad,
                    feedback_age_s=now - feedback.source_timestamp,
                    q_mpc=self.guard.target.positions_rad,
                    q_segment_start=self.guard.q_start,
                    segment_started=self.guard.segment_started,
                    proposal_computed_at=self.guard.target.computed_at,
                    q_last_published=self.guard.q_last_published,
                    sampled_at=now,
                    published_at=self.guard.last_published_at,
                    dropped_ticks=self.guard.dropped_ticks,
                )
            return command
        except Exception as error:
            self.close(str(error))
            raise

    def close(self, reason: str = "cancel/shutdown") -> None:
        """Stop returning commands, invoke existing mock stop handling and destroy publishers."""
        if self.guard is not None and self.guard.running:
            self.guard.stop(reason)
        if self.runtime is not None:
            rt, self.runtime = self.runtime, None
            self.log.event("mpc_shutdown", reason=reason, command_count=self.published)
            if rt.pubs:
                rt.abort()
            rt.node.destroy_node()
