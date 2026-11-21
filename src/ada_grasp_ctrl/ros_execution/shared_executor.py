"""Run an explicit shared packet on verified mock drivers using one publishing loop.

Legacy single-side senders/actions remain unchanged. This entry cannot enable
physical execution, even if invoked from a hardware ROS environment.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import threading
import time

from ada_grasp_ctrl.experiments.right_ros_contract import check_kinematics, kinematic_contract, side_names
from ada_grasp_ctrl.paths import resolve_from_root
from .settings import validate_domain
from .shared_packet import SIDES, load_shared_packet, receiver_config
from .shared_stream import SharedStageStream


def execute_shared(packet: dict, log: object, canceled: threading.Event) -> dict:
    """Reuse existing model/preflight/watchdog/abort ownership for one combined stage stream.

    Args:
        packet: Validated explicit mock-only transport packet.
        log: Existing RunLog instance.
        canceled: Process cancellation flag, checked before every publication.

    Returns:
        Execution report; object grasp/lift success is not inferred from command completion.

    Raises:
        Exception: Any preflight, feedback, routing or combined-stage failure after bounded abort.
    """
    validate_domain("mock", os.environ.get("ROS_DOMAIN_ID"))
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
        raise ValueError("Shared mock execution requires localhost discovery")
    from .runtime import Runtime, ARM_COMMAND
    from .model import Model
    from dual_sharpa_wave.qos import hand_qos

    arms = packet["plan_identity"]["active_arms"]
    runtimes, models, stream = {}, {}, None
    published = 0
    log.result.update(
        active_arms=arms,
        operation="shared_mock_stage_schedule",
        packet=packet,
        joint_names={s: list(sum(side_names(s), ())) for s in SIDES},
    )

    def spin() -> None:
        if canceled.is_set():
            raise RuntimeError("Shared execution canceled")
        for rt in runtimes.values():
            rt.spin(0.0)

    def feedback() -> dict:
        result = {}
        for side, rt in runtimes.items():
            if not rt.fresh() or rt.fault:
                raise RuntimeError(f"{side}: stale/invalid feedback: {rt.fault}")
            a, h = rt.feedback["arm"], rt.feedback["hand"]
            result[side] = dict(
                joint_names=list(sum(side_names(side), ())),
                positions_rad=a.q.tolist() + h.q.tolist(),
                source_timestamp=min(a.received, h.received),
                receive_timestamp=max(a.received, h.received),
            )
        return result

    try:
        for side in SIDES:
            config = receiver_config(packet, side)
            if len(arms) == 1 and side in arms:
                passive = "left" if side == "right" else "right"
                config["passive_positions_rad"] = packet["snapshot"][passive]["positions_rad"]
            runtimes[side] = Runtime(config, log)
        # No command publisher exists during either preflight. Even a passive side
        # must identify itself as mock before its stationary arm target can be sent.
        for side, rt in runtimes.items():
            rt.preflight(side in arms)
            for branch in SIDES:
                check_kinematics(packet["scene_kinematics"][branch], kinematic_contract(rt.description, branch))
            models[side] = Model(rt.description, rt.c)
        log.result["driver_interfaces_by_side"] = {}
        for side, rt in runtimes.items():
            log.result["driver_interfaces_by_side"][side] = dict(mock_verified=True, side=side)
        limits = {
            side: dict(
                joint_names=list(sum(side_names(side), ())),
                **{
                    field: models[side].limits["arm"][i].tolist() + models[side].limits["hand"][i].tolist()
                    for i, field in enumerate(("lower", "upper", "velocity"))
                },
            )
            for side in SIDES
        }
        stream = SharedStageStream(packet, limits, log.event)
        owner = runtimes[arms[0]]
        arm_pub = owner.node.create_publisher(owner.JointState, ARM_COMMAND, 1)
        hand_pubs = {
            s: owner.node.create_publisher(owner.JointState, runtimes[s].hand_command, hand_qos()) for s in arms
        }
        for side in arms:
            runtimes[side].pubs = dict(arm=arm_pub, hand=hand_pubs[side])
        log.result["command_publishers_created"] = True
        owner.wait(
            lambda: (
                arm_pub.get_subscription_count() >= 1
                and all(p.get_subscription_count() >= 1 for p in hand_pubs.values())
            ),
            "paired command discovery",
        )
        for _ in range(4):
            spin()
        for side in arms:
            runtimes[side].watchdog()
        stream.start(feedback(), time.monotonic())
        next_tick = stream.last_tick + 0.01
        while not stream.done:
            spin()
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.001, next_tick - now))
                continue
            # Validate both sides and all publishers before preparing any command.
            for side in arms:
                runtimes[side].watchdog()
            frame = stream.tick(now, feedback())
            if frame is None:
                continue
            if canceled.is_set():
                raise RuntimeError("Shared execution canceled before publication")
            stamp = owner.node.get_clock().now().to_msg()
            arm = owner.JointState(name=frame["arm"]["joint_names"], position=frame["arm"]["positions_rad"])
            arm.header.stamp = stamp
            hands = {}
            for side, data in frame["hands"].items():
                hands[side] = owner.JointState(name=data["joint_names"], position=data["positions_rad"])
                hands[side].header.stamp = stamp
            # DDS publication is not a transaction. A publish exception aborts the
            # complete stream; identical stamps expose any incomplete transport pair.
            arm_pub.publish(arm)
            for side in arms:
                hand_pubs[side].publish(hands[side])
            published += 1
            log.event(
                "paired_published",
                ros_stamp_ns=stamp.sec * 10**9 + stamp.nanosec,
                publisher_node=owner.node.get_fully_qualified_name(),
                **frame,
            )
            next_tick = now + 0.01
        log.result.update(
            final_state="DONE",
            task_completed=True,
            published_frames=published,
            completed_stages=stream.index,
            hand_arrival_required=False,
        )
        return log.result
    except BaseException as error:
        if stream is not None:
            stream.abort(str(error))
        log.result.update(
            final_state="FAULT",
            task_completed=False,
            abort_reason=str(error),
            completed_stages=stream.index if stream is not None else 0,
            published_frames=published,
        )
        for side in arms:
            if side in runtimes:
                runtimes[side].abort()
        raise
    finally:
        for rt in runtimes.values():
            rt.node.destroy_node()


def main(argv: list[str] | None = None) -> int:
    """Validate explicit isolated mock scope before ROS initialization; never use domain 87."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ros-domain-id", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        domain = validate_domain("mock", args.ros_domain_id)
        if os.environ.get("ROS_DOMAIN_ID") != str(domain):
            raise ValueError("ROS_DOMAIN_ID must explicitly match --ros-domain-id before startup")
        if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST":
            raise ValueError("Shared mock execution requires ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST")
        path = resolve_from_root(args.targets)
        packet = load_shared_packet(path)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from .runtime import RunLog, save

    root = resolve_from_root(args.output_root, "output")
    root.mkdir(parents=True, exist_ok=False)
    canceled = threading.Event()
    previous = {
        sig: signal.signal(sig, lambda signum, frame: canceled.set()) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    log = None
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    try:
        config = receiver_config(packet, packet["plan_identity"]["active_arms"][0])
        log = RunLog(str(root), config, args)
        save(
            root / "run_manifest.yaml",
            dict(
                schema_version=1,
                environment="mock",
                ros_domain_id=domain,
                packet_path=str(path),
                packet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        )
        execute_shared(packet, log, canceled)
        return 0
    except BaseException as error:
        if log is not None:
            log.result.update(final_state="FAULT", task_completed=False, abort_reason=str(error))
        print(f"Shared mock execution failed: {error}", flush=True)
        return 1
    finally:
        if log is not None:
            log.finish()
            save(root / "run_report.json", log.result)
        rclpy.try_shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
