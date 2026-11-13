"""Opt-in compact read-only contact capture using ROS Python; no viewer or SDK calls."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
import json
from pathlib import Path
import time

from .data import FINGERS, TOPICS, normalize_frame
from .geometry import GeometryCatalog


def main() -> None:
    """Record same-message F6/contact evidence and explicitly diagnostic local geometry."""
    import rclpy
    from dual_sharpa_wave.msg import TactileFrame
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import JointState

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--finger", choices=FINGERS, default="index")
    parser.add_argument("--geometry-config", type=Path)
    args = parser.parse_args()
    if not 0 < args.duration <= 600:
        parser.error("duration must be finite and in (0, 600] seconds")
    catalog = (
        GeometryCatalog.from_config(args.geometry_config)
        if args.geometry_config
        else GeometryCatalog.installed_candidate()
    )
    counts: Counter = Counter()
    pending: deque = deque(maxlen=64)
    joint_stamp = [None]
    profile = catalog.maps.get(args.finger)
    profile_metadata = (
        dict(
            source=profile.source,
            units=profile.units,
            meters_per_unit=profile.meters_per_unit,
            T_E_S=profile.T_E_S.tolist(),
            uv_to_col_row=profile.uv_to_col_row.tolist(),
            verified=profile.verified,
            evidence=profile.evidence,
        )
        if profile
        else dict(error=catalog.errors.get(args.finger, "no profile"))
    )
    rclpy.init(args=[])
    node = rclpy.create_node("ada_contact_capture", enable_rosout=False, start_parameter_services=False)
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    # Exclusive creation prevents silently overwriting evidence from an earlier run.
    try:
        with args.output.open("x") as stream:
            stream.write(
                json.dumps(
                    dict(
                        kind="metadata",
                        finger=args.finger,
                        clock="ROS receipt and local monotonic",
                        interpretation="p_E/n_E are diagnostic only; no world pose or certified alignment",
                        geometry_profile=profile_metadata,
                    )
                )
                + "\n"
            )

            def receive(msg: TactileFrame) -> None:
                try:
                    frame = normalize_frame(
                        msg, args.finger, time.monotonic(), node.get_clock().now().nanoseconds * 1e-9
                    )
                except (ValueError, TypeError, AttributeError) as exc:
                    counts["REJECTED"] += 1
                    pending.append(dict(kind="rejected", error=str(exc)))
                    return
                counts[frame.contact_status] += 1
                record = dict(kind="frame", **asdict(frame), joint_stamp=joint_stamp[0])
                if len(pending) == pending.maxlen:
                    counts["DROPPED_RECORDS"] += 1
                pending.append(record)

            def write_record(record: dict) -> None:
                if record.get("contact_status") == "VALID":
                    try:
                        contact = catalog.lookup(args.finger, record["contact_row"][:2], True)
                        record["geometry"] = dict(
                            pixel_row=contact.row,
                            pixel_col=contact.col,
                            p_S=contact.p_S.tolist(),
                            n_S=contact.n_S.tolist(),
                            p_E=contact.p_E.tolist(),
                            n_E=contact.n_E.tolist(),
                            status=contact.verification,
                            units=contact.units,
                            source=contact.source,
                        )
                    except ValueError as exc:
                        record["geometry_error"] = str(exc)
                stream.write(json.dumps(record, allow_nan=False) + "\n")

            def receive_joints(msg: JointState) -> None:
                joint_stamp[0] = dict(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec)

            node.create_subscription(TactileFrame, TOPICS[args.finger], receive, qos)
            node.create_subscription(JointState, "/sharpa/right_hand/joint_states", receive_joints, qos)
            print(f"CAPTURE_READY finger={args.finger} duration={args.duration}s output={args.output}", flush=True)
            end = time.monotonic() + args.duration
            try:
                while time.monotonic() < end and rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.05)
                    while pending:
                        write_record(pending.popleft())
            except KeyboardInterrupt:
                pass
            while pending:
                write_record(pending.popleft())
            stream.write(json.dumps(dict(kind="summary", counts=dict(counts))) + "\n")
        print(json.dumps(dict(output=str(args.output), counts=dict(counts))), flush=True)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
