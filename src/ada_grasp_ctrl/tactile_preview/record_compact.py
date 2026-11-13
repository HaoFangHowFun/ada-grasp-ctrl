"""Record right-hand ROS data to MCAP while explicitly omitting dist_force payloads."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from ada_grasp_ctrl.paths import resolve_external_root

JOINT_TOPICS = (
    "/crx5ia/right/joint_states",
    "/sharpa/right_hand/joint_states",
    "/crx5ia/joint_targets",
    "/sharpa/right_hand/joint_command",
)
TACTILE_TOPICS = tuple(
    f"/sharpa/right_hand/tactile/{finger}/frame" for finger in ("thumb", "index", "middle", "ring", "pinky")
)
TOPIC_TYPES = {
    **dict.fromkeys(JOINT_TOPICS, "sensor_msgs/msg/JointState"),
    **dict.fromkeys(TACTILE_TOPICS, "dual_sharpa_wave/msg/TactileFrame"),
}


def omit_distribution(message: Any) -> Any:
    """Copy a tactile frame and replace only its distribution block with an omission marker.

    The original block metadata is retained in the marker, without its numeric payload.
    MISSING means unavailable in this recording; it does not mean measured zero or EMPTY.
    """
    from dual_sharpa_wave.msg import TactileBlock

    original = message.dist_force
    metadata = {
        "reason": "omitted_by_compact_recorder",
        "original_status": int(original.status),
        "original_shape": list(original.shape),
        "original_shape_declared": original.shape_declared,
        "original_declared_shape_json": original.declared_shape_json,
        "original_dtype": original.dtype,
        "original_error": original.error,
    }
    result = copy(message)
    result.dist_force = TactileBlock(status=TactileBlock.MISSING, error=json.dumps(metadata, sort_keys=True))
    return result


class CompactBag:
    """Write original joint CDR and reduced tactile CDR without publishing ROS messages."""

    def __init__(self, path: Path) -> None:
        import rosbag2_py

        if path.exists():
            raise FileExistsError(path)
        self.counts: Counter[str] = Counter()
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(
            rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"), rosbag2_py.ConverterOptions("", "")
        )
        for index, (topic, kind) in enumerate(TOPIC_TYPES.items()):
            self.writer.create_topic(
                rosbag2_py.TopicMetadata(id=index, name=topic, type=kind, serialization_format="cdr")
            )

    def write(self, topic: str, payload: bytes, timestamp_ns: int) -> None:
        """Preserve callback receipt time and all fields except the declared omission."""
        if topic not in TOPIC_TYPES:
            raise ValueError(f"Unexpected recording topic: {topic}")
        if topic in TACTILE_TOPICS:
            from dual_sharpa_wave.msg import TactileFrame
            from rclpy.serialization import deserialize_message, serialize_message

            payload = serialize_message(omit_distribution(deserialize_message(payload, TactileFrame)))
        self.writer.write(topic, payload, timestamp_ns)
        self.counts[topic] += 1

    def close(self) -> None:
        """Flush the MCAP and rosbag metadata."""
        self.writer.close()


def main() -> int:
    """Record until Ctrl+C; this process subscribes and writes files only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", help="Parent directory; relative paths are anchored to the source checkout")
    parser.add_argument("--run-seconds", type=float, help="Optional recording duration; otherwise stop with Ctrl+C")
    args = parser.parse_args()
    if args.run_seconds is not None and (not math.isfinite(args.run_seconds) or args.run_seconds <= 0):
        parser.error("--run-seconds must be finite and positive")
    domain = os.environ.get("ROS_DOMAIN_ID", "87")
    os.environ["ROS_DOMAIN_ID"] = domain
    try:
        import rclpy
        from rclpy.event_handler import SubscriptionEventCallbacks
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rosidl_runtime_py.utilities import get_message

        kinds = {topic: get_message(kind) for topic, kind in TOPIC_TYPES.items()}
        parent = resolve_external_root("output", args.output_root)
    except Exception as exc:
        print(f"Recorder preflight failed: {exc}", flush=True)
        return 2
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / f"physical_compact_{tag}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "ros_domain_id": domain,
        "bag": str(output / "bag"),
        "topics": TOPIC_TYPES,
        "omitted": "TactileFrame.dist_force numeric payload; marked MISSING with explicit omission provenance",
        "timestamps": "Recorder ROS-clock callback receipt times; source message timestamps preserved",
        "qos": "VOLATILE; BEST_EFFORT tactile, RELIABLE joints; depth 100",
        "actuator_publishers": False,
    }
    (output / "run_manifest.yaml").write_text(json.dumps(manifest, indent=2) + "\n")
    bag, node = None, None
    lost: Counter[str] = Counter()
    status, error = "completed", ""
    try:
        bag = CompactBag(output / "bag")
        rclpy.init(args=[])
        node = rclpy.create_node("ada_compact_recorder", enable_rosout=False, start_parameter_services=False)

        def receive(payload: bytes, topic: str) -> None:
            timestamp = node.get_clock().now().nanoseconds
            bag.write(topic, payload, timestamp)

        def message_lost(event: Any, topic: str) -> None:
            lost[topic] += event.total_count_change

        for topic, kind in kinds.items():
            reliability = ReliabilityPolicy.BEST_EFFORT if topic in TACTILE_TOPICS else ReliabilityPolicy.RELIABLE
            node.create_subscription(
                kind,
                topic,
                lambda payload, t=topic: receive(payload, t),
                QoSProfile(depth=100, reliability=reliability),
                raw=True,
                event_callbacks=SubscriptionEventCallbacks(message_lost=lambda event, t=topic: message_lost(event, t)),
            )
        print(f"Recording without dist_force on ROS domain {domain}: {output / 'bag'}\nStop with Ctrl+C.", flush=True)
        began, updated = time.monotonic(), 0.0
        while rclpy.ok() and (args.run_seconds is None or time.monotonic() - began < args.run_seconds):
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - updated >= 5:
                print(json.dumps({"messages": dict(bag.counts), "dds_reported_lost": dict(lost)}), flush=True)
                updated = time.monotonic()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        status, error = "execution_error", str(exc)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
        if bag is not None:
            try:
                bag.close()
            except Exception as exc:
                status, error = "execution_error", f"{error}; bag close failed: {exc}"
        counts = {topic: bag.counts[topic] if bag is not None else 0 for topic in TOPIC_TYPES}
        report = {
            "schema_version": 1,
            "status": status,
            "error": error,
            "messages": counts,
            "topics_without_messages": [topic for topic, count in counts.items() if not count],
            "dds_reported_lost": dict(lost),
            "loss_note": "DDS events do not include all upstream losses or SDK cache overwrites.",
        }
        (output / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {output}\n{json.dumps(report)}", flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
