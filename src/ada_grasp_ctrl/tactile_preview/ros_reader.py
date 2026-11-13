"""ROS Jazzy worker: subscriptions only, compact stdout snapshots, no SDK imports."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from typing import Any

from .data import LatestFrames, TOPICS, stamp_seconds


class Reader:
    """Keep DDS callbacks independent of browser and pipe I/O."""

    def __init__(self, joint_topic: str) -> None:
        import rclpy
        from dual_sharpa_wave.msg import TactileFrame
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import JointState

        self.node = rclpy.create_node("ada_tactile_preview", enable_rosout=False, start_parameter_services=False)
        self.cache = LatestFrames()
        self.lock = threading.Lock()
        self.joints: dict[str, Any] = {}
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.subscriptions = [
            self.node.create_subscription(
                TactileFrame,
                topic,
                lambda msg, f=finger: self.cache.receive(msg, f, self.node.get_clock().now().nanoseconds * 1e-9),
                qos,
            )
            for finger, topic in TOPICS.items()
        ]
        self.subscriptions.append(self.node.create_subscription(JointState, joint_topic, self.receive_joints, qos))

    def receive_joints(self, message: Any) -> None:
        """Preserve names and stamps; malformed feedback immediately invalidates pose."""
        names, positions = list(message.name), list(message.position)
        valid = len(names) == len(positions) and len(names) == len(set(names)) and all(map(math.isfinite, positions))
        packet = dict(
            names=names if valid else [],
            positions=positions if valid else [],
            received=time.monotonic(),
            receipt_ros=stamp_seconds(message.header.stamp),
            subscriber_ros=self.node.get_clock().now().nanoseconds * 1e-9,
            error="" if valid else "Malformed joint state",
        )
        with self.lock:
            self.joints = packet

    def snapshot(self) -> dict[str, Any]:
        """Copy current tactile/pose data and observed publisher endpoint QoS."""
        packet = self.cache.snapshot()
        with self.lock:
            packet["joints"] = dict(self.joints)
        return packet

    def endpoints(self) -> dict[str, Any]:
        """Inspect actual endpoint QoS, including zero or ambiguous publishers."""
        return {
            f: [
                dict(
                    node=e.node_namespace.rstrip("/") + "/" + e.node_name,
                    type=e.topic_type,
                    reliability=e.qos_profile.reliability.name,
                    durability=e.qos_profile.durability.name,
                    depth=e.qos_profile.depth,
                )
                for e in self.node.get_publishers_info_by_topic(t)
            ]
            for f, t in TOPICS.items()
        }


def main() -> None:
    """Stream at 20 Hz while a separate executor keeps only latest samples."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-topic", default="/sharpa/right_hand/joint_states")
    args = parser.parse_args()
    rclpy.init(args=[])
    reader = Reader(args.joint_topic)
    executor = SingleThreadedExecutor()
    executor.add_node(reader.node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    endpoints, updated = {}, 0.0
    try:
        while rclpy.ok():
            now = time.monotonic()
            if now - updated > 1:
                endpoints, updated = reader.endpoints(), now
            packet = reader.snapshot()
            packet.update(kind="snapshot", endpoints=endpoints, worker_monotonic=now)
            sys.stdout.write(json.dumps(packet, allow_nan=False) + "\n")
            sys.stdout.flush()
            time.sleep(0.05)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        executor.shutdown(timeout_sec=2)
        thread.join(timeout=2)
        reader.node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
