"""Latest-only, read-only full-right telemetry; executable under ROS Python 3.12."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

from ada_grasp_ctrl.experiments.right_ros_contract import ARM_NAMES, HAND_NAMES, check_kinematics, kinematic_contract
from ada_grasp_ctrl.tactile_preview.ros_reader import Reader
from ada_grasp_ctrl.tactile_preview.data import stamp_seconds

ARM_TOPIC = "/crx5ia/right/joint_states"
HAND_TOPIC = "/sharpa/right_hand/joint_states"
DESCRIPTION = "/bimanual/robot_description"


def measured_positions(packet: dict, names: tuple[str, ...], now: float) -> list[float]:
    """Validate actual named feedback and both receipt clocks, never infer missing joints."""
    if not packet or packet.get("error"):
        raise ValueError("UNAVAILABLE joints: " + packet.get("error", "missing feedback"))
    received, stamp, ros_now = (packet.get(k, float("nan")) for k in ("received", "receipt_ros", "subscriber_ros"))
    if not all(math.isfinite(t) for t in (received, stamp, ros_now, now)):
        raise ValueError("INVALID joint clocks")
    age = now - received
    if stamp <= 0 or not 0 <= age <= 0.5 or not -0.05 <= ros_now - stamp + age <= 0.5:
        raise ValueError("STALE joint feedback")
    actual, q = packet.get("names", []), packet.get("positions", [])
    if len(actual) != len(q) or len(actual) != len(set(actual)) or set(actual) != set(names):
        raise ValueError("INVALID joint names: require unique actual 6+22 joints")
    if not all(math.isfinite(v) for v in q):
        raise ValueError("INVALID nonfinite joint feedback")
    by_name = dict(zip(actual, q))
    return [by_name[n] for n in names]


def full_feedback(packet: dict, now: float) -> list[float]:
    """Reject missing/mismatched model, ambiguous publishers and incoherent actual state."""
    if packet.get("model_valid") is not True:
        raise ValueError("MODEL UNAVAILABLE: " + packet.get("model_error", "waiting for robot description"))
    if packet.get("joint_sources") != {"arm": 1, "hand": 1, "description": 1}:
        raise ValueError("UNAVAILABLE / AMBIGUOUS joint or model publisher")
    arm, hand = packet.get("arm", {}), packet.get("joints", {})
    q = measured_positions(arm, ARM_NAMES, now) + measured_positions(hand, HAND_NAMES, now)
    if abs(arm["received"] - hand["received"]) > 0.15:
        raise ValueError("STALE / INCOHERENT arm and hand receipts")
    return q


class RobotReader(Reader):
    """Extend the existing tactile subscriber with arm feedback and model identity only."""

    def __init__(self, model_urdf: Path) -> None:
        from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        super().__init__(HAND_TOPIC)
        description = model_urdf.read_text()
        self.expected = kinematic_contract(description)
        self.model_sha256 = hashlib.sha256(description.encode()).hexdigest()
        self.arm: dict[str, Any] = {}
        self.model_valid = False
        self.model_error = "Waiting for compatible ROS description"
        self.description: str | None = None
        self.world_T_base = None
        self.subscriptions.extend(
            [
                self.node.create_subscription(JointState, ARM_TOPIC, self.receive_arm, qos_profile_sensor_data),
                self.node.create_subscription(
                    String,
                    DESCRIPTION,
                    self.receive_model,
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
                ),
            ]
        )

    def receive_arm(self, message: Any) -> None:
        """Copy arm state; all structural/freshness checks run before display."""
        packet = dict(
            names=list(message.name),
            positions=list(message.position),
            received=time.monotonic(),
            receipt_ros=stamp_seconds(message.header.stamp),
            subscriber_ros=self.node.get_clock().now().nanoseconds * 1e-9,
            error="" if message.header.frame_id == "base_link" else "Unexpected arm feedback frame",
        )
        with self.lock:
            self.arm = packet

    def receive_model(self, message: Any) -> None:
        """Latch model changes as monitor faults; no robot commands or driver services."""
        from ada_grasp_ctrl.grasp_selection.adapters.state import world_base_from_description

        with self.lock:
            try:
                if self.description is not None and self.description != message.data:
                    raise ValueError("ROS robot model changed during monitoring")
                if self.description is None:
                    check_kinematics(self.expected, kinematic_contract(message.data))
                    self.world_T_base = world_base_from_description(message.data)
                    self.description = message.data
                    self.model_valid, self.model_error = True, ""
            except (ValueError, KeyError, TypeError) as error:
                self.model_valid, self.model_error = False, str(error)

    def snapshot(self) -> dict[str, Any]:
        packet = super().snapshot()
        with self.lock:
            packet.update(
                arm=dict(self.arm),
                model_valid=self.model_valid,
                model_error=self.model_error,
                model_sha256=self.model_sha256,
                world_T_right_base=self.world_T_base,
            )
        return packet


def main() -> None:
    """Publish no ROS messages; stream bounded latest snapshots to the owning UI process."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-urdf", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init(args=[])
    reader = RobotReader(args.model_urdf)
    executor = SingleThreadedExecutor()
    executor.add_node(reader.node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    endpoints, sources, updated = {}, {}, -float("inf")
    try:
        while rclpy.ok():
            now = time.monotonic()
            if now - updated >= 1:
                endpoints, updated = reader.endpoints(), now
                sources = {
                    g: reader.node.count_publishers(t)
                    for g, t in (("arm", ARM_TOPIC), ("hand", HAND_TOPIC), ("description", DESCRIPTION))
                }
            packet = reader.snapshot()
            packet.update(kind="snapshot", endpoints=endpoints, joint_sources=sources, worker_monotonic=now)
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
