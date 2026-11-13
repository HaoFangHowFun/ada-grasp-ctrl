"""Ada-owned feedback, command ownership checks and bounded abort handling."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable
import xml.etree.ElementTree as ET

import numpy as np
from ada_grasp_ctrl.ros_execution.model import Sample, named_positions
from ada_grasp_ctrl.experiments.right_ros_contract import side_names

ARM_COMMAND = "/crx5ia/joint_targets"
HAND_COMMAND = "/sharpa/right_hand/joint_command"
DESCRIPTION = "/bimanual/robot_description"
MANAGER = "/crx5ia/right/controller_manager"


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class RunLog:
    def __init__(self, root: str, config: dict[str, Any], args: argparse.Namespace) -> None:
        self.path = Path(root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{os.getpid()}")
        self.path.mkdir(parents=True)
        self.stream = (self.path / "events.jsonl").open("w", buffering=1)
        self.result = dict(
            final_state="WAIT_FEEDBACK",
            task_completed=False,
            object_lifted=None,
            object_held=None,
            abort_reason=None,
            environment=config.get("environment"),
            ros_domain_id=os.getenv("ROS_DOMAIN_ID"),
            joint_names=dict(zip(("arm", "hand"), side_names(config.get("side", "right")))),
            force_limiter=config.get("force_limiter"),
            sensors={},
            hardware_validated=False,
            command_publishers_created=False,
        )
        save(self.path / "config.json", config)
        versions = {"python": sys.version, "ros_distro": os.getenv("ROS_DISTRO"), "arguments": vars(args)}
        # Hash the installed/source files actually used, independent of git availability.
        import inspect
        from ada_grasp_ctrl.ros_execution import bridge, model as core, settings
        from dual_crx_control.robot import kinematics, ik_solver
        from dual_sharpa_wave import joint_names

        versions["source_sha256"] = {
            str(Path(inspect.getfile(m))): hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest()
            for m in (sys.modules[__name__], bridge, core, settings, kinematics, ik_solver, joint_names)
        }
        versions["repositories"] = {}
        for module in (sys.modules[__name__], kinematics, joint_names):
            location = Path(inspect.getfile(module)).resolve().parent
            result = subprocess.run(
                ["git", "-C", str(location), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            )
            status = subprocess.run(
                ["git", "-C", str(location), "status", "--short"], capture_output=True, text=True, check=False
            )
            versions["repositories"][str(location)] = dict(commit=result.stdout.strip(), status=status.stdout.strip())
        save(self.path / "versions.json", versions)
        print(f"Run log: {self.path}", flush=True)

    def event(self, kind: str, **data: Any) -> None:
        self.stream.write(json.dumps(dict(time=time.monotonic(), kind=kind, **data), allow_nan=False) + "\n")

    def finish(self) -> None:
        save(self.path / "result.json", self.result)
        self.stream.close()


class Runtime:
    @property
    def side(self) -> str:
        return self.c.get("side", "right")

    @property
    def hand_command(self) -> str:
        return f"/sharpa/{self.side}_hand/joint_command"

    @property
    def manager(self) -> str:
        return f"/crx5ia/{self.side}/controller_manager"

    @property
    def passive_side(self) -> str:
        return "left" if self.side == "right" else "right"

    def check_passive(self) -> None:
        """Reject passive-side drift from the state used for selection and preview."""
        if "passive_positions_rad" not in self.c:
            return
        from ada_grasp_ctrl.pipeline.plan import check_start

        if not self.fresh():
            raise RuntimeError("Dual-arm feedback expired")
        actual = np.r_[self.feedback["passive_arm"].q, self.feedback["passive_hand"].q]
        check_start(actual.tolist(), self.c["passive_positions_rad"])

    def __init__(self, c: dict[str, Any], log: RunLog) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState, Image
        from std_msgs.msg import String, Float64MultiArray
        from geometry_msgs.msg import WrenchStamped

        self.rclpy, self.JointState = rclpy, JointState
        self.c, self.log = c, log
        self.node = Node(f"ada_{self.side}_arm_hand_runtime")
        arm_names, hand_names = side_names(self.side)
        self.feedback, self.stamps = {}, {}
        self.description = self.arm_description = None
        self.fault = None
        self.limiter = self.hand_state = None
        self.require_hand_state = False
        self.pubs = {}
        self.readonly = True
        self.last_probe = 0.0
        self.probe_future = None
        self.controller_client = None
        transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for group, topic, names in (
            ("arm", f"/crx5ia/{self.side}/joint_states", arm_names),
            ("hand", f"/sharpa/{self.side}_hand/joint_states", hand_names),
        ):
            self.node.create_subscription(
                JointState, topic, lambda m, g=group, n=names: self.receive(g, n, m), qos_profile_sensor_data
            )
        if "passive_positions_rad" in c:
            for group, topic, names in (
                ("passive_arm", f"/crx5ia/{self.passive_side}/joint_states", side_names(self.passive_side)[0]),
                ("passive_hand", f"/sharpa/{self.passive_side}_hand/joint_states", side_names(self.passive_side)[1]),
            ):
                self.node.create_subscription(
                    JointState, topic, lambda m, g=group, n=names: self.receive(g, n, m), qos_profile_sensor_data
                )
        self.node.create_subscription(
            String, DESCRIPTION, lambda m: self.description_received("description", m.data), transient
        )
        self.node.create_subscription(
            String,
            f"/crx5ia/{self.side}/robot_description",
            lambda m: self.description_received("arm_description", m.data),
            transient,
        )
        self.node.create_subscription(
            String, "/crx5ia/collision_force_limiter/state", lambda m: setattr(self, "limiter", m.data), transient
        )
        self.node.create_subscription(
            String, f"/sharpa/{self.side}_hand/control_state", self.receive_hand_state, transient
        )
        self.node.create_subscription(
            WrenchStamped,
            f"/crx5ia/{self.side}/force_torque_sensor_broadcaster/wrench",
            self.wrench,
            qos_profile_sensor_data,
        )
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for suffix, msg_type in (("dist_force", Image), ("f6", Float64MultiArray)):
                topic = f"/sharpa/{self.side}_hand/tactile/{finger}/{suffix}"
                self.node.create_subscription(
                    msg_type, topic, lambda m, t=topic: self.tactile(t, m), qos_profile_sensor_data
                )

    def receive_hand_state(self, message: Any) -> None:
        try:
            self.hand_state = json.loads(message.data)["state"]
        except (ValueError, KeyError, TypeError):
            self.fault = "Malformed hand control_state"

    def description_received(self, field: str, value: str) -> None:
        old = getattr(self, field)
        if old is not None and old != value:
            self.fault = f"{field} changed during task"
        setattr(self, field, value)

    def receive(self, group: str, names: list[str], message: Any) -> None:
        try:
            q = named_positions(message.name, message.position, names)
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            age = self.node.get_clock().now().nanoseconds * 1e-9 - stamp
            if stamp <= 0 or age > self.c["feedback_timeout"] or age < -self.c["feedback_timeout"]:
                raise ValueError("old, zero or future timestamp")
            if stamp <= self.stamps.get(group, -1):
                return  # Replayed samples never refresh the watchdog.
            self.stamps[group] = stamp
            self.feedback[group] = Sample(q, time.monotonic())
        except (ValueError, TypeError) as exc:
            self.fault = f"{group}: invalid feedback: {exc}"

    def wrench(self, m: Any) -> None:
        values = [
            m.wrench.force.x,
            m.wrench.force.y,
            m.wrench.force.z,
            m.wrench.torque.x,
            m.wrench.torque.y,
            m.wrench.torque.z,
        ]
        valid = bool(np.isfinite(values).all())
        self.log.result["sensors"]["wrench"] = dict(
            received=time.monotonic(), valid=valid, frame=m.header.frame_id, synthetic=self.c["environment"] == "mock"
        )
        if valid:
            self.log.event("wrench", values=values, frame=m.header.frame_id)

    def tactile(self, topic: str, m: Any) -> None:
        self.log.result["sensors"][topic] = dict(received=time.monotonic(), units="uncalibrated SDK units")
        if hasattr(m, "encoding"):
            self.log.event("tactile", topic=topic, encoding=m.encoding, width=m.width, height=m.height)
        elif np.isfinite(m.data).all():
            self.log.event("tactile_f6", topic=topic, values=list(m.data))

    def spin(self, seconds: float = 0.01) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=seconds)
        if not self.rclpy.ok():
            raise KeyboardInterrupt
        if self.fault:
            raise RuntimeError(self.fault)

    def wait(self, predicate: Callable[[], bool], reason: str) -> None:
        deadline = time.monotonic() + self.c["startup_timeout"]
        while not predicate():
            self.spin()
            if time.monotonic() > deadline:
                raise TimeoutError("Startup timeout: " + reason)

    def fresh(self) -> bool:
        now = time.monotonic()
        return all(
            g in self.feedback and now - self.feedback[g].received < self.c["feedback_timeout"]
            for g in (
                ("arm", "hand", "passive_arm", "passive_hand") if "passive_positions_rad" in self.c else ("arm", "hand")
            )
        )

    def call(self, client: Any, request: Any, timeout: float = 3.0) -> Any:
        deadline = time.monotonic() + timeout
        while not client.service_is_ready():
            self.spin()
            if time.monotonic() > deadline:
                raise TimeoutError(f"Service unavailable: {client.srv_name}")
        future = client.call_async(request)
        while not future.done():
            self.spin()
            if time.monotonic() > deadline:
                raise TimeoutError(f"Service timeout: {client.srv_name}")
        return future.result()

    def parameters(self, node: str, names: list[str]) -> dict[str, Any]:
        from rcl_interfaces.srv import GetParameters
        from rclpy.parameter import parameter_value_to_python

        client = self.node.create_client(GetParameters, node + "/get_parameters")
        try:
            response = self.call(client, GetParameters.Request(names=names))
            return dict(zip(names, [parameter_value_to_python(v) for v in response.values]))
        finally:
            self.node.destroy_client(client)

    def preflight(self, execute: bool) -> None:
        self.wait(
            lambda: self.description is not None and self.arm_description is not None and self.fresh(),
            f"combined model, driver model and fresh {self.side}-arm/{self.side}-hand feedback",
        )
        hand = self.parameters(f"/sharpa/{self.side}_hand/hand_node", ["side", "backend", "read_only", "control_mode"])
        if hand["side"] != self.side:
            raise RuntimeError(f"Hand driver is not configured for {self.side} side")
        plugins = [e.text for e in ET.fromstring(self.arm_description).findall("ros2_control/hardware/plugin")]
        mock = bool(plugins) and all(p == "mock_components/GenericSystem" for p in plugins)
        expected_plugin = (
            "mock_components/GenericSystem"
            if self.c["environment"] == "mock"
            else "fanuc_robot_driver/FanucHardwareInterface"
        )
        if not plugins or any(plugin != expected_plugin for plugin in plugins):
            raise RuntimeError("Configured environment does not match actual arm hardware plugin")
        if (self.c["environment"] == "mock") != mock or hand["backend"] != ("mock" if mock else "sharpa_sdk"):
            raise RuntimeError("Configured environment does not match actual arm/hand drivers")
        self.log.result["driver_interfaces"] = dict(hand=hand, arm_plugins=plugins)
        self.log.result["sensor_note"] = (
            "Mock wrench is synthetic; tactile may be absent; no sensing validation"
            if mock
            else "Availability only; tactile units/frame not calibrated"
        )
        if not execute:
            return
        if hand["read_only"]:
            raise RuntimeError("Hand driver is read-only")
        if self.c["environment"] == "hardware":
            if hand["control_mode"] != "mit":
                raise RuntimeError("Hardware task requires MIT; SDK position timeout closes the session")
            reviewed_joint_goal = self.c["mode"] == "joint" and self.c.get("operator_reviewed_joint_goal") is True
            if not self.c["path_verified"] and not reviewed_joint_goal:
                raise RuntimeError("Hardware requires a reviewed path and path_verified: true")
            if self.c["mode"] == "cartesian" and (not self.c["mount_verified"] or not self.c.get("mount_config")):
                raise RuntimeError("Hardware Cartesian execution requires a verified mount_config")
        interp = self.parameters("/crx5ia/joint_interpolation", ["active_arms", "input_rate_hz", "method"])
        if (
            interp["active_arms"] != self.c.get("active_arms", [self.side])
            or interp["input_rate_hz"] != self.c["rate_hz"]
        ):
            raise RuntimeError("Require configured active_arms and matching input_rate_hz")
        self.log.result["driver_interfaces"]["interpolator"] = interp
        self.hand_mode = hand["control_mode"]
        self.require_hand_state = hand["backend"] == "sharpa_sdk" and self.hand_mode == "mit"
        if self.require_hand_state:
            self.wait(lambda: self.hand_state == "normal", "hand MIT control state")
        if self.c["force_limiter"] == "required":
            self.wait(lambda: self.limiter == "ARMED", "force limiter ARMED")
        from controller_manager_msgs.srv import ListControllers

        self.controller_client = self.node.create_client(ListControllers, self.manager + "/list_controllers")
        self.verify_controllers(self.call(self.controller_client, ListControllers.Request()))
        self.wait(
            lambda: all(self.receiver_present(t) for t in (ARM_COMMAND, self.hand_command)),
            f"{self.side}-side command receiver discovery",
        )
        for topic in (ARM_COMMAND, self.hand_command):
            if self.node.count_publishers(topic):
                raise RuntimeError(f"Another sender owns {topic}")
            if not self.receiver_present(topic):
                raise RuntimeError(f"Expected driver command receiver: {topic}")
        low = f"/crx5ia/{self.side}/forward_position_controller/commands"
        self.wait(lambda: bool(self.node.get_publishers_info_by_topic(low)), "interpolator output discovery")
        owners = self.node.get_publishers_info_by_topic(low)
        if len(owners) != 1 or owners[0].node_name != "joint_interpolation":
            raise RuntimeError("Unexpected direct arm command sender")
        self.check_passive()
        self.readonly = False

    def receiver_present(self, topic: str) -> bool:
        expected = (
            ("joint_interpolation", "/crx5ia") if topic == ARM_COMMAND else ("hand_node", f"/sharpa/{self.side}_hand")
        )
        receivers = self.node.get_subscriptions_info_by_topic(topic)
        return sum((i.node_name, i.node_namespace) == expected for i in receivers) == 1

    @staticmethod
    def verify_controllers(response: Any) -> None:
        states = {c.name: c.state for c in response.controller}
        for name in ("forward_position_controller", "joint_state_broadcaster"):
            if states.get(name) != "active":
                raise RuntimeError(f"Arm controller not active: {name}")

    def watchdog(self) -> None:
        self.check_passive()
        now = time.monotonic()
        if self.c["force_limiter"] == "required":
            if self.limiter != "ARMED" or self.node.count_publishers("/crx5ia/collision_force_limiter/state") != 1:
                raise RuntimeError(f"Force limiter unavailable/not ARMED: {self.limiter}")
        if self.require_hand_state and (
            self.hand_state != "normal" or self.node.count_publishers(f"/sharpa/{self.side}_hand/control_state") != 1
        ):
            raise RuntimeError(f"Hand driver control state: {self.hand_state}")
        for group, topic in (("arm", ARM_COMMAND), ("hand", self.hand_command)):
            if self.node.count_publishers(topic) != 1 or not self.receiver_present(topic):
                raise RuntimeError(f"Command ownership/receiver changed: {topic}")
        from controller_manager_msgs.srv import ListControllers

        if self.probe_future is not None:
            if self.probe_future.done():
                self.verify_controllers(self.probe_future.result())
                self.probe_future = None
            elif now - self.last_probe > 2.0:
                raise RuntimeError("Controller status probe timed out")
        if self.probe_future is None and now - self.last_probe > 0.5:
            self.probe_future = self.controller_client.call_async(ListControllers.Request())
            self.last_probe = now

    def abort(self) -> None:
        """Best-effort software arm deactivation; bounded hand target is retained.

        Never issue a new hand pose from stale feedback or claim motor disable.
        """
        if not self.pubs:
            self.log.result["stop"] = {"requested": False, "reason": "No command publishers created"}
            return
        from controller_manager_msgs.srv import SwitchController

        managers = self.c.get("active_arms", [self.side]) if "passive_positions_rad" in self.c else [self.side]
        for side in managers:
            self._abort_arm(side, SwitchController)

    def _abort_arm(self, side: str, service_type: Any) -> None:
        """Stop all controlled arms after a dual-stack fault, without issuing a hand target."""
        SwitchController = service_type
        client = self.node.create_client(SwitchController, f"/crx5ia/{side}/controller_manager/switch_controller")
        request = SwitchController.Request(deactivate_controllers=["forward_position_controller"], strictness=1)
        request.timeout.sec = 2
        # Do not route stop spins through fault checks, which caused this abort.
        outcome = {
            "requested": False,
            "controller_deactivated": False,
            "hand": "No further commands. Last bounded target / backend timeout policy; motors not disabled",
            "physical_stop_confirmed": False,
        }
        try:
            if client.service_is_ready():
                future = client.call_async(request)
                outcome["requested"] = True
                deadline = time.monotonic() + 3
                while self.rclpy.ok() and not future.done() and time.monotonic() < deadline:
                    self.rclpy.spin_once(self.node, timeout_sec=0.02)
                outcome["controller_deactivated"] = bool(future.done() and future.result().ok)
        except Exception as exc:
            outcome["error"] = str(exc)
        self.log.result.setdefault("stops", {})[side] = outcome
        self.log.result["stop"] = outcome
        self.node.destroy_client(client)
