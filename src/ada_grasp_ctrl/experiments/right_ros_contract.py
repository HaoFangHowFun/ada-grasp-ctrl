"""Pure-data contract for the experimental Ada-to-ROS right-arm sender.

This module can run under ROS system Python without importing MuJoCo or Torch.
It does not register a new supported hand or change the evaluation pipeline.
"""

from __future__ import annotations

import math
from typing import Any
import xml.etree.ElementTree as ET

PHASES = ("pregrasp", "grasp", "squeeze")
LIFT_PHASES = PHASES + ("lift",)
CYCLE_PHASES = LIFT_PHASES + ("hold", "release", "return")
PHASE_SEQUENCES = (PHASES, LIFT_PHASES, CYCLE_PHASES) + tuple(
    ("above_object",) + phases for phases in (PHASES, LIFT_PHASES, CYCLE_PHASES)
)
ARM_NAMES = tuple(f"right_J{i}" for i in range(1, 7))
HAND_NAMES = tuple(
    "right_" + name
    for name in (
        "thumb_CMC_FE",
        "thumb_CMC_AA",
        "thumb_MCP_FE",
        "thumb_MCP_AA",
        "thumb_IP",
        "index_MCP_FE",
        "index_MCP_AA",
        "index_PIP",
        "index_DIP",
        "middle_MCP_FE",
        "middle_MCP_AA",
        "middle_PIP",
        "middle_DIP",
        "ring_MCP_FE",
        "ring_MCP_AA",
        "ring_PIP",
        "ring_DIP",
        "pinky_CMC",
        "pinky_MCP_FE",
        "pinky_MCP_AA",
        "pinky_PIP",
        "pinky_DIP",
    )
)
JOINT_NAMES = ARM_NAMES + HAND_NAMES
ACTION = "/bimanual/right_arm_hand/follow_joint_trajectory"


def side_names(side: str = "right") -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the explicit arm/hand order without importing a robot runtime."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    return tuple(f"{side}_J{i}" for i in range(1, 7)), tuple(side + n[5:] for n in HAND_NAMES)


def packet_side(packet: dict[str, Any]) -> str:
    """Keep legacy right packets explicit while accepting side-bound exports."""
    side = packet.get("side", "right")
    side_names(side)
    if packet.get("kind") == "ada_right_joint_targets":
        if side != "right":
            raise ValueError("Legacy right packet cannot target the left side")
    elif packet.get("kind") != "ada_side_joint_targets" or "side" not in packet:
        raise ValueError("Unsupported Ada ROS packet kind")
    return side


def action_name(side: str = "right") -> str:
    """Return the selected side's passive trajectory action."""
    side_names(side)
    return f"/bimanual/{side}_arm_hand/follow_joint_trajectory"


def finite_vector(value: Any, size: int, field: str) -> list[float]:
    """Validate one finite vector without silently padding or clipping values."""
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{field}: expected {size} values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError(f"{field}: expected finite numbers")
    return [float(v) for v in value]


def validate_packet(packet: dict[str, Any]) -> None:
    """Reject malformed transport input before any ROS goal is created.

    Args:
        packet: JSON-compatible export of a validated Ada prepared sample.

    Raises:
        ValueError: If schema, frame, names or target values do not match the wire contract.
    """
    side = packet_side(packet)
    names = sum(side_names(side), ())
    if packet.get("schema_version") != 1:
        raise ValueError("Unsupported Ada ROS packet schema")
    if packet.get("units") != "rad" or packet.get("joint_names") != list(names):
        raise ValueError("Expected named right CRX + Sharpa joints in radians")
    if packet.get("reference_frame") != "world" or packet.get("pose_frame") != f"{side}_hand_flange":
        raise ValueError("Expected registered world-frame right_hand_flange targets")
    if packet.get("environment") not in ("mock", "hardware"):
        raise ValueError("Expected an explicit mock or hardware environment")
    if packet["environment"] == "hardware" and (
        packet.get("source", {}).get("kind") != "grasp_selection"
        or packet.get("reviewed_plan", {}).get("environment") != "hardware"
    ):
        raise ValueError("Hardware targets require a reviewed selection plan")
    phases = tuple(packet.get("execution_phases", PHASES))
    if phases not in PHASE_SEQUENCES or len(packet.get("targets", [])) != len(phases):
        raise ValueError("Expected grasp phases, optional lift, or the complete hold/release/return cycle")
    for phase, target in zip(phases, packet["targets"]):
        if target.get("phase") != phase:
            raise ValueError("Unexpected phase ordering")
        finite_vector(target["positions"], 28, phase)
        pose = finite_vector(target["flange_target_wxyz"], 7, phase + ".flange_target_wxyz")
        if abs(sum(v * v for v in pose[3:]) - 1) > 1e-4:
            raise ValueError(f"{phase}: nonunit WXYZ quaternion")
    if not packet.get("kinematics"):
        raise ValueError("Missing source model kinematic contract")


def kinematic_contract(description: str, side: str = "right") -> dict[str, Any]:
    """Extract the named moving joints and their ancestors, ignoring visual assets.

    Returns:
        Numeric joint origins, axes and limits indexed by child link, including
        world/base/mount/flange/palm chains. Fixed joint names may differ by exporter.
    """
    joint_names = sum(side_names(side), ())
    root = ET.fromstring(description)
    joints = root.findall("joint")
    names = {j.get("name"): j for j in joints}
    children = {j.find("child").get("link"): j for j in joints}
    selected = set()
    for name in joint_names:
        if name not in names or names[name].get("type") != "revolute":
            raise ValueError(f"Missing revolute {name}")
    starts = [names[n].find("child").get("link") for n in joint_names]
    starts.extend([f"{side}_hand_flange", f"{side}_hand_C_MC"])
    for link in starts:
        seen = set()
        while link != "world":
            if link in seen or link not in children:
                raise ValueError(f"Disconnected/cyclic world chain at {link}")
            seen.add(link)
            selected.add(link)
            link = children[link].find("parent").get("link")
    result = {}
    for child in sorted(selected):
        joint = children[child]
        origin = joint.find("origin")
        row = {"parent": joint.find("parent").get("link"), "type": joint.get("type")}
        for key in ("xyz", "rpy"):
            raw = "0 0 0" if origin is None else origin.get(key, "0 0 0")
            row[key] = finite_vector([float(v) for v in raw.split()], 3, child + "." + key)
        if joint.get("type") != "fixed":
            axis = joint.find("axis")
            row["name"] = joint.get("name")
            row["axis"] = finite_vector(
                [float(v) for v in ("1 0 0" if axis is None else axis.get("xyz")).split()], 3, child + ".axis"
            )
            limit = joint.find("limit")
            row["limits"] = finite_vector(
                [float(limit.get(key)) for key in ("lower", "upper", "velocity")], 3, child + ".limits"
            )
        result[child] = row
    return result


def check_kinematics(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Reject model/base/mount drift; tolerate only floating-point serialization noise."""
    if expected.keys() != actual.keys():
        raise ValueError("ROS/source kinematic link sets differ")
    for link, row in expected.items():
        if row.keys() != actual[link].keys():
            raise ValueError(f"ROS/source kinematic fields differ: {link}")
        for key, value in row.items():
            other = actual[link][key]
            if isinstance(value, list):
                if len(value) != len(other) or any(abs(a - b) > 1e-8 for a, b in zip(value, other)):
                    raise ValueError(f"ROS/source kinematics differ: {link}.{key}")
            elif value != other:
                raise ValueError(f"ROS/source kinematics differ: {link}.{key}")
