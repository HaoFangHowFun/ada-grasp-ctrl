"""Official hand loading and joint validation, independent of ROS and Viser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yourdfpy

from ada_grasp_ctrl.experiments.right_ros_contract import HAND_NAMES
from .data import FINGERS


def load_hand(path: Path, package_share: Path, *, load_meshes: bool = True) -> yourdfpy.URDF:
    """Load original URDF visual meshes and FK without copying or altering assets."""

    def filename_handler(fname: str) -> str:
        filename = fname
        prefix = "package://dual_sharpa_wave/"
        if filename.startswith(prefix):
            return str((package_share / filename[len(prefix) :]).resolve(strict=True))
        if filename.startswith("package://"):
            raise ValueError(f"Unsupported mesh package: {filename}")
        return str((path.parent / filename).resolve(strict=True))

    hand = yourdfpy.URDF.load(
        str(path), filename_handler=filename_handler, load_collision_meshes=False, load_meshes=load_meshes
    )
    if hand.base_link != "right_hand_flange" or set(hand.actuated_joint_names) != set(HAND_NAMES):
        raise ValueError("Expected the official right-hand-only URDF with 22 named joints and right_hand_flange root")
    for finger in FINGERS:
        hand.get_transform(f"right_{finger}_elastomer")
    return hand


def joint_configuration(
    packet: dict[str, Any],
    hand: yourdfpy.URDF,
    now: float,
    stale_after: float,
    synchronized: bool,
) -> tuple[np.ndarray | None, str]:
    """Reject incomplete/stale poses instead of extrapolating or padding missing joints."""
    if not packet:
        return None, "STATIC / UNCERTAIN: neutral or last pose; no joint feedback"
    if packet.get("error"):
        return None, "INVALID POSE: " + packet["error"]
    clocks = [packet.get(name, float("nan")) for name in ("received", "receipt_ros", "subscriber_ros")]
    if not np.isfinite(clocks).all():
        return None, "INVALID POSE: nonfinite or missing receipt clock"
    age = now - packet["received"]
    if age < -0.05 or age > stale_after:
        return None, "STALE POSE: last posture retained; world arrows hidden"
    if synchronized:
        source_age = packet["subscriber_ros"] - packet["receipt_ros"] + age
        if packet["receipt_ros"] <= 0 or source_age < -0.05 or source_age > stale_after:
            return None, "STALE / CLOCK ERROR POSE: world arrows hidden"
    names, positions = packet["names"], packet["positions"]
    if len(names) != len(positions) or len(set(names)) != len(names) or set(names) != set(hand.actuated_joint_names):
        return None, "INVALID POSE: expected all 22 unique right-hand joint names"
    lookup = dict(zip(names, positions))
    q = np.array([lookup[n] for n in hand.actuated_joint_names], dtype=float)
    if not np.isfinite(q).all():
        return None, "INVALID POSE: nonfinite joint position"
    for name, value in zip(hand.actuated_joint_names, q):
        limit = hand.joint_map[name].limit
        if limit is not None and not limit.lower - 1e-3 <= value <= limit.upper + 1e-3:
            return None, f"INVALID POSE: {name} outside URDF limits"
    suffix = "ROS receipt clocks asserted synchronized" if synchronized else "source age unverified"
    return q, f"LIVE hand FK ({suffix}); viewer world = fixed right_hand_flange"
