"""Live joint-model validation for the Ada-owned ROS receiver."""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from dual_crx_control.robot.kinematics import CRXKinematics

from ada_grasp_ctrl.experiments.right_ros_contract import ARM_NAMES as ARM_ORDER, HAND_NAMES as HAND_ORDER

ARM_NAMES = list(ARM_ORDER)
HAND_NAMES = list(HAND_ORDER)


def vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected {size} finite values")
    return result


def named_positions(names: list[str], positions: Any, expected: list[str]) -> np.ndarray:
    if len(names) != len(positions) or len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError("feedback must contain exactly the expected unique joint names")
    mapping = dict(zip(names, positions))
    return vector([mapping[n] for n in expected], len(expected), "feedback")


class Model:
    def __init__(self, description: str, config: dict[str, Any]) -> None:
        self.description = description
        self.kin = CRXKinematics(description, config["tip_frame"], config["reference_frame"])
        if self.kin.joint_names != ARM_NAMES:
            raise ValueError("Kinematic chain must contain exactly right_J1..right_J6")
        root = ET.fromstring(description)
        self.limits = {}
        for group, names in (("arm", ARM_NAMES), ("hand", HAND_NAMES)):
            rows = []
            for name in names:
                j = root.find(f"joint[@name='{name}']")
                if j is None or j.get("type") != "revolute" or j.find("limit") is None:
                    raise ValueError(f"Missing revolute joint/limits: {name}")
                rows.append([float(j.find("limit").get(k, "nan")) for k in ("lower", "upper", "velocity")])
            low, high, speed = np.asarray(rows).T
            if not np.isfinite(rows).all() or np.any(low >= high) or np.any(speed <= 0):
                raise ValueError("Invalid model limits")
            self.limits[group] = (low, high, speed)
        # Preserve the base/tool/mount/palm contract explicitly in each plan.
        self.fixed_joints = [
            ET.tostring(j, encoding="unicode")
            for j in root.findall("joint")
            if j.get("type") == "fixed" and j.get("name", "").startswith("right_")
        ]

    def check(self, group: str, q: Any) -> np.ndarray:
        low, high, _ = self.limits[group]
        q = vector(q, len(low), group)
        if np.any(q < low - 1e-8) or np.any(q > high + 1e-8):
            raise ValueError(f"{group}: joint-limit violation")
        return q


@dataclass
class Sample:
    q: np.ndarray
    received: float
