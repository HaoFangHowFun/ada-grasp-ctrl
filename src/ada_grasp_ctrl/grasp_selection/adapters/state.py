"""Named snapshot validation and URDF identity; only standard-library imports here."""

import hashlib
import math
import xml.etree.ElementTree as ET

ARM_NAMES = tuple(f"right_J{i}" for i in range(1, 7))


def named_arm(names: list[str], positions: list[float]) -> list[float]:
    """Extract exact right joints regardless of order; reject mixed-side snapshots."""
    if len(names) != len(positions) or len(names) != len(set(names)):
        raise ValueError("Joint names/positions are incomplete or duplicated")
    if any(not isinstance(n, str) or n.startswith("left_") for n in names):
        raise ValueError("Mixed or left-side state is not a right-only snapshot")
    if not set(ARM_NAMES).issubset(names):
        raise ValueError("Missing right_J1..right_J6")
    if any(isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) for q in positions):
        raise ValueError("Nonfinite or nonnumeric joint position")
    return [float(positions[names.index(n)]) for n in ARM_NAMES]


def kinematic_contract(description: str) -> dict:
    """Extract every right-side joint and its world ancestors, excluding visuals."""
    root = ET.fromstring(description)
    joints = root.findall("joint")
    children = {j.find("child").get("link"): j for j in joints}
    selected = {}
    for j in joints:
        if not j.get("name", "").startswith("right_"):
            continue
        link, visited = j.find("child").get("link"), set()
        while link != "world":
            if link in visited or link not in children:
                raise ValueError("Missing/cyclic world chain")
            visited.add(link)
            selected[link] = children[link]
            link = children[link].find("parent").get("link")
    if not set(ARM_NAMES).issubset(j.get("name") for j in selected.values()):
        raise ValueError("Not a real named right CRX model")
    result = {}
    for child, joint in sorted(selected.items()):
        row = {"parent": joint.find("parent").get("link"), "type": joint.get("type")}
        origin = joint.find("origin")
        for key in ("xyz", "rpy"):
            row[key] = [float(x) for x in (origin.get(key, "0 0 0") if origin is not None else "0 0 0").split()]
        if joint.get("type") != "fixed":
            row["name"] = joint.get("name")
            row["axis"] = [float(x) for x in joint.find("axis").get("xyz").split()]
            row["limits"] = [float(joint.find("limit").get(k)) for k in ("lower", "upper")]
        for key in ("xyz", "rpy", "axis", "limits"):
            if key in row and (
                len(row[key]) != (2 if key == "limits" else 3) or not all(math.isfinite(x) for x in row[key])
            ):
                raise ValueError("Invalid URDF kinematics: " + child)
        result[child] = row
    return result


def check_description(expected: str, actual: str) -> None:
    """Require every expected chain, axis, mount and limit in the observed description."""
    first, second = kinematic_contract(expected), kinematic_contract(actual)
    moving_first = {c for c, row in first.items() if row["type"] != "fixed"}
    moving_second = {c for c, row in second.items() if row["type"] != "fixed"}
    if moving_first != moving_second:
        raise ValueError("Right-side moving joint sets differ")
    for child, row in first.items():
        if child not in second or row.keys() != second[child].keys():
            raise ValueError("Description link mismatch: " + child)
        for key, value in row.items():
            other = second[child][key]
            if isinstance(value, list):
                if len(value) != len(other) or any(abs(a - b) > 1e-8 for a, b in zip(value, other)):
                    raise ValueError(f"Description kinematics mismatch: {child}.{key}")
            elif value != other:
                raise ValueError(f"Description kinematics mismatch: {child}.{key}")


def world_base_from_description(description: str) -> list[list[float]]:
    """Read the explicit fixed world-to-right-base joint from the deployment URDF."""
    base = kinematic_contract(description)["right_base_link"]
    if base["parent"] != "world" or base["type"] != "fixed":
        raise ValueError("Expected explicit fixed world-to-right-base transform")
    r, p, y = base["rpy"]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    x, yy, z = base["xyz"]
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, yy],
        [-sp, cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def load_state(payload: dict, model_id: str, *, replay: bool, now_s: float | None = None, max_age_s: float = 0.5):
    """Validate a captured state; replay evaluates freshness at the recorded receipt time."""
    from ..types import RobotState

    if payload.get("schema_version") != 1 or payload.get("unit") != "rad":
        raise ValueError("Expected schema 1 state in radians")
    if payload.get("model_id") != model_id:
        raise ValueError("State/model identity mismatch")
    q = named_arm(payload["joint_names"], payload["positions_rad"])
    if payload.get("environment") != "fixture":
        if payload.get("source_frame_id") != payload["expected_source_frame_id"]:
            raise ValueError("Joint state header frame mismatch")
        expected, observed = payload["expected_description"], payload["observed_description"]
        if hashlib.sha256(expected.encode()).hexdigest() != model_id:
            raise ValueError("Expected description hash mismatch")
        check_description(expected, observed)
        if payload.get("world_T_right_base") != world_base_from_description(observed):
            raise ValueError("Snapshot transform differs from observed description")
    state = RobotState(
        q,
        payload["stamp_s"],
        payload["received_s"],
        model_id,
        payload["source_topic"],
        payload["environment"],
        payload["frame_id"],
        payload["clock"],
    )
    if not replay and state.environment != "hardware":
        raise ValueError("Live selection requires hardware state; use explicit replay for mock/fixtures")
    if not replay and state.clock != "ros_system":
        raise ValueError("Live snapshot evaluation requires the system ROS clock")
    when = state.received_s if replay else now_s
    if when is None:
        raise ValueError("Live evaluation requires current time in the state clock")
    state.check_fresh(when, max_age_s)
    return state, when
