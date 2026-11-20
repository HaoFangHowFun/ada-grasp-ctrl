"""Canonical schema-1 grasp data and explicit adapters; never execution commands.

The legacy ``grasp_selection.types.GraspCandidate`` API and all wire formats stay
unchanged. TypedDict constructors do not validate: call the validators at input
boundaries. Validated records are independent, JSON-compatible copies.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypedDict, cast

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.right_ros_contract import finite_vector
from .multi_arm import STAGES, hand_names, validate_candidate

SCHEMA_VERSION = 1
Arm = Literal["left", "right"]


class StageTarget(TypedDict):
    """World-from-hand-flange rigid matrix and named finger angles in radians."""

    frame_id: Literal["world"]
    pose_frame: str
    T_world_hand: list[list[float]]
    joint_positions_rad: dict[str, float]


class _GraspData(TypedDict):
    schema_version: Literal[1]
    candidate_id: str
    mode: str
    frame_id: Literal["world"]
    unit: Literal["m"]
    joint_unit: Literal["rad"]
    targets: dict[Arm, dict[str, StageTarget]]
    provenance: dict[str, Any]


class GraspCandidate(_GraspData):
    """Candidate requirements, independent of run-level arm availability."""

    required_arms: list[Arm]


class _PlanOptions(TypedDict, total=False):
    available_arms: list[Arm]


class GraspPlan(_GraspData, _PlanOptions):
    """Selected grasp targets, without trajectories or execution authorization."""

    active_arms: list[Arm]
    execution_ready: Literal[False]
    hardware_ready: Literal[False]


def _json_copy(value: Any, path: str) -> Any:
    """Copy strict JSON values, preserving keys and rejecting nonfinite metadata."""
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError(f"{path}: expected string keys")
        return {k: _json_copy(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path}: expected finite JSON data")


def _mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def _arms(value: Any, path: str) -> list[Arm]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(s, str) or s not in ("left", "right") for s in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{path}: expected nonempty, duplicate-free left/right arms")
    return list(value)


def _text(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: expected a nonempty string")


def _validate(row: object, arm_field: str) -> dict:
    raw = _mapping(row, "grasp")
    label = f"grasp[{raw.get('candidate_id', '<missing>')}]"
    data = _json_copy(raw, label)
    for name in ("candidate_id", "mode"):
        _text(data.get(name), f"{label}.{name}")
    if type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version: expected {SCHEMA_VERSION}")
    arms = _arms(data.get(arm_field), f"{label}.{arm_field}")
    data[arm_field] = arms
    targets = _mapping(data.get("targets"), f"{label}.targets")
    _mapping(data.get("provenance"), f"{label}.provenance")
    if set(targets) != set(arms):
        raise ValueError(f"{label}.targets: keys must exactly match {arm_field}")
    for side in arms:
        path = f"{label}.targets.{side}"
        stages = _mapping(targets[side], path)
        if set(stages) != set(STAGES):
            raise ValueError(f"{path}: expected exactly pregrasp/grasp/squeeze")
        for stage in STAGES:
            path = f"{label}.targets.{side}.{stage}"
            target = _mapping(stages[stage], path)
            matrix = target.get("T_world_hand")
            if not isinstance(matrix, list) or len(matrix) != 4:
                raise ValueError(f"{path}.T_world_hand: expected a 4x4 matrix")
            for i, values in enumerate(matrix):
                finite_vector(values, 4, f"{path}.T_world_hand[{i}]")
            joints = _mapping(target.get("joint_positions_rad"), f"{path}.joint_positions_rad")
            if set(joints) != set(hand_names(side)):
                raise ValueError(f"{path}.joint_positions_rad: expected 22 canonical {side} joint names")
            finite_vector(list(joints.values()), 22, f"{path}.joint_positions_rad")
            for field, expected in (("unit", "m"), ("joint_unit", "rad"), ("quaternion_order", "wxyz")):
                if field in target and target[field] != expected:
                    raise ValueError(f"{path}.{field}: expected {expected!r}")
            # Legacy targets may carry redundant XYZ/WXYZ. Check, never reinterpret them.
            if "position_m" in target:
                position = finite_vector(target["position_m"], 3, f"{path}.position_m")
                if not np.allclose(position, np.asarray(matrix)[:3, 3], atol=1e-6, rtol=0):
                    raise ValueError(f"{path}.position_m: disagrees with T_world_hand")
            if "quaternion_wxyz" in target:
                quat = np.array(finite_vector(target["quaternion_wxyz"], 4, f"{path}.quaternion_wxyz"))
                if abs(np.linalg.norm(quat) - 1) > 1e-4:
                    raise ValueError(f"{path}.quaternion_wxyz: expected a unit WXYZ quaternion")
                rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                if not np.allclose(rotation, np.asarray(matrix)[:3, :3], atol=1e-6, rtol=0):
                    raise ValueError(f"{path}.quaternion_wxyz: disagrees with T_world_hand")
    # Reuse the established schema-1 world/flange, units and proper-rigid checks.
    try:
        checked = validate_candidate({**data, "required_arms": arms})
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"{label}: {error}") from error
    data["targets"] = checked["targets"]
    return data


def validate_grasp_candidate(row: object) -> GraspCandidate:
    """Return an independent canonical candidate or raise ValueError on invalid data."""
    return cast(GraspCandidate, _validate(row, "required_arms"))


def validate_grasp_plan(row: object) -> GraspPlan:
    """Validate selected arms and targets, returning a data-only independent copy.

    Raises:
        ValueError: On malformed data, unavailable active arms or readiness claims.
    """
    data = _validate(row, "active_arms")
    if "required_arms" in data:
        raise ValueError("GraspPlan uses active_arms; candidate requirements belong in GraspCandidate")
    if "available_arms" in data:
        available = _arms(data["available_arms"], "available_arms")
        if not set(data["active_arms"]) <= set(available):
            raise ValueError("active_arms must be a subset of available_arms")
        data["available_arms"] = available
    for field in ("execution_ready", "hardware_ready"):
        if data.get(field) is not False:
            raise ValueError(f"{field} must be false: grasp data does not authorize execution")
    return cast(GraspPlan, data)


def candidate_from_schema1(row: dict, *, mode: str) -> GraspCandidate:
    """Adapt an existing per-arm candidate without changing its ID or target values.

    Args:
        row: Existing ``multi_arm.validate_candidate`` schema-1 data.
        mode: Explicit mode label; never inferred from available arms or a generator.

    Returns:
        Validated canonical candidate with original provenance and extension fields.

    Raises:
        ValueError: On invalid data or a conflicting existing mode label.
    """
    _mapping(row, "candidate")
    if "mode" in row and row["mode"] != mode:
        raise ValueError("candidate.mode differs from the supplied mode")
    return validate_grasp_candidate({**row, "mode": mode})


def candidate_to_schema1(row: GraspCandidate) -> dict:
    """Return legacy per-arm data; the caller must retain the canonical mode separately."""
    result = dict(validate_grasp_candidate(row))
    result.pop("mode")
    return result


def plan_from_candidate(candidate: GraspCandidate, *, available_arms: object = None) -> GraspPlan:
    """Select every required arm as data; no IK, ranking, trajectory or readiness grant.

    Args:
        candidate: Canonical candidate chosen by the caller.
        available_arms: Optional run-level permissions, distinct from selected arms.

    Returns:
        Validated data-only plan, including both-arm plans for future consumers.

    Raises:
        ValueError: On invalid candidates or unavailable selected arms.
    """
    row = validate_grasp_candidate(candidate)
    plan = {key: row[key] for key in _GraspData.__annotations__}
    plan.update(active_arms=row["required_arms"], execution_ready=False, hardware_ready=False)
    if available_arms is not None:
        plan["available_arms"] = _arms(available_arms, "available_arms")
    return validate_grasp_plan(plan)


def plan_from_selection(selection: dict, *, mode: str) -> GraspPlan:
    """Adapt a schema-1 multi-arm selector result, preserving its single selected side.

    Args:
        selection: Full ``multi_arm.select_candidates`` result with embedded candidate.
        mode: Explicit mode of the selected candidate.

    Returns:
        Canonical plan; the input and its legacy ``selected_side`` remain unchanged.

    Raises:
        ValueError: On no selection, inconsistent identities/arms or unavailable arms.
    """
    _mapping(selection, "selection")
    if type(selection.get("schema_version")) is not int or selection["schema_version"] != SCHEMA_VERSION:
        raise ValueError("selection.schema_version: expected 1")
    if selection.get("status") != "selected":
        raise ValueError("selection.status: no selected candidate")
    selected = _mapping(selection.get("selected"), "selection.selected")
    candidate = candidate_from_schema1(selected.get("candidate"), mode=mode)
    if selected.get("candidate_id") != candidate["candidate_id"]:
        raise ValueError("selection.selected.candidate_id differs from candidate")
    side = selected.get("selected_side")
    if candidate["required_arms"] != [side] or selected.get("required_arms") != [side]:
        raise ValueError("selection.selected_side/required_arms must match the single selected candidate arm")
    available = _arms(selection.get("enabled_arms"), "selection.enabled_arms")
    return plan_from_candidate(candidate, available_arms=available)
