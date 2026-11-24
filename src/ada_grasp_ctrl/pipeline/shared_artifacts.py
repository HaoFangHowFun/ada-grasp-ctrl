"""Validate canonical planning data before explicit mock or hardware ROS projection."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from ada_grasp_ctrl.experiments.right_ros_contract import kinematic_contract, side_names
from ada_grasp_ctrl.ros_execution.shared_packet import KIND, SIDES, validate_shared_packet
from .shared_timing import validate_stage_schedule


def shared_mock_packet(schedule: dict, snapshot: dict[str, dict[str, float]], description: str) -> dict:
    """Project validated Task 1/3 data without adding physical execution capability.

    Args:
        schedule: Complete Task 3 shared stage schedule, including canonical GraspPlan.
        snapshot: Both sides' measured, named 28-joint starting states.
        description: Combined URDF used to bind the live mock model's kinematics.

    Returns:
        Explicit versioned mock-only JSON packet. Write with allow_nan=False.

    Raises:
        ValueError: On invalid planning data, missing states or mismatched starting positions.
    """
    return validate_shared_packet(_shared_projection(schedule, snapshot, description, environment="mock", kind=KIND))


def shared_hardware_packet(schedule: dict, snapshot: dict, description: str, source: dict) -> dict:
    """Export explicit hardware intent with source identity; operator review remains required."""
    from ada_grasp_ctrl.ros_execution.shared_hardware import KIND as HARDWARE_KIND, validate_hardware_packet

    packet = _shared_projection(schedule, snapshot, description, environment="hardware", kind=HARDWARE_KIND)
    packet.update(source=source, operator_review_required=True)
    return validate_hardware_packet(packet)


def _shared_projection(schedule: dict, snapshot: dict, description: str, *, environment: str, kind: str) -> dict:
    """Project canonical data once; public exporters validate their distinct scope."""
    checked = validate_stage_schedule(schedule)
    plan = checked.pop("plan")
    if set(snapshot) != set(SIDES):
        raise ValueError("Snapshot must contain both measured sides")
    states = {}
    for side in SIDES:
        names = list(sum(side_names(side), ()))
        if set(snapshot[side]) != set(names):
            raise ValueError(f"Snapshot requires all named joints for {side}")
        states[side] = dict(joint_names=names, positions_rad=[snapshot[side][n] for n in names])
    identity = {
        name: deepcopy(plan[name]) for name in ("schema_version", "candidate_id", "mode", "active_arms", "provenance")
    }
    identity["available_arms"] = plan.get("available_arms", plan["active_arms"]).copy()
    return dict(
        schema_version=1,
        kind=kind,
        environment=environment,
        hardware_ready=False,
        plan_identity=identity,
        schedule=checked,
        snapshot=states,
        source_schedule_sha256=hashlib.sha256(
            json.dumps(schedule, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        scene_kinematics={side: kinematic_contract(description, side) for side in SIDES},
    )
