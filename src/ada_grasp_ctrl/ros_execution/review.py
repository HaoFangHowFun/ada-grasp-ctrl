"""Bind operator-reviewed execution to one exact, ordered pipeline packet."""

from __future__ import annotations

from typing import Any

from ada_grasp_ctrl.experiments.right_ros_contract import CYCLE_PHASES, finite_vector, validate_packet
from ada_grasp_ctrl.pipeline.plan import validate_plan


def check_reviewed_goal(
    packet: dict[str, Any], index: int, positions: list[float], duration: float, config: dict[str, Any]
) -> None:
    """Reject goals outside the reviewed sequence, environment, limits or timing.

    This is command identity validation, not a collision/path verification claim.
    Hardware authorization applies only to the complete pipeline cycle; arbitrary
    external joint goals retain the runtime's original hardware path requirement.
    """
    validate_packet(packet)
    plan = packet["reviewed_plan"]
    validate_plan(plan)
    if packet["environment"] != config["environment"] or plan["environment"] != config["environment"]:
        raise ValueError("Reviewed environment differs from receiver")
    phases = tuple(plan["phases"])
    base_phases = phases[1:] if phases[0] == "above_object" else phases
    if config["environment"] == "hardware" and base_phases != CYCLE_PHASES:
        raise ValueError("Hardware review requires the complete grasp/hold/release/return cycle")
    if not 0 <= index < len(packet["targets"]):
        raise ValueError("Reviewed sequence is already complete")
    reviewed_settings = dict(plan["receiver_settings"])
    reviewed_settings.setdefault("hand_arrival_required", True)
    if set(reviewed_settings) != set(config):
        raise ValueError("Receiver settings differ from reviewed plan")
    for key, value in reviewed_settings.items():
        if config.get(key) != value:
            raise ValueError("Receiver settings differ from reviewed plan: " + key)
    actual = finite_vector(positions, 28, "received goal")
    target = packet["targets"][index]
    if actual != target["positions"] or abs(duration - target["duration_s"]) > 1e-8:
        raise ValueError("Goal does not match the next reviewed target and duration")
