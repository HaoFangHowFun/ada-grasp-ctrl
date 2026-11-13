"""ROS-independent validation and playback of an explicitly reviewed joint plan."""

from __future__ import annotations

import math
from typing import Any

from ada_grasp_ctrl.experiments.right_ros_contract import (
    JOINT_NAMES,
    CYCLE_PHASES,
    PHASE_SEQUENCES,
    finite_vector,
    side_names,
)
from ada_grasp_ctrl.ros_execution.motion import smoothstep
from ada_grasp_ctrl.mpc.reference import linear_reference
from ada_grasp_ctrl.mpc.contracts import controller_mode


def validate_plan(plan: dict[str, Any]) -> None:
    """Reject incomplete or malformed execution plans before preview or sending."""
    if plan.get("schema_version") != 1 or plan.get("environment") not in ("mock", "hardware"):
        raise ValueError("Expected a schema-1 mock or hardware plan")
    phases = tuple(plan.get("phases", []))
    if plan.get("joint_names") != list(sum(side_names(plan.get("side", "right")), ())) or phases not in PHASE_SEQUENCES:
        raise ValueError("Plan joint order or phases differ")
    if plan.get("receiver_settings", {}).get("side", "right") != plan.get("side", "right"):
        raise ValueError("Plan side differs from receiver")
    if "passive_positions_rad" in plan:
        finite_vector(plan["passive_positions_rad"], 28, "passive_positions_rad")
        if plan.get("receiver_settings", {}).get("active_arms") != ["left", "right"]:
            raise ValueError("Passive state requires the dual-arm stack")
    base_phases = phases[1:] if phases[0] == "above_object" else phases
    if plan["environment"] == "hardware" and (
        base_phases != CYCLE_PHASES or plan.get("receiver_settings", {}).get("environment") != "hardware"
    ):
        raise ValueError("Hardware requires the complete cycle and matching receiver settings")
    mode = controller_mode(
        {
            "environment": plan["environment"],
            "side": plan.get("side", "right"),
            "controller": {"mode": plan.get("controller_mode", "baseline")},
        }
    )
    if plan.get("interpolation") not in ("quintic_smoothstep", "linear"):
        raise ValueError("Unsupported plan interpolation")
    receiver_interpolation = plan.get("receiver_settings", {}).get("interpolation", "quintic_smoothstep")
    if plan["interpolation"] != receiver_interpolation or (mode == "linear" and receiver_interpolation != "linear"):
        raise ValueError("Preview/receiver interpolation differs from controller mode")
    if plan["interpolation"] == "linear" and (
        mode != "linear" or plan.get("receiver_settings", {}).get("interpolation") != "linear"
    ):
        raise ValueError("Linear preview requires the same explicit mock receiver mode")
    if len(plan.get("keyframes", [])) != len(phases) + 1 or len(plan.get("durations_s", [])) != len(phases):
        raise ValueError("Plan requires a measured start and all declared phase targets")
    for q in plan["keyframes"]:
        finite_vector(q, 28, "plan.keyframe")
    for t in [*plan["durations_s"], plan.get("settle_time_s")]:
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 < t <= 300:
            raise ValueError("Invalid plan duration")
    if "above_object" in phases:
        above = plan.get("above_object", {})
        if (
            above.get("solver_success") is not True
            or above.get("frame_id") != "world"
            or above.get("axis") != "+Z"
            or above.get("hand_policy")
            not in ("zero_joint_open", "retain_selected_pregrasp", "extrapolate_grasp_to_pregrasp")
            or above.get("used_for_candidate_selection") is not False
            or above.get("path_checked") is not False
            or above.get("collision_checked") is not False
        ):
            raise ValueError("Invalid above-object endpoint policy")
        values = [
            above.get(k)
            for k in (
                "clearance_m",
                "object_top_world_z_m",
                "approach_hand_lowest_world_z_m"
                if "approach_hand_lowest_world_z_m" in above
                else "pregrasp_lowest_world_z_m",
                "lowest_world_z_m",
                "height_m",
            )
        ]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Invalid above-object geometry")
        clearance, top, lowest, final, height = values
        if clearance <= 0 or abs(height - max(0.0, top + clearance - lowest)) > 1e-8:
            raise ValueError("Above-object height differs from requested clearance")
        if final < top + clearance - 0.001:
            raise ValueError("Above-object clearance is insufficient")
        start = finite_vector(above.get("start_flange_wxyz"), 7, "above-object start")
        target = finite_vector(above.get("target_flange_wxyz"), 7, "above-object target")
        if any(abs(target[i] - start[i] - (height if i == 2 else 0)) > 1e-8 for i in range(7)):
            raise ValueError("Above-object endpoint must retain pregrasp XY and orientation")
        if above.get("positions") != plan["keyframes"][1]:
            raise ValueError("Above-object joints differ from preview")
        if above["hand_policy"] == "zero_joint_open":
            if plan["keyframes"][1][6:] != [0.0] * 22:
                raise ValueError("Above-object endpoint must open all fingers to zero")
        elif above["hand_policy"] == "retain_selected_pregrasp":
            if plan["keyframes"][1][6:] != plan["keyframes"][2][6:]:
                raise ValueError("Above-object endpoint must preserve pregrasp fingers")
        else:
            factor = above.get("hand_extrapolation_factor")
            if (
                isinstance(factor, bool)
                or not isinstance(factor, (int, float))
                or not math.isfinite(factor)
                or factor < 1
            ):
                raise ValueError("Invalid above-object hand extrapolation factor")
            pregrasp = plan["keyframes"][phases.index("pregrasp") + 1][6:]
            grasp = plan["keyframes"][phases.index("grasp") + 1][6:]
            expected = [g + factor * (p - g) for p, g in zip(pregrasp, grasp)]
            if any(abs(a - b) > 1e-10 for a, b in zip(expected, plan["keyframes"][1][6:])):
                raise ValueError("Above-object hand differs from grasp-to-pregrasp extrapolation")
    if "lift" in phases:
        lift = plan.get("lift", {})
        if (
            lift.get("used_for_candidate_selection") is not False
            or lift.get("path_checked") is not False
            or lift.get("collision_checked") is not False
            or lift.get("solver_success") is not True
            or lift.get("frame_id") != "world"
            or lift.get("axis") != "+Z"
        ):
            raise ValueError("Lift must be an explicitly generated post-selection target without path/collision checks")
        height = lift.get("height_m")
        if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0:
            raise ValueError("Invalid lift height")
        start = finite_vector(lift.get("start_flange_wxyz"), 7, "lift start")
        target = finite_vector(lift.get("target_flange_wxyz"), 7, "lift target")
        if any(abs(target[i] - start[i] - (height if i == 2 else 0)) > 1e-8 for i in range(7)):
            raise ValueError("Lift must retain squeeze orientation and move only along world +Z")
        index = phases.index("lift") + 1
        if (
            lift.get("positions") != plan["keyframes"][index]
            or plan["keyframes"][index][6:] != plan["keyframes"][index - 1][6:]
        ):
            raise ValueError("Lift target must preserve the selected squeeze hand")
    if base_phases == CYCLE_PHASES:
        frames = plan["keyframes"]
        lift_index, hold_index, release_index, return_index = (
            phases.index(p) + 1 for p in ("lift", "hold", "release", "return")
        )
        cycle = plan.get("cycle", {})
        if (
            cycle.get("hold_s") != plan["durations_s"][hold_index - 1]
            or cycle.get("hold_starts_after") != "lift_succeeded"
        ):
            raise ValueError("Hold must start after lift arrival and match the reviewed duration")
        if frames[hold_index] != frames[lift_index]:
            raise ValueError("Hold must retain all lift joints")
        if (
            cycle.get("release_policy") != "zero_joint_open"
            or frames[release_index] != frames[lift_index][:6] + [0.0] * 22
        ):
            raise ValueError("Release must keep the lifted arm and open all hand joints")
        if (
            frames[return_index] != frames[0][:6] + [0.0] * 22
            or cycle.get("return_policy") != "measured_start_arm_open_hand"
        ):
            raise ValueError("Return must restore the measured arm start with the hand open")
        for phase in ("hold", "release", "return"):
            finite_vector(cycle.get("flange_targets_wxyz", {}).get(phase), 7, phase + " pose")


def check_start(
    actual: list[float], expected: list[float], *, hand_tolerance: float = 0.02, ignore_hand_error: bool = False
) -> None:
    """Keep strict preview-start checks; subsequent phases may use reviewed finger tolerance."""
    if isinstance(hand_tolerance, bool) or not math.isfinite(hand_tolerance) or hand_tolerance <= 0:
        raise ValueError("Hand start tolerance must be positive and finite")
    finite_vector(actual, 28, "actual start")
    finite_vector(expected, 28, "expected start")
    for index, (a, b) in enumerate(zip(actual, expected)):
        if index >= 6 and ignore_hand_error:
            continue
        tolerance = 0.01 if index < 6 else hand_tolerance
        if abs(a - b) > tolerance:
            raise ValueError(f"Robot moved since preview: {JOINT_NAMES[index]}; rerun selection and preview")


def phase_progress(plan: dict[str, Any], index: int) -> float:
    """Return the timeline percentage at a target endpoint, including preceding settles."""
    durations = plan["durations_s"]
    settle = plan["settle_time_s"]
    if not 0 <= index <= len(durations):
        raise ValueError("Phase index outside the plan")
    return (sum(durations[:index]) + max(0, index - 1) * settle) / (sum(durations) + len(durations) * settle) * 100


def sample_plan(plan: dict[str, Any], progress: float) -> list[float]:
    """Sample the reviewed Ada reference, including target settling periods."""
    if not math.isfinite(progress) or not 0 <= progress <= 100:
        raise ValueError("Progress must be between 0 and 100")
    elapsed = progress / 100 * (sum(plan["durations_s"]) + len(plan["durations_s"]) * plan["settle_time_s"])
    for index, duration in enumerate(plan["durations_s"]):
        if elapsed <= duration:
            a, b = plan["keyframes"][index : index + 2]
            if plan.get("interpolation") == "linear":
                return linear_reference(a, b, elapsed, duration)
            alpha = smoothstep(min(1.0, elapsed / duration))
            return [x + alpha * (y - x) for x, y in zip(a, b)]
        elapsed -= duration
        if elapsed <= plan["settle_time_s"]:
            return list(plan["keyframes"][index + 1])
        elapsed -= plan["settle_time_s"]
    return list(plan["keyframes"][-1])


def phase_at_progress(plan: dict[str, Any], progress: float) -> str:
    """Identify the displayed phase, including its nominal settling interval."""
    elapsed = progress / 100 * (sum(plan["durations_s"]) + len(plan["phases"]) * plan["settle_time_s"])
    for phase, duration in zip(plan["phases"], plan["durations_s"]):
        if elapsed <= duration + plan["settle_time_s"]:
            return phase
        elapsed -= duration + plan["settle_time_s"]
    return plan["phases"][-1]
