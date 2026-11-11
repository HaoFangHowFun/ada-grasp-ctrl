"""Add one overhead endpoint before the selected pregrasp, without reranking."""

from __future__ import annotations

import numpy as np

from ada_grasp_ctrl.experiments.table_clearance import HandGeometry
from ada_grasp_ctrl.experiments.crx_sharpa_right_model import matrix_pose
from .lift import lift_target


class AboveObjectTargetError(ValueError):
    """The selected grasp has no valid overhead approach endpoint."""


def above_object_target(
    adapter: object,
    pregrasp: object,
    object_points_world: object,
    clearance_m: float,
    *,
    grasp: object | None = None,
    hand_extrapolation_factor: float = 1.0,
) -> dict:
    """Raise pregrasp until the lowest hand mesh clears the observed object top.

    Retain pregrasp XY and wrist orientation. Optionally extend the finger motion
    from grasp toward pregrasp, then recompute mesh clearance for that hand shape.
    Only a local endpoint IK is computed; execution uses joint interpolation.
    """
    if isinstance(clearance_m, bool) or not np.isfinite(clearance_m) or clearance_m <= 0:
        raise AboveObjectTargetError("Above-object clearance must be positive and finite")
    points = np.asarray(object_points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise AboveObjectTargetError("Above-object target requires finite observed object points")
    q = np.asarray(pregrasp, dtype=float).copy()
    if q.shape != (28,) or not np.isfinite(q).all():
        raise AboveObjectTargetError("Above-object target requires 28 finite pregrasp joints")
    geometry = HandGeometry(adapter, {"normal_xyz": [0, 0, 1], "d": 0})
    pregrasp_lowest = geometry.measure(q)["gap_m"]
    if (
        isinstance(hand_extrapolation_factor, bool)
        or not np.isfinite(hand_extrapolation_factor)
        or hand_extrapolation_factor < 1
    ):
        raise AboveObjectTargetError("Hand extrapolation factor must be finite and at least 1")
    if hand_extrapolation_factor != 1:
        closing = np.asarray(grasp, dtype=float)
        if closing.shape != (28,) or not np.isfinite(closing).all():
            raise AboveObjectTargetError("Hand extrapolation requires 28 finite grasp joints")
        q[6:] = closing[6:] + hand_extrapolation_factor * (q[6:] - closing[6:])
        if np.any(q[6:] < adapter.limits[6:, 0]) or np.any(q[6:] > adapter.limits[6:, 1]):
            raise AboveObjectTargetError("Extrapolated hand exceeds joint limits; reduce --above-object-hand-factor")
    top = float(points[:, 2].max())
    lowest = geometry.measure(q)["gap_m"]
    height = max(0.0, top + clearance_m - lowest)
    start = np.linalg.inv(adapter.model_T_world) @ adapter.kin.forward(q)
    target = start.copy()
    target[2, 3] += height
    try:
        if height > 0:
            q = np.asarray(lift_target(adapter, q, height)["positions"])
    except ValueError as error:
        raise AboveObjectTargetError("Selected grasp retained; above-object IK failed: " + str(error)) from error
    final = geometry.measure(q)["gap_m"]
    if final < top + clearance_m - 0.001:
        raise AboveObjectTargetError("Above-object endpoint misses clearance by more than 1 mm")
    if np.any(q < adapter.limits[:, 0]) or np.any(q > adapter.limits[:, 1]):
        raise AboveObjectTargetError("Above-object endpoint exceeds joint limits")
    return {
        "positions": q.tolist(),
        "clearance_m": float(clearance_m),
        "object_top_world_z_m": top,
        "pregrasp_lowest_world_z_m": pregrasp_lowest,
        "approach_hand_lowest_world_z_m": lowest,
        "lowest_world_z_m": final,
        "height_m": height,
        "start_flange_wxyz": matrix_pose(start).tolist(),
        "target_flange_wxyz": matrix_pose(target).tolist(),
        "frame_id": "world",
        "axis": "+Z",
        "hand_policy": "retain_selected_pregrasp"
        if hand_extrapolation_factor == 1
        else "extrapolate_grasp_to_pregrasp",
        "hand_extrapolation_factor": float(hand_extrapolation_factor),
        "solver_success": True,
        "used_for_candidate_selection": False,
        "path_checked": False,
        "collision_checked": False,
    }
