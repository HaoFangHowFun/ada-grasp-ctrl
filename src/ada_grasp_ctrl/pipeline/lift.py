"""Generate a post-selection lift target without candidate scoring or collision checks."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.crx_sharpa_right_model import matrix_pose
from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import ANGLE_TOLERANCE, POSITION_TOLERANCE, residual
from ada_grasp_ctrl.grasp_selection.types import vector


class LiftTargetError(ValueError):
    """The selected grasp is retained, but no valid lift command can be generated."""


def inward_offset_limit(value: object) -> float:
    """Validate the operator-approved horizontal endpoint search bound in metres."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or not 0 <= value <= 0.1
    ):
        raise ValueError("Maximum inward offset must be finite and between 0 and 0.1 m")
    return float(value)


def lift_target(
    adapter: object,
    squeeze: object,
    height_m: float,
    *,
    max_inward_offset_m: float = 0.0,
    base_position_world: object = None,
) -> dict:
    """Convert one selected squeeze pose plus world +Z into a bounded joint target.

    Args:
        adapter: The selected candidate's model and world transform.
        squeeze: Selected 28-axis squeeze target, in canonical joint order.
        height_m: Positive world-Z displacement in meters.
        max_inward_offset_m: Optional horizontal search radius, at most 0.1 m.
            Try vertical first, then increasing radii toward the active base.
        base_position_world: Active arm base XYZ in world; required for offset search.

    Returns:
        Exact hand-preserving target and computation provenance. Collision and
        path checking are deliberately absent; candidate ranking is not consulted.

    Raises:
        ValueError: If no finite converged target can be generated. This blocks
            execution of the plan and never triggers candidate reselection.
    """
    if isinstance(height_m, bool) or not np.isfinite(height_m) or height_m <= 0:
        raise ValueError("Lift height must be positive and finite")
    max_inward_offset_m = inward_offset_limit(max_inward_offset_m)
    start_q = vector(squeeze, 28, "squeeze")
    start = adapter.kin.forward(start_q).copy()
    world_model = np.linalg.inv(adapter.model_T_world)
    start_world = world_model @ start
    base = vector(base_position_world, 3, "active base position") if max_inward_offset_m else None
    target = start.copy()

    def offsets() -> Iterator[np.ndarray]:
        yield np.zeros(2)
        if base is None:
            return
        toward = base[:2] - start_world[:2, 3]
        distance = np.linalg.norm(toward)
        if distance < 1e-10:
            return
        direction = toward / distance
        # A fine angular fan covers near-tangential solutions without permitting
        # an endpoint farther from the selected base than the original endpoint.
        angles = [0, *[signed for a in (15, 30, 45, 60, 75, 80, 85) for signed in (a, -a)]]
        radii = np.linspace(0, max_inward_offset_m, int(np.ceil(max_inward_offset_m / 0.01)) + 1)[1:]
        for radius in radii:
            for angle in np.deg2rad(angles):
                c, s = np.cos(angle), np.sin(angle)
                offset = radius * (np.array([[c, -s], [s, c]]) @ direction)
                if np.linalg.norm(toward - offset) <= distance:
                    yield offset

    def error(arm: np.ndarray) -> np.ndarray:
        actual = adapter.kin.forward(np.r_[arm, start_q[6:]])
        return np.r_[
            10 * (actual[:3, 3] - target[:3, 3]), Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).as_rotvec()
        ]

    attempts = []
    for offset in offsets():
        target = start.copy()
        target[:3, 3] += adapter.model_T_world[:3, :3] @ [*offset, height_m]
        # Every trial starts from the selected pose; a rejected solve never seeds
        # a command or changes the candidate, orientation, height or tolerances.
        fit = least_squares(
            error, start_q[:6], bounds=adapter.kin.bounds, max_nfev=160, ftol=1e-10, xtol=1e-10, gtol=1e-10
        )
        arm = vector(fit.x, 6, "lift arm")
        q = vector(np.r_[arm, start_q[6:]], 28, "lift")
        actual = adapter.kin.forward(q)
        errors = residual(actual, target)
        accepted = bool(
            fit.success
            and np.all(arm >= adapter.kin.bounds[0])
            and np.all(arm <= adapter.kin.bounds[1])
            and all(np.isfinite(value) for value in errors.values())
            and errors["position_m"] <= POSITION_TOLERANCE
            and errors["orientation_rad"] <= ANGLE_TOLERANCE
        )
        attempts.append(
            dict(offset_world_xy_m=offset.tolist(), accepted=accepted, residual=errors, message=str(fit.message))
        )
        if accepted:
            break
    else:
        raise LiftTargetError(
            f"Selected candidate retained, but lift target could not be computed: {errors}; {fit.message}; "
            f"tried {len(attempts)} endpoint(s), inward offset limit {max_inward_offset_m:g} m"
        )
    result = {
        "positions": q.tolist(),
        "height_m": float(height_m),
        "frame_id": "world",
        "axis": "+Z_with_inward_offset" if np.any(offset) else "+Z",
        "start_flange_wxyz": matrix_pose(world_model @ start).tolist(),
        "target_flange_wxyz": matrix_pose(world_model @ target).tolist(),
        "actual_flange_wxyz": matrix_pose(world_model @ actual).tolist(),
        "solver_success": bool(fit.success),
        "solver_message": str(fit.message),
        "residual": errors,
        "ik_used_to_compute_target": True,
        "used_for_candidate_selection": False,
        "path_checked": False,
        "collision_checked": False,
        "hand_policy": "retain_selected_squeeze",
        "orientation_policy": "retain_squeeze_fk_orientation",
    }
    if max_inward_offset_m:
        result["horizontal_adjustment"] = dict(
            policy="toward_active_base",
            max_distance_m=float(max_inward_offset_m),
            base_position_world_m=base.tolist(),
            offset_world_xy_m=offset.tolist(),
            attempts=attempts,
        )
    return result
