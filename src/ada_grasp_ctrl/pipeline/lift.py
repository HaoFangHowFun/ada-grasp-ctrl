"""Generate a post-selection lift target without candidate scoring or collision checks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.crx_sharpa_right_model import matrix_pose
from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import ANGLE_TOLERANCE, POSITION_TOLERANCE, residual
from ada_grasp_ctrl.grasp_selection.types import vector


class LiftTargetError(ValueError):
    """The selected grasp is retained, but no valid lift command can be generated."""


def lift_target(adapter: object, squeeze: object, height_m: float) -> dict:
    """Convert one selected squeeze pose plus world +Z into a bounded joint target.

    Args:
        adapter: The selected candidate's model and world transform.
        squeeze: Selected 28-axis squeeze target, in canonical joint order.
        height_m: Positive world-Z displacement in meters.

    Returns:
        Exact hand-preserving target and computation provenance. Collision and
        path checking are deliberately absent; candidate ranking is not consulted.

    Raises:
        ValueError: If no finite converged target can be generated. This blocks
            execution of the plan and never triggers candidate reselection.
    """
    if isinstance(height_m, bool) or not np.isfinite(height_m) or height_m <= 0:
        raise ValueError("Lift height must be positive and finite")
    start_q = vector(squeeze, 28, "squeeze")
    start = adapter.kin.forward(start_q).copy()
    target = start.copy()
    target[:3, 3] += adapter.model_T_world[:3, :3] @ [0.0, 0.0, height_m]

    def error(arm: np.ndarray) -> np.ndarray:
        actual = adapter.kin.forward(np.r_[arm, start_q[6:]])
        return np.r_[
            10 * (actual[:3, 3] - target[:3, 3]), Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).as_rotvec()
        ]

    # One local solve seeded by the already-selected squeeze, with no fallback
    # candidates, contact evaluation, table gate, or intermediate path sampling.
    fit = least_squares(error, start_q[:6], bounds=adapter.kin.bounds, max_nfev=160, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    q = vector(np.r_[fit.x, start_q[6:]], 28, "lift")
    actual = adapter.kin.forward(q)
    errors = residual(actual, target)
    if (
        not fit.success
        or not all(np.isfinite(value) for value in errors.values())
        or errors["position_m"] > POSITION_TOLERANCE
        or errors["orientation_rad"] > ANGLE_TOLERANCE
    ):
        raise LiftTargetError(
            f"Selected candidate retained, but lift target could not be computed: {errors}; {fit.message}"
        )
    world_model = np.linalg.inv(adapter.model_T_world)
    return {
        "positions": q.tolist(),
        "height_m": float(height_m),
        "frame_id": "world",
        "axis": "+Z",
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
