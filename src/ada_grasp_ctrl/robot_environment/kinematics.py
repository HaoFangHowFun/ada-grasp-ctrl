"""Bounded, single-arm Pinocchio IK with explicit frame and residual diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .model import side_slice

if TYPE_CHECKING:
    from .model import DualArmModel


def rigid(value: object) -> np.ndarray:
    """Reject nonfinite, reflected, scaled or malformed target transforms."""
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-10, rtol=0)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Expected a finite proper rigid transform")
    return matrix.copy()


def residual(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Measure translation in meters and shortest orientation error in radians."""
    return {
        "position_m": float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        "orientation_rad": float(Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude()),
    }


def solve_arm_ik(
    robot: DualArmModel,
    q_current: object,
    side: str,
    target_world_T_flange: object,
    *,
    frame: str = "hand_flange",
    max_nfev: int = 200,
) -> tuple[np.ndarray, dict]:
    """Return a bounded candidate, keeping the other 50 joints exactly as supplied.

    Args:
        robot: Model loaded from the canonical snapshot.
        q_current: Finite, in-bounds canonical 56-vector.
        side: Active left or right arm.
        target_world_T_flange: Proper world transform of the explicitly selected frame.
        frame: hand_flange (default), crx_flange, or tcp. No silent substitution.
        max_nfev: Positive least-squares evaluation budget.

    Returns:
        Candidate and diagnostic including solver success, residuals and acceptance.

    Raises:
        ValueError: On malformed input or malformed/nonfinite/out-of-bounds solver output.
    """
    q = robot.configuration(q_current)
    target = rigid(target_world_T_flange)
    if frame not in ("hand_flange", "crx_flange", "tcp"):
        raise ValueError("Arm IK frame must be hand_flange, crx_flange or tcp")
    name = robot.frame_name(side, frame)
    arm = side_slice(side, arm_only=True)
    if not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")

    def error(angles: np.ndarray) -> np.ndarray:
        candidate = q.copy()
        candidate[arm] = angles
        actual = robot.fk(candidate, side, frame)
        return np.r_[
            10 * (actual[:3, 3] - target[:3, 3]), Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).as_rotvec()
        ]

    fit = least_squares(
        error,
        q[arm],
        bounds=(robot.limits[arm, 0], robot.limits[arm, 1]),
        max_nfev=max_nfev,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if np.shape(fit.x) != (6,) or not np.isfinite(fit.x).all():
        raise ValueError("IK returned a malformed/nonfinite candidate")
    q[arm] = fit.x
    q = robot.configuration(q)
    errors = residual(robot.fk(q, side, frame), target)
    accepted = bool(fit.success and errors["position_m"] <= 1e-3 and errors["orientation_rad"] <= 1e-2)
    return q, {
        **errors,
        "accepted": accepted,
        "solver_success": bool(fit.success),
        "solver_status": int(fit.status),
        "message": str(fit.message),
        "nfev": int(fit.nfev),
        "side": side,
        "frame": name,
        "position_tolerance_m": 1e-3,
        "orientation_tolerance_rad": 1e-2,
        "collision_checked": False,
    }
