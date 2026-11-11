"""Joint displacement proxy, not measured force, torque, effort or energy."""

import numpy as np
from ..types import IKOption, SelectionContext, vector


class JointDisplacement:
    """Weighted squared difference in URDF radians, without angle wrapping."""

    name = "weighted_squared_joint_displacement"

    def __init__(self, weights: object = (1, 1, 1, 1, 1, 1)) -> None:
        self.weights = vector(weights, 6, "diagonal weights")
        if (self.weights < 0).any():
            raise ValueError("Joint weights must be nonnegative")

    def score(self, option: IKOption, context: SelectionContext) -> tuple[float, dict]:
        """Return the exact quadratic cost and unwrapped raw delta in radians."""
        delta = option.q_rad - context.robot_state.positions_rad
        cost = float(np.dot(delta * self.weights, delta))
        return cost, {
            "delta_rad": delta,
            "unweighted_norm_rad": float(np.linalg.norm(delta)),
            "weighted_squared_cost": cost,
            "weights": self.weights,
        }
