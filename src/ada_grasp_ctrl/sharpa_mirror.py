"""Stored, ROS-free Sharpa joint and flange/wrist pose mirroring."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml


class SharpaHandMirror:
    """Apply a versioned mapping; joint arrays use the stored names and radians."""

    def __init__(self, config: str | Path | None = None) -> None:
        path = Path(config) if config else files("ada_grasp_ctrl").joinpath("config/hand/sharpa_mirror.yaml")
        self.config = yaml.safe_load(path.read_text())
        if self.config["schema_version"] != 1:
            raise ValueError("Unsupported mirror configuration version")
        self.right_joint_names = tuple(self.config["right_joint_names"])
        self.left_joint_names = tuple(self.config["left_joint_names"])
        self.permutation = np.asarray(self.config["permutation"], dtype=int)
        self.sign = np.asarray(self.config["sign"], dtype=float)
        self.offset = np.asarray(self.config["offset"], dtype=float)
        self.local_reflection = np.asarray(self.config["local_reflection"], dtype=float)
        if sorted(self.permutation.tolist()) != list(range(22)) or not np.isin(self.sign, [-1, 1]).all():
            raise ValueError("Invalid joint permutation/signs")
        self._orthogonal(self.local_reflection, -1)

    @staticmethod
    def _orthogonal(matrix: np.ndarray, determinant: int) -> None:
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(matrix), determinant, atol=1e-8, rtol=0)
        ):
            raise ValueError("Invalid rotation/reflection matrix")

    def right_to_left(self, q_right: np.ndarray) -> np.ndarray:
        """Map one 22-vector in radians, rejecting nonfinite and out-of-limit values."""
        q = np.asarray(q_right, dtype=float)
        if q.shape != (22,) or not np.isfinite(q).all():
            raise ValueError("Expected a finite (22,) joint vector in radians")
        result = q[self.permutation] * self.sign + self.offset
        for label, value in (("right", q), ("left", result)):
            limits = np.asarray(self.config[f"{label}_limits"])
            if np.any(value < limits[:, 0] - 1e-8) or np.any(value > limits[:, 1] + 1e-8):
                raise ValueError(f"{label} joint limit violation")
        return result

    def mirror_wrist_pose(
        self, right_wrist_pose: np.ndarray, normal: np.ndarray, point: np.ndarray | None = None
    ) -> np.ndarray:
        """Mirror a 4x4 world-from-flange (or wrist) pose across a world plane.

        Args:
            right_wrist_pose: Rigid homogeneous pose, meters; no quaternion ambiguity.
            normal: Nonzero world-plane normal, normalized internally.
            point: World point on the plane, default world origin.

        Returns:
            Proper left pose. The stored local relationship includes the fixed
            flange/wrist origin correspondence; arbitrary TCP poses are not accepted.

        Raises:
            ValueError: For malformed poses or reflection planes.
        """
        pose = np.asarray(right_wrist_pose, dtype=float)
        n = np.asarray(normal, dtype=float)
        c = np.zeros(3) if point is None else np.asarray(point, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("Expected a finite homogeneous pose")
        self._orthogonal(pose[:3, :3], 1)
        if n.shape != (3,) or c.shape != (3,) or not np.isfinite(n).all() or not np.isfinite(c).all():
            raise ValueError("Expected finite plane vectors of shape (3,)")
        if np.linalg.norm(n) < 1e-12:
            raise ValueError("Plane normal must be nonzero")
        n = n / np.linalg.norm(n)
        reflection = np.eye(3) - 2 * np.outer(n, n)
        result = np.eye(4)
        result[:3, :3] = reflection @ pose[:3, :3] @ self.local_reflection.T
        result[:3, 3] = c + reflection @ (pose[:3, 3] - c)
        self._orthogonal(result[:3, :3], 1)
        return result
