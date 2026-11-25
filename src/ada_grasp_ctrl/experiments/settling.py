"""Windowed pose stability for interactive scenes with contact jitter."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SettlingPolicy:
    """Explicit simulation-time and pose tolerances, separate from grasp success."""

    timeout_s: float = 10.0
    window_s: float = 0.5
    position_span_m: float = 0.003
    orientation_span_deg: float = 2.0

    @classmethod
    def from_config(cls, value: dict) -> "SettlingPolicy":
        """Reject unknown, nonfinite and physically meaningless configuration values."""
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown simulation settling settings")
        policy = cls(**value)
        for field, number in vars(policy).items():
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number <= 0
            ):
                raise ValueError(f"settling.{field} must be positive and finite")
        if policy.window_s >= policy.timeout_s or policy.orientation_span_deg >= 180:
            raise ValueError("Settling requires window < timeout and orientation tolerance < 180 degrees")
        return policy


class PoseWindow:
    """Require bounded pose variation and tabletop support over a full time window.

    Linear and angular velocity are intentionally not combined. Pose samples from
    every physics step catch rocking even when the window endpoints coincide.
    The rotation-vector bounding-box diagonal conservatively bounds small-angle
    variation relative to the first pose in the window.
    """

    def __init__(self, policy: SettlingPolicy) -> None:
        self.policy = policy
        self.samples: deque[tuple[float, np.ndarray, np.ndarray, bool]] = deque()

    def add(self, now: float, pose: np.ndarray, supported: bool) -> None:
        """Collect world translation and WXYZ quaternion, retaining the time boundary."""
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or not math.isfinite(now):
            raise ValueError("Settling requires finite object pose and simulation time")
        if not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-6):
            raise ValueError("Settling requires a normalized object quaternion")
        if self.samples and now <= self.samples[-1][0]:
            raise ValueError("Settling simulation time must advance")
        self.samples.append((now, pose[:3].copy(), pose[3:].copy(), supported))
        while len(self.samples) > 2 and self.samples[1][0] <= now - self.policy.window_s:
            self.samples.popleft()

    def measure(self) -> dict:
        """Evaluate observed motion in metres/degrees without requiring zero velocity."""
        from scipy.spatial.transform import Rotation

        if not self.samples:
            raise ValueError("No settling samples")
        times, positions, quaternions, supports = zip(*self.samples)
        rotations = Rotation.from_quat(np.asarray(quaternions)[:, [1, 2, 3, 0]])
        relative = (rotations[0].inv() * rotations).as_rotvec()
        position_span = float(np.linalg.norm(np.ptp(positions, axis=0)))
        orientation_span = float(np.rad2deg(np.linalg.norm(np.ptp(relative, axis=0))))
        covered = times[-1] - times[0]
        supported = all(supports)
        stable = (
            covered + 1e-9 >= self.policy.window_s
            and supported
            and position_span <= self.policy.position_span_m
            and orientation_span <= self.policy.orientation_span_deg
        )
        return dict(
            stable=stable,
            observed_window_s=covered,
            position_span_m=position_span,
            orientation_span_deg=orientation_span,
            supported_throughout_window=supported,
        )
