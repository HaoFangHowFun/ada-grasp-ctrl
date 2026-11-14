"""Named, clock-explicit control contracts shared across Python/ROS processes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES

MODES = ("baseline", "linear", "mpc_passive", "mpc_active")
ACTIVE_STAGES = ("grasp", "squeeze")


def named_vector(names: tuple[str, ...], values: Any) -> tuple[float, ...]:
    """Require canonical right CRX + Sharpa order at every model/IPC boundary."""
    if tuple(names) != tuple(JOINT_NAMES):
        raise ValueError("Expected canonical named right 6 + 22 joint order")
    result = tuple(float(v) for v in values)
    if len(result) != 28 or not all(map(math.isfinite, result)):
        raise ValueError("Expected 28 finite joint values")
    return result


def fresh(timestamp: float, received: float, now: float, max_age: float) -> None:
    """Validate one monotonic clock domain; repeated reads cannot refresh source time."""
    if not all(map(math.isfinite, (timestamp, received, now, max_age))) or max_age <= 0:
        raise ValueError("Invalid clock or freshness threshold")
    if timestamp > received + 1e-9 or received > now + 1e-9 or now - timestamp > max_age + 1e-9:
        raise ValueError("Stale, future or reordered timestamps")


@dataclass(frozen=True)
class JointFeedback:
    """Positions are measured; never substitute the latest actuator command."""

    joint_names: tuple[str, ...]
    positions_rad: tuple[float, ...]
    source_timestamp: float
    receive_timestamp: float
    source: str
    velocities_rad_s: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "positions_rad", named_vector(self.joint_names, self.positions_rad))
        if self.velocities_rad_s is not None:
            object.__setattr__(self, "velocities_rad_s", named_vector(self.joint_names, self.velocities_rad_s))
        if not self.source or not all(map(math.isfinite, (self.source_timestamp, self.receive_timestamp))):
            raise ValueError("Feedback requires source and finite timestamps")


@dataclass(frozen=True)
class MPCProposal:
    """Absolute candidate; valid does not mean accepted or executed by a transport."""

    joint_names: tuple[str, ...]
    positions_rad: tuple[float, ...] | None
    valid: bool
    computed_at: float
    solve_duration_s: float
    contact_status: str
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        if tuple(self.joint_names) != tuple(JOINT_NAMES):
            raise ValueError("Proposal joint order differs")
        if self.positions_rad is not None:
            object.__setattr__(self, "positions_rad", named_vector(self.joint_names, self.positions_rad))
        if self.valid and self.positions_rad is None:
            raise ValueError("Valid proposal requires an absolute target")
        if not all(map(math.isfinite, (self.computed_at, self.solve_duration_s))) or self.solve_duration_s < 0:
            raise ValueError("Invalid proposal timing")


def controller_mode(config: dict, override: str | None = None) -> str:
    """Validate mode identity; only the launcher/receiver can authorize hardware motion."""
    mode = override or config.get("controller", {}).get("mode", "baseline")
    if mode not in MODES:
        raise ValueError(f"Unknown controller mode: {mode}")
    real_active = mode == "mpc_active" and config.get("environment") == "hardware"
    if config.get("controller", {}).get("hardware_enable", False) is not False and not real_active:
        raise ValueError("Hardware enablement is limited to explicit real mpc_active")
    if mode != "baseline" and config.get("environment", "mock") != "mock" and not real_active:
        raise ValueError("Phase 2 controller modes are offline/mock-only")
    if mode != "baseline" and config.get("side", "right") != "right":
        raise ValueError("Phase 2 is right-side only")
    return mode
