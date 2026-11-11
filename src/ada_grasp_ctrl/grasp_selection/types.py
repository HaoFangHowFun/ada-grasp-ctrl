"""Validated, ROS-independent data contracts for static candidate selection."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ARM_NAMES = tuple(f"right_J{i}" for i in range(1, 7))
HAND_NAMES = tuple(
    "right_" + name
    for name in (
        "thumb_CMC_FE",
        "thumb_CMC_AA",
        "thumb_MCP_FE",
        "thumb_MCP_AA",
        "thumb_IP",
        "index_MCP_FE",
        "index_MCP_AA",
        "index_PIP",
        "index_DIP",
        "middle_MCP_FE",
        "middle_MCP_AA",
        "middle_PIP",
        "middle_DIP",
        "ring_MCP_FE",
        "ring_MCP_AA",
        "ring_PIP",
        "ring_DIP",
        "pinky_CMC",
        "pinky_MCP_FE",
        "pinky_MCP_AA",
        "pinky_PIP",
        "pinky_DIP",
    )
)
STAGES = ("pregrasp", "grasp", "squeeze")


def vector(value: Any, size: int, name: str) -> np.ndarray:
    """Copy a finite numeric vector and prevent accidental in-place changes."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (size,) or not np.isfinite(raw).all():
        raise ValueError(f"{name}: expected {size} finite numbers")
    result = raw.astype(float, copy=True)
    result.setflags(write=False)
    return result


def rigid(value: Any, name: str) -> np.ndarray:
    """Validate a proper rigid transform without repairing rotation or translation."""
    t = np.array(value, dtype=float, copy=True)
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"{name}: expected a finite proper rigid transform")
    t.setflags(write=False)
    return t


def jsonable(value: Any) -> Any:
    """Convert diagnostics to JSON values; callers reject NaN during serialization."""
    if is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class StageTarget:
    """Original world-frame right flange pose and canonical hand joints."""

    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    joint_positions_rad: np.ndarray
    frame_id: str = "world"
    pose_frame: str = "right_hand_flange"

    def __post_init__(self) -> None:
        for name, size in (("position_m", 3), ("quaternion_wxyz", 4), ("joint_positions_rad", 22)):
            object.__setattr__(self, name, vector(getattr(self, name), size, name))
        if abs(np.linalg.norm(self.quaternion_wxyz) - 1) > 1e-4:
            raise ValueError("quaternion_wxyz: expected unit WXYZ quaternion")
        if self.frame_id != "world" or self.pose_frame != "right_hand_flange":
            raise ValueError("Expected world-frame right_hand_flange, not palm/TCP/camera")

    def transform(self) -> np.ndarray:
        """Return world-from-flange, without applying camera extrinsics again."""
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat(self.quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
        t[:3, 3] = self.position_m
        return t


@dataclass(frozen=True)
class GraspCandidate:
    """One attributable grasp; phase data and source payload remain available."""

    candidate_id: str
    stages: Mapping[str, StageTarget]
    source_id: str
    calibration_id: str
    original: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        from types import MappingProxyType

        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or not self.source_id
            or not self.calibration_id
        ):
            raise ValueError("Missing stable candidate/source/calibration ID")
        if set(self.stages) != set(STAGES) or not all(isinstance(s, StageTarget) for s in self.stages.values()):
            raise ValueError("All original pregrasp/grasp/squeeze targets are required")
        object.__setattr__(self, "stages", MappingProxyType(dict(self.stages)))


@dataclass(frozen=True)
class RobotState:
    """Named right-arm state, bound to one clock, model and measured source."""

    positions_rad: np.ndarray
    stamp_s: float
    received_s: float
    model_id: str
    source_topic: str
    environment: str
    frame_id: str = "world"
    clock: str = "ros_system"
    joint_names: tuple[str, ...] = ARM_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions_rad", vector(self.positions_rad, 6, "positions_rad"))
        if tuple(self.joint_names) != ARM_NAMES or self.frame_id != "world":
            raise ValueError("State requires canonical right joints and world model frame")
        if not self.model_id or not self.source_topic or self.environment not in ("mock", "hardware", "fixture"):
            raise ValueError("Missing or invalid state provenance")
        if self.clock not in ("ros_system", "ros_sim"):
            raise ValueError("Unknown state clock")
        if (
            any(isinstance(v, (bool, np.bool_)) for v in (self.stamp_s, self.received_s))
            or not np.isfinite([self.stamp_s, self.received_s]).all()
            or min(self.stamp_s, self.received_s) <= 0
        ):
            raise ValueError("Invalid state timestamps")

    def check_fresh(self, now_s: float, max_age_s: float) -> None:
        """Check both transport age and evaluation age in the same declared clock."""
        if not np.isfinite([now_s, max_age_s]).all() or max_age_s <= 0:
            raise ValueError("Invalid freshness configuration")
        if not 0 <= self.received_s - self.stamp_s <= max_age_s or not 0 <= now_s - self.stamp_s <= max_age_s:
            raise ValueError("Missing, stale or future joint state")


@dataclass(frozen=True)
class SelectionContext:
    """Explicit transforms and acceptance facts for one evaluation snapshot."""

    robot_state: RobotState
    world_T_right_base: np.ndarray
    transform_source: str
    calibration_id: str
    now_s: float
    max_age_s: float = 0.5
    replay: bool = True
    acceptance: Mapping[str, bool] = field(default_factory=dict)
    scene_complete: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "world_T_right_base", rigid(self.world_T_right_base, "world_T_right_base"))
        if not self.transform_source or not self.calibration_id:
            raise ValueError("Missing transform/calibration provenance")
        self.robot_state.check_fresh(self.now_s, self.max_age_s)


@dataclass
class IKOption:
    """One grasp branch with independent endpoint checks for all required stages."""

    solution_index: int
    q_rad: np.ndarray
    stages: dict
    reasons: list[str] = field(default_factory=list)
    score: float | None = None
    score_details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.q_rad = vector(self.q_rad, 6, "IK q_rad")


@dataclass
class CandidateEvaluation:
    """Per-candidate rejections, cached IK options and source references."""

    candidate_id: str
    source_id: str
    options: list[IKOption] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    best_solution_index: int | None = None


@dataclass
class SelectionResult:
    """Static selection only; motion readiness always requires an external path check."""

    selected_candidate_id: str | None
    selected_q_rad: np.ndarray | None
    cost: float | None
    selected: dict | None
    evaluations: list[CandidateEvaluation]
    configuration: dict
    status: str
    readiness_reasons: list[str]
    schema_version: int = 1
    execution_ready: bool = False
    readiness: str = "needs_validation"
