"""Immutable sensing results; deliberately separate from controller wrench contacts."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GeometryProvenance:
    """Explicit map identity and assumptions; force units are independent of map units."""

    profile: str
    source: str
    units: str
    meters_per_unit: float
    transform_s_to_e: tuple[tuple[float, ...], ...]
    uv_to_col_row: tuple[tuple[float, ...], ...]
    verification: str
    evidence: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Freeze nested sequences, including caller-supplied matrices and evidence."""
        for name in ("transform_s_to_e", "uv_to_col_row"):
            object.__setattr__(self, name, tuple(tuple(float(x) for x in row) for row in getattr(self, name)))
        evidence = tuple(tuple(pair) for pair in self.evidence)
        if any(len(pair) != 2 or any(not isinstance(x, str) for x in pair) for pair in evidence):
            raise ValueError("Geometry evidence must contain string key/value pairs")
        object.__setattr__(self, "evidence", evidence)


@dataclass(frozen=True)
class TactileContactObservation:
    """One first-contact sample, with vectors in one explicit frame (meters/N).

    Timestamp is local subscriber monotonic receipt time, not acquisition time.
    Force is the fingertip resultant, provisionally object-on-finger; it is not
    allocated among contacts. No torque, object identity or controller readiness
    is implied. Tuple storage prevents consumers from mutating shared samples.
    """

    finger: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    force: tuple[float, float, float]
    frame_id: str
    timestamp: float
    receipt_ros: float
    subscriber_ros: float
    session_id: str
    sequence: int
    sdk_frame_id: int | None
    sdk_frame_id_raw: str
    sdk_timestamp: float | None
    sdk_timestamp_repr: str
    sdk_timestamp_unit: str
    sdk_timestamp_epoch: str
    source_frame: str
    contact_uv: tuple[float, float]
    contact_count: int
    contact_row: tuple[float, ...]
    contact_shape: tuple[int, ...]
    raw_f6: tuple[float, ...]
    force_indices: tuple[int, int, int]
    geometry: GeometryProvenance
    pixel_row: int
    pixel_col: int
    joint_received: float | None = None
    joint_receipt_ros: float | None = None
    warnings: tuple[str, ...] = ()
    force_convention: str = "object_on_finger"
    force_convention_quality: str = "provisional; operator-observed direction and reasonable N units"
    timestamp_clock: str = "subscriber host monotonic"

    def __post_init__(self) -> None:
        """Own all sequences and reject invalid physical vectors at construction."""
        for name, size in (("position", 3), ("normal", 3), ("force", 3), ("contact_uv", 2), ("raw_f6", 6)):
            value = tuple(float(x) for x in getattr(self, name))
            if len(value) != size or not all(map(math.isfinite, value)):
                raise ValueError(f"{name} must contain {size} finite values")
            object.__setattr__(self, name, value)
        for name in ("contact_row", "contact_shape", "force_indices", "warnings"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not math.isclose(sum(x * x for x in self.normal), 1.0, abs_tol=1e-6):
            raise ValueError("Expected a normalized surface normal")
        if not self.frame_id or not math.isfinite(self.timestamp) or self.contact_count < 1:
            raise ValueError("Expected a frame, finite receipt time and at least one contact")
        if not isinstance(self.geometry, GeometryProvenance):
            raise ValueError("Expected immutable geometry provenance")
        if not all(map(math.isfinite, self.contact_row)) or self.contact_row[:2] != self.contact_uv:
            raise ValueError("Contact UV must match the finite original row")
