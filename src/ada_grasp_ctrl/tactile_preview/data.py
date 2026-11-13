"""Atomic F6/first-contact snapshots shared by ROS and viewer Python."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Any

FINGERS = ("pinky", "ring", "middle", "index", "thumb")
TOPICS = {finger: f"/sharpa/right_hand/tactile/{finger}/frame" for finger in FINGERS}
BLOCK_STATUS = {0: "MISSING", 1: "VALID", 2: "EMPTY", 3: "INVALID"}


def stamp_seconds(stamp: Any) -> float:
    """Convert a ROS stamp without interpreting an SDK timestamp."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass(frozen=True)
class Frame:
    """Owned F6 and first contact from one message; deformation arrays are never copied."""

    finger: str
    status: str
    raw: tuple[float, ...]
    shape: tuple[int, ...]
    error: str
    received: float
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
    warnings: tuple[str, ...]
    contact_status: str = "MISSING"
    contact_shape: tuple[int, ...] = ()
    contact_row: tuple[float, ...] = ()
    contact_count: int = 0
    contact_error: str = ""


def first_contact(block: Any) -> dict[str, Any]:
    """Validate the full contact block and retain row 0 with every raw column.

    The final dimension is columns; preceding dimensions enumerate C-order rows.
    A one-dimensional K-vector denotes one row. Shape is always retained.
    """
    result = dict(contact_status="INVALID", contact_shape=(), contact_row=(), contact_count=0, contact_error="")
    try:
        shape = tuple(block.shape)
        result["contact_shape"] = shape
        status = BLOCK_STATUS.get(block.status, "INVALID")
        result["contact_status"], result["contact_error"] = status, block.error
        if status in ("MISSING", "INVALID"):
            if status == "MISSING" and len(block.raw_values):
                raise ValueError("MISSING contact block contains values")
            return result
        if not 1 <= len(shape) <= 8 or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("Invalid contact shape dimensions")
        size = math.prod(shape)
        if size > 262144 or size != len(block.raw_values):
            raise ValueError("Contact shape/flat length mismatch")
        if size == 0:
            result.update(contact_status="EMPTY", contact_count=0)
            return result
        if shape[-1] < 2:
            raise ValueError("Contact rows require at least two columns")
        count = math.prod(shape[:-1]) if len(shape) > 1 else 1
        if status == "EMPTY" and count != 0:
            raise ValueError("EMPTY contact block has nonzero row count")
        if not all(math.isfinite(v) for v in block.raw_values):
            raise ValueError("Contact payload contains nonfinite values")
        result.update(contact_status="VALID" if count else "EMPTY", contact_count=count)
        if count:
            result["contact_row"] = tuple(float(v) for v in block.raw_values[: shape[-1]])
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        result.update(contact_status="INVALID", contact_row=(), contact_count=0, contact_error=str(exc))
    return result


def normalize_frame(message: Any, expected: str, now: float, ros_now: float) -> Frame:
    """Validate identity and F6 against the installed Phase 1A wire contract.

    Raises:
        ValueError: If schema or topic identity is inconsistent.
    """
    if (
        expected not in FINGERS
        or message.schema_version != 1
        or message.source != "sharpa_sdk"
        or message.side != "right"
        or message.finger != expected
        or message.channel != FINGERS.index(expected)
    ):
        raise ValueError("Wrong schema/source/side/finger/channel for subscribed topic")
    block = message.f6
    status = BLOCK_STATUS.get(block.status, "INVALID")
    shape = tuple(block.shape)
    raw = tuple(float(v) for v in block.raw_values)
    error = block.error
    valid_shape = (
        0 < len(shape) <= 8
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in shape)
        and math.prod(shape) == 6
    )
    if status == "VALID" and (not valid_shape or len(raw) != 6 or not all(math.isfinite(v) for v in raw)):
        status, error = "INVALID", "F6 requires a consistent shape and exactly six finite values"
    if status != "VALID":
        raw = ()
    sdk_timestamp = float(message.sdk_timestamp_raw) if message.sdk_timestamp_valid else None
    if sdk_timestamp is not None and not math.isfinite(sdk_timestamp):
        sdk_timestamp = None
    return Frame(
        finger=expected,
        status=status,
        raw=raw,
        shape=shape,
        error=error,
        received=now,
        receipt_ros=stamp_seconds(message.header.stamp),
        subscriber_ros=ros_now,
        session_id=message.session_id,
        sequence=message.receipt_sequence,
        sdk_frame_id=message.sdk_frame_id if message.sdk_frame_id_valid else None,
        sdk_frame_id_raw=message.sdk_frame_id_raw,
        sdk_timestamp=sdk_timestamp,
        sdk_timestamp_repr=message.sdk_timestamp_repr,
        sdk_timestamp_unit=message.sdk_timestamp_unit,
        sdk_timestamp_epoch=message.sdk_timestamp_epoch,
        source_frame=message.header.frame_id,
        warnings=tuple(message.identity_warnings) + tuple(message.validation_errors),
        **first_contact(message.contact_points),
    )


def frame_status(frame: Frame, now: float, stale_after: float, synchronized: bool) -> tuple[str, float, float | None]:
    """Age data using monotonic arrival time plus optional verified ROS-clock delay."""
    arrival_age = max(0.0, now - frame.received)
    receipt_age = frame.subscriber_ros - frame.receipt_ros + arrival_age if synchronized else None
    if receipt_age is not None and (frame.receipt_ros <= 0 or receipt_age < -0.05):
        return "CLOCK_ERROR", arrival_age, receipt_age
    if arrival_age > stale_after or (receipt_age is not None and receipt_age > stale_after):
        return "STALE", arrival_age, receipt_age
    return frame.status, arrival_age, receipt_age


def select_force(raw: tuple[float, ...], indices: tuple[int, int, int], flip: bool = False) -> tuple[float, ...]:
    """Select explicitly configured channels, preserving sign unless diagnostic flip is enabled."""
    if len(indices) != 3 or len(set(indices)) != 3 or any(type(i) is not int or i not in range(6) for i in indices):
        raise ValueError("Choose three distinct F6 channel indices from 0 to 5")
    if len(raw) != 6 or not all(math.isfinite(v) for v in raw):
        raise ValueError("Expected six finite raw F6 values")
    return tuple((-1 if flip else 1) * raw[i] for i in indices)


def force_to_world(rotation: Any, force: Any) -> Any:
    """Rotate a force with R_WE only; translations never act on directions.

    Raises:
        ValueError: If the inputs are nonfinite or the matrix is not a proper rotation.
    """
    import numpy as np

    rotation, force = np.asarray(rotation, dtype=float), np.asarray(force, dtype=float)
    if (
        rotation.shape != (3, 3)
        or force.shape != (3,)
        or not np.isfinite(rotation).all()
        or not np.isfinite(force).all()
    ):
        raise ValueError("Expected finite 3x3 rotation and 3-vector")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("Expected a proper rotation")
    return rotation @ force


class LatestFrames:
    """Bounded, thread-safe latest-per-finger cache with measured arrival rates."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames: dict[str, Frame] = {}
        self.times: dict[str, deque] = {f: deque(maxlen=120) for f in FINGERS}
        self.errors: dict[str, str] = {}
        self.rejections = {f: 0 for f in FINGERS}

    def receive(self, message: Any, finger: str, ros_now: float) -> None:
        """Copy F6 and contact row 0 together, invalidating malformed topic identity."""
        now = time.monotonic()
        try:
            frame = normalize_frame(message, finger, now, ros_now)
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            with self.lock:
                self.frames.pop(finger, None)
                self.errors[finger] = str(exc)
                self.rejections[finger] += 1
            return
        with self.lock:
            self.frames[finger] = frame
            self.times[finger].append(now)
            self.errors.pop(finger, None)

    def snapshot(self) -> dict[str, Any]:
        """Return owned JSON-compatible records; never hold the lock during I/O."""
        with self.lock:
            frames, times = dict(self.frames), {f: tuple(t) for f, t in self.times.items()}
            errors, rejections = dict(self.errors), dict(self.rejections)
        now = time.monotonic()
        recent = {f: [t for t in ts if now - t <= 2] for f, ts in times.items()}
        rates = {
            f: (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0 for f, ts in recent.items()
        }
        return dict(frames={f: asdict(v) for f, v in frames.items()}, rates=rates, errors=errors, rejections=rejections)
