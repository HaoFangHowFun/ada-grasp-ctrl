"""Shared read-only tactile conversion; no ROS, Viser or SDK device ownership."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any, Callable

import numpy as np

from ada_grasp_ctrl.contact.tactile_observation import GeometryProvenance, TactileContactObservation
from ada_grasp_ctrl.experiments.right_ros_contract import HAND_NAMES
from .data import FINGERS, Frame, force_to_world, frame_status, select_force
from .geometry import GeometryCatalog, transform_contact
from .kinematics import joint_configuration


@dataclass(frozen=True)
class FingerObservation:
    """Availability and diagnostics accompany every optional contact."""

    status: str
    detail: str = ""
    observation: TactileContactObservation | None = None
    local: TactileContactObservation | None = None
    frame: Frame | None = None
    transport_status: str = "NO DATA"
    force_local: tuple[float, ...] | None = None
    force_base: tuple[float, ...] | None = None
    arrival_age: float = 0.0
    receipt_age: float | None = None
    pose_skew: float | None = None
    source_position: tuple[float, ...] | None = None
    source_normal: tuple[float, ...] | None = None


@dataclass(frozen=True)
class TactileSnapshot:
    """One coherent evaluation; get_contacts/get_statuses are convenience views."""

    evaluated_at: float
    fingers: tuple[tuple[str, FingerObservation], ...]
    configuration: tuple[float, ...] | None
    pose_status: str


class TactileAdapter:
    """Convert existing reader snapshots to per-finger sensing observations.

    Args:
        hand: Existing official yourdfpy model, owned by this adapter's update
            thread. Rendering may share it but must not independently change FK.
        geometry: Existing explicit per-finger geometry catalog.
        stale_after: Maximum monotonic arrival age in seconds.
        pose_skew: Maximum tactile/joint receipt-time difference in seconds.
        receipt_clock_synchronized: Explicit assertion enabling ROS receipt ages.
        clock: Same-host monotonic clock; injectable for deterministic tests.

    update replaces the complete latest-per-finger snapshot, rather than merging
    old contacts. Input must come from normalize_frame/LatestFrames (not arbitrary
    ROS messages). Arrival order is retained, including SDK ID regressions; source
    identity is never manufactured. get_snapshot re-evaluates age without requiring
    callbacks, so a stopped reader cannot leave a valid contact indefinitely.
    """

    def __init__(
        self,
        hand: Any,
        geometry: GeometryCatalog,
        *,
        stale_after: float = 0.5,
        pose_skew: float = 0.15,
        receipt_clock_synchronized: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if any(not math.isfinite(x) or x <= 0 for x in (stale_after, pose_skew)):
            raise ValueError("Freshness and skew limits must be finite and positive")
        if hand.base_link != "right_hand_flange":
            raise ValueError("Expected right_hand_flange base; laboratory world is unsupported")
        if set(hand.actuated_joint_names) != set(HAND_NAMES):
            raise ValueError("Expected all 22 official right-hand joints")
        for finger in FINGERS:
            hand.get_transform(f"right_{finger}_elastomer")
        self.hand, self.geometry, self.clock = hand, geometry, clock
        self.stale_after, self.pose_skew = stale_after, pose_skew
        self.synchronized = receipt_clock_synchronized
        self.lock = threading.RLock()
        self._frames: dict[str, Any] = {}
        self._joints: dict[str, Any] = {}
        self._endpoints: dict[str, Any] | None = None
        self._errors: dict[str, str] = {}
        self._connected = False
        self._indices = (0, 1, 2)
        self._allow_geometry = False

    def update(
        self,
        tactile_frames: dict[str, Frame | dict],
        joint_state: dict[str, Any] | None,
        *,
        connected: bool = True,
        endpoints: dict[str, Any] | None = None,
        errors: dict[str, str] | None = None,
        force_indices: tuple[int, int, int] = (0, 1, 2),
        allow_unverified_geometry: bool = False,
    ) -> TactileSnapshot:
        """Own a complete normalized snapshot and evaluate all five fingers.

        Missing keys clear previous readings. Optional endpoints enforce the
        existing one-publisher/type gate; offline normalized fixtures may omit it.
        Display-only sign flipping is intentionally not an adapter option.

        Raises:
            ValueError: If a snapshot contains unsupported finger keys.
        """
        with self.lock:
            unknown = set(tactile_frames) - set(FINGERS)
            if unknown:
                self._frames, self._connected = {}, False
                raise ValueError(f"Unsupported tactile fingers: {sorted(unknown)}")
            self._frames = deepcopy(tactile_frames)
            self._joints = deepcopy(joint_state or {})
            self._endpoints, self._errors = deepcopy(endpoints), dict(errors or {})
            self._connected = connected
            self._indices, self._allow_geometry = tuple(force_indices), allow_unverified_geometry
            return self.get_snapshot()

    def get_contacts(self) -> dict[str, TactileContactObservation | None]:
        """Return fresh hand-base contacts; None always has a separate status."""
        return {f: r.observation for f, r in self.get_snapshot().fingers}

    def get_statuses(self) -> dict[str, str]:
        """Return distinct missing/empty/invalid/stale/pose/geometry status codes."""
        return {f: r.status for f, r in self.get_snapshot().fingers}

    def get_snapshot(self) -> TactileSnapshot:
        """Evaluate age at read time and return contact/status pairs atomically."""
        with self.lock:
            now = self.clock()
            try:
                q, pose_status = joint_configuration(self._joints, self.hand, now, self.stale_after, self.synchronized)
                if q is not None and self._connected and not np.array_equal(q, self.hand.cfg):
                    self.hand.update_cfg(q)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                q, pose_status = None, f"INVALID POSE: {exc}"
            results = tuple((f, self._evaluate(f, now, q, pose_status)) for f in FINGERS)
            return TactileSnapshot(now, results, None if q is None else tuple(q), pose_status)

    def _evaluate(self, finger: str, now: float, q: Any, pose_status: str) -> FingerObservation:
        payload = self._frames.get(finger)
        if payload is None:
            error = self._errors.get(finger, "")
            return FingerObservation("INVALID" if error else "NO_DATA", error)
        try:
            frame = Frame(**payload) if isinstance(payload, dict) else payload
            if not isinstance(frame, Frame) or frame.finger != finger:
                raise ValueError("Normalized frame finger does not match snapshot key")
            frame = replace(
                frame,
                **{n: tuple(getattr(frame, n)) for n in ("raw", "shape", "warnings", "contact_shape", "contact_row")},
            )
            if not all(math.isfinite(x) for x in (frame.received, frame.receipt_ros, frame.subscriber_ros)):
                raise ValueError("Nonfinite frame receipt clock")
            status, age, receipt_age = frame_status(frame, now, self.stale_after, self.synchronized)
            if frame.received > now + 0.05:
                status = "CLOCK_ERROR"
        except (TypeError, ValueError, AttributeError, OverflowError) as exc:
            return FingerObservation("INVALID", str(exc), transport_status="INVALID")
        if not self._connected:
            status = "DISCONNECTED"
        elif self._endpoints is not None:
            endpoints = self._endpoints.get(finger, [])
            if len(endpoints) != 1 or endpoints[0].get("type") != "dual_sharpa_wave/msg/TactileFrame":
                status = "NO / AMBIGUOUS PUBLISHER"
        result = FingerObservation(
            status, frame=frame, transport_status=status, arrival_age=age, receipt_age=receipt_age
        )
        try:
            if frame.status == "VALID" and (
                not 1 <= len(frame.shape) <= 8
                or any(type(d) is not int or d < 1 for d in frame.shape)
                or math.prod(frame.shape) != 6
            ):
                raise ValueError("Invalid normalized F6 shape")
            force = select_force(frame.raw, self._indices) if frame.status == "VALID" else None
        except (ValueError, TypeError, OverflowError) as exc:
            return replace(result, status="INVALID_FORCE", detail=str(exc))
        result = replace(result, force_local=force)
        if status != "VALID":
            code = status if status in ("STALE", "CLOCK_ERROR", "DISCONNECTED") else "F6/TRANSPORT " + status
            return replace(result, status=code, detail=frame.error)
        if force is None:
            return replace(result, status="INVALID_FORCE")
        joints = self._joints
        skew = (
            None
            if q is None
            else (
                abs(frame.receipt_ros - joints.get("receipt_ros", 0))
                if self.synchronized
                else abs(frame.received - joints.get("received", 0))
            )
        )
        pose_ok = skew is not None and math.isfinite(skew) and skew <= self.pose_skew
        pose = self.hand.get_transform(f"right_{finger}_elastomer")
        result = replace(
            result, pose_skew=skew, force_base=tuple(force_to_world(pose[:3, :3], force)) if pose_ok else None
        )
        if frame.contact_status != "VALID":
            code = frame.contact_status if frame.contact_status in ("MISSING", "EMPTY", "INVALID") else "INVALID"
            return replace(result, status=code, detail=frame.contact_error)
        try:
            shape = frame.contact_shape
            if (
                not 1 <= len(shape) <= 8
                or any(type(d) is not int or d < 1 for d in shape)
                or shape[-1] != len(frame.contact_row)
                or shape[-1] < 2
                or math.prod(shape) > 262144
                or math.prod(shape[:-1]) != frame.contact_count
                or not all(map(math.isfinite, frame.contact_row))
            ):
                raise ValueError("Invalid normalized first-contact row/count/shape")
        except (TypeError, ValueError) as exc:
            return replace(result, status="INVALID", detail=str(exc))
        profile = self.geometry.maps.get(finger)
        if profile is None:
            return replace(result, status="GEOMETRY_UNAVAILABLE", detail=self.geometry.errors.get(finger, "no profile"))
        if not profile.verified and not self._allow_geometry:
            return replace(result, status="GEOMETRY_UNVERIFIED", detail="Diagnostic geometry opt-in required")
        try:
            contact = self.geometry.lookup(finger, frame.contact_row[:2], self._allow_geometry)
            result = replace(result, source_position=tuple(contact.p_S), source_normal=tuple(contact.n_S))
            provenance = GeometryProvenance(
                self.geometry.profile_names.get(finger, profile.source),
                profile.source,
                profile.units,
                profile.meters_per_unit,
                tuple(map(tuple, profile.T_E_S)),
                tuple(map(tuple, profile.uv_to_col_row)),
                contact.verification,
                tuple(sorted(profile.evidence.items())),
            )
            local = TactileContactObservation(
                finger=finger,
                position=contact.p_E,
                normal=contact.n_E,
                force=force,
                frame_id=f"right_{finger}_elastomer",
                timestamp=frame.received,
                receipt_ros=frame.receipt_ros,
                subscriber_ros=frame.subscriber_ros,
                session_id=frame.session_id,
                sequence=frame.sequence,
                sdk_frame_id=frame.sdk_frame_id,
                sdk_frame_id_raw=frame.sdk_frame_id_raw,
                sdk_timestamp=frame.sdk_timestamp,
                sdk_timestamp_repr=frame.sdk_timestamp_repr,
                sdk_timestamp_unit=frame.sdk_timestamp_unit,
                sdk_timestamp_epoch=frame.sdk_timestamp_epoch,
                source_frame=frame.source_frame,
                contact_uv=frame.contact_row[:2],
                contact_count=frame.contact_count,
                contact_row=frame.contact_row,
                contact_shape=frame.contact_shape,
                raw_f6=frame.raw,
                force_indices=self._indices,
                geometry=provenance,
                pixel_row=contact.row,
                pixel_col=contact.col,
                warnings=frame.warnings,
                force_convention_quality=(
                    "provisional; operator-observed direction and reasonable N units"
                    if self._indices == (0, 1, 2)
                    else "provisional; diagnostic channel selection not operator-validated"
                ),
            )
            if not pose_ok:
                code = (
                    "POSE_SKEWED"
                    if q is not None
                    else ("POSE_MISSING" if not joints else "POSE_STALE" if "STALE" in pose_status else "POSE_INVALID")
                )
                return replace(result, status=code, detail=pose_status, local=local)
            position, normal, force_base = transform_contact(pose, local.position, local.normal, local.force)
            observation = replace(
                local,
                position=position,
                normal=normal,
                force=force_base,
                frame_id="right_hand_flange",
                joint_received=joints["received"],
                joint_receipt_ros=joints["receipt_ros"],
            )
            return replace(result, status="VALID", observation=observation, local=local)
        except (ValueError, TypeError, KeyError) as exc:
            return replace(result, status="GEOMETRY_INVALID", detail=str(exc))
