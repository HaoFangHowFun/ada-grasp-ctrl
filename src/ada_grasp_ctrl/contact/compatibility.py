"""Audited point-contact compatibility; original tactile observations stay untouched."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import HAND_NAMES
from ada_grasp_ctrl.mpc.contracts import JointFeedback, fresh
from ada_grasp_ctrl.optimization import friction_cone_slack
from ada_grasp_ctrl.tactile_preview.adapter import TactileSnapshot
from ada_grasp_ctrl.tactile_preview.data import FINGERS
from .types import ContactObservation
from .validation import validate_observation


class ContactKinematics(Protocol):
    """Existing full-arm model, using canonical measured joints in meters."""

    def compute_jaco_a(self, q: np.ndarray) -> None: ...
    def get_frame_pose(self, frame_name: str) -> np.ndarray: ...
    def get_frame_jaco(self, frame_name: str, type: str = "body") -> np.ndarray: ...


@dataclass(frozen=True)
class ContactEstimate:
    """Valid empty, eligible point contacts and insufficient data are distinct."""

    status: str
    observation: ContactObservation | None
    reasons: tuple[str, ...]
    provenance: dict

    def require_observation(self) -> ContactObservation:
        if self.observation is None:
            raise ValueError("Contact data unavailable: " + "; ".join(self.reasons))
        return self.observation


def contact_frame(normal: np.ndarray) -> np.ndarray:
    """Choose the MuJoCo default point-contact tangent gauge, with +X hand→object.

    Tangent axes are a declared coordinate choice, not a tactile measurement.
    MuJoCo contacts with explicit anisotropic tangent directions require a supplied
    matching gauge; this adapter supports the native isotropic point-contact model.
    """
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or not np.isclose(n @ n, 1, atol=1e-7, rtol=0):
        raise ValueError("Invalid unit contact normal")
    tangent = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.5 else np.array([0.0, 0.0, 1.0])
    tangent -= n * (n @ tangent)
    tangent /= np.linalg.norm(tangent)
    return np.column_stack((n, tangent, np.cross(n, tangent)))


def rigid_transform(value: Any) -> np.ndarray:
    """Reject malformed FK/mount transforms without repairing calibration."""
    pose = np.asarray(value, dtype=float)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7, rtol=0)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-7, rtol=0)
    ):
        raise ValueError("Invalid rigid transform")
    return pose


def point_jacobian(kinematics: ContactKinematics, body: str, local: np.ndarray, frame_local: np.ndarray) -> np.ndarray:
    """Use the exact native Ks point-offset and body-local Jacobian convention."""
    x, y, z = np.asarray(local).reshape(3)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return frame_local.T @ np.block([np.eye(3), -skew]) @ kinematics.get_frame_jaco(body, type="body")


class TactileContactEstimator:
    """Convert only complete, coherent adapter snapshots under explicit assumptions.

    No SDK, geometry maps or hand FK are duplicated here. Existing full-CRX
    kinematics supplies the world hand-base transform and contact-body Jacobian.
    Sensor acquisition clock is preserved; an unknown SDK epoch is never equated
    to host time. All outputs remain offline-only while force/mount calibration
    and acquisition synchronization have not been commissioned.
    """

    def __init__(
        self,
        kinematics: ContactKinematics,
        *,
        object_bodies: tuple[str, ...],
        tactile_joint_names: tuple[str, ...],
        friction_mu: float,
        allow_candidate_geometry: bool = False,
        max_age_s: float = 0.2,
        max_skew_s: float = 0.15,
        measured_force_policy: str = "strict",
        ambiguous_contact_policy: str = "reject",
    ) -> None:
        if len(object_bodies) != 1 or not object_bodies[0]:
            raise ValueError("Exactly one externally associated task object is required")
        if len(tactile_joint_names) != 22 or set(tactile_joint_names) != set(HAND_NAMES):
            raise ValueError("Explicit TactileAdapter joint configuration order is required")
        if not np.isfinite([friction_mu, max_age_s, max_skew_s]).all() or min(friction_mu, max_age_s, max_skew_s) <= 0:
            raise ValueError("Invalid compatibility thresholds")
        if measured_force_policy not in ("strict", "diagnostic") or ambiguous_contact_policy not in ("reject", "omit"):
            raise ValueError("Unknown contact compatibility policy")
        self.measured_force_policy = measured_force_policy
        self.ambiguous_contact_policy = ambiguous_contact_policy
        self.kinematics, self.object = kinematics, object_bodies[0]
        self.joint_names, self.mu = tuple(tactile_joint_names), friction_mu
        self.allow_candidate = allow_candidate_geometry
        self.max_age, self.max_skew = max_age_s, max_skew_s
        self.seen: dict[str, tuple] = {}
        self.last_now: float | None = None

    def estimate(self, snapshot: TactileSnapshot, feedback: JointFeedback, now: float) -> ContactEstimate:
        """Use valid contacts and healthy no-contact fingers; unavailable sensing rejects the tick."""
        provenance = {
            "model": "point_contact_with_friction",
            "local_moment": "assumed_zero_not_measured",
            "object_association": {"body": self.object, "source": "external_single_object_task_assumption"},
            "force_conversion": "f_Ada=-f_Sharpa; n_Ada=n_Sharpa",
            "force_calibration": "provisional_operator_observation_not_formal_calibration",
            "tangent_gauge": "MuJoCo_default_isotropic_point_contact",
            "acquisition_clock": "preserved_sdk_metadata; unknown_epochs_not_synchronized",
            "hardware_eligible": False,
            "measured_force_policy": self.measured_force_policy,
            "ambiguous_contact_policy": self.ambiguous_contact_policy,
            "missing_contact_semantics": "operator_confirmed_Sharpa_no_contact_when_F6_transport_healthy",
            "fingers": {finger: asdict(row.frame) if row.frame else None for finger, row in snapshot.fingers},
            "finger_statuses": {
                finger: {
                    "raw_status": row.frame.contact_status if row.frame else None,
                    "adapter_status": row.status,
                    "transport_status": row.transport_status,
                    "interpretation": "UNAVAILABLE",
                    "detail": "not evaluated",
                }
                for finger, row in snapshot.fingers
            },
        }
        try:
            fresh(feedback.source_timestamp, feedback.receive_timestamp, now, self.max_age)
            fresh(snapshot.evaluated_at, snapshot.evaluated_at, now, self.max_age)
            if self.last_now is not None and now < self.last_now:
                raise ValueError("Compatibility clock regressed")
            self.last_now = now
            if tuple(f for f, _ in snapshot.fingers) != FINGERS:
                raise ValueError("Missing, duplicate or reordered finger snapshot")
            if snapshot.configuration is None or len(snapshot.configuration) != 22:
                raise ValueError("Missing tactile joint-state association")
            by_name = dict(zip(self.joint_names, snapshot.configuration))
            if not np.allclose([by_name[n] for n in HAND_NAMES], feedback.positions_rad[6:], atol=1e-6, rtol=0):
                raise ValueError("Tactile and full-state hand configurations differ")
            self.kinematics.compute_jaco_a(np.asarray(feedback.positions_rad))
            base = rigid_transform(self.kinematics.get_frame_pose("right_hand_flange"))
            converted, timestamps = [], []
            for finger, result in snapshot.fingers:
                frame = result.frame
                if frame is None or result.status not in ("VALID", "EMPTY", "MISSING"):
                    raise ValueError(f"{finger}: {result.status}: {result.detail}")
                fresh(frame.received, frame.received, now, self.max_age)
                if abs(frame.received - feedback.source_timestamp) > self.max_skew:
                    raise ValueError(f"{finger}: feedback/contact skew")
                if frame.status != "VALID" or not frame.session_id or frame.finger != finger:
                    raise ValueError(f"{finger}: invalid transport identity")
                if (
                    len(frame.raw) != 6
                    or not np.isfinite(frame.raw).all()
                    or not frame.shape
                    or any(d <= 0 for d in frame.shape)
                    or np.prod(frame.shape) != 6
                    or frame.error
                    or frame.contact_error
                ):
                    raise ValueError(f"{finger}: invalid F6/contact source data")
                # A forward gap alone is not invalid sensing; freshness still applies.
                if any(warning != "sdk_frame_id_forward_gap" for warning in frame.warnings):
                    raise ValueError(f"{finger}: source identity/validation warning: {frame.warnings}")
                if result.status == "MISSING" and result.transport_status != "VALID":
                    raise ValueError(f"{finger}: MISSING requires valid transport")
                identity = (frame.session_id, frame.sequence, frame.sdk_frame_id, frame.received, repr(asdict(frame)))
                previous = self.seen.get(finger)
                if previous is not None and identity != previous:
                    if (
                        identity[0] != previous[0]
                        or identity[1] <= previous[1]
                        or identity[3] <= previous[3]
                        or (identity[2] is not None and previous[2] is not None and identity[2] <= previous[2])
                    ):
                        raise ValueError(f"{finger}: session change, duplicate mutation or reordered frame")
                self.seen[finger] = identity
                timestamps.append(frame.received)
                if result.status in ("MISSING", "EMPTY"):
                    if (
                        frame.contact_status != result.status
                        or frame.contact_count != 0
                        or frame.contact_row
                        or result.observation is not None
                    ):
                        raise ValueError(f"{finger}: inconsistent no-contact frame")
                    provenance["finger_statuses"][finger].update(
                        interpretation="NO_CONTACT", detail="fresh valid F6/transport; no contact points"
                    )
                    continue
                if frame.contact_status != "VALID":
                    raise ValueError(f"{finger}: inconsistent VALID contact status")
                obs = result.observation
                if obs is None or obs.finger != finger or obs.frame_id != "right_hand_flange":
                    raise ValueError(f"{finger}: missing hand-base observation")
                if (
                    obs.timestamp != frame.received
                    or obs.sequence != frame.sequence
                    or obs.session_id != frame.session_id
                    or obs.sdk_frame_id != frame.sdk_frame_id
                    or obs.raw_f6 != tuple(frame.raw)
                    or obs.contact_count != frame.contact_count
                ):
                    raise ValueError(f"{finger}: observation/frame identity mismatch")
                if obs.joint_received is None or abs(obs.joint_received - feedback.source_timestamp) > self.max_skew:
                    raise ValueError(f"{finger}: joint association unavailable/skewed")
                if obs.contact_count > 1 and self.ambiguous_contact_policy == "omit":
                    provenance["finger_statuses"][finger].update(
                        interpretation="AMBIGUOUS", detail="resultant not allocated; omitted from contact set"
                    )
                    continue
                if obs.contact_count != 1:
                    raise ValueError(f"{finger}: resultant force cannot be allocated to {obs.contact_count} contacts")
                if obs.force_convention != "object_on_finger" or obs.force_indices != (0, 1, 2):
                    raise ValueError(f"{finger}: unsupported force convention/channels")
                geometry = obs.geometry
                rigid_transform(geometry.transform_s_to_e)
                uv = np.asarray(geometry.uv_to_col_row, dtype=float)
                if (
                    not geometry.profile
                    or not geometry.source
                    or not geometry.units
                    or not geometry.verification
                    or not np.isfinite(geometry.meters_per_unit)
                    or geometry.meters_per_unit <= 0
                    or uv.shape != (2, 3)
                    or not np.isfinite(uv).all()
                    or abs(np.linalg.det(uv[:, :2])) < 1e-12
                    or not obs.force_convention_quality
                ):
                    raise ValueError(f"{finger}: invalid geometry/force provenance")
                verified = obs.geometry.verification == "VERIFIED BY CONFIG EVIDENCE"
                if not verified and not self.allow_candidate:
                    raise ValueError(f"{finger}: unverified candidate geometry requires offline opt-in")
                body = f"right_{finger}_elastomer"
                body_pose = rigid_transform(self.kinematics.get_frame_pose(body))
                position = base[:3, :3] @ obs.position + base[:3, 3]
                normal = base[:3, :3] @ obs.normal  # Never flip normal to align display arrows.
                force = -(base[:3, :3] @ obs.force)
                rotation = contact_frame(normal)
                local_force = rotation.T @ force
                if local_force[0] < 0:
                    raise ValueError(f"{finger}: negative normal projection; no automatic sign repair")
                slack = float(friction_cone_slack(local_force, self.mu)[0])
                if slack < -1e-9 and self.measured_force_policy == "strict":
                    raise ValueError(f"{finger}: force outside native circular friction cone")
                local = body_pose[:3, :3].T @ (position - body_pose[:3, 3])
                frame_local = body_pose[:3, :3].T @ rotation
                meta = {
                    "observation": asdict(obs),
                    "moment": "point_contact_assumed_zero_not_measured",
                    "object_identity": "external_task_assumption",
                    "friction_slack_n": slack,
                    "measured_cone_pass": slack >= -1e-9,
                    "measured_force_policy": self.measured_force_policy,
                    "geometry_candidate": not verified,
                    "hardware_eligible": False,
                }
                provenance["finger_statuses"][finger].update(interpretation="CONTACT", detail="valid point contact")
                converted.append(
                    {
                        "body1_name": body,
                        "body2_name": self.object,
                        "contact_pos": position,
                        "contact_pos_local": local.reshape(3, 1),
                        "contact_frame": rotation,
                        "contact_frame_local": frame_local,
                        "contact_force": np.r_[local_force, np.zeros(3)],
                        "normal_force": float(local_force[0]),
                        "hand_side": "right",
                        "valid": True,
                        "source": "tactile_replay_point_contact",
                        "provenance": meta,
                        "point_jacobian": point_jacobian(self.kinematics, body, local, frame_local),
                    }
                )
            # Use the oldest contributing receipt, never evaluation time. Original
            # per-finger acquisition/receipt identity is retained in provenance.
            timestamp = min(timestamps)
            if max(timestamps) - timestamp > self.max_skew:
                raise ValueError("Inter-finger frame skew")
            for contact in converted:
                contact["timestamp"] = timestamp
            if not converted and any(
                row["interpretation"] == "AMBIGUOUS" for row in provenance["finger_statuses"].values()
            ):
                raise ValueError("All sensed contacts are ambiguous; not a valid empty contact set")
            observation = ContactObservation(timestamp, tuple(converted), "tactile_replay_point_contact")
            validate_observation(observation, timestamp)
            return ContactEstimate("eligible" if converted else "valid_empty", observation, (), provenance)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            return ContactEstimate("insufficient_data", None, (str(error),), provenance)
