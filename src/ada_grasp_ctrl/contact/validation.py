"""Reject unavailable, stale or malformed contact quantities before control."""

from __future__ import annotations

import numpy as np

from .types import ContactObservation


def validate_observation(observation: ContactObservation, timestamp: float) -> None:
    """Reject invalid data; an empty valid snapshot means genuine contact loss."""
    if not observation.valid or observation.errors or not observation.source:
        raise ValueError(f"Invalid contact observation: {observation.errors}")
    if not np.isfinite([timestamp, observation.timestamp]).all() or abs(timestamp - observation.timestamp) > 1e-9:
        raise ValueError("Contact observation is stale or has an invalid timestamp")
    for i, contact in enumerate(observation.contacts):
        if contact.get("valid") is not True or contact.get("hand_side") not in ("right", "left"):
            raise ValueError(f"Contact {i}: invalid validity/hand metadata")
        if not contact.get("body1_name") or not contact.get("body2_name"):
            raise ValueError(f"Contact {i}: missing body identity")
        if contact.get("timestamp") != observation.timestamp or contact.get("source") != observation.source:
            raise ValueError(f"Contact {i}: mismatched timestamp/source")
        for key, shape in (
            ("contact_pos", (3,)),
            ("contact_pos_local", (3, 1)),
            ("contact_frame", (3, 3)),
            ("contact_frame_local", (3, 3)),
            ("contact_force", (6,)),
        ):
            value = np.asarray(contact.get(key), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Contact {i}: unavailable or nonfinite {key}")
            if "frame" in key and (
                not np.allclose(value.T @ value, np.eye(3), atol=1e-7, rtol=0)
                or not np.isclose(np.linalg.det(value), 1.0, atol=1e-7, rtol=0)
            ):
                raise ValueError(f"Contact {i}: invalid {key}")
        normal = contact.get("normal_force")
        if normal is None or not np.isfinite(normal) or normal < -1e-7:
            raise ValueError(f"Contact {i}: unavailable or negative normal force")
        if abs(normal - contact["contact_force"][0]) > 1e-9:
            raise ValueError(f"Contact {i}: inconsistent normal force")
