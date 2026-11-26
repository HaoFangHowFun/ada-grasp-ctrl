"""Read existing fingertip adapters without changing simulation or contact contracts."""

import mujoco
import numpy as np

from ada_grasp_ctrl.contact.force_display import FINGERS, SIDES, FingerForce, ForceSnapshot
from ada_grasp_ctrl.contact.mujoco_adapter import MuJoCoContactAdapter
from ada_grasp_ctrl.contact.types import ContactObservation
from ada_grasp_ctrl.contact.validation import validate_observation

EPSILON = 1e-12  # Numerical degeneracy only; not a physical load threshold.


def unavailable(timestamp: float, error: str) -> ForceSnapshot:
    """Represent failed sampling explicitly, never as measured zero or contact loss."""
    return ForceSnapshot(timestamp, tuple(FingerForce(s, f, "UNAVAILABLE") for s in SIDES for f in FINGERS), (error,))


def aggregate(observation: ContactObservation, side: str, geom_fingers: dict[int, str]) -> tuple[FingerForce, ...]:
    """Sum object forces and normal-load-weighted positions/normals for display only.

    Raises:
        ValueError: If any measurement is invalid; no apparently valid partial sum escapes.
    """
    validate_observation(observation, observation.timestamp)
    groups = {finger: [] for finger in FINGERS}
    for contact in observation.contacts:
        if contact["hand_side"] != side:
            raise ValueError("Mismatched adapter side")
        fingers = [geom_fingers[g] for g in contact["geom_ids"] if g in geom_fingers]
        if len(fingers) != 1:
            continue
        if contact["contact_force"][0] > 0:
            groups[fingers[0]].append(contact)
    result = []
    for finger, rows in groups.items():
        if not rows:
            result.append(FingerForce(side, finger, "NO_CONTACT", normal_load=0.0))
            continue
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            weights = np.array([c["contact_force"][0] for c in rows])
            total = float(weights.sum())
            if total <= EPSILON:
                result.append(FingerForce(side, finger, "DEGENERATE"))
                continue
            force = sum((c["contact_frame"] @ c["contact_force"][:3] for c in rows), np.zeros(3))
            position = sum((w * c["contact_pos"] for w, c in zip(weights / total, rows)), np.zeros(3))
            normal = sum((w * c["contact_frame"][:, 0] for w, c in zip(weights / total, rows)), np.zeros(3))
            length = np.linalg.norm(normal)
            if not np.isfinite([force, position, normal]).all():
                raise ValueError("Nonfinite fingertip resultant")
            result.append(
                FingerForce(
                    side,
                    finger,
                    "VALID" if length > EPSILON else "NORMAL_DEGENERATE",
                    tuple(position),
                    tuple(force),
                    tuple(normal / length) if length > EPSILON else None,
                    normal_load=total,
                )
            )
    return tuple(result)


class FingertipForceSource:
    """Two adapters on the actual model/data; exact elastomer collision geoms only."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.data = data
        self.adapters, self.geom_fingers, self.errors = {}, {}, {}
        root = model.body("object").id
        object_ids = {root}
        for body in range(root + 1, model.nbody):
            if int(model.body_parentid[body]) in object_ids:
                object_ids.add(body)
        object_bodies = {model.body(i).name for i in object_ids}
        for side in SIDES:
            try:
                bodies, geoms = set(), {}
                for finger in FINGERS:
                    name = f"{side}_{finger}_elastomer"
                    body, geom = model.body(name), model.geom(name + "_collision_0")
                    if model.geom_bodyid[geom.id] != body.id or not (geom.contype[0] or geom.conaffinity[0]):
                        raise ValueError(f"Invalid sensing collision geometry: {name}")
                    bodies.add(name)
                    geoms[geom.id] = finger
                self.adapters[side] = MuJoCoContactAdapter(
                    model, data, hand_bodies=bodies, object_bodies=object_bodies, hand_side=side
                )
                self.geom_fingers[side] = geoms
            except Exception as error:
                self.errors[side] = f"{side}: {type(error).__name__}: {error}"

    def sample(self) -> ForceSnapshot:
        """Call under the physics owner's lock, after its step/forward, without advancing time."""
        rows, errors = [], []
        for side in SIDES:
            try:
                if side in self.errors:
                    raise ValueError(self.errors[side])
                rows.extend(aggregate(self.adapters[side].observe(), side, self.geom_fingers[side]))
            except Exception as error:
                # The existing adapter validates a whole side and may fail on an
                # environment record too. Never reinterpret that failure as no contact.
                rows.extend(FingerForce(side, f, "UNAVAILABLE") for f in FINGERS)
                errors.append(f"{side}: {type(error).__name__}: {str(error)[:180]}")
        return ForceSnapshot(float(self.data.time), tuple(rows), tuple(errors))
