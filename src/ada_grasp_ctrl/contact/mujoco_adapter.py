"""Ground-truth contact sampling using the repository's canonical wrench transform."""

from __future__ import annotations

import mujoco
import numpy as np

from ada_grasp_ctrl.utils.hand_util import canonicalize_contact_frame_wrench
from .types import ContactObservation
from .validation import validate_observation


class MuJoCoContactAdapter:
    """Read exact named hand/object pairs, retaining every contact in MuJoCo order.

    Args:
        model: Existing compiled model; no settings are modified.
        data: Live simulation data, sampled without forwarding or stepping.
        hand_bodies: Exact body names, excluding arm and table bodies.
        object_bodies: Exact object body names, including descendants if needed.
        hand_side: Explicit side metadata for a future interchangeable source.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        hand_bodies: set[str],
        object_bodies: set[str],
        hand_side: str = "right",
    ) -> None:
        if not hand_bodies or not object_bodies or hand_bodies & object_bodies or hand_side not in ("left", "right"):
            raise ValueError("Expected disjoint nonempty hand/object body sets and a known hand side")
        self.model, self.data, self.hand_side = model, data, hand_side
        self.hand_ids = {model.body(name).id for name in hand_bodies}
        self.object_ids = {model.body(name).id for name in object_bodies}

    def observe(self) -> ContactObservation:
        """Return all hand-object contacts; invalid readings reject the whole snapshot."""
        contacts, errors, environment = [], [], []
        timestamp = float(self.data.time)
        for index, contact in enumerate(self.data.contact):
            a, b = (int(self.model.geom_bodyid[g]) for g in contact.geom)
            direct = a in self.hand_ids and b in self.object_ids
            if not direct and not (b in self.hand_ids and a in self.object_ids):
                if a in self.object_ids or b in self.object_ids:
                    raw = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, index, raw)
                    if not np.isfinite(raw).all():
                        errors.append(f"environment contact {index}: nonfinite wrench")
                    else:
                        environment.append(
                            {
                                "contact_id": index,
                                "body_names": [self.model.body(i).name for i in (a, b)],
                                "contact_pos": contact.pos.copy(),
                                "frame_world_columns": contact.frame.reshape(3, 3).T.copy(),
                                "wrench_on_geom2": raw,
                                "timestamp": timestamp,
                            }
                        )
                continue
            hand, obj = (a, b) if direct else (b, a)
            try:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                frame, wrench = canonicalize_contact_frame_wrench(contact.frame, wrench, hand_is_geom1=direct)
                rotation = self.data.xmat[hand].reshape(3, 3)
                position = contact.pos.copy()
                contacts.append(
                    {
                        "body1_name": self.model.body(hand).name,
                        "body2_name": self.model.body(obj).name,
                        "contact_dist": float(contact.dist),
                        "contact_pos": position,
                        "contact_pos_local": rotation.T @ (position - self.data.xpos[hand]).reshape(3, 1),
                        "contact_frame": frame,
                        "contact_frame_local": rotation.T @ frame,
                        "contact_force": wrench,
                        "normal_force": float(wrench[0]),
                        "timestamp": timestamp,
                        "hand_side": self.hand_side,
                        "source": "mujoco",
                        "valid": True,
                        "contact_id": index,
                        "geom_ids": [int(g) for g in contact.geom],
                    }
                )
            except ValueError as error:
                errors.append(f"contact {index}: {error}")
        observation = ContactObservation(
            timestamp, tuple(contacts), "mujoco", not errors, tuple(errors), tuple(environment)
        )
        # Never drop a broken measurement and continue with an apparently valid subset.
        validate_observation(observation, timestamp)
        return observation
