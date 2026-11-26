"""Single-arm placement IK with the existing static contact policy; no transport planner."""

import numpy as np

from ada_grasp_ctrl.grasp_selection.multi_arm import SEEDS, forbidden_contacts
from ada_grasp_ctrl.robot_environment.kinematics import rigid
from ada_grasp_ctrl.robot_environment.model import side_slice


class PlacementGate:
    """Use the existing bounded IK and choose the solution closest to the held stage."""

    def __init__(self, robot: object, static: object, target: dict, canceled: object = lambda: False) -> None:
        self.robot, self.static, self.target, self.canceled = robot, static, target, canceled
        self.delta = rigid(target["Delta_T"])

    def __call__(self, row: dict, side: str, held: np.ndarray) -> dict:
        """Test the adjusted terminal squeeze wrist, keeping its fingers and passive arm."""
        source = rigid(row["targets"][side]["squeeze"]["T_world_hand"])
        destination = self.delta @ source
        arm = side_slice(side, True)
        mask = np.ones(56, dtype=bool)
        mask[arm] = False
        attempts, valid = [], []
        for index, seed in enumerate([held[arm], *SEEDS]):
            if self.canceled():
                raise RuntimeError("Placement evaluation canceled")
            q = held.copy()
            q[arm] = seed
            try:
                q = self.robot.configuration(q)
                q, diagnostic = self.robot.solve_arm_ik(q, side, destination)
                q = self.robot.configuration(q)
                if not np.array_equal(q[mask], held[mask]):
                    raise ValueError("Placement IK changed fingers or passive arm")
                attempt = dict(seed_index=index, ik=diagnostic, q56=q.tolist(), accepted=False, collision_checked=False)
                if diagnostic["accepted"]:
                    contacts = self.static.check_static_contacts(q)
                    bad = forbidden_contacts(contacts, side, phase="squeeze")
                    attempt.update(
                        contacts=contacts,
                        forbidden_contacts=bad,
                        collision_checked=True,
                        accepted=not bad,
                        reason="place_endpoint_collision" if bad else "accepted",
                    )
                    if not bad:
                        valid.append((float(np.sum((q[arm] - held[arm]) ** 2)), index, q, diagnostic))
                else:
                    attempt["reason"] = "place_endpoint_ik_rejected"
                attempts.append(attempt)
            except (ValueError, RuntimeError) as error:
                attempts.append(dict(seed_index=index, error=str(error), accepted=False))
        result = dict(
            accepted=bool(valid),
            reason="accepted"
            if valid
            else (
                "place_endpoint_collision"
                if any(a.get("forbidden_contacts") for a in attempts)
                else "place_endpoint_ik_rejected"
            ),
            held_reference_stage="adjusted_squeeze_before_lift",
            pose_frame=f"{side}_hand_flange",
            T_world_hand_source_held=source.tolist(),
            T_world_hand_place=destination.tolist(),
            source_held_q56=held.tolist(),
            attempts=attempts,
            collision_checked=any(a.get("collision_checked", False) for a in attempts),
            collision_policy="existing squeeze contact policy; static robot/table scene without object solid",
            target_revision=self.target["target_revision"],
            target_sha256=self.target["target_sha256"],
        )
        if valid:
            _, index, q, diagnostic = min(valid, key=lambda item: item[:2])
            result.update(q56=q.tolist(), arm_positions_rad=q[arm].tolist(), ik=diagnostic, seed_index=index)
        return result
