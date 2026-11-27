"""Placement endpoint IK with existing static contact policies; no transport search."""

import numpy as np

from ada_grasp_ctrl.grasp_selection.multi_arm import SEEDS, forbidden_contacts
from ada_grasp_ctrl.robot_environment.kinematics import rigid
from ada_grasp_ctrl.robot_environment.model import side_slice
from .face_coverage import FaceCoverage


class PlacementGate:
    """Use the existing bounded IK and choose the solution closest to the held stage."""

    def __init__(self, robot: object, static: object, target: dict, canceled: object = lambda: False) -> None:
        self.robot, self.static, self.target, self.canceled = robot, static, target, canceled
        self.delta = rigid(target["Delta_T"])
        self.face_coverage = FaceCoverage(static, target) if len(target.get("contact_patch_world_m", [])) >= 3 else None

    def __call__(self, row: dict, side: str, held: np.ndarray) -> dict:
        """Test the adjusted terminal squeeze wrist, keeping its fingers and passive arm."""
        if self.canceled():
            raise RuntimeError("Placement evaluation canceled")
        if side == "both":
            return self.paired(row, held)
        source = rigid(row["targets"][side]["squeeze"]["T_world_hand"])
        destination = self.delta @ source
        coverage = self.face_coverage.check(side, held, destination) if self.face_coverage else None
        if coverage is not None and not coverage["accepted"]:
            return dict(
                accepted=False,
                reason="placement_face_covered",
                face_coverage=coverage,
                attempts=[],
                collision_checked=False,
                target_revision=self.target["target_revision"],
                target_sha256=self.target["target_sha256"],
            )
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
            face_coverage=coverage,
        )
        if valid:
            _, index, q, diagnostic = min(valid, key=lambda item: item[:2])
            result.update(q56=q.tolist(), arm_positions_rad=q[arm].tolist(), ik=diagnostic, seed_index=index)
        return result

    def paired(self, row: dict, held: np.ndarray) -> dict:
        """Combine independent bounded IK branches and check the complete endpoint.

        Paired hands may cover the support face. Grasp ranking remains owned by
        the selector; this gate only chooses the nearest feasible Place branch.
        """
        from itertools import product

        sides = ["left", "right"]
        source = {s: rigid(row["targets"][s]["squeeze"]["T_world_hand"]) for s in sides}
        destination = {s: self.delta @ source[s] for s in sides}
        branches, attempts = {}, []
        for side in sides:
            arm = side_slice(side, True)
            branches[side] = []
            for index, seed in enumerate([held[arm], *SEEDS]):
                if self.canceled():
                    raise RuntimeError("Placement evaluation canceled")
                q = held.copy()
                q[arm] = seed
                try:
                    q, diagnostic = self.robot.solve_arm_ik(self.robot.configuration(q), side, destination[side])
                    q = self.robot.configuration(q)
                    mask = np.ones(56, dtype=bool)
                    mask[arm] = False
                    if not np.array_equal(q[mask], held[mask]):
                        raise ValueError("Placement IK changed other joints")
                    attempts.append(dict(side=side, seed_index=index, ik=diagnostic))
                    if diagnostic["accepted"] and not any(np.allclose(q[arm], b[0]) for b in branches[side]):
                        branches[side].append((q[arm].copy(), diagnostic, index))
                except (ValueError, RuntimeError) as error:
                    attempts.append(dict(side=side, seed_index=index, error=str(error)))
        valid, combined = [], []
        for left, right in product(branches["left"], branches["right"]):
            if self.canceled():
                raise RuntimeError("Placement evaluation canceled")
            q = held.copy()
            for side, branch in zip(sides, (left, right)):
                q[side_slice(side, True)] = branch[0]
            rows = self.static.check_static_contacts(q)
            bad = forbidden_contacts(rows, sides, phase="squeeze")
            combined.append(dict(forbidden_contacts=bad, accepted=not bad))
            if not bad:
                effort = sum(float(np.sum((q[side_slice(s, True)] - held[side_slice(s, True)]) ** 2)) for s in sides)
                valid.append((effort, q, {s: b[1] for s, b in zip(sides, (left, right))}))
        result = dict(
            accepted=bool(valid),
            reason="accepted" if valid else "place_endpoint_collision" if combined else "place_endpoint_ik_rejected",
            source_held_q56=held.tolist(),
            T_world_hand_source_held={s: source[s].tolist() for s in sides},
            T_world_hand_place={s: destination[s].tolist() for s in sides},
            attempts=attempts,
            combined_endpoints=combined,
            collision_checked=bool(combined),
            face_coverage=dict(skipped=True, reason="paired_support_face_contact_allowed"),
            target_revision=self.target["target_revision"],
            target_sha256=self.target["target_sha256"],
        )
        if valid:
            _, q, diagnostic = min(valid, key=lambda item: item[0])
            result.update(q56=q.tolist(), ik=diagnostic)
        return result
