"""Versioned per-arm candidates and minimum CRX displacement selection, without ROS."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.robot_environment.model import HAND_SUFFIXES, side_slice
from ada_grasp_ctrl.robot_environment.kinematics import rigid
from ada_grasp_ctrl.sharpa_mirror import SharpaHandMirror
from .criteria.joint_effort import JointDisplacement

STAGES = ("pregrasp", "grasp", "squeeze")
SIDES = {"left", "right"}
# Matches the incumbent right selector; no additional preference or finger cost.
SEEDS = np.deg2rad(
    [
        [0, -30, 130, 0, -60, 0],
        [0, 60, 150, 180, 60, 0],
        [160, 30, 50, 0, 60, 0],
        [160, -30, 130, 0, -60, 0],
        [-160, 30, 50, 0, 60, 0],
    ]
)


def enabled_set(value: object) -> set[str]:
    """Validate a nonempty set of manipulator permissions."""
    if not isinstance(value, (list, tuple, set, frozenset)) or not value or not set(value) <= SIDES:
        raise ValueError("enabled_arms must be a nonempty subset of left/right")
    if len(value) != len(set(value)):
        raise ValueError("Duplicate arm names")
    return set(value)


def hand_names(side: str) -> tuple[str, ...]:
    """Return the canonical hand order for a validated side."""
    side_slice(side)
    return tuple(f"{side}_{n}" for n in HAND_SUFFIXES)


def validate_candidate(row: dict) -> dict:
    """Validate schema, all required stages, proper world flanges and named radians.

    Returns:
        An independent JSON-compatible candidate; two-arm candidates remain representable.

    Raises:
        ValueError: On incompatible frames, units, names, stages or nonfinite values.
    """
    row = deepcopy(row)
    for field, expected in (("schema_version", 1), ("frame_id", "world"), ("unit", "m"), ("joint_unit", "rad")):
        if row.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}")
    if not isinstance(row.get("candidate_id"), str) or not row["candidate_id"]:
        raise ValueError("Missing candidate_id")
    arms = enabled_set(row["required_arms"])
    if set(row["targets"]) != arms or not isinstance(row.get("provenance"), dict):
        raise ValueError("Targets/provenance must match required_arms")
    for side in sorted(arms):
        if set(row["targets"][side]) != set(STAGES):
            raise ValueError("Exactly pregrasp/grasp/squeeze are required")
        for stage, target in row["targets"][side].items():
            if target.get("frame_id") != "world" or target.get("pose_frame") != f"{side}_hand_flange":
                raise ValueError(f"{side}/{stage}: expected world hand_flange")
            target["T_world_hand"] = rigid(target["T_world_hand"]).tolist()
            joints = target["joint_positions_rad"]
            if set(joints) != set(hand_names(side)) or not np.isfinite(list(joints.values())).all():
                raise ValueError(f"{side}/{stage}: expected 22 finite canonical joint angles")
    return row


def from_right_record(row: dict, provenance: dict, *, side: str = "right", plane: dict | None = None) -> dict:
    """Adapt unchanged decoded right stages; optionally map reflected-world predictions left.

    No pregrasp extrapolation is performed. Source record, likelihood and raw index survive.
    """
    side_slice(side)
    mirror = SharpaHandMirror()
    targets = {}
    for stage in STAGES:
        original = row["stages"][stage]
        if original["frame_id"] != "world" or original["pose_frame"] != "right_hand_flange":
            raise ValueError("Legacy adapter requires decoded world right_hand_flange stages")
        pose = rigid(original["T_world_hand"])
        if "position_m" in original and not np.allclose(original["position_m"], pose[:3, 3], atol=1e-6, rtol=0):
            raise ValueError("Legacy XYZ and matrix disagree")
        if "quaternion_wxyz" in original:
            quat = np.asarray(original["quaternion_wxyz"], dtype=float)
            if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1, atol=1e-4):
                raise ValueError("Legacy quaternion must be unit WXYZ")
            if not np.allclose(Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix(), pose[:3, :3], atol=1e-6, rtol=0):
                raise ValueError("Legacy WXYZ and matrix disagree")
        if set(original["joint_positions_rad"]) != set(hand_names("right")):
            raise ValueError("Legacy right joint names differ")
        q = np.array([original["joint_positions_rad"][n] for n in hand_names("right")])
        if side == "left":
            if plane is None:
                raise ValueError("Left conversion requires the inference reflection plane")
            pose = mirror.mirror_wrist_pose(pose, plane["normal_world"], plane["point_world_m"])
            q = mirror.right_to_left(q)
        targets[stage] = dict(
            frame_id="world",
            pose_frame=f"{side}_hand_flange",
            T_world_hand=pose.tolist(),
            joint_positions_rad=dict(zip(hand_names(side), q.tolist())),
        )
    return validate_candidate(
        dict(
            schema_version=1,
            candidate_id=f"{side}:{row['candidate_id']}",
            required_arms=[side],
            frame_id="world",
            unit="m",
            joint_unit="rad",
            targets={side: targets},
            provenance={
                **provenance,
                "source_candidate_id": row["candidate_id"],
                "source_record": deepcopy(row),
                "generator": "mirrored_right_full" if side == "left" else "right_full",
                "reflection_plane": plane if side == "left" else None,
            },
        )
    )


def forbidden_contacts(rows: list[dict], active_side: str, *, phase: str = "pregrasp") -> list[dict]:
    """Apply stage-aware hand/object checks and retain other legacy collision gates.

    Arm penetration may reach 10 micrometers. Same-hand self and hand/table
    penetration may reach 10 mm in every phase. Active hand/object penetration
    has a 10 mm gate only during initialization and pregrasp; later contacts are
    recorded without that gate. Inter-robot, passive hand/object and unknown
    environment contacts retain the 10 micrometer gate. Hand membership comes
    from the compiled hand-flange subtrees in the contact diagnostics.

    The exact adjacent base/J1 pair is excluded only after source adjacency is
    verified. Keep it in diagnostics and leave the physical model unchanged.

    Args:
        rows: Contact diagnostics from static checking or live dynamics.
        active_side: Arm executing the grasp.
        phase: Stage used to apply the active hand/object gate.

    Returns:
        Contacts rejected by the applicable gates; input diagnostics are unchanged.

    Raises:
        ValueError: If the side or phase is unknown.
    """
    side_slice(active_side)
    if phase not in ("initial", *STAGES, "lift", "hold"):
        raise ValueError(f"Unknown contact-check phase: {phase}")
    rejected = []
    for row in rows:
        bodies = set(row["bodies"])
        category = row["category"]
        single_side = len(row["sides"]) == 1 and row["sides"][0] in SIDES
        if (
            category == "within_robot"
            and single_side
            and bodies == {f"{row['sides'][0]}_base_link", f"{row['sides'][0]}_J1_link"}
            and row.get("neighboring_source_links") is True
        ):
            continue
        threshold = 1e-5
        is_arm = any(b == f"{s}_base_link" or b.startswith(f"{s}_J") for b in bodies for s in SIDES)
        hands = bodies & set(row.get("hand_bodies", []))
        if (
            phase not in ("initial", "pregrasp")
            and single_side
            and not is_arm
            and category == "robot_object"
            and hands
            and row["sides"] == [active_side]
        ):
            continue
        if single_side and not is_arm:
            if (
                (category == "within_robot" and bodies == hands)
                or (category == "robot_table" and hands)
                or (category == "robot_object" and hands and row["sides"] == [active_side])
            ):
                threshold = 0.01
        if row["penetration_m"] > threshold:
            rejected.append(row)
    return rejected


def select_candidates(
    robot: object,
    sim: object,
    rows: list[dict],
    enabled_arms: object,
    q_current: object,
    weights: object = None,
    max_nfev: int = 200,
) -> dict:
    """Evaluate bounded seeds and all endpoints, then select one globally minimum effort.

    Cost is squared six-joint displacement from the supplied current state, in rad².
    Equal costs break by candidate ID then seed index; static success is not execution readiness.
    """
    enabled = enabled_set(enabled_arms)
    current = robot.configuration(q_current)
    w = JointDisplacement().weights if weights is None else JointDisplacement(weights).weights
    ids = [r.get("candidate_id") for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be present and globally unique")
    evaluations, feasible = [], []
    for raw in rows:
        result = dict(candidate_id=raw["candidate_id"], reasons=[], attempts=[])
        evaluations.append(result)
        try:
            row = validate_candidate(raw)
        except (ValueError, KeyError, TypeError) as exc:
            result.update(reasons=["invalid_candidate"], error=str(exc))
            continue
        arms = set(row["required_arms"])
        if len(arms) != 1:
            result["reasons"] = ["unsupported_multi_arm_candidate"]
            continue
        if not arms <= enabled:
            result["reasons"] = ["disabled_arm"]
            continue
        side = next(iter(arms))
        arm, block = side_slice(side, True), side_slice(side)
        finger = slice(block.start + 6, block.stop)
        targets = row["targets"][side]
        hand = {s: np.array([targets[s]["joint_positions_rad"][n] for n in hand_names(side)]) for s in STAGES}
        if any(np.any(q < robot.limits[finger, 0]) or np.any(q > robot.limits[finger, 1]) for q in hand.values()):
            result["reasons"] = ["hand_joint_limits"]
            continue
        for index, seed in enumerate([current[arm], *SEEDS]):
            attempt = dict(seed_index=index, stages={}, reasons=[])
            result["attempts"].append(attempt)
            if np.any(seed < robot.limits[arm, 0]) or np.any(seed > robot.limits[arm, 1]):
                attempt["reasons"].append("seed_out_of_bounds")
                continue
            q = current.copy()
            q[arm] = seed
            stage_q = {}
            for stage in ("grasp", "pregrasp", "squeeze"):
                if stage != "grasp":
                    q = stage_q["grasp"].copy()
                q[finger] = hand[stage]
                try:
                    q, diag = robot.solve_arm_ik(q, side, targets[stage]["T_world_hand"], max_nfev=max_nfev)
                    q = robot.configuration(q)
                    if not np.array_equal(
                        q[side_slice("right" if side == "left" else "left")],
                        current[side_slice("right" if side == "left" else "left")],
                    ):
                        raise ValueError("IK changed the passive robot state")
                    contacts = sim.check_static_contacts(q)
                    bad = forbidden_contacts(contacts, side, phase=stage)
                    attempt["stages"][stage] = dict(ik=diag, contacts=contacts, forbidden_contacts=bad, q56=q.tolist())
                    if not diag["accepted"] or bad:
                        attempt["reasons"].append(stage + (":ik_rejected" if not diag["accepted"] else ":collision"))
                        break
                    stage_q[stage] = q.copy()
                except (ValueError, RuntimeError) as exc:
                    attempt["reasons"].append(f"{stage}:solver_error:{exc}")
                    break
            if attempt["reasons"]:
                continue
            delta = stage_q["grasp"][arm] - current[arm]
            cost = float(np.dot(w * delta, delta))
            if not np.isfinite(cost):
                attempt["reasons"].append("nonfinite_cost")
                continue
            attempt["effort"] = cost
            feasible.append(
                dict(
                    candidate_id=row["candidate_id"],
                    selected_side=side,
                    required_arms=[side],
                    source_candidate_id=row["provenance"].get("source_candidate_id"),
                    effort=cost,
                    delta_rad=delta.tolist(),
                    seed_index=index,
                    crx_target_rad=stage_q["grasp"][arm].tolist(),
                    stages=attempt["stages"],
                    candidate=row,
                )
            )
        if not any(x["candidate_id"] == row["candidate_id"] for x in feasible):
            result["reasons"] = ["no_feasible_ik_branch"]
    feasible.sort(key=lambda x: (x["effort"], x["candidate_id"], x["seed_index"]))
    return dict(
        schema_version=1,
        status="selected" if feasible else "no_feasible_candidate",
        enabled_arms=sorted(enabled),
        weights=w.tolist(),
        effort_unit="rad^2",
        effort_definition="sum(W * (q_grasp_CRX - q_current_CRX)^2); no angle wrapping",
        active_hand_object_penetration_policy="10 mm during initialization/pregrasp only",
        q_current=current.tolist(),
        selected=feasible[0] if feasible else None,
        evaluations=evaluations,
        execution_ready=False,
        hardware_ready=False,
    )
