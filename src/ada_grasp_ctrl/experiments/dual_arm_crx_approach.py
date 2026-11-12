"""Extend the successful right-CRX overhead route to a named active arm."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.grasp_selection.multi_arm import forbidden_contacts
from ada_grasp_ctrl.robot_environment.model import side_slice
from .crx_sharpa_right_approach import HEIGHT_M, SEGMENTS, interpolate_pose
from .simulated_object_cloud import body_pose


def prepare_overhead(scene: object, chosen: dict) -> dict:
    """Plan the historical four segments without moving either simulated robot or object.

    Hold the configured open fingers until descent. Retain the selected pregrasp
    branch, existing IK acceptance criteria, and all passive-arm collision checks.
    """
    robot, sim = scene.robot, scene.sim
    side = chosen["selected_side"]
    block, arm = side_slice(side), side_slice(side, True)
    initial = robot.configuration(sim.configuration())
    pregrasp = np.asarray(chosen["stages"]["pregrasp"]["q56"])
    palm = robot.fk(initial, side, f"{side}_hand_C_MC")
    palm_to_flange = np.linalg.solve(palm, robot.fk(initial, side))
    pregrasp_palm = robot.fk(pregrasp, side, f"{side}_hand_C_MC")
    overhead = palm.copy()
    overhead[:3, 3] = body_pose(sim)[:3, 3] + [0, 0, HEIGHT_M]
    clearance = palm.copy()
    clearance[2, 3] = max(palm[2, 3], overhead[2, 3])
    aligned = overhead.copy()
    aligned[:3, :3] = pregrasp_palm[:3, :3]
    q, start, segments = initial.copy(), palm.copy(), []
    velocities = robot.pin_model.velocityLimit[robot.pin_v[arm]]
    for name, target in zip(SEGMENTS, (clearance, overhead, aligned, pregrasp_palm)):
        angle = Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T).magnitude()
        steps = math.ceil(10 * max(2.0, np.linalg.norm(target[:3, 3] - start[:3, 3]) / 0.1, angle / 0.5))
        path = []
        for index, pose in enumerate(interpolate_pose(start, target, steps), 1):
            seed = q.copy()
            hand = initial[block][6:].copy()
            if name == "descend_to_pregrasp":
                hand += index / steps * (pregrasp[block][6:] - hand)
            seed[block][6:] = hand
            solved, diagnostic = robot.solve_arm_ik(seed, side, pose @ palm_to_flange)
            failure = dict(
                status="motion_failed", execution_ready=False, phase="pregrasp", segment=name, waypoint=index
            )
            if not diagnostic["accepted"]:
                return dict(**failure, reason="overhead_ik_failed", ik=diagnostic, target_palm=pose.tolist())
            delta = np.abs(solved[arm] - q[arm])
            if np.any(delta > np.minimum(0.35, velocities * 0.1)):
                return dict(**failure, reason="overhead_branch_or_velocity_jump", joint_delta_rad=delta.tolist())
            for intermediate in np.linspace(q, solved, 6)[1:]:
                contacts = sim.check_static_contacts(intermediate)
                bad = forbidden_contacts(contacts, side, phase="pregrasp")
                premature = name != "descend_to_pregrasp" and any(
                    c["category"] == "robot_object" and c["sides"] == [side] for c in contacts
                )
                if bad or premature:
                    return dict(**failure, reason="overhead_path_collision", contacts=bad or contacts)
            path.append(solved.tolist())
            q = solved
        segments.append(dict(name=name, path=path, target_palm=target.tolist()))
        start = target
    if np.max(np.abs(q[arm] - pregrasp[arm])) > 0.05:
        return dict(status="motion_failed", execution_ready=False, phase="pregrasp", reason="pregrasp_branch_mismatch")
    return dict(status="ready", segments=segments, overhead_height_m=HEIGHT_M)
