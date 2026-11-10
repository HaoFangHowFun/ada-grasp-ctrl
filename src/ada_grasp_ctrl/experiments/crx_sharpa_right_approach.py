"""Collision-checked overhead approach for the private right CRX experiment."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .crx_sharpa_right_model import JOINTS, matrix_pose, pose_matrix
from .crx_sharpa_right_preparation import RightKinematics, body_pose
from .crx_sharpa_right_simulation import ApprovedScene, invalid_initialization
from .sharpa_model import PALM

HEIGHT_M = 0.30
SEGMENTS = ("raise_clearance", "move_above_object", "orient_above_object", "descend_to_pregrasp")
INITIALIZATION_SEGMENTS = frozenset((*SEGMENTS, *(f"settle_{s}" for s in SEGMENTS)))


def overhead_pose(palm: np.ndarray, object_pose: np.ndarray) -> np.ndarray:
    """Place the palm frame 30 cm above object center, retaining its orientation."""
    target = palm.copy()
    target[:3, 3] = object_pose[:3] + np.array([0.0, 0.0, HEIGHT_M])
    return target


def interpolate_pose(start: np.ndarray, end: np.ndarray, steps: int) -> list[np.ndarray]:
    """Interpolate translation and shortest-path rotation, excluding the start pose."""
    rotations = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], end[:3, :3]])))
    poses = []
    for fraction in np.linspace(0, 1, steps + 1)[1:]:
        pose = np.eye(4)
        pose[:3, 3] = (1 - fraction) * start[:3, 3] + fraction * end[:3, 3]
        pose[:3, :3] = rotations(fraction).as_matrix()
        poses.append(pose)
    return poses


def prepare_approach(row: dict, record: dict, model_info: dict) -> dict:
    """Plan lift-clearance, overhead translation/rotation, then the original pregrasp.

    Return an attributable invalid status on failed IK or sampled collision. The
    open hand is held until overhead alignment; original pregrasp fingers are
    interpolated only during descent. No targets or scene registration are moved.
    """
    kin = RightKinematics(model_info)
    sim = ApprovedScene(Path(row["scene_path"]))
    initial = np.asarray(record["initial_qpos"])
    pregrasp = np.asarray(record["pregrasp_qpos"])
    flange = kin.forward(initial)
    palm = body_pose(kin.data, PALM)
    palm_to_flange = np.linalg.solve(palm, flange)
    kin.forward(pregrasp)
    pregrasp_palm = body_pose(kin.data, PALM)
    overhead = overhead_pose(palm, record["obj_pose"])
    clearance = palm.copy()
    clearance[2, 3] = max(palm[2, 3], overhead[2, 3])
    aligned = overhead.copy()
    aligned[:3, :3] = pregrasp_palm[:3, :3]
    waypoints = [clearance, overhead, aligned, pregrasp_palm]
    q = initial.copy()
    start = palm.copy()
    segments = []
    try:
        for segment, target in zip(SEGMENTS, waypoints):
            angle = Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T).magnitude()
            duration = max(2.0, np.linalg.norm(target[:3, 3] - start[:3, 3]) / 0.1, angle / 0.5)
            steps = math.ceil(duration * 10)
            path = []
            for index, pose in enumerate(interpolate_pose(start, target, steps), 1):
                fraction = index / steps
                hand = initial[6:].copy()
                if segment == "descend_to_pregrasp":
                    hand = (1 - fraction) * initial[6:] + fraction * pregrasp[6:]
                solved, diagnostic = kin.solve(pose @ palm_to_flange, hand, q[:6])
                if not diagnostic["reachable"] or not diagnostic["arm_table_clear"]:
                    return {
                        "status": "ik_invalid",
                        "reason": f"Overhead approach IK failed: {segment} step {index}",
                        "diagnostic": diagnostic,
                        "target_palm_wxyz": matrix_pose(pose).tolist(),
                    }
                delta = np.abs(solved[:6] - q[:6])
                limits = np.array([float(model_info["arm_limits"][n]["velocity"]) for n in JOINTS[:6]])
                if np.any(delta > np.minimum(0.35, limits * 0.1)):
                    return {"status": "ik_invalid", "reason": f"Overhead IK branch/velocity discontinuity: {segment}"}
                # Check actuator interpolation as well as Cartesian endpoints.
                for intermediate in np.linspace(q, solved, 6)[1:]:
                    sim.reset_pose_qpos(intermediate, record["obj_pose"])
                    contacts = sim.contact_diagnostics()
                    premature = segment != "descend_to_pregrasp" and bool(sim.get_curr_contact_info())
                    if invalid_initialization(contacts) or premature:
                        return {
                            "status": "initialization_invalid",
                            "reason": f"Overhead path contact: {segment}",
                            "waypoint": index,
                            "diagnostic": contacts,
                        }
                path.append(solved.tolist())
                q = solved
            segments.append({"name": segment, "path": path, "target_palm_wxyz": matrix_pose(target).tolist()})
            start = target
        # Do not hand off to a different arm branch than the frozen grasp references.
        if np.max(np.abs(q[:6] - pregrasp[:6])) > 0.05:
            return {"status": "ik_invalid", "reason": "Overhead descent ends on a different pregrasp IK branch"}
        return {"status": "ready", "overhead_palm_wxyz": matrix_pose(aligned).tolist(), "segments": segments}
    finally:
        sim.close_view_and_render()


def actual_palm_error(sim: ApprovedScene, target_wxyz: list[float]) -> dict:
    """Measure actual palm pose after updating FK, without a physics step."""
    import mujoco

    mujoco.mj_forward(sim.model, sim.data)
    actual = body_pose(sim.data, PALM)
    target = pose_matrix(np.asarray(target_wxyz))
    return {
        "position_m": float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        "orientation_rad": float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()),
        "actual_palm_wxyz": matrix_pose(actual).tolist(),
    }
