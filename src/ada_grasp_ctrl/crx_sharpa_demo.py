"""Experimental single-arm CRX-5iA and Sharpa grasp example utilities."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.crx_sharpa_contract import (
    ARM_JOINTS,
    ASSET,
    HAND_JOINTS,
    JOINTS,
    OBJECT,
    OBJECT_MASS,
    OBJECT_SCALE,
    PALM_FRAME,
)
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.schema import validate_grasp_record, with_current_schema
from ada_grasp_ctrl.utils.hand_util import MjHO
from ada_grasp_ctrl.utils.pin_helper import PinocchioHelper


def build_control_model(destination: Path) -> Path:
    """Convert the preview URDF to a MuJoCo model with 28 position actuators.

    Args:
        destination: New MJCF path below the example run directory.

    Returns:
        Absolute MJCF path.
    """
    source = resolve_from_root(ASSET, root_kind="asset")
    spec = mujoco.MjSpec.from_file(str(source))
    names = {joint.name for joint in spec.joints}
    if names != set(JOINTS):
        raise ValueError(f"CRX-Sharpa joint mismatch: missing={set(JOINTS) - names}, extra={names - set(JOINTS)}")
    for mesh in spec.meshes:
        mesh.file = str((source.parent / mesh.file).resolve(strict=True))
    root = ET.fromstring(spec.to_xml())
    actuators = ET.SubElement(root, "actuator")
    for name in JOINTS:
        is_arm = name in ARM_JOINTS
        ET.SubElement(
            actuators, "position", name=name, joint=name,
            kp="100" if is_arm else "5", kv="10" if is_arm else "0.5",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(destination))
    if model.nu != len(JOINTS) or model.nq != len(JOINTS):
        raise ValueError(f"Control model expected 28 joints/actuators, got nq={model.nq}, nu={model.nu}")
    return destination.resolve()


def _pose(helper: PinocchioHelper, arm_q: np.ndarray, hand_q: np.ndarray | None = None) -> np.ndarray:
    """Return the world-frame palm transform for named joint vectors.

    Args:
        helper: Full URDF kinematics model.
        arm_q: Six arm joint positions.
        hand_q: Optional 22-joint hand position.

    Returns:
        Homogeneous palm transform in the world frame.
    """
    values = dict(zip(JOINTS, np.r_[arm_q, np.zeros(22) if hand_q is None else hand_q]))
    q = np.array([values[name] for name in helper.dof_joint_names])
    helper.compute_forward_kinematics(q)
    return helper.get_frame_pose(PALM_FRAME).copy()


def solve_arm_ik(helper: PinocchioHelper, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Solve the six CRX joints for one world-frame palm target.

    Args:
        helper: Pinocchio model built from the source URDF.
        target: Palm transform as a finite 4-by-4 matrix.
        seed: Six arm angles in radians.

    Returns:
        Bounded six-joint IK solution.

    Raises:
        RuntimeError: If the target remains outside the accepted pose error.
    """
    indices = [helper.dof_joint_names.index(name) for name in ARM_JOINTS]
    lower, upper = helper.joint_limits[:, indices]

    def error(arm_q: np.ndarray) -> np.ndarray:
        actual = _pose(helper, arm_q)
        position = target[:3, 3] - actual[:3, 3]
        orientation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
        return np.r_[10 * position, orientation]

    result = least_squares(error, np.clip(seed, lower, upper), bounds=(lower, upper), max_nfev=80)
    residual = error(result.x)
    if not result.success or np.linalg.norm(residual[:3]) > 0.01 or np.linalg.norm(residual[3:]) > 0.01:
        raise RuntimeError(f"CRX palm IK failed: success={result.success}, residual={residual}")
    return result.x


def build_grasp_record() -> dict[str, object]:
    """Create a reproducible reach, open, and close grasp for the bundled bottle.

    Returns:
        Validated 28-joint common grasp record.
    """
    helper = PinocchioHelper(str(resolve_from_root(ASSET, root_kind="asset")), "urdf")
    seed_arm = np.deg2rad([0.0, 30.0, -30.0, 0.0, 60.0, 0.0])
    grasp_target = _pose(helper, seed_arm)
    # Keep the bottle's 0.06-scaled collision mesh supported by the tabletop.
    grasp_target[2, 3] = 0.081
    arm_grasp = solve_arm_ik(helper, grasp_target, seed_arm)
    palm = _pose(helper, arm_grasp)
    approach = palm.copy()
    approach[:3, 3] -= 0.12 * palm[:3, 2]
    arm_pregrasp = solve_arm_ik(helper, approach, arm_grasp)
    hand_open = np.zeros(len(HAND_JOINTS))
    hand_close = hand_open.copy()
    for index, name in enumerate(HAND_JOINTS):
        if any(token in name for token in ("_FE", "_PIP", "_DIP", "_IP")):
            limit_index = helper.dof_joint_names.index(name)
            hand_close[index] = min(0.8 * helper.joint_limits[1, limit_index], 1.0)
    object_position = palm[:3, 3] + palm[:3, :3] @ np.array([0.06, -0.02, 0.07])
    record = with_current_schema({
        "obj_path": str(resolve_from_root(OBJECT, root_kind="data")),
        "obj_scale": OBJECT_SCALE,
        "obj_pose": np.r_[object_position, [1.0, 0.0, 0.0, 0.0]],
        "joint_names": list(JOINTS),
        "pregrasp_qpos": np.r_[arm_pregrasp, hand_open],
        "grasp_qpos": np.r_[arm_grasp, hand_open],
        "squeeze_qpos": np.r_[arm_grasp, hand_close],
    })
    return validate_grasp_record(record, "CRX-Sharpa generated grasp", expected_joint_dim=28,
                                 expected_joint_names=JOINTS, require_joint_names=True, require_assets=True)


def create_simulation(control_model: Path, record: dict[str, object], *, viewer: bool = False) -> MjHO:
    """Build the project's MuJoCo hand-object scene around a generated grasp.

    Args:
        control_model: Generated MJCF with 28 position actuators.
        record: Common grasp record with object path and scale.
        viewer: Open a passive viewer when true.

    Returns:
        Initialized MuJoCo scene.
    """
    object_path = resolve_from_root(record["obj_path"], root_kind="data")
    info = json.loads((object_path / "info" / "simplified.json").read_text())
    coefficient = info["mass"] / (info["density"] * info["scale"] ** 3)
    density = OBJECT_MASS / (coefficient * float(record["obj_scale"]) ** 3)
    return MjHO(
        obj_path=str(object_path), obj_scale=record["obj_scale"], obj_density=density,
        hand_xml_path=str(control_model), hand_mocap=False, exclude_table_contact=None,
        friction_coef=[0.6, 0.02], has_floor_z0=True, debug_viewer=viewer,
    )


def run_phase1(run_dir: Path, *, viewer: bool = False) -> dict[str, object]:
    """Generate a model and grasp record, then replay fixed open/close targets.

    Args:
        run_dir: Fresh destination directory.
        viewer: Open an interactive MuJoCo viewer.

    Returns:
        Summary of the executed example and generated paths.
    """
    if run_dir.exists():
        raise FileExistsError(f"Example run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    model_path = build_control_model(run_dir / "control_model.xml")
    record = build_grasp_record()
    record_path = run_dir / "grasp.npy"
    np.save(record_path, record)
    simulation = create_simulation(model_path, record, viewer=viewer)
    try:
        simulation.get_contact_info(record["pregrasp_qpos"], record["obj_pose"])
        simulation.ctrl_qpos_a(JOINTS, record["pregrasp_qpos"])
        contacts = []
        peak_contacts = []
        for target in (record["grasp_qpos"], record["squeeze_qpos"]):
            start = simulation.get_qpos_a()
            peak = 0
            for q in np.linspace(start, target, 40):
                simulation.ctrl_qpos_a(JOINTS, q)
                simulation.control_hand_step(5)
                peak = max(peak, len(simulation.get_curr_contact_info()))
            peak_contacts.append(peak)
            contacts.append(len(simulation.get_curr_contact_info()))
        actual = simulation.get_qpos_f(JOINTS)
        summary = {
            "model": str(model_path), "grasp": str(record_path.resolve()),
            "joint_count": len(JOINTS), "actuator_count": simulation.model.nu,
            "contact_counts": contacts, "peak_contact_counts": peak_contacts,
            "final_joint_error_norm": float(np.linalg.norm(actual - record["squeeze_qpos"])),
            "object_pose": simulation.get_obj_pose().tolist(),
        }
        (run_dir / "phase1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        simulation.close_view_and_render()
