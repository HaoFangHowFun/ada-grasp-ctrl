"""Experimental single-arm CRX-5iA and Sharpa grasp example utilities."""

from __future__ import annotations

import json
import time
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


def build_grasp_record(*, object_scale: float = OBJECT_SCALE) -> dict[str, object]:
    """Create a reproducible reach, open, and close grasp for the bundled bottle.

    Args:
        object_scale: Uniform physical scale for the bottle collision meshes.

    Returns:
        Validated 28-joint common grasp record.

    Raises:
        ValueError: If the requested bottle scale is invalid.
    """
    if not np.isfinite(object_scale) or object_scale <= 0:
        raise ValueError("object_scale must be a positive finite number")
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
    # Move a larger bottle outside the open fingers at initialization. The
    # three-times-scale example uses a 2.5 cm X and 22.5 cm Y clearance shift.
    size_growth = max(0.0, object_scale / OBJECT_SCALE - 1.0)
    object_position[:2] += size_growth * np.array([0.0125, 0.1125])
    # The bundled bottle's collider extends approximately 0.35 scale units
    # below its origin; keep the bottom near the tabletop when resizing it.
    object_position[2] += 0.35 * (object_scale - OBJECT_SCALE)
    record = with_current_schema({
        "obj_path": str(resolve_from_root(OBJECT, root_kind="data")),
        "obj_scale": object_scale,
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


def configure_crx_sharpa_viewer(simulation: MjHO, object_pose: np.ndarray) -> None:
    """Frame the robot and bottle while drawing only their visual meshes.

    Args:
        simulation: Scene with an attached native MuJoCo viewer.
        object_pose: Initial world-space WXYZ object pose.
    """
    # Robot collision geoms use group 0; keep the bottle's visual geoms
    # visible when hiding that group in the native viewer.
    for geom_id in range(simulation.model.ngeom):
        name = simulation.model.geom(geom_id).name
        if name.startswith("object_visual_"):
            simulation.model.geom_group[geom_id] = 1
            collision_id = simulation.model.geom(name.replace("object_visual_", "object_collision_", 1)).id
            simulation.model.geom_rgba[geom_id] = simulation.model.geom_rgba[collision_id]
    simulation.debug_viewer.set_native_geom_group(0, False)
    object_position = np.asarray(object_pose[:3])
    simulation.debug_viewer.set_native_camera(
        lookat=[0.5 * object_position[0], 0.5 * object_position[1], 0.3],
        distance=1.5,
        azimuth=125,
        elevation=-22,
    )


def prepare_crx_sharpa_example(
    run_dir: Path, *, object_scale: float = OBJECT_SCALE
) -> tuple[Path, Path, dict[str, object]]:
    """Create the shared model and grasp input for an independent example run.

    Args:
        run_dir: Fresh destination directory.
        object_scale: Uniform physical scale for the bottle collision meshes.

    Returns:
        Control model path, grasp record path, and validated grasp record.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If the object scale is invalid.
    """
    if not np.isfinite(object_scale) or object_scale <= 0:
        raise ValueError("object_scale must be a positive finite number")
    if run_dir.exists():
        raise FileExistsError(f"Example run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    model_path = build_control_model(run_dir / "control_model.xml")
    record = build_grasp_record(object_scale=object_scale)
    record_path = run_dir / "grasp.npy"
    np.save(record_path, record)
    return model_path, record_path, record


def run_phase1(run_dir: Path, *, viewer: bool = False, object_scale: float = OBJECT_SCALE) -> dict[str, object]:
    """Run fixed open/close joint targets without contact adaptation.

    Args:
        run_dir: Fresh destination directory.
        viewer: Replay in a MuJoCo window and wait until it is closed.
        object_scale: Uniform physical scale for the bottle collision meshes.

    Returns:
        Summary of the executed fixed-target example.
    """
    model_path, record_path, record = prepare_crx_sharpa_example(run_dir, object_scale=object_scale)
    simulation = create_simulation(model_path, record, viewer=viewer)
    try:
        simulation.get_contact_info(record["pregrasp_qpos"], record["obj_pose"])
        if viewer:
            configure_crx_sharpa_viewer(simulation, record["obj_pose"])
        simulation.ctrl_qpos_a(JOINTS, record["pregrasp_qpos"])
        contacts = []
        peak_contacts = []
        for target in (record["grasp_qpos"], record["squeeze_qpos"]):
            start = simulation.get_qpos_a()
            peak = 0
            for q in np.linspace(start, target, 40):
                simulation.ctrl_qpos_a(JOINTS, q)
                simulation.control_hand_step(5)
                if viewer:
                    # MjHO sleeps for one physics step; pace the remaining four.
                    time.sleep(4 * simulation.model.opt.timestep)
                peak = max(peak, len(simulation.get_curr_contact_info()))
            peak_contacts.append(peak)
            contacts.append(len(simulation.get_curr_contact_info()))
        actual = simulation.get_qpos_f(JOINTS)
        summary = {
            "model": str(model_path), "grasp": str(record_path.resolve()),
            "joint_count": len(JOINTS), "actuator_count": simulation.model.nu,
            "object_scale": object_scale,
            "contact_counts": contacts, "peak_contact_counts": peak_contacts,
            "final_joint_error_norm": float(np.linalg.norm(actual - record["squeeze_qpos"])),
            "object_pose": simulation.get_obj_pose().tolist(),
        }
        (run_dir / "phase1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if viewer:
            print("Phase 1 finished. Close the viewer window to exit.", flush=True)
            while simulation.debug_viewer.native_is_running():
                simulation.debug_viewer.sync()
                time.sleep(0.05)
        return summary
    finally:
        simulation.close_view_and_render()
