"""Experimental CRX-Sharpa execution through the project's ours controller."""

from __future__ import annotations

import json
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
import mujoco
import numpy as np

from ada_grasp_ctrl.crx_sharpa_contract import JOINTS, OBJECT_SCALE, PALM_FRAME
from ada_grasp_ctrl.crx_sharpa_demo import (
    configure_crx_sharpa_viewer,
    create_simulation,
    prepare_crx_sharpa_example,
    solve_arm_ik,
)
from ada_grasp_ctrl.paths import project_root, runtime_root
from ada_grasp_ctrl.schema import load_npy_record, validate_control_record, validate_grasp_record
from ada_grasp_ctrl.tasks.control_eval_func.base import BaseEval
from ada_grasp_ctrl.tasks.control_eval_func.optimized_runner import WrenchControlPolicy, run_wrench_control_episode
from ada_grasp_ctrl.utils.hand_util import MjHO


class CrxSharpaOursEval(BaseEval):
    """Adapt the maintained ours policy to one experimental CRX-Sharpa model."""

    def _initialize(self) -> None:
        """Initialize the maintained contact controller with URDF kinematics."""
        self._initialize_controller("ours")

    def _simulate_under_extforce_details(
        self, pregrasp_qpos: np.ndarray, grasp_qpos: np.ndarray, squeeze_qpos: np.ndarray
    ) -> None:
        """Run the shared Stage 1 and Stage 2 control policy.

        Args:
            pregrasp_qpos: Applied initial position.
            grasp_qpos: Open-hand grasp target.
            squeeze_qpos: Closed-hand target.
        """
        run_wrench_control_episode(
            self,
            pregrasp_qpos,
            grasp_qpos,
            squeeze_qpos,
            WrenchControlPolicy(use_approaching_arm_motion=True),
        )

    def _eval_simulate_under_extforce(self, obj_pose: np.ndarray, file_suffix: str):
        """Keep the live viewer open after this example episode finishes.

        Args:
            obj_pose: Initial object WXYZ pose.
            file_suffix: Output suffix for this episode.

        Returns:
            Saved control output path and episode status.
        """
        result = super()._eval_simulate_under_extforce(obj_pose, file_suffix)
        viewer = self.mj_ho.debug_viewer
        if self.configs.task.debug_viewer and viewer.native_is_running():
            print("Phase 2 finished. Close the viewer window to exit.", flush=True)
            while viewer.native_is_running():
                viewer.sync()
                time.sleep(0.05)
        return result

    def _lift_target_qpos(self, current_qpos_a: np.ndarray, lift_height: float) -> np.ndarray:
        """Lift the palm in Cartesian Z with a bounded CRX IK solve.

        Args:
            current_qpos_a: Current arm and hand actuator targets.
            lift_height: Requested lift in metres.

        Returns:
            Full actuator target with six new CRX arm joint positions.
        """
        self.robot_adaptor.compute_fk_a(current_qpos_a)
        target_pose = self.robot_adaptor.get_frame_pose(PALM_FRAME).copy()
        target_pose[2, 3] += lift_height
        target = current_qpos_a.copy()
        target[:6] = solve_arm_ik(self.robot_model, target_pose, target[:6])
        return target


def _config_for_example(run_dir: Path, model_path: Path, *, viewer: bool = False):
    """Compose a private hand configuration without changing public CLI support.

    Args:
        run_dir: Run directory containing generated control model and grasp input.
        model_path: Generated control MJCF.
        viewer: Whether to show the live native MuJoCo scene.

    Returns:
        Hydra configuration for one CRX-Sharpa control evaluation.
    """
    config_dir = project_root() / "src" / "ada_grasp_ctrl" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        configs = compose(config_name="base", overrides=["hand=crx_sharpa_left", "task=control_eval"])
    configs.hand_name = "crx_sharpa_left"
    configs.task_name = "control_eval"
    configs.asset_root = str(runtime_root("asset"))
    configs.data_root = str(runtime_root("data"))
    configs.output_root = str(runtime_root("output"))
    configs.save_dir = str(run_dir)
    configs.grasp_dir = str(run_dir)
    configs.control_dir = str(run_dir / "control")
    configs.log_dir = str(run_dir / "log" / "phase2")
    configs.hand.xml_path = str(model_path)
    # The maintained Stage-1 objective has no world-X/Y palm tracking weight.
    # Start at the grasp arm pose so the fixed demo's fingers can reach the object.
    configs.task.arm_pregrasp_is_grasp = True
    configs.task.debug_render = False
    configs.task.debug_viewer = viewer
    return configs


def run_phase2(run_dir: Path, *, viewer: bool = False) -> dict[str, object]:
    """Run an existing grasp input through coordinated contact control.

    Args:
        run_dir: Directory containing generated model and grasp input.
        viewer: Show the running simulation and hold its final frame.

    Returns:
        Current-run path, status, and contact summary.

    Raises:
        FileNotFoundError: If the model or grasp input is missing.
        FileExistsError: If Phase 2 was already attempted in the run directory.
    """
    model_path = run_dir / "control_model.xml"
    record_path = run_dir / "grasp.npy"
    if not model_path.is_file() or not record_path.is_file():
        raise FileNotFoundError(f"Control model and grasp are required below {run_dir}")
    summary_path = run_dir / "phase2_summary.json"
    if summary_path.exists() or (run_dir / "control").exists():
        raise FileExistsError(f"Phase 2 output already exists below {run_dir}")
    configs = _config_for_example(run_dir.resolve(), model_path.resolve(), viewer=viewer)
    started = time.monotonic()
    evaluator = CrxSharpaOursEval(str(record_path), configs)
    if viewer:
        try:
            configure_crx_sharpa_viewer(evaluator.mj_ho, evaluator.grasp_data["obj_pose"])
        except Exception:
            evaluator.mj_ho.close_view_and_render()
            raise
    outputs, statuses = evaluator.run()
    result = validate_control_record(load_npy_record(outputs[0]), outputs[0])
    contacts = result["contacts"]
    optimized_contact_steps = int(
        sum(
            bool(step_contacts)
            and len(control["cf"]) == 3 * len(step_contacts)
            and np.linalg.norm(np.asarray(control["q_a"]) - np.asarray(target)) > 1e-4
            for step_contacts, control, target in zip(contacts, result["opt_res"], result["planned_dof"])
        )
    )
    object_poses = np.asarray(result["obj_pose"])
    summary = {
        "run_dir": str(run_dir.resolve()),
        "control_output": outputs[0],
        "statuses": [status.value for status in statuses],
        "object_scale": float(load_npy_record(record_path)["obj_scale"]),
        "elapsed_seconds": time.monotonic() - started,
        "control_steps": len(result["stage"]),
        "contact_steps": sum(bool(step) for step in contacts),
        "optimized_contact_steps": optimized_contact_steps,
        "max_contacts": max((len(step) for step in contacts), default=0),
        "stage_values": sorted({int(stage) for stage in result["stage"]}),
        "solver_degraded": any(not entry["accepted"] for entry in result["solver_diagnostics"]),
        "object_lift_delta_m": (
            float(object_poses[-1, 2] - object_poses[0, 2]) if len(object_poses) >= 2 else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_phase2_example(
    run_dir: Path, *, object_scale: float = OBJECT_SCALE, viewer: bool = False
) -> dict[str, object]:
    """Generate independent grasp inputs and run adaptive Phase 2 directly.

    Args:
        run_dir: Fresh Phase 2 output directory.
        object_scale: Uniform physical scale for the bottle collision meshes.
        viewer: Show the running simulation and hold its final frame.

    Returns:
        Adaptive control summary from the new run.
    """
    prepare_crx_sharpa_example(run_dir, object_scale=object_scale)
    return run_phase2(run_dir, viewer=viewer)


def _apply_recorded_state(simulation: MjHO, joint_positions: np.ndarray, object_pose: np.ndarray) -> None:
    """Display one saved state without advancing MuJoCo physics.

    Args:
        simulation: CRX-Sharpa hand-object scene.
        joint_positions: Recorded positions in the exact 28-joint contract order.
        object_pose: Recorded world-space WXYZ object pose.
    """
    for name, position in zip(JOINTS, joint_positions):
        simulation.data.joint(f"{simulation.hand_prefix}{name}").qpos[0] = position
    simulation.data.joint("obj_freejoint").qpos[:] = object_pose
    simulation.data.qvel[:] = 0
    mujoco.mj_forward(simulation.model, simulation.data)


def replay_phase2(run_dir: Path, *, fps: float = 10.0) -> dict[str, object]:
    """Show the recorded Phase 2 control states in a native MuJoCo viewer.

    Args:
        run_dir: Existing run directory with model, grasp, and Phase 2 output.
        fps: Playback rate for the recorded control samples.

    Returns:
        Read-only replay summary after the viewer window closes.

    Raises:
        FileNotFoundError: If any required input is missing.
        ValueError: If playback rate or trajectory shape is invalid.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    model_path = run_dir / "control_model.xml"
    grasp_path = run_dir / "grasp.npy"
    control_path = run_dir / "control" / "ours_default" / "grasp_dist_0_pos_0.npy"
    for path in (model_path, grasp_path, control_path):
        if not path.is_file():
            raise FileNotFoundError(f"Phase 2 replay input is missing: {path}")
    grasp = validate_grasp_record(
        load_npy_record(grasp_path), grasp_path,
        expected_joint_dim=len(JOINTS), expected_joint_names=JOINTS,
        require_joint_names=True, require_assets=True,
    )
    result = validate_control_record(load_npy_record(control_path), control_path)
    positions = np.asarray(result.get("dof", []), dtype=float)
    object_poses = np.asarray(result["obj_pose"], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != len(JOINTS) or len(positions) == 0:
        raise ValueError(f"Phase 2 replay requires recorded (T, {len(JOINTS)}) joint positions")
    if object_poses.ndim != 2 or object_poses.shape[0] < len(positions):
        raise ValueError("Phase 2 replay requires an object pose for every recorded control step")

    simulation = create_simulation(model_path, grasp, viewer=True)
    frames_replayed = 0
    try:
        _apply_recorded_state(simulation, positions[0], object_poses[0])
        configure_crx_sharpa_viewer(simulation, object_poses[0])
        print(f"Replaying {len(positions)} recorded Phase 2 control steps at {fps:g} fps.", flush=True)
        for joint_positions, object_pose in zip(positions, object_poses):
            if not simulation.debug_viewer.native_is_running():
                break
            _apply_recorded_state(simulation, joint_positions, object_pose)
            simulation.debug_viewer.sync()
            frames_replayed += 1
            time.sleep(1.0 / fps)
        if simulation.debug_viewer.native_is_running():
            print("Replay finished. Close the viewer window to exit.", flush=True)
            while simulation.debug_viewer.native_is_running():
                simulation.debug_viewer.sync()
                time.sleep(0.05)
    finally:
        simulation.close_view_and_render()
    return {
        "control_output": str(control_path.resolve()),
        "frames_replayed": frames_replayed,
        "lift_replayed": False,
    }
