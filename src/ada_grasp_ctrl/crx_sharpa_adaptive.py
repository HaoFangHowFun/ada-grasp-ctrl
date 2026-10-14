"""Experimental CRX-Sharpa execution through the project's ours controller."""

from __future__ import annotations

import json
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np

from ada_grasp_ctrl.crx_sharpa_contract import PALM_FRAME
from ada_grasp_ctrl.crx_sharpa_demo import solve_arm_ik
from ada_grasp_ctrl.paths import project_root, runtime_root
from ada_grasp_ctrl.schema import load_npy_record, validate_control_record
from ada_grasp_ctrl.tasks.control_eval_func.base import BaseEval
from ada_grasp_ctrl.tasks.control_eval_func.optimized_runner import WrenchControlPolicy, run_wrench_control_episode


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


def _config_for_example(run_dir: Path, model_path: Path):
    """Compose a private hand configuration without changing public CLI support.

    Args:
        run_dir: Completed Phase 1 run directory.
        model_path: Generated control MJCF.

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
    configs.task.debug_viewer = False
    return configs


def run_phase2(run_dir: Path) -> dict[str, object]:
    """Run an existing Phase 1 grasp through coordinated contact control.

    Args:
        run_dir: Completed Phase 1 output directory.

    Returns:
        Current-run path, status, and contact summary.

    Raises:
        FileNotFoundError: If Phase 1 output is incomplete.
        FileExistsError: If Phase 2 was already attempted in the run directory.
    """
    model_path = run_dir / "control_model.xml"
    record_path = run_dir / "grasp.npy"
    if not model_path.is_file() or not record_path.is_file():
        raise FileNotFoundError(f"Phase 1 model and grasp are required below {run_dir}")
    summary_path = run_dir / "phase2_summary.json"
    if summary_path.exists() or (run_dir / "control").exists():
        raise FileExistsError(f"Phase 2 output already exists below {run_dir}")
    configs = _config_for_example(run_dir.resolve(), model_path.resolve())
    started = time.monotonic()
    outputs, statuses = CrxSharpaOursEval(str(record_path), configs).run()
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
        "control_output": outputs[0],
        "statuses": [status.value for status in statuses],
        "elapsed_seconds": time.monotonic() - started,
        "control_steps": len(result["stage"]),
        "contact_steps": sum(bool(step) for step in contacts),
        "optimized_contact_steps": optimized_contact_steps,
        "max_contacts": max((len(step) for step in contacts), default=0),
        "stage_values": sorted({int(stage) for stage in result["stage"]}),
        "solver_degraded": any(not entry["accepted"] for entry in result["solver_diagnostics"]),
        "object_lift_delta_m": float(object_poses[-1, 2] - object_poses[0, 2]),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
