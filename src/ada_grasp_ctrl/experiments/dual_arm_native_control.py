"""Named active-arm adapters for the existing native coordinated grasp controller."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pinocchio as pin
from omegaconf import OmegaConf

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.tasks.control_eval_func.optimized_runner import WrenchControlPolicy, run_wrench_control_episode
from ada_grasp_ctrl.utils.grasp_controller import GraspController
from ada_grasp_ctrl.utils.hand_util import MjHO
from ada_grasp_ctrl.utils.pin_helper import PinocchioHelper
from ada_grasp_ctrl.utils.robot_adaptor import RobotAdaptor
from ada_grasp_ctrl.utils.robots.base import Arm, ArmHand, Hand
from .phase_comparison import build_config
from .simulated_object_cloud import apply_arm_bias_compensation


class ActiveKinematics(PinocchioHelper):
    """Keep world mounts and active frames while locking the other arm for control only.

    The full dual-arm MuJoCo model remains dynamic. Reducing this separate
    kinematics model satisfies the native adaptor's exact joint-name contract.
    """

    def __init__(self, robot: object, side: str, q56: np.ndarray) -> None:
        reference = pin.neutral(robot.pin_model)
        reference[robot.pin_q] = robot.configuration(q56)
        passive = side_slice("left" if side == "right" else "right")
        locked = [robot.pin_model.getJointId(n) for n in robot.joint_names[passive]]
        self.model = pin.buildReducedModel(robot.pin_model, locked, reference)
        self.data = self.model.createData()


class ActiveRobot(ArmHand):
    """Read active actuator gains and names from the already compiled scene."""

    def __init__(self, scene: object, side: str) -> None:
        super().__init__("")
        names = list(scene.robot.joint_names[side_slice(side)])
        self.arm, self.hand = Arm(""), Hand("")
        for part, subset in ((self.arm, names[:6]), (self.hand, names[6:])):
            part._dof_names = subset
            part._doa_names = [f"{n}_position" for n in subset]
            part._doa2dof_matrix = np.eye(len(subset))
            part._doa_kp = [float(scene.sim.model.actuator(n).gainprm[0]) for n in part._doa_names]
            part._body_names = []
        self.arm._doa_max_vel = scene.robot.pin_model.velocityLimit[scene.robot.pin_v[side_slice(side, True)]].tolist()
        self.hand._doa_max_vel = [2.0] * 22  # Historical Sharpa controller velocity bound.
        self.arm._base_name = f"{side}_base_link"
        self.hand._base_name = f"{side}_hand_C_MC"
        self._urdf_path = str(scene.robot.urdf_path)


class ActiveSimulation(MjHO):
    """Use native canonical contacts/interpolation on the same live dual-arm scene.

    No scene rebuild, reset, object force override, or gravity modification occurs.
    The callback validates and records every stepped state in the enclosing episode.
    """

    hand_prefix = ""
    hand_mocap = False

    def __init__(self, scene: object, side: str, after_step: Callable[[], None], *, compensate: bool) -> None:
        self.sim, self.side = scene.sim, side
        self.model, self.data = self.sim.model, self.sim.data
        self.block = side_slice(side)
        self.after_step, self.compensate = after_step, compensate
        self.target_qpos_a = self.sim.targets[self.block].copy()
        self.actuator_names = [f"{n}_position" for n in scene.robot.joint_names[self.block]]
        self.object_address = int(self.model.joint("object_free").qposadr[0])
        if self.model.body("object").id != self.model.nbody - 1:
            raise ValueError("Native contact extraction requires object as the last body")
        self.hand_bodies = {self.model.body(i).name for i in self.sim.hand_body_ids if self.sim.body_labels[i] == side}
        self.hand_contact_prefixes = tuple(sorted(self.hand_bodies))
        self.control_action = None
        self.initializing_control = False
        self.initial_command_projection = None
        self.command_roundoff_projections = []
        self.soft_limit_tolerance = float(scene.config["execution"]["soft_limit_tolerance_rad"])
        self.phase = "pregrasp"
        self.debug_viewer = self.debug_render = None

    def reset_pose_qpos(self, *args: object, **kwargs: object) -> None:
        """Disallow accidental native initialization from teleporting the live scene."""
        raise RuntimeError("Reset is forbidden during a dual-arm episode")

    def get_obj_pose(self) -> np.ndarray:
        """Return the live free object's XYZ/WXYZ pose."""
        return self.data.qpos[self.object_address : self.object_address + 7].copy()

    def get_curr_contact_info(self) -> list[dict]:
        """Retain native six-dimensional canonical wrenches for the active hand only."""
        return [c for c in super().get_curr_contact_info() if c["body1_name"] in self.hand_bodies]

    def ctrl_qpos_a(self, names: list[str], q_a: np.ndarray) -> None:
        """Validate named commands and preserve all passive-side targets."""
        if list(names) != self.actuator_names:
            raise ValueError("Native active actuator order differs from the compiled scene")
        if self.initializing_control:
            measured = self.sim.configuration()[self.block]
            value = np.asarray(q_a, dtype=float)
            if (
                value.shape != (28,)
                or not np.isfinite(value).all()
                or not np.allclose(value, measured, atol=1e-12, rtol=0)
            ):
                raise ValueError("Native initialization requires the current measured configuration")
            bounds = self.sim.robot.limits[self.block]
            bounded = np.clip(value, bounds[:, 0], bounds[:, 1])
            projection = float(np.max(np.abs(bounded - value)))
            if projection > self.soft_limit_tolerance:
                raise ValueError("Native initialization exceeds the existing soft-limit tolerance")
            self.initial_command_projection = dict(
                reason="Project measured soft-limit overshoot into command limits; measured state unchanged",
                maximum_projection_rad=projection,
                measured_rad=value.tolist(),
                command_rad=bounded.tolist(),
            )
            q_a = bounded
            self.initializing_control = False
        # SLSQP and interpolation can land a few machine ulps outside an exact
        # bound. Project only floating-point roundoff; larger violations still fail.
        value = np.asarray(q_a, dtype=float)
        if value.shape == (28,) and np.isfinite(value).all():
            bounds = self.sim.robot.limits[self.block]
            bounded = np.clip(value, bounds[:, 0], bounds[:, 1])
            delta = np.abs(value - bounded)
            roundoff = 64 * np.finfo(float).eps * np.maximum(1, np.max(np.abs(bounds), axis=1))
            if np.any(delta > 0) and np.all(delta <= roundoff):
                self.command_roundoff_projections.append(
                    dict(
                        time_s=float(self.data.time),
                        phase=self.phase,
                        maximum_projection_rad=float(np.max(delta)),
                        joints=[
                            dict(name=n, requested_rad=float(a), applied_rad=float(b))
                            for n, a, b in zip(names, value, bounded)
                            if a != b
                        ],
                    )
                )
                q_a = bounded
        self.sim.set_target(q_a, side=self.side)
        self.target_qpos_a = np.asarray(q_a).copy()

    def ctrl_qpos_a_with_interp(
        self,
        hand_qpos1: np.ndarray,
        hand_qpos2: np.ndarray,
        names: list[str],
        step_outer: int = 10,
        step_inner: int = 10,
    ) -> None:
        """Label native reference stages without changing the shared runner's progression."""
        if self.control_action is not None:
            self.phase = "grasp" if self.control_action < 20 else "squeeze"
            self.control_action += 1
        super().ctrl_qpos_a_with_interp(hand_qpos1, hand_qpos2, names, step_outer, step_inner)

    def control_hand_step(self, step_inner: int) -> None:
        """Step with the existing arm compensation and full-scene diagnostic callback."""
        for _ in range(step_inner):
            if self.compensate:
                apply_arm_bias_compensation(self.sim)
            self.sim.step()
            self.after_step()


class NativeEvaluator:
    """Supply the existing native runner with side-specific robot and simulator adapters."""

    def __init__(self, scene: object, side: str, adapter: ActiveSimulation, output: Path) -> None:
        self.robot = ActiveRobot(scene, side)
        self.robot_model = ActiveKinematics(scene.robot, side, scene.sim.targets)
        self.robot_adaptor = RobotAdaptor(
            self.robot_model, self.robot.dof_names, self.robot.doa_names, self.robot.doa2dof_matrix
        )
        self.configs = build_config("control_eval", "dummy_arm_sharpa_right", output, project_root(), seed=12)
        self.configs.task.method = "ours"
        self.configs.task.arm_pregrasp_is_grasp = False
        self.configs.task.graspdata.pregrasp_t = self.configs.task.graspdata.squeeze_t = 0
        self.configs.task.miu_coef = list(scene.config["object"]["physics"]["friction"][:2])
        self.grasp_ctrl = GraspController(self.configs.task.control, self.robot, self.robot_adaptor)
        self.mj_ho = adapter
        self.ctrl_freq, self.action_dt = 10, 0.1
        self.sim_step_per_action = round(self.action_dt / adapter.model.opt.timestep)
        if self.sim_step_per_action % 5 or not np.isclose(
            self.sim_step_per_action * adapter.model.opt.timestep, self.action_dt
        ):
            raise ValueError("Native 10 Hz control requires an integral number of physics steps divisible by five")
        # Save only the consumed controller settings: dummy-arm asset paths are not used.
        OmegaConf.save(self.configs.task.control, output / "native_controller.yaml", resolve=True)

    def _dof_data2user(self, qpos: np.ndarray) -> np.ndarray:
        """Targets already use the explicitly validated active canonical order."""
        if np.shape(qpos) != (28,) or not np.isfinite(qpos).all():
            raise ValueError("Expected a finite canonical 28-joint target")
        return np.asarray(qpos).copy()

    def grasp(self, chosen: dict) -> None:
        """Run the unchanged coordinated controller and shared lifecycle."""
        self.mj_ho.control_action = 0
        self.mj_ho.initializing_control = True
        try:
            values = [
                np.asarray(chosen["stages"][s]["q56"])[self.mj_ho.block] for s in ("pregrasp", "grasp", "squeeze")
            ]
            run_wrench_control_episode(self, *values, WrenchControlPolicy(use_approaching_arm_motion=True))
        finally:
            self.mj_ho.control_action = None
            self.mj_ho.initializing_control = False
