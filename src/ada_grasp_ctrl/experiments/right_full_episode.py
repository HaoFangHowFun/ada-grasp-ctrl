"""Real CRX backend using native episode/approach/lift and replaceable observations."""

from __future__ import annotations

import numpy as np

from ada_grasp_ctrl.contact import ContactObservation, ContactSource
from ada_grasp_ctrl.contact.mujoco_adapter import MuJoCoContactAdapter
from ada_grasp_ctrl.tasks.control_eval_func.episode_runner import DummyArmEpisodeRunner, EpisodeStep
from .crx_sharpa_right_evaluation import RightEvaluator, ApproachInvalidError
from .crx_sharpa_right_model import JOINTS
from .right_full_control import CONTROL_DT, Controller, linear_reference, physics_substeps, validate_command


class RightFullEvaluator(RightEvaluator):
    """Adapt existing real-arm execution; no dummy-arm model is constructed."""

    def __init__(
        self, row: dict, report: dict, config: object, method: str, *, contact_source: ContactSource | None = None
    ) -> None:
        self.controller = Controller(config.controller.mode)
        super().__init__(row, report, config, method)
        self.sim_step_per_action = physics_substeps(self.mj_ho.model.opt.timestep, config.controller.ours.frequency_hz)
        self.contacts = (
            contact_source
            if contact_source is not None
            else MuJoCoContactAdapter(
                self.mj_ho.model, self.mj_ho.data, hand_bodies=self.mj_ho.hand_bodies, object_bodies={"object"}
            )
        )
        self.ticks: list[dict] = []
        self._solver_offset = 0
        self.completion: dict = {}
        self.limits = self.robot_adaptor.joint_limits_f
        self.velocity = np.asarray(self.robot.doa_max_vel)
        if self.robot.dof_names != list(JOINTS) or not np.array_equal(self.robot.doa2dof_matrix, np.eye(28)):
            raise ValueError("Original contact stiffness requires this exact independent 28-axis mapping")
        model = self.mj_ho.model
        actuator_ids = self.mj_ho.a_addresses
        if not np.allclose(model.actuator_gainprm[actuator_ids, 0], self.robot.doa_kp, atol=1e-12, rtol=0):
            raise ValueError("MPC stiffness gains differ from MuJoCo actuator gains")
        if not np.allclose(model.actuator_ctrlrange[actuator_ids], self.limits.T, atol=1e-6, rtol=0):
            raise ValueError("Actuator bounds differ from canonical joint limits")

    def motion(self, path: np.ndarray, segment: str) -> None:
        """Retain existing approach/lift targets and inner holds with validated timing."""
        self.mj_ho.segment = segment
        for target in path:
            previous = self.mj_ho.get_qpos_a()
            validate_command(target, previous, self.limits, self.velocity)
            observation = self.contacts.observe()
            self.mj_ho.ctrl_qpos_a_with_interp(
                previous, target, self.robot.doa_names, step_outer=self.sim_step_per_action // 5, step_inner=5
            )
            self.episode_recorder.capture(segment, reference=target, command=target)
            self.record_tick(observation, target, target, segment, {})

    def record_tick(
        self, observation: ContactObservation, reference: np.ndarray, command: np.ndarray, stage: str, diag: dict
    ) -> None:
        """Record timestamped observations and post-action state with explicit alignment."""
        forces = np.asarray([c["contact_force"] for c in observation.contacts]).reshape(-1, 6)
        measured = self.mj_ho.get_qpos_f(JOINTS)
        native = self.grasp_ctrl.r_data["solver_diagnostics"][self._solver_offset :] if diag else []
        self._solver_offset = len(self.grasp_ctrl.r_data["solver_diagnostics"])
        self.ticks.append(
            {
                "control_step": len(self.ticks),
                "timestamp": observation.timestamp,
                "state_timestamp": float(self.mj_ho.data.time),
                "controller_mode": self.controller.mode,
                "stage": stage,
                "command_applied": True,
                "mpc_stage": diag.get("stage"),
                "reference_joint_position": reference.copy(),
                "commanded_actuator_position": command.copy(),
                "measured_joint_position": measured,
                "joint_tracking_error": measured - reference,
                "object_pose": self.mj_ho.get_obj_pose(),
                "object_displacement": self.mj_ho.get_obj_pose()[:3] - self.grasp_data["obj_pose"][:3],
                "contacts": list(observation.contacts),
                "contact_count": len(forces),
                "environment_contacts": list(observation.environment_contacts),
                "max_contact_force": float(np.linalg.norm(forces[:, :3], axis=1).max()) if len(forces) else 0.0,
                "sum_normal_force": float(forces[:, 0].sum()),
                "wrench_balance": diag.get("balance_metric"),
                "solver_time_ms": None if "t_ctrl_opt" not in diag else 1000 * diag["t_ctrl_opt"],
                "solver_status": None
                if "t_ctrl_opt" not in diag
                else ("degraded" if self.grasp_ctrl.solver_degraded else "success"),
                "solver_outcome": native,
            }
        )

    def native_grasp(self) -> None:
        """Run the same native linear reference and MPC policy on fresh observations."""
        sim = self.mj_ho
        sim.segment = "control"
        # A moving object invalidates a frozen pregrasp, even without premature contact.
        # Use the existing IK position/angle acceptance tolerances, without moving targets.
        from .crx_sharpa_right_preparation import POSITION_TOLERANCE, ANGLE_TOLERANCE, residual
        from .crx_sharpa_right_model import pose_matrix

        drift = residual(pose_matrix(sim.get_obj_pose()), pose_matrix(self.grasp_data["obj_pose"]))
        if drift["position_m"] > POSITION_TOLERANCE or drift["orientation_rad"] > ANGLE_TOLERANCE:
            raise ApproachInvalidError(f"Object moved before pregrasp: {drift}")
        runner = DummyArmEpisodeRunner(self, self.grasp_data["grasp_qpos"], self.grasp_data["squeeze_qpos"])
        # The original helper is already np.linspace; prove its contract independently.
        start = sim.get_qpos_f(JOINTS)
        for path, end in (
            (runner.path_approach, self.grasp_data["grasp_qpos"]),
            (runner.path_squeeze, self.grasp_data["squeeze_qpos"]),
        ):
            expected = np.array([linear_reference(start, end, (i + 1) * CONTROL_DT, 2.0) for i in range(20)])
            if not np.allclose(path, expected, atol=1e-14, rtol=0):
                raise ValueError("Native joint reference no longer follows linear interpolation")
            start = end
        self.controller.reset(runner)
        stop = "timeout"
        for index in range(self.controller.max_steps()):
            waypoint = min(index, len(runner.qpos_path) - 1)
            observation = self.contacts.observe()
            contacts = observation.controller_contacts(float(sim.data.time))
            forces = np.asarray([c["contact_force"][:3] for c in contacts]).reshape(-1, 3)
            state = EpisodeStep(
                index,
                waypoint,
                runner.qpos_path[waypoint],
                sim.get_qpos_f(JOINTS),
                sim.get_qpos_a(),
                contacts,
                sim.get_obj_pose(),
                forces,
                float(forces[:, 0].sum()),
            )
            if self.controller.should_stop(state):
                stop = "native_force_stop"
                break
            try:
                command = self.controller.step(state, observation, self.action_dt, timestamp=float(sim.data.time))
                validate_command(command.target_qpos_a, state.current_qpos_a, self.limits, self.velocity)
            except Exception as error:
                self.record_tick(observation, state.target_qpos_f, state.current_qpos_a, "controller_error", {})
                self.ticks[-1].update(
                    solver_status="failed",
                    error=f"{type(error).__name__}: {error}",
                    command_applied=False,
                    solver_outcome=self.grasp_ctrl.r_data["solver_diagnostics"].copy(),
                )
                raise
            runner.apply_control(state, command)
            self.record_tick(
                observation,
                state.target_qpos_f,
                command.target_qpos_a,
                "grasp" if waypoint < len(runner.path_approach) else "squeeze",
                command.diagnostics,
            )
            if sim.collision_history:
                raise ApproachInvalidError("Collision gate exceeded during grasp; lift skipped")
        else:
            if self.controller.mode == "linear":
                stop = "reference_complete"
        self.completion = {
            "termination": stop,
            "reference_complete": index + 1 >= len(runner.qpos_path),
            "terminal_joint_error_rad": (sim.get_qpos_f(JOINTS) - runner.qpos_path[-1]).tolist(),
            "physical_success_evaluated_by": "existing Cartesian lift criterion",
        }
