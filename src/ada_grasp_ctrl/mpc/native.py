"""Externally clocked original Ours policy, with no simulator/actuator writes."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from ada_grasp_ctrl.contact import ContactObservation
from ada_grasp_ctrl.experiments.right_full_control import Controller, validate_command
from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES
from ada_grasp_ctrl.tasks.control_eval_func.episode_runner import EpisodeStep
from .contracts import ACTIVE_STAGES, JointFeedback, MPCProposal, fresh, named_vector


class NativeMPC:
    """Reuse native force/stiffness/solver policies; reject failed proposals externally.

    The native policy's configured failure decision is retained in diagnostics.
    The execution boundary never approves a rejected candidate, even when native
    simulation is configured to record/apply that candidate. Rejection latches
    this runtime to avoid continuing with history from an unapplied command.
    """

    def __init__(self, runner: Any, *, clock: Callable[[], float] = time.monotonic, deadline_s: float = 0.1) -> None:
        if tuple(runner.robot.dof_names) != tuple(JOINT_NAMES):
            raise ValueError("MPC requires canonical real CRX + Sharpa joints")
        if not np.array_equal(runner.robot.doa2dof_matrix, np.eye(28)):
            raise ValueError("MPC requires independent named 28-axis mapping")
        if not np.isfinite(deadline_s) or deadline_s <= 0 or deadline_s > 0.1:
            raise ValueError("MPC deadline must be within the 100 ms period")
        self.runner, self.clock, self.deadline_s = runner, clock, deadline_s
        self.controller = Controller("ours")
        self.controller.reset(runner)
        self.index = 0
        self.last_tick: float | None = None
        self.last_feedback: float | None = None
        self.fault: str | None = None
        self.completed = False

    def tick(
        self,
        now: float,
        feedback: JointFeedback,
        q_cmd: tuple[float, ...],
        q_ref: tuple[float, ...],
        observation: ContactObservation,
        *,
        stage: str,
    ) -> MPCProposal:
        """Compute one absolute target using measured q and independently supplied q_cmd."""
        started = self.clock()
        diagnostics: dict = {"stage": stage, "index": self.index, "command_applied": False}
        candidate = None
        status = "unavailable"
        try:
            if self.fault:
                raise ValueError(self.fault)
            if stage not in ACTIVE_STAGES:
                raise ValueError("MPC may only run in grasp/squeeze")
            if self.last_tick is not None and now - self.last_tick < 0.1 - 1e-9:
                raise ValueError("MPC tick is earlier than 100 ms")
            if self.last_tick is not None and now - self.last_tick > 0.2 + 1e-9:
                raise ValueError("MPC tick deadline missed")
            fresh(feedback.source_timestamp, feedback.receive_timestamp, now, 0.2)
            if self.last_feedback is not None and feedback.source_timestamp <= self.last_feedback:
                raise ValueError("Repeated or reordered MPC feedback")
            self.last_tick, self.last_feedback = now, feedback.source_timestamp
            measured = np.asarray(feedback.positions_rad)
            commanded = np.asarray(named_vector(feedback.joint_names, q_cmd))
            reference = np.asarray(named_vector(feedback.joint_names, q_ref))
            limits = self.runner.robot_adaptor.joint_limits_f
            for q in (measured, commanded, reference):
                if np.any(q < limits[0] - 1e-8) or np.any(q > limits[1] + 1e-8):
                    raise ValueError("MPC input joint-limit violation")
            fresh(observation.timestamp, observation.timestamp, now, 0.2)
            contacts = observation.controller_contacts(observation.timestamp)
            if any(c["hand_side"] != "right" for c in contacts):
                raise ValueError("MPC received a non-right contact")
            status = "valid_contacts" if contacts else "valid_empty"
            forces = np.asarray([c["contact_force"][:3] for c in contacts]).reshape(-1, 3)
            diagnostics.update(
                contact_count=len(contacts),
                contact_source=observation.source,
                contact_timestamp=observation.timestamp,
                measured_q=measured.tolist(),
                q_cmd=commanded.tolist(),
                q_ref=reference.tolist(),
            )
            state = EpisodeStep(
                self.index,
                min(self.index, len(self.runner.qpos_path) - 1),
                reference,
                measured,
                commanded,
                contacts,
                np.full(7, np.nan),
                forces,
                float(forces[:, 0].sum()),
            )
            # Object pose is not consumed by this policy. Do not fabricate a measured pose.
            should_stop = self.controller.should_stop(state)
            max_steps = self.controller.max_steps()
            diagnostics.update(
                native_stage=self.controller.policy.stage,
                reference_index=min(self.index, len(self.runner.qpos_path) - 1),
                path_length=len(self.runner.qpos_path),
                current_sum_force=state.current_sum_force,
                final_sum_force=self.controller.policy.final_sum_force,
                should_stop=should_stop,
                max_steps=max_steps,
                remaining_steps=max(0, max_steps - self.index),
            )
            if should_stop or self.completed:
                self.completed = True
                diagnostics["termination"] = "native_complete"
                return MPCProposal(
                    tuple(JOINT_NAMES), None, False, now, max(0.0, self.clock() - started), status, diagnostics
                )
            if self.index >= max_steps:
                diagnostics["termination"] = "native_timeout"
                raise ValueError("Ours native timeout: native termination not achieved")
            offset = len(self.runner.grasp_ctrl.r_data["solver_diagnostics"])
            control = self.controller.step(state, observation, 0.1, timestamp=observation.timestamp)
            native = self.runner.grasp_ctrl.r_data["solver_diagnostics"][offset:]
            diagnostics.update(control.diagnostics)
            diagnostics["native_stage"] = self.controller.policy.stage
            diagnostics["desired_sum_force"] = self.controller.policy.desired_sum_force
            diagnostics["solver_outcomes"] = native
            if any(not d["accepted"] for d in native):
                raise ValueError("Native solver rejected/degraded; execution boundary rejects candidate")
            validate_command(control.target_qpos_a, commanded, limits, np.asarray(self.runner.robot.doa_max_vel))
            candidate = tuple(control.target_qpos_a)
            if self.clock() - started > self.deadline_s:
                raise ValueError("MPC solve deadline missed")
            self.index += 1
        except Exception as error:
            self.fault = f"{type(error).__name__}: {error}"
            diagnostics["reason"] = self.fault
            candidate = None
            status = "rejected"
        return MPCProposal(
            tuple(JOINT_NAMES),
            candidate,
            candidate is not None,
            now,
            max(0.0, self.clock() - started),
            status,
            diagnostics,
        )
