"""Linear and original Ada policy adapters for the real right CRX experiment."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ada_grasp_ctrl.mpc.reference import linear_reference as sample_linear

from ada_grasp_ctrl.contact import ContactObservation
from ada_grasp_ctrl.tasks.control_eval_func.episode_runner import EpisodeStep, StepControl
from ada_grasp_ctrl.tasks.control_eval_func.optimized_runner import _CoordinatedWrenchPolicy, WrenchControlPolicy

MODES = ("linear", "ours")
CONTROL_DT = 0.1


def physics_substeps(physics_dt: float, frequency_hz: float = 10) -> int:
    """Require exact 10 Hz updates and the native five-physics-step inner hold."""
    if frequency_hz != 10 or not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("right_full requires 10 Hz and a positive finite physics timestep")
    count = round(CONTROL_DT / physics_dt)
    if count < 1 or not np.isclose(count * physics_dt, CONTROL_DT, atol=1e-12, rtol=0):
        raise ValueError("0.1 s control period is not representable by this physics timestep")
    if count % 5:
        raise ValueError("Control period cannot preserve the native five-step inner actuator hold")
    return count


def linear_reference(start: np.ndarray, end: np.ndarray, elapsed: float, duration: float) -> np.ndarray:
    """Sample q0 + clip(t/T, 0, 1) * (qf-q0), with no smoothing."""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite([*a, *b, elapsed, duration]).all() or duration <= 0:
        raise ValueError("Invalid linear segment")
    return np.asarray(sample_linear(a, b, elapsed, duration))


def validate_command(command: np.ndarray, previous: np.ndarray, limits: np.ndarray, velocity: np.ndarray) -> None:
    """Reject malformed targets and position/speed violations before actuator writes."""
    q = np.asarray(command, dtype=float)
    if q.shape != previous.shape or not np.isfinite(q).all():
        raise ValueError("Invalid controller command: wrong shape or nonfinite values")
    if np.any(q < limits[0] - 1e-8) or np.any(q > limits[1] + 1e-8):
        raise ValueError("Controller command violates joint limits")
    if np.any(np.abs(q - previous) > velocity * CONTROL_DT + 1e-8):
        raise ValueError("Controller command violates joint velocity limits")


class Controller:
    """Small state/reference interface around the original coordinated policy.

    The runner supplies robot/controller configuration at reset. step only returns
    a command; actuator writes belong to the execution backend. Original policy
    owns stage switching, force trajectory, optimization and solver-failure policy.
    """

    def __init__(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown controller mode {mode!r}; expected linear or ours")
        self.mode = mode
        self.policy = _CoordinatedWrenchPolicy(WrenchControlPolicy(True)) if mode == "ours" else None
        self._diagnostics: dict = {}

    def reset(self, runner: Any) -> None:
        """Bind one episode's trajectory and native controller without global state."""
        self.runner = runner
        self._diagnostics = {}
        if self.policy is not None:
            self.policy.initialize(runner)

    def max_steps(self) -> int:
        """Retain original MPC timeout; linear ends at the final reference time."""
        return self.policy.max_steps(self.runner) if self.policy else len(self.runner.qpos_path)

    def should_stop(self, state: EpisodeStep) -> bool:
        """Retain original force-based stop separately from physical grasp success."""
        return bool(self.policy and self.policy.should_stop(self.runner, state))

    def step(self, state: EpisodeStep, observation: ContactObservation, dt: float, *, timestamp: float) -> StepControl:
        """Compute a command from the current state and reference at exactly 10 Hz."""
        if dt != CONTROL_DT:
            raise ValueError("Controller timestep must be 0.1 seconds")
        if self.policy is None:
            command = StepControl(self.runner.robot_adaptor._dof2doa(state.target_qpos_f))
        else:
            contacts = observation.controller_contacts(timestamp)
            forces = np.asarray([c["contact_force"][:3] for c in contacts]).reshape(-1, 3)
            state = replace(
                state, contacts=contacts, contact_forces=forces, current_sum_force=float(forces[:, 0].sum())
            )
            command = self.policy.control(self.runner, state)
        self._diagnostics = command.diagnostics.copy()
        return command

    def diagnostics(self) -> dict:
        """Return the native diagnostics; MPC fields are absent in linear mode."""
        return self._diagnostics.copy()
