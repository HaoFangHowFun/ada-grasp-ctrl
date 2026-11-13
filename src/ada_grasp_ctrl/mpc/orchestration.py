"""Stage boundary for baseline ownership and actuator-free passive proposals."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Any

from .contracts import ACTIVE_STAGES, MODES, JointFeedback, MPCProposal, fresh, named_vector


class StageOrchestrator:
    """Compute proposals only in grasp/squeeze; publication is a separate capability.

    A caller must confirm baseline arrival before advancing. This object has no
    publisher, action client, device handle or automatic mode-switch capability.
    """

    def __init__(
        self, mode: str, phases: tuple[str, ...], solve: Callable[..., MPCProposal], log: Callable[[dict], None]
    ):
        from ada_grasp_ctrl.experiments.right_ros_contract import PHASE_SEQUENCES

        if mode not in MODES or tuple(phases) not in PHASE_SEQUENCES:
            raise ValueError("Invalid mode or stage sequence")
        self.mode, self.phases, self.solve, self.log = mode, tuple(phases), solve, log
        self.index = -1
        self.failed = False
        self.last_time: float | None = None
        self.next_mpc: float | None = None

    def enter(self, phase: str, *, previous_arrived: bool = False) -> None:
        """Require ordered, externally confirmed transitions; never skip lift/hold gates."""
        if self.failed or self.index + 1 >= len(self.phases) or phase != self.phases[self.index + 1]:
            self.failed = True
            self.log({"event": "fault", "mode": self.mode, "owner": "none", "safety_reasons": "Invalid stage sequence"})
            raise ValueError("Invalid stage sequence")
        if self.index >= 0 and not previous_arrived:
            self.failed = True
            self.log(
                {"event": "fault", "mode": self.mode, "owner": "none", "safety_reasons": "Previous stage not arrived"}
            )
            raise ValueError("Previous stage has not arrived")
        self.index += 1

    def cancel(self) -> None:
        """Latch cancellation; a new invocation is required to restart."""
        self.failed = True
        self.log({"event": "cancel", "mode": self.mode, "owner": "none"})

    def tick(
        self, now: float, feedback: JointFeedback, q_cmd: tuple[float, ...], q_ref: tuple[float, ...], contact: Any
    ) -> MPCProposal | None:
        """Log measured/reference/command separately and return an optional candidate."""
        if self.failed or self.index < 0:
            raise ValueError("Orchestrator is not running")
        phase = self.phases[self.index]
        row = {"event": "tick", "timestamp": now, "stage": phase, "mode": self.mode}
        try:
            fresh(feedback.source_timestamp, feedback.receive_timestamp, now, 0.5)
            if self.last_time is not None and now <= self.last_time:
                raise ValueError("Nonmonotonic control clock")
            self.last_time = now
            q_cmd = named_vector(feedback.joint_names, q_cmd)
            q_ref = named_vector(feedback.joint_names, q_ref)
            active = phase in ACTIVE_STAGES and self.mode.startswith("mpc_")
            row.update(
                joint_names=feedback.joint_names,
                measured_q=feedback.positions_rad,
                feedback_source_timestamp=feedback.source_timestamp,
                feedback_age_s=now - feedback.source_timestamp,
                q_cmd=q_cmd,
                q_ref=q_ref,
                owner="mpc_mock" if active and self.mode == "mpc_active" else "baseline",
                proposal=None,
                proposal_executed=False,
                contact_source=getattr(contact, "source", "unavailable"),
                contact_valid=getattr(contact, "valid", False),
            )
            proposal = None
            if active and (self.next_mpc is None or now + 1e-9 >= self.next_mpc):
                try:
                    proposal = self.solve(now, feedback, q_cmd, q_ref, contact, stage=phase)
                except Exception as error:
                    if self.mode != "mpc_passive":
                        raise
                    proposal = MPCProposal(
                        feedback.joint_names,
                        None,
                        False,
                        now,
                        0,
                        "unavailable",
                        {"reason": f"Passive computation failed: {type(error).__name__}: {error}"},
                    )
                self.next_mpc = now + 0.1  # Drop missed ticks; never replay a backlog of solves.
                row["proposal"] = asdict(proposal)
                row["safety_reasons"] = proposal.diagnostics.get("reason", "")
            self.log(row)
            return proposal
        except Exception as error:
            self.failed = True
            self.log({**row, "event": "fault", "owner": "none", "safety_reasons": str(error)})
            raise
