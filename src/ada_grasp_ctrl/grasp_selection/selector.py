"""Deterministic hard-constraint-first selection; no ROS or inference imports."""

from typing import Any, Sequence
import numpy as np

from .criteria.base import HardCriterion, ScoringCriterion
from .criteria.joint_effort import JointDisplacement
from .criteria.reachability import Reachability
from .types import HAND_NAMES, CandidateEvaluation, GraspCandidate, SelectionContext, SelectionResult, jsonable


class GraspSelector:
    """Select minimum-cost feasible endpoint options and retain failed candidates."""

    def __init__(
        self,
        robot_adapter: Any,
        hard_criteria: Sequence[HardCriterion] | None = None,
        scoring_criteria: Sequence[ScoringCriterion] | None = None,
    ) -> None:
        self.robot_adapter = robot_adapter
        self.hard_criteria = list(hard_criteria) if hard_criteria is not None else [Reachability(robot_adapter)]
        self.scoring_criteria = list(scoring_criteria) if scoring_criteria is not None else [JointDisplacement()]
        if not self.hard_criteria or not self.scoring_criteria:
            raise ValueError("At least one hard criterion and one score are required")
        names = [c.name for c in self.scoring_criteria]
        if len(names) != len(set(names)):
            raise ValueError("Scoring criterion names must be unique")

    def select(self, candidates: Sequence[GraspCandidate], context: SelectionContext) -> SelectionResult:
        """Evaluate all candidates, resolving exact ties by candidate ID and branch index."""
        ids = [c.candidate_id for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate IDs")
        context.robot_state.check_fresh(context.now_s, context.max_age_s)
        self.robot_adapter.validate_context(context)
        evaluations, ranked = [], []
        for candidate in sorted(candidates, key=lambda c: c.candidate_id):
            ev = CandidateEvaluation(candidate.candidate_id, candidate.source_id)
            evaluations.append(ev)
            if candidate.calibration_id != context.calibration_id:
                ev.reasons.append("calibration_mismatch")
                continue
            for criterion in self.hard_criteria:
                criterion.evaluate(candidate, context, ev)
            if ev.reasons:
                continue
            feasible = []
            for option in ev.options:
                if option.reasons:
                    continue
                total = 0.0
                for criterion in self.scoring_criteria:
                    cost, details = criterion.score(option, context)
                    if not np.isfinite(cost):
                        option.reasons.append("nonfinite_score:" + criterion.name)
                        break
                    total += cost
                    option.score_details[criterion.name] = details
                if option.reasons or not np.isfinite(total):
                    if not option.reasons:
                        option.reasons.append("nonfinite_total_score")
                    continue
                option.score = total
                feasible.append((total, candidate.candidate_id, option.solution_index, candidate, option))
            if feasible:
                best = min(feasible, key=lambda r: r[:3])
                ev.best_solution_index = best[2]
                ranked.append(best)
            else:
                ev.reasons.append("no_feasible_ik_option")
        best = min(ranked, key=lambda r: r[:3]) if ranked else None
        readiness = ["continuous_path_not_checked", "execution_layer_must_revalidate_current_state"]
        for gate in ("camera_world", "world_right_base", "hand_mount", "table", "environment"):
            if context.acceptance.get(gate) is not True:
                readiness.append(gate + "_unverified")
        if not context.scene_complete:
            readiness.append("collision_scene_incomplete")
        if context.replay or context.robot_state.environment != "hardware":
            readiness.append("replay_or_nonhardware_state")
        selected = (
            None
            if best is None
            else {
                "candidate_id": best[3].candidate_id,
                "source_id": best[3].source_id,
                "solution_index": best[2],
                "arm_joint_names": context.robot_state.joint_names,
                "hand_joint_names": HAND_NAMES,
                "stages": best[3].stages,
                "original": best[3].original,
                "stage_checks": best[4].stages,
                "score_details": best[4].score_details,
            }
        )
        return SelectionResult(
            None if best is None else best[1],
            None if best is None else best[4].q_rad,
            None if best is None else best[0],
            jsonable(selected),
            evaluations,
            {
                "context": jsonable(context),
                "robot": self.robot_adapter.provenance,
                "hard_criteria": [type(c).__name__ for c in self.hard_criteria],
                "scores": [
                    {"name": c.name, "weights": jsonable(getattr(c, "weights", None))} for c in self.scoring_criteria
                ],
                "tie_break": ["cost", "candidate_id", "solution_index"],
                "angle_wrapping": False,
            },
            "selected" if best else "no_feasible_candidate",
            readiness,
        )
