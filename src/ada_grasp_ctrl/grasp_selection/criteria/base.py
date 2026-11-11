"""Criteria extend selection without changing the orchestration loop."""

from typing import Protocol
from ..types import CandidateEvaluation, GraspCandidate, IKOption, SelectionContext


class HardCriterion(Protocol):
    """Populate cached options and/or append candidate/option rejection reasons."""

    def evaluate(self, candidate: GraspCandidate, context: SelectionContext, evaluation: CandidateEvaluation) -> None:
        """Apply a hard condition; scores cannot override any rejection."""
        ...


class ScoringCriterion(Protocol):
    """Return a finite scalar and details using already computed options."""

    name: str

    def score(self, option: IKOption, context: SelectionContext) -> tuple[float, dict]:
        """Evaluate one feasible grasp solution."""
        ...
