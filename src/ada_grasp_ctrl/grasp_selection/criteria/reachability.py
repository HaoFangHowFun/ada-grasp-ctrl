"""Adapter-backed hard endpoint feasibility, cached once per candidate."""

from typing import Any
from ..types import CandidateEvaluation, GraspCandidate, SelectionContext


class Reachability:
    """Use the selected robot adapter without importing a simulator into core."""

    def __init__(self, robot_adapter: Any) -> None:
        self.robot_adapter = robot_adapter

    def evaluate(self, candidate: GraspCandidate, context: SelectionContext, evaluation: CandidateEvaluation) -> None:
        """Record every bounded IK branch and all failed endpoint checks."""
        self.robot_adapter.evaluate(candidate, context, evaluation)
