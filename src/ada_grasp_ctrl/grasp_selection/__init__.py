"""Read-only grasp selection; ROS and robot backends are imported only on demand."""

__all__ = ["GraspSelector", "GraspCandidate", "RobotState", "SelectionContext", "SelectionResult", "StageTarget"]


def __getattr__(name: str):
    """Keep the ROS snapshot tool usable without Ada's numeric dependencies."""
    if name == "GraspSelector":
        from .selector import GraspSelector

        return GraspSelector
    if name in __all__:
        from . import types

        return getattr(types, name)
    raise AttributeError(name)
