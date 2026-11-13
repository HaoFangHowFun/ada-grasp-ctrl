"""Experimental read-only Sharpa tactile preview; independent of grasp control."""

from ada_grasp_ctrl.contact.tactile_observation import TactileContactObservation

__all__ = ["TactileAdapter", "TactileContactObservation"]


def __getattr__(name: str) -> object:
    """Keep system-Python ROS ingestion independent of geometry/FK dependencies."""
    if name == "TactileAdapter":
        from .adapter import TactileAdapter

        return TactileAdapter
    raise AttributeError(name)
