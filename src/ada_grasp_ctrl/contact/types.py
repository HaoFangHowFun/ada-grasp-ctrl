"""Observation contract independent of MuJoCo, ROS and tactile hardware."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ContactObservation:
    """One timestamped snapshot in meters, newtons and newton-meters.

    Each contact retains the original controller dictionary keys. Frames contain
    column axes; +X points from hand to object, and the six-vector wrench acts on
    the object. Missing sensed quantities must never be represented as measured
    zeros. The audited tactile point-contact compatibility layer may represent
    zero local moments as an explicit model assumption in per-contact provenance;
    these are not sensor measurements. Other unavailable fields reject control.
    """

    timestamp: float
    contacts: tuple[dict, ...]
    source: str
    valid: bool = True
    errors: tuple[str, ...] = ()
    environment_contacts: tuple[dict, ...] = ()

    def controller_contacts(self, timestamp: float) -> list[dict]:
        """Validate freshness and return disposable copies for the native MPC."""
        from .validation import validate_observation

        validate_observation(self, timestamp)
        return deepcopy(list(self.contacts))


class ContactSource(Protocol):
    """A source reads the latest available snapshot without advancing physics."""

    def observe(self) -> ContactObservation:
        """Return measurements and explicit availability metadata."""
