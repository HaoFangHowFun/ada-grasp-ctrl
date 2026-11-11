"""Shared quintic timing for preview and passive execution."""

import math


def arrival_reached(error: list[float], config: dict) -> bool:
    """Require arm arrival; optionally record finger errors without gating completion."""
    if len(error) != 28 or not all(math.isfinite(v) for v in error):
        raise ValueError("Arrival requires 28 finite joint errors")
    return max(abs(v) for v in error[:6]) <= config["arm_tolerance"] and (
        not config.get("hand_arrival_required", True) or max(abs(v) for v in error[6:]) <= config["hand_tolerance"]
    )


def smoothstep(u: float) -> float:
    """Return the existing zero-end-velocity quintic coefficient for normalized time."""
    return 10 * u**3 - 15 * u**4 + 6 * u**5
