"""One Ada linear reference used by preview, MPC and command sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def linear_reference(start: Sequence[float], end: Sequence[float], elapsed: float, duration: float) -> list[float]:
    """Return q0 + clip(elapsed / duration, 0, 1) * (qf - q0), without easing."""
    a, b = tuple(start), tuple(end)
    if not a or len(a) != len(b) or not all(map(math.isfinite, (*a, *b, elapsed, duration))) or duration <= 0:
        raise ValueError("Invalid linear segment")
    alpha = min(1.0, max(0.0, elapsed / duration))
    return [x + alpha * (y - x) for x, y in zip(a, b)]


@dataclass
class LinearReference:
    """Clocked segment with endpoint limits and rejection of backward sampling."""

    start: Sequence[float]
    end: Sequence[float]
    began: float
    duration: float
    lower: Sequence[float]
    upper: Sequence[float]
    _last_time: float | None = None

    def __post_init__(self) -> None:
        self.start, self.end, self.lower, self.upper = map(tuple, (self.start, self.end, self.lower, self.upper))
        linear_reference(self.start, self.end, self.began, self.duration)
        if len(self.lower) != len(self.start) or len(self.upper) != len(self.start):
            raise ValueError("Reference limit dimensions differ")
        for a, b, lo, hi in zip(self.start, self.end, self.lower, self.upper):
            if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi or not lo <= a <= hi or not lo <= b <= hi:
                raise ValueError("Reference violates joint limits")

    def sample(self, now: float) -> list[float]:
        """Sample monotonically without advancing the controller or any actuator."""
        if self._last_time is not None and now < self._last_time:
            raise ValueError("Reference clock moved backward")
        result = linear_reference(self.start, self.end, now - self.began, self.duration)
        self._last_time = now
        return result
