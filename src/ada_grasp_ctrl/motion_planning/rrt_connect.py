"""NumPy bidirectional RRT-Connect with directed edges and fail-closed deadlines."""

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np

EdgeValid = Callable[[np.ndarray, np.ndarray], bool]


@dataclass
class PlanResult:
    """A complete checked path, or a failure with no executable prefix."""

    status: str
    path: list[np.ndarray] = field(default_factory=list)
    method: str = "rrt_connect"
    wall_s: float = 0.0
    iterations: int = 0
    edge_queries: int = 0
    original_waypoints: int = 0


def plan(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    joint_limits: np.ndarray,
    edge_valid: EdgeValid,
    *,
    extension: float = 0.20,
    max_iterations: int = 5000,
    timeout_s: float = 3.0,
    shortcut_s: float = 0.05,
    seed: int | None = None,
    deadline: float | None = None,
    canceled: Callable[[], bool] = lambda: False,
) -> PlanResult:
    """Find the first feasible path and greedily shortcut it, checking forward edges.

    Args:
        q_start: Initial configuration of arbitrary positive dimension.
        q_goal: Goal configuration of the same dimension.
        joint_limits: Finite (dimension, 2) closed bounds.
        edge_valid: Full sampled-edge checker. Called in execution order, including
            zero-length endpoint checks. It must implement its own sampling and
            observe the same deadline if a single query can take appreciable time.
        extension: Maximum L2 extension in joint units.
        max_iterations: Maximum alternating search iterations.
        timeout_s: Total budget including shortcuts and final validation.
        shortcut_s: Maximum greedy shortcut budget within the total budget.
        seed: Injectable NumPy random seed.
        deadline: Optional earlier absolute monotonic deadline, e.g. including IK.
        canceled: Cancellation predicate, checked before and after every edge.

    Returns:
        Structured result; only status ``success`` contains a path. Sampling is
        not continuous collision certification. Deadline overruns fail closed.

    Raises:
        ValueError: On malformed bounds or planner settings. Callback exceptions
            other than TimeoutError propagate rather than concealing failures.
    """
    began = time.monotonic()
    limits = np.asarray(joint_limits, dtype=float)
    if (
        limits.ndim != 2
        or limits.shape[1] != 2
        or not len(limits)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] > limits[:, 1])
        or not np.isfinite([extension, timeout_s, shortcut_s]).all()
        or extension <= 0
        or timeout_s <= 0
        or shortcut_s < 0
        or type(max_iterations) is not int
        or max_iterations < 1
        or (deadline is not None and not np.isfinite(deadline))
    ):
        raise ValueError("Invalid planner bounds/settings")
    end = min(began + timeout_s, deadline if deadline is not None else float("inf"))
    reserve = min(0.6, max(0, end - began) * 0.2)
    search_end = end - reserve
    result = PlanResult("no_solution")

    def guard() -> None:
        if canceled():
            raise InterruptedError("canceled")
        if time.monotonic() >= end:
            raise TimeoutError("timeout")

    def check(a: np.ndarray, b: np.ndarray) -> bool:
        guard()
        result.edge_queries += 1
        valid = edge_valid(a, b)
        guard()
        return bool(valid)

    def finish(status: str) -> PlanResult:
        result.status, result.wall_s = status, time.monotonic() - began
        if status != "success":
            result.path = []
        return result

    def endpoint(value: np.ndarray) -> np.ndarray | None:
        q = np.asarray(value, dtype=float)
        if q.shape != (len(limits),) or not np.isfinite(q).all():
            return None
        return q.copy() if np.all((q >= limits[:, 0]) & (q <= limits[:, 1])) else None

    start, goal = endpoint(q_start), endpoint(q_goal)
    try:
        guard()
        if start is None or not check(start, start):
            return finish("invalid_start")
        if goal is None or not check(goal, goal):
            return finish("invalid_goal")
        if check(start, goal):
            path = [start] if np.array_equal(start, goal) else [start, goal]
            result.method = "direct"
        else:
            rng = np.random.default_rng(seed)
            # Each tree retains its execution direction across swaps.
            trees = [([start], [-1], True), ([goal], [-1], False)]
            path = []

            def extend(tree: tuple, target: np.ndarray) -> tuple[int | None, bool]:
                nodes, parents, forward = tree
                guard()
                index = int(np.argmin(np.sum((np.asarray(nodes) - target) ** 2, axis=1)))
                delta = target - nodes[index]
                distance = float(np.linalg.norm(delta))
                if distance == 0:
                    return index, True
                q = target.copy() if distance <= extension else nodes[index] + delta * (extension / distance)
                valid = check(nodes[index], q) if forward else check(q, nodes[index])
                if not valid:
                    return None, False
                nodes.append(q)
                parents.append(index)
                return len(nodes) - 1, distance <= extension

            def trace(tree: tuple, index: int) -> list[np.ndarray]:
                nodes, parents, _ = tree
                branch = []
                while index != -1:
                    branch.append(nodes[index])
                    index = parents[index]
                return branch[::-1]

            for iteration in range(max_iterations):
                guard()
                if time.monotonic() >= search_end:
                    return finish("timeout")
                result.iterations = iteration + 1
                a, b = trees
                index, _ = extend(a, rng.uniform(limits[:, 0], limits[:, 1]))
                if index is not None:
                    while time.monotonic() < search_end:
                        other, reached = extend(b, a[0][index])
                        if other is None:
                            break
                        if reached:
                            left, right = (trace(a, index), trace(b, other))
                            if not a[2]:
                                left, right = right, left
                            path = left + right[-2::-1]
                            break
                if path:
                    break
                trees.reverse()
            if not path:
                return finish("timeout" if time.monotonic() >= search_end else "no_solution")

        result.original_waypoints = len(path)
        shortcut_end = min(time.monotonic() + shortcut_s, search_end)
        i = 0
        while i + 2 < len(path) and time.monotonic() < shortcut_end:
            for j in range(len(path) - 1, i + 1, -1):
                if time.monotonic() >= shortcut_end:
                    break
                if check(path[i], path[j]):
                    path[i + 1 : j] = []
                    break
            i += 1
        # The same callback and direction are used for direct paths, shortcuts,
        # and final validation. Never expose a partial tree or unchecked prefix.
        if len(path) == 1:
            valid = check(path[0], path[0])
        else:
            valid = all(check(a, b) for a, b in zip(path, path[1:]))
        if not valid:
            return finish("final_validation_failed")
        guard()
        result.path = [q.copy() for q in path]
        return finish("success")
    except InterruptedError:
        return finish("canceled")
    except TimeoutError:
        return finish("timeout")
