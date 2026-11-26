"""Explicit mode priorities followed by single-arm or combined-state paired feasibility."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .contracts import validate_grasp_candidate, plan_from_candidate
from .criteria.joint_effort import JointDisplacement
from .multi_arm import enabled_set, select_candidates, select_paired_candidates
from .types import jsonable


def _labels(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, (list, tuple))
        or any(not isinstance(label, str) or not label.strip() for label in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field}: expected unique nonempty mode names")
    return list(value)


def _versioned(value: object, fields: set[str], field: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{field}: expected fields {sorted(fields)}")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError(f"{field}.schema_version: expected 1")
    return value


def _priorities(prior: dict) -> tuple[list[str], dict[str, float | None]]:
    if not isinstance(prior, dict) or prior.get("kind") not in ("scores", "ranking"):
        raise ValueError("prior.kind: supply explicit scores or ranking; raw K_t counts are not probabilities")
    kind = prior["kind"]
    _versioned(prior, {"schema_version", "kind", kind}, "prior")
    if kind == "ranking":
        order = _labels(prior[kind], "prior.ranking")
        return order, dict.fromkeys(order)
    scores = prior[kind]
    if not isinstance(scores, dict):
        raise ValueError("prior.scores: expected a mode-to-score object")
    _labels(list(scores), "prior.scores")
    for mode, score in scores.items():
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"prior.scores.{mode}: expected a finite score")
    return sorted(scores, key=lambda mode: (-scores[mode], mode)), {m: float(v) for m, v in scores.items()}


def select_hierarchical(
    robot: object,
    sim: object,
    batch: dict,
    prior: dict,
    available_arms: object,
    q_current: object,
    *,
    supported_modes: Sequence[str] = ("left_full", "right_full"),
    enabled_modes: Sequence[str] | None = None,
    weights_by_arm: dict[str, object] | None = None,
    max_nfev: int = 200,
    endpoint_gate: object = None,
) -> dict[str, Any]:
    """Select within the first feasible externally ranked mode; never compare modes by effort.

    Args:
        robot: Incumbent dual-scene robot/IK adapter.
        sim: Incumbent static contact checker.
        batch: Schema-1 object with ``candidates_by_mode: {mode: [GraspCandidate, ...]}``.
        prior: Schema-1 ``kind: scores`` with a ``scores`` mapping, or ``kind: ranking``
            with a descending ``ranking`` list. Scores need not be probabilities.
        available_arms: Nonempty subset of left/right allowed for this invocation.
        q_current: Current canonical 56-joint state, including the passive branch.
        supported_modes: Explicit caller capability labels, independent of inference.
        enabled_modes: Optional subset of supported modes; defaults to all supported modes.
        weights_by_arm: Optional six-joint nonnegative weights for each available arm.
        max_nfev: Incumbent per-seed IK evaluation budget.
        endpoint_gate: Optional single-arm post-grasp gate forwarded before mode selection.
            Callers must exclude paired modes when using a single-arm-only gate.

    Returns:
        Schema-1 report with ``SELECTED`` and a data-only plan, or ``NO_FEASIBLE_GRASP``
        with ``plan=None``. Mode and candidate traces include skips and backend diagnostics.
        Equal scores use lexical mode order; within a mode ties use candidate ID then seed.

    Raises:
        ValueError: For malformed run-level input, duplicate candidate IDs or invalid state.
            Malformed attributable candidates are recorded individually and permit fallback.
    """
    _versioned(batch, {"schema_version", "candidates_by_mode"}, "batch")
    groups = batch["candidates_by_mode"]
    if not isinstance(groups, dict):
        raise ValueError("batch.candidates_by_mode: expected an object")
    _labels(list(groups), "batch.candidates_by_mode")
    order, scores = _priorities(prior)
    supported = set(_labels(supported_modes, "supported_modes"))
    enabled = supported if enabled_modes is None else set(_labels(enabled_modes, "enabled_modes"))
    if not enabled <= supported:
        raise ValueError("enabled_modes must be a subset of supported_modes")
    try:
        available = sorted(enabled_set(available_arms))
    except (ValueError, TypeError) as error:
        raise ValueError("available_arms: expected nonempty unique left/right arms") from error
    if type(max_nfev) is not int or max_nfev <= 0:
        raise ValueError("max_nfev: expected a positive integer")
    if weights_by_arm is not None and (not isinstance(weights_by_arm, dict) or set(weights_by_arm) != set(available)):
        raise ValueError("weights_by_arm keys must exactly match available_arms")
    weights = {
        side: JointDisplacement((1,) * 6 if weights_by_arm is None else weights_by_arm[side]).weights.tolist()
        for side in available
    }
    current = robot.configuration(q_current)
    ids = set()
    for mode, rows in groups.items():
        if not isinstance(rows, list):
            raise ValueError(f"batch.candidates_by_mode.{mode}: expected a list")
        for row in rows:
            name = row.get("candidate_id") if isinstance(row, dict) else None
            if not isinstance(name, str) or not name.strip() or name in ids:
                raise ValueError("Candidate IDs must be present and globally unique across modes")
            ids.add(name)

    ranked = {mode: i + 1 for i, mode in enumerate(order)}
    # Retain groups omitted by the prior as explicit unranked diagnostics, never defaults.
    order += sorted(set(groups) - set(order))
    trace, chosen, chosen_mode = [], None, None
    for mode in order:
        rows = sorted(groups.get(mode, []), key=lambda row: row["candidate_id"])
        reasons = []
        if mode not in supported:
            reasons.append("unsupported_mode")
        elif mode not in enabled:
            reasons.append("disabled_mode")
        if mode not in ranked:
            reasons.append("missing_priority")
        if not rows:
            reasons.append("empty_candidate_set")
        if not reasons and chosen is not None:
            reasons.append("not_evaluated_after_selection")
        entry = dict(
            mode=mode,
            rank=ranked.get(mode),
            score=scores.get(mode),
            candidate_count=len(rows),
            status="skipped" if reasons else "no_feasible_candidate",
            reasons=reasons,
            evaluated_candidate_count=0,
            feasible_candidate_count=0,
            candidates=[],
            selected_candidate_id=None,
            effort=None,
        )
        trace.append(entry)
        eligible = {side: [] for side in available}
        eligible["both"] = []
        diagnostics = {}
        for raw in rows:
            diagnostic = dict(candidate_id=raw["candidate_id"], reasons=list(reasons), attempts=[])
            diagnostics[raw["candidate_id"]] = diagnostic
            try:
                row = validate_grasp_candidate(raw)
                if row["mode"] != mode:
                    raise ValueError(f"candidate.mode differs from group {mode!r}")
            except ValueError as error:
                diagnostic["reasons"].append("invalid_candidate")
                diagnostic["error"] = str(error)
                continue
            if not set(row["required_arms"]) <= set(available):
                diagnostic["reasons"].append("disabled_arm")
            elif not reasons:
                eligible["both" if len(row["required_arms"]) == 2 else row["required_arms"][0]].append(row)
        winners = []
        for side, candidates in eligible.items():
            if not candidates:
                continue
            result = (
                select_paired_candidates(robot, sim, candidates, available, current, max_nfev)
                if side == "both"
                else select_candidates(
                    robot,
                    sim,
                    candidates,
                    available,
                    current,
                    weights[side],
                    max_nfev,
                    **({"endpoint_gate": endpoint_gate} if endpoint_gate is not None else {}),
                )
            )
            entry["evaluated_candidate_count"] += len(candidates)
            for diagnostic in result["evaluations"]:
                diagnostics[diagnostic["candidate_id"]] = diagnostic
                if any("effort" in attempt for attempt in diagnostic["attempts"]):
                    entry["feasible_candidate_count"] += 1
            if result["selected"] is not None:
                winners.append(result["selected"])
        entry["candidates"] = [diagnostics[row["candidate_id"]] for row in rows]
        if winners:
            chosen = min(winners, key=lambda row: (row["effort"], row["candidate_id"], row["seed_index"]))
            chosen_mode = mode
            entry.update(status="selected", selected_candidate_id=chosen["candidate_id"], effort=chosen["effort"])
        elif not reasons:
            entry["reasons"] = ["no_feasible_candidate"]

    report = dict(
        schema_version=1,
        status="SELECTED" if chosen is not None else "NO_FEASIBLE_GRASP",
        available_arms=available,
        priority_kind=prior["kind"],
        tie_break=["mode_priority", "mode_name_for_equal_scores", "effort", "candidate_id", "seed_index"],
        weights_by_arm=weights,
        effort_unit="rad^2",
        effort_definition="sum(W_side * (q_grasp_CRX - q_current_CRX)^2); no angle wrapping",
        q_current=current.tolist(),
        selected_mode=chosen_mode,
        selected_side=chosen["selected_side"] if chosen is not None else None,
        selected=chosen,
        effort=chosen["effort"] if chosen is not None else None,
        plan=plan_from_candidate(chosen["candidate"], available_arms=available) if chosen is not None else None,
        mode_trace=trace,
        execution_ready=False,
        hardware_ready=False,
    )
    return jsonable(report)
