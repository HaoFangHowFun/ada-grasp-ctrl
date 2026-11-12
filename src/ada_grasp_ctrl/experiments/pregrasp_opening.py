"""Explicit pre-selection finger opening for local Phase 2 experiments."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ada_grasp_ctrl.grasp_selection.multi_arm import hand_names, validate_candidate
from ada_grasp_ctrl.robot_environment.model import side_slice


def opening_multiplier(config: dict) -> float:
    """Validate the optional multiplier of the decoded grasp-to-pregrasp finger delta.

    One preserves decoded candidates. Larger values extrapolate fingers only and
    clip them to named source joint limits; wrist and grasp/squeeze stay unchanged.
    """
    settings = config.get("candidate_preparation", {})
    if not isinstance(settings, dict) or set(settings) - {"pregrasp_open_multiplier"}:
        raise ValueError("candidate_preparation only supports pregrasp_open_multiplier")
    value = settings.get("pregrasp_open_multiplier", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 1:
        raise ValueError("candidate_preparation.pregrasp_open_multiplier must be finite and at least 1")
    return float(value)


def prepare_candidates(rows: list[dict], limits: np.ndarray, multiplier: float) -> list[dict]:
    """Extrapolate pregrasp fingers, clip to limits and retain raw values in provenance.

    Args:
        rows: Unmodified inference candidates with canonical per-side joint names.
        limits: Source limits in canonical 56-joint order, shaped (56, 2).
        multiplier: Factor applied to pregrasp minus grasp; 3 means triple that delta.

    Returns:
        Copies with only pregrasp fingers adjusted. Collision selection and
        execution consume these same copies; raw inference remains replayable.

    Raises:
        ValueError: On malformed settings, limits, overflow, or already adjusted input.
    """
    multiplier = opening_multiplier({"candidate_preparation": {"pregrasp_open_multiplier": multiplier}})
    limits = np.asarray(limits, dtype=float)
    if limits.shape != (56, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Expected finite canonical (56, 2) source joint limits")
    if multiplier == 1:
        return deepcopy(rows)
    prepared = []
    for raw in rows:
        row = validate_candidate(raw)
        if "pregrasp_opening" in row["provenance"]:
            raise ValueError("Use raw candidates.json; pregrasp opening must not be applied twice")
        details = {}
        for side in row["required_arms"]:
            targets = row["targets"][side]
            names = hand_names(side)
            original = np.array([targets["pregrasp"]["joint_positions_rad"][n] for n in names])
            grasp = np.array([targets["grasp"]["joint_positions_rad"][n] for n in names])
            with np.errstate(over="raise", invalid="raise"):
                try:
                    requested = grasp + multiplier * (original - grasp)
                except FloatingPointError as exc:
                    raise ValueError("Nonfinite extrapolated pregrasp") from exc
            bounds = limits[side_slice(side)][6:]
            result = np.clip(requested, bounds[:, 0], bounds[:, 1])
            targets["pregrasp"]["joint_positions_rad"] = dict(zip(names, result.tolist()))
            details[side] = dict(
                original_joint_positions_rad=dict(zip(names, original.tolist())),
                requested_joint_positions_rad=dict(zip(names, requested.tolist())),
                clipped_joints=[
                    dict(name=n, requested_rad=float(a), applied_rad=float(b), limits_rad=limit.tolist())
                    for n, a, b, limit in zip(names, requested, result, bounds)
                    if a != b
                ],
            )
        row["provenance"]["pregrasp_opening"] = dict(
            multiplier=multiplier,
            formula="clip(q_grasp + multiplier * (q_pregrasp - q_grasp), source_limits)",
            sides=details,
        )
        prepared.append(row)
    return prepared
