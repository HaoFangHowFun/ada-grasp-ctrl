"""Explicit hierarchical selection adapter for the existing single-side live exporter."""

from __future__ import annotations

from copy import deepcopy

from ada_grasp_ctrl.grasp_selection.contracts import candidate_from_schema1, plan_from_candidate, validate_grasp_plan
from ada_grasp_ctrl.grasp_selection.hierarchical import select_hierarchical
from ada_grasp_ctrl.grasp_selection.multi_arm import enabled_set

MODES = ("right_full", "left_full")


def selector_settings(value: object) -> dict:
    """Validate the live adapter's explicit ranking before capture or process startup.

    Returns:
        Independent settings using the selector's schema-1 ranking contract.

    Raises:
        ValueError: For malformed, contradictory or unimplemented mode configuration.
    """
    if not isinstance(value, dict) or set(value) != {"prior", "supported_modes"}:
        raise ValueError("hierarchical_selector requires prior and supported_modes")
    prior, supported = value["prior"], value["supported_modes"]
    if (
        not isinstance(prior, dict)
        or set(prior) != {"schema_version", "kind", "ranking"}
        or type(prior["schema_version"]) is not int
        or prior["schema_version"] != 1
        or prior["kind"] != "ranking"
    ):
        raise ValueError("hierarchical_selector.prior requires schema_version: 1, kind: ranking and ranking")
    for name, labels in (("supported_modes", supported), ("prior.ranking", prior["ranking"])):
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(m, str) or m not in MODES for m in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(f"hierarchical_selector.{name}: expected unique right_full/left_full labels")
    if set(prior["ranking"]) != set(supported):
        raise ValueError("hierarchical_selector.prior.ranking must name exactly the supported_modes")
    return deepcopy(value)


def canonical_batch(rows: list[dict], available_arms: object) -> dict:
    """Adapt fresh right-full / mirrored-left rows without modifying legacy candidates.

    Returns:
        Canonical batch containing only enabled sides, retaining checkpoint and mirror provenance.

    Raises:
        ValueError: For duplicate IDs, unsupported sources or inconsistent side/mode identity.
    """
    available = enabled_set(available_arms)
    groups = {side + "_full": [] for side in sorted(available)}
    ids = set()
    if not isinstance(rows, list):
        raise ValueError("Live candidates must be a list of schema-1 rows")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Live candidates must be schema-1 objects")
        arms = row.get("required_arms")
        if not isinstance(arms, list) or len(arms) != 1 or not isinstance(arms[0], str) or arms[0] not in available:
            raise ValueError("Live hierarchical candidates must require one enabled arm")
        side = arms[0]
        candidate = candidate_from_schema1(row, mode=side + "_full")
        name = candidate["candidate_id"]
        if not name.startswith(side + ":") or name in ids:
            raise ValueError("Live candidate IDs must be unique and retain their left:/right: prefix")
        ids.add(name)
        source = candidate["provenance"]
        generator = "mirrored_right_full" if side == "left" else "right_full"
        if source.get("generator") != generator or not all(source.get(k) for k in ("checkpoint", "metadata")):
            raise ValueError(f"{name}: expected existing {generator} checkpoint/metadata provenance")
        if side == "left" and not source.get("reflection_plane"):
            raise ValueError(f"{name}: missing left reflection provenance")
        tags = dict(source_model_mode="right_full", generation_method=generator)
        if any(k in source and source[k] != v for k, v in tags.items()):
            raise ValueError(f"{name}: contradictory source model provenance")
        source.update(tags)
        groups[side + "_full"].append(candidate)
    return dict(schema_version=1, candidates_by_mode=groups)


def legacy_selection(report: dict, batch: dict, rows: list[dict], available_arms: object) -> dict:
    """Project a checked hierarchical decision into the existing export envelope.

    The canonical report is retained as audit data inside the hash-bound selection;
    neither it nor its plan can grant execution authorization.

    Raises:
        ValueError: If selected candidate, mode, side, availability or plan disagree.
    """
    available = sorted(enabled_set(available_arms))
    if report["available_arms"] != available or any(
        report.get(field) is not False for field in ("execution_ready", "hardware_ready")
    ):
        raise ValueError("Hierarchical report availability/readiness differs from the live run")
    chosen = deepcopy(report["selected"])
    if report["status"] == "SELECTED":
        if chosen is None:
            raise ValueError("SELECTED requires a candidate")
        side, mode, name = chosen["selected_side"], report["selected_mode"], chosen["candidate_id"]
        if side not in available or mode != side + "_full" or report["selected_side"] != side:
            raise ValueError("Hierarchical selected mode/side differs")
        matches = [r for r in batch["candidates_by_mode"].get(mode, []) if r["candidate_id"] == name]
        originals = [r for r in rows if r["candidate_id"] == name]
        if (
            len(matches) != 1
            or len(originals) != 1
            or chosen["required_arms"] != [side]
            or chosen["candidate"] != matches[0]
            or matches[0]["required_arms"] != [side]
            or originals[0]["required_arms"] != [side]
            or originals[0]["targets"] != matches[0]["targets"]
        ):
            raise ValueError("Hierarchical selected identity/arms differ from the input candidate")
        plan = validate_grasp_plan(report["plan"])
        if plan != plan_from_candidate(matches[0], available_arms=available):
            raise ValueError("Hierarchical plan differs from the selected candidate")
        # Keep the legacy row, including its original provenance, exactly as inferred.
        chosen["candidate"] = deepcopy(originals[0])
    elif report["status"] != "NO_FEASIBLE_GRASP" or any(
        report.get(k) is not None for k in ("selected", "plan", "selected_mode", "selected_side")
    ):
        raise ValueError("Inconsistent hierarchical no-selection result")
    return dict(
        schema_version=1,
        status="selected" if chosen is not None else "no_feasible_candidate",
        enabled_arms=available,
        weights=[1.0] * 6,
        effort_unit="rad^2",
        effort_definition="sum(W * (q_grasp_CRX - q_current_CRX)^2); no angle wrapping",
        active_hand_object_penetration_policy="10 mm during initialization/pregrasp only",
        q_current=deepcopy(report["q_current"]),
        selected=chosen,
        evaluations=[deepcopy(c) for mode in report["mode_trace"] for c in mode["candidates"]],
        execution_ready=False,
        hardware_ready=False,
        hierarchical_report=deepcopy(report),
    )


def select_live_hierarchical(
    robot: object, sim: object, rows: list[dict], available_arms: object, q_current: object, settings: dict
) -> tuple[dict, dict, dict]:
    """Call the existing selector and return legacy selection, canonical batch and audit report."""
    settings = selector_settings(settings)
    batch = canonical_batch(rows, available_arms)
    report = select_hierarchical(
        robot, sim, batch, settings["prior"], available_arms, q_current, supported_modes=settings["supported_modes"]
    )
    return legacy_selection(report, batch, rows, available_arms), batch, report
