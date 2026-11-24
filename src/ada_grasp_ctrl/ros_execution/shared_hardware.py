"""Explicit hardware packet identity and entry point; never accepts mock transport packets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .shared_packet import _validate_packet
from ada_grasp_ctrl.experiments.right_ros_contract import CYCLE_PHASES

KIND = "ada_shared_hardware_joint_targets"


def validate_hardware_packet(packet: dict) -> dict:
    """Validate hardware intent without granting operator review or physical validation."""
    checked = _validate_packet(packet, environment="hardware", kind=KIND)
    if checked["plan_identity"]["active_arms"] != ["left", "right"]:
        raise ValueError("Shared hardware requires a paired plan")
    if tuple(s["phase"] for s in checked["schedule"]["stages"]) != ("above_object", *CYCLE_PHASES):
        raise ValueError("Shared hardware requires the complete reviewed cycle")
    source = checked.get("source", {})
    if not source.get("references") or source.get("plan_path") not in source["references"]:
        raise ValueError("Hardware packet requires bound plan and source references")
    if checked.get("operator_review_required") is not True:
        raise ValueError("Hardware packet requires explicit operator review")
    return checked


def load_hardware_packet(path: Path) -> dict:
    """Verify all source files and the exported schedule before any ROS initialization."""
    packet = validate_hardware_packet(json.loads(path.read_text()))
    source = packet["source"]
    for name, digest in source["references"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reviewed hardware source changed: {name}")
    plan = json.loads(Path(source["plan_path"]).read_text())
    if (
        hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
        != packet["source_schedule_sha256"]
    ):
        raise ValueError("Hardware schedule identity changed")
    identity = packet["plan_identity"]
    if any(identity[k] != plan["plan"][k] for k in ("candidate_id", "mode", "active_arms", "provenance")):
        raise ValueError("Hardware candidate identity differs from bound plan")
    if {k: v for k, v in plan.items() if k != "plan"} != packet["schedule"]:
        raise ValueError("Hardware transport differs from reviewed schedule")
    return packet


def main(argv: list[str] | None = None) -> int:
    """Use the shared executor with explicit hardware scope and reviewed packet digest."""
    from .shared_executor import executor_main

    return executor_main(argv, environment="hardware")


if __name__ == "__main__":
    raise SystemExit(main())
