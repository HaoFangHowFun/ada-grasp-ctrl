"""Explicit one-/two-arm packet loading and preview identity; no execution permission."""

from pathlib import Path

from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json, file_hash
from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
from ada_grasp_ctrl.ros_execution.shared_packet import load_shared_packet, KIND
from ada_grasp_ctrl.ros_execution.shared_hardware import load_hardware_packet, KIND as HARDWARE_KIND


def load_execution_packet(path: Path) -> dict:
    """Dispatch validated packets by their explicit transport kind."""
    kind = read_json(path).get("kind")
    if kind == KIND:
        return load_shared_packet(path)
    if kind == HARDWARE_KIND:
        return load_hardware_packet(path)
    return load_packet(path)


def preview_identity(packet: dict, path: Path) -> dict:
    """Bind displayed arms, candidate and complete packet contents to operator review."""
    if "plan_identity" in packet:
        return dict(
            required_arms=packet["plan_identity"]["active_arms"],
            candidate_id=packet["plan_identity"]["candidate_id"],
            packet_sha256=file_hash(path),
        )
    return dict(side=packet["side"], candidate_id=packet["sample_id"], packet_sha256=file_hash(path))
