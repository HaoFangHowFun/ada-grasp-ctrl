"""Versioned replay of immutable TactileAdapter output, without SDK/geometry parsing."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from ada_grasp_ctrl.mpc.contracts import JointFeedback
from ada_grasp_ctrl.tactile_preview.adapter import FingerObservation, TactileSnapshot
from ada_grasp_ctrl.tactile_preview.data import Frame
from .tactile_observation import GeometryProvenance, TactileContactObservation


def encode_snapshot(snapshot: TactileSnapshot) -> dict:
    """Serialize adapter results with original receipt, SDK and geometry provenance."""
    return asdict(snapshot)


def decode_snapshot(value: dict) -> TactileSnapshot:
    """Restore types, preserving invalid/empty statuses and original sequence order."""
    rows = []
    for finger, raw in value["fingers"]:
        raw = dict(raw)
        if raw.get("frame") is not None:
            raw["frame"] = Frame(**raw["frame"])
        for field in ("observation", "local"):
            if raw.get(field) is not None:
                observation = dict(raw[field])
                observation["geometry"] = GeometryProvenance(**observation["geometry"])
                raw[field] = TactileContactObservation(**observation)
        rows.append((finger, FingerObservation(**raw)))
    configuration = value["configuration"]
    return TactileSnapshot(
        value["evaluated_at"],
        tuple(rows),
        None if configuration is None else tuple(configuration),
        value["pose_status"],
    )


class TactileReplay:
    """Arrival-ordered, zero-order replay; reading a frame never changes its age.

    JSON contains schema_version=1, kind=ada_tactile_adapter_replay, origin_s,
    provenance (including synthetic/recorded), tactile_joint_names, and events.
    Each event has delivered_at, snapshot, and full JointFeedback. SDK regressions
    remain in the file and are rejected by the estimator, not sorted away.
    """

    def __init__(self, path: Path) -> None:
        def reject(value: str) -> None:
            raise ValueError(f"Nonfinite replay JSON: {value}")

        payload = json.loads(path.read_text(), parse_constant=reject)
        if payload.get("schema_version") != 1 or payload.get("kind") != "ada_tactile_adapter_replay":
            raise ValueError("Unsupported tactile replay schema")
        if not payload.get("provenance") or not payload.get("events"):
            raise ValueError("Replay needs provenance and events")
        self.origin = float(payload["origin_s"])
        self.joint_names = tuple(payload["tactile_joint_names"])
        self.provenance = payload["provenance"]
        self.events = []
        last = self.origin
        for event in payload["events"]:
            delivered = float(event["delivered_at"])
            if not math.isfinite(delivered) or delivered < last:
                raise ValueError("Replay delivery clock must be finite and nondecreasing")
            snapshot = decode_snapshot(event["snapshot"])
            feedback = JointFeedback(**event["feedback"])
            if snapshot.evaluated_at > delivered or feedback.receive_timestamp > delivered:
                raise ValueError("Replay delivery predates observation/feedback")
            self.events.append((delivered, snapshot, feedback))
            last = delivered
        self.index = -1
        self.now = self.origin

    def sample(self, now: float) -> tuple[TactileSnapshot, JointFeedback] | None:
        """Return the latest delivered snapshot; no frame means unavailable, not empty."""
        if not math.isfinite(now) or now < self.now:
            raise ValueError("Replay clock regressed")
        self.now = now
        while self.index + 1 < len(self.events) and self.events[self.index + 1][0] <= now:
            self.index += 1
        return None if self.index < 0 else self.events[self.index][1:]
