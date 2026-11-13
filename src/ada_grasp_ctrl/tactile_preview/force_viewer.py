"""Three-vector diagnostic overlay on the existing Viser hand preview."""

from __future__ import annotations

from typing import Any

import numpy as np

from ada_grasp_ctrl.contact.compatibility import contact_frame
from ada_grasp_ctrl.optimization import friction_cone_slack
from .adapter import FingerObservation
from .data import FINGERS
from .viewer import Preview

ORANGE = (255, 140, 20)
BLUE = (40, 100, 255)
GREEN = (30, 210, 60)


def contact_diagnostic(result: FingerObservation, mu: float, source_issue: str = "") -> dict:
    """Diagnose in hand-flange coordinates without changing or filtering failing forces."""
    frame = result.frame
    row = dict(
        status="UNAVAILABLE",
        raw_status=frame.contact_status if frame else None,
        adapter_status=result.status,
        reason=source_issue or result.detail,
    )
    if source_issue or frame is None or result.transport_status != "VALID":
        return row
    if (
        frame.error
        or frame.contact_error
        or not frame.session_id
        or any(w != "sdk_frame_id_forward_gap" for w in frame.warnings)
    ):
        row["reason"] = "source error or identity warning"
        return row
    if result.status in ("MISSING", "EMPTY"):
        if frame.contact_status == result.status and frame.contact_count == 0 and not frame.contact_row:
            row.update(status="NO_CONTACT", reason="healthy source has no contact points")
        return row
    if result.status != "VALID" or result.observation is None:
        return row
    if frame.contact_count != 1:
        row.update(status="AMBIGUOUS", reason="one F6 resultant cannot be allocated to multiple contacts")
        return row
    obs = result.observation
    if obs.frame_id != "right_hand_flange" or obs.force_convention != "object_on_finger":
        row["reason"] = "unsupported coordinate frame or force convention"
        return row
    normal, force = np.asarray(obs.normal), -np.asarray(obs.force)
    local = contact_frame(normal).T @ force
    fn = float(normal @ force)
    ft = float(np.linalg.norm(force - fn * normal))
    magnitude = float(np.linalg.norm(force))
    angle = float(np.degrees(np.arccos(np.clip(fn / magnitude, -1, 1)))) if magnitude else None
    slack = float(friction_cone_slack(local, mu)[0])
    cone = "NEGATIVE_NORMAL_FORCE" if fn <= 0 else "PASS" if slack >= -1e-9 else "FAIL"
    row.update(
        status="VALID",
        frame=obs.frame_id,
        position=list(obs.position),
        normal=list(obs.normal),
        f_sharpa=list(obs.force),
        f_object=force.tolist(),
        fn_n=fn,
        ft_n=ft,
        angle_deg=angle,
        mu_required=ft / fn if fn > 0 else None,
        mu=mu,
        cone=cone,
        cone_slack_n=slack,
        force_convention=obs.force_convention,
        source=dict(
            session_id=frame.session_id,
            sequence=frame.sequence,
            sdk_frame_id=frame.sdk_frame_id,
            received=frame.received,
            receipt_ros=frame.receipt_ros,
            raw_f6=list(frame.raw),
            sdk_timestamp=frame.sdk_timestamp,
            sdk_timestamp_unit=frame.sdk_timestamp_unit,
            sdk_timestamp_epoch=frame.sdk_timestamp_epoch,
            source_frame=frame.source_frame,
        ),
    )
    return row


class ForcePreview(Preview):
    """Reuse mesh/FK, adapter, GUI and arrow primitives; only overlay forces and diagnostics."""

    def __init__(self, server: Any, hand: Any, args: Any, mu: float, geometry: Any = None) -> None:
        super().__init__(server, hand, args, geometry=geometry)
        self.mu = mu
        server.gui.configure_theme(control_width="large")
        server.initial_camera.position = (0.18, -0.25, 0.23)
        server.initial_camera.look_at = (0, 0, 0.12)
        self.intro.content = (
            "# Recorded contact forces · read only\n\n"
            "**Orange:** object on finger. **Blue:** finger on object (−orange). "
            "**Green:** outward surface normal, unchanged.\n\n"
            "All arrows use **right_hand_flange** coordinates; hand-only view. "
            "No object mesh is inferred from this recording. Geometry/alignment and force units remain provisional."
        )
        self.legend.content = (
            "Red point = contact position. Both force arrows share one m/N scale; normal length is visual."
        )
        for widget in (*self.indices, self.flip, self.mode):
            widget.visible = False
            widget.disabled = True
        self.flip.value = False
        self.mode.value = "World f_W"
        self.scale.value = 0.01
        self.zero.value = 0.0
        self.zero.visible = False
        self.object_arrows = {}
        for finger in FINGERS:
            self.arrows[finger].colors = np.array([ORANGE], dtype=np.uint8)
            self.normals[finger].colors = np.array([GREEN], dtype=np.uint8)
            self.object_arrows[finger] = server.scene.add_arrows(
                f"/object_force/{finger}",
                points=np.zeros((1, 2, 3)),
                colors=BLUE,
                shaft_radius=0.001,
                head_radius=0.003,
                head_length=0.008,
                visible=False,
            )

    def _update(self, packet: dict, connected: bool, error: str = "") -> dict:
        state = super()._update(packet, connected, error)
        diagnostics = {}
        lines = [
            f"**MPC μ = {self.mu:g}** · forces in N · angle in degrees",
            "",
            "| Finger | Fn | Ft | Angle | μ required | Cone / status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for finger, result in self.last_snapshot.fingers:
            diagnostic = contact_diagnostic(result, self.mu, packet.get("source_issues", {}).get(finger, ""))
            diagnostics[finger] = diagnostic
            for handle in (
                self.arrows[finger],
                self.object_arrows[finger],
                self.normals[finger],
                self.markers[finger],
                self.geometry_labels[finger],
            ):
                handle.visible = False
            if diagnostic["status"] == "VALID":
                point = np.asarray(diagnostic["position"])
                for handle, field, scale in (
                    (self.arrows[finger], "f_sharpa", self.scale.value),
                    (self.object_arrows[finger], "f_object", self.scale.value),
                    (self.normals[finger], "normal", self.normal_length.value),
                ):
                    handle.points = np.array([[point, point + scale * np.asarray(diagnostic[field])]])
                    handle.visible = True
                self.markers[finger].position = point
                self.markers[finger].visible = True

                def fmt(value: float | None) -> str:
                    return "—" if value is None else f"{value:.3f}"

                values = " | ".join(fmt(diagnostic[k]) for k in ("fn_n", "ft_n", "angle_deg", "mu_required"))
                lines.append(f"| {finger} | {values} | {diagnostic['cone']} |")
            else:
                label = "NO CONTACT" if diagnostic["status"] == "NO_CONTACT" else diagnostic["status"]
                lines.append(f"| {finger} | — | — | — | — | {label} |")
            self.readouts[finger].content = (
                f"**Replay interpretation: {diagnostic['status']}** {diagnostic.get('reason', '')}\n\n"
                + self.readouts[finger].content
            )
            state["fingers"][finger].update(
                arrow=self.arrows[finger].visible, contact_visible=self.markers[finger].visible
            )
        self.mapping_status.content = "\n".join(lines)
        self.connection.content = "Recorded MCAP · receipt-time playback · 1× · no ROS connection or robot commands"
        state.update(
            force_diagnostics=diagnostics,
            arm_feedback=packet.get("arm", {}),
            receipt_time_s=self.last_snapshot.evaluated_at,
        )
        self.last_state = state
        return state
