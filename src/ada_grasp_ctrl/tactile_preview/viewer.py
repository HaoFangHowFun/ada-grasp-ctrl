"""Viser URDF kinematics and tactile display, with no physics/control runtime."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yourdfpy

from ada_grasp_ctrl.experiments.right_ros_contract import HAND_NAMES
from .data import FINGERS, Frame, force_to_world, frame_status, select_force
from .geometry import GeometryCatalog, transform_contact


def load_hand(path: Path, package_share: Path) -> yourdfpy.URDF:
    """Load original URDF visual meshes and FK without copying or altering assets."""

    def filename_handler(fname: str) -> str:
        filename = fname
        prefix = "package://dual_sharpa_wave/"
        if filename.startswith(prefix):
            return str((package_share / filename[len(prefix) :]).resolve(strict=True))
        if filename.startswith("package://"):
            raise ValueError(f"Unsupported mesh package: {filename}")
        return str((path.parent / filename).resolve(strict=True))

    hand = yourdfpy.URDF.load(str(path), filename_handler=filename_handler, load_collision_meshes=False)
    if hand.base_link != "right_hand_flange" or set(hand.actuated_joint_names) != set(HAND_NAMES):
        raise ValueError("Expected the official right-hand-only URDF with 22 named joints and right_hand_flange root")
    for finger in FINGERS:
        hand.get_transform(f"right_{finger}_elastomer")
    return hand


def joint_configuration(
    packet: dict[str, Any],
    hand: yourdfpy.URDF,
    now: float,
    stale_after: float,
    synchronized: bool,
) -> tuple[np.ndarray | None, str]:
    """Reject incomplete/stale poses instead of extrapolating or padding missing joints."""
    if not packet:
        return None, "STATIC / UNCERTAIN: neutral or last pose; no joint feedback"
    if packet.get("error"):
        return None, "INVALID POSE: " + packet["error"]
    age = now - packet["received"]
    if age < -0.05 or age > stale_after:
        return None, "STALE POSE: last posture retained; world arrows hidden"
    if synchronized:
        source_age = packet["subscriber_ros"] - packet["receipt_ros"] + age
        if packet["receipt_ros"] <= 0 or source_age < -0.05 or source_age > stale_after:
            return None, "STALE / CLOCK ERROR POSE: world arrows hidden"
    names, positions = packet["names"], packet["positions"]
    if len(names) != len(positions) or len(set(names)) != len(names) or set(names) != set(hand.actuated_joint_names):
        return None, "INVALID POSE: expected all 22 unique right-hand joint names"
    lookup = dict(zip(names, positions))
    q = np.array([lookup[n] for n in hand.actuated_joint_names], dtype=float)
    if not np.isfinite(q).all():
        return None, "INVALID POSE: nonfinite joint position"
    for name, value in zip(hand.actuated_joint_names, q):
        limit = hand.joint_map[name].limit
        if limit is not None and not limit.lower - 1e-3 <= value <= limit.upper + 1e-3:
            return None, f"INVALID POSE: {name} outside URDF limits"
    suffix = "ROS receipt clocks asserted synchronized" if synchronized else "source age unverified"
    return q, f"LIVE hand FK ({suffix}); viewer world = fixed right_hand_flange"


class Preview:
    """One URDF model supplies both mesh transforms and elastomer force frames."""

    def __init__(
        self, server: viser.ViserServer, hand: yourdfpy.URDF, args: Any, geometry: GeometryCatalog | None = None
    ) -> None:
        self.server, self.hand, self.args = server, hand, args
        config = getattr(args, "geometry_config", None)
        self.geometry = (
            geometry
            if geometry is not None
            else (
                GeometryCatalog.from_config(config.expanduser().resolve())
                if config
                else GeometryCatalog.installed_candidate()
            )
        )
        server.scene.set_up_direction("+z")
        server.initial_camera.position = (0.35, -0.5, 0.32)
        server.initial_camera.look_at = (0, 0, 0.13)
        server.initial_camera.up = (0, 0, 1)
        self.robot = ViserUrdf(server, hand, root_node_name="/hand")
        # Viser creates joint frames at identity until update_cfg is called, even
        # for fixed joints. Assemble the static URDF before telemetry is available.
        self.robot.update_cfg(hand.cfg)
        server.scene.add_frame("/world", axes_length=0.06, axes_radius=0.0015)
        server.scene.add_label("/world_label", "Viewer world = right_hand_flange", position=(0, 0, -0.025))
        server.gui.add_markdown(
            "# Right Sharpa tactile · read only\n\n"
            "**HYPOTHESIS: selected F6 axes = elastomer axes. Alignment and force sign are unverified.**\n\n"
            "Force on fingertip is the interpretation under test; no default negation. "
            "Operator reports force in N at pad center. Other F6 channels stay raw (torque units unknown).\n\n"
            "World is the fixed hand flange, **not the laboratory/arm world**. "
            "URDF meters, right-handed XYZ, Z up; no renderer axis conversion. "
            "Axes: X red, Y green, Z blue."
        )
        self.mapping_status = server.gui.add_markdown("")
        self.geometry_status = server.gui.add_markdown("")
        self.allow_geometry = server.gui.add_checkbox(
            "Show UNVERIFIED geometry (diagnostic only)",
            initial_value=getattr(args, "allow_unverified_geometry", False),
        )
        with server.gui.add_folder("Geometry source and explicit mapping", expand_by_default=False):
            for finger in FINGERS:
                profile = self.geometry.maps.get(finger)
                server.gui.add_markdown(
                    f"**{finger}**\n\n"
                    + (
                        f"{profile.source}\n\nUnits: {profile.units}; m/unit={profile.meters_per_unit}\n\n"
                        f"UV to (col,row): `{profile.uv_to_col_row.tolist()}`\n\n"
                        f"T_E_S (translation in meters): `{profile.T_E_S.tolist()}`\n\n"
                        f"Evidence: `{profile.evidence}`"
                        if profile
                        else "UNAVAILABLE: " + self.geometry.errors.get(finger, "no profile")
                    )
                )
        self.indices = [
            server.gui.add_dropdown(
                f"Assumed F{axis} raw index", options=tuple(map(str, range(6))), initial_value=str(index)
            )
            for axis, index in zip("xyz", args.force_indices)
        ]
        self.flip = server.gui.add_checkbox("Diagnostic sign flip (visualization only)", initial_value=False)
        self.scale = server.gui.add_slider(
            "Arrow length (m / presumed N)", min=0.001, max=0.1, step=0.001, initial_value=0.02
        )
        self.zero = server.gui.add_slider("Hide below presumed N", min=0.0, max=0.5, step=0.005, initial_value=0.02)
        self.normal_length = server.gui.add_slider(
            "Normal length (display only, m)", min=0.005, max=0.08, step=0.001, initial_value=0.025
        )
        self.selected = server.gui.add_dropdown("Finger axes / local vector", options=FINGERS, initial_value="index")
        self.mode = server.gui.add_dropdown(
            "Force coordinate display", options=("World f_W", "Local f_E"), initial_value="World f_W"
        )
        self.pose_status = server.gui.add_markdown("")
        self.connection = server.gui.add_markdown("")
        server.gui.add_markdown(
            "**Red point = contact row 0; blue = surface normal; green = fingertip F6 resultant.** "
            "Both arrows start at the red point. Placing resultant F6 there is a visualization approximation, "
            "not per-contact force allocation or wrench relocation. Missing contact hides all three. "
            "Near-zero force hides only green. Local mode shows point and vectors in the selected elastomer axes."
        )
        self.axes, self.labels, self.arrows, self.readouts = {}, {}, {}, {}
        self.markers, self.normals, self.geometry_labels = {}, {}, {}
        for finger in FINGERS:
            self.axes[finger] = server.scene.add_frame(f"/axes/{finger}", axes_length=0.025, axes_radius=0.0007)
            self.labels[finger] = server.scene.add_label(f"/labels/{finger}", finger)
            self.arrows[finger] = server.scene.add_arrows(
                f"/force/{finger}",
                points=np.zeros((1, 2, 3)),
                colors=(30, 210, 60),
                shaft_radius=0.001,
                head_radius=0.003,
                head_length=0.008,
                visible=False,
            )
            self.markers[finger] = server.scene.add_icosphere(
                f"/contact/{finger}", radius=0.002, color=(255, 30, 30), visible=False
            )
            self.normals[finger] = server.scene.add_arrows(
                f"/normal/{finger}",
                points=np.zeros((1, 2, 3)),
                colors=(40, 100, 255),
                shaft_radius=0.0007,
                head_radius=0.002,
                head_length=0.005,
                visible=False,
            )
            self.geometry_labels[finger] = server.scene.add_label(
                f"/geometry_warning/{finger}", f"{finger}: UNVERIFIED geometry", visible=False
            )
            with server.gui.add_folder(finger, expand_by_default=True):
                self.readouts[finger] = server.gui.add_markdown("NO DATA")
        self.local_origin = np.array([0.22, 0, 0.1])
        self.local_axes = server.scene.add_frame(
            "/local", position=self.local_origin, axes_length=0.06, axes_radius=0.001, visible=False
        )
        self.local_label = server.scene.add_label(
            "/local_label",
            "Selected elastomer coordinates: p_E, n_E, f_E",
            position=self.local_origin + [0, 0, -0.025],
            visible=False,
        )
        self.local_arrow = server.scene.add_arrows(
            "/local_force",
            points=np.zeros((1, 2, 3)),
            colors=(30, 210, 60),
            shaft_radius=0.001,
            head_radius=0.003,
            head_length=0.008,
            visible=False,
        )
        self.local_marker = server.scene.add_icosphere(
            "/local_contact", radius=0.002, color=(255, 30, 30), visible=False
        )
        self.local_normal = server.scene.add_arrows(
            "/local_normal",
            points=np.zeros((1, 2, 3)),
            colors=(40, 100, 255),
            shaft_radius=0.0007,
            head_radius=0.002,
            head_length=0.005,
            visible=False,
        )
        self.local_warning = server.scene.add_label("/local_warning", "UNVERIFIED geometry", visible=False)
        self.last_state: dict[str, Any] = {}

    def update(self, packet: dict[str, Any], connected: bool, error: str = "") -> dict[str, Any]:
        """Send each marker/normal/force update atomically to avoid mixed-frame graphics."""
        with self.server.atomic():
            return self._update(packet, connected, error)

    def _update(self, packet: dict[str, Any], connected: bool, error: str = "") -> dict[str, Any]:
        """Render an owned snapshot and expose the same state for integration checks."""
        now = time.monotonic()
        synchronized = self.args.receipt_clock_synchronized
        joints = packet.get("joints", {})
        q, pose_state = joint_configuration(joints, self.hand, now, self.args.stale_after, synchronized)
        if q is not None and connected:
            self.robot.update_cfg(q)
        self.pose_status.content = f"**{pose_state}**\n\nJoint ROS stamp: {joints.get('receipt_ros', 'unavailable')}"
        self.connection.content = (
            f"**{'ROS reader connected' if connected else 'DISCONNECTED / OFFLINE'}** {error}\n\n"
            + (
                "Freshness: ROS receipt age + monotonic arrival age; synchronized clocks asserted."
                if synchronized
                else "Freshness: local subscriber arrival age only. Source receipt age UNVERIFIED; "
                "SDK timestamp epoch/unit unknown and never used for freshness."
            )
        )
        indices = tuple(int(widget.value) for widget in self.indices)
        mapping_valid = len(set(indices)) == 3
        self.mapping_status.content = (
            f"**{'UNVERIFIED mapping' if mapping_valid else 'INVALID mapping: indices must differ'}**: "
            f"Fx/Fy/Fz = raw {indices}. All force values in presumed N pending channel verification. "
            + ("**SIGN FLIP ACTIVE (display only)**" if self.flip.value else "Raw signs preserved.")
        )
        unresolved = [f for f in FINGERS if f not in self.geometry.maps or not self.geometry.maps[f].verified]
        self.geometry_status.content = (
            "**⚠ UNVERIFIED GEOMETRY — "
            + ("DIAGNOSTIC DISPLAY ACTIVE" if self.allow_geometry.value else "HIDDEN")
            + "**\n\n"
            + ", ".join(unresolved)
            + ". Pixel mapping, units, finger model, S-to-E alignment and outward-normal convention "
            "require verification. Inspect explicit profile details below."
            if unresolved
            else "Geometry verification evidence supplied by configuration; inspect profile details."
        )
        local_mode = self.mode.value == "Local f_E"
        self.local_axes.visible = self.local_label.visible = local_mode
        self.local_arrow.visible = self.local_marker.visible = self.local_normal.visible = (
            self.local_warning.visible
        ) = False
        state = dict(pose=pose_state, connected=connected, fingers={})
        for finger in FINGERS:
            pose = self.hand.get_transform(f"right_{finger}_elastomer")
            origin, rotation = pose[:3, 3], pose[:3, :3]
            self.axes[finger].position = origin
            self.axes[finger].wxyz = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
            self.axes[finger].visible = finger == self.selected.value
            self.labels[finger].position = origin + [0, 0, 0.015]
            arrow = self.arrows[finger]
            marker, normal_arrow, warning = self.markers[finger], self.normals[finger], self.geometry_labels[finger]
            arrow.visible = marker.visible = normal_arrow.visible = warning.visible = False
            endpoints = packet.get("endpoints", {}).get(finger, [])
            endpoint_ok = len(endpoints) == 1 and endpoints[0]["type"] == "dual_sharpa_wave/msg/TactileFrame"
            endpoint_text = (
                "; ".join(f"{e['node']} {e['reliability']}/{e['durability']} depth={e['depth']}" for e in endpoints)
                or "no publisher discovered"
            )
            payload = packet.get("frames", {}).get(finger)
            rejection = packet.get("errors", {}).get(finger)
            if payload is None:
                status = "INVALID: " + rejection if rejection else "NO DATA"
                self.readouts[finger].content = f"**{status}**\n\n{endpoint_text}"
                state["fingers"][finger] = dict(
                    status=status, contact_status="NO DATA", arrow=False, contact_visible=False
                )
                continue
            frame = Frame(**payload)
            status, arrival_age, receipt_age = frame_status(frame, now, self.args.stale_after, synchronized)
            if not connected:
                status = "DISCONNECTED"
            elif not endpoint_ok:
                status = "NO / AMBIGUOUS PUBLISHER"
            rate = packet.get("rates", {}).get(finger, 0)
            f_e = (
                select_force(frame.raw, indices, self.flip.value) if frame.status == "VALID" and mapping_valid else None
            )
            f_w = force_to_world(rotation, f_e) if f_e is not None else None
            norm = float(np.linalg.norm(f_e)) if f_e is not None else None
            # A recent pose may still be unrelated to the current force sample.
            skew = (
                abs(frame.receipt_ros - joints.get("receipt_ros", 0))
                if synchronized
                else abs(frame.received - joints.get("received", 0))
            )
            pose_ok = q is not None and skew <= self.args.pose_skew
            contact_status, geometry_note, contact_state = frame.contact_status, frame.contact_error, {}
            if status != "VALID":
                contact_status = (
                    status if status in ("STALE", "CLOCK_ERROR", "DISCONNECTED") else "F6/TRANSPORT " + status
                )
            elif not mapping_valid:
                contact_status = "INVALID FORCE MAPPING"
            elif frame.contact_status == "VALID":
                try:
                    contact = self.geometry.lookup(finger, frame.contact_row[:2], self.allow_geometry.value)
                    p_w, n_w, _ = transform_contact(pose, contact.p_E, contact.n_E, f_e)
                    contact_state = dict(
                        p_E=contact.p_E.tolist(),
                        n_E=contact.n_E.tolist(),
                        p_W=p_w.tolist() if pose_ok else None,
                        n_W=n_w.tolist() if pose_ok else None,
                        pixel_row=contact.row,
                        pixel_col=contact.col,
                        geometry=contact.verification,
                    )
                    geometry_note = (
                        f"**{contact.verification}** · pixel(row,col)=({contact.row},{contact.col})\n\n"
                        f"p_S ({contact.units}): `{contact.p_S.round(5).tolist()}`\n\n"
                        f"n_S: `{contact.n_S.round(5).tolist()}`\n\n"
                        f"p_E (m): `{contact.p_E.round(6).tolist()}` · n_E: `{contact.n_E.round(5).tolist()}`\n\n"
                        f"p_W (m): `{contact_state['p_W']}` · n_W: `{contact_state['n_W']}`"
                    )
                    visible = pose_ok if not local_mode else finger == self.selected.value
                    if not local_mode and not pose_ok:
                        contact_status = "POSE UNAVAILABLE / STALE / SKEWED"
                    if visible:
                        point, direction, force = (
                            (p_w, n_w, f_w)
                            if not local_mode
                            else (self.local_origin + contact.p_E, contact.n_E, np.asarray(f_e))
                        )
                        handles = (
                            (marker, normal_arrow, arrow, warning)
                            if not local_mode
                            else (self.local_marker, self.local_normal, self.local_arrow, self.local_warning)
                        )
                        point_handle, normal_handle, force_handle, warning_handle = handles
                        point_handle.position, point_handle.visible = point, True
                        normal_handle.points = np.array([[point, point + self.normal_length.value * direction]])
                        normal_handle.visible = True
                        if norm is not None and norm > self.zero.value:
                            force_handle.points = np.array([[point, point + self.scale.value * force]])
                            force_handle.visible = True
                        warning_handle.position = point + [0, 0, 0.012]
                        warning_handle.visible = contact.verification == "UNVERIFIED DIAGNOSTIC"
                except ValueError as exc:
                    contact_status, geometry_note = "GEOMETRY ERROR / UNVERIFIED", str(exc)
            pose_note = "" if pose_ok else "WORLD VECTOR UNCERTAIN: missing/stale/skewed pose; world arrow hidden"
            self.readouts[finger].content = (
                f"**{status}** · {rate:.2f} Hz measured\n\n"
                f"F6 raw: `{list(frame.raw)}` · shape `{list(frame.shape)}`\n\n"
                f"Selected f_E: `{f_e}` · f_W: `{None if f_w is None else f_w.round(5).tolist()}`\n\n"
                f"Norm (presumed N): `{norm}`\n\n"
                f"**Contact: {contact_status}** · block={frame.contact_status} · N={frame.contact_count} · "
                f"selected_index={'0' if frame.contact_count else 'none'}\n\n"
                f"CONTACT_POINT row 0 (extra columns unknown): `{list(frame.contact_row)}` · "
                f"shape `{list(frame.contact_shape)}`\n\n"
                f"(u,v): `{list(frame.contact_row[:2])}`\n\n{geometry_note}\n\n"
                f"Arrival age: {arrival_age:.3f} s · receipt age: "
                f"{f'{receipt_age:.3f} s' if receipt_age is not None else 'UNVERIFIED'}\n\n"
                f"ROS receipt: `{frame.receipt_ros:.9f}` · source frame: `{frame.source_frame}`\n\n"
                f"SDK frame ID: `{frame.sdk_frame_id}` (raw `{frame.sdk_frame_id_raw}`) · sequence {frame.sequence}\n\n"
                f"SDK ts: `{frame.sdk_timestamp_repr}` · unit={frame.sdk_timestamp_unit}, "
                f"epoch={frame.sdk_timestamp_epoch}"
                f"\n\nSession: `{frame.session_id}`\n\n{endpoint_text}\n\n{pose_note}\n\n"
                f"{frame.error} {'; '.join(frame.warnings)}"
            )
            state["fingers"][finger] = dict(
                status=status,
                arrow=arrow.visible,
                norm=norm,
                origin=origin.tolist(),
                f_E=f_e,
                f_W=None if f_w is None else f_w.tolist(),
                contact_status=contact_status,
                contact_visible=marker.visible
                or (local_mode and finger == self.selected.value and self.local_marker.visible),
                contact_count=frame.contact_count,
                contact_row=list(frame.contact_row),
                **contact_state,
            )
        self.last_state = state
        return state
