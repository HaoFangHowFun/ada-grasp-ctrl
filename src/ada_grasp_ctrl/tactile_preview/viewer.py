"""Viser URDF kinematics and tactile display, with no physics/control runtime."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yourdfpy

from .data import FINGERS
from .adapter import TactileAdapter
from .kinematics import load_hand as load_hand, joint_configuration as joint_configuration
from .geometry import GeometryCatalog


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
        self.adapter = TactileAdapter(
            hand,
            self.geometry,
            stale_after=args.stale_after,
            pose_skew=args.pose_skew,
            receipt_clock_synchronized=args.receipt_clock_synchronized,
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
        self.intro = server.gui.add_markdown(
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
        self.legend = server.gui.add_markdown(
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
            with server.gui.add_folder(finger, expand_by_default=not getattr(args, "force_replay", False)):
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
        synchronized = self.args.receipt_clock_synchronized
        joints = packet.get("joints", {})
        indices = tuple(int(widget.value) for widget in self.indices)
        self.adapter.geometry = self.geometry
        snapshot = self.adapter.update(
            packet.get("frames", {}),
            joints,
            connected=connected,
            endpoints=None if getattr(self.args, "force_replay", False) else packet.get("endpoints", {}),
            errors=packet.get("errors", {}),
            force_indices=indices,
            allow_unverified_geometry=self.allow_geometry.value,
        )
        self.last_snapshot = snapshot
        derived = dict(snapshot.fingers)
        q, pose_state = snapshot.configuration, snapshot.pose_status
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
            endpoint_text = (
                "; ".join(f"{e['node']} {e['reliability']}/{e['durability']} depth={e['depth']}" for e in endpoints)
                or "no publisher discovered"
            )
            result = derived[finger]
            frame = result.frame
            if frame is None:
                status = "INVALID: " + result.detail if result.detail else "NO DATA"
                self.readouts[finger].content = f"**{status}**\n\n{endpoint_text}"
                state["fingers"][finger] = dict(
                    status=status, contact_status=result.status, arrow=False, contact_visible=False
                )
                continue
            status, arrival_age, receipt_age = result.transport_status, result.arrival_age, result.receipt_age
            rate = packet.get("rates", {}).get(finger, 0)
            sign = -1 if self.flip.value else 1
            f_e = tuple(sign * x for x in result.force_local) if result.force_local is not None else None
            f_w = sign * np.asarray(result.force_base) if result.force_base is not None else None
            norm = float(np.linalg.norm(f_e)) if f_e is not None else None
            pose_ok = result.force_base is not None
            contact_status, geometry_note, contact_state = result.status, result.detail, {}
            local, base = result.local, result.observation
            if local is not None:
                contact_state = dict(
                    p_E=list(local.position),
                    n_E=list(local.normal),
                    p_W=list(base.position) if base else None,
                    n_W=list(base.normal) if base else None,
                    pixel_row=local.pixel_row,
                    pixel_col=local.pixel_col,
                    geometry=local.geometry.verification,
                )
                geometry_note = (
                    f"**{local.geometry.verification}** · pixel(row,col)=({local.pixel_row},{local.pixel_col})\n\n"
                    f"p_S ({local.geometry.units}): `{result.source_position}` · n_S: `{result.source_normal}`\n\n"
                    f"p_E (m): `{local.position}` · n_E: `{local.normal}`\n\n"
                    f"p_W (m): `{contact_state['p_W']}` · n_W: `{contact_state['n_W']}`\n\n"
                    f"Force convention: {local.force_convention} ({local.force_convention_quality})"
                )
            observation = local if local_mode and finger == self.selected.value else base if not local_mode else None
            if observation is not None:
                point = np.asarray(observation.position) + (self.local_origin if local_mode else 0)
                direction, force = np.asarray(observation.normal), sign * np.asarray(observation.force)
                handles = (
                    (self.local_marker, self.local_normal, self.local_arrow, self.local_warning)
                    if local_mode
                    else (marker, normal_arrow, arrow, warning)
                )
                point_handle, normal_handle, force_handle, warning_handle = handles
                point_handle.position, point_handle.visible = point, True
                normal_handle.points = np.array([[point, point + self.normal_length.value * direction]])
                normal_handle.visible = True
                if norm is not None and norm > self.zero.value:
                    force_handle.points = np.array([[point, point + self.scale.value * force]])
                    force_handle.visible = True
                warning_handle.position = point + [0, 0, 0.012]
                warning_handle.visible = observation.geometry.verification == "UNVERIFIED DIAGNOSTIC"
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
