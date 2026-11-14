"""Read-only 8089 scene and force presentation; no control or SDK device interfaces."""

from __future__ import annotations

from collections import deque
from html import escape
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
import webbrowser
from typing import Any

import numpy as np
import yaml

from ada_grasp_ctrl.experiments.crx_sharpa_right_model import matrix_pose, pose_matrix
from ada_grasp_ctrl.experiments.fixed_camera import load_plane, load_snapshot, verify_calibration_acceptance
from ada_grasp_ctrl.experiments.fixed_camera_geometry import file_hash
from ada_grasp_ctrl.experiments.fixed_camera_preview import plane_mesh
from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import body_pose
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.tactile_preview.adapter import TactileAdapter, TactileSnapshot
from ada_grasp_ctrl.tactile_preview.data import FINGERS
from ada_grasp_ctrl.tactile_preview.force_viewer import GREEN, ORANGE, contact_diagnostic
from ada_grasp_ctrl.tactile_preview.geometry import GeometryCatalog, rigid_transform, transform_contact
from ada_grasp_ctrl.tactile_preview.kinematics import load_hand
from ada_grasp_ctrl.tactile_preview.__main__ import RosProcess
from .telemetry import full_feedback


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def force_rows(snapshot: TactileSnapshot, world_flange: np.ndarray, mu: float) -> dict:
    """Reuse audited point-contact diagnostics; expose an honest partial monitor estimate."""
    rows, values = {}, []
    for finger, observed in snapshot.fingers:
        row = contact_diagnostic(observed, mu)
        row.update(age_s=observed.arrival_age if observed.frame else None, adapter_status=observed.status)
        if row["status"] == "VALID":
            if row["fn_n"] < 0:
                row.update(status="INVALID", reason="negative normal projection", fn_n=None)
            else:
                p, n, f = transform_contact(world_flange, row["position"], row["normal"], row["f_object"])
                row.update(position=p.tolist(), normal=n.tolist(), f_object=f.tolist(), frame="world")
                values.append(row["fn_n"])
        elif row["status"] == "NO_CONTACT":
            row["fn_n"] = 0.0  # Only explicitly healthy no-contact, never a missing sensor.
            values.append(0.0)
        rows[finger] = row
    complete = len(values) == len(FINGERS)
    return dict(
        fingers=rows,
        total_n=sum(values) if complete else None,
        partial_n=sum(values) if values else None,
        total_status="MONITOR ESTIMATE" if complete else "PARTIAL" if values else "UNAVAILABLE",
    )


class PhaseObserver:
    """Read only new bounded event bytes; never add work to the control loop."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.offsets: dict[Path, int] = {}
        self.native: dict = {}
        self.completed_force: float | None = None
        self.scanned = -float("inf")
        self.paths: list[Path] = []

    def update(self, now: float) -> dict:
        status = read_json(self.root / "pipeline_status.json") if (self.root / "pipeline_status.json").exists() else {}
        progress = (
            read_json(self.root / "sender/progress.json") if (self.root / "sender/progress.json").exists() else {}
        )
        if now - self.scanned >= 0.5:
            self.paths = sorted((self.root / "receiver").glob("*/events.jsonl"))
            self.scanned = now
        for path in self.paths:
            # Bound both per-update work and partial-line retention; old display frames may be dropped.
            size = path.stat().st_size
            offset = self.offsets.get(path, 0)
            with path.open("rb") as stream:
                if size - offset > 524288:
                    stream.seek(size - 262144)
                    stream.readline()
                    offset = stream.tell()
                stream.seek(offset)
                data = stream.read(262144)
            boundary = data.rfind(b"\n") + 1
            self.offsets[path] = offset + boundary
            for line in data[:boundary].splitlines():
                try:
                    event = json.loads(line)
                    if event.get("kind") == "native_complete" and self.completed_force is None:
                        # current_sum_force is the measured completion value; final_sum_force is the threshold.
                        completed = event.get("diagnostics", {}).get("current_sum_force")
                        if isinstance(completed, (float, int)) and math.isfinite(completed) and completed >= 0:
                            self.completed_force = float(completed)
                    if event.get("kind") in ("mpc_proposal", "native_complete") and event.get(
                        "time", 0
                    ) >= self.native.get("time", 0):
                        self.native = event
                except (ValueError, TypeError):
                    continue
        phase = status.get("stage", "preparing")
        state = status.get("state", "waiting")
        terminal = phase == "pipeline"
        if terminal:
            phase = state
        elif progress:
            phase = (
                "dynamic lift planning"
                if progress["kind"] in ("native_complete_ack", "lift_snapshot", "lift_planning_started")
                else progress.get("phase", phase)
            )
        diagnostics = self.native.get("diagnostics", {})
        native_age = now - self.native.get("time", -float("inf"))
        if phase == "grasp" and 0 <= native_age <= 0.5:
            phase = "MPC squeeze" if diagnostics.get("native_stage") == 2 else "MPC grasp"
        force = diagnostics.get("current_sum_force")
        if not isinstance(force, (float, int)) or not math.isfinite(force):
            force = None
        return dict(
            phase=phase,
            state=state,
            reason=status.get("reason"),
            native_force_n=force,
            completed_force_n=self.completed_force,
            native_force_live=phase.startswith("MPC ") and 0 <= native_age <= 0.5,
            native_age_s=native_age if math.isfinite(native_age) else None,
            mpc_active=phase in ("grasp", "MPC grasp", "MPC squeeze"),
            terminal=terminal,
        )


class RobotScene:
    """Existing verified CRX/Sharpa meshes with kinematics only and explicit world placement."""

    def __init__(self, server: Any, config: dict, world_base: list) -> None:
        import mujoco
        from mjviser import ViserMujocoScene
        from ada_grasp_ctrl.grasp_selection.adapters.crx_sharpa import CrxSharpaAdapter

        self.world_base = rigid_transform(world_base)
        self.table_path = resolve_from_root(config["table_plane"])
        if config.get("calibration_acceptance"):
            verify_calibration_acceptance(self.table_path, resolve_from_root(config["calibration_acceptance"]))
            table = load_plane(self.table_path)
        else:
            table = load_plane(self.table_path, require_accepted=True)
        self.table_hash = file_hash(self.table_path)
        adapter = CrxSharpaAdapter(resolve_from_root(config["model_report"]), table, self.world_base)
        self.model, self.data = adapter.model, adapter.data
        self.addresses, self.limits = adapter.addresses, adapter.limits
        self.model_sha256 = adapter.model_id
        world_model = np.linalg.inv(adapter.model_T_world)
        # Transform this private presentation model once; never modify source assets or control models.
        for body in range(1, self.model.nbody):
            if self.model.body_parentid[body] == 0:
                local = pose_matrix(np.r_[self.model.body_pos[body], self.model.body_quat[body]])
                pose = matrix_pose(world_model @ local)
                self.model.body_pos[body], self.model.body_quat[body] = pose[:3], pose[3:]
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] == 0 or self.model.geom_group[geom] != 1:
                self.model.geom_rgba[geom, 3] = 0
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                # mjviser renders planes as infinite grids even at alpha zero. This private
                # kinematics-only model uses hidden boxes; the calibrated patch is drawn below.
                self.model.geom_type[geom] = mujoco.mjtGeom.mjGEOM_BOX
                self.model.geom_rgba[geom, 3] = 0
        mujoco.mj_kinematics(self.model, self.data)
        self.scene = ViserMujocoScene(server, self.model, num_envs=1)
        self.scene.camera_tracking_enabled = False
        self.scene.geom_groups_visible = [False, True, False, False, False, False]
        self.scene.refresh_visualization()
        vertices, faces = plane_mesh(table)
        server.scene.add_mesh_simple(
            "/observed_table", vertices=vertices, faces=faces, color=(125, 140, 155), opacity=0.6, side="double"
        )
        self.last_q = None

    def update(self, q: list[float]) -> np.ndarray:
        """Pose every joint from measurements; no trajectory interpolation or dynamics steps."""
        import mujoco

        values = np.asarray(q)
        if np.any(values < self.limits[:, 0] - 1e-3) or np.any(values > self.limits[:, 1] + 1e-3):
            raise ValueError("INVALID measured joint outside model limits")
        self.data.qpos[self.addresses] = values
        mujoco.mj_kinematics(self.model, self.data)
        self.scene.update_from_mjdata(self.data)
        self.scene.request_update()
        self.last_q = values.copy()
        return body_pose(self.data, "right_hand_flange")


def object_cloud(root: Path, table_hash: str) -> tuple[np.ndarray, dict]:
    """Load exactly the selected original object once, with saved calibration/selection provenance."""
    info = read_json(root / "object.json")
    if info.get("frame_id") != "world" or info.get("unit") != "m" or info["plane_sha256"] != table_hash:
        raise ValueError("Object/table world-frame provenance mismatch")
    if file_hash(root / "object.npz") != info["object_sha256"] or file_hash(Path(info["plane_path"])) != table_hash:
        raise ValueError("Object/table artifact changed")
    arrays, metadata = load_snapshot(Path(info["source_snapshot"]))
    if (
        info["snapshot_sha256"] != metadata["snapshot_sha256"]
        or info["extrinsics"]["sha256"] != metadata["extrinsics"]["sha256"]
    ):
        raise ValueError("Object/capture calibration mismatch")
    with np.load(root / "object.npz", allow_pickle=False) as archive:
        points = archive["points_world_m"].copy()
        if not np.array_equal(points, arrays["points_world_m"][archive["source_indices"]]):
            raise ValueError("Object differs from selected capture points")
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Invalid static object cloud")
    return points, info


def object_surface(root: Path, object_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the same saved visible surface as candidate preview, expressed in world coordinates."""
    from ada_grasp_ctrl.grasp_selection.preview import transform_points

    info = read_json(root / "asset.json")
    if info.get("frame_id") != "object_local" or info.get("unit") != "m":
        raise ValueError("Unsupported surface frame/units")
    if info["source_object_sha256"] != file_hash(object_root / "object.json"):
        raise ValueError("Surface belongs to a different selected object")
    if file_hash(root / "surface.npz") != info["files_sha256"]["surface.npz"]:
        raise ValueError("Surface artifact changed")
    with np.load(root / "surface.npz", allow_pickle=False) as archive:
        vertices = transform_points(archive["vertices"], rigid_transform(archive["T_world_asset"]))
        return vertices, archive["faces"].copy(), archive["colors_rgb"].copy()


def force_overlay(live: float | None, completed: float | None, status: str) -> str:
    """Read-only viewport overlay: live estimate and the latched native completion measurement."""
    live_text = "—" if live is None else f"{live:.2f} N"
    completed_text = "—" if completed is None else f"{completed:.2f} N"
    return (
        '<div id="contact-force-overlay" style="position:fixed;top:16px;left:16px;z-index:1000;'
        "pointer-events:none;background:rgba(255,255,255,.96);color:#17212b;padding:16px 22px;"
        'border-radius:10px;box-shadow:0 2px 14px #0002;min-width:240px;font-family:sans-serif">'
        '<div style="font-size:15px;font-weight:600">Live contact ΣFn</div>'
        f'<div style="font-size:34px;font-weight:700;color:#c45b00">{live_text}</div>'
        f'<div style="font-size:11px;color:#58616b">{escape(status)}</div>'
        '<div style="margin-top:14px;font-size:15px;font-weight:600">Grasp complete ΣFn</div>'
        f'<div style="font-size:34px;font-weight:700;color:#087b51">{completed_text}</div>'
        '<div style="font-size:11px;color:#58616b">Native completion · saved value</div></div>'
    )


class Dashboard:
    """Best-effort presentation only; all I/O failures remain monitor faults."""

    def __init__(
        self, server: Any, config: dict, root: Path, *, clock=time.monotonic, adapter: TactileAdapter | None = None
    ) -> None:
        self.server, self.config, self.root, self.clock = server, config, root, clock
        self.robot = None
        self.cloud = None
        self.surface = None
        self.surface_attempted = False
        self.cloud_attempted = False
        self.robot_attempted = False
        self.error = ""
        self.adapter = adapter
        self.phase = PhaseObserver(root)
        self.count = 0
        self.samples: deque = deque(maxlen=512)
        self.last_state: dict = {}
        self.last_gui = -float("inf")
        server.gui.configure_theme(control_width="large")
        self.banner = server.gui.add_markdown("**mpc_active · Preparing** · read-only actual feedback")
        self.health = server.gui.add_markdown("Waiting for actual ROS joint feedback")
        self.static = server.gui.add_markdown("Waiting for perception · Static pre-grasp object cloud")
        self.forces = server.gui.add_markdown("**Normal force (N): UNAVAILABLE**")
        self.force_hud = server.gui.add_html(force_overlay(None, None, "UNAVAILABLE"))
        self.native = server.gui.add_markdown("**15 N native completion threshold — not a force limit**")
        self.scale = server.gui.add_slider(
            "Force display scale (m/N)", min=0.001, max=0.05, step=0.001, initial_value=0.01
        )
        server.gui.add_markdown(
            "Orange: **force on object**. Green: **finger-to-object surface normal**. "
            "Contact geometry / force calibration are provisional. Receipt-aligned; not acquisition-synchronized."
        )
        self.markers, self.normals, self.arrows = {}, {}, {}
        for finger in FINGERS:
            self.markers[finger] = server.scene.add_icosphere(
                f"/contacts/{finger}", radius=0.003, color=(255, 70, 70), visible=False
            )
            self.normals[finger] = server.scene.add_arrows(
                f"/normals/{finger}",
                points=np.zeros((1, 2, 3)),
                colors=GREEN,
                shaft_radius=0.001,
                head_radius=0.003,
                head_length=0.008,
                visible=False,
            )
            self.arrows[finger] = server.scene.add_arrows(
                f"/forces/{finger}",
                points=np.zeros((1, 2, 3)),
                colors=ORANGE,
                shaft_radius=0.001,
                head_radius=0.003,
                head_length=0.008,
                visible=False,
            )
        control = yaml.safe_load(resolve_from_root("src/ada_grasp_ctrl/config/task/control_eval.yaml").read_text())
        self.mu = float(control["control"]["friction_cone_mu"])
        server.initial_camera.position = (1.65, -1.5, 1.0)
        server.initial_camera.look_at = (0.5, -0.45, 0.1)
        server.initial_camera.up = (0, 0, 1)

    def update(self, packet: dict, connected: bool, error: str = "", *, phase: dict | None = None) -> dict:
        began, cpu = self.clock(), time.process_time()
        pose_status, world_flange = "UNAVAILABLE", None
        q = None
        try:
            if not connected:
                raise ValueError("DISCONNECTED / STALE telemetry: " + error)
            q = full_feedback(packet, began)
            if self.robot is None and not self.robot_attempted:
                self.robot_attempted = True
                self.robot = RobotScene(self.server, self.config, packet["world_T_right_base"])
            if self.robot is None:
                raise ValueError(self.error or "Robot scene unavailable")
            if packet.get("model_sha256") != self.robot.model_sha256 or not np.allclose(
                packet["world_T_right_base"], self.robot.world_base, atol=1e-9, rtol=0
            ):
                raise ValueError("MODEL / WORLD TRANSFORM MISMATCH")
            world_flange = self.robot.update(q)
            pose_status = (
                "REPLAY synthetic 6+22 joints"
                if self.config.get("dashboard_replay")
                else "LIVE actual 6+22 measured joints"
            )
            self.error = ""
            if self.adapter is None:
                package = resolve_from_root(self.config["mpc_package_share"])
                hand = load_hand(package / "urdf/right_sharpa_wave_with_flange.urdf", package, load_meshes=False)
                self.adapter = TactileAdapter(
                    hand,
                    GeometryCatalog.installed_candidate(),
                    stale_after=0.2,
                    receipt_clock_synchronized=True,
                    clock=self.clock,
                )
        except Exception as exc:
            pose_status = str(exc)
            self.error = pose_status
            world_flange = None
        state = dict(
            fingers={f: dict(status="UNAVAILABLE", adapter_status=pose_status) for f in FINGERS},
            total_n=None,
            partial_n=None,
            total_status="UNAVAILABLE",
        )
        if self.adapter is not None and world_flange is not None:
            try:
                snapshot = self.adapter.update(
                    packet.get("frames", {}),
                    packet.get("joints", {}),
                    connected=connected,
                    endpoints=packet.get("endpoints", {}),
                    errors=packet.get("errors", {}),
                    allow_unverified_geometry=True,
                )
                state = force_rows(snapshot, world_flange, self.mu)
            except Exception as exc:
                self.error = "Tactile monitor: " + str(exc)
        for finger in FINGERS:
            row = state["fingers"][finger]
            valid = row["status"] == "VALID"
            for handle in (self.markers[finger], self.normals[finger], self.arrows[finger]):
                handle.visible = valid
            if valid:
                point = np.asarray(row["position"])
                self.markers[finger].position = point
                self.normals[finger].points = np.array([[point, point + 0.025 * np.asarray(row["normal"])]])
                self.arrows[finger].points = np.array([[point, point + self.scale.value * np.asarray(row["f_object"])]])
        if (
            self.robot is not None
            and not self.cloud_attempted
            and (self.root / "grouplive/object/object.json").exists()
            and (self.root / "grouplive/object/run_report.json").exists()
        ):
            self.cloud_attempted = True
            try:
                points, info = object_cloud(self.root / "grouplive/object", self.robot.table_hash)
                self.cloud = self.server.scene.add_point_cloud(
                    "/static_object", points=points, colors=(205, 35, 175), point_size=0.004
                )
                self.server.scene.add_label(
                    "/static_object_label",
                    text="Static object cloud",
                    position=points.mean(axis=0) + np.array([0, 0, 0.07]),
                    depth_test=False,
                )
                self.static.content = (
                    f"**Static pre-grasp object cloud** · {len(points)} points · cluster {info['cluster_id']}"
                    "\n\nNot tracking the object; stays fixed during Lift and later motion."
                )
            except Exception as exc:
                self.static.content = "Object overlay UNAVAILABLE: " + str(exc)
        asset_root = self.root / "grouplive/asset"
        if (
            self.cloud is not None
            and not self.surface_attempted
            and (asset_root / "asset.json").exists()
            and (asset_root / "run_report.json").exists()
        ):
            self.surface_attempted = True
            try:
                import trimesh

                vertices, faces, colors = object_surface(asset_root, self.root / "grouplive/object")
                # Candidate preview renders both saved triangles and colored points. Keep both static.
                self.surface = self.server.scene.add_mesh_trimesh(
                    "/measured_surface",
                    trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False),
                )
                self.cloud.points, self.cloud.colors, self.cloud.point_size = vertices, colors, 0.0015
                self.static.content += "\n\nCandidate preview display: recorded surface + colored points."
            except Exception as exc:
                self.static.content += "\n\nRecorded surface UNAVAILABLE: " + str(exc)
        try:
            stage = phase if phase is not None else self.phase.update(began)
        except Exception as exc:
            stage = dict(phase="status unavailable", reason=str(exc), mpc_active=False)
        if began - self.last_gui >= 0.2:
            self.last_gui = began
            self.force_hud.content = force_overlay(
                state["total_n"], stage.get("completed_force_n"), state["total_status"]
            )
            self.banner.content = f"**mpc_active · {stage['phase']}** · " + (
                "MPC episode" if stage.get("mpc_active") else "Linear/preparation stage · MPC inactive"
            )
            if self.config.get("dashboard_replay"):
                self.banner.content = "**SYNTHETIC REPLAY · no ROS**\n\n" + self.banner.content
            self.health.content = pose_status + (
                " · last robot pose retained" if world_flange is None and self.robot else ""
            )
            if self.error and world_flange is not None:
                self.health.content += "\n\n" + self.error
            if stage.get("reason"):
                self.health.content += "\n\n" + stage["reason"]
            lines = ["| Finger | Fn (N) | Status / age |", "|---|---:|---|"]
            for finger, row in state["fingers"].items():
                force = "—" if row.get("fn_n") is None else f"{row['fn_n']:.2f}"
                age = "—" if row.get("age_s") is None else f"{row['age_s']:.3f}s"
                lines.append(f"| {finger} | {force} | {row['status']} · {row.get('adapter_status', '')} · {age} |")
            total = "—" if state["total_n"] is None else f"{state['total_n']:.2f} N"
            lines += [f"\n**Eligible normal sum: {total} · {state['total_status']}**"]
            if state["total_n"] is None and state["partial_n"] is not None:
                lines += [f"Known eligible subtotal: {state['partial_n']:.2f} N (incomplete)"]
            self.forces.content = "\n".join(lines)
            native = stage.get("native_force_n")
            label = "live MPC event" if stage.get("native_force_live") else "last MPC event; historical"
            self.native.content = "**15 N native completion threshold — not a force limit**\n\n"
            self.native.content += (
                "MPC completion force: —" if native is None else f"MPC completion force: **{native:.2f} N** ({label})"
            )
            if not stage.get("mpc_active"):
                self.native.content += "\n\nThreshold is context only; no MPC force regulation in this stage."
        lag = max(0.0, began - packet.get("worker_monotonic", began)) if connected else None
        state.update(
            pose_status=pose_status,
            phase=stage,
            connected=connected,
            measured_q=q,
            static_cloud_loaded=self.cloud is not None,
            static_surface_loaded=self.surface is not None,
            telemetry_lag_s=lag,
        )
        self.samples.append(dict(update_s=self.clock() - began, cpu_s=time.process_time() - cpu, lag_s=lag))
        self.count += 1
        self.last_state = state
        return state


def start_reader(config: dict) -> RosProcess:
    """Source only the existing ROS environment; the child contains subscriptions alone."""
    report = read_json(resolve_from_root(config["model_report"]))
    command = [
        "/bin/bash",
        "-c",
        'source "$1" && source "$2" && shift 2 && exec nice -n 10 "$@"',
        "ada-monitor",
        config["ros_setup"],
        config["workspace_setup"],
        config["ros_python"],
        "-m",
        "ada_grasp_ctrl.pipeline.telemetry",
        "--model-urdf",
        report["model"]["urdf"],
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = str(resolve_from_root("src"))
    environment["PATH"] = "/usr/bin:/bin:" + environment.get("PATH", "")
    environment["ROS_DOMAIN_ID"] = str(config["ros_domain_id"])
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return RosProcess(config["ros_python"], "unused", command=command, environment=environment)


def serve_dashboard(status: Path, ready: Path, config_path: Path, *, replay: Any = None, duration: float = 0) -> None:
    """Own one 8089 server; a stopped reader or browser never affects the controller."""
    import viser

    with socket.socket() as probe:
        # Allow a restart after closed connections; an active listener still prevents binding.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", 8089))
        except OSError as exc:
            raise RuntimeError("Port 8089 is occupied; refusing another port") from exc
    config = read_json(config_path)
    server = viser.ViserServer(host="127.0.0.1", port=8089, label="Right CRX + Sharpa · Live monitor")
    reader = dashboard = None
    observed_start, cpu_start = time.monotonic(), time.process_time()
    serializer = server.get_scene_serializer() if replay is not None else None
    try:
        if server.get_port() != 8089:
            raise RuntimeError("Port 8089 became occupied")
        ready.write_text(json.dumps(dict(port=8089, observational=True)))
        if replay is None:
            threading.Thread(target=webbrowser.open, args=("http://localhost:8089",), daemon=True).start()
        dashboard = Dashboard(server, config, status.parent)
        startup_error = ""
        try:
            reader = replay if replay is not None else start_reader(config)
        except Exception as exc:
            startup_error = "Telemetry unavailable: " + str(exc)
        began, saved = time.monotonic(), 0.0
        while not duration or time.monotonic() - began < duration:
            now = time.monotonic()
            packet, connected, error = reader.snapshot() if reader else ({}, False, startup_error)
            try:
                state = dashboard.update(packet, connected, error)
                if serializer is not None:
                    serializer.insert_sleep(0.05)
                if now - saved >= 1:
                    saved = now
                    temporary = status.parent / "dashboard_state.tmp"
                    temporary.write_text(json.dumps(state, allow_nan=False))
                    temporary.replace(status.parent / "dashboard_state.json")
            except Exception as exc:
                dashboard.health.content = "MONITOR ERROR: " + str(exc)
                for handles in (dashboard.markers, dashboard.normals, dashboard.arrows):
                    for handle in handles.values():
                        handle.visible = False
            time.sleep(max(0.001, 0.05 - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.close()
        if dashboard is not None:
            (status.parent / "dashboard_metrics.json").write_text(
                json.dumps(
                    dict(
                        updates=dashboard.count,
                        elapsed_s=time.monotonic() - observed_start,
                        process_cpu_s=time.process_time() - cpu_start,
                        recent_samples=list(dashboard.samples),
                    ),
                    indent=2,
                )
            )
        if serializer is not None:
            (status.parent / "dashboard_scene.viser").write_bytes(serializer.serialize())
        server.stop()
