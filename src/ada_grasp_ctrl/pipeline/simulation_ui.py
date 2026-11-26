"""Persistent single-viewport scene editor, review, physics and reset lifecycle."""

from copy import deepcopy
import os
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from mjviser import ViserMujocoScene
import yaml

from ada_grasp_ctrl.calibration import calibrated_snapshot, DEFAULT_CALIBRATION
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.placement.workflow import prepare_simulation_session, run_uop
from ada_grasp_ctrl.placement.ui import PlacementPanel
from ada_grasp_ctrl.placement.target_ui import PlacementTargetPanel
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json
from .dual_viewer import create_server
from .simulation_scene import SceneBuilder
from .simulation_plan import PHASES, prepare_plan, preview, verify_plan
from .simulation_executor import SimulationExecutor
from .simulation_processes import SimulationProcesses


class SimulationApp:
    """One server and one execution owner; generation IDs isolate every confirmed attempt."""

    def __init__(self, server: object, builder: SceneBuilder, args: object, root: Path) -> None:
        self.server, self.builder, self.args, self.root = server, builder, args, root
        self.placement_aware = getattr(args, "placement_aware", False)
        self.lock = threading.RLock()
        self.generation = 0
        self.state = "EDITING"
        self.attempt = None
        self.scene = self.plan = self.executor = self.processes = self.panel = None
        self.placement_folder = None
        self.worker = None
        self.closed = threading.Event()
        self.requested = builder.defaults()
        self.view = ViserMujocoScene(server, builder.display.model, num_envs=1)
        self.view.camera_tracking_enabled = False
        self.view.geom_groups_visible[0] = False
        self.view.geom_groups_visible[3] = False
        self.view._sync_visibilities()
        self.view.update_from_mjdata(builder.display.data)
        server.initial_camera.position = (1.7, -1.5, 1.1)
        server.initial_camera.look_at = (0.4, -0.4, 0)
        server.initial_camera.up = (0, 0, 1)
        self.object_handle = None
        self.status = server.gui.add_markdown("**EDITING**")
        with server.gui.add_folder("Scene Builder"):
            self.choice = server.gui.add_dropdown(
                "Object", options=list(builder.entries), initial_value=self.requested["object_id"]
            )
            corners = (
                np.array(
                    [
                        [x, y, 0]
                        for x in (-builder.half_extents[0], builder.half_extents[0])
                        for y in (-builder.half_extents[1], builder.half_extents[1])
                    ]
                )
                @ builder.rotation.T
                + builder.center
            )
            self.xy = [
                server.gui.add_slider(
                    axis + " (m)",
                    min=float(corners[:, i].min()),
                    max=float(corners[:, i].max()),
                    step=0.001,
                    initial_value=self.requested["xy_m"][i],
                )
                for i, axis in enumerate(("X", "Y"))
            ]
            self.yaw = server.gui.add_slider(
                "Yaw (deg)", min=-180, max=180, step=1, initial_value=self.requested["yaw_deg"]
            )
            entry = builder.entries[self.requested["object_id"]]
            self.scale = server.gui.add_slider(
                "Scale",
                min=entry["scale_range"][0],
                max=entry["scale_range"][1],
                step=0.001,
                initial_value=self.requested["scale"],
            )
            server.gui.add_markdown("Fixed support orientation. Height follows the table; XY stays inside its edges.")
            self.seed = server.gui.add_number("Seed", initial_value=self.requested["seed"], min=0, step=1)
            self.randomize = server.gui.add_button("Randomize")
            self.defaults_button = server.gui.add_button("Reset edit controls")
            self.confirm = server.gui.add_button("Confirm Scene")
        self.reset_button = server.gui.add_button("Reset Scene")
        self.execute_button = server.gui.add_button(
            "Execute Grasp", disabled=True, visible=not (args.placement_only or self.placement_aware)
        )
        self.progress = server.gui.add_slider(
            "Playback (%)",
            min=0,
            max=100,
            step=0.1,
            initial_value=0,
            disabled=True,
            visible=not (args.placement_only or self.placement_aware),
        )
        self.phase = server.gui.add_dropdown(
            "Phase",
            options=PHASES,
            initial_value=PHASES[0],
            disabled=True,
            visible=not (args.placement_only or self.placement_aware),
        )
        self.phase.on_update(self.select_phase)
        self.summary = server.gui.add_markdown("")
        self.edit_controls = [
            self.choice,
            *self.xy,
            self.yaw,
            self.scale,
            self.seed,
            self.randomize,
            self.defaults_button,
            self.confirm,
        ]
        for control in (self.choice, *self.xy, self.yaw, self.scale, self.seed):
            control.on_update(self.edit)
        self.confirm.on_click(self.confirm_scene)
        self.reset_button.on_click(self.reset)
        self.execute_button.on_click(self.execute)
        self.randomize.on_click(self.random_scene)
        self.defaults_button.on_click(self.reset_controls)
        self.update_object()
        self.render_thread = threading.Thread(target=self.render, daemon=True)
        self.render_thread.start()

    def transition(self, state: str, detail: str = "") -> None:
        """Publish guarded lifecycle state and persist same-attempt diagnostics."""
        self.state = state
        self.status.content = f"**{state}**\n\n{detail}"
        for control in self.edit_controls:
            control.disabled = state != "EDITING"
        self.execute_button.disabled = (
            self.placement_aware or state != "PREVIEW_READY" or self.args.preview_only or self.args.export_only
        )
        self.progress.disabled = state != "PREVIEW_READY"
        self.phase.disabled = state != "PREVIEW_READY"
        record = dict(
            schema_version=1,
            state=state,
            detail=detail,
            generation=self.generation,
            attempt=str(self.attempt) if self.attempt else None,
            port=8089,
        )
        write_json(self.root / "viewer_state.json", record)
        if self.attempt:
            write_json(self.attempt / "run_report.json", record)

    def select_phase(self, _: object = None) -> None:
        """Scrub the existing reference endpoint without advancing physical state."""
        with self.lock:
            if self.state == "PREVIEW_READY":
                stages = self.plan["schedule"]["stages"][:6]
                stage = next(s for s in stages if s["phase"] == self.phase.value)
                self.progress.value = stage["stage_end_time"] / stages[-1]["settled_time"] * 100

    def read_controls(self) -> dict:
        """Read a complete request atomically under the app lock."""
        seed = self.seed.value
        if int(seed) != seed or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        scale = (
            self.builder.entries[self.choice.value]["default_scale"]
            if self.choice.value != self.requested["object_id"]
            else self.scale.value
        )
        return self.builder.tabletop_request(
            self.choice.value, [c.value for c in self.xy], self.yaw.value, int(seed), scale=scale
        )

    def update_object(self) -> None:
        """Update cached mesh vertices only; no physics compilation or cloud sampling."""
        pose, vertices = self.builder.pose(self.requested)
        mesh = self.builder.meshes[self.requested["object_id"]]
        if self.object_handle is not None:
            self.object_handle.remove()
        self.object_handle = self.server.scene.add_mesh_simple(
            "/editable_object",
            vertices=mesh.vertices * self.requested["scale"],
            faces=mesh.faces,
            color=(70, 180, 90),
            position=pose[:3, 3],
            wxyz=Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]],
        )
        try:
            self.builder.check_footprint(vertices)
            self.status.content = "**EDITING** · " + self.builder.entries[self.requested["object_id"]]["label"]
            self.confirm.disabled = False
        except ValueError as error:
            self.status.content = str(error)
            self.confirm.disabled = True

    def edit(self, _: object = None) -> None:
        """Keep the selected support pose and fit XY to the complete object footprint."""
        with self.lock:
            if self.state != "EDITING":
                return
            try:
                self.set_controls(self.read_controls())
            except ValueError as error:
                self.status.content = str(error)
                self.confirm.disabled = True

    def set_controls(self, request: dict) -> None:
        """Apply complete deterministic edits before refreshing their visual mesh."""
        self.requested = request
        self.choice.value = request["object_id"]
        for c, v in zip(self.xy, request["xy_m"]):
            c.value = v
        self.yaw.value = request["yaw_deg"]
        self.scale.min, self.scale.max = self.builder.entries[request["object_id"]]["scale_range"]
        self.scale.value = request["scale"]
        self.seed.value = request["seed"]
        self.update_object()

    def reset_controls(self, _: object = None) -> None:
        with self.lock:
            if self.state == "EDITING":
                self.set_controls(self.builder.defaults(self.choice.value))

    def random_scene(self, _: object = None) -> None:
        """Randomize only the object, supported XY and yaw with a recorded seed."""
        with self.lock:
            if self.state != "EDITING":
                return
            try:
                seed = self.read_controls()["seed"]
                self.set_controls(self.builder.random_tabletop_request(seed))
            except ValueError as error:
                self.status.content = str(error)
                self.confirm.disabled = True

    def confirm_scene(self, _: object = None) -> None:
        """Only confirmation creates physics and runs fresh model inference."""
        with self.lock:
            if self.state != "EDITING" or self.confirm.disabled:
                return
            self.requested = self.read_controls()
            self.generation += 1
            generation = self.generation
            root = self.root / f"attempt_{generation:04d}"
            root.mkdir()
            self.attempt = root
            processes = SimulationProcesses(root)
            self.processes = processes
            request = deepcopy(self.requested)
            self.transition("SETTLING")

        def prepare() -> None:
            try:
                scene, obj, state = self.builder.confirm(request, root, processes.cancel)
                with self.lock:
                    if generation != self.generation:
                        return
                    self.scene = scene
                    if self.args.placement_only or self.placement_aware:
                        session = prepare_simulation_session(obj, root / "placement", state)
                        uop = resolve_from_root(self.args.uop_root or os.getenv("UOP_NET_ROOT") or "../uop-net")
                        python = (
                            resolve_from_root(Path(self.args.uop_python).expanduser().parent)
                            / Path(self.args.uop_python).name
                            if self.args.uop_python
                            else uop / ".venv/bin/python"
                        )
                        self.placement_folder = self.server.gui.add_folder("Placement")
                        with self.placement_folder:

                            def infer() -> list[dict]:
                                return run_uop(session, uop, python, request["seed"])

                            if self.placement_aware:
                                self.panel = PlacementTargetPanel(self, session, infer, state)
                            else:
                                self.panel = PlacementPanel(self.server, session, infer)
                        self.object_handle.visible = False
                        self.transition("PLACEMENT_EDITING" if self.placement_aware else "PLACEMENT_READY")
                        return
                    self.transition("INFERENCE/PLANNING")
                plan = prepare_plan(self.builder.config, self.args.selector, root, processes, state)
                with self.lock:
                    if generation != self.generation:
                        return
                    self.plan = plan
                    if plan is None:
                        self.transition(
                            "NO_FEASIBLE_GRASP", "See selection diagnostics; Reset Scene permits a new attempt."
                        )
                    else:
                        self.progress.value = 0
                        self.summary.content = (
                            f"Source: **{plan['source_mode']}** · execution: **{plan['execution_mode']}**\n\n"
                            f"Candidate: {plan['candidate_id']} · effort: {plan['effort']:.6f} rad²\n\n"
                            f"Stage progression: **{plan.get('stage_completion', 'arrival')}** · "
                            f"Squeeze ×{plan.get('pose_adjustment', {}).get('squeeze_extrapolate_ratio', 1):g}\n\n"
                            + " · ".join(
                                f"{r['mode']}: {r['feasible_candidate_count']}/"
                                f"{r['evaluated_candidate_count']} feasible"
                                for r in plan.get("candidate_counts", [])
                            )
                            + "\n\n"
                            "Preview is a reference; Execute Grasp starts physical simulation."
                        )
                        self.transition("PREVIEW_READY")
            except Exception as error:
                with self.lock:
                    if generation == self.generation:
                        self.transition("ERROR", f"{type(error).__name__}: {error}")
            finally:
                processes.close()

        with self.lock:
            if generation != self.generation:
                return
            self.worker = threading.Thread(target=prepare, daemon=True)
            self.worker.start()

    def execute(self, _: object = None) -> None:
        """Require explicit review, unchanged artifacts and exactly one physics owner."""
        with self.lock:
            if self.placement_aware or self.state != "PREVIEW_READY" or self.execute_button.disabled:
                return
            try:
                verify_plan(self.plan)
                if read_json(self.attempt / "simulation_plan.json") != self.plan:
                    raise ValueError("Reviewed plan changed")
                executor = SimulationExecutor(self.scene, self.plan, self.builder.config, self.attempt, self.lock)
                self.executor = executor
                generation = self.generation
                self.transition("EXECUTING")
            except Exception as error:
                self.transition("ERROR", str(error))
                return

        def run() -> None:
            try:
                result = executor.run()
            except Exception as error:
                result = dict(status="failed", error=f"{type(error).__name__}: {error}")
                write_json(executor.root / "execution_report.json", result)
            with self.lock:
                if generation == self.generation:
                    self.transition("ERROR" if result["status"] != "holding" else "HOLDING", result.get("error", ""))

        with self.lock:
            if generation != self.generation or self.executor is not executor:
                return
            self.worker = threading.Thread(target=run, daemon=True)
            self.worker.start()

    def reset(self, _: object = None, *, reason: str = "Reset Scene") -> None:
        """Invalidate callbacks first, then join physics before restoring initial display."""
        with self.lock:
            if self.state == "RESETTING":
                return
            self.generation += 1
            was_settling = self.state == "SETTLING"
            if self.attempt:
                write_json(
                    self.attempt / "cancellation.json",
                    dict(reason=reason, state=self.state, prior_report=read_json(self.attempt / "run_report.json")),
                )
            self.transition("RESETTING")
            if self.attempt:
                write_json(self.attempt / "run_report.json", dict(schema_version=1, state="CANCELED", reason=reason))
            executor, worker, processes = self.executor, self.worker, self.processes
            if executor:
                executor.cancel.set()
            if self.panel:
                self.panel.active = False
        if processes:
            processes.close()
        if (executor or was_settling) and worker:
            worker.join(timeout=10)
        with self.lock:
            if (executor or was_settling) and worker and worker.is_alive():
                self.transition("ERROR", "Physics worker has not stopped; reset refused")
                return
            if self.panel:
                if self.placement_aware:
                    self.panel.remove()
                self.panel.cloud.remove()
                self.placement_folder.remove()
            self.panel = self.placement_folder = None
            self.scene = self.plan = self.executor = self.processes = self.worker = None
            self.attempt = None
            self.builder.display.reset(self.builder.initial)
            self.view.update_from_mjdata(self.builder.display.data)
            self.progress.value = 0
            self.phase.value = PHASES[0]
            self.summary.content = ""
            self.transition("EDITING")
            self.update_object()

    def render(self) -> None:
        """Copy measured geometry under lock; no physics work or mesh retransmission here."""
        counter = 0
        while not self.closed.wait(1 / 30):
            with self.lock:
                if self.scene is None:
                    continue
                sim = self.scene.sim
                q = (
                    preview(self.plan, float(self.progress.value))
                    if self.state == "PREVIEW_READY"
                    else sim.configuration()
                )
                if self.placement_aware and self.panel is not None:
                    endpoint_q = self.panel.preview_q()
                    if endpoint_q is not None:
                        q = endpoint_q
                display = self.builder.display
                display.data.qpos[display.qpos] = q
                mujoco.mj_forward(display.model, display.data)
                self.view.update_from_mjdata(display.data)
                body = sim.data.body("object")
                self.object_handle.position = body.xpos.copy()
                self.object_handle.wxyz = body.xquat.copy()
                counter += 1
                if self.executor and counter % 3 == 0:
                    if self.executor.result["status"] == "holding" and self.state == "EXECUTING":
                        self.transition("HOLDING")
                    latest = self.executor.latest
                    if latest:
                        self.summary.content = (
                            f"**{latest['phase']}** · lift {latest['lift_height_m']:.3f} m\n\n"
                            f"Scheduled hold success: {self.executor.result['scheduled_hold_success']}\n\n"
                            f"Retained now: {latest['retained']} · "
                            f"post-hold drop: {self.executor.result['post_hold_drop_latched']}"
                        )

    def close(self) -> None:
        """Stop all owned physics before closing the sole viewport."""
        self.closed.set()
        self.reset(reason="process_exit")
        self.render_thread.join(timeout=5)


def run_simulation(args: object, config: dict, root: Path) -> int:
    """Create the existing server once and retain it across all scene attempts."""
    root.mkdir(parents=True, exist_ok=False)
    config = deepcopy(config)
    config["environment"] = "simulation"
    server = app = None
    report = dict(schema_version=1, environment="simulation", status="preflight")
    try:
        config["robot_snapshot"] = str(
            calibrated_snapshot(
                config["robot_snapshot"], root / "robot_snapshot", config.get("extrinsics") or DEFAULT_CALIBRATION
            )
        )
        config["table_plane"] = str(resolve_from_root(config["table_plane"]))
        (root / "effective_config.yaml").write_text(yaml.safe_dump(config))
        (root / "run_manifest.yaml").write_text(
            yaml.safe_dump(dict(schema_version=1, arguments=vars(args), config=config, mode="simulation"))
        )
        builder = SceneBuilder(config)
        server = create_server()
        app = SimulationApp(server, builder, args, root)
        app.transition("EDITING")
        report["status"] = "serving"
        write_json(root / "run_report.json", report)
        print("PREVIEW_URL=http://localhost:8089", flush=True)
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        report["status"] = "closed"
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if app:
            app.close()
        if server:
            server.stop()
        report["attempt_directories"] = [str(p) for p in sorted(root.glob("attempt_*"))]
        write_json(root / "run_report.json", report)
