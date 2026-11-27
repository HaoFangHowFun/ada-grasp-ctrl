"""PlacementPanel extension for finite-table XY and static single-arm endpoint review."""

import threading

import numpy as np

from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json
from ada_grasp_ctrl.pipeline.placement_preview import PlacementPreview, LABEL
from ada_grasp_ctrl.pipeline.simulation_processes import SimulationProcesses
from ada_grasp_ctrl.pipeline.placement_execution import prepare_execution, preview_points, LABEL as EXECUTION_LABEL
from ada_grasp_ctrl.pipeline.simulation_plan import preview_stage
from .session import write_json
from .target import TargetGeometry, transform_points
from .ui import PlacementPanel


class PlacementTargetPanel(PlacementPanel):
    """Reuse P00/UOP controls and bind each worker to an explicit target revision."""

    def __init__(self, app: object, session: object, infer: object, state: dict) -> None:
        self.app = app
        self.revision = 0
        self.endpoint = None
        self.executing = False
        self.preview_key = None
        self.execution_enabled = getattr(app, "placement_execute", False)
        self.syncing = False
        self.worker_lock = threading.Lock()
        self.geometry = TargetGeometry(app.builder, app.scene, session)
        self.engine = None
        super().__init__(app.server, session, infer)
        # One lock orders app Reset and target callbacks; no panel/app lock inversion.
        self.lock = app.lock
        self.processes = SimulationProcesses(app.attempt)
        app.processes = self.processes
        self.source_state = state
        self.attempt = app.attempt
        self.confirm_button.label = "Confirm Placement Target"
        marker = self.geometry.marker(session.center[:2])
        corners = np.array([[x, y, 0] for x in (-1, 1) for y in (-1, 1)], dtype=float)
        corners[:, :2] *= self.geometry.half_extents
        corners = corners @ self.geometry.table_rotation.T + self.geometry.table_center
        self.xy = [
            app.server.gui.add_slider(
                "Destination " + axis + " (m)",
                min=float(corners[:, i].min()),
                max=float(corners[:, i].max()),
                step=0.001,
                initial_value=float(marker[i]),
            )
            for i, axis in enumerate(("X", "Y"))
        ]
        self.marker = app.server.scene.add_transform_controls(
            "/placement_destination",
            scale=0.12,
            position=marker,
            active_axes=(True, True, False),
            disable_rotations=True,
            depth_test=False,
        )
        self.marker.on_update(self.drag)
        for slider in self.xy:
            slider.on_update(self.move)
        self.endpoint_choice = app.server.gui.add_dropdown(
            "Endpoint",
            options=("Grasp", "Place"),
            initial_value="Place",
            disabled=True,
            visible=not self.execution_enabled,
        )
        self.edit_button = app.server.gui.add_button("Edit Placement Target", disabled=True)
        self.edit_button.on_click(self.edit_target)
        self.target_cloud = app.server.scene.add_point_cloud(
            "/placement_target",
            points=session.points,
            colors=(70, 160, 245),
            point_size=0.003,
            visible=False,
            precision="float32",
        )
        app.server.gui.add_markdown(
            EXECUTION_LABEL
            if self.execution_enabled
            else LABEL + ". Grasp shows the adjusted squeeze posture before lift."
        )
        if self.execution_enabled:
            app.server.gui.add_markdown(
                "Blue: confirmed target. Green: preview cloud, or actual object during execution."
            )
        self.validate_target()

    def validate_target(self) -> None:
        """Check support/footprint only; editing never invokes HUGS or IK."""
        try:
            target = self.geometry.resolve(self.session, [s.value for s in self.xy], self.revision + 1)
            self.notice.content = f"Target centroid XYZ (m): {np.round(target['target_centroid_world_m'], 4).tolist()}"
            self.confirm_button.disabled = self.session.inference_status == "running"
        except ValueError as error:
            self.notice.content = str(error)
            self.confirm_button.disabled = True

    def invalidate(self) -> None:
        """Invalidate derived state while retaining cached source/UOP predictions."""
        self.revision += 1
        self.endpoint = None
        self.app.plan = None
        self.app.plan_root = None
        self.preview_key = None
        self.target_cloud.visible = False
        self.endpoint_choice.disabled = True
        self.edit_button.disabled = True
        self.session.confirmed_id = None
        self.cloud.points = self.session.select(self.choice.value.split(" — ")[0])
        self.refresh()
        self.app.summary.content = ""
        write_json(
            self.session.root / "target_state.json", dict(target_revision=self.revision, status="editing", preview=None)
        )
        self.app.transition("PLACEMENT_EDITING")

    def edit_target(self, _: object = None) -> None:
        with self.lock:
            if not self.active or self.executing:
                return
            self.invalidate()
            self.validate_target()

    def switch(self, _: object = None) -> None:
        with self.lock:
            if not self.active or self.executing:
                return
            self.invalidate()
            self.validate_target()

    def move(self, _: object = None) -> None:
        with self.lock:
            if self.active and not self.executing and not self.syncing:
                self.set_xy([s.value for s in self.xy])

    def drag(self, _: object = None) -> None:
        with self.lock:
            if self.active and not self.executing and not self.syncing:
                self.set_xy(self.marker.position[:2])

    def set_xy(self, xy: object) -> None:
        """Synchronize both controls to the realized table point without recursive edits."""
        marker = self.geometry.marker(xy)
        self.syncing = True
        try:
            self.marker.position = marker
            for slider, value in zip(self.xy, marker[:2]):
                slider.value = float(value)
        finally:
            self.syncing = False
        self.invalidate()
        self.validate_target()

    def run(self, _: object = None) -> None:
        # Keep the original one-shot adapter and failure semantics; do not hold the app
        # lock while inference runs, so Reset can invalidate the panel immediately.
        with self.lock:
            if (
                not self.active
                or self.executing
                or self.run_button.disabled
                or self.session.inference_status != "not_run"
            ):
                return
            self.run_button.disabled = True
            self.confirm_button.disabled = True
            self.notice.content = "Running UOP-Net on the original snapshot…"
        self.session.infer_once(self.infer)
        with self.lock:
            if self.active:
                self.choice.options = self.options()
                self.refresh()
                self.validate_target()

    def confirm(self, _: object = None) -> None:
        with self.lock:
            if not self.active or self.executing or self.confirm_button.disabled:
                return
            try:
                self.invalidate()
                target = self.geometry.resolve(self.session, [s.value for s in self.xy], self.revision)
                self.session.confirm()
                destination = self.session.root / f"target_{self.revision:04d}"
                destination.mkdir()
                write_json(destination / "placement_target.json", target)
                write_json(
                    self.session.root / "target_state.json",
                    dict(
                        target_revision=self.revision,
                        status="preparing",
                        target=str(destination / "placement_target.json"),
                        preview=None,
                    ),
                )
                self.confirm_button.disabled = True
                self.edit_button.disabled = False
                revision = self.revision
                simulation = self.app.scene.sim
                self.app.transition("PLACEMENT_PREPARING", "Source inference / grasp feasibility / place endpoint IK")
                self.refresh()
            except (OSError, ValueError) as error:
                self.notice.content = str(error)
                return

        def canceled() -> bool:
            return not self.active or revision != self.revision or self.processes.cancel.is_set()

        def prepare() -> None:
            try:
                with self.worker_lock:
                    if canceled():
                        return
                    if self.engine is None:
                        self.engine = PlacementPreview(
                            self.app.builder.config,
                            self.app.args.selector,
                            self.attempt,
                            self.source_state,
                            self.processes,
                        )
                    result = self.engine.prepare(target, destination, canceled)
                    if canceled():
                        return
                    planner_args = (
                        {"sim": simulation}
                        if self.engine.config.get("simulation", {}).get("placement_planner") == "rrt_connect"
                        else {}
                    )
                    plan = (
                        prepare_execution(self.engine, result, destination, canceled, **planner_args)
                        if result and self.execution_enabled
                        else None
                    )
                with self.lock:
                    if canceled():
                        return
                    self.endpoint = result
                    write_json(
                        self.session.root / "target_state.json",
                        dict(
                            target_revision=revision,
                            status="selected" if result else "no_feasible_grasp",
                            target=str(destination / "placement_target.json"),
                            preview=str(destination / "placement_endpoint_preview.json") if result else None,
                            simulation_reference=str(destination / "simulation_plan.json") if plan else None,
                        ),
                    )
                    if result is None:
                        report = read_json(destination / "placement_selection_report.json")
                        self.notice.content = (
                            f"No feasible grasp-to-place endpoint. Counts: {report['counts']}. "
                            f"Rejections: {report.get('rejection_counts', {})}. "
                            f"Edit orientation/XY and confirm again. Diagnostics: {destination}"
                        )
                        self.app.transition("PLACEMENT_NO_FEASIBLE_GRASP", self.notice.content)
                    else:
                        self.cloud.points = self.session.points
                        self.target_cloud.points = transform_points(self.session.points, target)
                        self.target_cloud.visible = True
                        self.endpoint_choice.disabled = False
                        review_label = plan.get("label", EXECUTION_LABEL) if plan is not None else LABEL
                        self.notice.content = review_label
                        ik = result["placement"]["ik"]
                        residuals = (
                            f"{ik['position_m']:.6f} m / {ik['orientation_rad']:.6f} rad"
                            if "position_m" in ik
                            else "accepted"
                        )
                        self.app.summary.content = (
                            f"**{result['arm']} / {result['mode']} / {result['candidate_id']}** · "
                            f"{target['candidate_id']}\n\n"
                            f"Target XYZ: {np.round(target['target_centroid_world_m'], 4).tolist()} m · "
                            f"effort {result['effort']:.6f} rad²\n\n"
                            f"Grasp feasible · place IK accepted · residuals: {residuals}\n\n{review_label}"
                        )
                        if plan is not None:
                            self.app.plan = plan
                            self.app.plan_root = destination
                            self.app.phase.options = plan["execution_phases"]
                            self.app.phase.value = plan["execution_phases"][0]
                            self.app.progress.value = 0
                            self.app.summary.content += f"\n\nPreview preparation: {plan['preparation_s']:.3f} s."
                            self.app.transition("PREVIEW_READY", review_label)
                        else:
                            self.app.transition("PLACEMENT_PREVIEW_READY", LABEL)
            except Exception as error:
                write_json(
                    destination / "failure.json",
                    dict(error=f"{type(error).__name__}: {error}", canceled=canceled(), target_revision=revision),
                )
                with self.lock:
                    if not canceled():
                        write_json(
                            self.session.root / "target_state.json",
                            dict(
                                target_revision=revision,
                                status="error",
                                preview=None,
                                target=str(destination / "placement_target.json"),
                            ),
                        )
                        self.notice.content = f"{type(error).__name__}: {error}. Edit target or Reset Scene."
                        self.app.transition("PLACEMENT_ERROR", self.notice.content)

        with self.lock:
            if canceled():
                return
            self.app.worker = threading.Thread(target=prepare, daemon=True)
            self.app.worker.start()

    def preview_q(self) -> object:
        """Return the accepted solution exactly; rendering uses separate static state."""
        return None if self.endpoint is None else self.endpoint[self.endpoint_choice.value.lower() + "_q56"]

    def update_preview(self, q: np.ndarray, progress: float) -> None:
        """Update immutable cloud transforms only when the reviewed scrub location changes."""
        key = (self.revision, progress)
        if key != self.preview_key:
            stage, _ = preview_stage(self.app.plan, progress)
            self.cloud.points = preview_points(self.app.plan, self.engine.robot, self.session.points, q, stage["phase"])
            self.preview_key = key

    def begin_execution(self) -> None:
        """Freeze target controls for the physics owner; Reset is the only new-scene action."""
        if self.endpoint is None or self.app.plan["target_revision"] != self.revision:
            raise ValueError("Reviewed target revision changed")
        self.executing = True
        for control in (self.choice, self.run_button, self.confirm_button, self.edit_button, *self.xy):
            control.disabled = True
        self.marker.visible = False
        self.cloud.visible = False

    def remove(self) -> None:
        """Remove scene handles; app owns folder removal and subprocess cancellation."""
        self.active = False
        write_json(
            self.session.root / "target_state.json",
            dict(target_revision=self.revision, status="canceled", preview=None),
        )
        self.marker.remove()
        self.target_cloud.remove()
