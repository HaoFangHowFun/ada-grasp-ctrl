"""Opt-in, just-in-time placement planning within the existing simulation owner."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time
from typing import Callable, Generator, Iterator

import numpy as np

from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json, file_hash
from ada_grasp_ctrl.motion_planning.rrt_connect import PlanResult, plan as search
from ada_grasp_ctrl.motion_planning.placement_collision import (
    PlanningScene,
    EdgePolicy,
    SUPPORT_TOLERANCE,
    NUMERICAL_EPS,
    contacts,
    separating,
)
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.robot_environment.kinematics import rigid
from ada_grasp_ctrl.ros_execution.settings import DEFAULTS
from .placement_schedule import planned_stage, parameters

LABEL = (
    "Simulation completion-first execution; collision warnings and explicit linear fallback enabled. "
    "Lift / transport / descent / retreat pending actual-state planning."
)
COMPLETION_POLICY = "complete_with_warnings"
LIFT_CLEARANCE_M = 0.015
TRANSPORT_CLEARANCE_M = 0.005
DEFAULTS_PLANNER = dict(extension=0.20, resolution=0.02, max_iterations=5000, timeout_s=3.0, shortcut_s=0.05)


def settings(config: dict) -> dict:
    """Resolve planner-only settings without altering existing physics or policies."""
    overrides = config.get("simulation", {}).get("rrt_connect", {})
    if set(overrides) - {*DEFAULTS_PLANNER, "seed"}:
        raise ValueError("Unknown simulation.rrt_connect setting")
    result = {**DEFAULTS_PLANNER, "seed": config.get("seed", 4), **overrides}
    for key in ("extension", "resolution", "timeout_s"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"Invalid rrt_connect.{key}")
    if (
        type(result["max_iterations"]) is not int
        or result["max_iterations"] < 1
        or not np.isfinite(result["shortcut_s"])
        or result["shortcut_s"] < 0
        or (result["seed"] is not None and (type(result["seed"]) is not int or result["seed"] < 0))
    ):
        raise ValueError("Invalid RRT iteration/shortcut/seed settings")
    return result


def arm_path(
    scene: PlanningScene,
    goal: np.ndarray,
    options: dict,
    canceled: Callable,
    began: float,
    *,
    direct_only: bool = False,
    allow_fallback: bool = False,
) -> tuple[list, dict]:
    """Plan only six joints and include preceding snapshot/IK time in the deadline."""
    if allow_fallback:
        scene.robot.configuration(scene.fixed)
        scene.robot.configuration(goal)
    deadline = began + options["timeout_s"]
    policy = EdgePolicy(
        scene,
        scene.fixed[scene.arm],
        goal[scene.arm],
        resolution=options["resolution"],
        connector_radius=options["extension"],
        deadline=deadline,
        canceled=canceled,
    )
    before = scene.queries
    if direct_only:
        # The real-table exception is only for the final local descent connector;
        # a blocked connector must not launch a new unconstrained RRT detour.
        result = PlanResult("invalid_descent", method="direct_descent", edge_queries=1)
        try:
            if policy(scene.fixed[scene.arm], goal[scene.arm]):
                result.status = "success"
                result.path = [scene.fixed[scene.arm].copy(), goal[scene.arm].copy()]
                result.original_waypoints = 2
        except TimeoutError:
            result.status = "timeout"
        except InterruptedError:
            result.status = "canceled"
    else:
        result = search(
            scene.fixed[scene.arm],
            goal[scene.arm],
            scene.robot.limits[scene.arm],
            policy,
            **{k: v for k, v in options.items() if k != "resolution"},
            deadline=deadline,
            canceled=canceled,
        )
    record = asdict(result)
    record.pop("path")
    record.update(
        wall_s=time.monotonic() - began,
        collision_queries=scene.queries - before,
        simplified_waypoints=len(result.path),
        failure_reason=None if result.status == "success" else result.status,
        last_collision=policy.last_failure,
    )
    path = [scene.full(q) for q in result.path]
    if allow_fallback and result.status not in ("success", "canceled"):
        if canceled():
            raise InterruptedError("Linear fallback canceled")
        start, target = scene.robot.configuration(scene.fixed), scene.robot.configuration(goal)
        mask = np.ones(56, dtype=bool)
        mask[scene.arm] = False
        if not np.array_equal(start[mask], target[mask]):
            raise ValueError("Linear fallback changed fingers/passive joints")
        path = [start, target]
        # Keep the failed search result; fallback is executable, not collision-certified.
        record.update(fallback_applied=True, execution_method="linear_fallback", simplified_waypoints=2)
    return path, record


def transition(
    scene: PlanningScene,
    start: np.ndarray,
    goal: np.ndarray,
    options: dict,
    canceled: Callable,
    *,
    warnings: list | None = None,
) -> set[tuple[int, int]]:
    """Validate a retained hand/arm transition against the current static object."""
    if warnings is not None:
        scene.robot.configuration(start)
        scene.robot.configuration(goal)
    policy = EdgePolicy(
        scene,
        start,
        goal,
        full=True,
        resolution=min(options["resolution"], DEFAULTS["hand_step"]),
        deadline=time.monotonic() + options["timeout_s"],
        canceled=canceled,
    )
    try:
        valid = policy(start, goal)
    except TimeoutError:
        if warnings is None:
            raise
        valid = False
        policy.last_failure = "transition validation timeout"
    if not valid:
        if warnings is None:
            raise ValueError(f"{scene.mode}: retained transition collision/separation failure: {policy.last_failure}")
        warnings.append(dict(phase=scene.mode, reason=policy.last_failure))
    return policy.hand_pairs


def prepare_execution(engine: object, endpoint: dict, destination: Path, canceled: Callable, sim: object) -> dict:
    """Prepare the selected approach only; future physics-dependent paths stay pending."""
    from .simulation_plan import verify_plan

    began = time.monotonic()
    chosen = read_json(destination / "placement_selection_report.json")["selected"]
    side = endpoint["arm"]
    if chosen["candidate_id"] != endpoint["candidate_id"] or chosen["required_arms"] != [side]:
        raise ValueError("Placement winner identity changed")
    initial = engine.robot.configuration(sim.configuration())
    if not np.array_equal(initial, engine.state["full_positions_rad"]):
        raise ValueError("Settled initial configuration changed")
    options = settings(engine.config)
    scene = PlanningScene(sim, side, engine.static)
    source_pose = scene.body_pose(sim.data, scene.object_body)
    scene.configure(initial, "approach", source_pose)
    endpoints = {
        phase: engine.robot.configuration(chosen["stages"][phase]["q56"]) for phase in ("pregrasp", "grasp", "squeeze")
    }
    passive = side_slice("right" if side == "left" else "left")
    if any(not np.array_equal(q[passive], initial[passive]) for q in endpoints.values()):
        raise ValueError("Selected endpoint changed passive arm")
    approach_goal = initial.copy()
    approach_goal[scene.arm] = endpoints["pregrasp"][scene.arm]
    path, report = arm_path(scene, approach_goal, options, canceled, time.monotonic(), allow_fallback=True)
    report["segment"] = "approach"
    write_json(destination / "path_checks.json", report)
    if report["status"] != "success" and not report.get("fallback_applied"):
        raise ValueError(f"approach: {report['status']}; selected candidate retained")
    stages = [planned_stage(side, "pregrasp", path, 0.0, dwell=DEFAULTS["settle_time"])]
    previous = approach_goal
    validation_warnings = []
    for phase, target in (
        ("pregrasp_fingers", endpoints["pregrasp"]),
        ("grasp", endpoints["grasp"]),
        ("squeeze", endpoints["squeeze"]),
    ):
        scene.configure(previous, "pregrasp" if phase == "pregrasp_fingers" else phase, source_pose)
        transition(scene, previous, target, options, canceled, warnings=validation_warnings)
        stages.append(
            planned_stage(side, phase, [previous, target], stages[-1]["settled_time"], dwell=DEFAULTS["settle_time"])
        )
        previous = target
    references = deepcopy(endpoint["references"])
    for p in (
        destination / "placement_endpoint_preview.json",
        destination / "placement_selection_report.json",
        engine.root / "scene_config.yaml",
        engine.robot.urdf_path,
        engine.robot.directory / "manifest.json",
        Path(engine.config["table_plane"]),
    ):
        references[str(p)] = file_hash(p)
    result = dict(
        schema_version=1,
        kind="ada_simulation_reference",
        environment="simulation",
        workflow="grasp_to_place",
        placement_planner="rrt_connect",
        execution_policy=COMPLETION_POLICY,
        preparation_warnings=validation_warnings,
        planner_settings=options,
        stage_completion="timed",
        reference_hz=engine.config.get("simulation", {}).get("reference_hz", 25),
        schedule=dict(
            schema_version=1,
            kind="ada_rrt_placement_prefix",
            active_arms=[side],
            stages=stages,
            settle_time_s=DEFAULTS["settle_time"],
            end_time=stages[-1]["settled_time"],
        ),
        execution_phases=[s["phase"] for s in stages],
        pending_segments=["lift", "carry", "place", "retreat"],
        initial_q56=initial.tolist(),
        full_joint_names=list(engine.robot.joint_names),
        references=references,
        nominal_endpoints={p: q.tolist() for p, q in endpoints.items()},
        accepted_place_q56=endpoint["place_q56"],
        source_object_pose=source_pose.tolist(),
        target_object_pose=(rigid(endpoint["target"]["Delta_T"]) @ source_pose).tolist(),
        candidate_id=chosen["candidate_id"],
        active_arms=[side],
        effort=chosen["effort"],
        source_mode=endpoint["mode"],
        execution_mode=chosen["candidate"].get("execution_mode", side + "_full"),
        pose_adjustment=endpoint.get("simulation_pose_adjustment"),
        target=deepcopy(endpoint["target"]),
        target_revision=endpoint["target_revision"],
        target_sha256=endpoint["target_sha256"],
        source_sha256=endpoint["source_sha256"],
        placement_parameters=parameters(engine.config.get("simulation", {}).get("placement", {})),
        planning_reports=[report],
        label=LABEL,
        preparation_s=time.monotonic() - began,
    )
    verify_plan(result)
    if canceled():
        raise RuntimeError("Placement preparation canceled")
    write_json(destination / "simulation_plan.json", result)
    return result


class PlacementPlanner:
    """Generate stages inside SimulationExecutor.run; never own a thread or step physics."""

    def __init__(self, executor: object, static: object) -> None:
        self.executor, self.plan = executor, executor.plan
        self.side = self.plan["active_arms"][0]
        self.scene = PlanningScene(executor.sim, self.side, static)
        self.options = self.plan["planner_settings"]
        self.arm, self.block = self.scene.arm, side_slice(self.side)
        self.reports = deepcopy(self.plan["planning_reports"])
        self.accepted = []
        self.cursor = 0.0
        self.stage = None
        self.monitor = None
        self.pending = None
        self.warnings = {}
        self.last_attachment = None
        self.last_pairs = set()
        self.grasp_evidence = dict(
            outcome="pending",
            drop_confirmed=False,
            cleanup_required=False,
            contact_absence_threshold_s=0.5,
            attachment_translation_threshold_m=0.02,
            maximum_contact_absence_s=0.0,
        )
        self.contact_absent_since = None

    @property
    def completion_first(self) -> bool:
        """Older saved plans keep their original strict execution policy."""
        return getattr(self, "plan", {}).get("execution_policy") == COMPLETION_POLICY

    def warn(self, code: str, detail: object) -> None:
        """Aggregate recurring observations without creating an unbounded per-tick log."""
        phase = self.pending["segment"] if self.pending else self.stage["phase"] if self.stage else "preparation"
        key = (phase, code)
        now = float(self.executor.sim.data.time)
        if key not in self.warnings:
            self.warnings[key] = dict(phase=phase, code=code, first_time_s=now, count=0)
        self.warnings[key].update(last_time_s=now, detail=detail, count=self.warnings[key]["count"] + 1)
        self.executor.result["execution_warnings"] = list(self.warnings.values())

    def validate_transition(self, start: np.ndarray, goal: np.ndarray) -> set:
        """Keep collision diagnostics without vetoing valid simulation references."""
        warnings = [] if self.completion_first else None
        pairs = transition(self.scene, start, goal, self.options, self.executor.cancel.is_set, warnings=warnings)
        for warning in warnings or ():
            self.warn("transition_validation", warning)
        return pairs

    def cleanup_requested(self) -> bool:
        """A confirmed missing/dropped grasp leads to opening and retreat, not fake placement."""
        return self.completion_first and self.grasp_evidence["cleanup_required"]

    def snapshot(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Pause force display and copy actual state without holding the UI lock in search."""
        self.pending = dict(segment=mode, began=time.monotonic(), queries=self.scene.queries)
        with self.executor.lock:
            self.executor.result["status"] = "planning"
            self.executor.result["planning_segment"] = mode
            self.executor.force_snapshot = None
            self.scene.snapshot()
            q = self.scene.robot.configuration(self.executor.sim.configuration())
            pose = self.scene.body_pose(self.executor.sim.data, self.scene.object_body)
        return q, pose

    def failure(self, error: Exception) -> None:
        """Retain diagnostics even when IK, snapshots or retained transitions fail."""
        if self.pending is not None:
            self.reports.append(
                dict(
                    segment=self.pending["segment"],
                    method="actual_state_validation",
                    status=(
                        "timeout"
                        if isinstance(error, TimeoutError)
                        else "canceled"
                        if self.executor.cancel.is_set()
                        else "failed"
                    ),
                    wall_s=time.monotonic() - self.pending["began"],
                    collision_queries=self.scene.queries - self.pending["queries"],
                    original_waypoints=0,
                    simplified_waypoints=0,
                    failure_reason=str(error),
                )
            )
            self.pending = None
        self.save()

    def save(self) -> None:
        """Persist dynamically accepted paths with the immutable reviewed target identity."""
        report = dict(
            target_revision=self.plan["target_revision"],
            target_sha256=self.plan["target_sha256"],
            candidate_id=self.plan["candidate_id"],
            execution_policy=self.plan.get("execution_policy", "strict"),
            warnings=list(self.warnings.values()),
            grasp_evidence=self.grasp_evidence.copy(),
            release_hand_object_policy="legacy_squeeze",
            object_table_policy=dict(
                physical_warning_penetration_m=SUPPORT_TOLERANCE,
                nominal_lift_and_preplace_clearance_m=LIFT_CLEARANCE_M,
                transport_planning_min_clearance_m=TRANSPORT_CLEARANCE_M,
                measured_lift_clearance_warning_m=TRANSPORT_CLEARANCE_M,
                descent="direct_connector_with_explicit_fallback"
                if self.completion_first
                else "checked_direct_connector",
            ),
            reports=self.reports,
            stages=self.accepted,
            preview_preparation_s=self.plan["preparation_s"],
            total_planning_s=self.plan["preparation_s"]
            + sum(r["wall_s"] for r in self.reports[len(self.plan["planning_reports"]) :]),
        )
        self.executor.result["motion_planning"] = {k: v for k, v in report.items() if k != "stages"}
        if self.completion_first:
            self.executor.result["grasp_evidence"] = self.grasp_evidence.copy()
        write_json(self.executor.root / "dynamic_paths.json", report)

    def accept(self, phase: str, path: list, *, fixed_duration: float | None = None, dwell: float = 0.0) -> dict:
        """Build a stage with the existing interpolator and initialize live contact latches."""
        if self.executor.cancel.is_set():
            raise RuntimeError("Planning canceled")
        stage = planned_stage(self.side, phase, path, self.cursor, fixed_duration=fixed_duration, dwell=dwell)
        self.cursor = stage["settled_time"]
        self.accepted.append(stage)
        self.stage = stage
        start = self.scene.observe(self.executor.sim.data)
        self.monitor = dict(hand=contacts(start.hand))
        if self.pending is not None:
            self.reports.append(
                dict(
                    segment=self.pending["segment"],
                    method="sampled_transition",
                    status="success",
                    wall_s=time.monotonic() - self.pending["began"],
                    collision_queries=self.scene.queries - self.pending["queries"],
                    original_waypoints=len(path),
                    simplified_waypoints=len(path),
                    failure_reason=None,
                )
            )
            self.pending = None
        self.save()
        with self.executor.lock:
            self.executor.result["status"] = "executing"
            self.executor.result["planning_segment"] = None
        return stage

    def planned(self, goal: np.ndarray, began: float, segment: str, **extra: object) -> list:
        """Log success/failure before authorizing any new arm path."""
        path, record = arm_path(
            self.scene,
            goal,
            self.options,
            self.executor.cancel.is_set,
            began,
            direct_only=segment == "place",
            allow_fallback=self.completion_first,
        )
        record.update(segment=segment, **extra)
        self.reports.append(record)
        if record.get("fallback_applied"):
            self.warn(
                "linear_fallback", dict(segment=segment, reason=record["status"], collision=record["last_collision"])
            )
        self.pending = None
        self.save()
        if record["status"] != "success" and not record.get("fallback_applied"):
            raise ValueError(f"{segment}: {record['status']}; no fallback")
        return path

    def held_state(
        self, q: np.ndarray, pose: np.ndarray, *, table_clearance: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, set]:
        """Validate measured-to-commanded fingers without releasing Squeeze preload."""
        self.scene.configure(q, "squeeze", pose)
        evidence = self.scene.observe(self.executor.sim.data)
        pairs = set(contacts(evidence.hand))
        if (evidence.forbidden or not pairs) and not self.completion_first:
            raise ValueError("carry: no valid measured active-hand grasp contact; attachment refused")
        if evidence.forbidden:
            self.warn("grasp_collision", evidence.forbidden)
        if not pairs and self.last_attachment is None:
            raise ValueError("carry: no measured grasp attachment available")
        held = q.copy()
        held[self.block][6:] = np.asarray(self.plan["nominal_endpoints"]["squeeze"])[self.block][6:]
        # New intended pairs must be observed on this checked finger transition;
        # arm, passive-hand and table collisions retain their original gates.
        actual_contact = bool(pairs)
        pairs.update(self.validate_transition(q, held))
        wrist = self.scene.body_pose(self.executor.sim.data, self.scene.wrist_body)
        if actual_contact:
            attachment = np.linalg.solve(wrist, pose)
            self.last_attachment, self.last_pairs = attachment.copy(), pairs.copy()
        else:
            attachment = self.last_attachment.copy()
            pairs.update(self.last_pairs)
            self.warn(
                "transient_contact_absence", "Planning retains the last measured attachment; grasp is not recaptured"
            )
        self.scene.configure(
            held,
            "carry",
            pose,
            attachment=attachment,
            allowed_pairs=pairs,
            measured_hand=q,
            table_plane=self.plan["target"]["table_plane_world"] if table_clearance else None,
            table_clearance=table_clearance,
        )
        return held, attachment, pairs

    def carry_path(
        self, q: np.ndarray, pose: np.ndarray, object_goal: np.ndarray | None, segment: str, began: float
    ) -> tuple[list, np.ndarray, set]:
        """Plan from actual arm/attachment while retaining the Squeeze command."""
        table_clearance = TRANSPORT_CLEARANCE_M if segment == "carry" else 0.0
        held, attachment, pairs = self.held_state(q, pose, table_clearance=table_clearance)
        if object_goal is None:
            object_goal = self.scene.clearance_pose(
                self.executor.sim.data, self.plan["target"]["table_plane_world"], LIFT_CLEARANCE_M
            )
        goal_pose = np.linalg.solve(attachment.T, object_goal.T).T
        seed = held.copy()
        if segment == "carry":
            seed[self.arm] = np.asarray(self.plan["accepted_place_q56"])[self.arm]
        if self.completion_first and segment == "carry":
            goal, diagnostic = self.transport_ik(held, seed, goal_pose, began)
        else:
            goal, diagnostic = self.scene.robot.solve_arm_ik(seed, self.side, goal_pose)
        if self.executor.cancel.is_set():
            raise InterruptedError(f"{segment}: IK canceled")
        if time.monotonic() >= began + self.options["timeout_s"]:
            if not self.completion_first:
                raise TimeoutError(f"{segment}: actual-grasp IK exhausted planning deadline")
            self.warn("ik_deadline", dict(segment=segment, ik=diagnostic))
        if not diagnostic["accepted"]:
            if not self.completion_first:
                raise ValueError(f"{segment}: actual-grasp IK rejected: {diagnostic}")
            # Never command the rejected solver vector. Reuse a previously accepted
            # joint endpoint, sacrificing object pose accuracy explicitly.
            goal = held.copy()
            if segment in ("carry", "place"):
                goal[self.arm] = np.asarray(self.plan["accepted_place_q56"])[self.arm]
            goal = self.scene.robot.configuration(goal)
            diagnostic = {
                **diagnostic,
                "command_source": "accepted_place_endpoint" if segment != "lift" else "hold_current",
            }
            self.warn("ik_endpoint_fallback", dict(segment=segment, ik=diagnostic))
        mask = np.ones(56, dtype=bool)
        mask[self.arm] = False
        if not np.array_equal(goal[mask], held[mask]):
            raise ValueError(f"{segment}: IK changed held fingers/passive joints")
        path = self.planned(
            goal,
            began,
            segment,
            ik=diagnostic,
            T_hand_object=attachment.tolist(),
            actual_start_q56=q.tolist(),
            held_command_q56=held.tolist(),
            max_finger_command_error_rad=float(np.max(np.abs(held[self.block][6:] - q[self.block][6:]))),
            actual_object_pose=pose.tolist(),
            goal_object_pose=object_goal.tolist(),
            planning_table_clearance_m=table_clearance,
        )
        return path, goal, pairs

    def transport_ik(
        self, held: np.ndarray, nominal: np.ndarray, pose: np.ndarray, began: float
    ) -> tuple[np.ndarray, dict]:
        """Prefer a nearby clear IK branch before allowing a collision-warning fallback."""
        from ada_grasp_ctrl.grasp_selection.multi_arm import SEEDS

        attempts, candidates = [], []
        mask = np.ones(56, dtype=bool)
        mask[self.arm] = False
        for index, arm in enumerate([held[self.arm], nominal[self.arm], *SEEDS]):
            if self.executor.cancel.is_set():
                raise InterruptedError("Transport IK canceled")
            if index and time.monotonic() >= began + self.options["timeout_s"]:
                break
            seed = held.copy()
            seed[self.arm] = arm
            goal, diagnostic = self.scene.robot.solve_arm_ik(self.scene.robot.configuration(seed), self.side, pose)
            row = dict(seed_index=index, ik=diagnostic)
            attempts.append(row)
            if not diagnostic["accepted"]:
                continue
            goal = self.scene.robot.configuration(goal)
            if not np.array_equal(goal[mask], held[mask]):
                raise ValueError("Transport IK changed held fingers/passive joints")
            collision = self.scene.query(goal).forbidden
            row["collisions"] = collision
            candidates.append((float(np.linalg.norm(goal[self.arm] - held[self.arm])), goal, diagnostic))
            if not collision:
                return goal, {**diagnostic, "seed_index": index, "attempts": attempts}
        if not candidates:
            # The caller selects the explicit saved-endpoint policy, never a rejected vector.
            return held.copy(), dict(accepted=False, attempts=attempts)
        _, goal, diagnostic = min(candidates, key=lambda item: item[0])
        self.warn("ik_goal_collision", attempts)
        return goal, {**diagnostic, "attempts": attempts, "collision_free_goal": False}

    def validate_waypoint_support(self) -> float:
        """Require a real 5 mm object gap before enabling the raised planning plane."""
        observed = self.scene.observe(self.executor.sim.data)
        distance = self.scene.clearance(self.executor.sim.data, self.plan["target"]["table_plane_world"])
        if observed.forbidden:
            if not self.completion_first:
                raise ValueError(f"lift: actual waypoint collision: {observed.forbidden}")
            self.warn("waypoint_collision", observed.forbidden)
        if distance < TRANSPORT_CLEARANCE_M - NUMERICAL_EPS:
            if not self.completion_first:
                raise ValueError(f"lift: actual object clearance {distance:.6f} m is below required 0.005 m")
            self.warn("low_lift_clearance", dict(actual_m=distance, preferred_m=TRANSPORT_CLEARANCE_M))
        return distance

    def carry_stages(self) -> Generator[dict, None, np.ndarray]:
        """Execute the clearance waypoint before planning the confirmed destination."""
        began = time.monotonic()
        q, pose = self.snapshot("lift")
        if self.completion_first:
            self.scene.configure(q, "squeeze", pose)
            if not contacts(self.scene.observe(self.executor.sim.data).hand):
                self.grasp_evidence.update(outcome="not_acquired", cleanup_required=True)
                self.warn("grasp_not_acquired", "No actual hand/object contact after Squeeze; opening and retreating")
                return q
            self.grasp_evidence["outcome"] = "carrying"
        path, _, _ = self.carry_path(q, pose, None, "lift", began)
        yield self.accept("lift", path, dwell=DEFAULTS["settle_time"])
        if self.cleanup_requested():
            return self.executor.sim.configuration()

        began = time.monotonic()
        q, pose = self.snapshot("carry")
        clearance = self.validate_waypoint_support()
        target = rigid(self.plan["target_object_pose"])
        preplace = self.scene.clearance_pose(
            self.executor.sim.data, self.plan["target"]["table_plane_world"], LIFT_CLEARANCE_M, pose=target
        )
        path, hover, _ = self.carry_path(q, pose, preplace, "carry", began)
        self.reports[-1]["actual_lift_clearance_m"] = clearance
        yield self.accept("transport", path, dwell=DEFAULTS["settle_time"])
        if self.cleanup_requested():
            return self.executor.sim.configuration()

        began = time.monotonic()
        q, pose = self.snapshot("place")
        if np.max(np.abs(q[self.arm] - hover[self.arm])) > DEFAULTS["arm_tolerance"]:
            if not self.completion_first:
                raise ValueError("place: actual arm has not reached preplace; descent refused")
            self.warn("preplace_tracking", dict(max_arm_error_rad=float(np.max(np.abs(q[self.arm] - hover[self.arm])))))
        path, goal, pairs = self.carry_path(q, pose, target, "place", began)
        yield self.accept("place", path)
        if self.cleanup_requested():
            return self.executor.sim.configuration()
        # Settle retains the exact Place reference, while live support gates remain active.
        measured, pose = self.snapshot("settle")
        self.scene.configure(goal, "settle", pose, allowed_pairs=pairs, measured_hand=measured)
        yield self.accept("settle", [goal, goal], fixed_duration=self.plan["placement_parameters"]["settle_s"])
        return goal

    def stages(self) -> Iterator[dict]:
        """Yield reviewed approach, retained grasp stages, and measured-state future stages."""
        first = self.plan["schedule"]["stages"][0]
        q, pose = self.snapshot("approach")
        self.scene.configure(q, "approach", pose)
        path = [q.copy()]
        for row in first["waypoints_rad"]:
            point = q.copy()
            point[self.block] = row[self.side]
            path.append(point)
        policy = EdgePolicy(
            self.scene,
            path[0][self.arm],
            path[-1][self.arm],
            resolution=self.options["resolution"],
            canceled=self.executor.cancel.is_set,
            deadline=time.monotonic() + self.options["timeout_s"],
        )
        try:
            valid = all(policy(a[self.arm], b[self.arm]) for a, b in zip(path, path[1:]))
        except TimeoutError:
            if not self.completion_first:
                raise
            valid = False
        if not valid:
            if not self.completion_first:
                raise ValueError("Reviewed approach no longer valid")
            self.warn("approach_validation", policy.last_failure)
        yield self.accept("pregrasp", path, dwell=DEFAULTS["settle_time"])
        for phase in ("pregrasp_fingers", "grasp", "squeeze"):
            q, pose = self.snapshot(phase)
            self.scene.configure(q, "pregrasp" if phase == "pregrasp_fingers" else phase, pose)
            goal = q.copy()
            nominal = "pregrasp" if phase == "pregrasp_fingers" else phase
            goal[self.block] = np.asarray(self.plan["nominal_endpoints"][nominal])[self.block]
            self.validate_transition(q, goal)
            yield self.accept(phase, [q, goal], dwell=DEFAULTS["settle_time"])

        goal = yield from self.carry_stages()
        if self.completion_first and not self.cleanup_requested():
            self.grasp_evidence["outcome"] = (
                "retained_to_release"
                if contacts(self.scene.observe(self.executor.sim.data).hand)
                else "contact_lost_before_release"
            )
        values = self.plan["placement_parameters"]
        for phase, source in (("release_1", "grasp"), ("release_2", "pregrasp")):
            q, pose = self.snapshot(phase)
            self.scene.configure(q, "release", pose)
            end = q.copy()
            end[self.arm] = goal[self.arm]
            end[self.block][6:] = np.asarray(self.plan["nominal_endpoints"][source])[self.block][6:]
            self.validate_transition(q, end)
            yield self.accept(phase, [q, end], fixed_duration=values[phase + "_s"])
        began = time.monotonic()
        q, pose = self.snapshot("retreat")
        self.scene.configure(q, "grasp", pose)
        pairs = set(contacts(self.scene.observe(self.executor.sim.data).hand))
        self.scene.configure(q, "retreat", pose, allowed_pairs=pairs)
        end = q.copy()
        end[self.arm] = np.asarray(self.plan["initial_q56"])[self.arm]
        path = self.planned(end, began, "retreat", actual_start_q56=q.tolist(), actual_object_pose=pose.tolist())
        yield self.accept("retreat", path)

    def check_live(self, actual: np.ndarray, reference: np.ndarray, elapsed: float) -> None:
        """Keep actual object collisions active independently of planning attachment."""
        observed = self.scene.observe(self.executor.sim.data)
        if observed.forbidden:
            if not self.completion_first:
                raise ValueError(f"live_object_collision: {observed.forbidden}")
            self.warn("live_object_collision", observed.forbidden)
        state, mode = self.monitor, self.scene.mode
        if mode == "carry":
            if not contacts(observed.hand):
                if not self.completion_first:
                    raise ValueError("live_grasp_lost: attached-object planning assumption no longer supported")
                self.warn(
                    "contact_absence", "No current active-hand contact; monitoring duration and object displacement"
                )
            if self.completion_first:
                self.monitor_grasp(observed)
        if mode == "retreat":
            current = contacts(observed.hand)
            if not separating(state["hand"], current):
                if not self.completion_first:
                    raise ValueError("live_separation: penetration increased or contact renewed")
                self.warn("retreat_contact_change", dict(pairs=[list(p) for p in current]))
            state["hand"] = current

    def monitor_grasp(self, observed: object) -> None:
        """Confirm loss only after sustained absence and displacement; never weld the object."""
        now = float(self.executor.sim.data.time)
        if contacts(observed.hand):
            self.contact_absent_since = None
            return
        if self.contact_absent_since is None:
            self.contact_absent_since = now
        absent = now - self.contact_absent_since
        self.grasp_evidence["maximum_contact_absence_s"] = max(self.grasp_evidence["maximum_contact_absence_s"], absent)
        wrist = self.scene.body_pose(self.executor.sim.data, self.scene.wrist_body)
        pose = self.scene.body_pose(self.executor.sim.data, self.scene.object_body)
        deviation = float(np.linalg.norm(pose[:3, 3] - (wrist @ self.last_attachment)[:3, 3]))
        if absent >= 0.5 and deviation >= 0.02 and not self.grasp_evidence["drop_confirmed"]:
            self.grasp_evidence.update(
                outcome="dropped",
                drop_confirmed=True,
                cleanup_required=True,
                drop_time_s=now,
                attachment_translation_error_m=deviation,
            )
            self.warn("confirmed_drop", self.grasp_evidence.copy())
