"""One actual MuJoCo owner for reviewed single/paired linear simulation schedules."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import mujoco
import numpy as np

from ada_grasp_ctrl.experiments.simulated_object_cloud import body_pose
from ada_grasp_ctrl.experiments.dual_arm_grasp_episode import live_contacts
from ada_grasp_ctrl.grasp_selection.multi_arm import forbidden_contacts
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.robot_environment.kinematics import residual
from ada_grasp_ctrl.ros_execution.settings import DEFAULTS
from ada_grasp_ctrl.ros_execution.motion import arrival_reached
from .simulation_plan import reference, verify_plan
from .fingertip_forces import FingertipForceSource, unavailable


class SimulationExecutor:
    """Step position actuators with a simulation clock; never assign object or robot poses."""

    def __init__(
        self, scene: object, plan: dict, config: dict, root: Path, lock: threading.RLock | None = None
    ) -> None:
        verify_plan(plan)
        self.scene, self.sim, self.plan, self.root = scene, scene.sim, deepcopy(plan), root
        self.placement = plan.get("workflow") == "grasp_to_place"
        self.lock = lock or threading.RLock()
        self.cancel = threading.Event()
        # Older reviewed plans retain their recorded arrival-gated semantics.
        self.stage_completion = plan.get("stage_completion", "arrival")
        self.settings = {
            **DEFAULTS,
            "hand_tolerance": config["hand_tolerance"],
            "hand_arrival_required": config["hand_arrival_required"],
        }
        self.dt = float(self.sim.model.opt.timestep)
        self.reference_hz = float(plan.get("reference_hz", 100))
        self.reference_interval_s = 1 / self.reference_hz
        self.substeps = round(self.reference_interval_s / self.dt)
        if self.substeps < 1 or not np.isclose(self.substeps * self.dt, self.reference_interval_s, atol=1e-12, rtol=0):
            raise ValueError("Physics timestep must divide the reviewed reference interval")
        snapshot_path = next(
            (Path(p) for p in plan["references"] if Path(p).name == "settled_state.npz"), root / "settled_state.npz"
        )
        if snapshot_path.exists():
            with np.load(snapshot_path, allow_pickle=False) as snapshot:
                for field in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied"):
                    if field not in snapshot and field.endswith("applied"):
                        continue
                    if not np.array_equal(getattr(self.sim.data, field), snapshot[field]):
                        raise ValueError(f"Settled {field} changed after review")
        self.initial_targets = self.sim.targets
        self.warning_start = self.sim.data.warning.number.copy()
        self.settled_pose = body_pose(self.sim)
        self.pre_lift_pose = None
        self.result = dict(
            schema_version=1,
            status="ready",
            gravity_environment=deepcopy(scene.metadata.get("gravity_environment", {})),
            stage_completion=self.stage_completion,
            pose_adjustment=deepcopy(plan.get("pose_adjustment", {})),
            pipeline_completed=False,
            scheduled_hold_success=None,
            post_hold_retained=None,
            post_hold_drop_latched=False,
            post_hold_retention_failure_latched=False,
            phase_events=[],
            timestep_s=self.dt,
            reference_hz=self.reference_hz,
            contact_profile=deepcopy(scene.metadata.get("contact_profile", {})),
            substeps=self.substeps,
            success_criterion=dict(
                minimum_lift_m=0.03,
                maximum_object_speed_m_s=0.02,
                contact_policy="Every active side must contact the object at every hold sample",
                baseline="Actual object mesh origin immediately before lift; world Z",
            ),
            selected_source_mode=plan["source_mode"],
            execution_mode=plan["execution_mode"],
            active_arms=plan["active_arms"],
            candidate_id=plan["candidate_id"],
            effort=plan["effort"],
        )
        self.latest = {}
        self.force_source = None
        self.force_snapshot = None
        self.force_error = ""
        try:
            self.force_source = FingertipForceSource(self.sim.model, self.sim.data)
        except Exception as error:
            self.force_error = f"Contact forces: {type(error).__name__}: {str(error)[:180]}"
        if self.placement:
            self.result.pop("success_criterion")
            self.result["hold_evaluation"] = "not_evaluated_no_hold_stage"
        self.hold_min, self.hold_speed, self.hold_contact = float("inf"), 0.0, True
        self.peak_lift = 0.0
        self.ever_retained = False

    def tick(self, stage: dict, elapsed: float, stream: object, *, post_hold: bool = False) -> bool:
        """Set every active target before the common step and record actual physical state."""
        q = reference(self.plan, stage, elapsed)
        phase = stage["phase"]
        with self.lock:
            for side in self.plan["active_arms"]:
                self.sim.set_target(q[side_slice(side)], side=side)
            for side in set(("left", "right")) - set(self.plan["active_arms"]):
                block = side_slice(side)
                if not np.array_equal(self.sim.targets[block], self.initial_targets[block]):
                    raise ValueError("Passive command changed")
            for _ in range(self.substeps):
                self.scene.apply_forces()
                self.sim.step()
                if np.any(self.sim.data.warning.number != self.warning_start):
                    raise FloatingPointError("MuJoCo numerical warning")
            mujoco.mj_forward(self.sim.model, self.sim.data)
            actual = self.sim.configuration()
            overshoot = max(
                0, np.max(self.scene.robot.limits[:, 0] - actual), np.max(actual - self.scene.robot.limits[:, 1])
            )
            if overshoot > 0.01:
                raise ValueError(f"live_joint_limit_overshoot: {overshoot}")
            rows = live_contacts(self.sim)
            contact_phase = "pregrasp" if phase == "above_object" else phase
            if self.placement and phase in (
                "reorient",
                "transport",
                "place",
                "settle",
                "release_1",
                "release_2",
                "retreat_lift",
                "retreat",
            ):
                contact_phase = "squeeze"
            if forbidden_contacts(rows, self.plan["active_arms"], phase=contact_phase):
                raise ValueError("live_collision")
            errors = {
                side: (actual[side_slice(side)] - stage["arms"][side]["positions_rad"]).tolist()
                for side in self.plan["active_arms"]
            }
            arrived = all(arrival_reached(e, self.settings) for e in errors.values())
            pose = body_pose(self.sim)
            baseline = self.pre_lift_pose if self.pre_lift_pose is not None else self.settled_pose
            lift = float(pose[2, 3] - baseline[2, 3])
            dof = int(self.sim.model.joint("object_free").dofadr[0])
            speed = float(np.linalg.norm(self.sim.data.qvel[dof : dof + 3]))
            contact = all(
                any(c["category"] == "robot_object" and side in c["sides"] for c in rows)
                for side in self.plan["active_arms"]
            )
            retained = None if self.placement else lift >= 0.03 and speed <= 0.02 and contact
            if phase == "hold" and not post_hold:
                self.hold_min = min(self.hold_min, lift)
                self.hold_speed = max(self.hold_speed, speed)
                self.hold_contact &= contact
            self.peak_lift = max(self.peak_lift, lift)
            if post_hold and not self.placement:
                self.result["post_hold_retained"] = retained
                self.result["post_hold_retention_failure_latched"] |= not retained
                self.result["post_hold_drop_latched"] |= self.ever_retained and (lift < 0.03 or not contact)
            if phase == "hold":
                self.ever_retained |= retained
            wrists = {}
            for side in self.plan["active_arms"]:
                body = self.sim.data.body(side + "_hand_flange")
                matrix = np.eye(4)
                matrix[:3, :3], matrix[:3, 3] = body.xmat.reshape(3, 3), body.xpos
                wrists[side] = matrix
            relative_actual = relative_expected = relative_error = None
            if len(wrists) == 2:
                relative_actual = np.linalg.inv(wrists["left"]) @ wrists["right"]
                relative_expected = np.linalg.inv(self.scene.robot.fk(q, "left")) @ self.scene.robot.fk(q, "right")
                relative_error = residual(relative_actual, relative_expected)
            self.latest = dict(
                time_s=float(self.sim.data.time),
                phase=phase,
                post_hold=post_hold,
                lift_height_m=lift,
                slip_from_peak_m=max(0.0, self.peak_lift - lift),
                object_speed_m_s=speed,
                contact_retained=contact,
                retained=retained,
                object_pose=pose.tolist(),
                arrival_reached=arrived,
                joint_errors=errors,
                q56=actual.tolist(),
                reference_q56=q.tolist(),
                target_q56=self.sim.targets.tolist(),
                contacts=rows,
                relative_wrist=(np.linalg.inv(wrists["left"]) @ wrists["right"]).tolist() if len(wrists) == 2 else None,
                relative_wrist_reference=relative_expected.tolist() if relative_expected is not None else None,
                relative_wrist_error=relative_error,
                numerical_warning_counts=self.sim.data.warning.number.tolist(),
            )
            # Display-only sampling shares this exact measured state and clock.
            # Keep failures outside safety checks and force payloads outside logs.
            try:
                self.force_snapshot = (
                    self.force_source.sample()
                    if self.force_source is not None
                    else unavailable(float(self.sim.data.time), self.force_error)
                )
            except Exception as error:
                self.force_snapshot = unavailable(
                    float(self.sim.data.time), f"Contact forces: {type(error).__name__}: {str(error)[:180]}"
                )
            stream.write(json.dumps(self.latest, allow_nan=False) + "\n")
            return arrived

    def run(self, *, realtime: bool = True, continue_hold: bool = True) -> dict:
        """Execute the reviewed prefix, then retain final targets until explicitly canceled."""
        verify_plan(self.plan)
        if not np.allclose(self.sim.configuration(), self.plan["initial_q56"], atol=1e-10, rtol=0):
            raise ValueError("Settled robot changed after review")
        wall, start = time.monotonic(), float(self.sim.data.time)
        self.result["status"] = "executing"

        def pace() -> None:
            if realtime:
                self.cancel.wait(max(0, wall + float(self.sim.data.time) - start - time.monotonic()))

        try:
            with (self.root / "physics.jsonl").open("w", buffering=1) as stream:
                for stage in self.plan["schedule"]["stages"][: len(self.plan["execution_phases"])]:
                    if self.cancel.is_set():
                        break
                    if stage["phase"] == "lift":
                        self.pre_lift_pose = body_pose(self.sim)
                    began = float(self.sim.data.time)
                    settled = None
                    # The existing nominal settle interval becomes a fixed endpoint dwell.
                    timed_duration = stage["settled_time"] - stage["stage_start_time"]
                    event = dict(
                        phase=stage["phase"],
                        start_s=began,
                        reviewed_duration_s=stage["duration_s"],
                        scheduled_total_s=timed_duration,
                        completion_policy=self.stage_completion,
                    )
                    self.result["phase_events"].append(event)
                    while not self.cancel.is_set():
                        elapsed = float(self.sim.data.time) - began
                        arrived = self.tick(stage, elapsed, stream)
                        elapsed = float(self.sim.data.time) - began
                        if self.stage_completion == "timed":
                            if elapsed >= timed_duration - 1e-9:
                                break
                        elif elapsed >= stage["duration_s"]:
                            settled = (elapsed if settled is None else settled) if arrived else None
                            if settled is not None and elapsed - settled >= self.settings["settle_time"] - 1e-9:
                                break
                            if elapsed > stage["duration_s"] + self.settings["settle_timeout"]:
                                raise ValueError("stage_arrival_timeout")
                        pace()
                    event.update(
                        end_s=float(self.sim.data.time),
                        joint_errors=self.latest.get("joint_errors"),
                        arrival_reached=self.latest.get("arrival_reached"),
                    )
                if not self.cancel.is_set():
                    if self.placement:
                        self.result.update(status="completed", pipeline_completed=True)
                    else:
                        success = self.hold_min >= 0.03 and self.hold_speed <= 0.02 and self.hold_contact
                        self.result.update(
                            status="holding",
                            pipeline_completed=True,
                            scheduled_hold_success=bool(success),
                            minimum_retained_lift_m=self.hold_min,
                            maximum_hold_speed_m_s=self.hold_speed,
                            retained_contact_every_hold_sample=bool(self.hold_contact),
                        )
                    write_json(self.root / "execution_report.json", self.result)
                    count = 0
                    while continue_hold and not self.cancel.is_set():
                        self.tick(stage, stage["duration_s"], stream, post_hold=not self.placement)
                        count += 1
                        if count % 100 == 0:
                            write_json(self.root / "execution_report.json", self.result)
                        pace()
                if self.cancel.is_set():
                    self.result["status"] = "canceled"
        except FloatingPointError as error:
            self.result.update(status="sim_numerical_failure", error=str(error))
        except (RuntimeError, ValueError) as error:
            self.result.update(status="motion_failed", error=str(error))
        finally:
            with self.lock:
                self.force_snapshot = None
            self.result.update(
                final_object_pose=body_pose(self.sim).tolist(),
                last_observation=self.latest,
                pre_lift_pose=self.pre_lift_pose.tolist() if self.pre_lift_pose is not None else None,
            )
            write_json(self.root / "execution_report.json", self.result)
        return self.result
