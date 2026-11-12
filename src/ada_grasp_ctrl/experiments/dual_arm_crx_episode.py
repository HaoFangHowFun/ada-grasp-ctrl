"""Historical CRX overhead/native-control protocol in the live dual-arm scene."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ada_grasp_ctrl.errors import ControlSolveEpisodeAbort
from ada_grasp_ctrl.grasp_selection.multi_arm import STAGES, forbidden_contacts
from ada_grasp_ctrl.robot_environment.kinematics import residual
from ada_grasp_ctrl.robot_environment.model import side_slice
from .dual_arm_crx_approach import prepare_overhead
from .dual_arm_grasp_episode import live_contacts
from .dual_arm_native_control import ActiveSimulation, NativeEvaluator
from .simulated_object_cloud import body_pose, positive

from .sharpa_contact_profile import diagnostic_step_interval


def execute_native(scene: object, selection: dict, config: dict, output: Path) -> dict:
    """Execute overhead approach, native ours, and continuous Cartesian lift.

    Preserve global gravity, object dynamics, passive commands, selection ranking,
    and the scene's retained-lift criterion. Save partial physics and controller
    records on failure; solver degradation takes precedence over physical success.
    """
    if not scene.physics_available:
        return dict(status="physics_unavailable", execution_ready=False)
    if config.get("phases") != [*STAGES, "lift"]:
        raise ValueError("execution.phases must be pregrasp/grasp/squeeze/lift")
    lift_height = positive(config["lift_height_m"], "lift_height_m")
    hold = positive(config["hold_s"], "hold_s")
    minimum = positive(config["success"]["minimum_lift_m"], "minimum_lift_m")
    stable_speed = positive(config["success"]["maximum_object_speed_m_s"], "maximum_object_speed_m_s")
    soft_limit = positive(config["soft_limit_tolerance_rad"], "soft_limit_tolerance_rad")
    compensate = config.get("arm_bias_compensation", False)
    if not isinstance(compensate, bool) or minimum > lift_height:
        raise ValueError("Invalid bias compensation or minimum lift setting")
    sim, robot = scene.sim, scene.robot
    sample_steps = diagnostic_step_interval(float(sim.model.opt.timestep))
    chosen = selection["selected"]
    side = chosen["selected_side"]
    block, arm = side_slice(side), side_slice(side, True)
    passive = side_slice("left" if side == "right" else "right")
    initial_commands = sim.targets.copy()
    initial_pose = body_pose(sim).copy()
    bad = forbidden_contacts(sim.check_static_contacts(sim.configuration()), side, phase="initial")
    if bad:
        return dict(status="invalid_initialization", contacts=bad, execution_ready=False)
    for contact in sim.data.contact:
        labels = {sim.body_labels[int(sim.model.geom_bodyid[g])] for g in contact.geom}
        if "object" in labels and labels & {"table", "environment"} and contact.dist < -0.002:
            return dict(status="invalid_initialization", reason="object_environment_penetration", execution_ready=False)
    plan = prepare_overhead(scene, chosen)
    if plan["status"] != "ready":
        return plan
    result = dict(
        status="motion_failed",
        execution_ready=True,
        hardware_ready=False,
        selected_side=side,
        protocol="crx_overhead_native_ours",
        phase_events=[],
        approach_diagnostics=[],
        lift_diagnostics=[],
        success_criterion=config["success"],
        arm_bias_compensation=compensate,
        active_hand_object_penetration_policy="10 mm during initialization/pregrasp only",
        physics=dict(
            gravity=sim.model.opt.gravity.tolist(),
            timestep_s=float(sim.model.opt.timestep),
            gravity_disabled=bool(sim.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)),
            object_downward_force_override=False,
        ),
        historical_differences=[
            "Live scene gravity retained; timestep recorded in physics",
            "Scene retained-lift criterion retained",
            "Passive arm remains dynamic with its original command",
        ],
        initial_object_pose=initial_pose.tolist(),
        contact_profile=scene.metadata.get("contact_profile", {"name": "source"}),
    )
    warning_start = sim.data.warning.number.copy()
    frames, contacts_log = [], []
    ticks = 0
    hold_lifts, hold_speeds, hold_contacts = [], [], []
    before_lift = None
    adapter = None
    evaluator = None

    def record() -> None:
        phase = adapter.phase if adapter is not None else "initial"
        frames.append(
            dict(
                time_s=float(sim.data.time),
                qpos=sim.data.qpos.copy(),
                qvel=sim.data.qvel.copy(),
                ctrl=sim.data.ctrl.copy(),
                qfrc_applied=sim.data.qfrc_applied.copy(),
                phase=phase,
                object_pose=body_pose(sim),
            )
        )

    def check() -> None:
        phase = adapter.phase
        rows = live_contacts(sim)
        contacts_log.append(dict(time_s=float(sim.data.time), phase=phase, contacts=rows))
        if forbidden_contacts(rows, side, phase=phase):
            raise ValueError("live_collision")
        q = sim.configuration()
        overshoot = float(max(0, np.max(robot.limits[:, 0] - q), np.max(q - robot.limits[:, 1])))
        result["maximum_soft_limit_overshoot_rad"] = max(result.get("maximum_soft_limit_overshoot_rad", 0), overshoot)
        if overshoot > soft_limit:
            raise ValueError("live_joint_limit_overshoot")
        if not np.array_equal(sim.targets[passive], initial_commands[passive]):
            raise ValueError("Passive command changed")

    def after_step() -> None:
        nonlocal ticks
        ticks += 1
        if np.any(sim.data.warning.number != warning_start):
            raise FloatingPointError("MuJoCo numerical warning")
        phase = adapter.phase
        events = result["phase_events"]
        if not events or events[-1]["phase"] != phase:
            if events:
                events[-1].update(completed=True, end_s=float(sim.data.time))
            events.append(dict(phase=phase, start_s=float(sim.data.time), completed=False))
        if ticks % sample_steps == 0:
            record()
            check()
            if phase == "hold":
                obj_dof = int(sim.model.joint("object_free").dofadr[0])
                hold_lifts.append(float(body_pose(sim)[2, 3] - before_lift[2, 3]))
                hold_speeds.append(float(np.linalg.norm(sim.data.qvel[obj_dof : obj_dof + 3])))
                hold_contacts.append(bool(adapter.get_curr_contact_info()))

    adapter = ActiveSimulation(scene, side, after_step, compensate=compensate)
    evaluator = NativeEvaluator(scene, side, adapter, output)
    names = evaluator.robot.doa_names
    per_action = evaluator.sim_step_per_action

    def motion(target: np.ndarray) -> None:
        adapter.ctrl_qpos_a_with_interp(adapter.get_qpos_a(), target[block], names, per_action // 5, 5)

    def measured_palm() -> np.ndarray:
        mujoco.mj_forward(sim.model, sim.data)
        pose = np.eye(4)
        body = sim.data.body(f"{side}_hand_C_MC")
        pose[:3, :3], pose[:3, 3] = body.xmat.reshape(3, 3), body.xpos
        return pose

    try:
        record()
        check()
        for segment in plan["segments"]:
            path = np.asarray(segment["path"])
            for target in path:
                motion(target)
                if segment["name"] != "descend_to_pregrasp" and adapter.get_curr_contact_info():
                    raise ValueError("premature_hand_object_contact")
            error = residual(measured_palm(), np.asarray(segment["target_palm"]))
            waits = 0
            while (error["position_m"] > 0.001 or error["orientation_rad"] > 0.01) and waits < 30:
                motion(path[-1])
                waits += 1
                error = residual(measured_palm(), np.asarray(segment["target_palm"]))
            result["approach_diagnostics"].append(dict(segment=segment["name"], settle_actions=waits, **error))
            if error["position_m"] > 0.001 or error["orientation_rad"] > 0.01:
                raise ValueError("overhead_arrival_failed")
        control_pose = body_pose(sim).copy()
        result.update(
            control_start_object_pose=control_pose.tolist(),
            object_drift_before_control_m=float(np.linalg.norm(control_pose[:3, 3] - initial_pose[:3, 3])),
        )
        evaluator.grasp(chosen)
        before_lift = body_pose(sim).copy()
        result["pre_lift_pose"] = before_lift.tolist()
        adapter.phase = "lift"
        # Use actual flange pose; seed bounded IK from the last valid command rather
        # than clipping soft-limit dynamics or teleporting measured finger positions.
        mujoco.mj_forward(sim.model, sim.data)
        body = sim.data.body(f"{side}_hand_flange")
        start = np.eye(4)
        start[:3, :3], start[:3, 3] = body.xmat.reshape(3, 3), body.xpos
        seed = sim.targets.copy()
        for index in range(1, 21):
            target_pose = start.copy()
            target_pose[2, 3] += lift_height * index / 20
            q, diagnostic = robot.solve_arm_ik(seed, side, target_pose)
            jump = float(np.max(np.abs(q[arm] - seed[arm])))
            result["lift_diagnostics"].append(dict(waypoint=index, joint_step_rad=jump, **diagnostic))
            if not diagnostic["accepted"] or jump > 0.35:
                raise ValueError("cartesian_lift_ik_failed")
            for intermediate in np.linspace(seed, q, 6)[1:]:
                if forbidden_contacts(sim.check_static_contacts(intermediate), side, phase="lift"):
                    raise ValueError("cartesian_lift_path_collision")
            motion(q)
            seed = q
        result["lift_completed"] = True
        adapter.phase = "hold"
        adapter.control_hand_step(max(1, int(np.ceil(hold / sim.model.opt.timestep))))
        if not hold_lifts:
            raise ValueError("Hold duration did not produce a validation sample")
        success = min(hold_lifts) >= minimum and max(hold_speeds) <= stable_speed and all(hold_contacts)
        result.update(
            status="success" if success else "failed_grasp",
            minimum_retained_lift_m=min(hold_lifts),
            maximum_hold_speed_m_s=max(hold_speeds),
            retained_contact_every_hold_sample=all(hold_contacts),
            object_lift_from_control_start_m=float(body_pose(sim)[2, 3] - control_pose[2, 3]),
        )
        result["phase_events"][-1].update(completed=True, end_s=float(sim.data.time))
    except ControlSolveEpisodeAbort as exc:
        result.update(status="solver_degraded", phase=adapter.phase, error=str(exc))
    except FloatingPointError as exc:
        result.update(status="sim_numerical_failure", phase=adapter.phase, error=str(exc))
    except (ValueError, RuntimeError) as exc:
        result.update(status="motion_failed", phase=adapter.phase, error=str(exc))
    finally:
        record()
        degraded = bool(evaluator.grasp_ctrl.solver_degraded)
        if degraded and result["status"] in ("success", "failed_grasp"):
            result["physical_status"] = result["status"]
            result["status"] = "solver_degraded"
        native_status = (
            "completed"
            if result["status"] in ("success", "failed_grasp")
            else ("solver_degraded" if degraded else "execution_error")
        )
        evaluator.grasp_ctrl.save_recorded_data(str(output / "control.npy"), episode_status=native_status)
        np.savez_compressed(
            output / "trajectory.npz",
            **{k: [f[k] for f in frames] for k in frames[0]},
            joint_names=robot.joint_names,
            robot_qpos_indices=sim.qpos,
        )
        result.update(
            trajectory="trajectory.npz",
            control_record="control.npy",
            native_initial_command_projection=adapter.initial_command_projection,
            command_roundoff_projections=adapter.command_roundoff_projections,
            contacts=contacts_log,
            solver_degraded=degraded,
            physics_steps=ticks,
            control_steps=len(evaluator.grasp_ctrl.r_data["dof"]),
            contact_control_steps=sum(bool(c) for c in evaluator.grasp_ctrl.r_data["contacts"]),
            passive_target_unchanged=bool(np.array_equal(sim.targets[passive], initial_commands[passive])),
            final_object_pose=body_pose(sim).tolist(),
        )
    return result
