"""Conservative position-controlled Phase 2 episode with sampled path and live checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.robot_environment.kinematics import residual
from ada_grasp_ctrl.grasp_selection.multi_arm import forbidden_contacts, STAGES
from .simulated_object_cloud import body_pose, positive, apply_arm_bias_compensation

from .sharpa_contact_profile import diagnostic_step_interval


def smoothstep(t: float) -> float:
    """Use the incumbent position-plan quintic interpolation, without ROS imports."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def live_contacts(sim: object) -> list[dict]:
    """Classify actual contacts without clipping soft-limit physics states for static IK APIs."""
    rows = []
    for contact in sim.data.contact:
        bodies = [int(sim.model.geom_bodyid[g]) for g in contact.geom]
        names = [sim.model.body(b).name for b in bodies]
        labels = [sim.body_labels[b] for b in bodies]
        sides = sorted(set(labels) & {"left", "right"})
        if not sides:
            continue
        category = (
            "left_right"
            if len(sides) == 2
            else "robot_object"
            if "object" in labels
            else "robot_table"
            if "table" in labels
            else "within_robot"
            if labels[0] == labels[1]
            else "robot_environment"
        )
        rows.append(
            dict(
                category=category,
                sides=sides,
                bodies=names,
                hand_bodies=[name for body, name in zip(bodies, names) if body in sim.hand_body_ids],
                neighboring_source_links=(
                    sim.robot.parent_links.get(names[0]) == names[1] or sim.robot.parent_links.get(names[1]) == names[0]
                ),
                penetration_m=max(0.0, float(-contact.dist)),
                distance_m=float(contact.dist),
            )
        )
    return rows


def execute(scene: object, selection: dict, config: dict, output: Path) -> dict:
    """Plan all four stages, validate sampled motion, then step actual dynamic physics.

    No robot/object reset or teleport occurs during execution. Every physics step is
    checked for numerical warnings; contacts and tracking are checked every 20 ms.
    Configuration defines an independent experimental lift-retention criterion.
    """
    protocol = config.get("protocol", "position")
    if protocol == "crx_overhead_native_ours":
        from .dual_arm_crx_episode import execute_native

        return execute_native(scene, selection, config, output)
    if protocol != "position":
        raise ValueError(f"Unknown execution protocol: {protocol}")
    if not scene.physics_available:
        return dict(
            status="physics_unavailable",
            execution_ready=False,
            reason="Explicit free-object mass/friction/collision assumptions are required",
        )
    if config.get("phases") != [*STAGES, "lift"]:
        raise ValueError("execution.phases must be pregrasp/grasp/squeeze/lift")
    durations = {s: positive(config["durations_s"][s], s + " duration") for s in [*STAGES, "lift"]}
    hold = positive(config["hold_s"], "hold_s")
    settle = positive(config["settle_timeout_s"], "settle_timeout_s")
    lift_height = positive(config["lift_height_m"], "lift_height_m")
    minimum = positive(config["success"]["minimum_lift_m"], "minimum_lift_m")
    stable_speed = positive(config["success"]["maximum_object_speed_m_s"], "maximum_object_speed_m_s")
    tolerance = positive(config["arm_arrival_tolerance_rad"], "arm_arrival_tolerance_rad")
    soft_limit_tolerance = positive(config["soft_limit_tolerance_rad"], "soft_limit_tolerance_rad")
    if minimum > lift_height:
        raise ValueError("minimum_lift_m must not exceed commanded lift_height_m")
    sim, robot = scene.sim, scene.robot
    sample_steps = diagnostic_step_interval(float(sim.model.opt.timestep))
    chosen = selection["selected"]
    side, arm, block = (
        chosen["selected_side"],
        side_slice(chosen["selected_side"], True),
        side_slice(chosen["selected_side"]),
    )
    passive = side_slice("right" if side == "left" else "left")
    initial_commands = sim.targets.copy()
    start = robot.configuration(sim.configuration())
    bad = forbidden_contacts(sim.check_static_contacts(start), side, phase="initial")
    if bad:
        return dict(status="invalid_initialization", contacts=bad, execution_ready=False)
    for contact in sim.data.contact:
        labels = {sim.body_labels[int(sim.model.geom_bodyid[g])] for g in contact.geom}
        if "object" in labels and labels & {"table", "environment"} and contact.dist < -0.002:
            return dict(
                status="invalid_initialization",
                reason="object_environment_penetration",
                penetration_m=float(-contact.dist),
                execution_ready=False,
            )
    plan = {s: np.array(chosen["stages"][s]["q56"]) for s in STAGES}
    lift = np.array(chosen["candidate"]["targets"][side]["squeeze"]["T_world_hand"])
    lift[2, 3] += lift_height
    plan["lift"], lift_diag = robot.solve_arm_ik(plan["squeeze"], side, lift)
    if not lift_diag["accepted"]:
        return dict(status="motion_failed", phase="lift_plan", ik=lift_diag, execution_ready=False)
    # Check the full command interpolation with the actual object and passive robot state.
    # This is sampled kinematic validation, followed by independent live dynamic checks.
    path_checks = []
    prev = start
    for phase, target in plan.items():
        count = max(2, int(np.ceil(1.875 * np.max(np.abs(target[block] - prev[block])) / 0.02)))
        for index in range(count + 1):
            q = prev + smoothstep(index / count) * (target - prev)
            bad = forbidden_contacts(sim.check_static_contacts(q), side, phase=phase)
            if bad:
                return dict(
                    status="motion_failed",
                    phase=phase,
                    reason="interpolated_path_collision",
                    sample=index,
                    contacts=bad,
                    execution_ready=False,
                )
        path_checks.append(dict(phase=phase, samples=count + 1, maximum_joint_increment_rad=0.02))
        prev = target
    result = dict(
        status="motion_failed",
        selected_side=side,
        execution_ready=True,
        hardware_ready=False,
        path_validation=path_checks,
        lift_ik=lift_diag,
        phase_events=[],
        success_criterion=config["success"],
        actuation="Phase 1 provisional position gains",
        active_hand_object_penetration_policy="10 mm during initialization/pregrasp only",
    )
    compensate = config.get("arm_bias_compensation", False)
    if not isinstance(compensate, bool):
        raise ValueError("arm_bias_compensation must be a boolean")
    result["arm_bias_compensation"] = compensate
    sim.select_side(side)
    times, qposes, qvels, controls, applied, phases, poses, contacts_log = [], [], [], [], [], [], [], []
    before_lift = None
    warning_start = sim.data.warning.number.copy()

    def record(phase: str) -> None:
        times.append(float(sim.data.time))
        qposes.append(sim.data.qpos.copy())
        qvels.append(sim.data.qvel.copy())
        controls.append(sim.data.ctrl.copy())
        applied.append(sim.data.qfrc_applied.copy())
        phases.append(phase)
        poses.append(body_pose(sim))

    def check(phase: str) -> None:
        if np.any(sim.data.warning.number != warning_start):
            raise FloatingPointError("MuJoCo numerical warning")
        q = sim.configuration()
        overshoot = float(max(0, np.max(robot.limits[:, 0] - q), np.max(q - robot.limits[:, 1])))
        result["maximum_soft_limit_overshoot_rad"] = max(result.get("maximum_soft_limit_overshoot_rad", 0), overshoot)
        if overshoot > soft_limit_tolerance:
            raise ValueError("live_joint_limit_overshoot")
        rows = live_contacts(sim)
        contacts_log.append(dict(time_s=float(sim.data.time), phase=phase, contacts=rows))
        if forbidden_contacts(rows, side, phase=phase):
            raise ValueError("live_collision")
        if not np.array_equal(sim.targets[passive], initial_commands[passive]):
            raise ValueError("Passive command changed")

    def step_target(target: np.ndarray, phase: str, index: int) -> None:
        sim.set_target(target[block])
        if compensate:
            apply_arm_bias_compensation(sim)
        sim.step()
        if np.any(sim.data.warning.number != warning_start):
            raise FloatingPointError("MuJoCo numerical warning")
        if index % sample_steps == 0:
            record(phase)
            check(phase)

    phase = "initial"
    try:
        record(phase)
        check(phase)
        for phase, target in plan.items():
            if phase == "lift":
                before_lift = body_pose(sim).copy()
            result["phase_events"].append(dict(phase=phase, start_s=float(sim.data.time)))
            prev = sim.targets.copy()
            steps = max(1, int(np.ceil(durations[phase] / sim.model.opt.timestep)))
            for index in range(1, steps + 1):
                step_target(prev + smoothstep(index / steps) * (target - prev), phase, index)
            for index in range(int(np.ceil(settle / sim.model.opt.timestep))):
                step_target(target, phase, index)
                if np.max(np.abs(sim.configuration()[arm] - target[arm])) <= tolerance:
                    break
            error = float(np.max(np.abs(sim.configuration()[arm] - target[arm])))
            actual_pose = np.eye(4)
            actual_pose[:3, :3] = sim.data.body(f"{side}_hand_flange").xmat.reshape(3, 3)
            actual_pose[:3, 3] = sim.data.body(f"{side}_hand_flange").xpos
            expected_pose = lift if phase == "lift" else chosen["candidate"]["targets"][side][phase]["T_world_hand"]
            pose_error = residual(actual_pose, np.asarray(expected_pose))
            result["phase_events"][-1].update(
                end_s=float(sim.data.time),
                arm_error_rad=error,
                pose_error=pose_error,
                finger_error_rad=float(np.max(np.abs(sim.configuration()[block][6:] - target[block][6:]))),
            )
            if error > tolerance or pose_error["position_m"] > 0.01 or pose_error["orientation_rad"] > 0.05:
                raise ValueError("stage_arrival_failed")
        phase = "hold"
        lift_samples, speeds, retained = [], [], []
        object_dof = int(sim.model.joint("object_free").dofadr[0])
        for index in range(int(np.ceil(hold / sim.model.opt.timestep))):
            step_target(plan["lift"], phase, index)
            if index % sample_steps == 0:
                lift_samples.append(float(body_pose(sim)[2, 3] - before_lift[2, 3]))
                speeds.append(float(np.linalg.norm(sim.data.qvel[object_dof : object_dof + 3])))
                retained.append(
                    any(c["category"] == "robot_object" and c["sides"] == [side] for c in contacts_log[-1]["contacts"])
                )
        success = min(lift_samples) >= minimum and max(speeds) <= stable_speed and all(retained)
        result.update(
            status="success" if success else "failed_grasp",
            lift_completed=True,
            minimum_retained_lift_m=min(lift_samples),
            maximum_hold_speed_m_s=max(speeds),
            retained_contact_every_hold_sample=all(retained),
            pre_lift_pose=before_lift.tolist(),
        )
    except FloatingPointError as exc:
        result.update(status="sim_numerical_failure", phase=phase, error=str(exc))
    except (ValueError, RuntimeError) as exc:
        numerical = not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all()
        result.update(status="sim_numerical_failure" if numerical else "motion_failed", phase=phase, error=str(exc))
    finally:
        record(phase)
        np.savez_compressed(
            output / "trajectory.npz",
            time_s=times,
            qpos=qposes,
            qvel=qvels,
            ctrl=controls,
            qfrc_applied=applied,
            phase=phases,
            object_pose=poses,
            joint_names=robot.joint_names,
            robot_qpos_indices=sim.qpos,
        )
        result.update(
            trajectory="trajectory.npz",
            contacts=contacts_log,
            passive_target_unchanged=bool(np.array_equal(sim.targets[passive], initial_commands[passive])),
            final_object_pose=body_pose(sim).tolist(),
        )
    return result
