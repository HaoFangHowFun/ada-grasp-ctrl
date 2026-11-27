"""Object-space dual IK inside the existing measured-state placement lifecycle."""

import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ada_grasp_ctrl.grasp_selection.multi_arm import SEEDS
from ada_grasp_ctrl.motion_planning.placement_collision import contacts
from ada_grasp_ctrl.robot_environment.kinematics import residual, rigid
from ada_grasp_ctrl.robot_environment.model import side_slice


def interpolate_pose(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    """Interpolate world translation and shortest-arc orientation on one clock."""
    pose = np.eye(4)
    pose[:3, 3] = (1 - progress) * start[:3, 3] + progress * goal[:3, 3]
    pose[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], goal[:3, :3]])))(progress).as_matrix()
    return pose


def contact_sides(planner: object, observed: object) -> set[str]:
    """Identify actual active-hand/object contact from the compiled geometry labels."""
    scene = planner.scene
    return {
        scene.sim.body_labels[int(scene.model.geom_bodyid[g])] for pair in contacts(observed.hand) for g in pair
    } & set(planner.sides)


def predicted_object(planner: object) -> np.ndarray:
    """Use the midpoint of both measured wrist predictions for the loss monitor."""
    poses = []
    for side in planner.sides:
        wrist = planner.scene.body_pose(planner.executor.sim.data, planner.scene.model.body(side + "_hand_flange").id)
        poses.append(wrist @ np.linalg.inv(planner.attachments[side]))
    return interpolate_pose(*poses, 0.5)


def solve_waypoint(
    robot: object, previous: np.ndarray, poses: dict, canceled: object, *, deadline: float = float("inf")
) -> tuple[np.ndarray, dict]:
    """Warm-start each arm, retry bounded seeds, and reject discontinuous branches."""
    q, diagnostics = previous.copy(), {}
    for side, pose in poses.items():
        arm = side_slice(side, True)
        mask = np.ones(56, dtype=bool)
        mask[arm] = False
        attempts = []
        for index, seed in enumerate([previous[arm], *SEEDS]):
            if canceled():
                raise InterruptedError("Paired carry IK canceled")
            if time.monotonic() >= deadline:
                raise TimeoutError("Paired carry shared planning deadline; hold current grasp")
            trial = q.copy()
            trial[arm] = seed
            try:
                candidate, diagnostic = robot.solve_arm_ik(robot.configuration(trial), side, pose)
                candidate = robot.configuration(candidate)
                if not np.array_equal(candidate[mask], q[mask]):
                    raise ValueError("Paired IK changed joints outside its arm")
                jump = float(np.max(np.abs(candidate[arm] - previous[arm])))
                attempts.append(dict(seed_index=index, ik=diagnostic, max_joint_change_rad=jump))
                # Dense 5 mm / 0.03 rad object steps must not switch IK branches.
                if diagnostic["accepted"] and jump <= 0.5:
                    q = candidate
                    break
            except (ValueError, RuntimeError) as error:
                attempts.append(dict(seed_index=index, error=str(error)))
        else:
            raise ValueError(f"{side}: paired carry IK rejected or discontinuous; hold current grasp: {attempts}")
        diagnostics[side] = attempts
    return q, diagnostics


def carry_path(
    planner: object, q: np.ndarray, pose: np.ndarray, object_goal: np.ndarray | None, segment: str, began: float
) -> tuple[list, np.ndarray, set]:
    """Build the whole synchronized path before authorizing any carry command.

    Attachments are captured together once after Squeeze. Later stages start from
    measured object state and keep those same attachments and Squeeze preload.
    Collision diagnostics are advisory; invalid IK never returns a partial path.
    """
    from .rrt_placement import LIFT_CLEARANCE_M, TRANSPORT_CLEARANCE_M

    scene, executor = planner.scene, planner.executor
    scene.configure(q, "squeeze", pose)
    observed = scene.observe(executor.sim.data)
    pairs = set(contacts(observed.hand))
    if planner.attachments is None:
        if not pairs:
            raise ValueError("Paired carry has no initial measured grasp")
        planner.attachments = {
            side: np.linalg.solve(pose, scene.body_pose(executor.sim.data, scene.model.body(side + "_hand_flange").id))
            for side in planner.sides
        }
        planner.last_attachment = np.linalg.inv(planner.attachments[planner.sides[0]])
        planner.grasp_evidence["initial_contact_sides"] = sorted(contact_sides(planner, observed))
    held = q.copy()
    held[planner.fingers] = np.asarray(planner.plan["nominal_endpoints"]["squeeze"])[planner.fingers]
    pairs.update(planner.last_pairs)
    pairs.update(planner.validate_transition(q, held))
    planner.last_pairs = pairs.copy()
    margin = TRANSPORT_CLEARANCE_M if segment == "carry" else 0.0
    scene.configure(
        held,
        "carry",
        pose,
        attachment=planner.last_attachment,
        allowed_pairs=pairs,
        measured_hand=q,
        table_plane=planner.plan["target"]["table_plane_world"] if margin else None,
        table_clearance=margin,
    )
    if object_goal is None:
        object_goal = scene.clearance_pose(
            executor.sim.data, planner.plan["target"]["table_plane_world"], LIFT_CLEARANCE_M
        )
    object_goal = rigid(object_goal)
    distance = residual(pose, object_goal)
    count = max(1, int(np.ceil(max(distance["position_m"] / 0.005, distance["orientation_rad"] / 0.03))))
    path, object_poses, ik = [held], [pose], []
    record = dict(
        segment=segment,
        method="object_interpolation_dual_ik",
        status="planning",
        collision_certified=False,
        actual_start_q56=q.tolist(),
        held_command_q56=held.tolist(),
        actual_object_pose=pose.tolist(),
        goal_object_pose=object_goal.tolist(),
        attachments={s: a.tolist() for s, a in planner.attachments.items()},
        planning_table_clearance_m=margin,
        ik=ik,
    )
    before = scene.queries
    try:
        for progress in np.linspace(0, 1, count + 1)[1:]:
            target = interpolate_pose(pose, object_goal, float(progress))
            wrists = {s: target @ planner.attachments[s] for s in planner.sides}
            solved, diagnostic = solve_waypoint(
                scene.robot, path[-1], wrists, executor.cancel.is_set, deadline=began + planner.options["timeout_s"]
            )
            path.append(solved)
            object_poses.append(target)
            ik.append(dict(progress=float(progress), arms=diagnostic))
        # Sample the same linear joint connectors used by the reference owner.
        # Keep a single explicit predicted object pose for each combined query.
        maximum = dict(position_m=0.0, orientation_rad=0.0)
        relative_max = maximum.copy()
        collisions, checked = [], 0
        expected_relative = np.linalg.solve(planner.attachments["left"], planner.attachments["right"])
        for index, (a, b) in enumerate(zip(path, path[1:])):
            samples = max(
                2, int(np.ceil(np.max(np.abs(b[planner.arm] - a[planner.arm])) / planner.options["resolution"])) + 1
            )
            for alpha in np.linspace(0, 1, samples):
                if executor.cancel.is_set():
                    raise InterruptedError("Paired carry validation canceled")
                sample = a + alpha * (b - a)
                target = interpolate_pose(object_poses[index], object_poses[index + 1], float(alpha))
                scene.object_path_pose = target
                bad = scene.query(sample).forbidden
                checked += 1
                if bad and len(collisions) < 20:
                    collisions.append(dict(edge=index, alpha=float(alpha), contacts=bad))
                wrists = {s: scene.robot.fk(sample, s) for s in planner.sides}
                for side in planner.sides:
                    errors = residual(wrists[side], target @ planner.attachments[side])
                    maximum = {k: max(maximum[k], errors[k]) for k in maximum}
                errors = residual(np.linalg.solve(wrists["left"], wrists["right"]), expected_relative)
                relative_max = {k: max(relative_max[k], errors[k]) for k in relative_max}
        record.update(
            status="success",
            original_waypoints=len(path),
            simplified_waypoints=len(path),
            object_waypoints=[p.tolist() for p in object_poses],
            sampled_configurations=checked,
            collision_warnings=collisions,
            maximum_wrist_residual=maximum,
            maximum_relative_wrist_residual=relative_max,
        )
        if collisions:
            planner.warn("paired_carry_collision", collisions)
        if maximum["position_m"] > 0.001 or maximum["orientation_rad"] > 0.01:
            planner.warn("paired_reference_drift", dict(wrist=maximum, relative=relative_max))
    except (ValueError, RuntimeError, InterruptedError, TimeoutError) as error:
        record.update(
            status="canceled" if executor.cancel.is_set() else "failed",
            failure_reason=str(error),
            execution_method="hold_current_grasp",
            partial_path_executed=False,
            failed_waypoint_index=len(path),
        )
        planner.grasp_evidence["outcome"] = "planning_failed_hold"
        raise
    finally:
        scene.object_path_pose = None
        record.update(wall_s=time.monotonic() - began, collision_queries=scene.queries - before)
        record["deadline_exceeded"] = record["wall_s"] > planner.options["timeout_s"]
        planner.reports.append(record)
        planner.pending = None
        planner.save()
    return path, path[-1], pairs


def approach_fallback(
    engine: object, scene: object, goal: np.ndarray, options: dict, canceled: object, began: float
) -> tuple[list, dict]:
    """Reuse the incumbent overhead endpoint when a paired direct approach is blocked.

    Both legs use the existing 12D planner and shared deadline. Any unchecked
    connector remains explicitly a simulation fallback, including the final
    open-finger endpoint that can already touch the object.
    """
    from .approach import above_object_target
    from .dual_scene import SideAdapter
    from .rrt_placement import arm_path

    initial, pose = scene.fixed.copy(), scene.pose.copy()
    with np.load(engine.root / "object/object.npz", allow_pickle=False) as archive:
        points = archive["points_world_m"]
    above, endpoints = goal.copy(), {}
    for side in scene.sides:
        if canceled():
            raise InterruptedError("Paired approach canceled")
        block = side_slice(side)
        adapter = SideAdapter(engine.robot, engine.static, goal, side)
        endpoint = above_object_target(adapter, goal[block], points, engine.config["above_object_clearance_m"])
        above[side_slice(side, True)] = np.asarray(endpoint["positions"])[:6]
        endpoints[side] = endpoint
    first, first_report = arm_path(scene, above, options, canceled, began, allow_fallback=True)
    if not first:
        raise InterruptedError("Paired overhead approach canceled")
    scene.configure(above, "approach", pose)
    try:
        second, second_report = arm_path(scene, goal, options, canceled, began, allow_fallback=True)
        if not second:
            raise InterruptedError("Paired overhead descent canceled")
    finally:
        scene.configure(initial, "approach", pose)
    return first + second[1:], dict(
        execution_method="existing_overhead_endpoint",
        endpoints=endpoints,
        legs=[first_report, second_report],
        collision_certified=False,
    )
