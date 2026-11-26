"""Winner-only endpoint preparation and lightweight grasp-to-place preview geometry."""

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json, file_hash
from ada_grasp_ctrl.grasp_selection.contracts import candidate_from_schema1, plan_from_candidate
from ada_grasp_ctrl.grasp_selection.multi_arm import SEEDS, forbidden_contacts
from ada_grasp_ctrl.grasp_selection.types import jsonable
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.placement.target import transform_points
from ada_grasp_ctrl.robot_environment.kinematics import rigid
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.ros_execution.settings import DEFAULTS
from .dual_artifacts import single_endpoints
from .paired_artifacts import named_state
from .placement_schedule import PHASES, build_schedule, parameters

LABEL = "Kinematic preview — ideal object attachment; not a physical prediction or continuous path guarantee"


def endpoint_poses(endpoint: dict, candidate: dict, settings: dict) -> dict:
    """Offset the accepted wrist along the table normal and rotated source approach."""
    side = endpoint["arm"]
    place = rigid(endpoint["placement"]["T_world_hand_place"])
    target = endpoint["target"]
    normal = np.asarray(target["table_plane_world"], dtype=float)[:3]
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-10:
        raise ValueError("preplace: invalid table normal")
    normal = normal / np.linalg.norm(normal)
    source = candidate["targets"][side]
    direction = rigid(source["pregrasp"]["T_world_hand"])[:3, 3] - rigid(source["grasp"]["T_world_hand"])[:3, 3]
    if np.linalg.norm(direction) < 1e-10:
        raise ValueError("retreat: degenerate source pregrasp-to-grasp direction")
    direction = rigid(target["Delta_T"])[:3, :3] @ (direction / np.linalg.norm(direction))
    preplace, retreat_lift = place.copy(), place.copy()
    preplace[:3, 3] += settings["preplace_height_m"] * normal
    retreat_lift[:3, 3] += settings["retreat_lift_height_m"] * normal
    retreat = retreat_lift.copy()
    retreat[:3, 3] += settings["retreat_distance_m"] * direction
    return dict(preplace=preplace, retreat_lift=retreat_lift, retreat=retreat)


def solve_endpoint(
    robot: object,
    static: object,
    side: str,
    pose: np.ndarray,
    held: np.ndarray,
    phase: str,
    canceled: object,
    checks: dict,
) -> np.ndarray:
    """Reuse bounded IK seeds and contact policy, choosing the closest accepted arm branch."""
    arm = side_slice(side, True)
    mask = np.ones(56, dtype=bool)
    mask[arm] = False
    attempts, valid = [], []
    checks[phase] = dict(T_world_hand=pose.tolist(), attempts=attempts)
    for index, seed in enumerate([held[arm], *SEEDS]):
        if canceled():
            raise RuntimeError("Placement preparation canceled")
        q = held.copy()
        q[arm] = seed
        try:
            q, diagnostic = robot.solve_arm_ik(robot.configuration(q), side, pose)
            q = robot.configuration(q)
            if not np.array_equal(q[mask], held[mask]):
                raise ValueError("Endpoint IK changed fingers or passive arm")
            attempt = dict(seed_index=index, ik=diagnostic, accepted=False)
            if diagnostic["accepted"]:
                contacts = static.check_static_contacts(q)
                bad = forbidden_contacts(contacts, side, phase="squeeze")
                attempt.update(contacts=contacts, forbidden_contacts=bad, accepted=not bad, q56=q.tolist())
                if not bad:
                    valid.append((float(np.sum((q[arm] - held[arm]) ** 2)), index, q))
            attempts.append(attempt)
        except (RuntimeError, ValueError) as error:
            attempts.append(dict(seed_index=index, accepted=False, error=str(error)))
    if not valid:
        raise ValueError(f"{phase}: no collision-free accepted IK branch; selected candidate retained")
    _, index, q = min(valid, key=lambda item: item[:2])
    checks[phase].update(seed_index=index, q56=q.tolist())
    return q


def check_reference(robot: object, static: object, plan: dict, canceled: object, checks: dict) -> None:
    """Sample the shared reference at existing arm/hand command-step resolutions."""
    from .simulation_plan import reference

    side = plan["active_arms"][0]
    checks["phases"] = []
    for stage in plan["schedule"]["stages"]:
        data = stage["arms"][side]
        delta = np.abs(np.array(data["positions_rad"]) - data["start_positions_rad"])
        samples = max(
            2, int(np.ceil(max(max(delta[:6]) / DEFAULTS["arm_step"], max(delta[6:]) / DEFAULTS["hand_step"]))) + 1
        )
        record = dict(phase=stage["phase"], sampled_configurations=0, forbidden_contacts=[])
        checks["phases"].append(record)
        for alpha in np.linspace(0, 1, samples):
            if canceled():
                raise RuntimeError("Placement preparation canceled")
            q = robot.configuration(reference(plan, stage, float(alpha * stage["duration_s"])))
            phase = "pregrasp" if stage["phase"] in ("above_object", "pregrasp") else "squeeze"
            bad = forbidden_contacts(static.check_static_contacts(q), side, phase=phase)
            record["sampled_configurations"] += 1
            if bad:
                record.update(alpha=float(alpha), forbidden_contacts=bad)
                raise ValueError(
                    f"{stage['phase']}: nominal path collision at {alpha:.3f}; selected candidate retained"
                )


def prepare_execution(engine: object, endpoint: dict, destination: Path, canceled: object) -> dict:
    """Prepare only the already-selected winner; failures never invoke selection again."""
    from .simulation_plan import verify_plan

    began = time.monotonic()
    root, config, robot, static = engine.root, engine.config, engine.robot, engine.static
    selection = read_json(destination / "placement_selection_report.json")
    chosen = selection["selected"]
    if chosen["candidate_id"] != endpoint["candidate_id"] or chosen["required_arms"] != [endpoint["arm"]]:
        raise ValueError("Placement winner identity changed")
    settings = parameters(config.get("simulation", {}).get("placement", {}))
    checks = dict(continuous_path_verified=False, geometry="static robot/table scene without object solid")
    try:
        if canceled():
            raise RuntimeError("Placement preparation canceled")
        poses = endpoint_poses(endpoint, chosen["candidate"], settings)
        # Finish a higher, source-oriented Lift before the separate Reorient stage.
        # Keep the shared grasp-only/hardware lift height and existing IK policy intact.
        lift_config = {**config, "lift_height_m": config.get("simulation", {}).get("placement_lift_height_m", 0.15)}
        q, _, _, _, above, lift, phases, frames = single_endpoints(
            robot, static, selection, root / "robot_state.json", root / "object", lift_config
        )
        checks.update(above_object=above, lift=lift)
        side = endpoint["arm"]
        block = side_slice(side)
        fingers = slice(block.start + 6, block.stop)
        held = robot.configuration(endpoint["grasp_q56"])
        place = robot.configuration(endpoint["place_q56"])
        passive = side_slice("left" if side == "right" else "right")
        if not np.array_equal(place[passive], q[passive]) or not np.array_equal(place[fingers], held[fingers]):
            raise ValueError("Place changed the passive arm or held fingers")
        endpoints = {}
        for phase, frame in zip(phases[:5], frames[1:6]):
            full = q.copy()
            full[block] = frame
            endpoints[phase] = full
        if not np.array_equal(endpoints["squeeze"], held):
            raise ValueError("Selected squeeze differs from the accepted held reference")
        rotation_pose = robot.fk(endpoints["lift"], side).copy()
        rotation_pose[:3, :3] = poses["preplace"][:3, :3]
        reorient = solve_endpoint(robot, static, side, rotation_pose, place, "reorient", canceled, checks)
        preplace = solve_endpoint(robot, static, side, poses["preplace"], place, "preplace", canceled, checks)
        release_1, release_2 = place.copy(), place.copy()
        release_1[fingers] = endpoints["grasp"][fingers]
        release_2[fingers] = endpoints["pregrasp"][fingers]
        retreat_lift = solve_endpoint(
            robot, static, side, poses["retreat_lift"], release_2, "retreat_lift", canceled, checks
        )
        retreat = solve_endpoint(robot, static, side, poses["retreat"], retreat_lift, "retreat", canceled, checks)
        endpoints.update(
            reorient=reorient,
            transport=preplace,
            place=place,
            settle=place,
            release_1=release_1,
            release_2=release_2,
            retreat_lift=retreat_lift,
            retreat=retreat,
        )
        grasp = plan_from_candidate(
            candidate_from_schema1(chosen["candidate"], mode=chosen["candidate"].get("mode", side + "_full")),
            available_arms=selection["enabled_arms"],
        )
        schedule = build_schedule(
            grasp, named_state(q), {p: {side: named_state(v)[side]} for p, v in endpoints.items()}, settings
        )
        references = deepcopy(endpoint["references"])
        for path in (
            destination / "placement_endpoint_preview.json",
            destination / "placement_selection_report.json",
            root / "object/object.json",
            root / "scene_config.yaml",
            robot.urdf_path,
            robot.directory / "manifest.json",
            Path(config["table_plane"]),
        ):
            references[str(path)] = file_hash(path)
        plan = dict(
            schema_version=1,
            kind="ada_simulation_reference",
            environment="simulation",
            workflow="grasp_to_place",
            reference_hz=config.get("simulation", {}).get("reference_hz", 25),
            stage_completion="timed",
            schedule=schedule,
            execution_phases=list(PHASES),
            initial_q56=q.tolist(),
            full_joint_names=list(robot.joint_names),
            references=references,
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
            T_world_hand_source_held=endpoint["placement"]["T_world_hand_source_held"],
            accepted_place_q56=place.tolist(),
            placement_lift_height_m=float(lift_config["lift_height_m"]),
            label=LABEL,
        )
        check_reference(robot, static, plan, canceled, checks)
        plan["preparation_s"] = time.monotonic() - began
        verify_plan(plan)
        if canceled():
            raise RuntimeError("Placement preparation canceled")
        write_json(destination / "simulation_plan.json", plan)
        return plan
    finally:
        checks["preparation_s"] = time.monotonic() - began
        write_json(destination / "path_checks.json", jsonable(checks))


def preview_points(plan: dict, robot: object, points: np.ndarray, q: np.ndarray, phase: str) -> np.ndarray:
    """Compute display-only attachment from immutable points, never from a prior frame."""
    if phase in PHASES[:4]:
        return points
    if phase in ("lift", "reorient", "transport", "place"):
        transform = robot.fk(q, plan["active_arms"][0]) @ np.linalg.inv(rigid(plan["T_world_hand_source_held"]))
        return points @ transform[:3, :3].T + transform[:3, 3]
    return transform_points(points, plan["target"])
