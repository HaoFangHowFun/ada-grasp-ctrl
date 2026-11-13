"""Reviewed stages on a deterministic mock plant, with optional tactile replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np

from ada_grasp_ctrl.contact import ContactObservation
from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, validate_packet
from ada_grasp_ctrl.pipeline.plan import validate_plan
from ada_grasp_ctrl.ros_execution.motion import arrival_reached, smoothstep
from .contracts import JointFeedback, MPCProposal, controller_mode
from .native import NativeMPC
from .orchestration import StageOrchestrator
from .reference import linear_reference
from .streaming import GuardedStream
from .baseline_stream import BaselineStream


def mujoco_contact_source(scene: Path) -> Callable[[float, JointFeedback], ContactObservation]:
    """Sample an explicitly selected mock scene at measured q without stepping hardware.

    This is a kinematic MuJoCo contact fixture, not a dynamically validated grasp.
    Real dynamic contact/trajectory parity is tested by the right-full evaluator.
    """
    import mujoco
    from ada_grasp_ctrl.contact.mujoco_adapter import MuJoCoContactAdapter

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    base = model.body("right_hand_flange").id
    ids = {base}
    for i in range(model.nbody):
        if model.body_parentid[i] in ids:
            ids.add(i)
    adapter = MuJoCoContactAdapter(model, data, hand_bodies={model.body(i).name for i in ids}, object_bodies={"object"})
    addresses = [model.joint(n).qposadr[0] for n in JOINT_NAMES]

    def observe(now: float, feedback: JointFeedback) -> ContactObservation:
        data.qpos[addresses] = feedback.positions_rad
        data.time = now
        mujoco.mj_forward(model, data)
        return adapter.observe()

    return observe


def json_value(value: object) -> object:
    """Normalize native numerical diagnostics for portable strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_runner(report: dict, plan: dict, output: Path) -> SimpleNamespace:
    """Construct native robot/adaptor/controller without a simulator actuator handle."""
    from ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation import make_config
    from ada_grasp_ctrl.experiments.crx_sharpa_right_simulation import RightCrxSharpa
    from ada_grasp_ctrl.utils.grasp_controller import GraspController
    from ada_grasp_ctrl.utils.robot_adaptor import RobotAdaptor
    from ada_grasp_ctrl.utils.pin_helper import PinocchioHelper

    from ada_grasp_ctrl.paths import configure_runtime_roots, runtime_root

    previous_roots = {kind + "_root": runtime_root(kind) for kind in ("asset", "data", "output")}
    try:
        config = make_config(report, report["samples"][0], output, "ours")
        robot = RightCrxSharpa(report["model"])
        kind = config.hand.get("kinematics_file_type", "mjcf")
        model = PinocchioHelper(robot.get_file_path(kind), kind)
        adaptor = RobotAdaptor(model, robot.dof_names, robot.doa_names, robot.doa2dof_matrix)
        grasp_ctrl = GraspController(config.task.control, robot, adaptor)
    finally:
        # The historical experiment factory activates process-global path roots.
        # This reusable offline boundary must not change subsequent pipeline calls.
        configure_runtime_roots(**previous_roots)
    paths = []
    for phase in ("grasp", "squeeze"):
        i = plan["phases"].index(phase)
        duration = plan["durations_s"][i]
        count = max(1, int(np.ceil(duration / 0.1)))
        paths.append(
            np.array([linear_reference(*plan["keyframes"][i : i + 2], (j + 1) * 0.1, duration) for j in range(count)])
        )
    return SimpleNamespace(
        robot=robot,
        robot_adaptor=adaptor,
        grasp_ctrl=grasp_ctrl,
        evaluator=SimpleNamespace(configs=config, action_dt=0.1),
        path_approach=paths[0],
        path_squeeze=paths[1],
        qpos_path=np.concatenate(paths),
    )


def run_packet(
    packet: dict,
    model_report: dict,
    output: Path,
    mode: str,
    *,
    tactile_replay: str | None = None,
    simulation_scene: str | None = None,
    contact_source: Callable[[float, JointFeedback], ContactObservation] | None = None,
) -> dict:
    """Execute reviewed mock endpoints with a 40 ms mock measurement lag.

    Missing contact is rejected, never silently replaced by a valid empty frame.
    A MuJoCo source can be injected with its own explicitly configured scene.
    This is a deterministic offline clock, not a ROS or wall-clock timing claim.
    """
    validate_packet(packet)
    plan = packet["reviewed_plan"]
    validate_plan(plan)
    controller_mode({"environment": packet["environment"], "side": plan.get("side", "right")}, mode)
    if packet["environment"] != "mock":
        raise ValueError("Offline packet executor refuses hardware packets")
    if mode == "linear" and plan["interpolation"] != "linear":
        raise ValueError("Linear mode requires a newly exported matching linear preview plan")
    from ada_grasp_ctrl.experiments.right_ros_contract import check_kinematics, kinematic_contract
    from ada_grasp_ctrl.paths import resolve_from_root
    from ada_grasp_ctrl.ros_execution.review import check_reviewed_goal
    from ada_grasp_ctrl.ros_execution.settings import bridge_config

    check_kinematics(packet["kinematics"], kinematic_contract(Path(model_report["model"]["urdf"]).read_text()))
    config = {k: v for k, v in plan["receiver_settings"].items() if k not in ("mode", "path_verified")}
    resolved_settings = bridge_config(config)
    for i, target in enumerate(packet["targets"]):
        check_reviewed_goal(packet, i, target["positions"], target["duration_s"], resolved_settings)
    if sum(bool(v) for v in (tactile_replay, simulation_scene, contact_source)) > 1:
        raise ValueError("Select exactly one contact source")
    if simulation_scene:
        simulation_scene = str(resolve_from_root(simulation_scene))
        contact_source = mujoco_contact_source(Path(simulation_scene))
    if tactile_replay:
        tactile_replay = str(resolve_from_root(tactile_replay))
    output.mkdir(parents=True, exist_ok=False)
    runner = make_runner(model_report, plan, output)
    native = NativeMPC(runner)
    replay, estimator = None, None
    if tactile_replay:
        from ada_grasp_ctrl.contact.replay import TactileReplay
        from ada_grasp_ctrl.contact.compatibility import TactileContactEstimator

        replay = TactileReplay(Path(tactile_replay))
        estimator = TactileContactEstimator(
            runner.robot_adaptor,
            object_bodies=("object",),
            tactile_joint_names=replay.joint_names,
            friction_mu=runner.grasp_ctrl.mu,
            allow_candidate_geometry=True,
        )
    now = replay.origin if replay else 0.0
    q = tuple(plan["keyframes"][0])
    rows, commands = [], []
    orchestrator = StageOrchestrator(mode, tuple(plan["phases"]), native.tick, rows.append)
    limits = runner.robot_adaptor.joint_limits_f
    result = {
        "schema_version": 1,
        "mode": mode,
        "hardware_used": False,
        "ros_publishers": 0,
        "mpc_command_count": 0,
        "baseline_command_count": 0,
        "left_command_count": 0,
        "clock": "injected_monotonic",
        "mock_feedback_lag_s": 0.04,
        "completed_stages": [],
        "reference_segments_completed": [],
        "contact_source": "tactile_replay" if replay else "injected" if contact_source else "unavailable",
        "simulation_scene": simulation_scene,
        "simulation_contact_semantics": "kinematic_mj_forward_fixture" if simulation_scene else None,
    }
    try:
        stream = BaselineStream(tuple(limits[0]), tuple(limits[1]), tuple(runner.robot.doa_max_vel), config=config)
        stream.start(JointFeedback(tuple(JOINT_NAMES), q, now, now, "offline_lagged_mock"), now, owner="baseline")
        for phase_index, phase in enumerate(plan["phases"]):
            orchestrator.enter(phase, previous_arrived=phase_index > 0)
            active = mode == "mpc_active" and phase in ("grasp", "squeeze")
            owner = "mpc_mock" if active else "baseline"
            if owner != stream.owner:
                stream.stop("phase ownership handoff")
                stream = (
                    GuardedStream(config=config)
                    if active
                    else BaselineStream(
                        tuple(limits[0]), tuple(limits[1]), tuple(runner.robot.doa_max_vel), config=config
                    )
                )
                stream.start(JointFeedback(tuple(JOINT_NAMES), q, now, now, "offline_lagged_mock"), now, owner=owner)
            began, start, settled = now, q, None
            duration = plan["durations_s"][phase_index]
            target = tuple(plan["keyframes"][phase_index + 1])
            while True:
                now = round(now + 0.01, 9)
                elapsed = now - began
                q_ref = (
                    tuple(runner.qpos_path[min(native.index, len(runner.qpos_path) - 1)])
                    if active
                    else tuple(linear_reference(start, target, elapsed, duration))
                )
                feedback = JointFeedback(tuple(JOINT_NAMES), q, now, now, "offline_lagged_mock")
                contact = ContactObservation(now, (), "unavailable", False, ("No contact source supplied",))
                try:
                    if contact_source and mode.startswith("mpc_"):
                        contact = contact_source(now, feedback)
                    elif replay and mode.startswith("mpc_"):
                        event = replay.sample(now)
                        if event:
                            snapshot, recorded_feedback = event
                            estimate = estimator.estimate(snapshot, recorded_feedback, now)
                            # A recording cannot supply the mock plant's current pose. Require
                            # agreement rather than mixing recorded contacts with a different q.
                            coherent = np.allclose(recorded_feedback.positions_rad, q, atol=1e-6, rtol=0)
                            if estimate.observation is not None and coherent:
                                contact = estimate.observation
                            else:
                                reasons = estimate.reasons or ("Replay state differs from current mock state",)
                                contact = ContactObservation(now, (), "tactile_replay", False, reasons)
                except Exception as error:
                    if mode != "mpc_passive":
                        raise
                    contact = ContactObservation(now, (), "unavailable", False, (str(error),))
                proposal = orchestrator.tick(now, feedback, stream.q, q_ref, contact)
                if active:
                    if proposal is not None:
                        if proposal.diagnostics.get("termination") == "native_complete":
                            # Run the executor safety checks before acknowledging completion.
                            stream.check(now, feedback)
                            result["native_termination"] = proposal.diagnostics
                            break
                        stream.accept(proposal, now)
                else:
                    reference = (
                        q_ref
                        if plan["interpolation"] == "linear"
                        else tuple(a + smoothstep(min(1, elapsed / duration)) * (b - a) for a, b in zip(start, target))
                    )
                    stream.accept(MPCProposal(tuple(JOINT_NAMES), reference, True, now, 0, "baseline"), now)
                command = stream.tick(now, feedback)
                if command is not None:
                    result["mpc_command_count" if active else "baseline_command_count"] += 1
                    commands.append(
                        {
                            "timestamp": now,
                            "stage": phase,
                            "owner": stream.owner,
                            "q_ref": q_ref,
                            "q_100Hz": command,
                            "measured_q": q,
                            "proposal_executed": active,
                        }
                    )
                    if active:
                        stream.published(command, now)
                    q = tuple(a + 0.25 * (b - a) for a, b in zip(q, command))
                if active:
                    # Grasp/squeeze label the reference segments, not Ours policy stages.
                    # Only native completion may leave the combined control episode.
                    if phase == "grasp" and native.index >= len(runner.path_approach):
                        break
                    continue
                error = [b - a for a, b in zip(q, target)]
                arrived = arrival_reached(error, stream.config)
                if elapsed >= duration and arrived:
                    settled = now if settled is None else settled
                    if now - settled >= plan["settle_time_s"]:
                        break
                else:
                    settled = None
                if elapsed > duration + stream.config["settle_timeout"]:
                    raise ValueError(f"{phase}: arrival timeout")
            if active and phase == "grasp":
                result["reference_segments_completed"].append(phase)
            else:
                if active:
                    result["completed_stages"].append("grasp")
                result["completed_stages"].append(phase)
        result["status"] = "completed"
    except Exception as error:
        result.update(status="aborted", reason=str(error))
        raise
    finally:
        if "stream" in locals() and stream.running:
            stream.stop("shutdown")
        for name, records in (("proposals.jsonl", rows), ("commands.jsonl", commands)):
            with (output / name).open("w") as file:
                for record in records:
                    file.write(json.dumps(json_value(record), allow_nan=False) + "\n")
        result["proposal_count"] = sum(bool(row.get("proposal")) for row in rows)
        result["valid_proposals"] = sum(
            bool(row.get("proposal", {}).get("valid")) for row in rows if row.get("proposal")
        )
        (output / "run_report.json").write_text(json.dumps(json_value(result), indent=2, allow_nan=False))
        (output / "run_manifest.yaml").write_text(
            json.dumps(json_value({"packet": packet, "mode": mode, "hardware_enable": False}), indent=2)
        )
    return result
