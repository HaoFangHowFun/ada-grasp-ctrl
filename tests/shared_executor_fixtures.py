"""Deterministic synthetic poses/model for offline and isolated DDS transport tests."""

import argparse
from copy import deepcopy
import hashlib
import json
import xml.etree.ElementTree as ET

from ada_grasp_ctrl.experiments.right_ros_contract import CYCLE_PHASES, PHASES, kinematic_contract, side_names
from ada_grasp_ctrl.experiments.right_ros_sender import motion_duration
from ada_grasp_ctrl.ros_execution.settings import DEFAULTS
from ada_grasp_ctrl.ros_execution.shared_packet import KIND, LIMIT_KEYS, SIDES, validate_shared_packet


def description():
    root = ET.Element("robot", name="synthetic_shared_mock")
    ET.SubElement(root, "link", name="world")
    for side in SIDES:
        parent = "world"
        for index, name in enumerate(side_names(side)[0]):
            child = f"{side}_J{index + 1}_link"
            ET.SubElement(root, "link", name=child)
            joint = ET.SubElement(root, "joint", name=name, type="revolute")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "origin", xyz="0 0 0.1", rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz="0 0 1")
            ET.SubElement(joint, "limit", lower="-1", upper="1", velocity="1", effort="100")
            parent = child
        for suffix in ("hand_flange", "hand_C_MC"):
            child = f"{side}_{suffix}"
            ET.SubElement(root, "link", name=child)
            joint = ET.SubElement(root, "joint", name=child + "_mount", type="fixed")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            parent = child
        for name in side_names(side)[1]:
            child = name + "_link"
            ET.SubElement(root, "link", name=child)
            joint = ET.SubElement(root, "joint", name=name, type="revolute")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=child)
            ET.SubElement(joint, "axis", xyz="0 0 1")
            ET.SubElement(joint, "limit", lower="-1", upper="1", velocity="1", effort="10")
    return ET.tostring(root, encoding="unicode")


def schedule(active=SIDES, *, full=False):
    active = list(active)
    mode = "both_full" if len(active) == 2 else active[0] + "_full"
    plan = dict(
        schema_version=1,
        mode=mode,
        candidate_id=mode + ":synthetic",
        active_arms=active,
        available_arms=list(SIDES),
        frame_id="world",
        unit="m",
        joint_unit="rad",
        provenance={"source": "synthetic_transport_fixture"},
        targets={s: {} for s in active},
        execution_ready=False,
        hardware_ready=False,
    )
    policies = {s: dict(minimum_duration=2.0, **{key: DEFAULTS[key] for key in LIMIT_KEYS}) for s in active}
    phases = ("above_object", *CYCLE_PHASES) if full else PHASES
    previous, stages, cursor = {s: [0.0] * 28 for s in active}, [], 0.0
    for phase in phases:
        data, required = {}, {}
        for side in active:
            arm = (0.03 if side == "left" else 0.4) + (0.02 if phase != "pregrasp" else 0)
            hand = 0.05 if phase == "pregrasp" else 0.1
            if phase in ("above_object", "return"):
                arm = hand = 0.0
            elif phase == "release":
                hand = 0.0
            end = [arm] * 6 + [hand] * 22
            data[side] = dict(
                joint_names=list(sum(side_names(side), ())),
                start_positions_rad=previous[side].copy(),
                positions_rad=end,
            )
            required[side] = (
                0.5 if phase == "hold" else motion_duration(previous[side], end, argparse.Namespace(**policies[side]))
            )
            previous[side] = end
            if phase in PHASES:
                plan["targets"][side][phase] = dict(
                    frame_id="world",
                    pose_frame=f"{side}_hand_flange",
                    T_world_hand=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    joint_positions_rad=dict(zip(side_names(side)[1], end[6:])),
                )
        duration = max(required.values())
        stages.append(
            dict(
                phase=phase,
                stage_start_time=cursor,
                stage_end_time=cursor + duration,
                settled_time=cursor + duration + 0.3,
                duration_s=duration,
                required_durations_s=required,
                arms=data,
            )
        )
        cursor = stages[-1]["settled_time"]
    return dict(
        schema_version=1,
        kind="ada_shared_stage_schedule",
        plan=plan,
        active_arms=active,
        reference="linear",
        joint_unit="rad",
        stage_start_time=0.0,
        settle_time_s=0.3,
        hold_s=0.5,
        policies_by_arm=policies,
        stages=stages,
        end_time=cursor,
        timeline="nominal_until_preceding_stage_succeeds",
        completion_policy="retain_existing_arrival_settle_timeout_and_abort_checks",
        acceleration_continuous=False,
        continuous_path_verified=False,
        execution_ready=False,
        hardware_ready=False,
    )


def packet(active=SIDES, *, full=False):
    source = schedule(active, full=full)
    transport = deepcopy(source)
    plan = transport.pop("plan")
    return validate_shared_packet(
        dict(
            schema_version=1,
            kind=KIND,
            environment="mock",
            hardware_ready=False,
            plan_identity={
                k: plan[k]
                for k in ("schema_version", "candidate_id", "mode", "active_arms", "available_arms", "provenance")
            },
            schedule=transport,
            snapshot={s: dict(joint_names=list(sum(side_names(s), ())), positions_rad=[0.0] * 28) for s in SIDES},
            scene_kinematics={s: kinematic_contract(description(), s) for s in SIDES},
            source_schedule_sha256=hashlib.sha256(
                json.dumps(source, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        )
    )


def limits():
    return {
        s: dict(joint_names=list(sum(side_names(s), ())), lower=[-1.0] * 28, upper=[1.0] * 28, velocity=[1.0] * 28)
        for s in SIDES
    }


def feedback(now, positions):
    return {
        s: dict(
            joint_names=list(sum(side_names(s), ())),
            positions_rad=list(positions[s]),
            source_timestamp=now,
            receive_timestamp=now,
        )
        for s in SIDES
    }
