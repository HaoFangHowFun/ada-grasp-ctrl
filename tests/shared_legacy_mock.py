"""Exercise the schema-1 right action receiver against synthetic DDS drivers."""

import hashlib
import json
import sys

from ada_grasp_ctrl.experiments.right_ros_contract import side_names, kinematic_contract
from ada_grasp_ctrl.ros_execution.settings import bridge_config
from shared_executor_fixtures import packet


def run_legacy(root, description, node, spin_until, start_child, stop_child):
    import yaml
    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    from ada_grasp_ctrl.experiments.right_ros_sender import load_packet

    def write(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False))

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    model, selected, snapshot, plan_path, target_path = (
        root / name for name in ("model.urdf", "selection.json", "snapshot.json", "motion_plan.json", "targets.json")
    )
    model.write_text(description)
    source = packet(("right",))
    names = list(sum(side_names("right"), ()))
    stages = source["schedule"]["stages"]
    config = bridge_config(
        dict(side="right", active_arms=["left", "right"], hand_arrival_required=False, interpolation="linear")
    )
    config_path = root / "receiver.yaml"
    config_path.write_text(yaml.safe_dump({k: v for k, v in config.items() if k not in ("mode", "path_verified")}))
    plan = dict(
        schema_version=1,
        side="right",
        environment="mock",
        candidate_id="legacy-fixture",
        joint_names=names,
        phases=[s["phase"] for s in stages],
        keyframes=[[0.0] * 28] + [s["arms"]["right"]["positions_rad"] for s in stages],
        durations_s=[s["duration_s"] for s in stages],
        settle_time_s=0.3,
        interpolation="linear",
        controller_mode="linear",
        receiver_settings=config,
        passive_positions_rad=[0.0] * 28,
    )
    write(plan_path, plan)
    write(
        selected,
        dict(
            selected_side="right",
            selected_candidate_id="legacy-fixture",
            selected=dict(stage_checks={s["phase"]: dict(q_rad=s["arms"]["right"]["positions_rad"]) for s in stages}),
        ),
    )
    write(
        snapshot,
        dict(side="right", environment="mock", full_positions_rad=[0.0] * 28, passive_positions_rad=[0.0] * 28),
    )
    provenance = dict(
        kind="grasp_selection",
        model_path=str(model),
        model_sha256=digest(model),
        selection_path=str(selected),
        snapshot_path=str(snapshot),
        plan_path=str(plan_path),
        plan_sha256=digest(plan_path),
        references={str(p): digest(p) for p in (model, selected, snapshot, plan_path)},
    )
    payload = dict(
        schema_version=1,
        kind="ada_right_joint_targets",
        side="right",
        environment="mock",
        sample_id="legacy-fixture",
        units="rad",
        joint_names=names,
        reference_frame="world",
        pose_frame="right_hand_flange",
        quaternion_order="WXYZ",
        execution_phases=plan["phases"],
        source=provenance,
        kinematics=kinematic_contract(description, "right"),
        scene_kinematics={s: kinematic_contract(description, s) for s in ("left", "right")},
        reviewed_plan=plan,
        targets=[
            dict(
                phase=s["phase"],
                positions=s["arms"]["right"]["positions_rad"],
                duration_s=s["duration_s"],
                flange_target_wxyz=[0, 0, 0, 1, 0, 0, 0],
            )
            for s in stages
        ],
    )
    write(target_path, payload)
    write(
        root / "run_manifest.yaml",
        dict(
            mode="ada_ros_target_export",
            sample_id="legacy-fixture",
            source=provenance,
            targets_sha256=digest(target_path),
        ),
    )
    assert load_packet(target_path) == payload
    child = start_child(
        [
            sys.executable,
            "-m",
            "ada_grasp_ctrl.ros_execution.bridge",
            "--config",
            str(config_path),
            "--targets",
            str(target_path),
            "--output-dir",
            str(root / "receiver"),
        ],
        root / "executor.log",
    )
    client = ActionClient(node, FollowJointTrajectory, "/bimanual/right_arm_hand/follow_joint_trajectory")
    outcomes = []
    try:
        spin_until(lambda: client.server_is_ready() or child.poll() is not None, 20)
        if child.poll() is not None:
            raise RuntimeError((root / "executor.log").read_text())
        for target in payload["targets"]:
            duration = target["duration_s"]
            goal = FollowJointTrajectory.Goal(
                trajectory=JointTrajectory(
                    joint_names=names,
                    points=[
                        JointTrajectoryPoint(
                            positions=target["positions"],
                            time_from_start=Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9)),
                        )
                    ],
                )
            )
            sent = client.send_goal_async(goal)
            spin_until(sent.done, 10)
            handle = sent.result()
            assert handle.accepted
            result = handle.get_result_async()
            spin_until(result.done, 30)
            response = result.result()
            assert response.result.error_code == 0, response.result.error_string
            outcomes.append(dict(phase=target["phase"], status=response.status, error_code=response.result.error_code))
        write(root / "legacy_actions.json", outcomes)
    finally:
        client.destroy()
        stop_child(child)
