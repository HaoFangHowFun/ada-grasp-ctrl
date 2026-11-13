"""Capture named live states for both robot branches without command publishers."""

from __future__ import annotations

import argparse
import json
import os
import time

from ada_grasp_ctrl.experiments.right_ros_contract import side_names, check_kinematics, kinematic_contract
from ada_grasp_ctrl.paths import resolve_from_root, resolve_external_root


def main(argv: list[str] | None = None) -> int:
    """Save one attributable dual-side snapshot for selection and execution review."""
    import rclpy

    from .runtime import Runtime, RunLog
    from .model import Model
    from .settings import bridge_config, validate_domain
    from .snapshot import snapshot_receiver

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-urdf", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--environment", choices=("mock", "hardware"), required=True)
    args = parser.parse_args(argv)
    validate_domain(args.environment, os.environ.get("ROS_DOMAIN_ID"))
    root = resolve_external_root("output", args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    expected = resolve_from_root(args.model_urdf).read_text()
    runtimes, logs = {}, []
    rclpy.init(args=[])
    try:
        for side in ("left", "right"):
            config = bridge_config({"side": side, "environment": args.environment, "active_arms": ["left", "right"]})
            log = RunLog(str(root / side), config, args)
            logs.append(log)
            rt = Runtime(config, log)
            runtimes[side] = rt
            rt.receive = snapshot_receiver(rt)
            rt.preflight(False)
            check_kinematics(kinematic_contract(expected, side), kinematic_contract(rt.description, side))
        deadline = time.monotonic() + 20
        while True:
            for rt in runtimes.values():
                rt.spin(0.01)
            if all(rt.fresh() for rt in runtimes.values()):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Fresh dual-arm feedback unavailable")
        result = dict(schema_version=1, environment=args.environment, unit="rad", frame_id="world", sides={})
        for side, rt in runtimes.items():
            model = Model(rt.description, rt.c)
            result["sides"][side] = dict(
                joint_names=list(sum(side_names(side), ())),
                positions_rad=[v for g in ("arm", "hand") for v in model.check(g, rt.feedback[g].q).tolist()],
                stamps_s=rt.stamps,
                environment_evidence=rt.log.result["driver_interfaces"],
            )
            rt.log.result.update(final_state="SNAPSHOT", task_completed=True)
        if runtimes["left"].description != runtimes["right"].description:
            raise ValueError("Combined model changed during dual snapshot")
        result["captured_wall_s"] = time.time()
        result["full_joint_names"] = [n for s in ("left", "right") for n in sum(side_names(s), ())]
        result["full_positions_rad"] = [v for s in ("left", "right") for v in result["sides"][s]["positions_rad"]]
        (root / "observed_model.urdf").write_text(runtimes["left"].description)
        (root / "robot_state.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return 0
    finally:
        for rt in runtimes.values():
            rt.node.destroy_node()
        for log in logs:
            log.finish()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
