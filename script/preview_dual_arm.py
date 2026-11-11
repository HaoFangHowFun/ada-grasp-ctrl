"""Validate a canonical dual-arm snapshot and optionally open a MuJoCo viewer."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import yaml

from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.robot_environment import DualArmModel, DualArmSimulation
from ada_grasp_ctrl.robot_environment.kinematics import residual
from ada_grasp_ctrl.robot_environment.model import side_slice


def body_transform(data: mujoco.MjData, name: str) -> np.ndarray:
    """Read world_T_body after MuJoCo forward kinematics."""
    pose = np.eye(4)
    pose[:3, :3] = data.body(name).xmat.reshape(3, 3)
    pose[:3, 3] = data.body(name).xpos
    return pose


def validate(robot: DualArmModel, sim: DualArmSimulation) -> dict:
    """Measure FK, IK, independent commands, contacts and headless physics."""
    home = robot.home_configuration()
    rng = np.random.default_rng(42)
    cases = [home]
    for i in range(56):
        q = home.copy()
        q[i] = min(robot.limits[i, 1], 0.1)
        cases.append(q)
    cases.extend(rng.uniform(robot.limits[:, 0], robot.limits[:, 1], size=(20, 56)))
    maxima = {"position_m": 0.0, "orientation_rad": 0.0}
    frame_maxima = {}
    frames = (
        "base",
        "crx_flange",
        "ee_mount",
        "hand_flange",
        "palm",
        "tcp",
        "thumb_fingertip",
        "index_fingertip",
        "middle_fingertip",
        "ring_fingertip",
        "pinky_fingertip",
    )
    for q in cases:
        sim.data.qpos[sim.qpos] = q
        mujoco.mj_forward(sim.model, sim.data)
        for side in ("left", "right"):
            for frame in frames:
                name = robot.frame_name(side, f"{side}_{frame}" if frame.endswith("fingertip") else frame)
                error = residual(robot.fk(q, side, name), body_transform(sim.data, name))
                frame_maxima.setdefault(name, dict.fromkeys(maxima, 0.0))
                for key, value in error.items():
                    maxima[key] = max(maxima[key], value)
                    frame_maxima[name][key] = max(frame_maxima[name][key], value)
    ik_results = {}
    for side in ("left", "right"):
        q = home.copy()
        arm = side_slice(side, True)
        q[arm] = [0.2, -0.3, 0.5, 0.2, -0.3, 0.1]
        candidate, diagnostic = robot.solve_arm_ik(home, side, robot.fk(q, side))
        held = np.ones(56, dtype=bool)
        held[arm] = False
        diagnostic["other_50_unchanged"] = bool(np.array_equal(candidate[held], home[held]))
        ik_results[side] = diagnostic
    sim.reset(home)
    command_checks = {}
    for side, other in (("left", "right"), ("right", "left")):
        sim.select_side(side)
        before = sim.targets[side_slice(other)]
        target = home[side_slice(side)].copy()
        target[0] = 0.1
        sim.set_target(target)
        command_checks[side] = bool(np.array_equal(sim.targets[side_slice(other)], before))
    contacts_home = sim.check_static_contacts(home)
    crafted = {}
    contact_rng = np.random.default_rng(42)
    for index in range(400):
        q = contact_rng.uniform(robot.limits[:, 0], robot.limits[:, 1])
        for contact in sim.check_static_contacts(q):
            category = contact["category"]
            if category not in crafted and contact["penetration_m"] > 1e-5:
                crafted[category] = {"seed": 42, "sample_index": index, "q_rad": q.tolist(), "contact": contact}
        if {"within_robot", "robot_table", "left_right"} <= crafted.keys():
            break
    timings = {}
    for count in (100, 1000):
        start = time.perf_counter()
        for _ in range(count):
            sim.check_static_contacts(home)
        elapsed = time.perf_counter() - start
        timings[str(count)] = {"total_s": elapsed, "mean_ms": elapsed * 1000 / count}
    sim.reset(home)
    sim.step(10)
    steps = {
        "count": 10,
        "time_s": float(sim.data.time),
        "finite": bool(np.isfinite(sim.data.qpos).all()),
        "max_joint_displacement_rad": float(np.max(np.abs(sim.configuration() - home))),
        "max_soft_limit_violation_rad": float(
            max(
                0,
                np.max(robot.limits[:, 0] - sim.configuration()),
                np.max(sim.configuration() - robot.limits[:, 1]),
            )
        ),
        "warnings": [int(w.number) for w in sim.data.warning],
    }
    transforms = {}
    for side in ("left", "right"):
        base = robot.fk(home, side, "base")
        mount = robot.fk(home, side, "ee_mount")
        hand = robot.fk(home, side, "hand_flange")
        transforms[side] = {
            "world_T_base": base.tolist(),
            "ee_mount_T_hand_flange": np.linalg.solve(mount, hand).tolist(),
            "world_T_hand_flange": hand.tolist(),
            "frames": robot.frames[side],
        }
    passed = (
        all(value <= 1e-6 for value in maxima.values())
        and all(d["accepted"] and d["other_50_unchanged"] for d in ik_results.values())
        and all(command_checks.values())
        and steps["finite"]
        and not any(steps["warnings"])
        and {"within_robot", "robot_table", "left_right"} <= crafted.keys()
        and "rclpy" not in sys.modules
    )
    return {
        "passed": passed,
        "robot_joint_count": len(robot.joint_names),
        "joint_names": list(robot.joint_names),
        "mapping": sim.mapping,
        "source_manifest": robot.manifest,
        "frames": transforms,
        "fk": {"cases": len(cases), "max_errors": maxima, "by_frame": frame_maxima, "tolerance": 1e-6},
        "ik": ik_results,
        "passive_command_unchanged": command_checks,
        "contacts": {
            "home": contacts_home,
            "home_counts": dict(Counter(c["category"] for c in contacts_home)),
            "crafted": crafted,
            "benchmark": timings,
            "explicit_exclusions": int(sim.model.nexclude),
            "policy": "diagnostic only; source colliders retained; default MuJoCo filtering applies",
        },
        "physics": steps,
        "simulation_parameters": {
            "timestep_s": float(sim.model.opt.timestep),
            "integrator": "implicitfast",
            "arm_kp_kv": [100, 10],
            "hand_kp_kv": [7.5, 0],
            "hand_damping": 0.05,
            "hand_armature": 1e-5,
            "status": "Provisional numerical gains, not hardware identification or trajectory validation",
        },
        "rclpy_imported": "rclpy" in sys.modules,
        "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__},
    }


def main() -> int:
    """Run a fresh validation report and optional viewer using only numerical packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    output = resolve_from_root(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    robot = DualArmModel.load(args.snapshot)
    sim = DualArmSimulation(robot)
    (output / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "snapshot": str(robot.directory),
                "output_root": str(output),
                "seed": 42,
                "viewer": args.viewer,
                "mount_calibrated": False,
            }
        )
    )
    report = validate(robot, sim)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "fk": report["fk"]["max_errors"],
                "ik": report["ik"],
                "report": str(output / "report.json"),
            },
            indent=2,
        )
    )
    if args.viewer and report["passed"]:
        from mujoco import viewer

        sim.reset(robot.home_configuration())
        with viewer.launch_passive(sim.model, sim.data) as window:
            while window.is_running():
                start = time.perf_counter()
                sim.step()
                window.sync()
                time.sleep(max(0, sim.model.opt.timestep - (time.perf_counter() - start)))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Preview preflight failed: {error}", file=sys.stderr)
        sys.exit(2)
