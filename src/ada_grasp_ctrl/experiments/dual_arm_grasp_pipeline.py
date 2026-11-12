"""Independent offline dual-arm scene, single-winner minimum-effort grasp pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json, file_hash
from ada_grasp_ctrl.grasp_selection.multi_arm import (
    enabled_set,
    hand_names,
    select_candidates,
    validate_candidate,
    STAGES,
)
from ada_grasp_ctrl.robot_environment.model import side_slice
from .simulated_object_cloud import ObjectScene
from .mirrored_left_grasp import infer_sides
from .dual_arm_grasp_episode import execute
from .pregrasp_opening import opening_multiplier, prepare_candidates


def write_json(path: Path, value: object) -> None:
    """Write strict JSON, never silently exporting NaN diagnostics."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixture_candidates(scene: ObjectScene, identity: str) -> list[dict]:
    """Build explicit FK motion fixtures at the configured pose, never claim inferred grasps."""
    q = scene.robot.configuration(scene.sim.configuration())
    rows = []
    for side in ("left", "right"):
        targets = {}
        for stage in STAGES:
            targets[stage] = dict(
                frame_id="world",
                pose_frame=f"{side}_hand_flange",
                T_world_hand=scene.robot.fk(q, side).tolist(),
                joint_positions_rad=dict(zip(hand_names(side), q[side_slice(side)][6:].tolist())),
            )
        rows.append(
            validate_candidate(
                dict(
                    schema_version=1,
                    candidate_id=f"{side}:fk_fixture",
                    required_arms=[side],
                    targets={side: targets},
                    frame_id="world",
                    unit="m",
                    joint_unit="rad",
                    provenance=dict(
                        generator="fk_motion_fixture",
                        inference="fixture_only",
                        scene_identity=identity,
                        source_candidate_id="fk_fixture",
                        warning="No intended object grasp",
                    ),
                )
            )
        )
    return rows


def run(
    config: dict, output: Path, *, replay: Path | None = None, fixture: bool = False, perception_only: bool = False
) -> dict:
    """Run one new scene; persist reports even when physical grasping is unavailable."""
    enabled_set(config["selector"]["enabled_arms"])
    if config.get("schema_version") != 1:
        raise ValueError("Expected scene schema_version: 1")
    multiplier = opening_multiplier(config)
    scene = ObjectScene(config, output)
    scene.settle()
    points, info = scene.cloud()
    identity_data = dict(perception=info, q_current=scene.sim.configuration().tolist())
    identity = hashlib.sha256(
        json.dumps(identity_data, sort_keys=True, allow_nan=False).encode() + np.ascontiguousarray(points).tobytes()
    ).hexdigest()
    info["scene_identity"] = identity
    write_json(output / "perception.json", info)
    # Binary stores the compiled geometry and exact engine model for independent replay.
    mujoco.mj_saveModel(scene.sim.model, str(output / "scene.mjb"), None)
    np.savez_compressed(
        output / "initial_state.npz",
        qpos=scene.sim.data.qpos,
        qvel=scene.sim.data.qvel,
        ctrl=scene.sim.data.ctrl,
        time_s=scene.sim.data.time,
    )
    if perception_only:
        return dict(
            status="perception_only",
            execution_ready=False,
            scene_identity=identity,
            physics_available=scene.physics_available,
        )
    rejected = []
    if replay is not None:
        payload = read_json(replay)
        if payload.get("schema_version") != 1 or payload.get("scene_identity") != identity:
            raise ValueError("Replay scene identity differs: mesh/pose/camera/cloud/robot state must match")
        candidates = [validate_candidate(r) for r in payload["candidates"]]
        rejected = payload.get("conversion_rejections", [])
        mode = "cached_candidates"
    elif fixture:
        candidates = fixture_candidates(scene, identity)
        mode = "fixture_only"
    else:
        candidates, rejected = infer_sides(points, config, output, identity)
        mode = "actual_checkpoint"
    write_json(
        output / "candidates.json",
        dict(
            schema_version=1, scene_identity=identity, mode=mode, candidates=candidates, conversion_rejections=rejected
        ),
    )
    if multiplier != 1:
        candidates = prepare_candidates(candidates, scene.robot.limits, multiplier)
        write_json(
            output / "execution_candidates.json",
            dict(schema_version=1, scene_identity=identity, pregrasp_open_multiplier=multiplier, candidates=candidates),
        )
    selector = config["selector"]
    selection = select_candidates(
        scene.robot,
        scene.sim,
        candidates,
        selector["enabled_arms"],
        scene.sim.configuration(),
        selector.get("weights"),
        selector.get("max_nfev", 200),
    )
    selection["conversion_rejections"] = rejected
    write_json(output / "selection.json", selection)
    report = dict(
        schema_version=1,
        status=selection["status"],
        inference_mode=mode,
        scene_identity=identity,
        selected_side=None,
        effort=None,
        execution_ready=False,
        hardware_ready=False,
    )
    if selection["selected"] is None:
        return report
    chosen = selection["selected"]
    report.update(selected_side=chosen["selected_side"], candidate_id=chosen["candidate_id"], effort=chosen["effort"])
    episode = execute(scene, selection, config["execution"], output)
    write_json(output / "episode.json", episode)
    report.update(status=episode["status"], execution_ready=episode["execution_ready"], episode="episode.json")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run simulation only; exit 0 completed/perception, 1 physical failure, 2 input/runtime failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=["simulation"])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--replay", help="Scene-bound candidates.json from an earlier run")
    group.add_argument(
        "--fixture-current-pose", action="store_true", help="Explicit FK fixture, never actual inference"
    )
    parser.add_argument("--perception-only", action="store_true")
    args = parser.parse_args(argv)
    output = None
    try:
        path = resolve_from_root(args.scene)
        config = yaml.safe_load(path.read_text())
        destination = resolve_from_root(args.output_root or config["output"]["root"], "output")
        destination.mkdir(parents=True, exist_ok=False)
        output = destination
        config["output"]["root"] = str(output)
        (output / "effective_scene.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(output / "invocation.json", vars(args))
        (output / "run_manifest.yaml").write_text(
            yaml.safe_dump(
                dict(
                    scene=str(path),
                    scene_sha256=file_hash(path),
                    arguments=vars(args),
                    environment="simulation",
                    mujoco_version=mujoco.__version__,
                    numpy_version=np.__version__,
                )
            )
        )
        report = run(
            config,
            output,
            replay=resolve_from_root(args.replay) if args.replay else None,
            fixture=args.fixture_current_pose,
            perception_only=args.perception_only,
        )
        write_json(output / "run_report.json", report)
        print(json.dumps({**report, "output_root": str(output)}, indent=2))
        return 0 if report["status"] in ("success", "perception_only") else 1
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError, mujoco.FatalError) as exc:
        report = dict(status="preflight_or_runtime_failed", error=f"{type(exc).__name__}: {exc}")
        if output is not None:
            write_json(output / "run_report.json", report)
        print(report["error"], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
