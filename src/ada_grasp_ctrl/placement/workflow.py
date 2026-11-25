"""Reuse live object selection, then stop at placement confirmation in the existing preview."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np
import yaml

from ada_grasp_ctrl.experiments.fixed_camera import load_plane, load_snapshot
from ada_grasp_ctrl.paths import project_root, resolve_from_root
from .session import PlacementSession, sha256, write_json


def prepare_session(object_root: Path, root: Path) -> PlacementSession:
    """Copy the exact selected partial cloud and its captured environment for reproducible review."""
    info = json.loads((object_root / "object.json").read_text())
    if info.get("schema_version") != 1 or info.get("frame_id") != "world" or info.get("unit") != "m":
        raise ValueError("Placement requires a world/metres captured object")
    object_file = object_root / "object.npz"
    if sha256(object_file) != info["object_sha256"]:
        raise ValueError("Selected object changed since extraction")
    plane_path = resolve_from_root(info["plane_path"])
    if sha256(plane_path) != info["plane_sha256"]:
        raise ValueError("Table changed since object extraction")
    plane = load_plane(plane_path)
    arrays, capture = load_snapshot(resolve_from_root(info["source_snapshot"]))
    if capture["snapshot_sha256"] != info["snapshot_sha256"]:
        raise ValueError("Capture changed since object extraction")
    if (
        plane["extrinsics"]["sha256"] != capture["extrinsics"]["sha256"]
        or info["extrinsics"]["sha256"] != capture["extrinsics"]["sha256"]
        or plane["serial"] != capture["serial"]
    ):
        raise ValueError("Object, capture and table calibration must match")
    with np.load(object_file, allow_pickle=False) as data:
        points = data["points_world_m"].copy()
        indices = data["source_indices"]
        if not np.array_equal(points, arrays["points_world_m"][indices]):
            raise ValueError("Object points differ from original captured snapshot")
    # Preserve all observed points. Only the display may subsample the environment.
    np.savez_compressed(
        root / "snapshot.npz",
        object_points_world_m=points,
        environment_points_world_m=arrays["points_world_m"],
        object_source_indices=indices,
        table_normal_world=plane["normal_xyz"],
        T_world_color=arrays["T_world_color"],
    )
    write_json(root / "table_plane.json", plane)
    write_json(root / "source_object.json", info)
    write_json(root / "source_capture.json", capture)
    metadata = dict(
        frame_id="world",
        unit="m",
        rotation_convention="active column vectors: v_new=R_relative@v; row points use @R_relative.T",
        preview_transform="(points_world_m-observed_centroid)@R_relative.T+observed_centroid",
        preview_translation="centroid retained for orientation review; contact height undetermined",
        target_support_normal=(-np.asarray(plane["normal_xyz"])).tolist(),
        calibration=capture["extrinsics"],
        T_world_color=np.asarray(arrays["T_world_color"]).tolist(),
        source_object=str(object_root),
        source_object_sha256=info["object_sha256"],
        source_capture_sha256=info["snapshot_sha256"],
        table_plane_sha256=info["plane_sha256"],
        snapshot_sha256=sha256(root / "snapshot.npz"),
    )
    session = PlacementSession(root, points, info["object_sha256"], metadata)
    session.save_candidates()
    return session


def run_uop(session: PlacementSession, uop_root: Path, python: Path, seed: int) -> list[dict]:
    """Invoke only UOP inference on the saved cloud; retain its log and exact result."""
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = str(project_root() / "src")
    environment.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    command = [
        str(python),
        "-m",
        "ada_grasp_ctrl.placement.uop_adapter",
        "--uop-root",
        str(uop_root),
        "--snapshot",
        str(session.root / "snapshot.npz"),
        "--output",
        str(session.root),
        "--seed",
        str(seed),
    ]
    write_json(session.root / "uop_invocation.json", dict(command=command, seed=seed))
    with (session.root / "uop.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    result = json.loads((session.root / "uop_result.json").read_text())
    if result["snapshot_sha256"] != session.metadata["snapshot_sha256"]:
        raise ValueError("UOP result references a different snapshot")
    session.metadata["uop"] = {key: value for key, value in result.items() if key != "candidates"}
    return result["candidates"]


def run_placement(args: argparse.Namespace, config: dict, root: Path) -> int:
    """Branch before all grasp/model/robot setup; reuse the existing manual ROI/cluster selector."""
    from ada_grasp_ctrl.pipeline.runner import Processes, perceive
    from ada_grasp_ctrl.pipeline.dual_viewer import create_server
    from ada_grasp_ctrl.ros_execution.settings import validate_domain
    from .ui import serve

    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", 8089)) == 0:
            raise RuntimeError("Port 8089 is occupied; close the previous preview")
    uop_root = resolve_from_root(args.uop_root or os.getenv("UOP_NET_ROOT") or "../uop-net")
    python = resolve_from_root(args.uop_python) if args.uop_python else uop_root / ".venv/bin/python"
    config = dict(config)
    if args.environment:
        config["environment"] = args.environment
    config["ros_domain_id"] = validate_domain(
        config["environment"], args.ros_domain_id if args.ros_domain_id is not None else os.getenv("ROS_DOMAIN_ID")
    )
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=1,
        mode="placement_only",
        arguments=vars(args),
        config=config,
        uop_root=str(uop_root),
        uop_python=str(python),
    )
    (root / "run_manifest.yaml").write_text(yaml.safe_dump(manifest))
    processes = Processes(config, root)
    report = dict(schema_version=1, mode="placement_only", status="preparing", execution_started=False)
    try:
        if args.object_root:
            obj = resolve_from_root(args.object_root)
        else:
            perception = argparse.Namespace(mode="live", reuse_candidates=False, roi=args.roi, cluster=args.cluster)
            obj = perceive(dict(group="placement"), perception, config, processes, root, infer=False, build_asset=False)
        session = prepare_session(obj, root)
        report["status"] = "preview_ready"
        write_json(root / "run_report.json", report)
        serve(session, create_server, lambda: run_uop(session, uop_root, python, config.get("seed", 4)))
        report.update(
            status="confirmed" if session.confirmed_id else "closed_without_confirmation",
            confirmed_candidate_id=session.confirmed_id,
            inference_status=session.inference_status,
        )
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        processes.close()
        write_json(root / "run_report.json", report)


def prepare_simulation_session(object_root: Path, root: Path, state: dict) -> PlacementSession:
    """Bind placement to a genuine immutable simulation cloud without camera records."""
    from ada_grasp_ctrl.pipeline.observation import read_observation

    info, points, _ = read_observation(object_root)
    if info.get("source_type") != "simulation_complete_mesh":
        raise ValueError("Expected simulation provenance")
    plane_path = resolve_from_root(info["plane_path"])
    if sha256(plane_path) != info["plane_sha256"]:
        raise ValueError("Table changed after scene confirmation")
    plane = json.loads(plane_path.read_text())
    root.mkdir()
    np.savez_compressed(root / "snapshot.npz", object_points_world_m=points, table_normal_world=plane["normal_xyz"])
    write_json(root / "source_object.json", info)
    write_json(root / "robot_state.json", state)
    metadata = dict(
        frame_id="world",
        unit="m",
        source_type=info["source_type"],
        simulation_source=info["simulation_source"],
        source_object_sha256=info["object_sha256"],
        table_plane_sha256=info["plane_sha256"],
        snapshot_sha256=sha256(root / "snapshot.npz"),
        rotation_convention="active column vectors: v_new=R_relative@v; row points use @R_relative.T",
        preview_transform="(points_world_m-observed_centroid)@R_relative.T+observed_centroid",
        preview_translation="centroid retained for orientation review; contact height undetermined",
        target_support_normal=(-np.asarray(plane["normal_xyz"])).tolist(),
    )
    session = PlacementSession(root, points, info["object_sha256"], metadata)
    session.save_candidates()
    return session
