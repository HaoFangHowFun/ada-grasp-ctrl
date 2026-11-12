"""Thin original/reflected-cloud integration with the unchanged right-full inference CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np

from ada_grasp_ctrl.paths import resolve_from_root, project_root, runtime_root
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json
from ada_grasp_ctrl.grasp_selection.multi_arm import enabled_set, from_right_record
from .simulated_object_cloud import vector


def reflection_plane(points: np.ndarray, config: dict) -> dict:
    """Fix one world plane before inference resampling and record its unit normal."""
    n = vector(config.get("plane_normal_world", [0, 1, 0]), 3, "reflection normal")
    if np.linalg.norm(n) < 1e-12:
        raise ValueError("Reflection normal must be nonzero")
    n /= np.linalg.norm(n)
    raw = config.get("plane_point", "object_point_cloud_centroid")
    c = (
        points.mean(axis=0)
        if isinstance(raw, str) and raw == "object_point_cloud_centroid"
        else vector(raw, 3, "plane point")
    )
    return dict(
        normal_world=n.tolist(),
        point_world_m=c.tolist(),
        convention="world diagnostic reflection; no hardware calibration",
    )


def reflect_points(points: np.ndarray, plane: dict) -> np.ndarray:
    """Reflect metric world points once around the recorded plane."""
    n, c = np.asarray(plane["normal_world"]), np.asarray(plane["point_world_m"])
    return c + (points - c) @ (np.eye(3) - 2 * np.outer(n, n))


def infer_sides(points: np.ndarray, config: dict, output: Path, scene_identity: str) -> tuple[list[dict], list[dict]]:
    """Run the same actual checkpoint only for enabled branches and retain raw artifacts.

    Existing inference owns checkpoint validation, preprocessing and stage decoding.
    A separate configured Python preserves its installed GPU dependencies without migration.
    """
    inference = dict(config["inference"])
    if "reference" in inference:
        reference = read_json(resolve_from_root(inference["reference"]))
        for key in ("checkpoint", "training_config", "metadata"):
            inference.setdefault(key, reference[key]["path"])
            if file_hash(resolve_from_root(inference[key])) != reference[key]["sha256"]:
                raise ValueError(f"Inference reference {key} checksum differs")
    plane = reflection_plane(points, config.get("mirroring", {}))
    rows, rejected = [], []
    for side in sorted(enabled_set(config["selector"]["enabled_arms"])):
        branch = output / f"inference_{side}"
        branch.mkdir()
        obj = branch / "object"
        obj.mkdir()
        cloud = reflect_points(points, plane) if side == "left" else points
        np.savez_compressed(obj / "object.npz", points_world_m=cloud, points_color_m=cloud, T_world_color=np.eye(4))
        info = dict(
            schema_version=1,
            frame_id="world",
            unit="m",
            table_status="simulation_diagnostic",
            object_sha256=file_hash(obj / "object.npz"),
            extrinsics=dict(kind="identity world adapter; no physical camera calibration", sha256=scene_identity),
        )
        (obj / "object.json").write_text(json.dumps(info, indent=2))
        destination = branch / "predictions"
        # Resolving a venv executable symlink escapes its pyvenv.cfg and loses installed packages.
        python = Path(inference["python"]).expanduser()
        if not python.is_absolute():
            python = runtime_root("data") / python
        cmd = [
            str(python),
            "-m",
            "ada_grasp_ctrl.experiments.fixed_camera_grasp",
            "--object",
            str(obj),
            "--output-root",
            str(destination),
            "--allow-unaccepted",
            "--samples",
            str(inference["samples_per_side"]),
            "--seed",
            str(config["point_cloud"]["seed"]),
            "--device",
            inference.get("device", "cuda:0"),
        ]
        for key in ("checkpoint", "training_config", "metadata", "dexlearn_root"):
            cmd += ["--" + key.replace("_", "-"), str(resolve_from_root(inference[key]))]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root() / "src")
        env.pop("PYTHONHOME", None)
        (branch / "command.json").write_text(json.dumps(cmd, indent=2))
        with (branch / "inference.log").open("w") as log:
            completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"{side} actual inference failed (exit {completed.returncode}); see {branch}/inference.log"
            )
        payload = read_json(destination / "candidates.json")
        provenance = dict(
            scene_identity=scene_identity,
            checkpoint=payload["checkpoint"],
            metadata=payload["metadata"],
            training_config=payload["training_config"],
            dexlearn_commit=payload["dexlearn_commit"],
            preprocessing=payload["preprocessing"],
            input_point_source=config["point_cloud"]["source"],
            inference="actual_checkpoint",
            pregrasp_extrapolation=1,
            object_id=config["object"]["name"],
        )
        for raw in payload["candidates"]:
            try:
                rows.append(from_right_record(raw, provenance, side=side, plane=plane))
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append(
                    dict(
                        candidate_id=f"{side}:{raw.get('candidate_id')}",
                        reason="candidate_conversion_failed",
                        error=str(exc),
                        source_record=raw,
                    )
                )
    return rows, rejected
