"""Original/right and reflected/left inference from each newly captured object."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from ada_grasp_ctrl.experiments.mirrored_left_grasp import reflection_plane, reflect_points
from ada_grasp_ctrl.grasp_selection.multi_arm import from_right_record, enabled_set
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json, verify_reference
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.paths import resolve_from_root


def infer_live_sides(object_root: Path, config: dict, processes: object, output: Path) -> list[dict]:
    """Run the established checkpoint for each enabled side, preserving capture provenance."""
    reference = config.get("inference_model")
    if reference is None:
        reference = read_json(resolve_from_root(config["inference_reference"]))
    for key in ("checkpoint", "training_config", "metadata"):
        verify_reference(reference[key])
    info = read_json(object_root / "object.json")
    with np.load(object_root / "object.npz", allow_pickle=False) as archive:
        original = {k: archive[k].copy() for k in archive.files}
    if file_hash(object_root / "object.npz") != info["object_sha256"]:
        raise ValueError("Captured object changed")
    plane = reflection_plane(original["points_world_m"], config.get("mirroring", {}))
    rows, rejected = [], []
    for side in sorted(enabled_set(config["enabled_arms"])):
        branch = output / ("inference_" + side)
        branch.mkdir()
        source = object_root
        if side == "left":
            source = branch / "reflected_object"
            source.mkdir()
            data = deepcopy(original)
            data["points_world_m"] = reflect_points(original["points_world_m"], plane)
            world_color = original["T_world_color"]
            data["points_color_m"] = (data["points_world_m"] - world_color[:3, 3]) @ world_color[:3, :3]
            np.savez_compressed(source / "object.npz", **data)
            derived = deepcopy(info)
            derived.update(
                object_sha256=file_hash(source / "object.npz"),
                derivation=dict(
                    kind="world_reflection",
                    plane=plane,
                    source_object=str(object_root),
                    source_sha256=info["object_sha256"],
                ),
            )
            write_json(source / "object.json", derived)
        destination = branch / "predictions"
        command = [
            config["dexlearn_python"],
            "-m",
            "ada_grasp_ctrl.experiments.fixed_camera_grasp",
            "--object",
            str(source),
            "--output-root",
            str(destination),
            "--allow-unaccepted",
            "--samples",
            str(config["samples"]),
            "--seed",
            str(config["seed"]),
            "--device",
            config["device"],
            "--dexlearn-root",
            config["dexlearn_root"],
        ]
        for key in ("checkpoint", "training_config", "metadata"):
            command += ["--" + key.replace("_", "-"), reference[key]["path"]]
        processes.run(command, "inference_" + side)
        payload = read_json(destination / "candidates.json")
        provenance = dict(
            inference="actual_checkpoint",
            checkpoint=payload["checkpoint"],
            metadata=payload["metadata"],
            calibration=info["extrinsics"],
            input_point_source="live_measured_object",
            source_object=str(object_root),
            source_object_sha256=info["object_sha256"],
            predictions=str(destination / "candidates.json"),
            predictions_sha256=file_hash(destination / "candidates.json"),
            preprocessing=payload["preprocessing"],
        )
        for candidate in payload["candidates"]:
            try:
                rows.append(from_right_record(candidate, provenance, side=side, plane=plane))
            except (ValueError, KeyError, TypeError) as error:
                rejected.append(dict(side=side, candidate_id=candidate.get("candidate_id"), error=str(error)))
    write_json(output / "candidates.json", dict(schema_version=1, candidates=rows, conversion_rejections=rejected))
    return rows
