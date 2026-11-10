"""Single measured-object inference using the existing right-full DexLearn checkpoint."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ada_grasp_ctrl.paths import resolve_external_root, resolve_from_root
from .fixed_camera import read_json, write_json
from .fixed_camera_geometry import file_hash, rigid, transform

STAGES = ("pregrasp", "grasp", "squeeze")


def preprocess(points: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, dict]:
    """Sample with replacement and center in world axes, preserving physical metric scale."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Input object points must be finite Nx3 meters")
    if count < 3:
        raise ValueError("Point count must be >= 3")
    # Same sampling policy as RobotMultiDexDataset._sample_point_cloud.
    indices = np.random.RandomState(seed).choice(len(points), count, replace=True)
    sampled = points[indices]
    center = sampled.mean(axis=0, keepdims=True)
    normalized = sampled - center
    error = float(np.max(np.abs(normalized + center - sampled)))
    return normalized, dict(
        frame_id="world",
        unit="m",
        center_world_m=center[0].tolist(),
        scale=1.0,
        axes="world XYZ, no rotation",
        sampling="numpy RandomState.choice with replacement",
        indices=indices.tolist(),
        seed=seed,
        count=count,
        roundtrip_max_m=error,
    )


def decode_candidates(
    raw: np.ndarray, scores: np.ndarray, joint_names: list[str], preprocessing: dict, world_color: np.ndarray
) -> list[dict]:
    """Decode all three predicted world-flange stages without pregrasp extrapolation.

    Only positions receive the inverse centering/scale operation. Rotations and the
    22 named finger angles are validated and preserved; no world transform is applied
    a second time because input axes already are world axes.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 3 or raw.shape[1:] != (3, 58) or not np.isfinite(raw).all():
        raise ValueError("Expected finite right-full predictions with shape (N,3,58)")
    if len(joint_names) != 22 or len(set(joint_names)) != 22 or any(not n.startswith("right_") for n in joint_names):
        raise ValueError("Expected 22 unique named right-hand joints")
    if len(scores) != len(raw) or not np.isfinite(scores).all():
        raise ValueError("Invalid candidate likelihoods")
    if preprocessing["frame_id"] != "world" or preprocessing["scale"] != 1.0:
        raise ValueError("This checkpoint adapter supports centered metric world points only")
    center = np.asarray(preprocessing["center_world_m"])
    color_world = np.linalg.inv(rigid(world_color))
    candidates = []
    for index, candidate in enumerate(raw):
        stages = {}
        for name, values in zip(STAGES, candidate):
            quat = values[3:7]
            if not np.isclose(np.linalg.norm(quat), 1, atol=1e-4, rtol=0):
                raise ValueError(f"Candidate {index}/{name}: non-unit WXYZ quaternion")
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            matrix[:3, 3] = values[:3] + center
            camera_hand = color_world @ matrix
            roundtrip = float(np.max(np.abs(rigid(world_color) @ camera_hand - matrix)))
            stages[name] = dict(
                frame_id="world",
                pose_frame="right_hand_flange",
                position_m=matrix[:3, 3].tolist(),
                quaternion_wxyz=quat.tolist(),
                T_world_hand=matrix.tolist(),
                T_color_hand=camera_hand.tolist(),
                camera_world_roundtrip_max=roundtrip,
                joint_positions_rad=dict(zip(joint_names, values[7:29].tolist())),
            )
        candidates.append(dict(candidate_id=f"c{index:04d}", log_probability=float(scores[index]), stages=stages))
    return candidates


def infer(args: argparse.Namespace, output: Path) -> dict:
    """Validate checkpoint contracts, run its actual sample method and save replayable results."""
    object_root = resolve_from_root(args.object)
    info = read_json(object_root / "object.json")
    if info.get("schema_version") != 1 or info.get("frame_id") != "world" or info.get("unit") != "m":
        raise ValueError("Unsupported object schema/frame/units")
    if info["table_status"] != "accepted" and not args.allow_unaccepted:
        raise ValueError("Object table has not been accepted; --allow-unaccepted is diagnostic preview only")
    if file_hash(object_root / "object.npz") != info["object_sha256"]:
        raise ValueError("Object snapshot hash mismatch")
    with np.load(object_root / "object.npz", allow_pickle=False) as archive:
        points = archive["points_world_m"]
        color_points = archive["points_color_m"]
        world_color = rigid(archive["T_world_color"])
    if not np.allclose(transform(color_points, world_color), points, atol=1e-7, rtol=0):
        raise ValueError("Object camera/world round-trip failed")
    checkpoint = resolve_from_root(args.checkpoint)
    training_config = resolve_from_root(args.training_config)
    metadata_path = resolve_from_root(args.metadata)
    metadata = read_json(metadata_path)
    if metadata.get("hand_family") != "sharpa_right" or metadata.get("wrist_body_names") != ["right_hand_flange"]:
        raise ValueError("Checkpoint metadata must identify right Sharpa flange")
    if metadata.get("grasp_type") != "right_full":
        raise ValueError("Checkpoint metadata must identify right_full")
    root = resolve_from_root(args.dexlearn_root)
    sys.path.insert(0, str(root))
    import torch
    from omegaconf import OmegaConf

    config = OmegaConf.load(training_config)
    if config.algo.model.name != "RobotHierarchicalModel" or config.algo.model.head.name != "DiffusionBiRT_MLPRTJ":
        raise ValueError("Unsupported checkpoint architecture")
    if config.data.model.joint_num != 44 or config.data.model.traj_length != 3:
        raise ValueError("Expected right/left 22+22 joints and three stages")
    if not config.data.augmentation.pc_centering or config.data.augmentation.rotation:
        raise ValueError("Adapter requires centered, non-rotated training points")
    if config.data.robot.metadata_group != "right_full" or config.data.point_cloud.source != "complete":
        raise ValueError("Unexpected training data contract")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    model_config = OmegaConf.create(OmegaConf.to_container(config.algo.model, resolve=True))
    OmegaConf.save(model_config, output / "resolved_model_config.yaml")
    normalized, preprocessing = preprocess(points, int(config.data.point_cloud.num_points), args.seed)
    preprocessing["training_point_source"] = "complete"
    preprocessing["input_point_source"] = "single-view partial measured cloud; distribution shift unvalidated"
    np.savez_compressed(
        output / "preprocessing.npz",
        normalized_points_m=normalized,
        sampled_world_points_m=points[np.asarray(preprocessing["indices"])],
        T_world_color=world_color,
    )
    write_json(output / "preprocessing.json", preprocessing)
    model_module = importlib.import_module("dexlearn.network.models.robot_hierarchical")
    dataset_module = importlib.import_module("dexlearn.dataset")
    actual_module = Path(model_module.__file__).resolve()
    if root not in actual_module.parents:
        raise ValueError(f"DexLearn imported from unexpected checkout: {actual_module}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = model_module.RobotHierarchicalModel(model_config)
    # This is the user's explicitly selected trusted local training checkpoint.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    tensor = torch.as_tensor(normalized[None], device=args.device)
    batch = dataset_module.get_sparse_tensor(tensor, float(model_config.backbone.voxel_size))
    batch["grasp_type_id"] = torch.tensor([3], dtype=torch.long, device=args.device)
    with torch.no_grad():
        raw, types, probabilities, scores = model.sample(batch, args.samples)
    raw, types, probabilities, scores = [
        value.detach().cpu().numpy()[0] for value in (raw, types, probabilities, scores)
    ]
    np.savez_compressed(
        output / "raw_predictions.npz",
        poses=raw,
        type_ids=types,
        type_probabilities=probabilities,
        log_probabilities=scores,
    )
    selected = np.flatnonzero((types == 3) & np.isfinite(scores) & (scores > np.finfo(scores.dtype).min / 2))
    if not len(selected):
        raise ValueError("Checkpoint produced no valid right_full candidates")
    selected = selected[np.argsort(-scores[selected], kind="stable")]
    candidates = decode_candidates(raw[selected], scores[selected], metadata["joint_names"], preprocessing, world_color)
    for row, raw_index in zip(candidates, selected):
        row["raw_candidate_index"] = int(raw_index)
    result = dict(
        schema_version=1,
        frame_id="world",
        unit="m",
        joint_unit="rad",
        quaternion_order="wxyz",
        pose_frame="right_hand_flange",
        stages=list(STAGES),
        candidates=candidates,
        object_root=str(object_root),
        object_sha256=info["object_sha256"],
        calibration=info["extrinsics"],
        table_status=info["table_status"],
        checkpoint=dict(path=str(checkpoint), sha256=file_hash(checkpoint), iteration=int(state["iter"])),
        training_config=dict(path=str(training_config), sha256=file_hash(training_config)),
        metadata=dict(path=str(metadata_path), sha256=file_hash(metadata_path), joint_names=metadata["joint_names"]),
        dexlearn_commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        preprocessing=preprocessing,
        reachability="not assessed; no calibrated CRX mount provided",
        execution="visualization only; no ROS command publisher",
        pregrasp_extrapolation=1,
    )
    write_json(output / "candidates.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run one real checkpoint inference without synthetic scene configuration files."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("object", "checkpoint", "training-config", "metadata", "dexlearn-root", "output-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-unaccepted", action="store_true")
    args = parser.parse_args(argv)
    output = None
    try:
        destination = resolve_external_root("output", args.output_root)
        destination.mkdir(parents=True, exist_ok=False)
        output = destination
        (output / "run_manifest.yaml").write_text(yaml.safe_dump(vars(args)))
        result = infer(args, output)
        report = dict(
            status="completed",
            candidate_count=len(result["candidates"]),
            table_status=result["table_status"],
            checkpoint=result["checkpoint"],
        )
        write_json(output / "run_report.json", report)
        print(report)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        if output is not None:
            write_json(output / "run_report.json", dict(status="preflight_failed", error=str(exc)))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
