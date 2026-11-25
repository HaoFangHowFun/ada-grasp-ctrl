"""One-load Native Sharpa Auto-Mode inference for the existing dual live pipeline."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.sharpa_mirror import SharpaHandMirror
from ada_grasp_ctrl.pipeline.grasp_models import SOURCE_MODES, SINGLE_MODES
from .fixed_camera_grasp import decode_candidates, preprocess, STAGES
from scipy.spatial.transform import Rotation


def decode_bimanual(raw: np.ndarray, joint_names: list[str], preprocessing: dict) -> dict:
    """Decode one native paired sample; dummy-arm metadata entries are never predictions."""
    values = np.asarray(raw, dtype=float)
    mirror = SharpaHandMirror()
    axes = ("TransX", "TransY", "TransZ", "RotX", "RotY", "RotZ")
    expected = [f"ra_{a}J" for a in axes] + list(mirror.right_joint_names)
    expected += [f"la_{a}J" for a in axes] + list(mirror.left_joint_names)
    if joint_names != expected or values.shape != (3, 58) or not np.isfinite(values).all():
        raise ValueError("Expected native (3,58) paired sample and ordered 56-joint metadata")
    center = np.asarray(preprocessing["center_world_m"], dtype=float)
    if (
        preprocessing["frame_id"] != "world"
        or preprocessing["scale"] != 1.0
        or center.shape != (3,)
        or not np.isfinite(center).all()
    ):
        raise ValueError("Expected centered metric world coordinates")
    targets = {}
    for side, offset, names in (("right", 0, joint_names[6:28]), ("left", 29, joint_names[34:56])):
        targets[side] = {}
        for stage, vector in zip(STAGES, values[:, offset : offset + 29]):
            quat = vector[3:7]
            if not np.isclose(np.linalg.norm(quat), 1, atol=1e-4, rtol=0):
                raise ValueError(f"{side}/{stage}: non-unit WXYZ quaternion")
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
            pose[:3, 3] = vector[:3] + center
            targets[side][stage] = dict(
                frame_id="world",
                pose_frame=f"{side}_hand_flange",
                T_world_hand=pose.tolist(),
                joint_positions_rad=dict(zip(names, vector[7:29].tolist())),
            )
    return targets


def metadata_contract(root: Path) -> dict:
    """Check native metadata explicitly, including combined dummy-arm joint names."""
    result = {}
    mirror = SharpaHandMirror()
    for mode in SOURCE_MODES:
        path = root / mode / "metadata.json"
        meta = read_json(path)
        dual = mode.startswith("both_")
        names = list(mirror.right_joint_names)
        if dual:
            axes = ("TransX", "TransY", "TransZ", "RotX", "RotY", "RotZ")
            names = [f"ra_{a}J" for a in axes] + names
            names += [f"la_{a}J" for a in axes] + list(mirror.left_joint_names)
        if (
            meta.get("hand_family") != "sharpa_dual"
            or meta.get("grasp_type") != mode
            or meta.get("joint_names") != names
            or meta.get("wrist_body_names") != ["right_hand_flange", "left_hand_flange"][: 2 if dual else 1]
        ):
            raise ValueError(f"{path}: unsupported native Sharpa metadata contract")
        result[mode] = dict(path=str(path), sha256=file_hash(path), **meta)
    return result


def infer(object_root: Path, config: dict, output: Path) -> dict:
    """Run real Auto-Mode and enabled mirrored modes in one DexLearn process.

    Returns all raw arrays, model diagnostics and only per-type Top-K executable rows.
    No robot state, ROS connection, camera capture or hardware action occurs here.
    """
    began = time.monotonic()
    deployment = config["grasp_models"]["native_multimode"]
    checkpoint, training = (resolve_from_root(deployment[k]) for k in ("checkpoint", "training_config"))
    metadata = metadata_contract(resolve_from_root(deployment["metadata_root"]))
    info = read_json(object_root / "object.json")
    if (info.get("schema_version"), info.get("frame_id"), info.get("unit")) != (1, "world", "m"):
        raise ValueError("Unsupported object schema/frame/units")
    if file_hash(object_root / "object.npz") != info["object_sha256"]:
        raise ValueError("Object snapshot hash mismatch")
    from ada_grasp_ctrl.pipeline.observation import read_observation, source_provenance

    info, points, camera_arrays = read_observation(object_root)
    world_color = camera_arrays.get("T_world_color")
    plane = config["native_reflection_plane"]
    arms = set(config["enabled_arms"])
    if not arms or not arms <= {"left", "right"}:
        raise ValueError("Invalid enabled arms")
    dexroot = resolve_from_root(config["dexlearn_root"])
    sys.path.insert(0, str(dexroot))
    import torch
    from omegaconf import OmegaConf

    model_module = importlib.import_module("dexlearn.network.models.robot_hierarchical")
    dataset = importlib.import_module("dexlearn.dataset")
    sampling = importlib.import_module("dexlearn.task.sample")
    types_module = importlib.import_module("dexlearn.dataset.grasp_types")
    if dexroot not in Path(model_module.__file__).resolve().parents:
        raise ValueError("DexLearn imported from unexpected checkout")
    if types_module.GRASP_TYPES != ["0_any"] + [f"{i}_{m}" for i, m in enumerate(SOURCE_MODES, 1)]:
        raise ValueError("Native model mode IDs differ")
    cfg = OmegaConf.load(training)
    if (
        cfg.algo.model.name != "RobotHierarchicalModel"
        or cfg.algo.model.head.name != "DiffusionBiRT_MLPRTJ"
        or cfg.data.model.joint_num != 44
        or cfg.data.model.traj_length != 3
        or not cfg.data.augmentation.pc_centering
        or cfg.data.point_cloud.source != "complete"
        or cfg.data.robot.metadata_group != "both_full"
        or cfg.algo.model.type_objective != "availability"
        or dict(cfg.algo.model.type_availability)
        != dict(score_threshold=0.9, min_available_types=1, use_score_prior=True)
        or not cfg.algo.sample_selection.enabled
        or cfg.algo.sample_selection.scope != "per_type"
        or cfg.algo.sample_selection.mode != "topk"
        or cfg.algo.test_grasp_num != 100
        or cfg.algo.test_topk != 10
    ):
        raise ValueError("Unsupported native model/training/sampling contract")
    # Rotation augmentation is train-only in RobotMultiDexDataset.__getitem__.
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.algo.model, resolve=True))
    OmegaConf.save(model_cfg, output / "resolved_model_config.yaml")
    seed, device = int(config["seed"]), config["device"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model_module.RobotHierarchicalModel(model_cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    refs = dict(
        checkpoint=dict(path=str(checkpoint), sha256=file_hash(checkpoint), iteration=int(state["iter"])),
        training_config=dict(path=str(training), sha256=file_hash(training)),
        dexlearn_commit=subprocess.check_output(["git", "-C", str(dexroot), "rev-parse", "HEAD"], text=True).strip(),
        source_object=str(object_root),
        source_object_sha256=info["object_sha256"],
        **source_provenance(info),
    )
    timings = dict(load_and_setup_s=time.monotonic() - began)
    rows, rejected, branches = [], [], {}

    def batch_for(cloud: np.ndarray) -> tuple[dict, dict, np.ndarray]:
        normalized, prep = preprocess(cloud, int(cfg.data.point_cloud.num_points), seed)
        prep.update(
            training_point_source="complete",
            input_point_source=info.get("source_type", "single-view partial measured cloud"),
        )
        batch = dataset.get_sparse_tensor(
            torch.as_tensor(normalized[None], device=device), float(model_cfg.backbone.voxel_size)
        )
        return batch, prep, normalized

    def sample_branch(label: str, batch: dict, prep: dict, normalized: np.ndarray, request: int) -> tuple:
        start = time.monotonic()
        batch["grasp_type_id"] = torch.tensor([request], dtype=torch.long, device=device)
        with torch.no_grad():
            raw, ids, scores, logp = model.sample(batch, int(cfg.algo.test_grasp_num))
            keep = sampling._sample_save_mask(raw, logp, ids, cfg)
            availability = scores[0, 0]
            available = model._availability_mask(availability[None], batch["grasp_type_id"])[0]
        raw, ids, scores, logp, keep, available = [
            x.detach().cpu().numpy() for x in (raw[0], ids[0], scores[0], logp[0], keep[0], available)
        ]
        if raw.shape != (500, 3, 58) or scores.shape != (500, 5) or not np.isfinite(scores).all():
            raise ValueError(f"{label}: incompatible native sample output")
        path = output / (label + ".npz")
        np.savez_compressed(
            path,
            poses=raw,
            type_ids=ids,
            availability_scores=scores,
            log_probabilities=logp,
            topk_mask=keep,
            available_mask=available,
            normalized_points_m=normalized,
            center_world_m=prep["center_world_m"],
            **camera_arrays,
        )
        branches[label] = dict(
            raw_predictions=str(path),
            sha256=file_hash(path),
            requested_type_id=request,
            preprocessing=prep,
            topk_raw_indices=np.flatnonzero(keep).tolist(),
        )
        timings[label + "_s"] = time.monotonic() - start
        return raw, ids, scores[0], logp, keep, available

    original, prep, normalized = batch_for(points)
    if "right" in arms:
        raw, ids, scores, logp, keep, available = sample_branch("original_auto", original, prep, normalized, 0)
    else:
        # The left-only path still ranks the original scene but generates no right candidates.
        start = time.monotonic()
        with torch.no_grad():
            feature, _ = model.backbone(original)
            score_tensor = model._availability_scores(model.type_classifier(feature))
            mask = model._availability_mask(score_tensor, torch.zeros(1, dtype=torch.long, device=device))
        scores, available = score_tensor[0].cpu().numpy(), mask[0].cpu().numpy()
        np.savez_compressed(
            output / "original_scores.npz",
            availability_scores=scores,
            available_mask=available,
            normalized_points_m=normalized,
        )
        timings["original_scores_s"] = time.monotonic() - start
    score_map = dict(zip(SOURCE_MODES, map(float, scores)))
    ranked = sorted([m for m, yes in zip(SOURCE_MODES, available) if yes], key=lambda m: (-score_map[m], m))

    def convert(
        raw: np.ndarray, ids: np.ndarray, logp: np.ndarray, keep: np.ndarray, side: str, prep: dict, label: str
    ) -> None:
        for index in np.flatnonzero(keep):
            mode = SOURCE_MODES[int(ids[index]) - 1]
            if mode not in ranked or (mode not in SINGLE_MODES and (side != "right" or len(arms) != 2)):
                continue
            provenance = dict(
                **refs,
                metadata=metadata[mode],
                model_score=score_map[mode],
                preprocessing=prep,
                inference="actual_checkpoint",
                grasp_model="native_multimode",
                raw_predictions=branches[label]["raw_predictions"],
                raw_predictions_sha256=branches[label]["sha256"],
            )
            try:
                decoded = (
                    dict(
                        targets=decode_bimanual(raw[index], metadata[mode]["joint_names"], prep),
                        log_probability=float(logp[index]),
                    )
                    if mode not in SINGLE_MODES
                    else decode_candidates(
                        raw[index : index + 1],
                        logp[index : index + 1],
                        metadata[mode]["joint_names"],
                        prep,
                        world_color,
                    )[0]
                )
                decoded.update(
                    candidate_id=f"{mode}:c{index:04d}",
                    raw_candidate_index=int(index),
                    source_model_mode=mode,
                    side=side,
                    provenance=provenance,
                )
                rows.append(decoded)
            except (ValueError, KeyError, TypeError) as error:
                rejected.append(dict(side=side, mode=mode, raw_index=int(index), error=str(error)))

    if "right" in arms:
        convert(raw, ids, logp, keep, "right", prep, "original_auto")
    if config.get("grasp_mode", "auto") == "auto" and "left" in arms and any(m in SINGLE_MODES for m in ranked):
        mirrored_points = np.load(config["native_mirrored_cloud"], allow_pickle=False)
        mirrored, mirror_prep, mirror_normalized = batch_for(mirrored_points)
        for mode in ranked:
            if mode not in SINGLE_MODES:
                continue
            label = "mirrored_" + mode
            raw, ids, _, logp, keep, _ = sample_branch(
                label, mirrored, mirror_prep, mirror_normalized, SOURCE_MODES.index(mode) + 1
            )
            convert(raw, ids, logp, keep, "left", mirror_prep, label)
    timings["total_s"] = time.monotonic() - began
    result = dict(
        schema_version=1,
        grasp_model="native_multimode",
        **refs,
        availability_scores=score_map,
        ranked_available_modes=ranked,
        eligible_modes=[
            m
            for m in ranked
            if (m in SINGLE_MODES or len(arms) == 2)
            and (config.get("grasp_mode", "auto") == "auto" or m.startswith("both_"))
        ],
        unsupported_modes={},
        score_semantics="independent sigmoid availability, not calibrated success probabilities",
        right_records=rows,
        conversion_rejections=rejected,
        branches=branches,
        reflection_plane=plane,
        timings=timings,
        samples_per_type=100,
        topk_per_type=10,
        pregrasp_extrapolation=1,
    )
    write_json(output / "candidates.json", result)
    return result


def main() -> int:
    """Offline-capable worker entry point; the parent owns the live capture lifecycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("object", "config", "output-root"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = resolve_from_root(args.output_root, "output")
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_manifest.yaml").write_text(yaml.safe_dump(vars(args)))
    try:
        result = infer(resolve_from_root(args.object), yaml.safe_load(Path(args.config).read_text()), output)
        write_json(
            output / "run_report.json",
            dict(status="completed", timings=result["timings"], candidate_count=len(result["right_records"])),
        )
        return 0
    except Exception as error:
        write_json(output / "run_report.json", dict(status="failed", error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
