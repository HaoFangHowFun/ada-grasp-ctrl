"""Derive, validate and render standalone Sharpa mirroring; no ROS or arm commands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ada_grasp_ctrl.experiments.sharpa_mirroring import HandGeometry, derive_mapping, diagnose, render_case
from ada_grasp_ctrl.paths import resolve_external_root, resolve_from_root
from ada_grasp_ctrl.sharpa_mirror import SharpaHandMirror


def run(args: argparse.Namespace) -> None:
    """Write a fresh report, reusable mapping and selected screenshots."""
    root = resolve_from_root(args.models)
    output = resolve_external_root("output", args.output_root)
    if output.exists():
        raise ValueError(f"Output must be a new directory: {output}")
    right = HandGeometry(root / "right_sharpa_wave_with_flange.urdf", "right")
    left = HandGeometry(root / "left_sharpa_wave_with_flange.urdf", "left")
    config, derivation = derive_mapping(right, left)
    output.mkdir(parents=True)
    config_path = output / "mapping.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    mirror = SharpaHandMirror(config_path)
    rng = np.random.default_rng(args.seed)
    cases = [("canonical", np.zeros(22))]
    for i, row in enumerate(derivation["joints"]):
        for d in row["deltas_rad"]:
            q = np.zeros(22)
            q[i] = d
            cases.append((f"joint_{i:02d}_{d:+.2f}", q))
    for i in range(22):
        for end in range(2):
            q = np.zeros(22)
            q[i] = right.limits[i, end]
            cases.append((f"boundary_{i:02d}_{end}", q))
    # Probe configurations fit the mapping; these random configurations are never used in fitting.
    for i in range(args.samples):
        cases.append((f"heldout_{i:04d}", rng.uniform(right.limits[:, 0], right.limits[:, 1])))
    candidate_sources = {}
    if args.candidates:
        for p in sorted(resolve_from_root(args.candidates).glob("*.npy"))[: args.candidate_count]:
            record = np.load(p, allow_pickle=True).item()  # Explicit trusted-local prepared records only.
            names = record["joint_names"]
            indices = [names.index(name) for name in right.names]
            candidate_sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            for field in ("pregrasp_qpos", "grasp_qpos", "squeeze_qpos"):
                value = np.asarray(record[field], dtype=float)
                if value.shape != (len(names),) or not np.isfinite(value).all():
                    raise ValueError(f"{p}: malformed {field}")
                cases.append((f"dexlearn_{p.stem}_{field}", value[indices]))
        if not candidate_sources:
            raise ValueError("No prepared DexLearn candidates found")
    results = [diagnose(right, left, mirror, q, name) for name, q in cases]
    heldout = [r for r in results if r["case_id"].startswith("heldout_")]
    worst = sorted(heldout, key=lambda r: r["max_landmark_error_mm"], reverse=True)[:5]
    # Exercise arbitrary world orientations/planes independently of finger mapping.
    max_world_error, max_rotation_error = 0.0, 0.0
    for _ in range(32):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        pose[:3, 3] = rng.normal(size=3)
        normal, center = rng.normal(size=3), rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        reflection = np.eye(3) - 2 * np.outer(normal, normal)
        target = mirror.mirror_wrist_pose(pose, normal, center)
        q = rng.uniform(right.limits[:, 0], right.limits[:, 1])
        reference = center + (right.fk(q) @ pose[:3, :3].T + pose[:3, 3] - center) @ reflection.T
        actual = left.fk(mirror.right_to_left(q)) @ target[:3, :3].T + target[:3, 3]
        max_world_error = max(max_world_error, float(np.linalg.norm(reference - actual, axis=1).max() * 1000))
        recovered = mirror.mirror_wrist_pose(target, normal, center)
        max_rotation_error = max(
            max_rotation_error, float(Rotation.from_matrix(pose[:3, :3].T @ recovered[:3, :3]).magnitude())
        )
    for case in worst:
        case["likely_discrepancy_cause"] = (
            "Small kinematic residual is consistent with rounded URDF angles; "
            "coarse independent surface sampling and source mesh asymmetry affect surface distances."
        )
    screenshots = []
    if not args.no_render:
        images = output / "images"
        images.mkdir()
        for mode in ("right", "reference", "left", "overlay"):
            path = images / f"canonical_{mode}.png"
            render_case(right, left, mirror, results[0], path, mode)
            screenshots.append(str(path.relative_to(output)))
        selected = worst + heldout[:2]
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            q = np.zeros(22)
            indices = [i for i, n in enumerate(right.names) if f"_{finger}_" in n]
            q[indices] = right.limits[indices].mean(axis=1)
            selected.append(diagnose(right, left, mirror, q, f"finger_{finger}"))
        selected += [r for r in results if r["case_id"].startswith("dexlearn_")][:3]
        for case in selected:
            path = images / (case["case_id"] + "_overlay.png")
            render_case(right, left, mirror, case, path, "overlay")
            case["screenshot"] = str(path.relative_to(output))
            screenshots.append(case["screenshot"])
    summary = {
        "cases": len(results),
        "heldout_cases": len(heldout),
        "max_landmark_error_mm": max(r["max_landmark_error_mm"] for r in results),
        "heldout_max_landmark_error_mm": max(r["max_landmark_error_mm"] for r in heldout),
        "heldout_mean_surface_error_mm": float(np.mean([r["mean_surface_error_mm"] for r in heldout])),
        "heldout_max_surface_error_mm": max(r["max_surface_error_mm"] for r in heldout),
        "invalid_cases": [r["case_id"] for r in results if not r["valid"]],
        "world_pose_max_landmark_error_mm": max_world_error,
        "pose_roundtrip_max_angle_deg": float(np.rad2deg(max_rotation_error)),
    }
    report = {
        "schema_version": 1,
        "summary": summary,
        "derivation": derivation,
        "cases": results,
        "worst_five": worst,
        "screenshots": screenshots,
        "limitations": [
            "Surface metrics have independent point-sampling error and include shared source mesh files.",
            "No physical calibration, collision or grasp-success validation.",
            "Joint and frame validity do not imply object or scene mirroring is integrated.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "diagnostics.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "valid", "max_landmark_error_mm", "mean_surface_error_mm", "max_surface_error_mm"]
        )
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in results)
    manifest = {
        "arguments": vars(args),
        "candidate_sources": candidate_sources,
        "models": config["assets"],
        "numpy": np.__version__,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                resolve_from_root("src/ada_grasp_ctrl/sharpa_mirror.py"),
                resolve_from_root("src/ada_grasp_ctrl/experiments/sharpa_mirroring.py"),
                resolve_from_root("script/validate_sharpa_mirroring.py"),
            ]
        },
        "command": sys.argv,
    }
    (output / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (output / "report.md").write_text(
        "# Sharpa mirror validation\n\n```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n\n"
        + "Surface metrics are nearest-sample estimates. Inspect report.json for joint evidence and provenance.\n"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Run the standalone CLI with explicit asset and output roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Directory with the two original with-flange URDFs")
    parser.add_argument("--output-root", required=True, help="Fresh output directory")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", help="Trusted local prepared DexLearn NPY directory with explicit joint_names")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.samples < 5 or args.candidate_count < 1:
        parser.error("Need at least five held-out samples and a positive candidate count")
    run(args)


if __name__ == "__main__":
    main()
