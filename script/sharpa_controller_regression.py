"""Prepare existing Sharpa 100K predictions and run paired controller checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from ada_grasp_ctrl.errors import AdaGraspError
from ada_grasp_ctrl.experiments.sharpa_regression import load_suite, prepare_suite, run_suite
from ada_grasp_ctrl.paths import source_checkout_root


def main() -> int:
    """Dispatch a private CPU evaluation; never retrain or silently resample."""
    repo = source_checkout_root()
    dataset = Path(os.environ.get("HUGS_DATASET_ROOT", str(Path.home() / "hugs-dataset")))
    base = dataset / "BimanBODex_sharpa_1264k_v1"
    train = base / "dexlearn_training/sharpaMulti_robotMultiHierar_sharpa_right_full_1m_100k_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare or verify inputs without simulation")
    parser.add_argument("--smoke", action="store_true", help="Four scales x one sample x two phases; eight episodes")
    parser.add_argument("--workers", type=int, default=1, help="CPU workers per phase; default 1")
    parser.add_argument("--suite-root", type=Path, default=os.environ.get("SHARPA_SUITE_ROOT"))
    parser.add_argument("--output-root", type=Path, help="New output directory; default is a unique timestamp")
    parser.add_argument("--baseline", type=Path, help="Compatible prior run directory or regression.json")
    args = parser.parse_args()
    try:
        if args.workers < 1:
            raise ValueError("workers must be positive")
        suite = (
            args.suite_root
            or repo
            / "output/regression"
            / ("sharpa-right-100k-tabletop-smoke-v1" if args.smoke else "sharpa-right-100k-tabletop-100x10-v1")
        ).resolve()
        path = suite / "suite_manifest.json"
        if not suite.exists():

            def configured(name: str, default: Path) -> Path:
                return Path(os.environ.get(name, str(default))).expanduser().resolve(strict=True)

            prepare_suite(
                suite,
                configured(
                    "SHARPA_PREDICTIONS_ROOT", train / "tests/step_100000/sharpaMulti_scale_matched/3_right_full"
                ),
                configured("SHARPA_METADATA", base / "sharpa_right/right_full/metadata.json"),
                configured("SHARPA_CHECKPOINT", train / "ckpts/step_100000.pth"),
                configured(
                    "SHARPA_SCENE_MANIFEST",
                    dataset / "object/DGN_2k/valid_split_sharpa_90_10_v1/test_scenes_sharpa_four_scales.json",
                ),
                dataset.resolve(strict=True),
                configured("SHARPA_SOURCE_XML", repo.parent / "HUGS-DexGraspBench/assets/hand/sharpa_wave/right.xml"),
                configured(
                    "SHARPA_SOURCE_URDF", repo.parent / "HUGS-Sharpa-Assets/urdf/right_sharpa_wave_with_flange.urdf"
                ),
                args.smoke,
            )
        if args.prepare_only:
            frozen = load_suite(path)
            print(f"Verified frozen suite: {len(frozen['samples'])} grasps at {path}")
            return 0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = (args.output_root or repo / f"output/regression/sharpa-runs/{stamp}-{os.getpid()}").resolve()
        baseline = args.baseline.resolve() if args.baseline else None
        if baseline and baseline.is_dir():
            baseline /= "regression.json"
        return run_suite(path, output, args.smoke, args.workers, baseline)
    except AdaGraspError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code
    except (OSError, ValueError, KeyError) as error:
        print(f"Sharpa preflight/input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
