"""Prepare or run the fixed native-asset Shadow controller regression suite."""

import argparse
from pathlib import Path
import sys

from ada_grasp_ctrl.errors import AdaGraspError
from ada_grasp_ctrl.experiments.shadow_regression import prepare_suite, run_suite


def main() -> int:
    """Dispatch preparation and headless execution with explicit exit semantics."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Generate the immutable 1000-grasp suite once")
    for name in ("suite-root", "dexlearn-root", "checkpoint", "metadata"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    run = sub.add_parser("run", help="Run 1000 grasps in each of two phases without model inference")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--baseline", type=Path, help="Previous full run directory, or its regression.json")
    run.add_argument("--smoke", action="store_true", help="One frozen sample per scale; 8 executions total")
    run.add_argument("--workers", type=int, default=8, help="CPU workers per phase (1 runs serially)")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_suite(
                args.suite_root.resolve(),
                args.dexlearn_root.resolve(),
                args.checkpoint.resolve(),
                args.metadata.resolve(),
            )
            return 0
        baseline = args.baseline.resolve() if args.baseline else None
        if baseline and baseline.is_dir():
            baseline /= "regression.json"
        return run_suite(args.suite.resolve(), args.output_root.resolve(), args.smoke, baseline, workers=args.workers)
    except AdaGraspError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code
    except (OSError, ValueError, KeyError) as error:
        print(f"Preflight/input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
