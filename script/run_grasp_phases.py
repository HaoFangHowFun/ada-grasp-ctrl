"""Prepare and compare HUGS Shadow predictions using native op/ours policies."""

import argparse
import json
from pathlib import Path
import sys

from ada_grasp_ctrl.experiments.episode_recording import replay_episode
from ada_grasp_ctrl.experiments.phase_comparison import prepare_grasp_suite, run_comparison
from ada_grasp_ctrl.errors import AdaGraspError
from ada_grasp_ctrl.utils.viewer_session import DebugViewerError


def main() -> int:
    """Parse the explicit experiment protocol CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--robot", choices=["dummy_arm_shadow"], default="dummy_arm_shadow")
    prepare.add_argument("--source", choices=["hugs-shadow-right-full"], default="hugs-shadow-right-full")
    for name in ("input-root", "metadata", "data-root", "output-root"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--selection-manifest", type=Path)
    prepare.add_argument("--ik-device", choices=["cpu"], default="cpu")
    run = commands.add_parser("run")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--phase", choices=["both", "phase1", "phase2"], default="both")
    run.add_argument("--protocol", choices=["predicted-three-pose"], default="predicted-three-pose")
    run.add_argument("--workers", type=int, choices=[1], default=1)
    run.add_argument("--seed", type=int, default=12)
    run.add_argument("--sample-id")
    run.add_argument("--viewer", choices=["mujoco", "mjviser"])
    replay = commands.add_parser("replay")
    replay.add_argument("--run-root", type=Path, required=True)
    replay.add_argument("--phase", choices=["phase1", "phase2"], required=True)
    replay.add_argument("--sample-id", required=True)
    replay.add_argument("--viewer", choices=["mujoco", "mjviser"], default="mujoco")
    replay.add_argument("--speed", type=float, default=1)
    replay.add_argument("--no-hold", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            manifest = prepare_grasp_suite(
                args.input_root.resolve(),
                args.metadata.resolve(),
                args.data_root.resolve(),
                args.output_root.resolve(),
                args.selection_manifest,
            )
            print(manifest)
            return int(any(e["status"] != "prepared" for e in json.loads(manifest.read_text())["samples"]))
        if args.command == "run":
            return run_comparison(
                args.suite.resolve(), args.output_root.resolve(), args.phase, args.seed, args.sample_id, args.viewer
            )
        rows = json.loads((args.run_root / args.phase / "phase_results.json").read_text())
        row = next(r for r in rows if r["sample_id"] == args.sample_id)
        path = Path(row["evaluation"]["output_paths"][0]).with_suffix(".replay.npz")
        replay_episode(path, args.viewer, args.speed, not args.no_hold)
        return 0
    except AdaGraspError as error:
        print(str(error), file=sys.stderr)
        return error.exit_code
    except (ValueError, OSError, KeyError, StopIteration, DebugViewerError) as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
