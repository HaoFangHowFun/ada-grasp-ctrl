"""Run an independent CRX-Sharpa grasp with the project's adaptive ours controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ada_grasp_ctrl.crx_sharpa_adaptive import run_phase2, run_phase2_example
from ada_grasp_ctrl.crx_sharpa_contract import OBJECT_SCALE
from ada_grasp_ctrl.paths import runtime_root


def main() -> None:
    """Execute an independent adaptive run or consume an existing grasp input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_run_dir", nargs="?", type=Path, help="Optional existing run with model and grasp inputs."
    )
    parser.add_argument("--run-dir", type=Path, help="Fresh output directory for an independent Phase 2 run.")
    parser.add_argument("--viewer", action="store_true", help="Show live adaptive control in MuJoCo.")
    parser.add_argument(
        "--object-scale", type=float, default=OBJECT_SCALE,
        help=f"Physical bottle mesh scale for an independent run (default: {OBJECT_SCALE:g}).",
    )
    args = parser.parse_args()
    if args.source_run_dir is not None:
        if args.run_dir is not None or args.object_scale != OBJECT_SCALE:
            parser.error("--run-dir and --object-scale apply only to an independent Phase 2 run")
        summary = run_phase2(args.source_run_dir, viewer=args.viewer)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-%f")
        run_dir = args.run_dir or runtime_root("output") / "examples" / "crx_sharpa" / f"phase2-{run_id}"
        summary = run_phase2_example(run_dir, object_scale=args.object_scale, viewer=args.viewer)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
