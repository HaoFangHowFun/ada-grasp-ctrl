"""Independent right CRX + right Sharpa smoke evaluator using an approved preview."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.experiments.crx_sharpa_right_evaluation import run_smoke


def main() -> int:
    """Run four fixed-scale op/ours pairs; no ROS or hardware commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Four fixed scales x op/ours (default and current scope)")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--preview-root", type=Path, default=Path("output/regression/crx-sharpa-right-preview-x080-y070-config-v1")
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        choices=[1],
        default=1,
        help="Serial paired execution; parallel execution is not yet validated",
    )
    args = parser.parse_args()
    root = project_root()
    preview = (root / args.preview_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        root / (args.output_root or Path(f"output/regression/crx-sharpa-right-smoke/{stamp}-{os.getpid()}"))
    ).resolve()
    try:
        code = run_smoke(preview, output, prepare_only=args.prepare_only)
        print(f"RUN_REPORT={output / 'run_report.json'}", flush=True)
        return code
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Right CRX smoke preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
