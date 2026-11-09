"""Prepare or reopen the right CRX/Sharpa placement review; no evaluation episodes."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from ada_grasp_ctrl.paths import project_root


def main() -> int:
    """Run the preview-only milestone with explicit fresh-output/reopen semantics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-root", type=Path, help="Reopen a previously generated preview")
    parser.add_argument("--sample-id")
    parser.add_argument("--prepare-only", action="store_true", help="Write report and screenshots, then exit")
    parser.add_argument("--port", type=int, default=8086)
    args = parser.parse_args()
    repo = project_root()

    def resolve(path: Path) -> Path:
        return (path if path.is_absolute() else repo / path).resolve()

    if args.run_root and (args.output_root or args.suite_root):
        parser.error("--run-root cannot be combined with --output-root or --suite-root")
    try:
        from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import load_preview, prepare_preview
        from ada_grasp_ctrl.experiments.crx_sharpa_right_preview import render_screenshots, serve_preview

        if args.run_root:
            output = resolve(args.run_root)
            report = load_preview(output)
        else:
            suite = resolve(args.suite_root or Path("output/regression/sharpa-right-100k-tabletop-100x10-v1"))
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = resolve(
                args.output_root or Path(f"output/regression/crx-sharpa-right-previews/{stamp}-{os.getpid()}")
            )
            report = prepare_preview(suite / "suite_manifest.json", output, args.sample_id)
            render_screenshots(report, output)
        if args.sample_id and args.sample_id not in {r["sample_id"] for r in report["samples"]}:
            raise ValueError("Sample is not in this preview; prepare a new run with --sample-id")
        print(f"PREVIEW_REPORT={output / 'preview_report.json'}", flush=True)
        if not args.prepare_only:
            serve_preview(report, port=args.port, sample_id=args.sample_id)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Preview preflight failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
