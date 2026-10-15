"""Run the experimental CRX-5iA and Sharpa fixed-target grasp example."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ada_grasp_ctrl.crx_sharpa_demo import run_phase1
from ada_grasp_ctrl.paths import runtime_root


def main() -> None:
    """Parse CLI arguments and execute one fresh example run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, default=None, help="Fresh output directory; defaults below output/.")
    parser.add_argument("--viewer", action="store_true", help="Show the MuJoCo run and wait for the window to close.")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-%f")
    run_dir = runtime_root("output") / "examples" / "crx_sharpa" / run_id if args.run_dir is None else args.run_dir
    print(json.dumps(run_phase1(Path(run_dir), viewer=args.viewer), indent=2))


if __name__ == "__main__":
    main()
