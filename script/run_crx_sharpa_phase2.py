"""Run the Phase 1 CRX-Sharpa grasp with the project's adaptive ours controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada_grasp_ctrl.crx_sharpa_adaptive import run_phase2


def main() -> None:
    """Parse one completed Phase 1 run and execute adaptive control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed Phase 1 output directory.")
    args = parser.parse_args()
    print(json.dumps(run_phase2(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
