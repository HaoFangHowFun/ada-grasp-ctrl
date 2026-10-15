"""Replay a completed CRX-Sharpa Phase 2 control record in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ada_grasp_ctrl.crx_sharpa_adaptive import replay_phase2


def main() -> None:
    """Replay a completed Phase 2 run until its native viewer closes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Existing Phase 2 run directory.")
    parser.add_argument("--fps", type=float, default=10.0, help="Recorded control frames per second (default: 10).")
    args = parser.parse_args()
    print(json.dumps(replay_phase2(args.run_dir, fps=args.fps), indent=2))


if __name__ == "__main__":
    main()
