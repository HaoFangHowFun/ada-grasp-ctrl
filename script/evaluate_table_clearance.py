"""Preview, then optionally execute, a right-hand tabletop clearance evaluation."""

import os
from pathlib import Path
import sys


def main() -> int:
    """Use the maintained geometry environment; ROS runs in a separate process."""
    root = Path(__file__).resolve().parents[1]
    python = root / ".conda/bin/python"
    source = str(root / "src")
    if not python.is_file():
        raise SystemExit("The checkout's .conda environment is required")
    if Path(sys.executable).resolve() != python.resolve() or os.environ.get("PYTHONPATH") != source:
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.update(PYTHONPATH=source, MUJOCO_GL="egl")
        os.execve(str(python), [str(python), *sys.argv], env)
    from ada_grasp_ctrl.experiments.table_clearance import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
