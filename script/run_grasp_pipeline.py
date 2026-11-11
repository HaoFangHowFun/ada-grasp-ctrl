"""Capture the current object, preview on 8089, then run one Enter-gated grasp cycle."""

from pathlib import Path
import os
import sys


def main() -> int:
    """Bootstrap the checkout's maintained Ada environment, isolating inherited ROS paths."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    checkout = Path(__file__).resolve().parents[1]
    source = str(checkout / "src")
    python = checkout / ".conda/bin/python"
    if not python.is_file():
        raise SystemExit("The maintained Ada .conda environment is required for this checkout entry point")
    # ROS Python 3.12 packages must never enter the Ada Python 3.10 process.
    if os.environ.get("PYTHONPATH") != source or Path(sys.executable).resolve() != python.resolve():
        environment = os.environ.copy()
        environment["PYTHONPATH"] = source
        environment.pop("PYTHONHOME", None)
        os.execve(str(python), [str(python), *sys.argv], environment)
    sys.path.insert(0, source)
    from ada_grasp_ctrl.pipeline.runner import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
