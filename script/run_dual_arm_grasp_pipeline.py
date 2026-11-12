"""Run the independent Phase 2 offline simulation; no ROS or hardware backend."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from ada_grasp_ctrl.experiments.dual_arm_grasp_pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
