#!/usr/bin/env bash
# Static placement review only: no physics stepping, ROS, or controller evaluation.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/.conda/bin/python}"
unset PYTHONHOME
export PYTHONPATH="${project_root}/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ada_grasp_ctrl_mpl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$python_bin" "${project_root}/script/preview_crx_sharpa_right.py" "$@"
