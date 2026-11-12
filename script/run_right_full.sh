#!/usr/bin/env bash
# Offline right_full execution; exclude inherited ROS Python search paths.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/.conda/bin/python}"
unset PYTHONHOME
export PYTHONPATH="${project_root}/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ada_grasp_ctrl_mpl}"
export CUDA_VISIBLE_DEVICES=""
exec "$python_bin" -m ada_grasp_ctrl.experiments.right_full_pipeline "$@"
