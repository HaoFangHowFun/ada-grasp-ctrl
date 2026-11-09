#!/usr/bin/env bash
# Frozen right-hand predictions; native op/ours, no training or viewer.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/.conda/bin/python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python unavailable: $python_bin" >&2
  exit 2
fi
unset PYTHONHOME
export PYTHONPATH="${project_root}/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ada_grasp_ctrl_mpl}"
# Preparation and simulation use CPU; this entry never performs model inference.
export CUDA_VISIBLE_DEVICES=""
exec "$python_bin" "${project_root}/script/sharpa_controller_regression.py" "$@"
