#!/usr/bin/env bash
# Paired MuJoCo smoke test of the approved right CRX + right Sharpa preview.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/.conda/bin/python}"
unset PYTHONHOME
export PYTHONPATH="${project_root}/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ada_grasp_ctrl_mpl}"
export CUDA_VISIBLE_DEVICES=""
exec "$python_bin" "${project_root}/script/test_crx_sharpa_right_controller.py" "$@"
