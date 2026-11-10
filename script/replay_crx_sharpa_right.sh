#!/usr/bin/env bash
# Replay recorded right CRX + right Sharpa episodes in a browser.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/.conda/bin/python}"
unset PYTHONHOME
export PYTHONPATH="${project_root}/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ada_grasp_ctrl_mpl}"
export CUDA_VISIBLE_DEVICES=""
exec "$python_bin" "${project_root}/script/replay_crx_sharpa_right.py" "$@"
