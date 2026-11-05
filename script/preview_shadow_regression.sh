#!/usr/bin/env bash
# Open recorded results only; never run inference or a controller.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/.conda/bin/python}"
unset PYTHONHOME
export PYTHONPATH="${project_root}/src"
exec "${python_bin}" "${project_root}/script/preview_shadow_regression.py" "$@"
