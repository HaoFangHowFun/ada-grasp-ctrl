#!/usr/bin/env bash
# Reuse the name-independent paired replay browser; no evaluation is performed.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/.conda/bin/python}"
unset PYTHONHOME
export PYTHONPATH="${project_root}/src"
exec "$python_bin" "${project_root}/script/preview_shadow_regression.py" \
  --label "Sharpa right 100K | friction 0.8 | experimental" "$@"
