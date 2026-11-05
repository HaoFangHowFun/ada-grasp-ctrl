#!/usr/bin/env bash
# Generate frozen inputs once, then run both phases without opening a viewer.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/.conda/bin/python}"
suite_root="${SHADOW_SUITE_ROOT:-${project_root}/output/regression/shadow-native-100x10-v1}"
run_root=""
baseline=""
prepare_only=0
smoke=0
workers=8
usage() {
  cat <<'EOF'
Usage: bash script/test_shadow_controller.sh [options]
  --prepare-only          Build/verify the frozen 1000-grasp suite without simulation
  --suite-root PATH       Override the frozen suite directory
  --output-root PATH      New result directory (default: unique timestamp under output/regression/runs)
  --baseline PATH         Previous result directory or regression.json; never overwritten
  --workers N             CPU workers per phase (default: 8; use 1 for serial)
  --smoke                 Run 4 frozen samples, one per scale (8 executions)
  -h, --help              Show this help

Default: 100 native scenes x 10 frozen grasps x 2 phases = 2000 executions.
Protocol: friction 0.8, matched initialization, no extra squeeze, seed 12, 8 workers.
Environment: PYTHON_BIN, DEXLEARN_ROOT, SHADOW_CHECKPOINT, SHADOW_METADATA, SHADOW_SUITE_ROOT.
Preparation uses DexLearn only when the suite does not exist. Existing partial/changed suites fail.
Exit: 0 completed cleanly; 1 solver degradation/execution errors or baseline regression; 2 preflight error.
Preview is separate: see script/preview_shadow_regression.sh --help.
EOF
}
while (( $# )); do
  case "$1" in
    --prepare-only) prepare_only=1; shift ;;
    --smoke) smoke=1; shift ;;
    --suite-root|--output-root|--baseline|--workers)
      if (( $# < 2 )) || [[ -z "$2" ]]; then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in --suite-root) suite_root="$2";; --output-root) run_root="$2";; --baseline) baseline="$2";; --workers) workers="$2";; esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then echo "Workers must be a positive integer: $workers" >&2; exit 2; fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then echo "Python unavailable: ${python_bin}" >&2; exit 2; fi
unset PYTHONHOME
export PYTHONPATH="${project_root}/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ada_grasp_ctrl_mpl}"
# Do not set BLAS thread counts here: changing them can change optimizer trajectories.
if [[ ! -e "${suite_root}" ]]; then
  dex_root="${DEXLEARN_ROOT:-${project_root}/../HUGS-DexLearn}"
  checkpoint="${SHADOW_CHECKPOINT:-${dex_root}/output/shadowMulti_robotMultiHierar_right_full_752k_v1/ckpts/step_050000.pth}"
  metadata="${SHADOW_METADATA:-/home/hao-fang/hugs-dataset/BimanBODex_shadow_tabletop_777k_v1/shadow/right_full/metadata.json}"
  "${python_bin}" "${project_root}/script/shadow_controller_regression.py" prepare \
    --suite-root "${suite_root}" --dexlearn-root "${dex_root}" --checkpoint "${checkpoint}" --metadata "${metadata}"
fi
if (( prepare_only )); then
  "${python_bin}" -c 'from pathlib import Path; import sys; from ada_grasp_ctrl.experiments.shadow_regression import load_suite; s=load_suite(Path(sys.argv[1])); print("Verified frozen suite:",len(s["samples"]),"grasps")' "${suite_root}/suite_manifest.json"
  exit 0
fi
if [[ -z "${run_root}" ]]; then
  run_root="${project_root}/output/regression/runs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
args=(run --workers "$workers" --suite "${suite_root}/suite_manifest.json" --output-root "${run_root}")
if (( smoke )); then args+=(--smoke); fi
if [[ -n "${baseline}" ]]; then args+=(--baseline "${baseline}"); fi
exec "${python_bin}" "${project_root}/script/shadow_controller_regression.py" "${args[@]}"
