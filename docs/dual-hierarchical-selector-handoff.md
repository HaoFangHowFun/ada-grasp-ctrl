# Dual-live hierarchical selector integration

Baseline: `0ad24f3c1b3625a08da31942bfb09db51a23f0d9`, original checkout on `main`.
The standalone right-full pipeline, shared mock executor, MPC/tactile controllers,
drivers, legacy candidate serialization and execution packet contract are unchanged.
No physical actuation, real checkpoint inference, camera capture or ROS driver
startup was performed for this integration. No commit or push was requested for it.

## Behavior and compatibility

`script/run_dual_grasp_pipeline.py --selector legacy` remains the default. It calls
the same `select_candidates()` with the same arguments, then the unchanged exporter.
The new YAML block is ignored by the legacy path. No YAML `selector` key can activate
hierarchical selection. The CLI/config manifests record the explicit switch.

`--selector hierarchical` validates a schema-1 `kind: ranking` prior and explicit
supported modes before creating processes or capturing an object. The live adapter
accepts only `right_full` and `left_full`; ranking and supported modes must be unique,
nonempty and name the same set. A supported subset is allowed; unranked/unsupported
groups remain selector diagnostics and cannot be executed. No counts or synthetic
probabilities are converted into a prior. See the [usage guide](dual-live-grasp-pipeline.md#optional-hierarchical-selection).

The unchanged `infer_live_sides()` performs only enabled inference branches and
keeps fresh-capture hashes, model verification, reflection geometry and the legacy
`candidates.json`. The adapter copies rows with `candidate_from_schema1`, checks
single-side permissions, source generator, checkpoint/metadata, reflection provenance
and unique prefixed IDs, and adds canonical provenance fields:

| Execution-side mode | Source model | Generation method |
| --- | --- | --- |
| `right_full` | `right_full` | `right_full` |
| `left_full` | `right_full` | `mirrored_right_full` |

These labels introduce no new trained inference capability. The existing
`select_hierarchical()` receives the current scene and complete measured 56-joint
state. It retains all backend IK, contact, joint-limit and effort checks. The first
feasible ranked mode wins; cheaper candidates in lower modes cannot override it.
Ties within a mode retain candidate-ID/seed ordering. `--arms` limits actual
inference and IK; an unavailable ranked side is reported as an empty mode.

The compatibility adapter validates the returned candidate ID, side, mode, arm
requirements, input targets, availability and canonical plan before producing a
legacy `status: selected` / `no_feasible_candidate` envelope. Its selected candidate
is the original schema-1 row, including unchanged provenance. The hierarchical
report is embedded separately under `hierarchical_report` for audit and written to
`hierarchical_selection.json`; the canonical batch and selected plan have separate
convenience files. The root selection, including its embedded audit, is already
hash-bound by `export_dual_selection()` and `load_packet()`. Standalone convenience
files are not execution inputs.

Neither `SELECTED` nor a canonical plan authorizes motion. Canonical readiness flags
stay false; existing packet validation, preview hash binding, Enter gate, fresh
snapshot checks, ROS probes and authorization remain in the same downstream flow.
Only one selected arm can be exported. A no-feasible result saves the full trace,
returns exit code 1 and creates no export, viewer, sender or receiver. Malformed
configuration returns preflight code 2 before any process launch.

## Files changed

- `pipeline/dual_runner.py`: explicit CLI switch, preflight configuration check,
  opt-in selector call and canonical audit artifacts.
- New `pipeline/dual_selection.py`: ranking validation, canonical input adaptation
  and checked legacy result projection. No feasibility algorithm is duplicated.
- `config/experiments/dual_grasp_pipeline.yaml`: editable example preference.
- New `tests/dual_selector_fixtures.py` and `tests/test_dual_hierarchical_pipeline.py`:
  synthetic CLI/export, compatibility, identity, provenance and authorization tests.
- New `script/test_dual_hierarchical_mock.py`: reproducible synthetic example output.
- This handoff and `docs/dual-live-grasp-pipeline.md`: behavior and usage.

Paths above are relative to `src/ada_grasp_ctrl/` for production modules/config.
`dual_inference.py`, `dual_artifacts.py`, selector/backend/contracts, all standalone
right-pipeline files and all pre-existing tests are untouched.

## Inspectable synthetic outputs

Evidence root: [`output/dual_selector/hierarchical_20261120_v1`](../output/dual_selector/hierarchical_20261120_v1/).
Its [manifest](../output/dual_selector/hierarchical_20261120_v1/run_manifest.yaml)
records source hashes, exit codes, selected identities and verified packet hashes.
Each case includes synthetic checkpoint/training/metadata files with real local
hashes, candidates, measured-state fixture, selection and applicable packet artifacts.
These checkpoint files are clearly marked synthetic and contain no network weights.

| Case | Outcome |
| --- | --- |
| `legacy_default` | Selects cheaper `left:c0000` (0.06 rad²) |
| `right_ranked` | Selects `right:c0000` (0.96 rad²), despite cheaper left |
| `left_ranked` | Selects `left:c0000` with reversed ranking |
| `fallback` | Right synthetic IK rejected, then selects `left:c0000` |
| `infeasible` | `NO_FEASIBLE_GRASP`, no target packet |
| `left_only` | Only left inference and evaluation, even with right ranked first |

Example selected files:

- [Right selection](../output/dual_selector/hierarchical_20261120_v1/right_ranked/run/selection.json),
  [canonical plan](../output/dual_selector/hierarchical_20261120_v1/right_ranked/run/grasp_plan.json),
  [trace](../output/dual_selector/hierarchical_20261120_v1/right_ranked/run/hierarchical_selection.json),
  [validated packet](../output/dual_selector/hierarchical_20261120_v1/right_ranked/run/export/targets.json).
- [Mirrored-left batch with checkpoint provenance](../output/dual_selector/hierarchical_20261120_v1/left_ranked/run/canonical_candidates.json)
  and [left packet](../output/dual_selector/hierarchical_20261120_v1/left_ranked/run/export/targets.json).
- [All-infeasible trace](../output/dual_selector/hierarchical_20261120_v1/infeasible/run/hierarchical_selection.json).

The fixture substitutes capture, network output, analytical robot IK/contacts and
hand geometry. It runs the real per-side inference orchestration/mirror adapter,
hierarchical/legacy selectors, overhead/lift helpers, exporter and packet validators.
It starts no ROS nodes or subprocess drivers. These examples prove routing and
serialization, not physical reachability, collision clearance or grasp performance.

## Validation

Results on 2026-11-20:

- New integration suite: **10 passed, 0 failed/skipped**.
- Focused integration + dual-live + hierarchical + canonical contracts:
  **36 passed, 0 failed/skipped**.
- Full Ada unit suite: **568 run, 483 passed, 85 skipped, 0 failures/errors** (39.427 s).
- Sourced ROS contract subset: **18 run, 14 passed, 4 skipped, 0 failures/errors**.
  Existing right `linear` / `mpc_active` and dual receiver tests pass. The four
  dynamic-lift tests require their combined sourced/offline fixture environment.
- Six synthetic audit cases passed with expected packet or no-feasible outcomes.
- Whole-tree Ruff lint, compilation, changed-file formatting and `git diff --check`
  passed. Full-tree formatting still reports the same eight pre-existing files;
  this task does not reformat them.

The maintained LEAP quick example completed with **100 outputs: 87 success,
6 failure, 7 invalid initialization, 0 solver degradation and 0 execution errors**,
exit code 0. Current-run evaluation/statistics reports are under
[`output/examples/leap_tac3d/dual_selector_20261120_v1`](../output/examples/leap_tac3d/dual_selector_20261120_v1/).
This is the existing offline bundled-example smoke test, not a dual-live grasp
performance test or a full three-hand release gate. Validation logs are preserved
under the synthetic evidence root's `test_logs/` directory.

Exact commands from the repository root:

```bash
env PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/ada_dual_selector_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 .conda/bin/python -m unittest test_dual_hierarchical_pipeline \
  test_dual_live_pipeline test_hierarchical_selection test_grasp_contracts -v

env PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_dual_selector_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 .conda/bin/python -m unittest discover -s tests -v

env PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/ada_dual_selector_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 .conda/bin/python script/test_dual_hierarchical_mock.py \
  --output-root output/dual_selector/hierarchical_20261120_v1

env ROS_DOMAIN_ID=88 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  bash --noprofile --norc -c '
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/src:/home/hao-fang/hugs-workspace/ada-grasp-ctrl/tests:${PYTHONPATH:-}
nice -n 10 /usr/bin/python3 -m unittest test_right_ros_receiver \
  test_dual_live_receiver test_mpc_p3_receiver test_dynamic_lift_ros -v
'

env PYTHONPATH=src PYTHON_BIN=.conda/bin/python MPLCONFIGDIR=/tmp/ada_dual_selector_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  ADA_GRASP_CTRL_RUN_ID=dual_selector_20261120_v1 \
  nice -n 10 bash script/run_example.sh leap_tac3d quick

.conda/bin/python -m compileall -q src tests script
/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff check src tests script
/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff format --check src tests script
git diff --check
```

Use new output roots/run IDs for reruns. Only the known full-tree format check is
expected to exit nonzero. Tests used low CPU priority, one numeric thread, no GPU
and no evaluation-worker changes. ROS-dependent checks explicitly used localhost
domain **88**; the ordinary domain **87** workflow remains unchanged.

Physical validation and actual camera/checkpoint integration runs remain separate
operator-controlled work. This patch adds no Human Prior, independent left model,
three-finger modes, bimanual execution, Place or Task 4 shared-executor integration.
