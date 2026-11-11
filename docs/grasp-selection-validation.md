# Grasp selection validation — 2026-11-10

Software implementation and the five-capture mock/offline evaluation are complete.
Physical deployment acceptance is pending. No real actuator was started, no motion
command was issued, and neither inference nor CRX physics was run for selection.

## Inputs and actual results

Artifacts are under `output/grasp_selection/mvd_20261110_v1/` (not committed).
The actual snapshot is `capture_checked/robot_state.json`, received from the
right-only mock launch on ROS domain 89. Right J1–J6 were
`[-90, -30, 210, 0, -60, 0]` degrees, read as radians from the broadcaster rather
than substituted from the offline initial-pose configuration.

The subscriber checked the observed combined URDF against the prepared model's
axes, limits, fixed base and flange mounting. Model URDF SHA-256:
`e013b12a14385478ce05f544615f7514eba7b376e0bee3fbb36b8aa6c32abff8`.
The camera calibration file exists and matches its recorded SHA-256:
`77f66ac38eea55774538f45f0b9c748bbc0106caea909e30f6141fcd6e4e99b5`.
The nominal transform chain and exact source topics are documented in
[grasp-selection.md](grasp-selection.md#frames-and-state-provenance).

`ros_graph_audit.json` records no command topics. The launch confirmed
`mock_components/GenericSystem` for CRX and `backend=mock, read_only=True` for
Sharpa. All processes started by that launch were stopped after capture.
The unrelated Viser preview stays available on port 8089.

Each group has eight real saved DexLearn candidates. All 40 candidates were
evaluated twice in independent fresh directories. Both selection and candidate
evaluation JSONs were byte-identical for all five pairs.

| Group | Captured object | Feasible under modeled static gates | Selected ID | Weighted squared displacement (rad²) |
| --- | --- | ---: | --- | ---: |
| 01 | Green pear-shaped object, first capture | 2 / 8 | c0003 | 1.155914413343405 |
| 02 | Green pear-shaped object, second capture | 3 / 8 | c0004 | 1.8853102093079903 |
| 03 | Rubik's cube | 3 / 8 | c0004 | 1.3207225999492476 |
| 04 | Transparent bottle | 4 / 8 | c0000 | 2.4587393588385362 |
| 05 | USB adapter body | 3 / 8 | c0000 | 2.719899873975981 |

The integration index is `five_groups_final/integration_summary.json`; each
`groupNN_repeat1/` contains the full three required reports, configuration and
manifest. These results are selection under the recorded partial scene, not
complete collision-free motion or grasp success.

For example, group 02 evaluates every candidate as follows:

| Candidate | Feasible retained branches | Best cost | Rejections |
| --- | ---: | ---: | --- |
| c0000 | 3 | 4.120696641443579 | None |
| c0001 | 0 | — | Grasp static collision |
| c0002 | 0 | — | Grasp IK convergence/residual |
| c0003 | 0 | — | Grasp IK convergence/residual |
| c0004 | 3 | 1.8853102093079903 | Selected minimum |
| c0005 | 4 | 14.025946884157504 | None |
| c0006 | 0 | — | Grasp IK convergence/residual and static collision |
| c0007 | 0 | — | Grasp IK convergence/residual |

The selected group 02 right-arm vector is
`[-0.7143480063, -1.0234463569, 3.4103615336, -0.8697652956, -1.0940889697, -0.2798752021]` rad.
Each option reports raw deltas, unweighted norm, exact cost, seed attempts, all
computed stage residuals and collisions. Rejected stages skipped after a grasp
failure are explicitly marked `not_checked`; they are not recorded as passing.

Group 01 c0001 is rejected for hand joint limits. The older hand-only visual
preview flagged seven group 05 candidates for any visual-mesh table intersection.
This selector uses the existing real CRX/Sharpa collision model and its unchanged
10 mm hand penetration gate. Group 05's selected c0000 has about **5.994 mm** hand/
table penetration at squeeze and therefore passes that experimental gate; it is
not a zero-collision or execution-ready result. Selected groups 01–03 also have
approximately 6–9 mm modeled hand self-penetration, below that same inherited gate.
Neither the geometry nor thresholds were changed to obtain these outcomes.

## Tests and checks

Commands were run in the Ada checkout with its existing `.conda` Python 3.10:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  .conda/bin/python -m unittest discover -s tests -p 'test_grasp_selection*.py' -v
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  .conda/bin/python -m unittest discover -s tests -v
.conda/bin/python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
```

- Focused final selector/CRX/preview tests: **33 passed**, no skips or failures.
  Real-model tests include 28 named joints, independent FK, selected FK within the
  existing tolerances, all three stages, known table collision and a clear FK
  fixture. Other tests cover ranking, ties, multiple branches, weights, empty/all
  failed sets, malformed poses, names, states, frames, no double transform,
  rejected hard constraints, command-free imports and output overwrite refusal.
- Final full suite including the preview test: **235 run, 234 passed,
  1 skipped, 0 failures**. The skip is the existing OpenCV projection boundary in
  Ada's environment. Right CRX preparation, fixed camera, Shadow/dummy arm and the
  pre-existing ROS bridge tests were included.
- `/usr/bin/python3` ran `test_fixed_camera_grasp.py` separately with system
  OpenCV: **5 passed**, covering the projection boundary skipped in Ada's environment.
- Full-repository Ruff check and compileall passed. Ruff was already available at
  `/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff`; nothing was installed.
- All added files pass Ruff format. Full-repository format check still reports
  eight pre-existing files, matching the documented prior CRX status. Those files
  were not changed.
- Actual ROS snapshot capture passed after explicitly checking the installed
  broadcaster's documented `base_link` header. The first attempt rejected that
  header rather than accepting an assumed frame, and its empty output directory
  was retained. No startup motion task was launched.
- Viser HTTP returned 200. A real local WebSocket client switched all five
  datasets and verified that playback changes progress. Evidence:
  `preview_8089_final/ui_smoke.json`. Ten rendered images cover all selected grasps.

## Existing Shadow quick baseline discrepancy

The maintained Shadow quick command ran all 100 samples in a fresh output root:

```bash
env -u PYTHONPATH \
  PYTHON_BIN=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/.conda/bin/python \
  ADA_GRASP_CTRL_RUN_ID=selector-mvd-20261110-v1 \
  ADA_GRASP_CTRL_EXAMPLE_BASE=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/output/grasp_selection/mvd_20261110_v1/regression \
  bash script/run_example.sh shadow quick
```

It exits 1 because seven solver-degraded samples are retained. Counts:
**68 success, 4 failure, 21 invalid initialization, 7 solver degraded,
0 execution error**. Every per-sample classification exactly matches the two
pre-task runs `output/examples/shadow/20261110T011021Z-25098` and
`output/examples/shadow/20261109T223837Z-2162080`.

Verification against `examples/quick_expected_status.json` still fails for three
pre-existing classifications: one mug changes failure→success, tape changes
solver_degraded→success, and pizza changes solver_degraded→failure. The exact
paths are in `shadow_quick_comparison.json`. This is not a passing release gate;
the golden inventory, solver policy and controller code were not altered.

## Physical acceptance still pending

Camera→world, world→right base and mounted flange/TCP need independent physical
acceptance. The table remains `preview_unaccepted`. The five object assets are
single-view surfaces; the bottle has material-related depth uncertainty and the
adapter asset excludes its cable and USB plug. Full object/environment collision
geometry and safe continuous paths are absent. Endpoint selection and display
interpolation cannot resolve these gates. Every result has `execution_ready=false`.

## Exact added files

No pre-existing tracked implementation file was modified. The user's pre-existing
untracked ROS bridge files were preserved. This change adds:

```text
src/ada_grasp_ctrl/grasp_selection/__init__.py
src/ada_grasp_ctrl/grasp_selection/types.py
src/ada_grasp_ctrl/grasp_selection/selector.py
src/ada_grasp_ctrl/grasp_selection/cli.py
src/ada_grasp_ctrl/grasp_selection/capture_state.py
src/ada_grasp_ctrl/grasp_selection/preview.py
src/ada_grasp_ctrl/grasp_selection/criteria/__init__.py
src/ada_grasp_ctrl/grasp_selection/criteria/base.py
src/ada_grasp_ctrl/grasp_selection/criteria/joint_effort.py
src/ada_grasp_ctrl/grasp_selection/criteria/reachability.py
src/ada_grasp_ctrl/grasp_selection/adapters/__init__.py
src/ada_grasp_ctrl/grasp_selection/adapters/dexlearn.py
src/ada_grasp_ctrl/grasp_selection/adapters/state.py
src/ada_grasp_ctrl/grasp_selection/adapters/crx_sharpa.py
script/evaluate_grasp_selection_gallery.py
tests/test_grasp_selection.py
tests/test_grasp_selection_crx.py
docs/grasp-selection.md
docs/grasp-selection-validation.md
```

The usage document gives the exact offline, read-only ROS and unified 8089 preview
invocations. Generated captures, model diagnostics, JSON reports and images remain
under the fresh ignored output tree.
