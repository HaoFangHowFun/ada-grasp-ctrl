# Task 2 — Read-only 8089 dashboard results

Implemented on the existing uncommitted Task 1 worktree at `main`, reference HEAD
`ecc07ce920ebf6f1d285f925f9dff025b11d6c9e`. Task 1 changes were preserved.
No robot motion, real active pipeline, live ROS subscriber, dependency installation, commit or push was performed for Task 2.

## Changes

- `pipeline/telemetry.py`: extends the existing tactile reader with arm joints and robot-description subscribers.
  Validates named actual 6+22 feedback, clocks, model/world placement and publisher uniqueness.
  Latest snapshots at approximately 20 Hz; callbacks run independently of stdout. No publishers, action clients or SDK sessions.
- `pipeline/dashboard.py`: existing Viser/MuJoCo renderer, actual-joint kinematics only, visual meshes only,
  calibrated table, verified static original cloud, five contacts and orange force-on-object / green normal arrows.
  Reuses tactile adapter/diagnostics and their provisional geometry policy. Shows honest partial/stale states,
  native-event force separately from monitor estimate, and phases through return. No physics stepping in updates.
  Collision meshes are hidden only in the private presentation model; controller collision checks are unchanged.
- `pipeline/dashboard_replay.py`: bounded synthetic replay around recorded joints, with five synthetic contacts,
  full phase sequence, static recorded cloud, scene serialization and metrics. No ROS connection.
- Small hooks in `pipeline/viewer.py`, `pipeline/runner.py`, `tactile_preview/__main__.py`:
  existing early 8089 ownership, optional runtime monitor, low-priority single-thread BLAS monitor processes,
  explicit reader command/environment, five-second terminal display and owned cleanup.
  Initial port conflict remains a preflight error. Runtime monitor failure cannot abort control.
- `tests/test_pipeline_dashboard.py`, operator guide and updated `docs/grasp-pipeline.md`.

No native solver/objective, 15 N completion logic, gains, lift planning, command rate or existing mode policy changed.
No overlap conflict remains with the current Task 1 implementation.

## Focused offline checks

```bash
PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl OPENBLAS_NUM_THREADS=1 \
  .conda/bin/python -m unittest test_pipeline_dashboard test_task1_active_pipeline \
  test_tactile_adapter test_tactile_preview test_tactile_force_replay -q
```

PASS: 59 tests, 58 passed and one expected skip (opt-in isolated synthetic ROS DDS integration).
Coverage includes actual-joint ordering/freshness/model checks, subscriber-only mocked ROS callbacks,
world point/direction transform and force sign, display scale and >15 N, invalid/missing/multi-contact data,
worker exit/cleanup, port refusal, monitor failure isolation, phase/native event semantics,
static cloud retention and all 28 measured joints on the recorded model. `mj_step` is forbidden in the model test.
Existing Task 1 authorization and baseline-mode routing tests passed.

Ruff check and format check passed on the seven changed Task 2 Python files:
`pipeline/{dashboard,dashboard_replay,telemetry,runner,viewer}.py`,
`tactile_preview/__main__.py`, `tests/test_pipeline_dashboard.py`.
Used existing `/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff`.
Syntax check passed:

```bash
.conda/bin/python -m compileall -q src/ada_grasp_ctrl/pipeline \
  src/ada_grasp_ctrl/tactile_preview/__main__.py tests/test_pipeline_dashboard.py
```

## Browser replay and measurements

The temporary offline smoke harness was run with:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .conda/bin/python /tmp/task2_smoke.py
```

Evidence: `/tmp/ada-task2-smoke-669t2nix/`:

- `dashboard.png`: actual browser screenshot after removing collision meshes and the renderer's implicit infinite grid.
- `replay/dashboard_scene.viser`, `dashboard_state.json`, `dashboard_metrics.json`, `replay_report.json`.
- `synthetic_tick_timing.json`, replay/browser logs.

The harness ran the documented replay for 15 seconds, opened headless Chrome, captured a screenshot,
reloaded and disconnected the browser, and let the replay finish. Its children exited normally.
All five contacts remained valid; the panel showed forces above 15 N without clipping;
2,311 recorded object points stayed fixed. The browser screenshot explicitly labels synthetic replay.

One local run measured 291 updates / 15.014 seconds (19.4 Hz), 2.862 CPU seconds
(about 19.1% of one core, excluding browser and ROS reader), steady update median 5.62 ms / p95 11.86 ms,
in-process synthetic telemetry arrival lag p95 0.83 ms, and mean telemetry JSON 5,668 bytes
(about 113 KB/s at 20 Hz). A separate no-command 100 Hz timer ran 1,621 ticks over 16.20 seconds,
maximum lateness 2.96 ms. This timer is not a native controller test.

These are local offline observations, not live DDS latency, browser-render latency, or hardware timing guarantees.
The screenshot browser used software WebGL. Real hardware telemetry performance, 10 Hz native MPC /
100 Hz driver timing under live load, and physical arrow alignment were not tested; those remain operator-run items.
No full release gate or unrelated calibration revalidation was run, consistent with the requested minimal effort.
