# Task 1 — One-command real MPC pipeline

Date: 2026-11-13. Starting HEAD: `ecc07ce920ebf6f1d285f925f9dff025b11d6c9e`.
Implementation remains an uncommitted working-tree diff; no push or physical trial was performed.
The original untracked task prompt was preserved.

## Behavior

```bash
python script/run_grasp_pipeline.py --controller-mode mpc_active
```

For configured, externally running right hardware, this explicit live CLI choice authorizes this run,
selects the scoped hardware configuration, allocates a unique timestamped output directory, starts the
observational Viser viewer on 8089, captures the current cloud, selects the largest cluster whose 5th-percentile table height is at most 0.05 m,
runs existing inference/selection/export, and starts the existing ROS receiver and sender.
No ROI dialog, Enter gate, browser approval, offline fallback, or manual process transfer is used.
Python 3.10/native and ROS Python 3.12 remain subprocess-separated. Existing receiver enable and
confirmation guards receive run-local authorization; saved mode settings cannot grant it.

The user explicitly confirmed that the existing table plane and extrinsics had already been verified.
A dated, hash-bound operator attestation records this fact without rewriting or recalibrating those
artifacts. Existing quality, camera identity, capture freshness and calibration consistency checks remain.
Automatic extraction defaults to the full valid camera field with existing height/voxel/cluster filters.
Per the subsequent operator request, it filters by 5th-percentile signed table height <= 0.05 m,
then selects the largest remaining cluster. No eligible cluster, malformed/insufficient clouds,
and manual cluster overrides still reject. This replaces the original exact-one-cluster requirement.

After receiver-confirmed `native_complete`, the sender captures fresh coherent measured joints and
performs the existing 15 cm Lift FK/IK and sampled robot/table feasibility pass once. The receiver no
longer spawns a planner. It checks the exact packet/review identity, controlled sender result file,
validation outcome, native-complete/phase state, fresh feedback and measured start, including before
first publication. Initial Lift submission must be within 20 s of capture; subsequent phases retain
the cached immutable plan and existing fresh start checks. Arm/hand Lift start tolerances remain
0.01/0.02 rad. Hold/release/return use the derived endpoint and prior return destination.

Trust boundary: the runner fixes the same-run `sender/dynamic-lift-result.json` path before launching
both processes. The receiver requires its contents to match the transmitted hash-bound plan.
This trusts the controlled local planner and filesystem; the hash is not a cryptographic signature
against a compromised host. Existing explicit `mpc.live` orchestration now passes this path too.
Direct bridge callers must supply `--dynamic-lift-plan` for post-MPC hardware Lift.

Native solver, force conventions, 15 N completion threshold (not a hard force limit), 10 Hz solving,
100 Hz streaming, speed/acceleration/settle policies and intentional five-second Hold are unchanged.
Baseline, linear, passive and explicit offline/debug paths retain their behavior. Full contact-sheet
rendering is omitted from automatic preparation; selection/plan/export provenance is retained.

## Changed files

- `script/run_grasp_pipeline.py`, `pipeline/runner.py`: existing entry point, scoped authorization/defaults,
  early viewer, real sender/receiver routing, stage timing and cleanup.
- `pipeline/viewer.py`, `pipeline/ros_probe.py`: observational status and existing tactile publisher readiness.
- `experiments/fixed_camera.py`, `config/experiments/fixed_camera.yaml`,
  `config/experiments/grasp_pipeline.yaml`, `config/experiments/fixed_camera_acceptance_2026-11-13.json`:
  configured ROI, largest near-table cluster policy and the user's calibration attestation.
- `pipeline/dynamic_lift.py`, `ros_execution/bridge.py`, `experiments/right_ros_sender.py`, `mpc/live.py`:
  one planning process, trusted handoff checks and action/handoff timing.
- `mpc/contracts.py`: allow right hardware active mode as a data/plan identity; execution authorization
  remains in launcher/receiver. Other Phase 2 hardware modes remain prohibited.
- `tests/test_task1_active_pipeline.py`, `tests/test_dynamic_lift_ros.py`, `tests/test_mpc_phase2.py`:
  focused regression coverage. `docs/grasp-pipeline.md`: happy path, scope and troubleshooting.

Python paths above are relative to `src/ada_grasp_ctrl/` unless otherwise qualified.

## Validation

Focused tests only, per the user's minimum-effort instruction; no full release gate or physical trial.

```bash
PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
ADA_DYNAMIC_LIFT_PACKET=output/grasp_pipeline/p3_plan_20261113_132559/grouplive/export/targets.json \
ADA_DYNAMIC_LIFT_SNAPSHOT=output/mpc_p3/active_20261113_132624/sender/dynamic-lift-request.json \
.conda/bin/python -m unittest test_task1_active_pipeline test_grasp_pipeline \
  test_fixed_camera test_dynamic_lift test_mpc_phase2 test_mpc_linear_stream -q
```

66 tests passed. Coverage includes mocked unified orchestration, unique output roots, no prompts,
mode isolation, early viewer/port conflict, 0/1/2 clusters, cloud/calibration failures, manual ROI,
real harmless subprocess timing/failure propagation, recorded Lift FK/IK/hand retention, infeasible
IK/limits/collision rejection, unchanged streaming and legacy plan behavior.

ROS-message tests use sourced Jazzy/workspace system Python, with **all ROS initialization and
hardware I/O mocked**. `/tmp/ada-task1-recorded-review.json` contains the `handoff` field extracted
from the original successful `sender/dynamic-lift-result.json`:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
PYTHONPATH="$PWD/src:$PWD/tests:$PYTHONPATH" \
ADA_DYNAMIC_LIFT_PACKET="$PWD/output/grasp_pipeline/p3_plan_20261113_132559/grouplive/export/targets.json" \
ADA_DYNAMIC_LIFT_REVIEW=/tmp/ada-task1-recorded-review.json \
/usr/bin/python3 -m unittest test_dynamic_lift_ros test_mpc_p3_receiver -q
```

8 tests passed, including no receiver planner call; native-complete requirement; abort/cancel/stale
feedback/start mismatch; invalid/tampered/expired/wrong-phase handoff; rejected Lift; baseline/passive
sender parity. Synthetic clocks are rebased; recorded measured joint values are unchanged.

Also passed: `.conda/bin/python -m compileall -q src tests`, Ruff check/format check on changed Python
files, `git diff --check`, and launcher `--help`. Ruff 0.17.3 came from the existing local cache;
no dependency was installed.

## Timing evidence

One offline native-process trial used the successful `active_20261113_132624` measured completion
snapshot and original source-bound packet. No ROS publishers or physics stepping:

| Measurement | Seconds |
| --- | ---: |
| Worker source loading plus geometry/feasibility | 1.541 |
| Whole planner subprocess including startup and polling | 1.741 |

181 known-geometry path samples passed; minimum sampled hand/table gap was 0.027168261 m.
Trial artifacts: `/tmp/ada-task1-recorded-24e0pcz_/`; metrics: `/tmp/task1-timing-measurements.json`.
These are one-run local timings, not a hardware latency improvement claim.

Historical milestone events contain comparable same-host monotonic clocks:

| Historical interval | Seconds |
| --- | ---: |
| Receiver native_complete → sender fresh snapshot | 0.076 |
| Receiver native_complete → sender Lift goal submission | 1.825 |
| Receiver native_complete → first Lift trajectory log event | 3.638 |

The last event is logged after command publication; it is not an instrumented exact first-publication
timestamp. No action-directory timestamps were used. New logs distinguish planner duration, action
wait, Lift submission and receiver first-command event. New physical MPC-to-Lift latency, live camera
capture/inference timing and end-to-end real grasp success: **not measured**.

## Remaining limits

The baseline is a single successful operator-observed hardware milestone, not a success-rate study.
This implementation has not been physically exercised. Cluster count is a pragmatic proxy: one object
may split, and touching objects may merge. The viewer is a stage/status display in automatic mode;
high-rate 3D contacts/forces remain Task 2. Lift clearance is the existing sampled known robot/table
check, not continuous or unknown-object/environment collision certification. Other phase motion and
safety policies retain the baseline's existing scope.

## Follow-up: 5 cm tabletop selection and live pear check

At the operator's request, automatic extraction now keeps clusters whose 5th-percentile signed
height above the accepted table is <= 0.05 m, then chooses the largest by valid point count.
No eligible cluster still aborts. Manual selection behavior is unchanged. The parameter is
`automatic_table_lower_height_m` in `fixed_camera.yaml`; `clusters.json` records lower heights,
eligibility and the selected cluster. This is proximity to the table, not proof of physical contact.

Focused follow-up validation: `PYTHONPATH=src:tests .conda/bin/python -m unittest
 test_task1_active_pipeline.AutomaticObjectTests test_fixed_camera -q`: 13 passed; changed-file
Ruff/compile and `git diff --check` passed. Tests cover multiple eligible clusters, rejecting a larger
elevated cluster, the 5 cm threshold using the fifth percentile, and manual multi-cluster rejection.

A fresh, read-only camera/robot-feedback/inference/selection/export check of the operator's pear
completed at `/tmp/ada-pear-check-uz6okdpn` (`run_report.json`: `prepared`, `motion_requested=false`).
The overlay selects the central tabletop object: 2,311 points, lower-height p05 = 0.020457 m.
The other cluster has 301 points and p05 = 0.054561 m and is excluded. Four of eight grasp candidates
were feasible; `c0003` was selected and a complete target packet exported. The test used the existing
runner with `export_only=True`, returning before sender/receiver startup. No motion commands were
sent and no physical grasp success is claimed. Temporary review artifacts are not committed.
