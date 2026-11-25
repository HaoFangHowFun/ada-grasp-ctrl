# Dual MuJoCo implementation and validation, 2026-11-24

## Baseline and outcome

Baseline and current Git HEAD: `0f3e292722f6ae0ae09d9db33dec82fd65c0a55f`.
The worktree was clean before implementation. The implementation is an uncommitted
working-tree change; no reset, stash, push, hardware connection or ROS launch was
performed. Validation was offline, using actual local models and MuJoCo.

The existing `script/run_dual_grasp_pipeline.py --environment simulation` now
provides scene editing, actual settling, complete-cloud inference, existing
selection/planning, explicit browser review/execution, continuous Hold and Reset
on one long-lived 8089 server. Placement-only shares the builder and immutable
cloud while preserving the separate CLI boundary. See
[operator instructions and launch commands](dual-mujoco-simulation.md).

## Implementation boundaries

| Files | Responsibility |
| --- | --- |
| `pipeline/dual_runner.py` | Simulation branches before ROS validation, capture and transport; existing hardware/mock paths retained. |
| `pipeline/simulation_scene.py`, `config/experiments/simulation_objects.yaml` | Cached local meshes, finite plane-aligned table, support height, footprint checks, actual settling, state serialization and immutable observation. |
| `experiments/simulated_object_cloud.py` | Existing ObjectScene accepts an omitted camera, explicit calibrated finite table and optional settling cancellation. Existing callers retain defaults. |
| `pipeline/observation.py`, `dual_inference.py`, `experiments/fixed_camera_grasp.py`, `native_multimode_grasp.py` | Shared source-specific cloud validation. Simulation carries genuine scene/model/mesh hashes. Real captures retain camera round-trip checks and calibration provenance. |
| `pipeline/dual_artifacts.py`, `paired_artifacts.py` | Existing endpoint and schedule construction extracted before ROS packet construction. Hardware exports keep their schemas, fields, interpolation and checks. |
| `pipeline/simulation_plan.py` | Existing inference/selector invocation, six-stage reviewed prefix, immutable source binding, same linear sampling for preview and execution. Old right-full rows are adapted to canonical mode only at the schedule boundary. |
| `pipeline/simulation_executor.py` | Named actuator targets, common physics steps, arrival/timeouts, actual object/contact metrics, continued hold, streamed evidence. |
| `pipeline/simulation_processes.py`, `simulation_ui.py` | Offline inference workers; single Viser server; guarded lifecycle and generation-based stale result rejection; cancellation and physics-owner joins. |
| `placement/workflow.py`, `placement/ui.py` | Simulation-specific session preparation and reusable panel without fabricated captured environment pixels. Existing real-capture validation stays separate. |
| `tests/test_dual_mujoco_simulation.py`, `test_dual_mujoco_web.py` | Geometry, provenance, actual physical stepping, static/physics isolation, reference equivalence, failure/retention, cancellation and actual WebSocket lifecycle. |

No existing MPC implementation, dependency pin, third-party tree, robot joint
order, learned ranking, IK tolerance, collision gate, golden or physics tuning
was changed. Pure existing helpers in ROS-named modules have no ROS runtime
requirement; they are reused only numerically.

The shared schedule retains all eight original stages. Simulation binds and
renders only the first six and never dispatches release/return. Single-side
simulation uses shared linear interpolation with existing endpoint/duration
policy; the existing single-side hardware quintic packet remains unchanged.
Paired lift uses the original five common IK waypoints. Actual relative-wrist
reference and error are logged without claiming continuous rigidity.

## Data and reproducibility

Each Confirm creates a new isolated `attempt_NNNN` directory. `requested_scene.json`
records object, XY, table-relative Yaw, derived world RPY, fixed scale, pose policy and seed. `scene_config.yaml`, `scene.mjb`,
`initial_state.npz`, `settled_state.npz` and `physics_scene.json` retain physical
provenance, requested/settled poses, actual inertia and canonical model identity.
`object/object.npz` contains only exactly 8,192 world-metre points; `object.json`
identifies `simulation_complete_mesh` and genuine source hashes. `robot_state.json`
contains canonical named q56 in radians and explicitly says simulation.

HUGS and UOP read the same immutable cloud. HUGS retains its existing preprocessing,
mirroring, checkpoints, availability/Top-K and WXYZ decoding. UOP retains its
2,048 distinct-point sampling, plane fitting and one-shot semantics. Physics
truth is kept outside their observation. Candidate filtering/planning receives
a separate `live_scene()` containing calibrated robots/table and no object solid;
static checks never modify execution data.

`simulation_plan.json` is a distinct transport-neutral reference. Before execution,
source hashes, the reviewed JSON and complete settled qpos/qvel/ctrl are checked.
`physics.jsonl` streams actual state, contacts and reference tracking at 25 Hz;
`execution_report.json` distinguishes completion, scheduled hold success, current
retention, retention failure and a drop after earlier retention. Reset/exit writes
cancellation evidence and preserves prior output.

Exact geometry and physics assumptions are documented in the operator guide:
finite 1.2 × 1.4 m unmeasured tabletop, original plane acceptance status, snapshot
initial arms, 100 g unmeasured objects, DexGraspBench contacts and original position actuators,
4 ms physics / 25 Hz reference, original arrival tolerances, 3 cm / 0.02 m/s
retention thresholds and contact on **each** active side throughout Hold.
No assumptions were presented as accepted laboratory calibration.

## Actual model and physics trials

Final model evidence:
`output/simulation/final_trials_20261124_152931/trial_report.json`.
Every trial freshly compiled/settled a local mesh and ran the actual checkpoint;
no fixture predictions were substituted. The default Mug uses uniform scale 0.14;
the larger paired trial uses 0.28, within the recorded range.

| Trial | Model / selector / eligible arms | Actual result |
| --- | --- | --- |
| 0 | Native 200k / hierarchical / left | Selected `left:right_full:c0269`; physical stage arrival timeout. |
| 1 | Native 200k / hierarchical / right | Selected `right:right_full:c0263`; physical stage arrival timeout. |
| 2 | Native 200k / hierarchical / paired-only | Selected `both_full:c0440`; both sides physically commanded; stage arrival timeout. |
| 3 | Native 200k / hierarchical / auto | Selected `left:right_full:c0235`; all six stages and scheduled Hold completed, followed by 100 continued-hold samples before explicit cancellation. **Grasp failed:** minimum lift about 0.000115 m, below 0.03 m. |
| 4 | Native 200k / hierarchical / paired-only, larger Mug | Selected `both_full:c0440`; physical arrival timeout. |
| 5 | Right-full 100k / legacy / auto | Selected `left:c0003`; physical arrival timeout. |
| 6 | Right-full 100k / hierarchical / auto | Selected `right:c0000`; physical arrival timeout. |

Trial 3 also ran actual UOP and produced P00 plus **two valid UOP candidates**.
P00 and a UOP candidate were both confirmed with relative rotations and
`translation_status: undetermined`. See its `placement/uop_result.json`, checkpoint
hash, `inference.npz`, immutable confirmation files and invocation/log.

Actual native browser evidence:
`output/simulation/ui_native_final_01/web_integration_report.json`. The process
started through the documented CLI, confirmed a scene, ran real native inference,
scrubbed the lift phase without physics, explicitly executed through Hold,
continued physical stepping, reset and performed fresh inference on a second
scene with the **same server and process**. Physical grasp was correctly reported
as unsuccessful. Placement browser evidence is in
`output/simulation/ui_development_01/`: actual UOP, P00 confirmation, Reset and
second scene in the same server. Earlier development results are retained and
are not substituted for final model evidence.

The deterministic physics tests separately cover left/right/paired full six-stage
execution, common-step targets, passive targets, persistent hold and a deliberately
unsuccessful stationary grasp. Browser tests use explicitly labeled synthetic
model results with real scenes/physics to exercise cancellation and lifecycle
independently of learned feasibility. These fixture tests are not claimed as
model-generated grasp successes.

## Regression evidence

Canonical full-suite invocation:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  .conda/bin/python -m unittest discover -s tests -v
.conda/bin/python -m compileall -q src tests
ruff check src tests script
# Ruff format --check was applied to every changed Python file.
git diff --check
```

Initial unchanged-test regression: 619 tests, 88 skips, passed. Final expanded
regression: **633 tests, 88 skips, passed**. Ruff, all 17 changed Python files' format
checks, compileall and whitespace checks passed. Full logs and results are copied to
`output/simulation/verification_20261124/` after completion. Installed Ruff is
`/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff`.

The maintained quick gate was run with inherited ROS Python paths cleared:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH= PYTHON_BIN=.conda/bin/python \
  ADA_GRASP_CTRL_RELEASE_GATE_ROOT=/tmp/ada-simulation-quick-current-isolated \
  bash script/run_release_gate.sh quick
```

It fails the existing Shadow expected-classification inventory. A fresh `git
archive HEAD` baseline in `/tmp/ada-simulation-baseline-0f3e292` reproduces the
**identical per-sample classification inventory**, not merely equal aggregate
counts: 68 successes, 4 failures, 21 invalid initializations, 7 degraded, 0 execution
errors. Classification digest:
`9a96b3bb4972b36c71914d49bdabca6167c95459a0d4e95a8ec07fedcdd65dbb`.
Comparison evidence is
`output/simulation/verification_20261124/quick_gate_comparison.json`.
The gate stops at Shadow, so later hand gates are not claimed as completed.
An initial non-isolated gate attempt hit inherited ROS Python 3.12 Pinocchio;
both that failed invocation and the corrected isolated invocations remain logged.
No golden inventory or tolerance was changed.

## Remaining limitations

- All real integration grasps reported here failed physically or timed out. One
  real model trial and the browser trial completed the full sequence, correctly
  without declaring grasp success. No parameters were tuned to hide these results.
- Several alternative default asset orientations failed the existing settling
  check. The editor reports such scenes and permits retry. The curated manifest
  validates available geometry, not arbitrary-pose stability.
- Finite table boundaries, object mass/inertia/friction and simulation actuator
  gains are assumptions/source defaults, not measured hardware identification.
- UOP Reset invalidates its panel; an already-running bounded UOP subprocess may
  finish into its original attempt. It cannot replace current scene results.
- Rendering/selection is single-object. Placement stops at orientation confirmation.
  No camera, tactile control, MPC, placement execution or physical-hardware
  validation is part of this change.

Suggested next step: use the default Mug scene to review the same physical failure
and tracking reports interactively, then compare other valid scenes. Any later
controller/physics identification or threshold change is separate work.

## Settling revision after interactive feedback

The user reported frequent false settling failures. Diagnostics on ten actual
manifest default scenes showed that the old six-component velocity norm mixed
metres/second and radians/second, and required 100 consecutive samples below
0.001. Only Mug and Bowl met that condition in five seconds. For example, Can
varied by about 0.4 mm / 0.1 degrees over the final half-second, but never passed.
Evidence: `output/simulation/settling_diagnostic_20261124_154443/diagnostics.json`.

Interactive simulation now explicitly opts into a pose window: maximum 3 mm
position span and 2 degree orientation span over 0.5 s with continuous table
support, timeout 10 s on the simulation clock. Every physics step contributes;
matching window endpoints alone do not suffice. Instantaneous linear/angular
speeds remain separate diagnostics. Legacy ObjectScene configurations without
`scene.settling` keep the old behavior. Contact, mass, friction, off-table,
penetration, numerical, robot-state and grasp-success checks are unchanged.

Actual scene validation at
`output/simulation/settling_validation_20261124_154701/report.json` passed settling
for all ten defaults in 0.52–4.58 s. Nine passed complete scene validation; Book
was still rejected by the independent 2 mm table-penetration check. This is
initialization coverage, not a new grasp-success claim. Earlier checkpoint
trial results above remain historical evidence from before this revision.


## Tabletop editor revision

The user requested only XY and yaw so objects retain a stable table support.
The editor now exposes Object, X, Y, Yaw and Seed, with no Roll/Pitch/Scale
controls. Each manifest entry records a fixed table-relative support quaternion;
the world orientation is table rotation × normal-axis yaw × support rotation.
Size is the original default scale. Full-mesh support determines Z, and XY is
clamped to retain 5 mm edge clearance with the realized XY reflected in sliders.
Randomization preserves these constraints; explicit legacy RPY/scale requests
remain supported offline.

`output/simulation/support_pose_20261124_155149/report.json` validates all ten
presets at yaw 0/90/180/270. Earlier Book failures were actual collision penetration
(about 14 mm), with only about 0.56 mm visual/collision discrepancy. Book's fixed
preset uses table-local roll 90 degrees, which passed the same penetration and
settling checks. This changes initial support, not the mass/friction/contact model.

The running viewer must be restarted to load these Python/UI changes; refreshing
an existing browser tab alone does not load the new implementation. No existing
user server was stopped.


Longer-duration validation found slow rotation in the first Bottle/Container
presets. Bottle now uses table-local roll 90 degrees; Container uses a supporting
face computed from its actual collision hull. Refinement evidence:
`output/simulation/support_refine_20261124_155730/report.json` and
`output/simulation/container_window_20261124_155936/report.json`.
Container remains supported but has millimetre-scale drift; the ten-second probe
in `output/simulation/container_refine_20261124_155807/report.json` measured about
10.5 mm translation for one yaw of that face. This limitation is retained rather
than changing friction, contacts, mass or immobilizing the free body.

Final fresh evidence:
`output/simulation/tabletop_validation_20261124_160034/report.json`: 30/30 scenes
passed support, footprint and penetration checks for five additional seconds.
The focused suite passed 14 tests after final preset refinement, including
sustained physics for all ten objects at two yaw/XY combinations and geometry
checks at table corners. The full suite before the last two preset refinements
passed 643 tests (88 skips), including all four actual 8089 WebSocket tests; the
user port had become free, so the tests used their own processes. Ruff check,
compileall and diff checks pass. Repository-wide format check still reports eight
pre-existing unrelated files; modified Python files pass formatting checks.
Logs: `/tmp/ada_tabletop_final_tests.log`, `/tmp/ada_tabletop_full_tests.log`,
`/tmp/ada_tabletop_final_validation.log`.

## Authorized squeeze and timed-stage revision

The user requested squeeze ×2 and timed progression after comparing the latest
DexGrasp Bench run (192/300 successes on 2026-11-24). The simulation pipeline now
extrapolates wrist translation and fingers as `grasp + 2 * (squeeze - grasp)`
before IK/selection, retaining squeeze rotation. Original predictions are kept;
`simulation_candidates.json` records the adjusted candidates and provenance.
The selected adjusted candidate drives preview, squeeze, lift and hold. Limits
are validated normally, without clipping. Hardware inference/export is unchanged.

New simulation plans record `stage_completion: timed`. Each phase runs its
existing interpolation duration plus a fixed 0.3 s endpoint dwell, with no
arrival-dependent wait. Arrival errors remain diagnostics. Existing collision,
actual joint overshoot, cancellation and numerical checks still stop execution.
Old saved plans lacking the field retain arrival semantics. The dual pipeline
YAML exposes `simulation.squeeze_extrapolate_ratio` (2.0) and
`simulation.stage_completion` (timed); gains, contact numerics, friction, mass,
pregrasp extrapolation and physical success criteria were not changed.

Fresh replay evidence:
`output/simulation/timed_squeeze_20261124_161939/report.json`. Reused the immutable
observations, physical settled states and selected candidate identities from
`live_20261124_153927/attempt_0020` and `live_20261124_160227/attempt_0003`, both
previous squeeze arrival timeouts. Each was reselected/replanned with ratio 1
and 2 under timed progression, without new inference. All four ran through lift
and hold with no execution error. None met the existing physical grasp-success
criterion. Paired ratio 2 retained a minimum 0.01117 m lift during hold, versus
0.00280 m at ratio 1; these are individual diagnostic trials, not a success-rate
claim. Earlier replay harness runs at 161702/161825 failed setup validation and
are not accepted physical trials.

Validation: 18 focused tests passed, including exact per-side extrapolation,
pre-selection application, no raw mutation/double application, timed advancement
when arrival is false, retained collision aborts and legacy timeout behavior.
Full suite: 647 tests, 94 skips, one failure because an existing mocked launcher
test probes the user's occupied 8089 port. That single test passes with only its
port probe mocked free; no server was stopped. Ruff check, changed-file format,
compileall and diff checks pass. The clean quick gate reproduces the existing
Shadow inventory mismatch exactly (68 success, 4 failure, 21 invalid, 7 degraded,
0 execution errors), per-sample digest
`9a96b3bb4972b36c71914d49bdabca6167c95459a0d4e95a8ec07fedcdd65dbb`.
The gate remains failed against the checked-in expected inventory and stops before
Allegro/LEAP; no baseline was changed. Initial gate invocation inherited ROS's
Python 3.12 path and failed Pinocchio import; rerunning with `PYTHONPATH=src`
produced the valid comparison. Evidence/logs: `/tmp/ada_timed_gate_comparison.json`,
`/tmp/ada_timed_tests.log`, `/tmp/ada_timed_full_tests.log`,
`/tmp/ada_timed_quick_clean.log`, `/tmp/ada_timed_replay_v3.log`.

Restart the simulation Python process and confirm a fresh scene to use the new
settings; existing in-memory plans retain their reviewed behavior.


## Historical gravity environment and restored scale control

The user authorized matching the prior single-arm gravity environment and restoring
Scale. The dual simulation now defaults to `simulation.gravity_mode: object_only`:
MuJoCo gravity disabled globally, with a world-frame object-COM force of mass ×
−10 N/kg applied before every settling/execution/continued-Hold step. The current
100 g objects receive 1 N downward. Arm bias compensation is disabled. Legacy
ObjectScene callers without this setting retain world gravity and their original
optional bias compensation. Gains, contact numerics, friction and masses are unchanged.
Initial/settled NPZ snapshots now include applied generalized/body forces; execution
verifies these fields when present and keeps older snapshots readable. Reports
record the effective gravity environment.

Without hand gravity, exact-zero default finger targets produced about 1e-7 rad
of negative soft-limit undershoot during initialization. Default snapshot-based
open hands now start 1e-4 rad inside finger limits in object-only mode. This avoids
clipping measured states or relaxing validation. The user's permission to relax
lower bounds was not needed; explicit initial states and source limits are preserved.

Scale is restored with per-object manifest bounds. XY, yaw and scale edits use the
same support orientation, recompute exact height and fit the complete footprint.
Object changes/reset restore default scale; seeded randomization retains default
sizes. Non-default sizes still undergo settling and penetration validation.

Fresh physical replay evidence:
`output/simulation/object_gravity_20261124_163622/report.json`. Both recorded single
and paired candidates were replanned with squeeze ×2 after fresh settling under the
new gravity environment, then completed all six timed stages without execution
errors. Neither met the physical success criterion. The paired case retained a
minimum lift of 0.01110 m; this does not establish a grasp-success improvement.

Validation for this revision: 21 focused tests passed, including actual free fall
at 10 m/s² for both 100 g and 200 g objects, no robot external torque, per-step
force restoration, force snapshot tampering rejection, Scale edits/reset and
all ten support presets under continued physics. Actual resized Mug scenes at
scale 0.10 and 0.20 both passed settling/penetration validation in
`output/simulation/scale_gravity_20261124_163910/report.json`.
Full suite: 650 tests, 94 skips, one known mocked-launcher failure due to the
user's occupied 8089 port. No user server was stopped. Ruff, changed-file format,
compileall and diff checks passed. The maintained quick gate again stopped at
the existing Shadow expected-inventory mismatch; its 100 per-sample classifications
exactly match the earlier baseline (68 success, 4 failure, 21 invalid, 7 degraded,
0 execution errors). Evidence: `/tmp/ada_gravity_gate_comparison.json`,
`/tmp/ada_gravity_scale_tests_final.log`, `/tmp/ada_gravity_scale_full_tests.log`,
`/tmp/ada_gravity_quick.log`. Restart Python and confirm a fresh scene to load
these force and UI changes.


## DexGraspBench contact and sampling alignment (2026-11-24)

Interactive defaults now select the explicit `dexgraspbench` contact profile and
25 Hz reviewed references. All geoms use friction `[0.8, 0.02, 0.0001]`, condim 4,
elliptic cone, impratio 10 and noslip iterations 2. Robot geoms alone receive
solref `[0.005, 1]` and solimp prefix `[0.5, 0.99, 0.0001]`; object/table stiffness
stays at source values. Physics uses implicitfast at 4 ms, ten steps per target.
The reference is `HUGS-DexGraspBench/src/util/hand_util.py` (`MjHO._add_hand`,
`_set_friction`, `control_hand_with_interp`). Historical `sharpa_native_v3` and
standalone `source` profiles retain their existing behavior. Old plans without
`reference_hz` retain 100 Hz; incompatible timestep/rate pairs fail before stepping.

Object-only gravity, CRX paths, stage durations and fixed endpoint dwell, squeeze
ratio 2, gains, torque limits, masses, inference checkpoint and success thresholds
are unchanged. Bench's dummy-arm paths, density-based masses and evaluation
criteria are not imported, so no Bench success-rate equivalence is claimed.

Diagnostic replay evidence:
`output/simulation/bench_contact_20261124_165541/report.json`.
Both saved single/paired candidates were reapplied to freshly settled scenes,
reselected and replanned, then completed all six phases without motion/numerical
failure. Neither passed scheduled Hold: minimum retained lift was 0.029199 m
(single) and 0.007461 m (paired). These are two diagnostic replays, not fresh
inference or a measured success rate; the 0.03 m threshold was preserved.

Validation for this change: 22 interactive simulation tests pass; four focused
contact-profile tests pass (two optional snapshot tests skip). The full canonical
unit suite reports 652 tests, OK with 88 skips. Ruff check, compileall, changed-file
format check and `git diff --check` pass. Repository-wide format checking still
reports eight pre-existing unrelated files.

Fresh quick gate root: `/tmp/ada_bench_gate_20261124_v1`. It stops at the existing
Shadow expected-inventory mismatch (68 success, 4 failure, 21 invalid, 7 degraded,
0 execution errors); Allegro/LEAP are not reached. All 100 classifications exactly
match the prior local baseline, verified in `/tmp/ada_bench_gate_comparison.json`,
digest `9a96b3bb4972b36c71914d49bdabca6167c95459a0d4e95a8ec07fedcdd65dbb`.
No golden inventory was changed. Restart the interactive simulation process and
confirm a new scene to load the new defaults; existing running attempts retain
their compiled physics and reviewed reference rate.


## Table-aligned placement previews (2026-11-24)

New simulation placement sessions persist the actual world table plane in
`metadata.preview_table_plane_world`. UOP candidate previews rotate the original
observed cloud around its centroid, then shift only Z to place the lowest point
on that plane. XY and orientation are preserved; P00 restores the exact source
cloud. The display shift is recorded as `preview_translation_world_m` when
confirming. Saved sessions reproduce it without inference. Old saved sessions
without the plane and captured-camera workflows retain centroid-only previews.

This adjusts display geometry only. It does not simulate settling, certify
stability, enforce finite-table bounds, or create robot placement trajectories;
`translation_status` remains `undetermined`. Regression tests cover lifting out
of penetration and lowering from a gap on horizontal/tilted tables, repeated
switching, immutable P00, serialization, invalid planes, and actual simulation
session preparation.

Replay of nine cached UOP candidates from three user simulation attempts under
`output/simulation/placement_20261124_172130` verified this display correction in
a fresh output root: `/tmp/ada_placement_ground_bsw2yaez/report.json`. Before
correction, gaps ranged from -0.05108 m (penetration) to +0.16867 m (floating).
After correction, every candidate's minimum signed table distance was below
1e-12 m in magnitude. No model rerun or source attempt modification was needed.

Validation: 15 placement-selection and 22 interactive-simulation tests pass.
Full suite: 654 tests, 94 skips, one existing launcher test fails because the
user's 8089 server is occupied. That launcher test passes in isolation with only
the port probe mocked free. The running user server was preserved. Ruff check,
changed-file format check, compileall and diff whitespace checks pass. No physics,
controller or classification behavior changes, so no quick trajectory gate rerun
was needed for this display-only change.
