# Opt-in single-arm RRT-Connect placement

This experimental simulation mode now prioritizes completing grasp-to-place while
retaining the object. New plans use `execution_policy: complete_with_warnings`.
Collision, clearance, contact-change and tracking diagnostics remain recorded but
no longer stop execution. The default non-RRT workflow and older saved plans
without this policy retain their existing behavior. This is simulation only.

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
env -u LD_LIBRARY_PATH -u PYTHONPATH .conda/bin/python script/run_dual_grasp_pipeline.py \
  --environment simulation --placement-aware --placement-execute \
  --placement-planner rrt_connect \
  --selector hierarchical --arms auto --grasp-model native_multimode \
  --output-root "output/simulation/rrt_placement_$(date +%Y%m%d_%H%M%S)"
```

Restart the Python viewer process to load changes; Reset Scene does not reload
Python modules. Confirm the scene and placement target, review the approach, then
click **Execute Grasp-to-Place**. Lift, transport, descent and retreat are planned
from measured state when their stages begin. Reset cancels the existing execution
owner. No extra approval screen, physics thread or server is introduced.

## Current execution policy

The sequence remains Approach, Pregrasp fingers, Grasp, Squeeze, Lift, Transport,
Place, Settle, Release 1, Release 2 and Retreat. All arm motion through Settle keeps
the original Squeeze finger command. It never substitutes measured finger angles
for the squeeze preload. Release retains the original finger targets/durations.
Retreat returns the six active arm joints to their captured initial angles.

- Lift targets the lowest compiled collision-mesh point at **15 mm** above the
  calibrated table. An actual gap below **5 mm** is a warning, not a gate.
- Transport RRT still checks a **5 mm planning-only table clearance** for the
  object, both arms, J6 and hand collision meshes. Both measured and commanded
  hand shapes are checked. The physical table and dynamics are unchanged.
- Actual object/table penetration beyond **2 mm**, unexpected contacts, retreat
  recontact and arm tracking error are warnings. They do not veto motion.
- RRT still tries collision-checked paths. If it fails or times out, the executor
  uses a direct joint interpolation to a finite, bounded goal. Reports retain the
  failed search status and set `fallback_applied` and `execution_method` explicitly;
  this path is not collision-certified. Descent uses a sampled direct connector
  with the same explicit fallback rather than a new RRT detour.
- Transport IK tries the actual arm, nominal accepted Place branch and existing
  bounded seeds, within the stage planning budget, preferring a clear endpoint.
  If no clear endpoint exists, a valid accepted IK endpoint may still be used
  with collision warnings. If IK rejects all poses, Carry/Place use the previously
  accepted Place joint endpoint with Squeeze fingers; rejected Lift IK holds the
  current valid command. The rejected solver vector is never commanded. The
  immutable requested object pose and rejected diagnostics remain in the report.
- A brief absence of hand contact does not stop carrying. Planning reuses the
  last measured attachment until actual contact returns. Loss is confirmed only
  after **0.5 s** continuously without active-hand contact **and** at least
  **20 mm** object displacement from that attachment. These are cleanup criteria,
  not placement accuracy or success thresholds.
- Confirmed loss interrupts the current carrying stage and opens/retreats. A
  missing initial grasp also opens/retreats without inventing an attachment.
  Cleanup ends with `cleanup_completed: true`, `pipeline_completed: false` and
  `grasp_evidence.outcome: dropped` or `not_acquired`. The viewer shows this outcome.
  `status: completed` means the execution owner finished its assigned sequence;
  it does not imply a successful placement.

Cancellation, invalid/nonfinite/out-of-bounds commands, passive-command mutation,
MuJoCo numerical warnings and excessive actual joint-limit overshoot remain hard
stops. Target/revision identity checks remain active. No gains, friction, gravity,
4 ms physics step, 25 Hz reference, assets or golden baselines were changed.

## Implementation and diagnostics

`motion_planning/rrt_connect.py` implements deterministic joint-space search.
`motion_planning/placement_collision.py` uses separate `MjData` with actual compiled
geometry; planning never moves the live object. `pipeline/rrt_placement.py` owns
stage generation and the explicit completion policy, while `simulation_executor.py`
remains the only physics owner. The existing placement scheduler and UI are reused.

The collision checker retains its original thresholds and contact policies.
Pregrasp/Grasp/Squeeze match the shared legacy contact rules. Release uses the
legacy Squeeze active-hand/object policy. Carry whitelists measured/intended hand
pairs; Retreat checks separating contacts. Under the new policy these observations
are advisory at execution and select RRT paths when possible. MuJoCo 3.6 ambiguous
zero mesh/box distances are resolved using actual collision-mesh/top-plane geometry.

Optional settings under `simulation`:

```yaml
rrt_connect:
  extension: 0.20
  resolution: 0.02
  max_iterations: 5000
  timeout_s: 3.0
  shortcut_s: 0.05
  seed: 4
```

IK, search and validation share the deadline. Calls are not forcibly interrupted;
a deadline overrun is diagnosed and can select the explicit fallback. Checks are
discrete, not continuous collision certification. Contact presence alone does not
certify a stable grasp. A linear fallback can scrape the table or drop the object.

`simulation_plan.json` records policy, preparation warnings and source identity.
`dynamic_paths.json` records search failures, fallback paths, IK attempts, actual
attachment, warnings and grasp evidence. `execution_report.json` separates full
pipeline completion from cleanup; `physics.jsonl` retains per-sample object pose,
contacts, actual joints and commands. Repeated warnings are aggregated by phase
and code rather than generating unbounded duplicate messages.

## Completion-policy validation, 2026-11-27

Fresh final replays: `output/simulation/rrt_completion_20261127_v3/`.
Each case restores its attributed original settled snapshot and actuator targets,
and reruns preparation and actual free-object MuJoCo physics. Cached selections
are reused; inference is not rerun. Each case has a `provenance.json` with source
and driver hashes. Historical outputs are not overwritten.

| Historical source | Final replay | Outcome |
| --- | --- | --- |
| `rrt_placement_20261127_105351/attempt_0001`, left c0169 | `low_lift_1` | Full pipeline; contact retained to Release despite 1.566 mm actual lift clearance |
| `rrt_placement_20261127_105351/attempt_0005`, left c0108 | `low_lift_5` | Dropped during Transport; opening/retreat completed |
| `rrt_placement_20261126_153127/attempt_0001`, left c0235 | `release` | Full pipeline; contact retained to Release |
| `rrt_placement_20261126_155203/attempt_0001`, left c0235 | `descent` | Full pipeline; former descent collision rejection used explicit fallback |
| `placement_aware_20261126_100638/attempt_0001`, right c0236 | `right` | Dropped during Transport; opening/retreat completed |

No case stopped at a collision/clearance/tracking gate while still holding the
object. **Three of five** completed the complete placement sequence with contact
retained to Release; **two of five** dropped and completed cleanup. This small,
selected regression set does not establish a general success rate. Its 40% drop
fraction is too high to claim robust retention. In both dropped cases Transport
used linear fallback; the right case still has colliding IK endpoints despite
trying alternate seeds. A trial extra lift correction did not improve retention
and was removed. None of the cleanup runs is counted as successful placement.

Validation commands and results for this policy:

- Focused RRT, placement and web suite: **39 tests, 3 skipped, no failures**.
  The three web cases require the currently occupied user port 8089.
- Full `unittest discover -s tests -v`: **721 tests, 97 skipped, 1 failure**.
  The sole failure is the unrelated active-launcher test that requires port 8089;
  its log explicitly reports the occupied port. The existing viewer was preserved.
- Canonical Ruff check, compileall and `git diff --check` pass. Canonical format
  check reports the same eight unchanged files as the previous validation; all
  files changed for this task are formatted.
- Fresh quick gate root `/tmp/ada-completion-quick-20261127-v1`: Shadow produces
  all 100 outputs, 68 success, 4 failure, 21 invalid initialization, 7 degraded and
  zero execution errors. The gate fails the same three known expected-inventory
  differences (mug failure→success, tape degraded→success, pizza degraded→failure).
  Allegro/LEAP are not reached after that failure. No golden inventory is changed.
- The physics audit confirms unchanged Squeeze targets for every Lift/Transport/
  Place/Settle sample in all five replays. The three full runs have no samples
  without active-hand contact during those stages. Both drop runs record an
  interrupted Transport followed by Release 1, Release 2 and Retreat.

## Initial evidence, before restoring legacy Pregrasp contacts, 2026-11-26

The sections below record earlier implementations and their validation results;
the current collision policy is described above.

Fresh physics replays in `output/simulation/rrt_validation_20261126_v4/` explicitly
reuse attributed cached selected grasps, settled snapshots and original actuator targets; HUGS/UOP inference
was **not** rerun. `provenance.json` identifies the source selection and its hash.

| Case | Approach | Carry including updated IK | Total planning | Physical outcome |
| --- | --- | --- | --- | --- |
| Left `c0235` | Direct, 0.032 s, 56 queries | RRT, 0.553 s, 447 queries; 18 → 18 points | 0.744 s | Stopped at initial carry: source support gained pairs/increased penetration |
| Right `c0236` | Direct, 0.027 s, 48 queries | Invalid goal, 0.008 s, 3 queries | 0.194 s | Grasp/Squeeze executed; actual-grasp Place IK collides with table |

Both last completed observations had active-hand contact, but neither case
completed placement. The left case genuinely required RRT after rejecting the
direct edge. An earlier right `c0202` case was rejected during pregrasp finger
validation (0.175 mm thumb/object penetration); its evidence remains in
`rrt_validation_20261126_v1/right/`. Full physical Settle/Release/Retreat remains
unverified because the real cases stopped earlier.

Focused tests cover planner ordering/determinism/timeout, directed contact
boundaries, actual attached/static object collisions, fixed joints, live-state
isolation, target binding, missing-grasp failure and cancellation. A real 8089
WebSocket test, with explicitly synthetic stationary grasp endpoints, verifies
pending preview, failure handling and Reset while planning without physics steps.

The full unit suite passed: **704 tests, 88 skipped** (environment/explicit opt-in
fixtures including ROS and model-dependent integrations). The focused planner,
collision and placement regression suite passed 19 tests. Commands used:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  .conda/bin/python -m unittest discover -s tests -v
env -u LD_LIBRARY_PATH PYTHONPATH=src:tests .conda/bin/python -m unittest \
  test_rrt_connect test_rrt_placement test_placement_execution -v
env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHON_BIN=.conda/bin/python \
  ADA_GRASP_CTRL_RELEASE_GATE_ROOT=/tmp/rrt_release_gate_20261126_v1 \
  bash script/run_release_gate.sh quick
```

The maintained quick release gate ran in the fresh directory above and
stopped at three Shadow inventory mismatches: 100 outputs, 72 completed (68 success,
4 failure), 21 invalid initializations, 7 degraded solvers. Allegro/LEAP were not
reached by that gate. No golden inventory was changed.

Canonical `ruff check src tests script`, compile and diff checks pass. Canonical
`ruff format --check src tests script` reports eight unchanged files outside this
change; all changed Python files pass formatting. Unit-suite results and logs are
recorded in `/tmp/rrt_full_tests_final.log`.

## Legacy Pregrasp contact restoration

The subsequent change restores the original contact standards for Pregrasp,
Grasp and Squeeze without changing the arm-only approach's stricter gate.
Twenty focused tests pass, including threshold boundaries, passive-hand/arm
rejection, and identical phase routing during preparation and live execution.

Fresh isolated replays of the user's same selected candidates are saved in
`output/simulation/rrt_legacy_contacts_20261126_v1/`. `attempt_0005` now passes
preparation and physically completes Pregrasp, its finger transition, Grasp and
Squeeze. It later stops at the separate carry source-support check; placement is
not complete. `attempt_0001` still fails Pregrasp finger validation at 10.080 mm
penetration, above even the restored 10 mm limit. No candidate was replaced.

Restart the running Python process to load the change; Reset Scene alone does
not reload the collision checker.

Validation for this update: 20 focused tests pass; lint, compile, changed-file
formatting and whitespace checks pass. The full suite ran 705 tests with 97 skips
and one failure: the unrelated active-launcher test requires a free 8089, which
was occupied by the user's running viewer. That viewer was left running. The
quick gate in `/tmp/rrt_legacy_release_gate_20261126_v1` reproduced the same three
Shadow classification mismatches. Logs are `/tmp/rrt_legacy_contact_tests.log`,
`/tmp/rrt_legacy_full_tests.log` and `/tmp/rrt_legacy_release_gate.log`.

## Initial live separation tolerance

The live source-support guard now accepts changing contact geometry pairs and
increased penetration up to the original 2 mm support limit, only inside the
initial separation region described above. The planner's sampled edge policy is
unchanged. Twenty-two focused tests pass, including actual MuJoCo checks at 1.9 mm
and 2.1 mm penetration and rejection outside the initial region/after separation.

The user's case from `rrt_placement_20261126_145536/attempt_0001` was replayed in
`output/simulation/rrt_source_support_20261126_v1/latest_left/`. It passes the
previous first-cycle source-support failure. Carry then runs for 1.88 simulation
seconds before active-hand/object contact disappears (`live_grasp_lost`); physical
placement remains incomplete. No grasp or target was replaced.

For this update, the full suite ran 707 tests with 97 skips and the same single
occupied-8089 launcher failure. The quick gate again reproduced the same three
Shadow inventory mismatches. Lint, compilation, changed-file formatting and
whitespace checks pass. Logs: `/tmp/rrt_source_support_tests.log`,
`/tmp/rrt_source_support_full_tests.log`, `/tmp/rrt_source_support_gate.log`.

## Retained Squeeze command and 5 mm waypoint

The latest change retains the Squeeze command throughout Lift, Place and Settle,
validates the measured/commanded hand discrepancy, and requires actual separation
before planning Place. It also fixes geometrically disproven zero mesh/box
distance results, without changing the 2 mm support cap or recontact policy.

Fresh cached-selection physics replays are in
`output/simulation/rrt_hold_lift_20261126_v3/`. Their provenance identifies the
source grasps; inference was not rerun. Both cases have exactly unchanged Squeeze
finger references throughout Lift and a kinematic waypoint clearance of 5 mm.
Both lift plans take the direct path, with 2.2 s motion and the existing 0.3 s dwell.

| Replay | Measured result | Stop condition |
| --- | --- | --- |
| Latest left `c0235`, source `rrt_placement_20261126_150622/attempt_0004/placement/target_0002` | Five hand/object pairs remain, but lowest collision vertex is still 0.0574 mm below the table at the end of Lift | Actual-separation gate refuses to plan Place |
| Right `c0236`, source `placement_aware_20261126_100638/attempt_0001/placement/target_0047` | Nine hand/object pairs remain; a transient 0.00159 mm gap is followed by 0.0863 mm penetration | Existing post-clearance recontact guard stops Lift |

Neither completes physical placement. A nominal 5 mm waypoint does not guarantee
5 mm actual clearance under compliant tracking and grasp motion. The first
implementation replay (`v2`) exposed false recontact from zero distance queries:
the left object's actual collision mesh was already 2.472 mm above the table,
with no raw object/table contacts. The final `v3` replay includes the geometric
distance correction; the failures in the table above are actual contact, not
those ambiguous zeros.

Validation: 28 focused tests pass. The full suite (before the final additional
IK/cancellation unit test) ran 712 tests, with 97 skips and the same one failure
because the user's viewer occupies 8089. The viewer was left running. The quick
gate reproduced the same three Shadow inventory mismatches (100 outputs,
68 success, 4 failure, 21 invalid, 7 degraded); Allegro/LEAP were not reached.
Lint, compilation, changed-file formatting and whitespace checks pass. Canonical
format checking still reports the eight unchanged files.

Logs: `/tmp/rrt_hold_lift_tests.log`, `/tmp/rrt_hold_lift_full_tests.log`,
`/tmp/rrt_hold_lift_gate.log`, `/tmp/rrt_hold_lift_gate_differences.json`.
Measured support audits are `/tmp/rrt_hold_lift_v3_left_support_audit.json` and
`/tmp/rrt_hold_lift_v3_right_support_audit.json`. Restart Python to load this change;
Reset Scene does not reload code.

## Uniform 2 mm object/table support tolerance

At the user's request, object/table support is now allowed at every planning and
execution sample within 2 mm penetration. Endpoint-only support, monotonic table
separation and post-clearance recontact restrictions have been removed. The
5 mm waypoint remains, but its measured-state gate now enforces the 2 mm cap
instead of requiring positive clearance. Other collisions, loss of grasp and
hand/object release-separation checks remain active. Dynamic path reports record
the new object/table policy explicitly.

Regression expectations were updated for persistent support, interior contact,
recontact, the waypoint gate and excessive penetration. Tests were paused at the
user's request pending the additional clearance requirement below; the combined
change has now been tested.

## 15 mm lift and 5 mm transport planning clearance

The current flow implements the subsequently approved three phases: a 15 mm
nominal lift, transport to a 15 mm preplace pose under a strict 5 mm planning
plane margin, then a checked direct Place connector using the real table. Actual
object clearance must reach 5 mm before Transport planning. The original Squeeze
finger command remains fixed through all three phases and Settle. Physical table
contact tolerance remains 2 mm, and the raised plane never changes live geometry.

Fresh cached-selection physics replays are in
`output/simulation/rrt_clearance15_20261126_v1/`. Source selections and restored
actuator targets are recorded in each case's `provenance.json`; inference was not
rerun. The latest left case comes from
`rrt_placement_20261126_153127/attempt_0001/placement/target_0002`.

- Left `c0235`: actual lift clearance is 8.502 mm, so the measured gate passes.
  Transport planning takes 1.688 s including IK, and produces a checked two-point
  path; physical transport takes 30.52 s including its dwell under the retained
  velocity/timing policy. It completes Transport, Place and Settle. Release 1 is
  refused before execution because the proposed opening introduces a new
  `left_thumb_DP`/object contact with 0.113 mm penetration. The existing release
  policy is retained; the complete placement pipeline does not finish.
- Right `c0236`: Lift completes; the preplace IK goal is rejected because J4/J5/J6
  intersect the table (up to 77.31 mm penetration at J5). No transport is executed.

An independent replay-path audit samples the left transport edge at the configured
0.02 rad maximum increment, checking both commanded and measured hand shapes.
Its minimum geometry-to-plane clearance is 5.426 mm with no margin violations.
Finger command error relative to Squeeze is exactly zero through Lift, Transport,
Place and Settle. Actual transport nevertheless briefly reaches 0.0744 mm table
penetration, within the allowed physical 2 mm cap, and ends at 7.590 mm clearance.
The margin therefore does not prove friction-free physical transport.

Validation: 30 focused tests pass. The full suite passes 715 tests with 88 skips,
including the real WebSocket lifecycle tests when port 8089 was available. The
quick gate in `/tmp/rrt_clearance15_gate_20261126_v1` reproduces the same three
Shadow classification differences: 100 outputs, 68 success, 4 failure, 21 invalid,
7 degraded; Allegro/LEAP were not reached. No baseline was changed. Lint,
compilation, changed-file formatting and whitespace checks pass; canonical
format checking still reports eight unchanged files outside this change.

Logs: `/tmp/rrt_clearance15_tests.log`, `/tmp/rrt_clearance15_full_tests.log`,
`/tmp/rrt_clearance15_gate.log`, `/tmp/rrt_clearance15_gate_differences.json`.
The geometry/command audit is `/tmp/rrt_clearance15_audit.json`. Restart Python to
load the new code; Reset Scene alone does not reload it.

## Release-only restoration of the original hand contact policy, 2026-11-27

At the user's request, Release 1/2 now delegate intended active-hand/object
contacts to the shared Squeeze policy, as the original non-RRT executor does.
Both planning and runtime remove Release's additional contact-pair whitelist and
monotonic separation requirement. Other collision checks, Carry/Settle contact
rules, the 15 mm / 5 mm clearance flow, and Retreat's original whitelist and
separation rules are unchanged. The Release finger targets and durations are
unchanged. Reports identify this as `release_hand_object_policy: legacy_squeeze`.

The exact earlier Release-blocked left case was replayed in
`output/simulation/rrt_release_regression_20261127_v1/left/`, using the cached
selection from `rrt_placement_20261126_153127/attempt_0001/placement/target_0002`.
It now executes both Release phases and completes Retreat. Compared with the
previous `rrt_clearance15_20261126_v1/left` run, all 1,338 pre-Release samples have
exactly equal robot configurations, object poses and reference commands. Thus the
new Release policy introduces no change to that case's prior physical trajectory.

At the end, active-hand/object contact is absent and object speed is approximately
`2.31e-7 m/s`. Object pose error relative to the confirmed target is 3.155 mm and
0.002469 rad (0.141 degrees). The executor reports `pipeline_completed: true`;
these measured errors are descriptive evidence, not a new placement-success
criterion. The sampled Retreat endpoint is the original arm configuration, with
the existing timed execution and tracking behavior retained.

A separate replay of the newer
`rrt_placement_20261126_155203/attempt_0001/placement/target_0002` is saved in
`output/simulation/rrt_release_policy_20261127_v1/left/`. It reproduces its
pre-existing earlier failure, `place: invalid_descent`, from an
`object_table_non_support` contact with signed distance about `-1.14e-14 m`.
Release is never reached in that case. The right case in the same replay root
also retains its pre-existing invalid preplace IK goal from robot/table collision.
Those rules were not changed by this Release-only update. Every replay records
cached-selection provenance; inference was not rerun and no running user scene
was reset.

Validation: 31 focused tests pass. The complete suite passes 716 tests with
88 skips, including WebSocket lifecycle tests. The fresh quick gate in
`/tmp/rrt_release_policy_gate_20261127_v1` reproduces the same three Shadow
classification differences (68 success, 4 failure, 21 invalid, 7 degraded from
100 outputs); Allegro/LEAP were not reached, and no baseline was changed.
Lint, compilation, changed-file formatting and whitespace checks pass. Canonical
format checking still identifies the eight unchanged files outside this work.

Logs and audit: `/tmp/rrt_release_policy_tests.log`,
`/tmp/rrt_release_policy_full_tests.log`, `/tmp/rrt_release_policy_gate.log`,
`/tmp/rrt_release_policy_gate_differences.json`, `/tmp/rrt_release_policy_audit.json`.
Restart Python to load the update; Reset Scene alone does not reload code.
