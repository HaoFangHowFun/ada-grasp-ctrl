# Phase 3: simulated grasp-to-place execution

Add `--placement-execute` to the existing placement-aware simulation command:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
env -u LD_LIBRARY_PATH -u PYTHONPATH .conda/bin/python script/run_dual_grasp_pipeline.py \
  --environment simulation --placement-aware --placement-execute \
  --selector hierarchical --arms auto --grasp-model native_multimode \
  --output-root "output/simulation/grasp_to_place_$(date +%Y%m%d_%H%M%S)"
```

Open <http://localhost:8089>. Confirm Scene, choose P00 or a cached UOP orientation,
move Destination X/Y, then Confirm Placement Target. Review the complete **Phase**
dropdown and **Playback (%)** slider. Only **Execute Grasp-to-Place** starts
physics. After Retreat, final joint targets and real-time physics continue until
**Reset Scene**. Target controls stay locked during physical execution. Target
edits before execution invalidate the plan; reconfirmation reuses source inference.

`--placement-aware` alone still provides the original endpoint-only preview.
Normal grasp simulation, standalone placement selection and hardware/mock flows
retain their existing behavior. The modifier requires simulation and placement-aware
mode; paired candidates remain excluded.

## Preparation and schedule

The Phase 2 selector is unchanged. Only its selected winner gets the additional
endpoint solves and sampled path checks. A failure reports the stage and stops
preparation; it does not select another candidate. Edit the target and confirm again.

The accepted Place joints and adjusted squeeze reference are reused exactly.
Placement execution completes a **0.15 m world-Z Lift, then Reorient, then
Transport, then Place**. The Lift endpoint retains the source squeeze orientation
and held fingers; its existing bounded inward-offset fallback still applies.
Configure this separately with `simulation.placement_lift_height_m` (default 0.15).
Normal grasp-only and hardware flows retain their original 0.05 m setting.
Reorient uses the Lift endpoint's actual FK wrist position and the accepted Place
orientation. Its IK starts near the accepted Place arm branch, retains held
fingers, and uses the existing bounded seed set and contact checks. Transport
then ends at Preplace with that same target orientation. Reorient fixes position
at the endpoint only: joint-space interpolation may move the wrist between
endpoints. Every segment still goes through the sampled robot/table path checks.
Preplace offsets Place by 0.15 m along the upward table normal. After Release 2,
Retreat Lift raises the open hand by 0.10 m along that normal. Retreat then offsets
this **raised wrist** by 0.10 m along the source pregrasp-minus-grasp direction
rotated by the confirmed relative placement rotation. A degenerate direction is
an error. These endpoints retain Place wrist orientation and use existing bounded
IK seeds, limits, tolerances and static contact policy. Retreat Lift starts near
the accepted Place arm; Retreat starts near the accepted raised arm.

| Stage | Arm | Fingers |
| --- | --- | --- |
| Above object / Pregrasp / Grasp / Squeeze | Existing endpoints | Existing policy |
| Lift | Source-oriented endpoint, 0.15 m higher | Adjusted squeeze |
| Reorient | Lift wrist position, Place orientation | Adjusted squeeze |
| Transport | Preplace | Adjusted squeeze |
| Place | Exact accepted Place | Adjusted squeeze |
| Settle | Fixed Place | Adjusted squeeze |
| Release 1 | Fixed Place | Original grasp |
| Release 2 | Fixed Place | Original pregrasp |
| Retreat Lift | Place + 0.10 m along upward table normal | Original pregrasp |
| Retreat | Retreat IK | Original pregrasp |

All segments use the shared joint-space linear reference. Durations are recomputed
with the existing policy for the higher Lift; endpoint dwell is unchanged.
Reorient, Transport and Place use the shared duration helper.
Reorient and Transport retain the normal endpoint dwell. Place immediately enters explicit
Settle. Settle, both Release segments, Retreat Lift and Retreat have no extra endpoint dwell.
There is no five-second Hold, extra Preplace stage, or return home.

New settings live under `simulation.placement` in the dual pipeline YAML:
`preplace_height_m: 0.15`, `settle_s: 0.5`, `release_1_s: 0.5`,
`release_2_s: 1.0`, `retreat_lift_height_m: 0.10`, `retreat_lift_s: 2.0`,
`retreat_distance_m: 0.10`, and `retreat_s: 2.0`.
The plan records resolved durations. Timed progression quantizes completion to
the existing reference interval (40 ms by default); the physics timestep stays
4 ms. Physics, gains, contact tolerances and gravity settings are unchanged.

## Preview and physical behavior

Preview never steps physics. The immutable source cloud stays at source through
Squeeze. During Lift/Reorient/Transport/Place it follows the displayed wrist using
`T_hand(q) @ inverse(T_source_held)`. From Settle onward it stays at the confirmed
target. Each scrub computes from the original cloud; there is no frame cache or
accumulated transform. The target cloud remains visible for comparison.

This is a kinematic preview with ideal attachment, not a physical prediction.
The static scene checks robots and the calibrated table plane, without an object
collision solid. Reference sampling uses existing arm/hand command-step spacing;
it does not certify continuous or Cartesian paths. Joint interpolation does not
describe a straight Cartesian path.

Execution uses the existing MuJoCo executor and displays measured robot joints
and the actual object mesh pose. The object remains a free body, with no weld,
teleport or pose overwrite. Release is open loop; the fixed Settle interval does
not guarantee table support or stability. Existing runtime collision, joint-limit,
numerical and cancellation protections remain active.

`COMPLETED` means the schedule finished. There is **no placement success metric**.
The legacy Hold result is `null`, labelled `not_evaluated_no_hold_stage`; normal
grasp-only Hold evaluation is unchanged.

## Artifacts and validation

Each historical `attempt_NNNN/placement/target_NNNN/` retains the Phase 2 target,
selection and endpoint records, plus `simulation_plan.json` and `path_checks.json`.
The simulation reference records canonical joints, source hashes, revision,
parameters, timing and the exact accepted Place joints. `preparation_s` measures
winner-only preparation, excluding cached inference and selection. Execution
adds `physics.jsonl` and `execution_report.json` in that same target directory.
Reset preserves history and cancels the current owner. No ROS packet is emitted.

Run the actual HUGS/UOP browser smoke with:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .conda/bin/python script/smoke_placement_aware.py --placement-execute \
  --output-root output/simulation/grasp_to_place_validation_NEW
```

The driver refuses an occupied port, reviews a translated P00 target and a
nonidentity UOP target if available, scrubs the full reference, explicitly executes
the final prepared plan, and retains actual completion or failure diagnostics.

### Local integration evidence, 2026-11-26

The evidence in this section predates the requested 0.15 m initial Lift and
separate Reorient updates and used the original 0.05 m Lift.

`output/simulation/phase3_validation_20261126_v4/` contains a fresh default Mug
run with real native HUGS and UOP, the browser invocation/report, fully loaded
Chrome screenshots (`P01_preview.png`, `execution_end.png`), and all target
artifacts. The source observation and settled snapshot hashes remained unchanged.

Both targets translated the source centroid by +0.02 m in world X:

- P00: Phase 2 found 16 placement-feasible candidates, but the selected winner's
  Preplace had no collision-free accepted IK branch. Preparation stopped and
  execution remained disabled. No automatic reselection occurred.
- P01: a nonidentity UOP rotation selected `left:right_full:c0235`, with 9
  placement-feasible candidates. Winner-only preparation took 0.216 s on this
  machine. Browser scrubbing covered all 11 stages before explicit execution.
  The roughly 65.63 s nominal schedule completed through Retreat, continued
  physics afterward, and Reset returned to editing.

The object physically lifted and moved. Release 1/2 ended without hand/object
contact, but Retreat recontacted and displaced the cup; the final image shows
the actual object tilted beside the blue nominal target. This is an observed
physical limitation of this open-loop sequence, not successful-placement evidence.
No physics parameters or checks were relaxed. `scheduled_hold_success` remains
null and no placement evaluator was added.

The preceding completed run in `phase3_validation_20261126_v3/` also retains
`independent_audit.json`: accepted Place joints were reused, the ideal Place
cloud agreed with the target transform within floating-point precision, and
passive actuator targets remained unchanged throughout the real trajectory.

The maintained quick gate ran in `/tmp/phase3-quick-gate-20261126/` and stopped
at the known historical Shadow expected-inventory discrepancy. Its 100 fresh
classifications were 68 success, 4 failure, 21 invalid initialization, 7 solver
degraded and 0 execution errors. Every classification matched the retained
pre-change evidence, digest
`9a96b3bb4972b36c71914d49bdabca6167c95459a0d4e95a8ec07fedcdd65dbb`.
The comparison is `phase3_validation_20261126_v3/quick_comparison.json`.
No golden data was changed; the gate did not advance to the other hands.

Final validation: **675 tests, 88 skipped, zero failures** with the maintained
`unittest discover -s tests -v` command in the isolated Ada environment
(`/tmp/phase3-full-tests.log`). Coverage includes geometry, preserved initial
durations, configurable release timing, sampled collision failure, winner-only
IK, passive targets, preview/reference agreement, actual free-body stepping,
browser invalidation/retry/explicit execution/Reset, and existing workflows.
`ruff check src tests script`, compileall, changed-file formatting and
`git diff --check` passed. Whole-tree formatting retains the same eight unrelated
pre-existing differences (`/tmp/phase3-format.log`).

Implementation is confined to the simulation CLI/lifecycle, winner preparation
in `pipeline/placement_execution.py`, the simulation schedule variant in
`pipeline/placement_schedule.py`, and the existing executor/reference/panel.
Phase 2 selection and the hardware shared-schedule contract are unchanged.

### Higher initial Lift update

The first higher-Lift update completed a 0.15 m source-oriented Lift before
the then-combined Transport/reorientation. The height is independent of the 0.15 m Preplace
offset and is recorded as `placement_lift_height_m` in each new plan. Restart
the viewer and reconfirm the target to prepare the updated reference.

Validation: 19 focused placement tests passed, including the default/custom
height and unchanged shared configuration; lint, compile and diff checks passed.
`output/simulation/phase3_higher_lift_6o9ncqee/` retains a fresh replay using the
saved real HUGS/UOP winner and restored settled MuJoCo snapshot. P01's Lift wrist
rose 0.149946 m, with a 0.003134 rad orientation residual within the existing
0.01 rad IK tolerance. The full physical schedule completed through Retreat.
P00 retained its earlier Preplace IK failure. No selector or physics policy was
changed, and completion is not a placement-success classification.

### Separate Reorient correction

The reported `transport: nominal path collision at 0.780` was reproduced from
`grasp_to_place_20261126_113019/attempt_0003/placement/target_0002` with winner
`left:right_full:c0298`. The direct Lift-to-Preplace joint interpolation struck
the passive right J2 link with the left thumb (about 1.6 mm penetration at the
first failing sample). Both endpoint poses were valid; this was not an IK
convergence failure. Merely choosing a Preplace branch near Lift moved the
collision to Place, and the nearby Place branch penetrated the table.

The corrected sequence inserts the separately checked Reorient endpoint at the
Lift wrist position with Place orientation, then moves to Preplace and descends
to the original accepted Place joints. Selection, passive-arm targets, collision
thresholds and the original Place solution are unchanged. Old eleven-stage
references remain readable; newly prepared references contain twelve stages.

`output/simulation/phase3_reorient_fix_03gnzzxp/` retains the exact failing
target's corrected schedule, all path checks and a fresh actual MuJoCo replay
from the saved settled snapshot. Preparation took 0.316 s; every sampled segment
passed and physical execution completed through Retreat. The earlier Mug/P01
case also completed with the new stage (`phase3_higher_lift_2wd06yq6/`). These
are execution-completion observations, not placement-success labels. Restart
the viewer and reconfirm the scene/target; the Phase dropdown should now include
`reorient` between `lift` and `transport`.

Validation: 20 focused placement tests and the full suite of **676 tests with
88 skips and zero failures** passed (`/tmp/phase3-reorient-full-tests.log`), as
did lint, compile, changed-file formatting and diff checks. The fresh maintained
quick gate again stopped at the known Shadow inventory discrepancy; all 100
classifications exactly matched the retained pre-change evidence, recorded in
`phase3_reorient_fix_03gnzzxp/quick_comparison.json`.

### Lift the open hand before withdrawal

The right-hand failures in
`grasp_to_place_20261126_114634/attempt_0003/placement/target_0002` and
`target_0018` were reproduced with winner `right:right_full:c0202`. The rotated
withdrawal vector descended 0.024675 m along the table normal, putting the
right little finger about 10.84 mm into the table. Accepted IK solutions were
therefore rejected by the unchanged contact policy.

New plans insert `retreat_lift` after `release_2`: lift the open hand 0.10 m
along the upward table normal, then apply the original withdrawal vector from
that raised origin. The default interpolation lasts 2 s, without extra dwell.
Both stages retain original pregrasp fingers and Place wrist orientation.
The target cloud stays fixed during this preview segment; physical execution
continues to use the actual free object.

`output/simulation/phase3_retreat_clearance_npj_ytdc/` contains fresh preparation
and MuJoCo replay of both failed targets from their saved settled snapshots.
Both passed all sampled path checks and completed the full thirteen-stage
schedule through Retreat. The retreat endpoint is now 0.075325 m above Place
along the table normal. Candidate identity and the accepted Place joints were
retained; no collision thresholds or physics settings changed.

The old eleven- and twelve-stage references remain readable without rewriting
their parameters. Restart the viewer and reconfirm the scene/target to use the
new sequence; the Phase dropdown should include `retreat_lift` before `retreat`.

Validation: 20 focused tests passed; lint, compilation, changed-file formatting
and diff checks passed. The full suite ran 676 tests with 96 skips and one
failure caused by the user's occupied port 8089 in an unrelated mocked launcher
test (`/tmp/phase3-retreat-lift-full-tests.log`). That single test passed when
its port probe was isolated in memory; no production checks or repository tests
were changed, and the user viewer was preserved. Browser tests requiring 8089
were skipped. The fresh quick gate retained the same historical Shadow inventory
discrepancy, with all 100 classifications matching the pre-change evidence
(`phase3_retreat_clearance_npj_ytdc/quick_comparison.json`).
