# Phase 2: placement-aware endpoint preview

This opt-in simulation workflow selects one left **or** right grasp whose existing
grasp checks and nominal placement endpoint IK both pass. It uses the existing
8089 viewer. It never executes a grasp, lift, transport, release or return.

For full trajectory review and physical simulation, explicitly add
`--placement-execute`; see [Phase 3 grasp-to-place execution](grasp-to-place-execution.md).
The endpoint-only behavior described here remains the default without that modifier.

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
env -u LD_LIBRARY_PATH python3 script/run_dual_grasp_pipeline.py \
  --environment simulation --placement-aware --selector hierarchical \
  --arms auto --grasp-model native_multimode \
  --output-root "output/simulation/placement_aware_$(date +%Y%m%d_%H%M%S)"
```

Use a fresh output root and open <http://localhost:8089>. An occupied port fails;
do not stop an unrelated viewer. `--arms left` and `--arms right` restrict the
eligible side. `--grasp-mode both_only`, hardware/mock environments, and combining
this mode with `--placement-only`, `--preview-only` or `--export-only` are rejected.
The existing right-full model also supports this mode with the legacy or
hierarchical selector. UOP installation overrides remain `--uop-root` and
`--uop-python`; no installation or training is performed.

## Operator workflow

1. Edit the object scene and click **Confirm Scene**. The existing physical
   settling produces one immutable observation and initial robot state.
2. **P00 — Original Pose** is selected. Optionally click **Run UOP-Net once**,
   then choose any returned orientation. Failure or zero planes preserves P00.
3. Drag the tabletop XY arrow/plane marker or use **Destination X/Y (m)**.
   The synchronized controls describe the observed-cloud centroid's destination,
   not the wrist or mesh origin. Marker Z follows the measured tabletop plane.
   A marker outside the table is clamped with the sliders updated accordingly.
   An off-table object footprint disables confirmation and displays an error;
   the confirmed object goal is never silently moved.
4. Click **Confirm Placement Target**. Only this starts source HUGS inference and
   selection. Review preparation, counts and errors in the viewer. Collapse the
   Scene Builder folder or scroll the controls to reach the placement controls.
5. On success, **Endpoint: Grasp / Place** switches the static robot state. Grasp
   means the selected adjusted terminal squeeze posture before lift. Place uses
   the exact accepted IK solution with the same fingers. The passive robot stays
   at its settled initial state. Green is the original cloud; blue is its target
   transform. There is no target mesh or interpolated trajectory.
6. **Edit Placement Target**, a new orientation, or XY edits invalidate the old
   preview. Confirm again to reevaluate feasibility. No-feasible targets remain
   editable without rebuilding the settled scene. **Reset Scene** clears the
   derived state and preserves the prior attempt directories.

UOP results and raw HUGS predictions are cached for that unchanged observation.
Every confirmation reevaluates grasp/placement feasibility. Revisions prevent
obsolete results from replacing an edited target. An inference failure remains
attributable to its attempt; Reset Scene starts a new inference attempt.

The review is labelled:

> Endpoint IK preview — transport path / placement stability not validated

## Geometry and feasibility

With immutable observed centroid `c`, selected relative rotation `R`, and resolved
target centroid `c_star`, the full rigid transform is
`Delta_T = [R, c_star - R @ c; 0, 1]`. Points use
`(X - c) @ R.T + c_star`. Switching never accumulates transforms. P00 retains
identity rotation but can translate; standalone `--placement-only` retains its
original exact P00 restoration and orientation-preview behavior.

Height is resolved against the effective upward tabletop plane using the
**compiled MuJoCo object collision mesh vertices in their settled world pose**.
MuJoCo's mesh centering/scaling and geom transforms are included. Linear support
extrema match the convex hull extrema, so this does not use sparse cloud minima.
The clearance policy is **0 m nominal contact**, distinct from the scene builder's
1 mm initialization drop clearance. There is no destination settling or stability
claim. The footprint uses the union of visual and collision vertices projected
into the existing finite table frame. Full geometry is used only for these
simulation geometry checks, never as additional learned-model inputs or a target
mesh display.

The held reference is the existing simulation-adjusted squeeze wrist and its
matching canonical fingers (`grasp + ratio * (squeeze - grasp)`, currently ratio
2). The place wrist is `Delta_T @ T_world_hand_source_held`, in the same world
hand-flange convention used by existing IK. It is never computed from the lifted
wrist. This is a rigid no-slip approximation.

The optional gate runs only after the existing pregrasp/grasp/squeeze IK, hand
limits and static contact checks pass. It uses the same bounded arm IK, tolerances
(1 mm / 0.01 rad), the held arm seed and existing seed set. Among accepted place
solutions passing the existing static squeeze contact policy, it chooses the
nearest to the held arm configuration. All colliding solutions reject the
candidate. This branch
choice does not affect grasp ranking: mode priority, grasp effort and existing
ties are preserved. Paired modes are disabled at the selector's eligibility
boundary; the native model's raw outputs are retained. A placement-infeasible
mode allows the existing next-mode fallback.

The existing arm IK solver itself has no collision check. Since the authorized
2026-11-26 update, the placement gate checks every successful IK solution in the
existing independent static robot/table scene before accepting it. This reuses
the existing infinite calibrated table plane and contact policy: arm/table
penetration greater than 0.01 mm is rejected; hand/table penetration retains the
existing 10 mm tolerance. Inter-arm and self-contact checks are also preserved.
The scene has no object collision solid. Transport paths, swept volumes,
object-related placement collisions and dynamic stability remain unverified.
Existing grasp checks and grasp effort ranking are unchanged. No ROS runtime or
execution packet is used. Rejected branches record `place_endpoint_collision`
and their contact diagnostics; successful IK alone no longer accepts the branch.
Restart an already-running viewer and confirm a new scene to load this change.

## Artifacts and review

Within `attempt_NNNN/`:

- `candidates.json` and model prediction directories retain the unchanged raw
  HUGS results; `simulation_candidates.json` records adjusted squeeze provenance.
- `placement/` retains the existing immutable snapshot, UOP results, candidate
  list and historical orientation-only selections. Those selections still say
  translation is undetermined; they are not relabelled execution poses.
- `placement/target_NNNN/placement_target.json` records source/target identities,
  revision, rotation, requested/realized XY, centroid Z, full transform, table,
  geometry provenance and explicit preview/contact assumptions.
- `placement_selection_report.json` contains evaluated/skipped diagnostics,
  original grasp-stage reasons, each attempted place pose/joints/residuals,
  grasp effort, counts and the selected candidate. Rejection counts are per
  seed attempt; candidate counts are separate. Skipped modes are not infeasible.
- `placement_endpoint_preview.json` records the accepted grasp/place full joint
  vectors and canonical names, passive state, wrist transforms, held stage,
  source/target hashes and preview-only markers.
- `placement/target_state.json` identifies the current revision and selected
  preview, or explicitly marks editing, failure or cancellation. Historical
  target directories remain available for inspection after invalidation/reset.

The JSON endpoints can be inspected without rerunning inference. The existing
saved placement viewer continues to replay orientation-only artifacts; it does
not interpret these new endpoint records as execution plans.

## Implementation and validation

Implementation baseline: clean commit `9510564`. Integration touches the
existing CLI and simulation lifecycle, adds `placement/target.py`,
`placement/feasibility.py`, `placement/target_ui.py` and
`pipeline/placement_preview.py`, and adds an optional gate to the shared
single-arm/hierarchical selectors. No learned weights, physics, IK thresholds,
third-party dependencies, existing execution behavior or golden data changed.

Original 2026-11-24 IK-only evidence (before the collision filter):
`output/simulation/placement_aware_validation_20261124_1816/` contains the CLI
invocation, screenshot, reports and independent artifact audit. The default Mug
scene produced P00 plus two real UOP orientations. Native HUGS produced 30
candidates; 20 single-arm candidates were evaluated and 10 paired candidates
were explicitly skipped. P00 at XY `[0.467745, -0.444090]` had 17 grasp/place
feasible candidates (9 left, 8 right). P01 at XY `[0.507745, -0.444090]` had
17 grasp-feasible and 16 placement-feasible candidates (8 left, 8 right).
Both selected `left:right_full:c0235` with unchanged grasp effort.

Accepted placement residuals were approximately `3.05e-13 m / 1.89e-12 rad`
(P00) and `1.53e-16 m / 1.41e-16 rad` (P01). Independent reload/FK checks matched
the recorded wrist and passive joints; transformed collision support lay on the
table to floating-point precision. Source observation and settled-state artifacts
were unchanged. No execution plan, physical execution log or transport export
was generated. These results establish endpoint software behavior, not physical
grasp or placement success.

Reproduce the actual model/UI smoke with:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .conda/bin/python script/smoke_placement_aware.py \
  --output-root output/simulation/placement_aware_validation_NEW
```

The script refuses an occupied 8089, forbids ROS imports, exercises P00 and one
real UOP candidate at two XYs, checks immutable source artifacts and the absence
of execution artifacts, and captures the actual viewer with local headless Chrome
when available. It uses the existing Viser wire decoder from the repository tests.

An additional real Chrome mouse-drag check is retained in
`output/simulation/placement_browser_drag_20261124_v3/`: dragging the X arrow
changed the destination slider to approximately `0.5237 m`, updated eight target
revisions, and kept UOP `not_run`. `before.png`, `after.png` and
`browser_drag.json` retain the interaction evidence. The first two CDP probes
selected Chrome's background extension tab and timed out; the successful check
explicitly selected the page target. The viewer and model smoke were unaffected.

Validation commands:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .conda/bin/python -m unittest test_placement_aware test_placement_aware_web -v
env -u LD_LIBRARY_PATH PYTHONPATH=src:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl .conda/bin/python -m unittest discover -s tests -v
.conda/bin/python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
git diff --check
```

The final full suite ran **664 tests with 88 skips and no failures**, including
native eligibility, inference caching, no-feasible recovery and Reset during
preparation. Lint, compile and changed-file formatting pass. Whole-tree formatting
retains eight pre-existing unrelated differences. Final test log:
`/tmp/placement-final-full-tests.log`; formatting log: `/tmp/placement-format.log`.
Validation-owned viewers and browser processes were stopped; port 8089 is free.

The maintained Shadow quick example was also run in fresh output roots. The
first attempt inherited ROS Python paths and failed to load Pinocchio. Repeating
with both `LD_LIBRARY_PATH` and `PYTHONPATH` unset produced all 100 outputs:
68 success, 4 failure, 21 invalid initialization, 7 solver degraded, 0 execution
errors. The checked-in expected inventory still rejects these classifications,
matching the documented pre-existing Shadow discrepancy; no baseline was changed.
Clean run: `/tmp/placement-aware-quick-clean-20261124/shadow/shadow-validation`.
All 100 scientific classifications exactly match the retained baseline, verified
in `output/simulation/placement_aware_validation_20261124_1816/quick_comparison.json`;
digest `9a96b3bb4972b36c71914d49bdabca6167c95459a0d4e95a8ec07fedcdd65dbb`.

## Placement collision filter update, 2026-11-26

The minimal update passes the existing static scene into `PlacementGate`, applies
`check_static_contacts` and `forbidden_contacts(..., phase="squeeze")` after each
successful IK solve, and excludes colliding branches before choosing the nearest
held-state solution. A candidate with no surviving branch is rejected; the
existing selector continues to other candidates/modes with unchanged grasp effort.
Diagnostics distinguish `place_endpoint_collision` from IK rejection and retain
the contacts for every checked branch. The IK optimizer and collision thresholds
are unchanged.

The actual saved source from `output/simulation/placement_aware_20261126_100638`
was replayed without rerunning HUGS or UOP and without changing any source files.
Fresh evidence is under
`output/simulation/placement_collision_validation_20261126_duz3dsuv/`.

| Target | Previous candidate (rejected for collision) | New candidate | Placement-feasible candidates |
| --- | --- | --- | --- |
| 0047 | `right:right_full:c0236` | `left:right_full:c0235` | 12 |
| 0057 | `right:right_full:c0236` | `right:right_full:c0202` | 3 |
| 0095 | `right:right_full:c0236` | `right:right_full:c0202` | 3 |

All three accepted endpoints were independently checked again against the static
contact policy and had no forbidden contacts. The original candidate files,
observation and settled-state hashes remained unchanged. Focused regressions
cover both arm sides, rejecting a closer colliding solution, all branches failing,
skipping collision checks after failed IK, candidate ranking and mode fallback.

Validation: 22 focused tests passed; the full suite ran 666 tests with 88 skips
and no failures (`/tmp/placement-collision-full-tests.log`). Ruff lint,
changed-file formatting, compileall and `git diff --check` passed. The replay
report is `replay_report.json` in the fresh evidence directory above.
