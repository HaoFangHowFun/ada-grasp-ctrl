# P3 dynamic Lift handoff — 2026-11-13

> Subsequent physical milestone: the operator-run `active_20261113_132624` completed Grasp, 15 cm Lift, Hold, Release and Return. See [physical milestone record](mpc-p3-real-grasp-lift-milestone-2026-11-13.md). The offline implementation results below retain their original scope.

## Result

Implemented a narrow hardware `mpc_active && native_complete` handoff. Lift now starts from a new measured 28-joint snapshot, computes a world **+Z 0.15 m** endpoint with the current flange orientation, and holds the measured 22 hand joints. The original 0.01 rad arm start tolerance is unchanged.

The recorded `active_20261113_111440` completion state reproduces the old `right_J1` failure and passes the new handoff. This is **offline evidence, not a physical lift result**. No drivers were started, no real robot commands were sent, and no commit/push was performed.

## Implementation

- `src/ada_grasp_ctrl/pipeline/dynamic_lift.py`: small JSON-based geometry helper and derived-goal validation. Reuses `CrxSharpaAdapter`, `lift_target`, `HandGeometry`, `static_check`, `motion_duration` and `check_start`. No physics stepping or grasp simulation.
- `src/ada_grasp_ctrl/experiments/right_ros_sender.py`: after successful native completion, waits for both new arm and hand feedback messages, records names, receipt timestamps, message-header stamps and topic sources, and creates the dynamic goal.
- `src/ada_grasp_ctrl/ros_execution/bridge.py`: independently verifies native completion, immutable source packet and derived goal; recomputes the same geometry review before creating command publishers, then checks fresh actual feedback against the captured start.
- `src/ada_grasp_ctrl/mpc/live.py`: forwards the existing native Python executable to the sender helper.
- `tests/test_dynamic_lift.py`, `tests/test_dynamic_lift_ros.py`: recorded-state geometry and synthetic sender/receiver regression.
- `tests/test_mpc_p3_receiver.py`: fixes comparison of unspecified ROS CDR padding. Every decoded message field and every numeric array byte must still match exactly; no numeric tolerance was relaxed.
- `docs/mpc-p3-operator-2026-11-12.md`: updates the handoff description and points to these operator steps.

The original controller, native completion/force policy, MPC streamer, contact adapter, drivers, and original plan validator/IK helper are unchanged by this task. Existing uncommitted linear-stream work is retained.

### Data flow and review

1. Sender receives action success with `Ours native_complete:`. Timeout, abort, solver failure or an invalid success message cannot enter the handoff.
2. Both feedback receipts must be newer than that result. Snapshot freshness uses the existing feedback timeout and 0.15 s receipt coherence; canonical order is six arm joints then 22 hand joints. Receipt time is not claimed to be sensor acquisition time.
3. A read-only native-Python subprocess loads the hash-verified selection/model/calibrated world frame. FK → current world pose → world +Z 15 cm with unchanged orientation → existing bounded local IK seeded by measured arm joints.
4. The derived Lift retains measured hand angles. Existing joint limits, sampled known robot/table collision gates, and mesh-to-table clearance are checked along the joint segment. Sampling reuses the existing table-clearance evaluator's 0.002 rad resolution. It rejects an intersecting or decreasing hand/table gap beyond the existing 1 mm allowance.
5. The immutable original packet is not rewritten. A versioned `ada_dynamic_lift_v1:` JSON extension in the trajectory header carries the captured start, derived segments, source hash and review identifier. This prefix is application metadata, not a TF coordinate frame; all pose calculations remain in world coordinates. Ordinary goals retain their original empty-header contract.
6. Receiver accepts the extension only after its own successful native completion and in the expected phase. It independently rebuilds the review from the same source packet/snapshot and requires exact equality. A hash alone is not accepted as path validation.
7. Sender continues processing feedback while planning. Receiver also processes feedback/cancellation while independently planning. Both check fresh actual positions against the captured start, and receiver checks again before first publication. A mismatch logs the complete actual/expected/delta vectors and aborts; no automatic retry or old-target fallback.

Planning does not refresh the snapshot's timestamps. Its age is validated at capture; fresh feedback is compared again after computation. Lift hand start tolerance remains the existing strict 0.02 rad rather than ignoring finger drift.

Hold must retain the new Lift endpoint, and Release must open at that new arm position. These derived endpoints and their starts are updated together to avoid a second stale-plan mismatch. Hold duration, zero-joint release policy and original return destination remain unchanged; motion durations are recomputed where the start changed. They use the existing conventional receiver trajectory, including its original quintic timing; no new force control or velocity planner is introduced.

## Recorded-state result

Input plan: `output/grasp_pipeline/p3_plan_20261113_111347/grouplive/export/targets.json`.

Input result: `output/mpc_p3/active_20261113_111440/receiver/20261113T191508_497475Z_478626/result.json`.

Artifacts: `output/dynamic-lift-handoff-20261113/{snapshot.json,handoff.json,summary.json}`. Snapshot positions are the recorded actual `final_positions`; the explicit offline clock is synthetic and is never used as live feedback.

| Observation | Result |
| --- | --- |
| Old preview start | Reproduces `Robot moved since preview: right_J1` |
| New start | Exactly equals recorded measured 28 joints |
| Requested world translation | `[0, 0, 0.15]` m |
| Flange world Z | −0.024254763 → +0.125745237 m |
| Endpoint position residual | 1.99e−13 m |
| Endpoint orientation residual | 6.96e−13 rad |
| Held finger angles | All 22 unchanged |
| Sampled path | 172 samples pass joint/known-geometry/table checks |
| Minimum sampled hand/table gap | 0.020175217 m |
| Geometry computation in recorded run | Approximately 1.42 s, excluding subprocess startup |
| Real command publications | 0 |

`handoff.json` includes original-preview joint deltas, current/desired/actual world poses, IK seed/solution, held-hand source, model/table/calibration provenance, validation scope, derived phase targets/durations and review identifier. The original endpoint helper's `path_checked=false` / `collision_checked=false` fields remain its own limited claim; the added `validation.sampled_known_geometry=true` describes the checks actually performed by the handoff helper.

## Focused validation

Run from the repository root. Native tests use the maintained Python environment:

```bash
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl OMP_NUM_THREADS=4
export ADA_DYNAMIC_LIFT_PACKET="$PWD/output/grasp_pipeline/p3_plan_20261113_111347/grouplive/export/targets.json"
export ADA_DYNAMIC_LIFT_SNAPSHOT="$PWD/output/dynamic-lift-handoff-20261113/snapshot.json"
.conda/bin/python -m unittest discover -s tests -p test_dynamic_lift.py -v
```

Result: **7 passed**. Covers recorded old/new start, FK/IK and hand mapping, malformed/stale/incoherent snapshots, unchanged start tolerances, invalid model/IK/path/table/limits, source/review binding, and independent subprocess reproduction.

ROS contract tests use mocked I/O; they do not initialize the ROS graph or operate hardware:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export ADA_DYNAMIC_LIFT_REVIEW="$PWD/output/dynamic-lift-handoff-20261113/handoff.json"
/usr/bin/python3 -m unittest discover -s tests -p test_dynamic_lift_ros.py -v
/usr/bin/python3 -m unittest discover -s tests -p test_mpc_p3_receiver.py -v
/usr/bin/python3 -m unittest discover -s tests -p test_right_ros_receiver.py -q
```

Results: **4 + 4 + 7 passed**. Covers successful sender/receiver handoff, held fingers, missing native completion/review, start races, cancel/stale/path failures, action rejection, and identical baseline/passive command fields.

Additional existing regression commands, using native Python with the environment above:

```bash
export ADA_RIGHT_FULL_PREVIEW="$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1"
export ADA_P3_PACKET="$PWD/output/grasp_pipeline/physical_baseline_20261112_194720/grouplive/export/targets.json"
.conda/bin/python -m unittest discover -s tests -p 'test_mpc*.py' -q
.conda/bin/python -m unittest discover -s tests -p 'test_pipeline*.py' -q
.conda/bin/python -m unittest discover -s tests -p test_grasp_pipeline.py -q
.conda/bin/python -m unittest discover -s tests -p test_right_ros_transport.py -q
```

Results: MPC **44 passed / 4 ROS skips**; pipeline **28 passed**; grasp pipeline **16 passed**; transport **11 passed**. Ruff on changed Python files, `python3 -m compileall -q src tests`, and `git diff --check` pass.

The first broad `test_right_ros*.py` invocation under system ROS Python could not import the native-only transport dependency `mujoco`; receiver tests passed there, and transport was then run successfully under `.conda/bin/python`. The initial CDR-byte comparison failed on padding only; decoded messages and numeric bytes were identical, and the corrected exact-field test passes.

## Limits and remaining physical validation

- Geometry checks cover the existing known right robot and measured table model, with the selector's existing contact tolerances. The incomplete observed object/environment is not a certified collision solid. Continuous collision clearance and unknown obstacles remain unverified; the report retains explicit false flags.
- The Lift endpoint is world-vertical; the intermediate path remains the existing joint-space interpolation, not a new Cartesian straight-line planner.
- Hold/release/return retain existing policies; this change does not newly certify their collision paths.
- Dual-scene dynamic validation is not implemented; packets requiring passive-arm scene state fail explicitly before Lift.
- If the arm/hand changes beyond the original tolerances while planning, the action stops with measured deltas. No tolerance relaxation or automatic replanning is performed.
- The physical 15 cm lift and object retention remain operator tests. Earlier downstream commissioning gaps in the linear-stream report are unchanged.

## Operator-only execution

These are commands for a **new operator-run trial**, not commands executed during implementation. Use the existing running camera/drivers. First capture/select the current object:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
export ROS_DOMAIN_ID=87
export P3_PLAN="$PWD/output/grasp_pipeline/p3_plan_$(date +%Y%m%d_%H%M%S)"
python3 script/run_grasp_pipeline.py \
  --environment hardware --ros-domain-id "$ROS_DOMAIN_ID" \
  --lift-height-m 0.15 --export-only --output-root "$P3_PLAN"
export P3_TARGETS="$P3_PLAN/grouplive/export/targets.json"
```

Then execute the existing P3 entrypoint:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
/usr/bin/python3 -m ada_grasp_ctrl.mpc.live \
  --targets "$P3_TARGETS" --controller-mode mpc_active \
  --execute --hardware-enable \
  --output-root "output/mpc_p3/active_$(date +%Y%m%d_%H%M%S)"
```

Enter `mpc_active` at the existing operator prompt. Successful native grasp now proceeds through dynamic 15 cm Lift, Hold, Release and Return. The existing Ctrl+C/action cancellation path is unchanged. New diagnostics are `sender/dynamic-lift-{request,result}.json`, sender `dynamic_lift_review` events, and corresponding files/events in the receiver Lift action directory.
