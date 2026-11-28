# Dual Sharpa feedback in the original grasp preview

The original `script/run_dual_grasp_pipeline.py` accepts `--tactile` with
`--environment hardware` for both preview-only and operator-reviewed grasp execution.
It reuses the simulation's ten-finger `ForceOverlay` in the existing 8089 viewer.
No new grasp pipeline or hardware driver is started. Existing invocations without
`--tactile` retain planned playback. Export-only and placement modes reject this
viewer option.

With the existing calibrated RGB-D/hardware launch already running on domain 87:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
env -u LD_LIBRARY_PATH -u PYTHONPATH \
  ROS_DOMAIN_ID=87 ROS_LOCALHOST_ONLY=1 \
  .conda/bin/python script/run_dual_grasp_pipeline.py \
  --environment hardware --ros-domain-id 87 \
  --selector hierarchical --arms auto --grasp-model native_multimode \
  --multi-camera --preview-only --tactile \
  --output-root "output/dual_live/tactile_$(date +%Y%m%d_%H%M%S)"
```

Select a current object ROI when prompted. The pipeline still needs an object for
capture, fusion and inference; empty-table tactile diagnostics do not count as a
fresh end-to-end grasp run.

For actual grasp execution, use the same command with a fresh output root and omit
`--preview-only`. The existing hardware stack must already be configured for motion;
the retained read-only launch cannot execute. `--arms auto` may select one hand;
add `--grasp-mode both_only` to require a feasible paired candidate. Review the new
8089 plan and press Enter in the pipeline terminal to start the existing complete
grasp/lift/hold/release/return cycle. The force display does not change that cycle
or make its release into a placement operation.

The viewer defaults to **Live measured pose + tactile**. Turning that off enables
planned playback for review and hides live force arrows. When the runner starts
execution, the viewer automatically returns to measured pose and locks out planned
playback for the rest of that run. The same viewer and two passive readers remain
alive during either the single-arm sender or paired hardware executor. The normal
runner cleanup closes them after completion, cancellation or failure.
Object and environment clouds remain fixed at their captured poses during motion;
only robot pose and tactile forces are live.

Missing, stale, ambiguous or model-incompatible feedback hides force geometry;
missing measured pose retains the last displayed pose with an explicit unavailable
status. No joint limits are clipped or loosened. Tactile availability does not gate
grasp execution or feed its controller. A reader dropout clears the display; a
viewer process exit retains the runner's existing owned-service failure handling.
The reader cleanup stops only its own subscriber children, not hardware drivers.

## Data and force meaning

- Subscribe to both hands' five atomic `TactileFrame` topics, named hand/arm joint
  feedback, and the existing robot description. There are no command publishers,
  driver restarts or SDK device owners.
- Official channels are right 0–4 and left 5–9 in pinky/ring/middle/index/thumb
  order. The earlier dual adapter erroneously expected right channels on the left;
  this integration fixes that identity check and tests crossed identities.
- Each side uses its official hand URDF. Live description kinematics must match
  the calibrated run model. Measured 56-joint FK places forces in world coordinates.
- F6 channels 0–2 provide the provisional object-on-finger resultant. The existing
  overlay receives the opposite finger-on-object vector: orange reaction, blue
  object force, green normal, red contact. Rotation transforms force vectors;
  translation affects contact positions only.
- Magnitudes use presumed N. SDK geometry remains an explicitly labeled diagnostic
  candidate. Operator direction checks do not establish calibrated magnitudes or
  contact-point accuracy. First-contact placement is a visualization, not an
  equivalent full six-dimensional wrench.
- Raw F6 and fresh resultant magnitudes remain visible when contact geometry is
  missing. Such fingers have no arrow; there is no fingertip-origin fallback.
- Hardware aggregate F6 cannot reconstruct the simulation's sum of original
  compressive normal loads, **ΣFn**. It remains unavailable for valid contacts;
  only explicit fresh empty-contact blocks provide a zero no-contact load.
- `tactile_state.json` is a bounded latest-state diagnostic updated once per second,
  separate from the execution packet and its review identity. Its `execution_started`
  field records the viewer's latched switch to live execution display; it does not
  establish successful motion or grasping.

## Execution integration validation

The execution display integration is covered by synthetic single-/paired-arm CLI
tests and the actual viewer loop with synthetic measured feedback. These verify the
Enter gate, packet-review mismatch rejection, viewer lifetime across transport,
unchanged reviewed packet bytes, automatic return from planned playback to measured
pose, stale-pose force clearing and subscriber cleanup. No hardware command or live
grasp was performed during these software checks. The earlier physical press
evidence below establishes passive visualization; subsequent operator execution
is recorded separately at the end of this document.

Validation: full unit discovery ran 785 tests (695 passed, 90 skipped); the focused
dual tactile set passed all 13 tests. Global Ruff lint, Python compilation,
touched-file formatting and `git diff --check` passed. The existing LEAP Tac3D
quick example completed 100 inputs (87 successes, 6 failures, 7 invalid starts,
0 degraded solves, 0 execution errors), under
`output/examples/leap_tac3d/20261128T044641Z-485067/`. These are software/simulation
checks, not physical grasp acceptance.

## Physical verification, 2026-11-27

Existing read-only hardware/camera launch PID 469559 and hand nodes 469574/469575
were reused on domain 87. No robot motion was commanded.

The fresh original-entry attempt
`output/dual_live/three_camera_tactile_20261127_v1` passed the initial robot
snapshot and captured three cameras, but stopped before inference because the
previous baseball ROI was empty. The fresh RGB image confirms the central object
was removed. That run remains failed and is not reported as successful inference.

Tactile verification then used the **original dual viewer**, with explicitly
labeled historical scene assets from
`output/dual_live/three_camera_table_corrected_20261127_1952`, in a separate
`output/dual_live/tactile_saved_scene_20261127_v1` diagnostic directory. Only the
scene was historical; joints and tactile feedback were live. The original source
run was not modified.

Evidence in that diagnostic directory:

- `tactile_validation_samples.json` and `operator_press_samples.json`: two bounded,
  partly overlapping recording windows.
- `operator_validation.json`: 63 unique snapshots; all ten fingers had valid F6
  transport in all 63. All ten fingers produced valid world-force observations at
  some point. Measured pose was valid in 62/63 snapshots; one stale-joint sample
  correctly withheld force geometry.
- During pressing, the left/right index magnitude maxima were approximately
  15.84/15.15 presumed N; thumb maxima were 6.88/4.90. These are observed diagnostic
  values, not reference-load calibration measurements.
- The operator explicitly confirmed the tested force directions looked correct.
  No claim is made about a comprehensive per-axis or calibrated force test.
- HTTP 8089 returned 200. At the operator's request, the preview and both owned
  ROS readers were stopped afterward; the existing hardware launch was preserved.

Validation: the full unit discovery ran 781 tests, with 96 skips and one failure
because the concurrent live preview occupied port 8089. After closing the preview,
that test and the focused tactile/telemetry/display/dual-live set were rerun:
71 tests, 70 passed, 1 skipped. No unresolved test failure remains. Global Ruff
lint and Python compilation pass; touched files pass formatting. Unrelated existing
formatting differences were left unchanged. Simulation physics, controllers,
thresholds, placement work and camera calibration acceptance were not changed.

## Operator execution follow-up, 2026-11-27

After removing `--grasp-mode both_only`, the operator reported success. Run
`output/dual_live/both_tactile_execute_20261127_210310` used hardware/domain 87,
`grasp_mode: auto` and `tactile: true`. It selected `left:right_three:c0127`
(`left_three`) and completed all eight action phases. Its latest tactile diagnostic
records measured display, execution started, live 56-joint pose and both readers
connected. This supports single-arm execution with the dual tactile display;
`object_lifted` and `object_held` remain null in the execution report.

A subsequent run, `output/dual_live/both_tactile_execute_20261127_210436`, selected
`both_full:c0440` and reached the paired hardware executor, but ended with
`Joint-limit violation` after 5,974 published frames and six completed stages.
The latest tactile diagnostic also records live measured display and both readers
connected. The executor recorded a fault and software stop handling; this run is
not a successful paired cycle. The joint-limit failure remains unresolved and no
limits, control policy or placement code were changed for this display commit.
