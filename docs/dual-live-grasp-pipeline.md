# Live automatic left/right grasp pipeline

Entry: `script/run_dual_grasp_pipeline.py`. Each invocation captures the current fixed camera,
lets the operator select the current object, runs fresh right-full inference on the original and
reflected object clouds, selects one feasible side/candidate, previews the full cycle, and executes
after terminal Enter. The default legacy selector uses the existing weighted CRX joint displacement
cost across eligible sides. An explicit hierarchical option uses configured mode priority first.
There is no fixed object, saved candidate, ROI, or object pose in the default workflow. The pear
capture used during development is test evidence only. The checkpoint/config/metadata references
identify the model, not an object to replay.

## Start the two physical arms, hands and fixed camera

Use a ROS terminal and keep the launch running. Replace an existing single-side launch before
starting this stack; it owns both devices and the fixed camera.

First generate the ROS hand-mount YAML from the same master calibration used by Ada.
Run this in the Ada terminal, choosing a new output path if it already exists:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
PYTHONPATH="$PWD/src" .conda/bin/python -m ada_grasp_ctrl.calibration \
  --output output/calibration/ros_mounts_20261122/hand_mounts.yaml
```

The exporter records the source JSON path/hash and does not start ROS. If the pipeline uses
custom `extrinsics`, pass that same master JSON with `--calibration`. Regenerate after changing
the calibration. In the ROS terminal, pass this file explicitly:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=87
export PYTHONPATH="/opt/sharpa-wave-sdk/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/sharpa-wave-sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ros2 launch bimanual_manipulation bimanual_system.launch.py \
  hand_mounts:=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/output/calibration/ros_mounts_20261122/hand_mounts.yaml \
  crx_mock:=false sharpa_backend:=sharpa_sdk \
  mit_kp_ratio:=1.0 mit_kd_ratio:=1.0 \
  method:=quintic input_rate_hz:=100.0 \
  enable_fix_camera:=true enable_wrist_camera:=false \
  camera_config:=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/src/ada_grasp_ctrl/config/experiments/fixed_camera_driver.yaml \
  rviz:=true viewer:=true
```

This launch configures writable hardware and enables hand motors. It does not run a grasp.
The existing force limiter settings remain in force. The single-side right grasp entry remains
`script/run_grasp_pipeline.py`; it retains its right-only interpolator requirement.

If initial-state capture reports `ROS/source kinematics differ: left_hand_flange.xyz`
(or another hand-flange XYZ/RPY field), compare the reported source and ROS values.
The ROS launch defaults to preview mounts unless `hand_mounts` is supplied, whereas Ada applies
the master calibration. Stop the existing launch in its operator terminal and restart it with
the generated YAML; changing a file alone does not reload the running model. Both arms' base
transforms must also match the calibration; this exporter only supplies hand mounts. Keep the
kinematics check enabled, including for `--export-only`, and retry with a fresh pipeline output root.

## Capture and grasp a new object

In another terminal:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
export ROS_DOMAIN_ID=87
python3 script/run_dual_grasp_pipeline.py \
  --environment hardware --ros-domain-id 87 \
  --output-root "output/dual_grasp/live_$(date +%Y%m%d_%H%M%S)"
```

The wrapper selects Ada's Python environment and isolates inherited ROS Python paths.
ROI selection runs in ROS system Python. Draw a box around this invocation's object, then press
Enter in the image window. This only confirms the object selection. `--roi x0 y0 x1 y1` supplies
an explicit ROI for a headless session; `--cluster ID` selects among multiple extracted clusters.

Open `http://localhost:8089`. The preview displays both robot branches, measured scene/object
points, the selected side/candidate, phase selection and timed playback. Press Enter in the
pipeline terminal to execute the selected cycle. `q`, EOF or Ctrl+C cancels. Each new object/run
requires a fresh output root and new capture/inference; no failed live capture falls back to old data.

Default `--arms auto` enables both inference branches and selects one side automatically. Use
`--arms left` or `--arms right` to limit eligible candidates while retaining the dual-stack state
and model. Both modes still require the two-side launch. Selection failure sends no motion.
`--export-only` prepares the selected plan without a viewer or execution; `--preview-only` displays
it without an Enter gate. Neither is a prerequisite for normal hardware use.

## Optional hierarchical selection

`--selector legacy` is the default and retains the existing selection algorithm.
Use `--selector hierarchical` to call the existing mode-priority selector. The
`hierarchical_selector` block in `dual_grasp_pipeline.yaml` supplies its ranking:

```yaml
hierarchical_selector:
  prior:
    schema_version: 1
    kind: ranking
    ranking: [right_full, left_full]
  supported_modes: [right_full, left_full]
```

This is an editable example preference, not a learned Human Prior or a claim that
the right hand is universally better. The first feasible mode wins even when
another mode has lower effort; within that mode the existing effort/ID/seed order
is retained. Reverse `ranking` to prefer left. The CLI flag is required: YAML alone
does not activate hierarchical selection. Unknown modes, duplicates, non-ranking
priors and disagreement between ranking and supported modes fail before capture.

For a separately configured mock capture/dual-stack source:

```bash
export ROS_DOMAIN_ID=88
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
python3 script/run_dual_grasp_pipeline.py \
  --environment mock --ros-domain-id 88 --selector hierarchical \
  --output-root output/dual_grasp/hierarchical_new_run --export-only
```

This command still requires fresh capture and state sources; it does not create
mock drivers or fabricate an object. Use a new output root each time. `--arms left`
or `--arms right` restricts both inference and selection to that side. An unavailable
ranked mode appears as an empty candidate set; no extra inference branch runs.

Both modes use only the existing **right_full checkpoint**. `left_full` labels
right_full inference on the reflected cloud followed by the established pose/joint
mirror. Canonical provenance records `source_model_mode: right_full` and
`generation_method: mirrored_right_full` for left candidates. There is no separate
trained left model, multimode inference, simultaneous dual grasp or Human Prior call.

Hierarchical runs add `canonical_candidates.json`, `hierarchical_selection.json`
and, on selection, `grasp_plan.json`. The full hierarchical report is also embedded
in the root `selection.json`, which the existing exporter includes in its hashed
references. The canonical plan remains data-only with both readiness flags false.
The selected legacy payload and normal single-side packet/export/preview/Enter
checks are retained. `NO_FEASIBLE_GRASP` saves the trace and exits without targets,
viewer or motion sender/receiver. See the [integration handoff](dual-hierarchical-selector-handoff.md)
for synthetic examples, exact test commands and validation limits. This option has
not been physically validated; the standalone right-full entry is unchanged.

## Execution

Both sides use the established real pipeline sequence:

`above_object -> pregrasp -> grasp -> squeeze -> lift -> hold -> release -> return`

- Above-object fingers target zero on all 22 joints. Whole-hand geometry is used to compute
  10 cm clearance over the current measured object cloud. Descent interpolates to selected pregrasp.
- Lift moves the selected squeeze flange endpoint 5 cm in world +Z. Hold lasts 5 seconds.
- Release opens at the lifted position; return restores this run's measured arm start, with an open hand.
- Arm/finger speed limits remain 0.15/0.4 rad/s and acceleration limits 0.3/0.8 rad/s².
  Commands enter the existing interpolator at 100 Hz.
- Finger arrival is recorded without gating phases, as in the current right-hand pipeline.
  Arm arrival, fresh feedback, finite/bounded commands, MIT state and force-limiter checks remain.
- The selected hand alone receives finger commands. Every arm message includes the selected arm
  trajectory and the other arm's measured stationary target, overriding any retained earlier target.
  Both sides' feedback is monitored; unexpected passive-side drift aborts execution. A dual-run
  abort requests deactivation of both arm position controllers. It does not claim hand motor disable.

The live scene uses the measured table plane and both robot models for existing endpoint collision
checks. A partial object cloud is not treated as a verified collision solid. Above-object, lift and
return retain endpoint IK and joint-space interpolation without adding continuous-path planning.
The preview and packet are hash-bound, including the selected side and both starting states.
A recorded action completion is not evidence that the object was held or lifted.

The dual-live config allows `endpoint_inward_offset_m: 0.1` for above-object and lift IK.
Both endpoints first try the original vertical target. If that local solve fails, the planner
tries increasing horizontal radii (at most 1 cm apart) and a deterministic angular fan on the
active-base side, accepting the first sampled feasible target within 10 cm. Each proposed endpoint
must be no farther from the active arm's calibrated base in XY than the original endpoint.
The selected grasp, wrist target orientation, 10 cm hand-to-object overhead clearance and 5 cm
post-grasp lift height are retained. IK still requires position error at most 1 mm and orientation
error at most 0.01 rad; these tolerances also apply to the shifted targets. Exhausting the bounded
search still fails without selecting another grasp or applying a rejected solve.

Set `endpoint_inward_offset_m: 0.0` in a custom dual config to restore strictly vertical endpoints.
Older configs without this key retain that behavior. Each enabled endpoint records
`horizontal_adjustment` with its world-XY offset, active base, bound and trial diagnostics.
Preview and sender validation require the declared displacement and unchanged height/orientation;
old records without this metadata remain strictly vertical. This is endpoint IK, not continuous
collision-checked motion planning. The recorded-scene regression at
`output/dual_live/inward_offset_replay_20261122_v1/run_report.json` exports the original `right:c0000`
with 10 cm above-object and 3 cm lift offsets; it does not represent a fresh capture or hardware execution.

## Configuration and records

The additive [unified grasp contract](unified-grasp-contract-handoff.md) defines
versioned candidate/selected-plan data and explicit legacy adapters for future
selector, time-scaling and executor work. Its `available_arms` and `active_arms`
are distinct; a both-arm data record does not enable both-arm execution. Current
selection, motion-plan and ROS-packet serializations remain unchanged.

The separate [shared ROS mock executor](shared-mock-executor-handoff.md) consumes
the unified plan's active arms and shared timing schedule for isolated mock tests.
It does not replace this live entry or enable dual-active physical operation;
domain 87 remains reserved for the existing hardware workflow.

`src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml` inherits the existing grasp pipeline's
motion/camera settings, adds the portable dual-robot snapshot and explicit model references, and
defaults to automatic side selection. Camera capture defaults to the installed unified calibration
JSON. Each invocation copies the source `robot_snapshot` into its new run directory and replaces
the base and hand-mount fixed origins with absolute values from that same JSON. The effective
config points FK, IK, collision checks, preview and export to this calibrated snapshot; the source
snapshot is preserved. Both branches retain the existing checks before execution. No camera-to-world
transform is applied twice.

The inherited table plane now uses the 2026-11-22 empty-table capture fitted with the master JSON.
After confirming a table-only change, the operator requested applying this result; the automatic
right pipeline uses the corresponding new operator confirmation record. Existing calibration checks
pass with the recorded new captures. Old table/acceptance artifacts remain available for replay.
See the [unified calibration handoff](unified-calibration-handoff.md) for paths, frame mappings,
validation results and the existing table recalibration procedure.

Records include `capture/`, `object/`, `inference_left/`, `inference_right/`, `candidates.json`,
`state/robot_state.json`, `selection.json`, `export/targets.json`, `export/motion_plan.json`,
`effective_config.yaml`, `run_manifest.yaml`, `run_report.json`, and execution `sender/` and `receiver/`.
Left inference retains the original capture provenance and records the world reflection explicitly.

The implementation shares `ros_execution/`, the sender, plan validation and endpoint helpers with
the original right pipeline. Legacy schema-1 right packets remain readable; new side-bound packets
use `kind: ada_side_joint_targets` with explicit `side`. MuJoCo Phase 2 execution and the public
research evaluation pipeline are independent of this entry.

## Validation evidence (2026-11-11)

- Full Ada unit suite: 379 tests, 316 passed and 63 skipped. ROS system Python separately passed
  all seven existing receiver tests and both new dual-side receiver tests.
- Two different synthetic object clouds exercise fresh per-side inference inputs, reflection and
  camera/world transform consistency. This regression test mocks the network predictions.
- The full CLI captured ROS camera messages again, extracted the object and ran actual checkpoint
  inference on domain 88. The camera source was a replay of the physical pear capture, not a new
  physical camera exposure. Automatic selection exported `left:c0003`; a separate `--arms right`
  run exported `right:c0003`. See `output/dual_arm_hardware/live_entry_v1` and
  `output/dual_arm_hardware/live_entry_right_v1`.
- Both sides completed all eight ROS mock stages. The left audit recorded 4034 arm/left-hand
  messages and zero right-hand messages; the right audit recorded 4656 arm, 4655 right-hand and
  zero left-hand messages. Every arm command retained the passive side's measured target.
  Audits: `output/dual_arm_hardware/mock_transport_left_v2/audit.json` and
  `output/dual_arm_hardware/mock_transport_right_v1/audit.json`.
- Those transport audits used a synthetic ARMED force-limiter state on isolated mock domain 88
  because mock wrench readings were invalid. They validate routing/playback, not hardware force
  sensing. Production limiter requirements were unchanged; no physical grasp was executed.
- The normal CLI reached the Enter gate with HTTP 200 on port 8089, and `q` canceled before creating
  a sender/receiver. Evidence: `output/dual_arm_hardware/live_preview_entry_v1`.
- Ruff lint, compileall and `git diff --check` passed. All 22 changed/new Python files pass format
  checks; the whole-tree format check still reports eight pre-existing unrelated files.
- The maintained LEAP quick example completed with 100 outputs (87 success, 6 failure, 7 invalid
  initialization, zero execution errors/degradation). Run: `output/examples/leap_tac3d/20261112T042551Z-317020`.
  Its first attempt inherited ROS Python 3.12 packages into Ada Python 3.10 and failed importing
  Pinocchio; rerunning with `PYTHONPATH=src` resolved that environment issue. The new pipeline wrapper
  already isolates these Python paths.

Checks used:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl .conda/bin/python -m unittest discover -s tests -v
.conda/bin/python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
PYTHONPATH=src PYTHON_BIN=.conda/bin/python bash script/run_example.sh leap_tac3d quick
```

In a sourced ROS terminal with the checkout's `src` on `PYTHONPATH`:

```bash
/usr/bin/python3 -m unittest discover -s tests -p test_right_ros_receiver.py -v
/usr/bin/python3 -m unittest discover -s tests -p test_dual_live_receiver.py -v
```
