# Live automatic left/right grasp pipeline

Entry: `script/run_dual_grasp_pipeline.py`. Each invocation captures the current fixed camera,
lets the operator select the current object, runs fresh right-full inference on the original and
reflected object clouds, selects one feasible side/candidate by the existing weighted CRX joint
displacement cost, previews the full cycle, and executes after terminal Enter.
There is no fixed object, saved candidate, ROI, or object pose in the default workflow. The pear
capture used during development is test evidence only. The checkpoint/config/metadata references
identify the model, not an object to replay.

## Start the two physical arms, hands and fixed camera

Use a ROS terminal and keep the launch running. Replace an existing single-side launch before
starting this stack; it owns both devices and the fixed camera.

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=87
export PYTHONPATH="/opt/sharpa-wave-sdk/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/sharpa-wave-sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ros2 launch bimanual_manipulation bimanual_system.launch.py \
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

## Configuration and records

`src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml` inherits the existing grasp pipeline's
motion/camera settings, adds the portable dual-robot snapshot and explicit model references, and
defaults to automatic side selection. The current calibrated camera extrinsics and table file are
inherited. The stored robot snapshot must match the live world/base/mount/joint geometry; both
branches are checked before execution. No camera-to-world transform is applied twice.

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
