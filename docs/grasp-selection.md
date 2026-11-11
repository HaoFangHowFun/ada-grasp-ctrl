# Read-only right CRX grasp selection

The experimental `ada_grasp_ctrl.grasp_selection` package selects a saved DexLearn
`right_full` candidate and a specific right CRX IK branch by minimum weighted
squared arm-joint displacement. It does not run inference, command ROS actuators,
step robot dynamics, plan a path, or change Ada's maintained four-stage workflow.

The core imports neither ROS, MuJoCo nor Torch. Its API is:

```python
from ada_grasp_ctrl.grasp_selection import GraspSelector
from ada_grasp_ctrl.grasp_selection.criteria.joint_effort import JointDisplacement

selector = GraspSelector(robot_adapter=adapter,
                         scoring_criteria=[JointDisplacement([1, 1, 1, 1, 1, 1])])
result = selector.select(candidates=batch.candidates, context=context)
```

`StageTarget`, `GraspCandidate`, `RobotState` and `SelectionContext` validate
finite values, dimensions, stages and frames. `CandidateEvaluation` retains the
bounded IK options, stage diagnostics, rejection codes and per-option costs.
`SelectionResult` retains original poses, the selected six-joint arm vector,
canonical hand targets, source IDs and acceptance status. Array inputs are copied
and made read-only. Metadata is preserved and must be treated as input data.

Hard criteria populate cached options and append candidate or option rejection
reasons. Scoring criteria return a finite scalar and details. Rejected options
cannot win through scoring; the default and only deployed score is
`sum(weights * (q_grasp - q_current)**2)`. Weights are six finite, nonnegative
diagonal entries. Limited joints are never wrapped at pi. Exact ties sort by
candidate ID, then solution index. All-failed and empty batches return structured
`no_feasible_candidate` results.

## Reused implementation and scope

The backend loads the existing right-side preparation report's hashed MJCF, URDF
and mesh dependencies. It verifies exactly six CRX plus 22 right Sharpa joints.
It reuses:

- `crx_sharpa_right_preparation.RightKinematics.solve`, `forward`, `audit`,
  `body_pose`, `residual`, `POSITION_TOLERANCE` and `ANGLE_TOLERANCE`.
- `crx_sharpa_right_model.JOINTS` and `matrix_pose`; the loaded models were built
  by its existing `build_model`, without rebuilding CRX IK.
- `crx_sharpa_right_simulation.exclude_base_joint_contact`,
  `ApprovedScene.contact_diagnostics` and `invalid_initialization`.
- Preview-only `crx_sharpa_right_preview.hand_mesh` and
  `fixed_camera_preview.plane_mesh` for the original targets and measured table.

Each candidate uses the captured current arm position followed by the five fixed
fallback seeds already used by `RightKinematics.solve`. The existing solver can
itself consult its fallback list. At most six returned grasp configurations are
retained; branches within `1e-6 rad` per joint are deduplicated, preserving the
first solution index. This is a recorded bounded subset, not exhaustive IK.
Every retained grasp option must also have acceptable original pregrasp and
squeeze endpoints. Those stages start from that grasp branch and use the existing
fallback workflow. Stage endpoints need not belong to a continuous motion branch.

IK acceptance stays at **1 mm / 0.01 rad**, including solver convergence.
Independent MuJoCo/Pinocchio auditing uses the existing `1e-6` tolerance; exported
MJCF origins contain rounding at approximately `4e-7 m`.

Collision exclusions preserve the existing hand exclusions and the intentional
adjacent `right_base_link`/`right_J1_link` bearing exclusion. Arm penetration must
not exceed **1e-5 m**; the existing experimental hand/self/table penetration gate
is **0.01 m**. Passing that gate does not mean zero penetration. No tolerance was
relaxed for these captures. The measured tilted table plane replaces the old
benchmark table for both solver fallback selection and final static checks.

The five saved assets explicitly contain single-view open surfaces, not complete
collision solids. They are displayed, but are not converted to guessed convex
objects. Robot/self/table checks run on the actual CRX/Sharpa collision geometry;
object and remaining environment collisions are **unverified**. Consequently,
`known_geometry_passed` never means that the complete scene is safe.

## Frames and state provenance

The saved candidates already declare world-frame `right_hand_flange` poses,
meters, WXYZ quaternions and radian joint values. Their per-stage hand targets
are dictionaries keyed by the 22 canonical joint names. The loader checks those
names and canonicalizes them without assuming JSON order. It preserves all three
original stages and never extrapolates pregrasp or reapplies camera extrinsics.

The camera calibration declares world as the deployment URDF root / left arm
base. The connected deployment model retains that world root even for right-only
launch. The captured description gives the fixed right base origin:

```text
xyz (m):   [0.03058936, -0.91156264, 0.00873476]
rpy (rad): [0.00994493, -0.01029918, -1.54694055]
```

The transform chain is explicit:

```text
base_T_flange = inv(world_T_right_base) @ world_T_flange
model_T_flange = model_T_right_base @ base_T_flange
```

The second line is needed because existing `RightKinematics` works in its model's
world coordinates. No legacy experiment `REGISTRATION` translation is applied to
the measured targets. Camera-world and base mounting have traced nominal sources
but have not been independently physically accepted.

The read-only boundary subscribes to `/crx5ia/right/joint_states` and
`/bimanual/robot_description`. The installed JointState broadcaster uses its
default header `base_link`; this is checked explicitly and retained as
`source_frame_id`. Joint angles are interpreted using the named URDF joint axes,
not by treating the header as a Cartesian pose frame. The observed description
must match every expected right-side joint, ancestor, axis, limit and mount. Extra
fixed display frames such as `right_tcp` are allowed; extra moving right joints
are rejected. The expected URDF SHA-256 binds the state to the selector model.

The aggregate `/bimanual/joint_states` is not the default. Snapshots containing
left-side joints are rejected. Freshness checks include transport delay, future
timestamps, completeness and finite values. Replay checks freshness at the saved
receipt time and is clearly labeled replay; it does not pretend an old snapshot
is current hardware feedback. Live snapshot evaluation requires a hardware-tagged
snapshot and the system ROS clock. The environment tag is operator-provided,
not independently proven hardware identity.

## Capture a mock snapshot

ROS uses system Python 3.12. Ada's numerical backend uses its existing Python 3.10
environment. No dependency installation or inference environment is needed.

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=89
export ROS_LOG_DIR=/tmp/grasp_selector_ros_logs
ros2 launch bimanual_manipulation single_arm_hand.launch.py \
  prefix:=right_ crx_mock:=true sharpa_backend:=mock read_only:=true \
  rviz:=false viewer:=false enable_fix_camera:=false enable_wrist_camera:=false
```

In another ROS shell with the same setup and domain:

```bash
export PYTHONPATH=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/src:$PYTHONPATH
/usr/bin/python3 -m ada_grasp_ctrl.grasp_selection.capture_state \
  --model-urdf /home/hao-fang/hugs-workspace/ada-grasp-ctrl/output/regression/crx-sharpa-right-preview-x080-y070-config-v1/model/model.urdf \
  --environment mock \
  --output-root /home/hao-fang/hugs-workspace/ada-grasp-ctrl/output/grasp_selection/new_capture
```

Only subscribers are created by this tool. It never launches a robot itself.
`--environment hardware` can label a snapshot from a separately managed, verified
hardware feedback source; hardware startup and execution are outside this tool.
Do not use mock or configured `offline_seed` values as real current hardware state.

## Offline selection and five-capture integration

Run from the Ada checkout. Relative input/output paths use Ada's existing path
helpers; every output directory must be fresh. These commands reuse the actual
mock snapshot captured during this task:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
env PYTHONPATH=src .conda/bin/python -m ada_grasp_ctrl.grasp_selection.cli \
  --candidates output/fixed_camera/green_pear_group02_grasps/candidates.json \
  --state output/grasp_selection/mvd_20261110_v1/capture_checked/robot_state.json \
  --model-report output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json \
  --replay --output-root output/grasp_selection/new_selection

env -u PYTHONPATH .conda/bin/python script/evaluate_grasp_selection_gallery.py \
  --gallery output/fixed_camera/five_objects_gallery/manifest.json \
  --state output/grasp_selection/mvd_20261110_v1/capture_checked/robot_state.json \
  --model-report output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json \
  --output-root output/grasp_selection/new_five_groups
```

The gallery tool evaluates each of the five captures independently twice. It
compares `selection_result.json` and `candidate_evaluations.json` byte-for-byte;
it does not combine unrelated object scenes into one competition. Each run also
writes `manifest.json`, `run_manifest.yaml` and `run_report.json`, recording input,
model, code, calibration and configuration provenance. Source inference files
and historical results are never overwritten. CLI exit 0 includes the valid
`no_feasible_candidate` outcome; preflight/input errors exit 2. The gallery returns
1 on reproducibility mismatch.

## Unified preview on 8089

```bash
env PYTHONPATH=src MUJOCO_GL=egl .conda/bin/python -m ada_grasp_ctrl.grasp_selection.preview \
  --summary output/grasp_selection/mvd_20261110_v1/five_groups_final/integration_summary.json \
  --output-root output/grasp_selection/new_preview --port 8089
```

Open <http://localhost:8089>. Select a dataset, inspect current-arm / pregrasp /
grasp / squeeze, or play illustrative joint interpolation. Overview and close-up
cameras, original target ghosts and collision geometry toggles share this one
port. An occupied 8089 is an error; no alternative port is silently selected.
`--export-only` produces ten overview/close-up PNGs without starting a server.

The preview rechecks saved endpoint FK, limits and static feasibility before
showing the selected solution. Interpolation is only a display convention: it is
not a planned path, controller result or successful physical grasp. It performs
no physics steps and publishes no commands. Current arm joints come from the
snapshot; initial fingers use the candidate's pregrasp target because no current
hand snapshot was captured. The UI reports sampled robot/self/table penetration,
including the existing 10 mm hand gate. These display samples never rerank the
selection. The object stays at its measured position throughout.

All results remain `execution_ready=false`, including when endpoint checks pass.
Camera extrinsics, world/base registration, hand mounting, table and complete
environment geometry must be independently accepted; a separate execution layer
must revalidate current state and a safe continuous motion path.

See [the factual validation report](grasp-selection-validation.md) for results,
known baseline failures and the exact changed-file inventory.
