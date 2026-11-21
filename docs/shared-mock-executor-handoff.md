# Task 4 — shared ROS mock executor

Base commit: `ae94ec1585c4444849ea0613ac87b152f8edd44d`. Implementation is in the
original checkout. The new execution path is additive. Existing live entry points,
ROS packet schemas, drivers and MPC mathematics are unchanged. One legacy linear
receiver timestamp initialization defect was corrected, as described below.

**Dual-active physical operation is not validated or enabled.** The new executor
accepts only an explicit mock packet, rejects domain 87 before ROS initialization,
requires localhost discovery, and verifies both sides' actual driver interfaces
identify themselves as mock. The usual hardware workflow on domain 87 remains
the existing `run_grasp_pipeline.py` / `run_dual_grasp_pipeline.py` workflow.

## Data and entry points

The Ada environment prepares the artifact:

```python
from ada_grasp_ctrl.pipeline.shared_artifacts import shared_mock_packet

packet = shared_mock_packet(schedule, snapshot, description)
# schedule: validated Task 3 output, including the complete canonical GraspPlan.
# snapshot: {"left": {joint_name: radians, ...}, "right": {...}}, 28 joints each.
# description: the combined URDF used to bind the live mock scene's kinematics.
# Write packet using json.dumps(packet, allow_nan=False).
```

`shared_mock_packet` validates the complete Task 1/3 data and projects it into an
explicit schema-1 `kind: ada_shared_mock_joint_targets` packet. It contains:

- `environment: mock`, `hardware_ready: false`;
- `plan_identity`: stable candidate ID, mode, provenance, `active_arms` and
  `available_arms` from the canonical plan (availability defaults to active arms
  only when omitted by the source plan);
- the Task 3 numeric schedule, retaining common clocks, ordered stage endpoints,
  per-side duration policies, original readiness/verification flags and timing;
- both measured 28-joint starting states and both model kinematic contracts;
- `source_schedule_sha256`, an audit fingerprint of the original complete schedule.

The ROS process validates the numeric transport projection with standard-library
code; it does not import the Ada robot planner, inference, MuJoCo or Torch.
`load_shared_packet` does not accept or auto-upgrade legacy packets. Legacy
`right_ros_sender.load_packet` and `right_ros_contract.validate_packet` remain
unchanged. The fingerprint is an audit reference, not an approval signature.

Run the new executor only from sourced ROS system Python in a verified mock setup:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=88
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PYTHONPATH=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/src:${PYTHONPATH:-}
/usr/bin/python3 -m ada_grasp_ctrl.ros_execution.shared_executor \
  --targets /absolute/path/shared_targets.json \
  --output-root /absolute/path/new_mock_run \
  --ros-domain-id 88
```

The output root must be new. This command does not start drivers. Existing
`Runtime.preflight` verifies mock plugins, hand backend, writable active hands,
interpolator configuration, controller state, required limiter state and command
ownership before command publication. Both sides' URDF kinematics and endpoints
are checked. The passive side is also verified as mock before any stationary arm
target is sent. Neither the source plan's `available_arms` nor the fact that both
devices exist makes both sides active.

## Shared clock, guards and completion

`SharedStageStream` is ROS-independent. One outer loop owns the nominal 100 Hz
cadence, one monotonic stage start and one normalized progress value. It computes
every active side's linear reference for that same instant, then prepares all
bounded outputs before returning any frame. It uses the existing `BaselineStream`
numerical guard unchanged, with explicit side/order/model-limit validation at its
boundary. The guard's historical right-joint names are an internal coordinate
template only; emitted names always correspond to the actual physical side.
No MPC solver or MPC controller is invoked.

The ROS owner publishes one 12-joint CRX message and one 22-joint Sharpa message per
active hand. All messages for a frame receive the **same ROS header timestamp**
from the owner node. For a single-active plan, the other CRX's six targets remain
the reviewed measured snapshot; no passive Sharpa publisher is created. Passive
arm and finger drift retain the existing `check_start` tolerances.

Duration starts from Task 3's shared maximum. Each side is checked against live
model velocity bounds and the existing receiver duration rule. Following the
legacy nonblocking-hand policy, a new stage starts from fresh measured joints;
if fingers lag their prior target, the execution duration may be extended to the
maximum requirement across **all active sides**. Reviewed and executed durations
are logged. No side receives an independent extension or stage clock.

The next stage begins only after common duration elapsed, all bounded command
vectors reached their endpoints, every active CRX meets existing arm arrival
tolerance, and the common settle interval has fresh feedback. Finger errors are
recorded with `hand_arrival_required=false` and do not create an extra barrier.
Hold begins after lift completion. Arrival timeout, clock stall, stale/invalid
feedback, joint-limit violation, unsafe bounded output, cancellation and ownership/
routing loss latch the whole stream stopped. Existing runtime abort handling
requests deactivation of both CRX position controllers; this is not hand motor
disable or proof of physical emergency-stop behavior.

The shared linear **reference** is synchronized. Existing acceleration/speed/step
guards can cause different per-side bounded tracking lag; physical arrival is
therefore not claimed to be simultaneous. Linear endpoint velocity discontinuities
remain subject to those existing lower-level protections. No driver interpolation
method or limit was changed.

DDS publishes separate messages, not a transaction. A publication exception stops
the combined stream and requests existing bounded aborts; it cannot retract a
message already delivered. Matching stamps and command records expose a missing
transport pair. No autonomous one-side stage progression is permitted.

## Audit and validation

`run_manifest.yaml` records the explicit domain and packet hash. The existing
`RunLog` records the model/runtime provenance and a result report. New events
include `stage_started`, `prepared_frame`, `paired_published`, `stage_completed`
and `shared_abort`, with common monotonic/ROS timestamps, progress, desired and
bounded per-side vectors, measured joints, feedback receive times and arrival
errors. Normal completion is not evidence of object grasp/lift success.

`script/test_shared_ros_mock.py` creates synthetic feedback/parameter/controller
services only. It requires localhost domain 88 and refuses to run if that domain
already has other nodes. It starts no physical drivers. Its synthetic limiter
state is explicitly test data; production limiter requirements remain intact.
The audit checks actual DDS wire timestamps and joint routing against executor
records, sole publisher ownership, passive-arm stationarity, no passive hand
messages, shared stage ordering and both-controller abort propagation.

The matrix includes both-active eight-stage completion, each single-active side,
a late CRX with persistent finger error, stale feedback, joint-limit violation,
a competing command publisher, and the legacy schema-1 right action
receiver executing three linear stages.

### Results and recordings (2026-11-20)

Preserved evidence: [`output/shared_executor/task4_20261120_verified`](../output/shared_executor/task4_20261120_verified/).
The [DDS audit](../output/shared_executor/task4_20261120_verified/dds/audit.json)
combines seven successful shared-executor cases from the initial matrix and a
successful legacy-only rerun. Original manifests, logs and wire recordings are
retained; this is explicitly a composite audit, not a claim that the first matrix
invocation passed. All DDS processes used domain **88**, localhost discovery and
synthetic drivers. No hardware drivers were started.

| Case | CRX messages | Left hand | Right hand | Outcome |
| --- | ---: | ---: | ---: | --- |
| Both active, asymmetric distances | 2,489 | 2,489 | 2,489 | All 8 stages completed |
| Right active | 1,070 | 0 | 1,070 | 3 stages; passive left stationary |
| Left active | 744 | 744 | 0 | 3 stages; passive right stationary |
| Late right CRX + persistent finger error | 1,124 | 1,124 | 1,124 | 3 stages; shared wait, no finger barrier |
| Stale right feedback | 69 | 69 | 69 | Both controllers stopped; no next stage |
| Left joint limit violation | 21 | 21 | 21 | Both controllers stopped; no next stage |
| Competing CRX publisher | 20 | 20 | 20 | Both controllers stopped; no next stage |
| Legacy schema-1 right linear actions | 1,023 | 0 | 1,023 | All 3 actions succeeded |

Every shared frame's observed DDS arm/hand timestamps and values matched its
`paired_published` record. Shared progress and stage ordering passed. All normal
shared arm and hand graph observations had exactly one publisher. The ownership
fault case permits the injected competing arm publisher while testing combined
abort. Legacy action publishers are destroyed between goals; one final in-flight
hand sample arrived after its publisher disappeared from the graph. This is
recorded as one empty graph observation, not treated as proof of an owner at that
instant. No competing legacy publisher was observed.

The first legacy test exposed `Reordered feedback`: the existing linear guard
was initialized with loop-start time instead of the older, still-fresh measured
sample's receive times. `bridge.py` now uses the actual arm/hand receive times,
consistent with subsequent ticks. The numeric guard, freshness/reordering checks
and MPC path are unchanged. A focused fake-I/O receiver regression uses feedback
20 ms older than the loop clock and completes successfully. The original failing
DDS log remains at `/tmp/ada_task4_dds_matrix_20261120_v1/legacy_right`.

The [fixed-clock recordings](../output/shared_executor/task4_20261120_verified/deterministic/run_manifest.yaml)
contain **two independently simulated, byte-identical JSON recordings per case**:
both-active (2,551 frames, 8 stages), right (1,096 frames), left (763 frames), and
late CRX/finger error (1,148 frames). Their hashes, source hashes, targets, feedback,
common progress and stage events are preserved. DDS wall-clock recordings are not
claimed to be byte-deterministic.

Validation results:

- Full Ada suite: 558 tests, 473 passed and 85 skipped (ROS-dependent tests).
- Sourced ROS subset: 28 tests, 23 passed and 5 skipped. The Ada exporter needs
  MuJoCo and is tested in Ada Python; four existing dynamic-lift tests require
  their separate offline review fixture. Linear and `mpc_active` regressions pass.
- Focused shared executor suite: all 10 tests passed in Ada Python.
- Whole-tree Ruff lint, Python compilation, changed-file formatting and
  `git diff --check` passed. Full-tree formatting still reports the same eight
  pre-existing files; none were reformatted for this task.

Exact commands, from the repository root (run directories must be new):

```bash
env PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_task4_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 .conda/bin/python -m unittest discover -s tests -v

env PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_task4_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 .conda/bin/python script/record_shared_mock.py \
  --output-root /tmp/ada_task4_deterministic_20261120

env ROS_DOMAIN_ID=88 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  bash --noprofile --norc -c '
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/src:/home/hao-fang/hugs-workspace/ada-grasp-ctrl/tests:${PYTHONPATH:-}
nice -n 10 /usr/bin/python3 script/test_shared_ros_mock.py \
  --output-root /tmp/ada_task4_dds_matrix_20261120_v1
'
# The initial matrix failed on legacy_right. After the timestamp correction,
# use the same ROS wrapper with this successful rerun command:
# nice -n 10 /usr/bin/python3 script/test_shared_ros_mock.py \
#   --output-root /tmp/ada_task4_legacy_20261120_v2 --cases legacy_right
# The sourced regression command inside the same ROS wrapper:
# nice -n 10 /usr/bin/python3 -m unittest test_shared_executor \
#   test_right_ros_receiver test_dual_live_receiver test_mpc_p3_receiver \
#   test_dynamic_lift_ros -v

.conda/bin/python -m compileall -q src tests script
/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff check src tests script
/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff format --check src tests script
git diff --check
```

The recorded DDS audit additionally checks hand publisher observations; those
assertions were also applied to the saved wire records after collection.
The full-tree format command exits nonzero only for the existing eight files.
Test logs are copied into the evidence directory. Evaluation workers in the
other checkout were left running; tests used one numeric thread, low CPU priority
and no GPU. No interference was observed.

## Files and remaining scope

New production modules:
`pipeline/shared_artifacts.py`, `ros_execution/shared_packet.py`,
`ros_execution/shared_stream.py`, `ros_execution/shared_executor.py`.
New test/audit files:
`tests/shared_executor_fixtures.py`, `tests/shared_legacy_mock.py`,
`tests/test_shared_executor.py`, `script/test_shared_ros_mock.py`,
`script/record_shared_mock.py`.
Legacy correction and regression: `ros_execution/bridge.py`,
`tests/test_mpc_p3_receiver.py`.
Documentation: this handoff and its link from the dual live pipeline guide.

The new path is explicitly selected; the live dual runner does not auto-route
existing packets through it. Task 2 still reports two-arm grasp candidates as
unsupported; mock test poses do not add a bimanual inference/IK capability.
Physical commissioning, simultaneous grasp generation, force sharing, Place,
tactile/MPC/15 N completion changes and physical dual-active authorization remain
outside this task. No commit or push is part of this implementation request.
