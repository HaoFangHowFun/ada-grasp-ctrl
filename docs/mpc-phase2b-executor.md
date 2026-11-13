# Phase 2B: original MPC and guarded mock streaming

## Data and process ownership

```mermaid
flowchart TD
  P[Reviewed stages and linear q_ref] --> O[StageOrchestrator]
  F[Named measured JointFeedback q] --> O
  C[MuJoCo or compatible tactile ContactObservation] --> O
  O --> W[Native Python 3.10 worker: original Ours at 10 Hz]
  Q[Last bounded command q_cmd] --> W
  W --> M[Absolute named 28-joint MPCProposal]
  M --> G[ROS Python 3.12: GuardedStream nominal 100 Hz]
  G --> A[/crx5ia/joint_targets]
  G --> H[/sharpa/right_hand/joint_command]
  A --> I[Unchanged CRX quintic reference interpolation: 500 Hz]
  I --> D[Existing right mock drivers]
  H --> D
  D --> F
  G --> Q
```

The passive orchestrator has no arrow to a publisher. The old action receiver must
be absent when the separate mock streaming owner acquires the topics. Local ownership
claims and DDS graph checks both enforce this boundary. The ROS adapter reuses the
existing Runtime preflight, backend/plugin verification, model limits, controller
state, limiter/watchdog and stop handling. It rejects hardware/left/dual configuration
before node creation; the actual arm plugin and hand backend must independently be
mock. No direct driver bypass or lower-level configuration changes are introduced.

`MPCWorker` uses versioned JSON lines and one outstanding request. The ROS process
does not import Pinocchio/Torch/MuJoCo. The native worker strips ROS Python/library
paths, follows the existing Torch-before-Pinocchio ABI setup and owns its own solver.
It rejects request backlog; pending IPC/solve latency over 100 ms aborts. This is an
asynchronous interface so the supervisor can keep running while optimization occurs.

## Mathematics and state

`NativeMPC` invokes the existing `Controller("ours")` and `_CoordinatedWrenchPolicy`.
It does not copy or modify the objective, force schedule, `Ks`, constraints, SLSQP
settings or centimeter scaling. The factory uses frozen right-full robot/config
metadata and canonical six CRX plus 22 Sharpa order, with identity actuation mapping.

Measured q and q_cmd are separate named inputs. Native `curr_q_f` is measured q;
`curr_q_a` is the latest bounded actuator target, not the latest optimizer proposal.
Object pose is not consumed by this policy and is not invented as a sensor reading.
`q_mpc` is an absolute named target. Native rejection diagnostics are retained even
when the native simulation policy is `apply_candidate`; the new execution boundary
rejects that candidate and latches a fault instead of publishing it. No solver math
or simulation failure policy was changed.

## Timing and failure policy

| Condition | Result |
| --- | --- |
| Valid state/contact, accepted solver | Candidate passed to guarded mock executor |
| Valid empty contact frame | Native no-contact policy, not a fabricated force |
| Invalid/missing/stale/reordered contact | No candidate; passive records reason, active stops |
| Solver rejected/degraded or malformed target | No candidate; runtime latched until explicit new instance |
| Native stop/timeout | No new candidate; no automatic stage advancement or fallback |
| Solver/IPC duration >100 ms | Reject deadline; retained diagnostic; no delayed output approval |
| Feedback age >200 ms at MPC / >500 ms at supervisor | Reject/stop; duplicate source stamps do not refresh age |
| Proposal older than 250 ms or reordered | Stop; cannot refresh an old target with a new receive time |
| Command loop gap >50 ms, backward clock, lost DDS/owner | Latch silence and invoke existing mock stop handling |
| Cancellation/shutdown | Release ownership, destroy command publishers, stop owned worker |

The 100 Hz guard enforces model position limits and the existing arm/hand speed,
step and acceleration defaults. Target reversals decelerate under the same bounds.
It uses a discrete braking envelope, may lag or overshoot an optimizer target, and
rejects an infeasible bounded step. Delayed callbacks emit at most one command, not
a catch-up burst. No hard real-time guarantees or physical emergency-stop claims
are made. Initial hold is explicitly distinguished from an MPC solve.

ROS measurement time is the host monotonic receipt guarded by the original ROS
header-age/replay checks; it is not claimed to be a synchronized hardware acquisition
timestamp. A test observer's callback-arrival times are not 500 Hz servo measurements.

## Test entrypoints

Start only the existing right mock stack on an empty isolated localhost domain:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=199 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ros2 launch bimanual_manipulation single_arm_hand.launch.py \
  prefix:=right_ crx_mock:=true sharpa_backend:=mock rviz:=false viewer:=false
```

From a separately sourced terminal on the same domain:

```bash
/usr/bin/python3 script/test_mpc_ros_mock.py \
  --output-root /tmp/ada-phase2-ros-new \
  --model-report "$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json" \
  --mpc-python "$PWD/.conda/bin/python"
/usr/bin/python3 script/test_mpc_retiming.py \
  --trace /tmp/ada-phase2-ros-new/trace.json --output-root /tmp/ada-phase2-retiming-new
```

The test uses explicitly synthetic valid-empty contact input to the real solver;
it does not claim tactile sensing or contact-grasp success. It checks right/left
publisher inventory, competing ownership, deliberate command stall and shutdown
silence. Unit tests also cover stale feedback, rejected contact/proposals, solver
failure, dropped transport, limits and cancellation. The unchanged quintic module
is replayed offline to quantify horizon extension and reference lag. See the
[validation report](mpc-phase2-validation-2026-11-12.md) for actual distributions.
