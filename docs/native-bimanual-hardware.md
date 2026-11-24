# Native 200k unified single/bimanual hardware path

The existing dual-live command now executes either one or two arms according to
the native model's first feasible ranked source mode. `--grasp-model right_full`
remains the default. Native inference still requires `--selector hierarchical`.
`--grasp-mode auto` is the default and adds no required flag. `--arms auto` makes
both arms available; it does not require a bimanual grasp.

## Operator rollout

Use the existing dual FANUC/Sharpa driver launch, with its normal interpolator and
protections. Do not run another target publisher. Each invocation captures a fresh
object and measured state; use a fresh output directory for each step.

Start with normal model-ranked export only:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
PYTHONPATH=src .conda/bin/python script/run_dual_grasp_pipeline.py \
  --environment hardware --ros-domain-id 87 \
  --selector hierarchical --arms auto --grasp-model native_multimode \
  --export-only \
  --output-root "output/dual_live/native_200k_export_$(date +%Y%m%d_%H%M%S)"
```

For an optional bimanual-only smoke test, start with:

```bash
PYTHONPATH=src .conda/bin/python script/run_dual_grasp_pipeline.py \
  --environment hardware --ros-domain-id 87 \
  --selector hierarchical --arms auto --grasp-model native_multimode \
  --grasp-mode both_only --export-only \
  --output-root "output/dual_live/native_both_export_$(date +%Y%m%d_%H%M%S)"
```

`both_only` is a post-model allowed-mode filter. Neither threshold nor Top-K is
changed. If the native mask has no paired mode, or no paired candidate is feasible,
the result is `NO_FEASIBLE_GRASP`; no substitute candidate or legacy model is used.
It requires both arms enabled and `native_multimode`. `--arms left/right` continues
to use the established single-arm path in `auto`.

Next, repeat the appropriate command with `--preview-only` instead of
`--export-only` and a new output directory. Open <http://localhost:8089>; the same
phase/progress slider drives both robot trajectories when a paired mode wins.
Review the source mode, executed arms, scores, approach, lift, release, and return.
Terminate the preview-only process before a new run.

For operator-reviewed live execution, use the normal command below. It creates a
fresh preview and waits for terminal Enter before any command transport starts.
Add `--grasp-mode both_only` only if intentionally restricting the mode filter.

```bash
PYTHONPATH=src .conda/bin/python script/run_dual_grasp_pipeline.py \
  --environment hardware --ros-domain-id 87 \
  --selector hierarchical --arms auto --grasp-model native_multimode \
  --output-root "output/dual_live/native_200k_live_$(date +%Y%m%d_%H%M%S)"
```

The runner checks displayed packet identity, rechecks both measured starting
states, and verifies source hashes before dispatch. One-arm selections retain the
existing bridge/sender. Two-arm selections use the shared hardware executor; the
operator never chooses an executor. Export-only and preview-only do not send motion.

For ROS mock validation, use the existing isolated harness (synthetic drivers):

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ROS_DOMAIN_ID=88 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  /usr/bin/python3 script/test_shared_ros_mock.py \
  --output-root "output/shared_mock/native_both_$(date +%Y%m%d_%H%M%S)"
```

Hardware packets cannot be passed to the mock executor. Mock packets cannot enable
hardware, and the mock entry still rejects domain 87 and non-local discovery.
Do not relabel a hardware export as a mock packet. The harness builds its own
explicit synthetic mock schedule; the 8089 preview uses the actual exported plan.

## Implementation and limits

- `experiments/native_multimode_grasp.py` decodes both native 29D hands from the
  same accepted original-cloud sample. Joint names come from metadata offsets
  6:28 and 34:56; dummy-arm metadata is never treated as a learned arm target.
  Centroid restoration and WXYZ conversion occur once. Existing single-hand
  decoding/mirroring remains unchanged. One worker loads one checkpoint.
- `grasp_selection/multi_arm.py` and `hierarchical.py` use existing arm IK and
  seeds, preserve all other joints, and validate combined 56-joint stage states.
  Both active hands retain the existing phase-aware object-contact policy;
  inter-arm contacts remain forbidden. The paired cost is the sum of squared
  six-joint displacements. Effort never compares different mode ranks.
- `pipeline/paired_artifacts.py` extends the existing export dispatcher. It uses
  existing overhead geometry and five common world-Z lift/IK waypoints, retains
  squeeze fingers, and checks relative wrist transforms at lift waypoints using
  existing IK tolerances. Eight shared phases remain
  `above_object → pregrasp → grasp → squeeze → lift → hold → release → return`.
  Each segment uses the slower side's existing duration policy.
- `shared_timing.py`, `shared_packet.py`, and `shared_stream.py` share the same
  waypoint reference between preview and transport. Both arms must meet the
  existing completion policy before the phase advances. Hand arrival remains
  nonblocking. Existing numeric bounds and low-level protections remain active.
- `ros_execution/shared_hardware.py` defines a distinct hardware packet and
  source/plan hash validation. `shared_executor.py` reuses one publishing loop,
  actual Runtime preflight, controller/ownership checks, and both-arm aborts.
  It verifies real FANUC plugins, `sharpa_sdk` MIT hands, matching joint models,
  fresh feedback and the existing required force limiter before commands.

Inspected `/home/hao-fang/ws_fanuc/src/bimanual_manipulation` and
`dual_crx_control`: the actual contracts remain `sensor_msgs/msg/JointState` on
`/crx5ia/joint_targets` (12 named CRX joints),
`/sharpa/left_hand/joint_command` (22 joints), and
`/sharpa/right_hand/joint_command` (22 joints). There is one CRX publisher and one
100 Hz owner with aligned ROS stamps. The existing 500 Hz driver interpolator is
unchanged. No driver interface change or additional robot controller was needed.

`export/path_checks.json` records endpoint/nominal segment checks, sampled at the
existing arm/hand command-step resolutions. A partial object cloud is not a verified
collision solid. Sampled checks are not continuous collision certification.
Relative wrist constraints at IK waypoints do not guarantee a rigid relative pose
between waypoints or under driver tracking errors. These distinctions remain in
the artifacts; `continuous_path_verified` and physical verification flags stay false.
This first paired implementation uses synchronized linear references, not tactile MPC.

## Validation evidence, 2026-11-23

Code-ready and physically verified are separate: **no physical motion was performed
for this change**. Current physical grasp/lift performance remains unverified.

Real saved cloud `native_200k_execute_20261123_133500/object` was passed through
the actual GPU worker with the 200k checkpoint. It returned ten model-available
`both_full` Top-K pairs, no conversion rejections, and only the original Auto-Mode
branch. Local evidence: `output/dual_live/native_paired_worker_20261123/`.

Actual dual-model IK and collision checking on the saved measured scene selected
`both_full:c0440`, effort **4.7079386627 rad²**, and exported the complete synchronized
hardware packet. Original replay evidence:
`output/dual_live/paired_offline_20261123_155246/`.
The export uses five common lift waypoints; relative pose residuals at those points
were below 1e-12 m/rad. The saved scene predates the latest table recalibration;
this demonstrates offline integration, not current-scene safety or a reusable live goal.
The newly generated GPU predictions also selected `both_full:c0440` and produced
a validated hardware export at `output/dual_live/paired_offline_20261123_155709/`
(effort 4.7079381109 rad²). This last replay includes both arm and finger step
resolutions in nominal path sampling.

A second saved scene (`native_200k_execute_20261123_133625`) correctly returned
`NO_FEASIBLE_GRASP`: 19 pairs failed left IK and one exceeded hand limits. Neither
the model mask nor physical tolerances were relaxed to obtain an export.

The existing domain-88 DDS harness passed the full eight-stage paired cycle and
late-arm/nonblocking-hand case. Separate fault runs passed stale-right-feedback,
left-limit, and competing-publisher cases, each requesting stops on both CRX arms.
Evidence: `output/shared_mock/native_paired_20261123_retry/{both,late_hand_error}`
and `output/shared_mock/native_paired_faults_20261123/`.
An added SIGTERM cancellation case passed with both stop requests at
`output/shared_mock/native_paired_cancel_20261123/`.
An initial harness invocation lacked the child package path; a combined run later
hit a DDS recording stamp mismatch on the fault case. The isolated fault rerun
passed without changing checks or tolerances.

Final checks:

| Check | Result |
| --- | --- |
| `PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl .conda/bin/python -m unittest discover -s tests -v` | 604 tests, OK; 87 optional/environment skips |
| Sourced Jazzy `/usr/bin/python3 -m unittest discover -s tests -p 'test_right_ros_receiver.py' -v` | 8 passed, including both-side real-driver/review checks |
| `.conda/bin/python -m compileall -q src tests` | Passed |
| `ruff check src tests script` | Passed |
| `ruff format --check` on all 21 changed Python files | Passed |
| `ruff format --check src tests script` | Eight unchanged files require formatting, as before this change |
| `git diff --check` | Passed |

Ruff was already cached at
`/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff`; no dependencies
were added. Unit logs are `/tmp/ada_full_tests_final.log` and
`/tmp/ada_ros_receiver_tests_final.log`.

The bundled quick release gate fails the existing
Shadow expected-classification comparison. Repeating it on untouched `91400ef`
in `/tmp/ada-native-baseline-91400ef` produced the identical summary: 68 successes,
4 failures, 21 invalid initializations, 7 degraded, 0 execution errors. No golden
data or tolerances were changed. Subsequent hand gates were not reached.

Focused regressions include real native samples with source/checkpoint provenance
in `tests/fixtures/native_paired_200k`, single-hand equivalence, ranked fallback,
paired state preservation, inter-arm collision rejection, hardware/mock isolation,
review identity, common waypoint sampling, full paired CLI export, and cancellation
before any executor starts.
