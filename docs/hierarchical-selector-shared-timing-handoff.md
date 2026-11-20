# Tasks 2/3 — selector and shared timing handoff

Implemented directly in the original checkout on `main`, based on
`10bef3ba0bd23a80ef16f9371b1357cd0a484a4d`. Existing uncommitted Task 1 files were
preserved. The [Task 1 contract](unified-grasp-contract-handoff.md) remains unchanged.
These additive APIs are offline planning utilities; live inference, exporters,
ROS executors, drivers and original MPC controllers are unchanged.

## Task 2: public selection API

Module: `ada_grasp_ctrl.grasp_selection.hierarchical`.

```python
select_hierarchical(
    robot: object, sim: object, batch: dict, prior: dict,
    available_arms: object, q_current: object, *,
    supported_modes: Sequence[str] = ("left_full", "right_full"),
    enabled_modes: Sequence[str] | None = None,
    weights_by_arm: dict[str, object] | None = None,
    max_nfev: int = 200,
) -> dict[str, Any]
```

`batch` is `{"schema_version": 1, "candidates_by_mode": {mode: [GraspCandidate, ...]}}`.
Candidate mode must agree with its group and IDs must be globally unique across
groups. Malformed identifiable candidates receive `invalid_candidate` diagnostics;
invalid collection structure, duplicate/missing IDs, run state or policies fail
preflight with `ValueError`.

Supply one explicit versioned priority form:

```json
{"schema_version": 1, "kind": "scores", "scores": {"left_full": 0.9, "right_full": 0.4}}
```

or `{"schema_version": 1, "kind": "ranking", "ranking": ["left_full", "right_full"]}`.
Scores are finite external priority values, not necessarily probabilities. Higher
scores rank first; ties use lexical mode name. An explicit ranking preserves its
listed order. Raw `K_t` / counts inputs are rejected: their producer must explicitly
choose a ranking or score interpretation. No inference or Human Prior model runs
inside this API. Supported modes are caller capability labels, not inferred from
the checkpoint, mode name or available-arm count.

For each eligible mode in order, the existing `multi_arm.select_candidates` backend
checks single-arm candidates against the current full 56-joint state. The wrapper
retains its seeds, IK, joint limits, contact/penetration checks and unwrapped
weighted squared CRX displacement cost. Each available arm may have six explicit
nonnegative weights; otherwise all weights are one. Within a mode the minimum
`(effort, candidate_id, seed_index)` wins. Effort is never compared between modes.
The original single-mode selector remains independently callable and unchanged.

The report includes mode ranks/scores, candidate/evaluated/feasible counts,
candidate diagnostics with complete backend attempts, selected mode/side/effort,
the legacy selected payload (including `selected_side`), and the canonical `plan`.
Unavailable arms, disabled/unsupported modes and unsupported two-arm candidates
are reported explicitly without fabricated IK failures. Groups omitted by the
prior have `missing_priority` and receive no IK evaluation. After a winner, lower
eligible modes have `not_evaluated_after_selection`. All failures or empty inputs
return `status: NO_FEASIBLE_GRASP`, `selected: null`, `plan: null`, and no execution
targets. Both readiness flags remain false even for a successful selection.

Deterministic fallback fixture: left target X=6 makes every fixture IK branch fail;
right target X=0.2 maps to six 0.2-rad CRX joints from zero. Its abbreviated report is:

```json
{
  "status": "SELECTED", "selected_mode": "right_full", "selected_side": "right",
  "effort": 0.24000000000000005,
  "mode_trace": [
    {"mode": "left_full", "rank": 1, "score": 0.9,
     "status": "no_feasible_candidate", "candidate_count": 1,
     "reasons": ["no_feasible_candidate"]},
    {"mode": "right_full", "rank": 2, "score": 0.4,
     "status": "selected", "candidate_count": 1, "reasons": []}
  ]
}
```

The first mode's full candidate trace retains `no_feasible_ik_branch` and each
`grasp:ik_rejected` attempt. These are deterministic test-double results, not
robot reachability evidence. See `tests/test_hierarchical_selection.py` for this
fixture and the counterexample where a feasible higher-priority mode wins despite
a cheaper candidate in another mode.

## Task 3: public timing API

Module: `ada_grasp_ctrl.pipeline.shared_timing`.

```python
build_stage_schedule(
    plan: GraspPlan,
    start_positions_rad: dict[str, dict[str, float]],
    stage_targets_rad: dict[str, dict[str, dict[str, float]]], *,
    policies_by_arm: dict[str, dict[str, float]] | None = None,
    stage_start_time: float = 0.0,
    settle_time_s: float = 0.3, hold_s: float = 5.0,
) -> dict[str, Any]
```

The canonical plan contains flange targets and fingers, not solved CRX joint
endpoints. Supply already-solved, named **6 CRX + 22 Sharpa** positions for each
active side: `start_positions_rad[side][joint]` and
`stage_targets_rad[phase][side][joint]`. Timing performs no IK or FK. The caller
owns CRX endpoint/pose consistency and joint-position-limit feasibility. Grasp
phase finger endpoints must exactly match the selected canonical plan.

Only `plan.active_arms` participate. A measured passive side may be present in
the start snapshot; it is ignored and never emitted in the schedule. Each phase's
targets and supplied policies must match the active sides exactly. Joint vectors
are ordered by `right_ros_contract.side_names`, independent of dictionary order.
Phase order follows existing `PHASE_SEQUENCES`: pregrasp/grasp/squeeze, optional
lift or the complete hold/release/return cycle, optionally prefixed by
`above_object`. Above Object / Locate on Top uses that existing `above_object`
name. The utility never invents missing endpoints.

Each moving side's duration calls the existing sender `motion_duration`:

```text
T_a = 1.05 * max(2.0,
                1.875 * max_arm_delta / arm_speed,
                sqrt(5.774 * max_arm_delta / arm_acceleration),
                1.875 * max_hand_delta / hand_speed,
                sqrt(5.774 * max_hand_delta / hand_acceleration)) + 0.1
T_stage = max(T_a for active sides)
```

Defaults remain 0.15/0.4 rad/s and 0.3/0.8 rad/s² for arm/hand. Optional policies
allow tighter limits or a longer minimum duration, never relaxed limits or a
shorter 2s floor. The existing 300s duration bound is enforced, never clamped.
Stationary hold retains the existing `hold_s` exception to the movement floor;
it must keep all lift joints. Lift retains squeeze fingers, release retains the
lifted arm and opens fingers, and return restores measured arm start with fingers
open. Existing positive settle intervals follow each phase, including the last.

The schedule records each side's required duration, one common start/end,
`settled_time`, ordered start/end joints, policies and the source canonical plan.
`validate_stage_schedule(schedule)` rebuilds these values and rejects inconsistent
serialized data. It is consistency checking, not cryptographic provenance or
execution approval. `sample_stage(schedule, phase, now, *, stage_start_time=None)`
returns a shared progress fraction and both sides' linear reference vectors:
`q = q0 + clip((now - t0) / T_stage, 0, 1) * (qf - q0)`. It reuses the existing
pure linear-reference math; no MPC controller runs.

Asymmetric deterministic fixture (`tests/test_shared_stage_timing.py`): arm
distances left=0.1, right=0.4 rad; both hand distances=0.2 rad; left arm speed=0.05
and right hand speed=0.2 rad/s; other limits use defaults. The first stage is:

```json
{
  "phase": "pregrasp", "stage_start_time": 10.0, "stage_end_time": 15.35,
  "settled_time": 15.65, "duration_s": 5.35,
  "required_durations_s": {"left": 4.0375, "right": 5.35}
}
```

Both references reach halfway at 12.675 and their own endpoints at 15.35.
This synchronizes **command references**, not physical tracking. Inherited
acceleration-based duration margins originate in the quintic policy; an ideal
linear segment still has nonzero velocity jumps at its endpoints. Accordingly
`acceleration_continuous`, `continuous_path_verified`, `execution_ready` and
`hardware_ready` remain false. Longer duration does not prove smooth acceleration.

## Task 4 integration boundary

1. Consume the successful Task 2 `plan` and selected backend joint solutions.
   For the three grasp stages, map `selected.stages[phase].q56` through the
   existing side slice and canonical names. The integration test exercises this.
   Supply existing upstream approach/lift/cycle endpoints separately.
2. Build and validate the schedule before preview or executor conversion. It is
   a new planning artifact, not a legacy ROS packet; no wire schema was changed.
3. Treat timestamps as nominal. A clock reaching a phase boundary does not
   authorize stage advancement. Retain existing arrival, settle, timeout and
   abort policies. When predecessor completion permits the next phase, sample it
   with one actual `stage_start_time` shared by all active sides. Hold starts after
   lift succeeds. A passive side does not become a synchronization participant.
4. Preserve lower-level speed/acceleration/step/feedback protections and physical
   approval gates. Two-arm schedule data cannot bypass the selector's unsupported
   two-arm gate or authorize dual-active execution. No new ROS barrier framework,
   driver interpolation change, Place support or MPC mathematics is included.

## Files and verification

New files: `grasp_selection/hierarchical.py`, `pipeline/shared_timing.py` (under
`src/ada_grasp_ctrl/`), `tests/test_hierarchical_selection.py`,
`tests/test_shared_stage_timing.py`, and this handoff. Existing Task 1 and runtime
files are unchanged. Test fixtures use analytical IK and synthetic poses; no
hardware, inference checkpoint, network connection or GPU is required. The timing
module also has an import test that blocks sockets and checks no ROS/Torch import.

Validation on 2026-11-20, from `/home/hao-fang/hugs-workspace/ada-grasp-ctrl`:

```bash
env PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_task23_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 .conda/bin/python -m unittest discover -s tests -p test_hierarchical_selection.py -v
env PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_task23_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 .conda/bin/python -m unittest discover -s tests -p test_shared_stage_timing.py -v
env PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_task23_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 .conda/bin/python -m unittest discover -s tests -v > /tmp/ada-task23-unittest.log 2>&1

/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff check src tests script
/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff format --check \
  src/ada_grasp_ctrl/grasp_selection/hierarchical.py \
  src/ada_grasp_ctrl/pipeline/shared_timing.py \
  tests/test_hierarchical_selection.py tests/test_shared_stage_timing.py
.conda/bin/python -m compileall -q src tests
git diff --check
```

Results: Task 2 **11/11 passed**; Task 3 including the selection-to-timing integration
**14/14 passed**. Full suite: **547 run, 463 passed, 84 skipped, zero failures/errors**
in 35.755 seconds. Existing selector, dual pipeline, legacy ROS-packet, linear
reference and offline `mpc_active` regressions pass. Fixture/ROS-dependent skips
remain skips; no new physical or ROS execution evidence is claimed.

Whole-tree Ruff lint, new-file format, compile and whitespace checks passed.
The canonical whole-tree `ruff format --check src tests script` was also run:
it reports eight pre-existing unformatted files (the two `run_crx_sharpa_phase*`
scripts, `crx_sharpa_adaptive.py`, `crx_sharpa_demo.py`, `utils/viewer_session.py`,
and tests `test_crx_sharpa_demo.py`, `test_crx_sharpa_preview.py`, `test_dummy_arm_ik.py`).
Those files are unchanged. All four Task 1 deliverables were checked against
pre-task SHA-256 hashes and remain byte-for-byte identical.

The separate data evaluation remained running. Tests used reduced CPU priority,
one numerical thread and no GPU. During validation, no commits, pushes, hardware drivers or physical
motion were issued. Live executor hookup and dual-active hardware capability
remain Task 4 work.
