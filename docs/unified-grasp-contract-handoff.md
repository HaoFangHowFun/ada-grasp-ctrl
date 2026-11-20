# Unified grasp contract — Task 1 handoff

Base: `main` at `10bef3ba0bd23a80ef16f9371b1357cd0a484a4d`.
Implementation branch: `task1/unified-graspplan-interface`.
Integrated into `/home/hao-fang/hugs-workspace/ada-grasp-ctrl` on `main` at the
user's request on 2026-11-20; the four deliverables were initially left uncommitted.

## Stable API for Tasks 2–4

Import from `ada_grasp_ctrl.grasp_selection.contracts`. `GraspCandidate`,
`GraspPlan` and `StageTarget` are TypedDicts; their constructors do **not** validate.
Use `validate_grasp_candidate` / `validate_grasp_plan` at input boundaries. Each
returns an independent JSON-compatible copy and raises `ValueError` for invalid
data. Serialize with `json.dumps(record, allow_nan=False)` and revalidate after
loading. The validators also reject nonfinite values in provenance.

Both records use `schema_version: 1`, stable `candidate_id`, an explicit `mode`,
`frame_id: world`, `unit: m`, `joint_unit: rad`, per-arm `targets`, and `provenance`.
The candidate extends the existing per-arm schema with a required `mode`; it
retains `required_arms`. The selected plan uses `active_arms` and optional
`available_arms`, with `execution_ready: false` and `hardware_ready: false`.

| Field | Meaning |
| --- | --- |
| `available_arms` | Run-level eligibility; legacy selector calls this `enabled_arms`. |
| `required_arms` | Arms needed by a candidate, before selection. |
| `active_arms` | Arms chosen for a plan; nonempty and a subset of availability when supplied. |
| `mode` | Opaque, nonempty mode label; no fixed enum, ranking or inference capability implied. |

Arm lists reject duplicates and names other than `left` / `right`. Target keys
must exactly match candidate requirements or selected active arms. Every arm
retains exactly `pregrasp`, `grasp`, `squeeze`; each target contains a proper 4×4
`T_world_hand` mapping that side's hand-flange coordinates into world coordinates,
`frame_id: world`, `pose_frame: <side>_hand_flange`, and all 22 canonical named
`joint_positions_rad`. Matrix translation is in metres. No palm/TCP substitution,
camera transform, quaternion reordering, extrapolation or joint reordering occurs.
If redundant `position_m` / `quaternion_wxyz` fields are present, they must agree
with the matrix. Redundant WXYZ retains its original sign. The rigid-transform
checks reuse the existing per-arm validator and its tolerances.

Mode labels remain extensible for the five intended modes; the contract does not
infer arms from their names or claim that the current checkpoint can generate
those modes. The caller supplies the mode and explicit arm requirements. Mode
eligibility and mode-to-arm policy belong to Task 2.

## Small data-only examples

These synthetic identity targets demonstrate serialization only; they are not
reachable or approved hardware targets.

```python
from copy import deepcopy
import json
from ada_grasp_ctrl.experiments.right_ros_contract import side_names
from ada_grasp_ctrl.grasp_selection.contracts import (
    validate_grasp_candidate, validate_grasp_plan, plan_from_candidate,
)

def example_candidate(side):
    target = dict(
        frame_id="world", pose_frame=f"{side}_hand_flange",
        T_world_hand=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        joint_positions_rad=dict.fromkeys(side_names(side)[1], 0.0),
    )
    return validate_grasp_candidate(dict(
        schema_version=1, candidate_id=f"{side}:c0004", mode=f"{side}_full",
        required_arms=[side], frame_id="world", unit="m", joint_unit="rad",
        targets={side: {s: deepcopy(target) for s in ("pregrasp", "grasp", "squeeze")}},
        provenance={"source": "synthetic_documentation_example"},
    ))

left_candidate = example_candidate("left")
right_candidate = example_candidate("right")
left = plan_from_candidate(left_candidate, available_arms=["left"])
right = plan_from_candidate(right_candidate, available_arms=["left", "right"])
assert right["active_arms"] == ["right"]  # Both eligible; only right selected.

both_candidate = validate_grasp_candidate({
    **right_candidate,
    "candidate_id": "both:c0001", "mode": "both_full",
    "required_arms": ["left", "right"],
    "targets": {**left_candidate["targets"], **right_candidate["targets"]},
})
both = plan_from_candidate(both_candidate, available_arms=["left", "right"])
assert both["execution_ready"] is False
assert validate_grasp_plan(json.loads(json.dumps(both, allow_nan=False))) == both
```

## Legacy adapters and compatibility

- `candidate_from_schema1(existing_per_arm_row, mode="right_full")` adds the
  explicitly supplied mode, preserving ID, stage targets, source record, reflection
  metadata and other provenance. The same API accepts left schema-1 rows with
  `mode="left_full"`. Conflicting existing mode labels are rejected.
- `candidate_to_schema1(canonical_candidate)` removes only `mode` and returns an
  independent legacy per-arm row. Keep canonical mode metadata separately when
  using this intentionally lossy legacy projection.
- For decoded original single-right records, keep using
  `multi_arm.from_right_record(raw, provenance)` first, then
  `candidate_from_schema1`. The existing left reflection adapter still requires
  its explicit plane; the new contract performs no additional mirroring.
- `plan_from_selection(selection, mode="right_full")` accepts the **full** result
  of `multi_arm.select_candidates`, maps `enabled_arms` to `available_arms`, and
  derives the single active arm from the selected candidate. It checks the
  selected ID, `selected_side` and `required_arms` agree. No feasible selection
  raises `ValueError` and produces no plan. Use `mode="left_full"` for a left winner.

Legacy input dictionaries and serializations are never mutated. In particular,
`selected_side` remains on all current selection/export paths. The older
`grasp_selection.types.GraspCandidate` and package-level import are unchanged.
The standalone single-right `SelectionResult`, derived execution motion plans
and ROS packets retain their existing APIs; this adapter takes the per-arm
selector result, not those distinct formats. New canonical records coexist with
them and are not automatically written into their files or hash-bound packets.

Task 2 should consume canonical candidates and explicit run availability, then
produce `plan_from_candidate(chosen, available_arms=...)`. Task 3 should consume
the plan's explicit stage targets; time scaling and CRX trajectories remain
separate artifacts. Task 4 must provide execution capability gates and review/
approval checks before building commands. Canonical validation is data validation,
not a reachability, collision, trajectory or hardware-readiness result. Even a
single-arm canonical plan cannot claim readiness. Both-arm candidates remain
rejected by the existing selector as `unsupported_multi_arm_candidate`; neither
legacy motion-plan nor ROS-packet validation accepts a canonical plan as input.

The implementation reuses the Ada per-arm validator and therefore runs in the
maintained Ada numeric environment (including its robot-model dependencies).
It does not add a ROS-system-Python transport API or implement multimode inference,
Human Prior, synchronization, Place, MPC, or hardware execution.

## Changed files and validation

- `src/ada_grasp_ctrl/grasp_selection/contracts.py`: additive contracts/adapters.
- `tests/test_grasp_contracts.py`: canonical, adapter and capability-gate coverage.
- `docs/unified-grasp-contract-handoff.md`: API and downstream handoff.
- `docs/dual-live-grasp-pipeline.md`: link to this contract.

Validation on 2026-11-20, from `/home/hao-fang/hugs-workspace/ada-grasp-ctrl-task1`:

```bash
env PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_task1_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 /home/hao-fang/hugs-workspace/ada-grasp-ctrl/.conda/bin/python \
  -m unittest discover -s tests -p test_grasp_contracts.py -v

env PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_task1_mpl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 10 /home/hao-fang/hugs-workspace/ada-grasp-ctrl/.conda/bin/python \
  -m unittest discover -s tests -v > /tmp/ada-task1-unittest-with-assets.log 2>&1

/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff check \
  src/ada_grasp_ctrl/grasp_selection/contracts.py tests/test_grasp_contracts.py
/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff format --check \
  src/ada_grasp_ctrl/grasp_selection/contracts.py tests/test_grasp_contracts.py
/home/hao-fang/hugs-workspace/ada-grasp-ctrl/.conda/bin/python -m compileall -q src tests
git diff --check
```

Results: focused **11/11 passed**; full suite **517 run, 430 passed, 87 skipped,
zero failures/errors**, 35.319 seconds. The full run includes `test_grasp_selection`,
`test_phase2_pipeline`, `test_dual_live_pipeline`, `test_grasp_pipeline`,
`test_mpc_phase2` (including exact linear reference/preview parity), and
`test_task1_active_pipeline` (mocked active orchestration). Changed-file lint,
format, compile and whitespace checks passed. The Python example above was also
extracted and executed successfully.

The first full-suite attempt failed because the new worktree had no submodule
meshes: 41 failed assertions and one error, all from missing Shadow/Allegro assets.
Restoring the local `mujoco_menagerie` at its exact pin
`0601a3160d773533b7c425d5484bf29bca7399ff` with a sparse checkout of `shadow_hand`
and `wonik_allegro` resolved those failures. No submodule pin or source was changed.
The successful run reuses the existing maintained Ada Python environment.

Fixture/ROS-dependent skips remain skips; this is offline regression evidence,
not new real-hardware or recorded-fixture execution evidence. No existing Python
source was edited, including the right-hand `linear`/`mpc_active` paths, selector,
exporter and ROS contracts. The original checkout was clean during isolated validation. Validation used
one numerical thread, reduced CPU priority and no GPU while the separate data
evaluation continued. During validation, no drivers, physical motion, commits or pushes were issued.

After integration, the focused command above was rerun from the original checkout:
all 11 tests passed (0.129 seconds). Changed-file lint, format, compile and
`git diff --check` also passed there. The full-suite result above is from the
isolated worktree before integration.
