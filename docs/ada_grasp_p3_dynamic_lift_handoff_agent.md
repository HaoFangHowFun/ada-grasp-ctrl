# Ada-Grasp-CTRL — Minimal MPC → Dynamic Lift Handoff

**Codex implementation task | 2026-11-13 | Right CRX-5iA + Right Sharpa Wave**

## Goal

Fix the **first Lift `check_start()` failure** after a successful real Ours MPC grasp+squeeze episode. The most recent operator-run physical test confirmed `native_complete` at index 48 (reference length 44), measured total normal force 16.375763 N (>15 N), and successful receiver→sender action result. **No Lift goal was sent** because sender compared post-MPC measured joints against the stale preview-time squeeze endpoint, failing `check_start()` at `right_J1`. The MPC joint trajectory is allowed to change during Grasp+Squeeze. A fixed preview Lift start and fixed absolute Lift IK target therefore cannot be reused as-is.

**Implement only a minimal, validated dynamic Lift handoff.** Do not change Ours controller, force thresholds, streaming/interpolation, drivers, contact adapter, or baseline Linear behavior. No autonomous robot operation.

## Source of truth / evidence

Repository: `/home/hao-fang/hugs-workspace/ada-grasp-ctrl`

Read the actual worktree (`git status`, relevant diffs) before editing; local changes may be ahead of GitHub. Do not discard or overwrite unrelated changes; do not commit/push without request.

Read:

- `docs/mpc-p3-native-complete-lift-handoff-2026-11-13.md` (latest physical trial report)
- `src/ada_grasp_ctrl/experiments/right_ros_sender.py` (`execute`, phase transition, `check_start` call)
- `src/ada_grasp_ctrl/pipeline/plan.py` (keyframes, fixed Lift generation, `check_start`, IK/path handling)
- `src/ada_grasp_ctrl/ros_execution/bridge.py` and receiving action/goal contract
- `src/ada_grasp_ctrl/mpc/live_execution.py` (native completion and final positions)
- Existing source for CRX FK/IK, trajectory review/validation and real robot feedback

Recorded evidence:

- `output/mpc_p3/active_20261113_111440/` (sender/receiver/action logs)
- `output/grasp_pipeline/p3_plan_20261113_111347/grouplive/export/motion_plan.json`
- Candidate `grouplive_c0003` / `c0003`

At native completion, measured arm joints (rad):

```text
right_J1 -1.043671126
right_J2 -0.993003288
right_J3  3.469827258
right_J4 -0.346057610
right_J5 -1.075276657
right_J6 -0.465424618
```

The old preview squeeze endpoint differs substantially (e.g. J1 ~0.04990 rad; J4 ~0.07573 rad). Existing `check_start()` arm tolerance is 0.01 rad; this check is correct for an *actual matching plan*, but the old plan no longer matches.

## Required behavior

1. **Mode scope:** Only for real `mpc_active` after **verified native Ours completion** of the combined Grasp+Squeeze episode. Do not alter `linear` or `mpc_passive`; do not alter earlier approach/pregrasp, hold/release/return without a demonstrated need. Native timeout, solver fault, cancel, stale feedback, invalid outcome, or abort must **never** trigger Lift.
2. **Fresh handoff snapshot:** Once the sender receives successful `native_complete`, obtain a **fresh, named, coherent 28-joint actual feedback snapshot** from the existing source. Use this *measured* joint state, not the old plan endpoint, not solely the last commanded target, and not an unverified earlier completion snapshot. Keep timestamp/provenance and require freshness/coherence using existing contracts. If no valid snapshot: stop without Lift.
3. **Dynamic world-vertical Lift:** Compute the **current actual right CRX tool/EEF world pose** using the existing FK and calibrated frames. Preserve the current EEF orientation and define the desired translation as **world +Z = +0.15 m** from the *current actual* pose (not local tool +Z and not old preview FK). Reuse the current IK/trajectory generation infrastructure, using measured arm configuration as seed and start. Require a valid IK solution, joint limits, and applicable existing path/collision/table checks; if these cannot be established, **fail closed** (report what is unverified and do not send Lift). Do not pretend the old preview's `collision_checked=false`, `path_checked=false` flags certify a new path.
4. **Hand hold:** Preserve the grasp achieved at MPC completion, using the fresh measured 22 hand joints as the new hand hold target for the Linear Lift (or, if existing hand backend requires a different explicit hold policy, justify it and verify no jump). **Do not silently revert to the preview's `retain_selected_squeeze` finger angles.** No new force control during Lift.
5. **Build a new coherent Lift goal:** Create a new Lift motion segment with a **start keyframe matching the fresh measured 28-joint snapshot**, a newly computed arm target, and held hand posture. Carry provenance/config and the existing reviewed-goal fields through the sender→receiver action contract. Both sender and receiver must validate the same dynamic start and goal; avoid bypassing checks or weakening `check_start()` tolerance. If an existing receiver review token/plan signature ties to an immutable preview, implement the smallest correct regeneration/validation path rather than forging or disabling it.
6. **Prevent start races:** A fresh feedback measurement used by sender is not guaranteed identical to receiver feedback at execution. Keep existing start checks and handle legitimate small feedback evolution under their present tolerance; do not inject a fake measured state, spoof timestamps, or permanently disable checks. If a new start mismatch occurs, record the actual delta and abort safely.
7. **Failure behavior:** On IK failure, invalid frame, stale feedback, missing model, validation failure or action rejection, do not publish Lift; return an explicit reason to sender. No automatic retry that might move hardware or command the former absolute Lift target. Preserve existing cancellation/ownership semantics.

## Constraints — do NOT change

- Original Ours solver, Jacobians, stage switching, `should_stop`, `max_steps`, force trajectory, `final_sum_force=15 N`, friction coefficient, measured-force convention or contact policy.
- The now-successful 10 Hz MPC → 100 Hz **linear** streamer (`GuardedStream` refactor); no added velocity/acceleration planner or Ada joint-limit guard.
- Robot drivers, Sharpa SDK, `bimanual_manipulation` existing safety protections and low-level control settings.
- Existing default Linear workflow, existing start verification for other phases, or the 0.01 rad arm `check_start()` tolerance.
- No speculative full motion-planning rewrite; favor a small helper + a narrowly scoped sender hook + corresponding receiving contract change only when needed.

## Offline tests (required before any physical trial)

- Read-only replay using **recorded actual native completion `final_positions`** and the old plan. Assert the old Lift start fails exactly as reported; then assert the new dynamic Lift start equals the recorded measured state, with world +Z 0.15 m target and preserved EEF orientation.
- FK/IK round-trip / endpoint and held-hand 22-joint mapping tests using current robot model; verify names/units and no arm/hand order swap.
- Sender **and receiver** acceptance for an internally coherent dynamic Lift goal; original `check_start()` still detects genuinely wrong/stale positions.
- Failure cases: native timeout/abort; stale/incoherent feedback; unreachable IK; invalid motion/path, missing review or source provenance; sender/receiver start mismatch; no Lift publication in all these cases.
- Regression: `linear` and `mpc_passive` unchanged (matched input/command fixtures), original `mpc_active` Grasp+Squeeze native behavior unchanged, no change to streamer and no forced MPC → Linear endpoint reset.
- Log from a synthetic/read-only run: measured start, previous preview start and per-axis difference, current and desired EEF world poses, IK seed/solution, hand hold source, path/limit validation result, reviewed-goal identifiers, and success/failure reason. Label synthetic success as synthetic, not proof of physical lift.

## Deliverables

1. Minimal implementation diff with the exact entrypoint(s), including an explicit `mpc_active && native_complete` handoff branch.
2. Short technical note: how fresh 28-joint snapshot flows into FK → world +Z 15 cm → IK → new Linear Lift goal → sender/receiver start validation; where hand posture is held.
3. Commands and results for focused tests plus one concrete offline regression from `active_20261113_111440` artifacts; any outstanding blockers stated explicitly.
4. Separate **operator-only** steps for a later physical trial; never start hardware or issue commands as part of implementation/testing.

**Success criterion:** The old preview mismatch is solved by reconstructing a valid Lift plan from real post-MPC state, **not by suppressing or relaxing the safety check**. Grasp remains controlled by Ours; Lift remains conventional Linear without force control.
