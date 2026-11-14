# Codex Implementation Task — Task 1: One-Command Real MPC Grasp Pipeline (Minimal Refactor)

**Date:** 2026-11-13\
**Repository:** https://github.com/HaoFangHowFun/ada-grasp-ctrl\
**Known baseline:** `main` commit `ecc07ce920ebf6f1d285f925f9dff025b11d6c9e` — *Complete MPC dynamic lift handoff and record hardware milestone*\
**Scope:** **Right FANUC CRX-5iA + Right Sharpa Wave, hardware `mpc_active` only**\
**Environment:** ROS 2 Jazzy; previously successful hardware run used `ROS_DOMAIN_ID=87`.

## Agent mandate

Implement **Task 1 only**: make the *already validated* right-hand real MPC grasp cycle operate through one command, eliminate manual ROI / preview / authorization steps, open the existing port-8089 viewer immediately, and reduce redundant MPC→Lift handoff computation. Do **not** reimplement the controller, change force policy, or expand to left/bimanual/simulation. Keep the diff minimal and reusable. **Do not operate real hardware or claim hardware validation autonomously.**

First inspect the current worktree, `git status`, latest commits, and the relevant code. Preserve uncommitted changes. Compare against the above baseline if the checkout has moved. The successful run is described in `docs/mpc-p3-real-grasp-lift-milestone-2026-11-13.md`, execution ID `active_20261113_132624`; operator observed successful grasp, followed by world +Z 15 cm lift, hold, release, return. All seven receiver actions completed. Treat this as a **single-run milestone**, not a statistically validated success rate. Keep its source/config/model traceability.

## Desired operator experience

The canonical entry point **already exists**: `script/run_grasp_pipeline.py`. **Reuse it; do not create a generic `run.py`, duplicate entry point, or monolithic replacement.**

Target CLI, for a configured/ready hardware workstation:

```bash
python script/run_grasp_pipeline.py --controller-mode mpc_active
```

This should be sufficient when the normal workspace configuration provides hardware backend, ROS domain, dependencies, and output location. Resolve the currently required `--output-root` automatically to a **unique timestamped run directory** for this particular mode; do not overwrite runs. Explicit configuration/CLI overrides may remain. Set hardware-specific defaults only for this explicitly requested real-active mode; do not silently change existing `linear`, `mpc_passive`, replay, mock, or dry-run defaults.

The explicit `--controller-mode mpc_active` token is the operator's **single authorization for this run**. No second interactive Enter gate, confirmation token, web click, or manual terminal transfer is permitted. However, reject incomplete or ambiguous hardware configuration, missing services, stale feedback, mismatched model/calibration, or other genuinely necessary preconditions; authorization is **not** a waiver of these checks. Do not infer hardware authorization from a default mode or a previously saved run. Preserve clean Ctrl+C/cancel/cleanup behavior and existing low-level safety behavior.

**No manual terminal coordination:** the one process should coordinate required existing sender/receiver components and existing supported services, capture and infer from the current RealSense cloud, select candidate, plan, execute MPC grasp+squeeze, then linear lift/hold/release/return. Do not rewrite subprocess-separated Python 3.10 vs ROS Python 3.12 interfaces: manage them transparently from the current launcher. Do not presume that the launcher should start or change real robot drivers when those are intentionally externally managed; detect readiness and fail clearly if not available.

## Work item A — Correctly route the unified entry point to real `mpc_active`

Inspect at least:
- `script/run_grasp_pipeline.py`
- `src/ada_grasp_ctrl/pipeline/runner.py`
- `src/ada_grasp_ctrl/experiments/right_ros_sender.py`
- `src/ada_grasp_ctrl/ros_execution/bridge.py`
- `src/ada_grasp_ctrl/mpc/live_execution.py`
- `src/ada_grasp_ctrl/config/experiments/grasp_pipeline.yaml`

**Known issue in the inspected `main` baseline:** after the preview gate, `pipeline/runner.py` currently branches `mpc_active` / `mpc_passive` to `mpc.offline.run_packet()` and returns; the real `mpc_active` path is elsewhere and requires additional manual invocation. Fix **only** the hardware `mpc_active` branch so it launches the existing real ROS receiver and sender with correct mode, target packet, native MPC Python environment, model report, ROS setup, hardware enable/authorization semantics, and all required args. Do not allow `mpc_active` to silently fall back to offline/mock execution. Preserve the current independent `linear` and `mpc_passive` behavior.

The receiver currently checks `hardware_enable` / `operator_confirmed` in `ros_execution/bridge.py` and `mpc/live_execution.py`. Feed these existing internal booleans from **the one explicit CLI choice** for this mode; do not delete the guards or require another human action. Avoid leaking authorization to other modes.

Plan/selection/export provenance and current execution logs should remain intact. Avoid redoing identical snapshot/probe/process setup if the same fresh state can be reliably reused; but **do not** use stale robot state merely to save time.

## Work item B — Automatic single-object extraction, no ROI prompt

Inspect:
- `src/ada_grasp_ctrl/experiments/fixed_camera.py` (`choose_roi`, `extract_object`, existing table-height filtering/clustering)
- `src/ada_grasp_ctrl/config/experiments/fixed_camera.yaml`
- `src/ada_grasp_ctrl/pipeline/runner.py` (`perceive` and `--roi` / `--cluster` flow)

For **this hardware `mpc_active` automatic path**, use a configured, calibrated fixed camera workspace ROI (existing relevant workspace geometry or full valid camera area if appropriate) without calling `cv2.selectROI()` or waiting for Enter. Reuse existing real-world plane-height filtering, voxel handling, and clustering; **no new segmentation model**. Require exactly **one valid cluster**:
- 0 → abort with clear `NO_OBJECT_DETECTED` diagnostic;
- 1 → continue with existing object cloud;
- >1 → abort with `MULTIPLE_OBJECTS_DETECTED` diagnostic;
- invalid/insufficient cloud → abort with an explicit reason.

Do **not** automatically choose a largest cluster or a specific cluster if multiple are valid. Document that clustering is a pragmatic single-object proxy (objects may split/merge), not semantic instance segmentation. Preserve manual `--roi` and debug preview functionality for other explicit workflows. Do not refit/calibrate camera extrinsics or the already accepted table plane during each normal grasp run.

## Work item C — Port 8089 starts at launch; is not an execution gate

Inspect `src/ada_grasp_ctrl/pipeline/runner.py` and `src/ada_grasp_ctrl/pipeline/viewer.py`.

For this real-active path, start port 8089 **at the beginning**, before expensive perception/inference, and show meaningful stage/status updates as preparation and execution progress. Remove only the **operator-confirmation dependency** (`wait_for_enter`, interactive gate) from the automatic path. The viewer must remain observational: opening, closing, or refreshing the browser must not pause execution. Avoid delaying the control pipeline waiting for a UI approval or full preview construction. Do not remove necessary perception/selection/plan validity checks.

**Task 2 (not this task):** high-rate robot-state 3D rendering, contact points/arrows, and prominent force readouts. In Task 1 only, make a small clean status/data hook if useful; do not build the real-time visualizer yet. Maintain the existing viewer/debug behavior outside this mode.

If 8089 is occupied, provide a clear conflict failure unless it is demonstrably a reusable **same-run** process; do not attach to an unrelated service.

## Work item D — Minimal post-MPC dynamic Lift, exactly one planning/feasibility pass

Inspect:
- `src/ada_grasp_ctrl/pipeline/dynamic_lift.py`
- `src/ada_grasp_ctrl/experiments/right_ros_sender.py`
- `src/ada_grasp_ctrl/ros_execution/bridge.py`
- `src/ada_grasp_ctrl/pipeline/plan.py`

**Current behavior:** after `native_complete`, sender captures fresh measured 28-joint state and invokes `dynamic_lift.compute()` (separate native Python process), generating world +Z 0.15 m lift with fixed current flange orientation and held hand joints. Receiver invokes an independent `dynamic_lift.compute()` and fully recomputes the geometry. Offline geometry computation was previously ~1.42 s *per pass*, not including process startup. This redundancy contributes to the perceptible pause before Lift, but exact hardware phase timings are not yet separately measured.

**New narrow contract:**
1. After a real MPC `native_complete` (not mere nominal stage completion), acquire a new **fresh/coherent actual** six arm + 22 hand feedback snapshot.
2. Run FK → world +Z **0.15 m** endpoint, unchanged flange orientation → bounded IK **once**. Preserve the 22 **measured** finger joint angles at completion.
3. Run necessary **joint-limit and actual Lift-path feasibility** against known robot/table geometry **once** in the trusted planning path. Reject infeasible motions. No new force control during Lift.
4. Pass the resulting immutable, source-bound lift plan to the existing receiver. The receiver must validate **goal structure/identity, command freshness, current joint start match, and legitimate phase/native-complete state** before dispatch. **Do not solve IK or repeat full path sampling in the receiver.** Treat duplicate full geometry replay as the work to remove, not all validation.
5. Preserve the existing **0.01 rad arm start tolerance**, relevant hand tolerance, action ownership, watchdogs, cancellation, ROS command validity and driver safety. Never substitute an old pre-MPC squeeze endpoint as Lift start. A robot that drifts too far must reject rather than jump to the goal.
6. Hold uses the derived Lift endpoint; release opens the hand at that arm position; return retains the prior baseline destination. Keep existing phase motion control and current 10 Hz MPC / 100 Hz streaming untouched.

Keep the change minimal. Use existing handoff structure and signed/hash-bound source identity where available, but remove the redundant independent **FK/IK/full-path recomputation**. Clearly document the changed trust boundary: the receiver accepts a single validated, source-bound lift plan from the controlled sender/runner plus its own start/goal freshness checks. If a required safety condition cannot be preserved with this contract, flag it explicitly rather than silently removing it.

Do not confuse the original Ours `15 N` **completion threshold** with a hard force limit. Do not modify native solver, SLSQP parameters, friction coefficient, completion logic, gains, contact force signs or tactile calibration.

## Work item E — Remove avoidable orchestration overhead; add objective timing

Measure and log wall-clock durations per stage, e.g.:
- ROS readiness / service setup
- camera capture and object extraction
- DexLearn inference
- grasp selection / planning
- preview/status initialization (should overlap with preparation)
- each action/phase
- `native_complete` → fresh snapshot → Lift plan/feasibility → Lift goal submitted → first Lift command
- cleanup

Log detailed timestamps in JSON/JSONL and a short terminal summary. Compare against existing run artifacts where clocks are comparable; **do not report timestamps of action directory names as exact MPC-to-Lift latency**. Existing 5-second Hold is **intentional**, not a performance bug. Preserve necessary settle/arrival behavior and speed limits. Do not speed up the trajectory by silently increasing velocities or changing the existing controller interpolation. Optimization target is redundant computation/launch/manual gates, not motion dynamics.

## Non-goals / strict invariants

- No left CRX, left Sharpa, dual-arm, simulation refactor or broad interface redesign.
- No modifications to original Ours solver (`utils/grasp_controller.py`, native completion semantics), 15 N completion threshold, 10 Hz solver frequency, 100 Hz command streaming, existing low-level driver or `bimanual_manipulation` safety.
- Do not alter DexLearn checkpoint/inference algorithm, candidate selector scoring/IK algorithm, calibrated world/table geometry or camera driver.
- Do not remove meaningful command-owner, feedback freshness, start consistency, joint limits, collision-feasibility, or abort checks.
- Preserve functional `linear`, `mpc_passive`, replay and existing explicit debugging tools; do not secretly enable active hardware motion in these modes.
- Avoid unnecessary code movement and extensive architecture cleanup. **Smallest coherent changes win.**

## Validation and acceptance

Implement tests (prefer existing test suites and fixtures) covering:

1. **One command / no prompts:** the explicit real-active CLI maps to hardware receiver+sender with correct args, auto output location, no `input()` / `cv2.selectROI()` / viewer approval; no unrequested hardware activation in other modes.
2. **Single object:** 0/1/2+ valid clusters; low-quality point cloud; accepted table/camera provenance; manual ROI still works as a debug option.
3. **Viewer:** port 8089 starts before heavy inference, displays stage transitions, and has no gate; occupied port fails clearly; UI disconnect doesn't stop the pipeline.
4. **MPC:** native_complete required before Lift; no solver/streaming policy changes; safe cancel / stale feedback / timeout / rejected proposal paths still abort.
5. **Minimal Lift:** recorded successful completion state produces world +Z 15 cm with measured hand pose; infeasible IK, joint limits, collision path, stale feedback and changed start reject. Ensure **one** FK/IK/geometry planning pass across sender+receiver; receiver does not spawn a second planning process.
6. **Handoff:** receiver rejects invalid/tampered/stale plan, phase mismatch, lost native completion, and start mismatch; 0.01 rad arm tolerance unchanged. Baseline non-MPC goals behave as before.
7. **Timing:** logs distinguish planner time, ROS action wait, Lift goal submission, first Lift command; measurable outputs on mock/recorded trials without hardware.

First run non-hardware unit/integration tests, lints and compile checks. If practical, dry-run the unified launcher using mocks/recorded inputs to verify full orchestration. **No live physical trial without the user/operator running it.** Do not infer that CI/mock passing proves real grasp success.

## Deliverables

- A focused implementation in the existing repository (not a new parallel execution pipeline).
- Updated CLI help/config/docs, showing the one-command happy path and explicit troubleshooting instructions.
- Tests with commands and outcomes.
- A concise `docs/task1-one-command-real-mpc-results-2026-11-13.md` reporting changed files, exact behavioral changes, regression coverage, remaining risks, and timing measurements (or `not measured` where applicable).
- Git diff / commit SHA if committed; do **not** push or operate hardware unless separately requested.

**Completion definition:** one explicitly authorized command launches the right-hand real-active pipeline, starts 8089 immediately, extracts exactly one observed object without manual box selection, generates and selects grasp candidates, performs existing native MPC grasp+squeeze, single-pass feasible 15 cm dynamic lift, hold/release/return, saves actionable stage logs, and requires **no extra Enter/confirmation**. Preserve failure-fast, control safety, and non-active modes.
