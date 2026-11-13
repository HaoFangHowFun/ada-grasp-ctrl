# Ada-Grasp Right CRX + Sharpa: MPC Phase 2A/2B/2C

**Codex Agent Implementation Handoff**  
**Date:** 2026-11-12  
**Target repo:** `https://github.com/HaoFangHowFun/ada-grasp-ctrl`  
**ROS integration:** `https://github.com/MSC-UCB/bimanual_manipulation`  
**CRX interpolation implementation:** `https://github.com/MSC-UCB/dual_crx_control`  
**Hardware:** right FANUC CRX-5iA (6 joints) + right Sharpa Wave (22 joints)  
**Scope:** **offline implementation and validation only**. Do not launch, connect to, command, arm, tare, or reconfigure physical devices.

## 0. Intent and confirmed design decisions

The existing `run_grasp_pipeline.py` has already completed **many successful real-world right-hand grasps without MPC**. Preserve this baseline. The repo also has a successful **right-CRX + Sharpa MuJoCo** implementation of the original `ours` contact-aware control policy at **10 Hz**, and has a completed, read-only `TactileAdapter` that produces `TactileContactObservation`. These three components are not yet one real-world closed-loop execution chain.

**Decisions (do not ask again):**

1. **MPC-active stages:** **Grasp and Squeeze only**. Above Object, Pregrasp, Lift, Hold, Release and Return remain baseline execution; all existing stage transitions must remain valid.
2. **Passive mode contact sources:** both **MuJoCo** and **recorded tactile replay**. No claim that replay provides a fully observed real contact wrench.
3. **MPC output:** absolute **28-joint position target**, in the canonical named order; not an unlabelled delta command.
4. **MPC math:** **preserve the original `ours` optimization / force objective / stiffness formulation**. Adapt measurements with an explicit compatibility layer, not by fabricating missing sensed quantities.
5. **Interpolation inside Ada:** **linear**. Do not use the Ada receiver's current `smoothstep`/quintic interpolation in the new linear reference / streaming path. The original working trajectory behavior must remain available as a regression-compatible legacy option. Existing lower-level `dual_crx_control` is currently launched with `method:=quintic input_rate_hz:=100.0`; **do not modify that repo or its launch settings** in these tasks. These are two distinct interpolation layers.
6. **Frequency split:** MPC update at **10 Hz**; Ada command output / safety supervision at nominal **100 Hz**; CRX interpolation node internally emits at **500 Hz** (not a claim about physical servo rate). Confirm actual timing with tests, no hard real-time claims.
7. **Phasing:** A/B/C developed and validated **without operating real hardware**. Hardware commissioning and active control are explicitly deferred.

**Critical separation:** The term `mpc_passive` (sometimes called *shadow mode*) refers to MPC calculations that are **never sent to actuators**. It does **not** refer to the Shadow robotic hand. All current hardware and model targets in this handoff are **Sharpa Wave**.

## 1. Inspect and preserve current source (mandatory first step)

Inspect actual HEAD before editing; report commit SHA and discrepancies with this handoff. Do not silently overwrite concurrent changes. Relevant paths to check:

- `script/run_grasp_pipeline.py`
- `src/ada_grasp_ctrl/pipeline/{runner,plan,viewer}.py`
- `src/ada_grasp_ctrl/ros_execution/{bridge,runtime,settings,motion,review,snapshot}.py`
- `src/ada_grasp_ctrl/config/experiments/grasp_pipeline.yaml`
- `src/ada_grasp_ctrl/experiments/{right_full_pipeline,right_full_control,right_full_episode}.py`
- `src/ada_grasp_ctrl/contact/{types,validation,mujoco_adapter,tactile_observation}.py`
- `src/ada_grasp_ctrl/tactile_preview/{adapter,geometry,kinematics,observe}.py`
- `src/ada_grasp_ctrl/utils/grasp_controller.py`
- `docs/{grasp-pipeline,grasp-pipeline-validation,right_full_mpc_integration,sharpa-tactile-adapter}.md`
- ROS launch: `bimanual_manipulation/launch/single_arm_hand.launch.py`
- CRX interpolation: `dual_crx_control/src/dual_crx_control/interpolation/{node,quintic}.py`

Baseline facts to verify in source:

- Current Ada ROS `FollowJointTrajectory` receiver accepts **one 28-joint endpoint per action**, rejects concurrent goals, and requires a reviewed packet in hardware mode. It generates a 100 Hz quintic/smoothstep trajectory. **It is not a live MPC streaming API.**
- The `dual_crx_control` interpolation node already accepts repeat joint targets, can replan, and publishes at 500 Hz. Keep its existing protections and do not bypass drivers.
- The simulation `ContactObservation` contract includes object identity, local contact position/frame and six-component wrench; tactile `TactileContactObservation` supplies a 3D point, unit normal and 3D force in **hand-base frame** plus validity/provenance. The types are intentionally separate.
- Existing `ours` simulation uses actual CRX + Sharpa joint kinematics and a canonical 28-DoF model. Preserve 10 Hz and solver policies.

**Do not mix up:** (a) 10 Hz MPC optimization cadence, (b) 100 Hz Ada guarded command publishing, and (c) 500 Hz CRX low-level reference interpolation.

## 2. Integration contract / high-level architecture

```text
RealSense -> DexLearn -> Selector -> reviewed grasp stages
                           |
                   run_grasp_pipeline.py
                           |
            +--------------+------------------+
            |                                 |
        baseline                         mpc_passive
     (legacy / linear)                 / mpc_active
            |                                 |
  existing ROS action path       linear reference generator
                                              |
                           measured 28-joint state + ContactSource
                                              |
                                  original Ours MPC (10 Hz)
                                              |
                            absolute named 28-joint target
                                              |
                     supervised 100 Hz command execution adapter
                                              |
                          existing arm/hand ROS driver topics
```

`mpc_passive` subscribes and computes only, recording candidate outputs and validity; **no command publisher or actuator-control action from the passive MPC path**. During real hardware testing later, the already approved baseline executes the actual motion and passive MPC observes it. In offline tests, use mock/replay sources.

The active executor must have one exclusive command owner. Never run an old trajectory receiver concurrently with an MPC publisher to the same device topics. Use explicit mode-specific ownership and fail-closed gate checks; hardware command enablement is **out of scope** for this handoff.

### Canonical control interface (design target, adapt names to repo conventions)

Define explicit dataclasses/protocols where useful, for example:

```python
JointFeedback(
    joint_names: tuple[str, ...],      # full six CRX followed by 22 Sharpa
    positions_rad: tuple[float, ...], # exactly 28 finite values
    velocities_rad_s: tuple[float, ...] | None,
    source_timestamp: float,
    receive_timestamp: float,
    source: str,
)

MPCProposal(
    joint_names: tuple[str, ...],
    positions_rad: tuple[float, ...], # absolute command, 28 values
    valid: bool,
    computed_at: float,
    solve_duration_s: float,
    contact_status: str,
    diagnostics: dict,
)
```

Use existing repository abstractions if equivalent. Never rely on unnamed positional arrays across ROS/model boundaries; enforce named order at input and output. Clearly separate **measured `q`** from **last commanded `q_cmd`**, as the original stiffness/contact formulation distinguishes them.

## 3. Task A — Pipeline controller mode + Ada linear reference + passive orchestration

### Goal

Add explicit controller selection to the existing right-hand grasp pipeline, while preserving the already successful non-MPC baseline. Scope MPC to **grasp/squeeze** stages. Implement **Ada-side linear references** with the same reference data in the preview and command/test implementations.

### Required work

1. Audit current CLI/config and propose a minimal, backwards-compatible mode switch: `baseline` (unchanged legacy default), `linear` (new Ada-linear baseline where necessary), `mpc_passive`, and `mpc_active` (offline/mock-only at this stage). If CLI contract differs, document exact aliases and defaults. Do not silently reinterpret the existing successful default.
2. Keep capture, calibration, object extraction, inference, candidate selection, reviewed stages, lifting and termination as-is; no reranking, pose mirroring rewrite, or checkpoint retraining.
3. Create one reusable **Ada linear joint reference** function. Its interpolation must be precisely `q(t) = q_start + clip((t-t0)/T,0,1) * (q_target-q_start)`; include endpoint, monotonic time, limit and numerical tests. Use it consistently for MPC reference trajectories and new linear-mode previews. If replacing the current full-motion preview animation is needed, ensure viewer/execution parity. Do not pretend downstream CRX quintic makes the Ada interpolation linear at the actuator.
4. Insert a controller policy boundary: outside grasp/squeeze, use baseline stage executor; in grasp/squeeze, passive mode computes MPC proposals while baseline commands remain the only commands. Active execution is gated and disabled for physical hardware in this task.
5. Add deterministic logging of stage, mode, timestamps, linear reference, measured 28-DoF state, MPC proposed command (if valid), contact/source status, safety reasons and command-source ownership. Proposals must never be mislabeled as executed targets.
6. Make passive execution usable with mock state + simulated contact and recorded tactile replay. If the replay does not contain an accepted controller-compatible contact, passive MPC may report **unavailable / rejected**; do not invent contact or treat rejected samples as no contact.
7. Preserve cancellation, reviewed packet identity, arrival checks, stable feedback, stage transitions, and existing camera / robot safety preflights. Avoid changing hardware permissions or operational defaults.

### Deliverables

- Mode/config/CLI changes in existing pipeline code with small reusable helper modules.
- Verified Ada-linear reference and preview/execution parity for the selected mode.
- Passive runner with no actuator output and structured logs.
- Docs: `docs/mpc-phase2a-pipeline.md`, test evidence under a **fresh** output directory.

### Acceptance tests (offline)

- Existing baseline tests and reviewed trajectory packet checks continue to pass; compare known baseline target arrays and stage ordering.
- Linear midpoint/endpoints for all 28 joints; new preview and reference samples identical, with no hidden `smoothstep` in new path.
- Grasp and squeeze alone invoke MPC; no MPC intervention in pregrasp/lift/etc.
- `mpc_passive` running with synthetic/mock feedback emits **zero MPC command publishes**; failures/invalid contact do not implicitly activate control.
- Cancel/stale feedback/invalid sequence fails closed and produces diagnostics.

## 4. Task B — 10 Hz MPC runtime + safe 100 Hz mock ROS streaming

### Goal

Package the **existing** Ours policy as an externally clocked, measured-state real-time-ish controller that outputs an **absolute named 28-joint position target** every 100 ms. Implement a mode-specific guarded 100 Hz ROS streaming adapter using mock drivers **only**, while keeping the existing driver and low-level interpolator untouched.

### Required work

1. Reuse the existing `Controller` / `_CoordinatedWrenchPolicy` / `GraspController` mathematics and numerical settings. Do not replace the solver with a PD/heuristic controller, and do not modify force objectives, stiffness, contact constraints or solver failure semantics to make tests pass.
2. Explicitly adapt measured joint position and commanded actuator targets to the original MPC API. Use Pinocchio / URDF kinematics and Jacobian conventions already validated in MuJoCo. Reject missing joint names, nonfinite values, invalid limits and stale feedback.
3. Implement deterministic MPC tick API at 10 Hz with timestamped input and output. Expose solver duration, stage, contact count/availability and failure reason. On infeasible/timeout/invalid samples, follow a documented safe **hold/reject/abort** policy consistent with the native controller; no silent fallback to unchecked new targets.
4. Build a 100 Hz **guarded** command adapter for the existing right arm and hand topics, preferably reusing safety preflights, backend/model/joint-order checks, watchdog, speed/accel/step limits and cancellation from existing Ada ROS runtime. The adapter tracks the most recent **accepted** MPC target and outputs explicitly bounded commands; do **not** blindly publish a new step of 28 joints whenever an optimizer returns.
5. Design explicit single-writer ownership, startup/shutdown behavior and mode transitions. Existing one-goal `FollowJointTrajectory` receiver is not fit for per-tick MPC updates: introduce a separate controlled streaming interface or carefully refactor a shared runtime, **without** weakening reviewed packet safeguards. Hardware write permission must remain disabled in this delivery.
6. Quantify response to low-rate targets and lower-level Quintic re-timing (nominal 10 ms input horizon, extendable under derivative limits). Account for measured feedback lag; do not assert an MPC target reaches the robot in the following 100 ms. Distinguish `q_mpc`, `q_100Hz`, and measured `q`.
7. Supply a rate/clock implementation that does not depend on exact timer execution. Check dropped ticks, stale inputs, missed deadlines, ROS queue backlog and shutdown/command silence. Record actual timestamps and percentiles in offline reports.
8. Allow mock/control-free testing without loading physical Sharpa SDK devices or real FANUC drivers; avoid any import-time hardware activation.

### Deliverables

- Independent, testable MPC runtime / safety-gated streaming adapter with clear contract.
- ROS mock transport tests and structured command/reference/measurement traces.
- Docs: `docs/mpc-phase2b-executor.md`, include architecture/ownership chart, topic names, timing traces, and policy table for invalid/stale/solver-fail states.

### Acceptance tests (offline)

- Exactly one active command publisher for a mock right side; no collision with the legacy receiver; left-side command count **zero**.
- MPC nominal 10 Hz proposals; guarded adapter nominal 100 Hz; recorded real scheduling deviation reported.
- 28 names and values preserved across adapters; command never exceeds configured joint/step/speed/accel limits (and existing safety guards are not bypassed).
- Fail closed on invalid feedback, stale tactile/contact, lost DDS, solver failure, cancellation, or command-rate stall.
- Active mock mode cannot be confused with hardware enable; hardware path rejects rather than moving a real robot.
- Existing right-full MuJoCo `ours` numerical parity / regression tests still pass.

## 5. Task C — TactileContactObservation → controller-compatible contact estimation

### Goal

Build a **source-independent contact estimation/compatibility layer** so MPC can accept (a) existing MuJoCo `ContactObservation` directly and (b) **recorded Sharpa tactile replay** as far as its measured quantities permit. Preserve original MPC mathematics and tell the truth about what is measured, reconstructed, assumed, or unavailable.

### Required work

1. Audit the exact `contact.ContactObservation` / native `Ks()` / friction and wrench conventions in the current controller. Write a field-by-field compatibility matrix against `TactileContactObservation` before implementing conversion.
2. Preserve `TactileAdapter` as the only source for UV→3D geometry and sensor→elastomer→hand-base FK. Do **not** reimplement SDK parsing, static geometry maps or URDF FK inside MPC. Keep the source read-only.
3. Define output as either a validated controller-compatible contact **with explicit assumptions/provenance**, or a typed **insufficient-data** result. Do not coerce invalid observations into valid zero contact. A **valid empty frame** is a different event from **invalid/stale/no frame**.
4. Frames: `TactileContactObservation` provides `p_B, n_B, f_B`. Transform into the frame used by the real-CRX MPC using current 28-DoF kinematics and any calibrated hand-mount transform. Compute contact-point Jacobian at the actual point; test via finite differences and MuJoCo `mj_jac` reference where applicable.
5. Force convention: current tactile hypothesis is `object_on_finger` reaction force, whereas the simulation controller canonical contact wrench is **on object**, with normal +X hand→object. Keep signed conversion explicit and test with controlled synthetic contacts. Do **not** assume aligning force with the outward surface normal is always correct.
6. Full-contact insufficiency: force is currently a 3-vector from F6 channels 0:3, not a verified 6D wrench; F6 channels 3:6 are **not verified torques**. Unknown moments must remain unknown. No fabricated zero torques if the solver interprets them as measured physical zeros. Resolve this by a mathematically justified reduced-contact compatibility policy *without changing original MPC optimization math*, or mark contact as not eligible and record a clear dependency/blocker if the original controller fundamentally requires unavailable 6D inputs.
7. Object identity: tactile sensing does not provide object ID. For single-object stages, association may be supplied by the selected grasp/task scene as a **declared external assumption**, never claimed as sensor observation. Refuse ambiguous multi-object associations.
8. Multi-contact limitation: current tactile adapter selects first contact per finger. Retain sensor reported counts, do not clone/aggregate additional contacts; identify any effect on the original per-contact optimizer.
9. Replay must preserve acquisition/receive timestamp, SDK frame ID, joint-state association, force sign, calibration/provenance, gaps, reordering and aging. Use monotonic timing with documented synchronization/skew thresholds; never refresh a stale frame merely because a ROS subscriber returns it again.
10. Use geometry alignment `T_elastomer_SDK=I` and `0.001m/native unit` **only as a clearly tagged candidate profile**, as prior geometry/operator observations are not a quantitative calibration. Prohibit treating this profile as hardware-calibrated for active closed-loop control.
11. Validate MuJoCo-contact source unchanged and replay-contact conversion independently. If passive calculation cannot safely proceed due to missing wrench information, return a diagnosable **no valid control proposal**, not a bogus success.

### Deliverables

- An independent `ContactSource` / `ContactEstimator` compatibility interface with strict type/provenance/time validation.
- Fixture-driven recorded tactile replay (sanitized; no hardcoded private workstation path required), with explicit format/versioning and reproducible example.
- Compatibility matrix explaining all six controller wrench components and each frame/force convention.
- Docs: `docs/mpc-phase2c-tactile-contact.md` including unresolved physical-calibration requirements and an explicit **go/no-go assessment** for active MPC.

### Acceptance tests (offline)

- Existing MuJoCo contact adapter remains numerically equivalent to old `ours` controller within existing tolerances.
- Synthetic known `p_B, n_B, f_B` transforms correctly for nonidentity transforms; Jacobian finite-difference tests pass.
- `object_on_finger` ↔ `finger_on_object` sign and hand→object normal conventions are explicitly tested, including tangential components.
- Stale, invalid, missing, out-of-order, skewed and partial tactile observations cannot silently enter the optimizer as valid contacts.
- Recorded replay is deterministic and shows correct contact on/off events and invalid-vs-empty distinction.
- Unknown torque/object identity are surfaced in diagnostics rather than fabricated.

## 6. Execution order and dependency management

Perform **A -> B -> C**, but keep contracts independent:

- **A** creates modes, reference and passive orchestration; use fake controller proposals if needed while B is developed.
- **B** supplies real 10 Hz solver runtime and 100 Hz mock streaming; can initially use existing MuJoCo contact source.
- **C** adds audited tactile replay and contact reconstruction without changing the solver math; may identify a **research blocker** if full wrench cannot be supported by the original formulation. Report blockers, do not hide them by changing objectives.

A and B should not gate C on actual hardware. All A/B/C final acceptance can be offline. Later **physical passive acceptance**, then **separately reviewed physical active control**, are distinct downstream tasks and must not be claimed complete here.

## 7. Non-negotiable invariants

- **No real hardware use**; never start `crx_mock:=false`, `sharpa_backend:=sharpa_sdk` or writable hardware launch in A/B/C tests.
- Do not change `MSC-UCB/dual_crx_control`, `MSC-UCB/bimanual_manipulation`, Sharpa SDK, hardware calibration, driver networking, hardware safety thresholds, speed limits, or gains.
- No change to checkpoint inference, candidate selection ranking, force objective, native MPC math, experimental collision tolerances or frozen scientific outputs.
- Right-side only, canonical **6 + 22** joint names/order. Left side must remain silent.
- No automatic hardware enable, no implicit fallback from invalid sensor data to fabricated zeros, no bypass of review/stop/watchdog/force safety protections.
- Existing successful baseline remains runnable, with regression evidence. Do not silently change the default working mode.
- Avoid Python ABI pollution: Ada inference environment may differ from system ROS Jazzy interpreter; use an explicit and tested serialization/IPC boundary where needed instead of importing incompatible native libraries across runtimes.
- Offline acceptance should be deterministic; use mocks, fixture recordings and explicit clock injection where practical. Do not depend on a running 8089 process; do not kill user-owned listeners.

## 8. Suggested configuration contract (adjust after inspecting repository)

```yaml
controller:
  mode: baseline  # baseline | linear | mpc_passive | mpc_active
  active_stages: [grasp, squeeze]
  ada_interpolation: linear
  mpc:
    frequency_hz: 10.0
    output: absolute_joint_position
    policy: ours
  command:
    output_rate_hz: 100.0
    hardware_enable: false   # hard-disabled for A/B/C delivery
  contact:
    source: mujoco  # mujoco | tactile_replay
    invalid_policy: reject
```

Do not use this illustrative schema to override existing config keys or published CLI contracts without an explicit compatibility and migration plan.

## 9. Required test and reporting discipline

- Snapshot original HEAD SHA, test baseline and preserve unrelated changes.
- Add focused unit tests, ROS mocked/DDS tests when available, and existing entire regression suite; report pass/skip/fail counts **without treating skips as passes**.
- Test legacy pipeline against a known successful mock fixture; preserve reviewed packet integrity and stage sequence.
- Run right-full simulated **linear** and **ours** regression; report contact/trajectory parity and explicit deviations.
- Produce single-source timing/command logs covering joint feedback age, MPC tick, solve duration, 100 Hz publish, CRX interpolation lag in mock/replay, and failure events.
- Include no-hardware proof: invoked commands, ROS_DOMAIN_ID, backend settings, command publisher inventory and absence of real device access.
- Run repository-native lint/format/compile checks, but do not opportunistically reformat unrelated files.
- Fresh output roots only; no overwriting or renaming frozen existing results.
- Document what is **implemented**, **tested offline**, **not tested physically**, and **blocked/unverified**, separately.

## 10. Definition of done / agent response format

At completion, return:

1. Current HEAD / final commit SHA (if committed; do not push unless explicitly requested), file changes and rationale.
2. Task A status, Task B status, Task C status; precisely identify incomplete or blocked items.
3. Exact tests, commands, pass/fail/skip counts, replay/mock artifacts and diagnostic output paths.
4. Architectural map: source of measured `q`, command `q_cmd`, reference `q_ref`, contact observation, MPC proposal, approved 100 Hz publish, and their ROS/topic/process ownership.
5. Timing and failure-handling results; invalid/unknown contact cases must be visible.
6. Preservation evidence for legacy successful grasp path and native MPC math.
7. A **separate future hardware validation plan** (no execution): passive measurements first; only then qualified active grasp/squeeze if safety/readiness criteria pass.

**Success criterion:** A/B/C provide a deterministic, safe-to-test-offline integration path from selected right-hand grasp to 10 Hz MPC proposals, 100 Hz mock command interface, and evidence-backed tactile replay contact conversion, without modifying existing safety-critical drivers or claiming that physical closed-loop MPC has already been validated.
