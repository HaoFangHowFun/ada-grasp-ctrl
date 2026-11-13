# Ada-Grasp-CTRL — P3 Real MPC Active Integration & Controlled Hardware Trial

**Agent implementation handoff | 2026-11-12 | Right CRX-5iA + Right Sharpa Wave**

## Objective

Integrate the **existing, already-tested Ours MPC** into the **existing, already-successful real Linear Grasp pipeline**, so that Ours controls **Grasp + Squeeze only**, with **10 Hz MPC** and the existing ROS command execution infrastructure. Preserve the successful non-MPC grasp workflow as the default and as a recovery path.

**This is an integration and controlled commissioning task, not a new controller or a new MuJoCo grasp experiment.** First implement and validate the software without hardware. Hardware actuation is a separate, explicitly operator-authorized step. Never start or command real hardware autonomously.

Repository: `/home/hao-fang/hugs-workspace/ada-grasp-ctrl` (also `~/hugs-workspace/ada-grasp-ctrl`). Keep all existing dirty/untracked work; check actual HEAD and diffs before changes. Do not commit or push unless explicitly requested.

## Existing state and evidence to inspect first

Read the current implementation, not just these summaries. Useful reports / modules:

- `docs/mpc-phase2-validation-2026-11-12.md`: A/B/C offline results; 10 Hz native proposals, 100 Hz paired mock publishing, single publisher ownership; legacy baseline unchanged.
- `docs/mpc-p1-p2a-results-2026-11-12.md`: P1 adopted native Ours completion and timeout; P2a MCAP decode/replay exists.
- `docs/mpc-p2a-contact-status-results-2026-11-12.md`: healthy fresh contact-point `MISSING` => **NO_CONTACT**; stale/invalid stream => **UNAVAILABLE**, not zero contact.
- `docs/tactile-force-replay-2026-11-12.md`: 8089 visualization; blue object-on-force, orange sensor reaction, green normal, and friction-cone diagnostics.
- `script/run_grasp_pipeline.py` and ROS execution bridge/receiver; original `src/ada_grasp_ctrl/tasks/control_eval_func/optimized_runner.py`, `src/ada_grasp_ctrl/experiments/right_full_control.py`, `src/ada_grasp_ctrl/utils/grasp_controller.py`; `src/ada_grasp_ctrl/mpc/`; `src/ada_grasp_ctrl/contact/`; `src/ada_grasp_ctrl/tactile_preview/`.
- Existing *real* baseline evidence: `/home/hao-fang/ws_fanuc/output/physical_mpc_observe_20261112_194701/` (MCAP) and `output/grasp_pipeline/physical_baseline_20261112_194720/` (run report).

Previously reported P1 result: an active mock without physical contacts reached native 88-step timeout, with 0 N versus the original 15 N completion threshold. **This is expected native behavior, not evidence of a broken ROS joint-arrival contract.** Do not force a contact-free mock grasp to complete.

Previously observed real force samples (provisional calibration): +64.5 s, index Fn=3.086 N, Ft=2.061 N; ring Fn=0.274 N, Ft=0.106 N. Existing MPC friction coefficient is `mu=0.3`; those samples fail cone due to excess tangential component, **not negative normal projection**. Do not infer that real forces or friction are formally calibrated, and do not alter `mu` to pass tests.

## Nonnegotiable control ownership

| Mode | Actual commands | Ours computations | Permitted scope |
| --- | --- | --- | --- |
| `linear` (default) | Existing successful Linear path | None | Entire existing pipeline unchanged |
| `mpc_passive` | **Exactly the Linear command path** | 10 Hz, diagnostics only | No MPC publisher, cannot abort Linear on MPC error |
| `mpc_active` | Existing Linear for approach/pregrasp/lift/hold/release/return; Ours for grasp+squeeze | 10 Hz active during grasp+squeeze | Requires explicit hardware enable and operator supervision |

**When MPC is active, original Ours is authoritative for all normal controller behavior:** optimizer, stage-1/stage-2 switching, force trajectory, solver failure policy, `should_stop`, `max_steps`, and final joint proposals. The ROS executor is responsible only for feedback delivery, bounded command transport and safety/watchdogs. Never use the Linear fixed-target joint-arrival gate as the *normal completion criterion* for Ours Grasp/Squeeze. Do not modify Ours mathematics or its original completion thresholds.

Ours native completion (verify in actual source): `step.index >= len(runner.qpos_path) and step.current_sum_force > self.final_sum_force`; native budget `len(runner.qpos_path) * 2`. Keep the original semantics; do not conflate grasp/squeeze reference boundaries with native completion. Record Stage 1 / Stage 2 transitions and native termination reason exactly.

## Implementation tasks (minimum changes)

### P3-A — Live contact → Ours observation bridge

Reuse the existing Sharpa ROS TactileAdapter, geometry, frame mapping, `TactileContactEstimator` and native MPC wrapper. Establish a coherent real-time **28 named joint** and tactile snapshot with explicit receipt/source clock metadata. No duplicate tactile parser or SDK polling implementation.

- Normal points outward, finger → object; F6 currently represents **provisional** object → finger reaction; Ours uses force on object, `f_object = -f_sharpa`, after both are in the same frame. Local force `f_c = R_c.T @ f_object`, normal component at index 0. Avoid double-negation or double-rotation; retain original raw values/provenance.
- A valid contact contributes only once, with identifiable finger/link, contact position, local position, local right-handed frame and correct Jacobian; do not invent 3D torque, object pose or contact identity.
- Fresh/healthy `contact_points=MISSING` or `EMPTY` means no contact on that finger **per operator-verified Sharpa convention**, not failure. `INVALID`, stale, disconnected, absent entire stream, inconsistent provenance or pose skew means unavailable; do not re-label as valid empty.
- Multiple points sharing one F6 resultant are ambiguous; never divide a single resultant among points. Log per-finger rejection reasons and keep other healthy contacts.
- **Important friction-cone distinction:** the original MPC optimization constrains *optimized* contact forces; do not assume all noisy/measured forces must pass the same cone to be considered a valid *measurement*. Audit the precise usage and safety implications of the current pre-solver rejection. If a change is justified, make it only in the compatibility policy, not the Ours solver, preserve all diagnostic values and invalid/negative-normal data gates, and show focused regression evidence. Do not silently remove safeguards.
- Real sensor acquisition timestamp units/epoch are not confirmed. Use bounded same-host receipt-time freshness and joint/tactile skew with explicit `receipt_aligned` provenance; **never claim it is acquisition synchronization**. Block or mark unavailable when coherence limits fail; do not interpolate future contacts.
- Native grasp reference must match the *actual selected real candidate* and its 28-joint/hand-base pose configuration; do not substitute the unrelated frozen `g0000` scene reference. Reuse the real pipeline's packet/selector outputs. Document any remaining pose/reference uncertainty and fail closed instead of fabricating alignment.

### P3-B — Real active execution wiring

Reuse the already implemented 10 Hz native worker → 100 Hz bounded target streaming and current `bimanual_manipulation` / `dual_crx_control` ROS paths. Do not modify ROS drivers, Sharpa SDK, existing force collision safeguards, or low-level interpolation configuration. Ada reference interpolation remains linear; underlying CRX interpolator may remain `method:=quintic`, nominal `input_rate_hz:=100`, output at its existing configuration.

- Commands must be **named, absolute 28-joint positions**, with verified canonical joint ordering / mapping. A single exclusive publisher/owner is required for each right arm/hand command topic. Existing fixed-trajectory receiver and new MPC streaming must not issue commands concurrently.
- Active mode uses existing Linear to reach pregrasp, then exclusively hands off Grasp+Squeeze to Ours, then returns to the existing Linear sequence only after native Ours **normal completion** and all transport/safety checks pass. Do not treat native timeout, no contact, solver rejection or transport fault as completion. No automatic lift after an unsuccessful grasp.
- Enforce existing limits, bounded per-tick delta, stale-feedback watchdog, missing-contact policy, explicit cancel/stop and clean ownership relinquishment. Safety stops are independent of native normal completion. If a 10 Hz cycle fails, choose a defined safe stop/hold policy consistent with existing hardware protection; no last-command runaway or automatic restart.
- Keep active real-hardware **disabled by default**. Hardware publish permission requires a separate explicit opt-in flag plus clear readiness diagnostics and operator consent at run time. Running ordinary `linear` or `mpc_passive` must never inadvertently start an MPC command publisher.
- Log actual feedback, linear reference, 28-joint MPC proposals, 100 Hz transmitted targets, solver status/time, stage, normal-force sum, target force, contact validity/provenance, controller completion or timeout, watchdog status, publisher ownership and intervention/stop cause. Logs are for post-run audit, not an excuse to add an unrelated dashboard.

### P3-C — Verification and operator-supervised commissioning

**First, offline only:**

1. Audit current source and preserve baseline hashes/fixtures. Run focused original MPC, ROS execution, tactile contact and mode-regression tests; confirm default `linear` unchanged.
2. Mock the real-time observation feed with a known coherent fixture. Test partial finger contacts, zero healthy contacts, stale topics, mismatched frames/time, negative normal, multi-point ambiguity, solver failure, duplicate publisher, stop/cancel and handoff.
3. Confirm `mpc_passive` and `linear` produce byte-identical **baseline commands for matched fixture inputs**, and passive never publishes MPC outputs or aborts Linear on MPC failure.
4. Confirm `mpc_active` produces the expected bounded 10 Hz → 100 Hz **mock** outputs, uses native Ours stage switching/completion, rejects native timeout and never lifts following failure. A synthetic valid completion test may exercise the transition, but must be labeled **synthetic**, never physical grasp evidence.
5. Build a **read-only readiness report** for a prospective live session: control command ownership, 28-joint mapping, current joint/tactile update age, force/normal sign and geometry version, reference alignment, solver period, maximum target delta, enabled safety, and explicit block reasons. If no valid contact can reach Ours, report **NOT READY**—do not override gates.

**Then, only with the operator physically present and explicitly authorizing the specific step:**

6. Start the existing proven baseline driver stack in the ordinary operator workflow. First perform read-only live diagnostics / short passive observation while the original Linear path remains in control. Do not modify the proven startup command or require a new robot driver.
7. Verify ownership and real feedback under live conditions; test cancel/stop while still non-actuating from the MPC side. Proceed to active only if all readiness checks pass and the operator approves.
8. For first controlled Active trial: maintain an accessible stop mechanism, use the existing approved workspace/object/trajectory and safety setup, switch to Ours **only for grasp+squeeze**, and record whether actual force evolves, stages switch, native completion occurs, and safe lift is permitted. Do not initiate a physical grasp automatically at the end of software changes.
9. After trial, restore the default Linear mode and check no MPC publisher remains. Report success/failure without masking the original control outcome. Never weaken a physical safety guard for convenience.

## Acceptance criteria

- [ ] Default `linear` remains the exact existing successful pathway; source/mode regression passes.
- [ ] `mpc_passive` outputs no MPC robot commands and cannot interfere with Linear.
- [ ] Native Ours policy retains its original math, Stage 1/2, force trajectory, `should_stop()` and `max_steps()`.
- [ ] Real telemetry adapter provides valid, source-traceable contact observations *when supported*; no bogus contact, torque, force-sign fix or fabricated pose.
- [ ] Correct reference for the **selected real grasp**; no accidental frozen-scene substitution.
- [ ] Real/hypothetical friction-cone violations reported distinctly from actual sensor corruption; any changed compatibility policy documented with tests.
- [ ] Exclusive 28-joint command ownership, bounded 10 Hz MPC / 100 Hz transport, watchdog/cancel/stop, no auto-lift on failure.
- [ ] No hardware commands without an explicit operator-authorized active enable; zero hardware use in initial implementation/test run.
- [ ] Focused tests and concrete commands/evidence paths supplied; incomplete live validation explicitly labeled, no unsupported physical success claim.

## Required deliverables

1. A short audit of **actual HEAD, changed files and current code paths**, preserving unrelated work.
2. Minimal implementation with precise entrypoints/flags for `linear`, `mpc_passive`, and explicitly gated `mpc_active`.
3. A command sequence for **offline test and mock qualification**; a **separate** operator runbook for live read-only/passive and eventual approved active commissioning (never automatically run the latter).
4. A machine-readable readiness summary and proposal/command/contact diagnostic artifacts.
5. Focused test results and a table distinguishing **implemented**, **offline-qualified**, **live-read-only-verified**, and **physical-active-unverified**.
6. A concise list of blockers, especially real source freshness/reference alignment, force convention/friction cone, missing-contact semantics, and the hardware authority handoff.

## Stop conditions / forbidden shortcuts

Do not modify original Ours math, solve strategy, 15 N completion setting, solver thresholds, CRX/Sharpa drivers, driver gains, safety constraints, or the successful baseline unless separately authorized. Do not set `mu` artificially, silently invert force, treat missing sensor streams as no contact, assume receipt clocks equal acquisition clocks, turn off contact checks broadly, or let the existing Linear receiver and MPC publisher race. Do not re-run previously successful MuJoCo grasp as the primary deliverable. **Stop software integration and report a blocker rather than remotely commanding the robot without the operator's specific authorization.**

**Success for this agent handoff is a minimally integrated, offline-qualified and operator-ready real active MPC pathway—not an unapproved autonomous hardware demonstration.**
