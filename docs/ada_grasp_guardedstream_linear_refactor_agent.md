# Ada-Grasp-CTRL — Simplify MPC GuardedStream to Linear Interpolation

**Agent implementation task | Right CRX-5iA + Right Sharpa Wave | 2026-11-13**

## Goal / agreed design

Refactor **only the MPC-active 10 Hz → 100 Hz streaming path** in `/home/hao-fang/hugs-workspace/ada-grasp-ctrl` so `GuardedStream` is a **thin, deterministic linear interpolator**, not a second trajectory planner or an independent robot safety layer.

```text
Original Ours MPC (10 Hz; 28 named absolute joint targets)
    → thin linear interpolation (100 Hz; starts from last actually published command)
    → existing bimanual_manipulation ROS interfaces and hardware safety
```

**Decisions already confirmed by operator:**
- A new interpolation segment **starts at the last successfully published joint command**, not current measured joint feedback. Initial stream starts from the fresh measured joint vector **before the first publication**.
- Use straight **linear interpolation**, **not** acceleration-based braking, velocity-state planning, quintic or ZOH.
- Remove the *duplicate Ada streaming* joint-position, speed, acceleration, per-step-delta and braking gates; do **not** loosen Ours MPC's own optimization constraints or modify the pre-existing bimanual_manipulation / device safety.
- Do not change the working default `linear` mode or `mpc_passive` output. `mpc_active` alone uses this interpolator.
- Keep command ownership exclusivity, stale/failure signaling, explicit cancellation, finite-number/shape/order validation, timing/watchdog and existing stop/abort behavior in the **execution interface**. Those are not trajectory constraints and must not disappear because the class is simplified.

## Source inspection before edits

Inspect actual HEAD, git status and dirty/untracked files; preserve existing work. Read:
- `src/ada_grasp_ctrl/mpc/streaming.py` (`GuardedStream`)
- `src/ada_grasp_ctrl/mpc/live_execution.py`
- `src/ada_grasp_ctrl/ros_execution/mpc_stream.py`
- `src/ada_grasp_ctrl/mpc/contracts.py` and `src/ada_grasp_ctrl/experiments/right_ros_contract.py`
- original Ours: `src/ada_grasp_ctrl/utils/grasp_controller.py`, `src/ada_grasp_ctrl/mpc/native.py`
- `src/ada_grasp_ctrl/ros_execution/bridge.py`, `model.py`, `runtime.py`, and tests `tests/test_mpc_phase2.py`, `tests/test_mpc_p3.py`, `tests/test_mpc_p3_receiver.py`
- `docs/mpc-p3-first-hardware-trial-2026-11-12.md` (previous real failure)

Original Ours already has an SLSQP `q_limits` inequality constraint and per-MPC-step joint delta bounds. Do not alter those. Previous hardware test had 33 valid proposals and 313 paired command ticks, then Ada `GuardedStream.check_limits()` fault; `right_ring_MCP_AA` approached a valid lower-bound MPC target. This motivates eliminating **Ada-generated** overshoot, not relaxing robot limits.

## Implementation specification

1. **Minimal interpolation behavior.** Maintain `q_last_published`, the latest accepted MPC target, and a segment start time. On receiving an accepted target at `t_k`, snapshot `q_start = q_last_published` and `q_goal = proposal.positions_rad`. For each scheduled 100 Hz publish opportunity, compute

   `alpha = clamp((t - t_k) / 0.1, 0.0, 1.0)`

   `q_cmd = (1-alpha) * q_start + alpha * q_goal`.

   Use 0.1 s as nominal 10 Hz interpolation duration, consistent with the current MPC clock. The *first* command must not jump at target acceptance; the exact endpoint must be reached when alpha=1 **if no newer MPC target arrives**. Never extrapolate past the target. Use stable interpolation arithmetic and clamp alpha, not joint position. Preserve floating-point finite/order validation.

2. **New target during an incomplete segment.** Re-anchor at **last successfully published** command, not an unsent computed sample, not the original segment start, not measured feedback, and not the previously requested target. Start a new 100 ms segment toward the new target. Do not emit any instantaneous catch-up burst or joint jump. Keep continuous position commands across retargeting; velocity discontinuity at segment boundaries is accepted in this design, with real hardware safety handled by the existing downstream stack.

3. **When no new MPC proposal arrives.** Complete the current segment and hold its endpoint while the existing *execution-interface* proposal-freshness watchdog remains satisfied. An expired/stale MPC proposal must trigger the existing defined fail-safe/abort; do not keep publishing indefinitely without valid upstream control. No silent fallback to `linear`, no invented target, and no auto-lift after a fault.

4. **Publishing contract.** The interpolation state used for the next segment must reflect the **last command actually published**, not merely a generated command. Audit the boundary between `GuardedStream.tick()` and the ROS publish calls. If publish can fail, do not advance `q_last_published` until the paired arm/hand send path has successfully completed according to the existing API; if the middleware cannot confirm physical delivery, describe this as `last_published` (not `last_applied`) and use consistent local publication semantics. Prevent mismatched arm/hand halves and preserve exclusive command ownership.

5. **Remove duplicated Ada movement constraints in this active path.** Delete or bypass the streamer's independent `self.velocity`, braking envelope, acceleration ramp, configured arm/hand `speed`, `acceleration`, `step` limiting and `check_limits()` movement rejection. Keep all original Ours constraints and preserve the downstream safety mechanisms. If the same `GuardedStream` is used by mock tests, update tests to reflect its new **interpolation-only** contract rather than deleting all safety tests elsewhere.

6. **Do not weaken operational control-plane checks.** Keep 28 canonical named joints, finite values, non-reordered joint feedback, solver proposal validity and deadline, command ownership, stale-feedback/source watchdog, command-rate stall policy, cancel/stop and fault latching in the appropriate existing interface. Do not create parallel publishers. Confirm downstream bimanual_manipulation validates joint targets and handles command timeout/stop on both CRX and Sharpa paths before removing redundant checks; **if not confirmed, report a commissioning blocker instead of asserting there is protection**. Do not independently restart the drivers or invoke actual hardware.

7. **Minimal changes elsewhere.** Update `live_execution.py`, `ros_execution/mpc_stream.py` or IPC contracts only as necessary to make `last_published` semantics and failure propagation correct. Do not touch original Ours solver math, friction coefficient, native stopping/completion, joint-order mapping, tactile conversion, robot drivers, or existing non-MPC linear reference generator. Avoid unrelated refactors.

## Offline tests (mandatory before any physical use)

- `linear` baseline and `mpc_passive` remain unchanged, using the existing byte-identical fixture regression tests.
- Constant target: 100 Hz samples on a straight line, in expected sequence, with exact start and endpoint and no extrapolation.
- Retarget at 30–70 ms: next segment starts at **last published** command; position is continuous, no repeated old-segment jump; reaches next endpoint if allowed to finish.
- Target exactly at a URDF lower/upper boundary: for an in-bounds segment, all interpolated commands remain within the interval. **Reproduce the recorded `right_ring_MCP_AA` case**, using documented target and command history, without assuming the rejected unsaved command's value.
- 28 axes and arm/hand named mapping preserved; no silent clipping, numerical NaN/Inf, or reordering.
- Missing/invalid/stale proposal, stale feedback, missed deadline, rate stall, duplicate owner, cancel and publishing failure prevent further actuation and correctly propagate fault to receiver/sender. Retain existing native Ours `should_stop()`, timeout handling and no-lift-after-failure.
- 10 Hz inputs -> approximately 100 Hz paired command output in ROS mock. Run the existing focused MPC, tactile, ROS receiver and pipeline suites where environments permit. No real hardware start.
- Confirm and document downstream `bimanual_manipulation` safety coverage for both right arm and hand; separate any unverified assumption as a **hardware commissioning blocker**.

## Deliverables

1. Minimal code patch with file-level change summary and explanations of `q_last_published`, retargeting, interpolation duration and command timestamps.
2. Offline test commands and numerical results, including `right_ring_MCP_AA` boundary regression and untouched Linear/passive comparison.
3. Read-only verification of downstream arm/hand position limits, timeout/stop and command ownership safety coverage; clearly state any gap.
4. Diagnostic artifacts that distinguish MPC target, segment start, interpolated command, actually-published command, feedback, and stop reason.
5. Short operator handoff: what changed, exactly what remains unverified, and why no physical test was run.

**Restrictions:** No autonomous hardware actuation, no implicit enable of `mpc_active`, no bypass of bimanual_manipulation safety, no alteration of Ours optimization or original stage transition, no weakening default `linear`, and no commit/push unless operator explicitly requests it.
