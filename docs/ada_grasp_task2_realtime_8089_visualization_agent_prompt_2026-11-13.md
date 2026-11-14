# Codex Agent Implementation Prompt — Task 2: Real-Time 8089 Robot & Tactile Dashboard

**Project:** `https://github.com/HaoFangHowFun/ada-grasp-ctrl`\
**Date:** 2026-11-13\
**Scope:** Right FANUC CRX-5iA + Right Sharpa Wave, hardware `mpc_active` only\
**Task:** Upgrade the existing port **8089** grasp preview into a **read-only, non-blocking, real-time monitoring dashboard**.

## 0. Read this first — concurrency with Task 1

Task 1 (one-command hardware `mpc_active` pipeline, auto object detection, 8089 early start, and minimal Lift handoff) is **currently being implemented by another agent**. Do **not** start from an old checkout and overwrite their work.

1. Inspect current `git status`, branch, HEAD, and the working tree **before** modifying anything. The reviewed reference at prompt-writing time was `main` at `ecc07ce920ebf6f1d285f925f9dff025b11d6c9e`; that is **not** necessarily the current HEAD.
2. Read the latest Task 1 changes/contract and integrate with the **current** 8089 launch/ownership mechanism. Avoid editing Task 1 orchestration files unless narrowly necessary and coordinated; prefer isolated visualization modules and small integration hooks.
3. If Task 1 is still changing the same code, implement the visualization modules with explicit interfaces, tests and a small documented integration patch; **do not silently revert, overwrite, or merge away Task 1 changes**.
4. Make no hardware motion and do not start real `mpc_active` execution. Any live hardware trial requires an operator-run step after code review.

## 1. User-approved requirements (do not expand scope)

- Existing **8089** web interface, launched **at the beginning** of the pipeline (Task 1 ownership). No separate application/port, no terminal Enter gate, no web approval gate.
- Full interactive **3D scene**: actual right CRX robot + right Sharpa hand meshes, calibrated table, and an **initial static observed object point cloud**.
- Continuously update **robot/hand pose from actual ROS joint feedback** (six CRX joints + 22 Sharpa joints). Never masquerade a planned trajectory, MPC target, or interpolated command as actual measured pose.
- Continuously update **finger contact points**, **surface normals**, and **force-on-object arrows** from the existing Sharpa tactile data and validated conversion interfaces.
- Requested visualization conventions: **orange = force on object**, **green = surface normal**, points visibly placed at corresponding fingertips; force arrow length proportional to measured force with user-adjustable display scale. Free 3D orbit/pan/zoom.
- Clear **force panel**: per-finger normal force (N), total eligible normal force (N), and the native MPC completion threshold of **15 N**. Display live controller mode and current pipeline phase (pregrasp, grasp, squeeze, lift, hold, release, return, plus preparing/idle/error).
- **Keep updating all robot and tactile visuals through Lift, Hold, Release, Return**, even though these stages run existing non-MPC motion, without enabling MPC force control during Lift.
- **Do not continuously re-capture/segment/reconstruct the object point cloud**. Only load the already captured pre-grasp object point cloud; do not move/attach it to the hand or claim that it tracks object motion after Lift. Label `Static pre-grasp object cloud` and consider dimming after movement begins.
- Read-only monitoring: viewer refresh/connection/browser state must not gate, slow, command, pause, cancel, or influence robot control. No ROS publishers/action clients or SDK device ownership in the UI.
- **Minimal implementation:** reuse existing rendering and data conversion. No new segmentation model, no new frontend framework, no robot physics simulation in the monitoring loop.

## 2. Reference implementation to inspect before coding

These paths were verified in the repository at the recorded reference commit, but inspect their **current** forms:

- `src/ada_grasp_ctrl/grasp_selection/preview.py` — existing Viser/MuJoCo 8089 viewer: `serve(...)`, `ViserMujocoScene`, scene point cloud/table overlays, 3D interaction, GUI panel.
- `src/ada_grasp_ctrl/pipeline/viewer.py` — existing 8089 pipeline entry and preview-data loading (Task 1 may have changed it).
- `src/ada_grasp_ctrl/pipeline/runner.py` and `script/run_grasp_pipeline.py` — Task 1 launch/early viewer and stage orchestration. **Do not rebuild the launcher.**
- `src/ada_grasp_ctrl/tactile_preview/adapter.py` — shared read-only `TactileAdapter`, `TactileSnapshot`, statuses and provenance. Avoid duplicate tactile kinematics/math.
- `src/ada_grasp_ctrl/tactile_preview/data.py` — normalized atomic F6/contact frames and `TOPICS` for the five right fingers.
- `src/ada_grasp_ctrl/tactile_preview/ros_reader.py` / `__main__.py` — existing ROS Jazzy subprocess-reader pattern (ROS Python 3.12 vs Ada Python 3.10).
- `src/ada_grasp_ctrl/tactile_preview/viewer.py` — existing Viser hand meshes/frames/contact markers/arrows, freshness behavior.
- `src/ada_grasp_ctrl/tactile_preview/force_viewer.py` — reusable force diagnostics and object-on-finger to force-on-object sign, visual conventions (existing replay also uses blue for other diagnostics; **Task 2 wants orange object-force and green normal**).
- `src/ada_grasp_ctrl/contact/compatibility.py` — actual native-compatible contact eligibility, measured-cone diagnostic policy and force convention. Do not invent a separate contact definition.
- `src/ada_grasp_ctrl/mpc/live_execution.py` / `mpc/native.py` — active MPC episode and native completion semantics.
- `src/ada_grasp_ctrl/ros_execution/runtime.py`, `experiments/right_ros_sender.py`, `ros_execution/bridge.py` — actual feedback and lifecycle phases. Prefer existing read-only events/status streams rather than extra logging in hot control loops.
- `src/ada_grasp_ctrl/experiments/fixed_camera.py`, `config/experiments/fixed_camera.yaml` — captured object/table assets; read once, do not restart camera.
- `docs/mpc-p3-real-grasp-lift-milestone-2026-11-13.md` — successful hardware baseline.

**Architecture suggestion, not a license for a rewrite:** one Viser 8089 server owned by the pipeline; a small independent **read-only telemetry subscriber/bridge** emits bounded latest-only joint and tactile snapshots; one presentation updater consumes telemetry and updates scene/GUI without blocking control. Reuse the working `ViserMujocoScene` robot model and existing tactile adapter/conversion, explicitly resolve the right-hand flange/world frame composition for arrows. If another small integration fits the actual Task 1 code better, use it and document the decision.

## 3. Scene and frames — implementation contract

### Robot
- Use **actual, named, fresh** 6+22 joint feedback. Validate names, order/uniqueness, finite values, stamps/arrival, and model compatibility. Prefer the existing joint topics and robot/model utilities; don't guess ROS topic names.
- Update the established CRX + Sharpa geometry in **world frame** using existing model/FK/rendering infrastructure. The hand is mounted to CRX; no independent hand floating at the origin.
- Treat feedback staleness or missing data visibly (`STALE`/`UNAVAILABLE`). Never continue to show old arrows as if valid.

### Table and object
- Load **calibrated** table geometry and the **single selected object** point cloud from the same run artifacts as perception/selection. Verify known frame/extrinsics assumptions and asset provenance; refuse misleading world-frame overlays if transforms are unavailable.
- Table and object are static for the entire run; clearly label that the object cloud is a pre-grasp snapshot and **not real-time object tracking**.
- Viewer may initialize before assets exist. Display a meaningful `Waiting for perception` state, then populate scene **without restarting 8089** when Task 1 creates validated assets.

### Contact arrows
- Reuse `TactileAdapter` and proven preview/contact geometry conventions; do not reconstruct contact position/normal from raw F6 alone.
- Existing native-compatible convention: Sharpa reported force interpreted as **object-on-finger**; force-on-object arrow is the sign-reversed vector. Normal direction remains finger-to-object. Apply **hand flange → world** rigid transform correctly; transform points with rotation+translation, directions by rotation only.
- Draw contact points only when a valid, geometrically interpretable contact exists. Unavailable/invalid/stale/ambiguous observations must be labeled, not silently rendered as zero force. For multi-contact with a single unresolved F6 resultant, **do not fabricate per-contact forces**. Follow the present audited compatibility/diagnostic policy.
- Conventions/force calibration and fingertip geometry are provisional where the project explicitly states so. Keep an unobtrusive provenance/warning; don't claim metrology-grade truth.

### Force panel
- Per-finger `Fn` and **sum of eligible positive normal forces**, using the same definitions/eligibility criteria as the native-compatible contact pipeline wherever possible. If the exact native controller's accepted-force sum is accessible from read-only telemetry/events, prefer showing that as the authoritative `MPC completion force`; otherwise label the derived sensor sum as a **monitor estimate**, not as the solver's authoritative internal value. These two values must **not** be silently conflated.
- A clearly marked **15 N native completion threshold**, **not** a force limit or cap. Going above 15 N is not itself a UI failure.
- During Lift/Hold/etc., continue showing current sensor forces with an explicit `Linear stage / MPC inactive` status; the 15 N grasp threshold is historical/context only and must not imply ongoing force regulation.
- Missing/invalid/stale fingers: show `—`, age/status, and `PARTIAL/UNAVAILABLE` total if a valid full sum cannot be established. No default-zero substitution for missing readings.
- UI-only display downsampling/aggregation is allowed; do not modify native solver/control cadence or force processing.

### Pipeline phase/status
- Surface phase and fault/status changes from the current Task 1 pipeline and/or existing sender/receiver events. No file polling that adds latency to a control loop; stage observation must be read-only.
- Useful states: startup, waiting camera, perception, inference, selection, planning, above_object, pregrasp, MPC grasp, MPC squeeze, dynamic lift planning, lift, hold, release, return, completed, error/interrupted.
- On terminal success/failure, keep the final state visible long enough for an operator to inspect it, under the owning process's lifecycle; don't leave uncontrolled zombie readers/servers.

## 4. Performance and isolation

- Preserve **10 Hz MPC** and **100 Hz command streaming** unchanged. UI is lower-priority, asynchronous, best-effort and read-only.
- Use **bounded latest-value buffers** and drop outdated visual frames instead of accumulating unbounded queues; don't create a large continuous raw-point-cloud stream. Static point cloud transferred only on initial load.
- Target a responsive viewer (e.g. display refresh approximately 10–20 Hz **if resources allow**, not a new hard real-time requirement). Measure and report typical refresh lag, update payloads, and CPU overhead where practical; do not claim a guaranteed frame rate without measurements.
- Keep ROS callbacks independent of browser/network writes and use the existing split Python environment strategy if required. Any reader disconnect or browser crash should be a **monitor fault only**; active motion safety decisions must remain solely in the controller/receiver.
- Own a **single instance of port 8089**. If occupied, clearly report the conflict; never silently open 8090 or another port. Ensure graceful shutdown on normal completion and Ctrl+C.
- Avoid new required dependencies if existing Viser, MuJoCo/mjviser, ROS Jazzy, and tactile preview tools suffice.

## 5. Deliverables

1. Minimal modules for the read-only live telemetry bridge and 8089 viewer update layer; small adapter to Task 1 orchestration where needed.
2. A single integrated 8089 dashboard with full robot+table+static point cloud and live five-finger contact/force indicators + phase/status.
3. Explicit statuses and provenance for stale/missing/multi-contact and unverified calibration.
4. Focused regression tests and a deterministic recorded/synthetic telemetry replay for verifying visualization **without a physical robot**. Include screenshots or equivalent evidence from a mock/replay if available.
5. Short operator README: how 8089 is launched from Task 1's one-command entrypoint, what the arrows/force values mean, what is static vs live, how to troubleshoot telemetry loss.
6. A results note under `docs/` listing files changed, unit/mock test results, performance observations, and any unverified hardware-only items.

## 6. Acceptance tests (must pass before claiming ready)

- With fake/replayed actual joint states, moving the six arm and 22 hand joints moves the full 3D robot consistently with the model; it is **not** following planned targets.
- With synthetic five-finger contacts, markers align with expected finger/world transforms; orange force-on-object flips Sharpa force direction, green surface normal does not flip; arrow lengths scale correctly. Contact positions are points, directions are rotated only.
- Scene includes calibrated table and static original object cloud; updates to live joints do **not** move the cloud. Initial viewer starts before perception assets and transitions to ready **without changing port**.
- GUI shows individual normal forces, total eligible force or honest partial/unavailable estimate, and a 15 N **completion** marker. Test >15 N without clipping or indicating a hard limit.
- Grasp → MPC squeeze → dynamic lift → hold → release → return phase sequence updates without manually pressing Enter or switching web apps. Live force updates continue through all non-MPC stages.
- Stale ROS data, lost hand joints, duplicate joint names, invalid contact geometry, multi-contact ambiguity, stopped telemetry worker, occupied 8089 and browser disconnect are represented safely and do not cause control commands.
- UI refresh and repeated browser connections do not create ROS **publishers**, command owners, ActionClients or SDK sessions; do not block/slow simulated 10 Hz controller ticks or 100 Hz command ticks.
- Existing `linear` and `mpc_passive` modes and Task 1 hardware `mpc_active` logic do not regress. Any tests that require hardware must be documented as **not executed** and run only by the operator.

## 7. Hard non-goals

- **No change** to original Ours MPC objective/solver, stage switching, 15 N native completion logic, friction assumptions, gains, or joint control.
- **No change** to the current dynamic Lift feasibility or driver safety mechanisms from Task 1.
- No new object tracking, camera re-capture per frame, live segmentation, object-attachment visualization, DexLearn reruns, or force control during Lift.
- No new manual approval/Enter gates, no additional physical robot commands, no interactive browser control widgets for robot motion.
- No reimplementation of Sharpa SDK handling, ROS driver control, or existing tactile contact geometry/calibration in a second codepath.
- No refactor of left arm, bimanual execution, or simulation architecture.

## 8. Execution instructions to Agent

1. Inspect live current repository state, read existing viewer + tactile module APIs, check Task 1 overlap.
2. Write a concise implementation plan and identify exact touch points. Keep edits limited.
3. Implement the read-only telemetry bridge, unified scene overlay and GUI with reliable stale-state handling.
4. Integrate minimally with whatever 8089 owner Task 1 actually established; do not start a second web server.
5. Add/run focused **offline** tests and replay demonstration. Do not start real robot movement.
6. Report: changed files, exact tested command(s), tests PASS/FAIL, performance/CPU observations, limitations, screenshots/replay location, and any Task 1 integration conflicts. If conflicts remain, leave clear patch instructions instead of overwriting concurrent changes.

**Desired user experience:** launch the Task 1 one-command right-hand hardware `mpc_active` pipeline once; 8089 is immediately available; the same 3D page shows the real moving CRX+Sharpa, static calibrated table and pre-grasp cloud, fingertip contact points, orange force-on-object arrows, green surface normals, force readouts and live phase; it remains responsive and read-only until the cycle completes.
