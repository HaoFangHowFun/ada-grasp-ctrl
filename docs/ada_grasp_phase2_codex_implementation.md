# Codex Implementation Task — Ada Grasp Phase 2

**Title:** Configurable Single-Arm Grasp Selection in a Dual-Arm MuJoCo Scene\
**Target:** `https://github.com/HaoFangHowFun/ada-grasp-ctrl`\
**Canonical robot description:** `https://github.com/MSC-UCB/bimanual_manipulation`\
**Status:** Implementation specification; **no Phase 2 implementation has been done by this document**\
**Phase 1 reference:** `dual-arm-environment.md` (user-supplied completion report)

## 0. Mission and non-negotiable scope

Deliver **one complete, independently runnable, offline simulation grasp pipeline**:

```text
Configurable object mesh + scale + world pose + camera + robot snapshot
                   ↓
          Full 56-DoF MuJoCo scene
                   ↓
     Object point cloud (complete_mesh or simulated_depth)
                   ↓
      ┌────────────┴────────────┐
      ↓                         ↓
  Original cloud           Reflected cloud
      ↓                         ↓
  Right DexLearn           SAME right DexLearn checkpoint
      ↓                         ↓
  Right candidates         Right-format predictions
                                ↓
                      SharpaHandMirror → Left candidates
      └────────────┬────────────┘
                   ↓
      Generic candidates: required_arms + per-arm targets
                   ↓
      Selector(enabled_arms, shared 56-DoF robot state)
                   ↓
      Single-arm IK + static contact eligibility + CRX joint effort
                   ↓
       ONE best feasible candidate, with selected side
                   ↓
       MuJoCo pregrasp → grasp → squeeze → lift
                   ↓
     Results, trajectory, diagnostics, replay/visualization
```

The `enabled_arms` input means **which arms are eligible to participate**, not “perform a simultaneous two-arm grasp”. Phase 2 candidates use exactly one arm. Both enabled means evaluate left- and right-only candidates together and select **one** side. A future two-arm grasp candidate can declare `required_arms={"left", "right"}` in the common schema, but **must be explicitly reported unsupported in Phase 2**. Do not implement two-hand cooperative grasp planning/execution now.

**Strong compatibility requirement:** Keep the current right-hand checkpoint, inference, selector, ROS bridge, action endpoints, hardware safety protections, original `script/run_grasp_pipeline.py` and its established behavior working unchanged. Do not refactor/migrate the old right-only pipeline as a prerequisite to adding Phase 2. Do not send hardware commands.

## 1. Inspect reality before editing

1. `git status --short`, current branch and HEAD. Preserve any uncommitted work, including output artifacts; never reset user changes.
2. Read `AGENTS.md` and relevant required rules, especially `agents/rules/development.md`, `control-simulation.md`, `testing-release.md`, `documentation-assets.md`. Current code/rules override this prompt if an implementation detail has changed; report conflicts rather than guessing.
3. Read `docs/dual-arm-environment.md` **or the user-supplied `dual-arm-environment.md`**, and inspect current `robot_environment/{model,kinematics,simulation}.py`, export/preview scripts, new tests, and actual Phase 1 snapshot directory.
4. Inspect the CURRENT right-grasp pipeline, `script/run_grasp_pipeline.py`, `experiments/fixed_camera_grasp.py`, `experiments/fixed_camera.py`, `experiments/hugs_sharpa_input.py`, `sharpa_mirror.py`, mirror config/doc, `grasp_selection/` including CRX adapter/types and tests, and existing right-side episode/simulation code. Do **not** blindly assume filenames or APIs from this specification are installed as-is.
5. Identify current right-hand candidate JSON/record frame conventions, pregrasp extrapolation policy, units, checkpoint metadata, object geometry handling and camera calibration; preserve them. Enumerate actual runnable test commands and preexisting known failures.
6. Inspect available object assets and license/provenance; support user-supplied local OBJ/STL/other mesh formats **only when the current MuJoCo/trimesh toolchain can read them**. Fail clearly on unsupported assets, missing collision data, or invalid mesh/scale; do not synthesize a fake asset.

Report any blocking discrepancy (e.g., the Phase 1 snapshot is missing, mesh conversion absent, existing inference not callable) with the smallest needed adaptation. New dependencies, changes to pinned submodules, benchmark semantics, or protected legacy APIs require explicit approval; do not silently make them.

## 2. Phase 1 API and grounded limitations

Phase 1's recorded working API is:

```python
from ada_grasp_ctrl.robot_environment import DualArmModel, DualArmSimulation
robot = DualArmModel.load(snapshot_directory)
sim = DualArmSimulation(robot)
q56 = robot.home_configuration()
T_world_hand = robot.fk(q56, side="left", frame="hand_flange")
q56_candidate, diagnostic = robot.solve_arm_ik(q56, "left", T_world_hand)
contacts = sim.check_static_contacts(q56_candidate)
sim.select_side("left")
sim.set_target(q56_candidate[:28])
sim.step(10)
```

Interpret this example with care:

- Named robot vector order: **left arm `[0:6]`**, **left hand `[6:28]`**, **right arm `[28:34]`**, **right hand `[34:56]`**. Never assume MuJoCo native `qpos` order; use provided explicit mappings. A free object adds coordinates (`nq` may be 63).
- `solve_arm_ik` changes the active **6 CRX joints only**, holds the other 50, returns acceptance diagnostics, **does not enumerate IK branches**. If multiple seeds are needed, implement a bounded, deterministic multistart wrapper without changing Phase 1 solver semantics.
- IK accepts `hand_flange`, `crx_flange`, or `tcp`; DexLearn's wrist targets refer specifically to `{side}_hand_flange`. Never confuse CRX flange, Sharpa flange, palm and TCP.
- `DualArmSimulation(robot, scene_spec=spec, object_bodies=("object",))` can include a caller-provided object. The contact checker is diagnostic, not an acceptance policy. It reports `left_right`, `robot_table`, `within_robot`, `robot_object`, and `robot_environment`, and does not step or mutate the live state. Build a narrow, documented selection policy on these diagnostics; do not globally forbid intentional hand/object contact.
- `set_target` operates on the selected side's **28 joints**; passive arm is not magically rigid under physics. Preserve its current command target and include it in contact checks.
- Mount values are upstream **mock preview placeholders**, not calibrated for hardware; source table is not necessarily the user's real table. Gains/soft limits are provisional. Static collision checking is not trajectory verification.
- Phase 1 report: 56 joints/actuators, 11 new tests passed, 56 relevant legacy tests passed, FK numerical agreement ~1e-15, but a separate **Shadow quick classification regression differs in three records** with unknown cause. Keep this issue visible; do not update or redefine baselines to make it pass.

For accurate current API names, inspect source code first. If a behavior is unavailable, implement it in new Phase 2 code; do not change the validated Phase 1 API casually.

## 3. Configurable simulation scene — first-class requirement

Provide a human-editable YAML scene and one CLI entry point. Example **target contract** (adjust relative paths for actual repo; values are illustrative, not a claim that `bottle.obj` exists):

```yaml
scene:
  robot_snapshot: output/dual_arm/phase1_20261111/snapshot_complete
  initial_robot_state: snapshot_initial_pose  # or an explicit named 56-joint state

object:
  name: bottle
  mesh: assets/objects/bottle.obj
  scale: [1.0, 1.0, 1.0]
  pose:
    frame_id: world
    position_m: [0.45, -0.20, 0.12]
    quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
  freejoint: true
  placement: explicit_pose  # optional: settle_on_table

camera:
  name: grasp_camera
  position_world_m: [0.35, -0.40, 0.85]
  look_at_world_m: [0.45, -0.20, 0.05]
  width: 640
  height: 480
  fovy_deg: 60.0
  near_m: 0.01
  far_m: 3.0

point_cloud:
  source: complete_mesh   # complete_mesh | simulated_depth
  object_only: true
  num_points: 2048
  seed: 42

inference:
  # Existing right_full DexLearn paths/parameters: reference or reuse
  # current established configuration rather than hardcoding new model settings.
  checkpoint: <user-provided-valid-checkpoint-path>
  training_config: <matching-config-path>
  metadata: <matching-right-sharpa-metadata-path>
  dexlearn_root: <installed-DexLearn-repo-path>
  samples_per_side: 32

mirroring:
  plane_normal_world: [0.0, 1.0, 0.0]
  plane_point: object_point_cloud_centroid

selector:
  enabled_arms: [left, right]
  # By default share the existing right-selector CRX joint displacement weights.

execution:
  phases: [pregrasp, grasp, squeeze, lift]

output:
  root: output/phase2/example_run
```

Avoid creating a huge configuration hierarchy. Validate all frames, quaternion normalization, mesh paths, mass/inertia or collision requirements, scale, poses, camera geometry, joint order and checkpoint metadata. Report errors explicitly. Exact file/syntax may differ if documented with working examples.

**Pose consistency:** Object position means mesh frame origin unless otherwise documented, not necessarily its bottom or centroid. `explicit_pose` respects user pose. Optional `settle_on_table` must advance physics to a stable pose **before** cloud generation and record that final pose; do not use an old pose for inference or collision checks. An object in MuJoCo needs meaningful collision and inertial properties for physical grasping; use genuine source data or explicit configuration with documented assumptions; do not invent validated mass/friction values. If physical properties are missing, permit perception/selector diagnostics but mark grasp-physics evaluation unsupported instead of faking a successful simulation.

The physical object must exist in the **same scene** used by perception, static collision diagnostics and grasp execution. Do not place an unrelated placeholder object in the execution scene. Keep object world transform and mesh origin consistent across all stages.

## 4. Object point cloud sources

Implement two selectable input paths with a **common world-frame `Nx3` metric point cloud contract**:

### 4A. `complete_mesh` (first milestone)

- Sample points from the *actual* configured object's surface, honoring mesh scale and full world pose. Use deterministic seeded sampling and record source, seed, object pose, mesh hash and sample count.
- Complete mesh points are **not a simulated RealSense observation**. Mark them as an oracle/diagnostic input.
- Use same downstream preprocessing expected by the trained right DexLearn: sample/center with existing invariants, avoid unapproved rotations or scale normalization.

### 4B. `simulated_depth` (required for full Phase 2 acceptance)

- Render the configured MuJoCo camera's **depth** from the same scene and object pose; reconstruct 3D with correct MuJoCo depth conventions and camera intrinsics/extrinsics. Do not treat nonlinear depth-buffer values as metric range without conversion.
- Restrict points to the chosen object via segmentation/render IDs or **equivalent verified visibility-aware object masking**. Exclude table, robot, background and other objects; **do not cheat by returning invisible complete-mesh points in depth mode**.
- Transform camera-frame points to `world`; validate projection/reprojection and coordinate conventions with deterministic fixtures. Record point count, masking coverage, camera metadata, and an optional depth/mask/point-cloud preview.
- Fail informatively if object is outside the FOV, occluded, depth rendering is unavailable or too few valid points remain. Headless execution is required; do not require a GUI.

For both modes: preserve raw point cloud, the inference input and relevant transforms. Never double-apply a camera-to-world transformation. `complete_mesh` is the recommended initial debugging mode; `simulated_depth` is the sensor-realism mode. State the known training-domain difference: existing `right_full` model was configured for `complete` point clouds, whereas simulated depth is partial-view.

## 5. Mirrored left-hand candidate generation

Reuse **the same trained right-full DexLearn checkpoint** and the existing `SharpaHandMirror` mapping. Do not retrain, replace the model or fabricate a left checkpoint. Existing verified utility:

```python
from ada_grasp_ctrl.sharpa_mirror import SharpaHandMirror
mirror = SharpaHandMirror()
q_left = mirror.right_to_left(q_right)  # 22 canonical joints; currently identity
T_left = mirror.mirror_wrist_pose(
    T_right_mirrored_world,
    normal=world_plane_normal,
    point=world_plane_point,
)
```

- For mirrored object cloud, `p' = c + (I - 2nn^T)(p - c)`, with **one explicitly recorded world-plane normal `n` and point `c`**. Default `n=[0,1,0]`, `c=centroid(original world object cloud)` **only as a configurable diagnostic default**, not a physical calibration. Compute `c` before random inference resampling. Permit explicit normal and point.
- Pass reflected world points through the existing right-hand DexLearn preprocessing/inference, not another model. Decode predicted right-hand-flange targets **in mirrored-world coordinates**, then map back to original world for left-hand-flange poses using `SharpaHandMirror.mirror_wrist_pose` with the *same* plane.
- Apply right-to-left 22-joint mapping for every stage. Keep `pregrasp`, `grasp`, `squeeze` stages and their actual target values. The existing `fixed_camera_grasp.py` stage decoder does **not** extrapolate pregrasp; other older adapter paths **do**. Verify the exact entry path to ensure **pregrasp extrapolation is never applied twice**.
- Existing right inference expects `right_full`, three stages, 22 named right joints, XYZ/WXYZ poses and centered metric world points. Preserve source metadata, probabilities, candidate original indices, stage names, units, provenance and reproducibility.
- Use exact `left_hand_flange` pose semantics; do not mirror a generic palm/TCP pose without converting frames explicitly. Left meshes/joints remain from the **real left URDF** supplied by Phase 1, never a geometrically mirrored right-hand stand-in.
- Retain the original object (unmirrored) in physical MuJoCo grasp evaluation. The reflected object exists **only in the inference branch**.
- A reflection may generate unreachable or colliding candidates; these should become normal selector rejections, not evidence that mapping must be modified.
- Test reflection involution, shared plane, quaternion validity/proper rotations, left joint limits, stage preservation, determinism and correct original-world coordinate recovery. Mirror correctness does not establish physical grasp success.

## 6. Generic candidate contract: immediate new API, right backward compatibility

Define a well-validated, ROS-independent common candidate type. Example shape:

```python
{
  "candidate_id": "left:c0012",  # globally unique when merging both sides
  "required_arms": ["left"],
  "targets": {
    "left": {
      "pregrasp": {"pose_frame": "left_hand_flange", "T_world_hand": ..., "joint_positions_rad": {...}},
      "grasp":    {"pose_frame": "left_hand_flange", "T_world_hand": ..., "joint_positions_rad": {...}},
      "squeeze":  {"pose_frame": "left_hand_flange", "T_world_hand": ..., "joint_positions_rad": {...}}
    }
  },
  "provenance": {
    "generator": "mirrored_right_full",
    "source_candidate_id": "c0012",
    "object_id": "bottle",
    "checkpoint_id": "...",
    "reflection_plane": "..."
  }
}
```

The exact Python dataclass or JSON version is up to Codex, but preserve these semantics:

- `required_arms` is a **set of required manipulators**, not a runtime mode flag. Targets are keyed by arm. Explicit `frame_id=world`, `pose_frame`, `rad`, `m`, canonical named joint order and three stages.
- Eligibility: `required_arms(candidate) ⊆ enabled_arms`. In Phase 2, only candidates with **exactly one required arm** enter feasibility evaluation. A candidate requiring two arms is recognized by schema but returns an explicit `unsupported_multi_arm_candidate` result rather than silently selecting one arm.
- Enabled arms can be `{left}`, `{right}`, `{left,right}`; enforce non-empty valid set. They express top-level user permission, **not a claim of hardware online status**.
- Legacy right-hand records remain accepted via a dedicated compatibility adapter; maintain existing public right-hand selector invocation, outputs, defaults and test expectations. The **generic schema is the new internal canonical representation**, but old data clients are not forced to change immediately.
- Do not duplicate inference across separate left/right codebases. Keep transforms and schemas in clean small modules with explicit validation and unit tests. Distinguish `candidate_id` for left and right candidates derived from the same raw index.

## 7. Selector semantics — joint effort, single winner

For each eligible single-arm candidate, evaluate **all required stages** and the relevant CRX's current joint state. Use Phase 1 full dual-arm environment and the inactive side's **current 28-joint state** in static checks.

For side `s ∈ {left,right}`, candidate `i`, and accepted IK branch/seed `k`:

```text
J_s(i,k) = (q_arm,grasp(s,i,k) − q_arm,current(s))^T
           W (q_arm,grasp(s,i,k) − q_arm,current(s))
```

- Exactly **six arm joints** contribute. **No Sharpa finger effort**, no invented distance-to-object metric, and no special left/right preference penalty. Use **the same existing right-selector joint weights** on both arms unless explicitly configured otherwise. Report `W` and units; do not change incumbent right-side default scoring or weights.
- Score grasp-arm joint displacement from **current measured or configured initial robot state to grasp**, not an arbitrary home pose or pregrasp trajectory length. Use named side slices.
- Preserve the existing right selector's multiple IK seed/solution strategy where practical; Phase 1 `solve_arm_ik` on its own returns **one** local solution. Ensure deterministic, finite, bounded multistart, with solver diagnostics and no silent clipping.
- Check joint bounds, IK residual, and static collisions at **pregrasp/grasp/squeeze** using current scene, passive robot configuration and actual object pose. A grasp intentionally touching an object is **not by itself a failure**. Distinguish hand-object from arm-object, arm-table, inter-arm and meaningful self collisions; account for known natural palm/thumb and base/J1 contacts in Phase 1, using narrow documented classifications, not broad disabling of collisions.
- A static contact is a diagnostic. Define limited, testable eligibility rules by contact pair/class; do not assume every MuJoCo contact is forbidden, or claim safety for untested intermediate motions. Do not add a separate hardware safety layer or watchdog.
- Filter candidates according to `enabled_arms` *before* expensive evaluation. For `both enabled`, find the best feasible left-only and right-only candidate and choose the **single** globally lowest-effort result. This is **NOT** `J_left + J_right`; summation applies only to a future candidate that actually requires both arms.
- Handle infeasible candidate sets, equal-cost deterministic tie-breaks, insufficient valid joint state, missing frame calibrations, incompatible source model and errors transparently. Never select a candidate with invalid/NaN cost or unaccepted IK.
- Selection result must include `selected_side`, candidate and source ID, required arms, selected CRX six-joint target, Sharpa 22-joint stage targets, score, chosen IK diagnostics, per-candidate rejection reasons and full provenance. Preserve right-only compatibility of existing JSON/report formats; new output can version schema.
- `execution_ready` is **not** implied by static selection: require explicit motion/execution validation before simulation launch; never claim hardware-readiness.

## 8. One complete offline simulation pipeline

Provide a **new** runnable entry point, preferably:

```bash
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py \
  --environment simulation \
  --scene config/simulation_scene.yaml
```

It must work for `selector.enabled_arms: [right]`, `[left]`, or `[left,right]` using the **same** code path. The historical `script/run_grasp_pipeline.py` must remain intact, including its existing `mock` meaning (ROS mock, **not** MuJoCo).

The new entry point must perform:

1. Load and validate scene config and complete portable robot snapshot.
2. Construct one dual-arm MuJoCo world with the **actual specified object** mesh, scale, transform and declared physics properties, source table and configured camera. Apply optional settling *before* perception.
3. Produce and persist selected world-frame point cloud and camera/mesh provenance.
4. Run right inference on original points **only if right is enabled**; run mirrored inference for left **only if left is enabled**. Allow a test/replay mode with cached, validated candidates to avoid rerunning GPU inference while debugging Selector/physics.
5. Wrap into common generic candidate schema, filter/score, emit full selection report. If no feasible candidate, fail gracefully with reports instead of pretending an execution occurred.
6. Execute **only the selected side** in MuJoCo with the other side present and maintaining its prior target; use selected stage wrist poses / finger commands and arm IK / prescribed motion to realize `pregrasp → grasp → squeeze → lift`.
7. Reuse the established right-side episode semantics and stage timing/control policy **where genuinely applicable**, but carefully adapt them to Phase 1 model and scene. Avoid copying or modifying legacy benchmark controllers wholesale, changing paper hyperparameters, or claiming the Phase 1 numerical hand gains are physically validated. Validate actual motion/path collisions in an appropriate simulation check; static endpoint checks alone are insufficient. If no safe motion to a stage is found, abort clearly and record diagnostics; do not teleport the arm/object during the grasp episode to obtain success.
8. Implement a **documented** simulation success criterion that checks actual object lift/retention relative to its pre-lift pose or table, stability, and relevant contact/trajectory conditions; distinguish `success`, `failed_grasp`, `invalid_initialization`, `no_feasible_candidate`, `motion_failed`, `physics_unavailable`, `sim_numerical_failure` and other meaningful statuses. Do not silently reuse a legacy Shadow benchmark threshold unless it genuinely applies and is explicitly approved.
9. Save deterministic artifacts sufficient to replay the scene and explain selection/execution: effective config, model/checkpoint/source IDs, object snapshot/pose, point cloud, raw and decoded per-side candidates, eligible/rejected candidates, selected side, IK/contacts/effort, trajectory and phase events, object trajectory, evaluation report, and optional renders/video/screenshots. Headless run must succeed without a GUI.

If a feature is not feasible without modifying protected legacy dependencies, implement as much as possible and **report precise incomplete acceptance items**. Never claim an end-to-end complete run when grasp motion/evaluation was skipped.

## 9. Proposed compact implementation organization

Prefer small additions rather than a large new framework; inspect current tree before choosing filenames:

```text
src/ada_grasp_ctrl/
    robot_environment/               # Phase 1 — reuse, avoid breaking public API
    sharpa_mirror.py                 # existing mapping — reuse unchanged
    grasp_selection/                # keep old right API; add generic candidate/side adapter as needed
    experiments/
        mirrored_left_grasp.py       # new, thin mirror+inference integration
        dual_arm_grasp_pipeline.py   # new orchestration, if appropriate
        simulated_object_cloud.py   # new complete/depth point-cloud source (if appropriate)
script/
    run_dual_arm_grasp_pipeline.py   # new independently runnable entry point
config/
    simulation_scene.yaml           # valid example with known asset or explicit user mesh input
/tests
    test_phase2_*.py                # isolated + integration tests
docs/
    phase2-dual-arm-grasp-pipeline.md
```

This is **illustrative layout**. Reuse packages where ownership is clear. Do not invent a second robot description or rewrite `robot_environment` to fit a guessed API. Make all new code/comments/docstrings in English, following `AGENTS.md`.

## 10. Incremental implementation sequence

Implement and validate in this order, keeping each milestone runnable:

**M1 — Scene, real object and complete mesh cloud.** Explicit configured pose/scale, stable source frame, deterministic cloud, same geometry in MuJoCo. Headless scene load; output clear diagnostics.

**M2 — Left mirrored DexLearn.** Original and reflected world clouds, same right checkpoint, proper right→left world wrist and 22 joints, three stages, unchanged legacy right outputs. Offline candidate preview without actuation.

**M3 — Generic candidates and enabled arms selector.** New canonical schema plus legacy right adapter, side-aware IK + static contacts + common effort. Tests for left-only/right-only/both and unsupported two-arm candidate. Do not change old Selector results.

**M4 — Simulated depth.** Actual camera depth and object-only segmentation/visibility-aware mask. World projection, roundtrip tests, debug image/point cloud; no complete-mesh fallback masquerading as depth.

**M5 — Full MuJoCo grasp episode.** Reusable single-selected-side stage control and grasp/lift evaluation. Report physics and movement limitations; avoid stage teleportation.

**M6 — One-command integration, docs and regressions.** Example scene, deterministic candidate replay, actual inference run when available, full summary and visualizations.

## 11. Acceptance tests (must report pass/fail/skip honestly)

### Scene & perception

- User can change **mesh, pose, scale, camera, enabled arms** via YAML only; these changes alter simulation and cloud as specified.
- `complete_mesh` points reproduce source surface and world pose under seeded sampling.
- `simulated_depth` points arise from visible object surfaces, not table/arm/background or hidden mesh faces; pixel reprojection consistent with camera calibration.
- Invalid asset, invisible object, wrong scale/frame, missing physics properties, invalid camera and missing snapshot yield explicit meaningful diagnostics.
- Object pose used for perception, candidate transformation, collision, and execution is the same final pose (including settling).

### Mirroring and candidates

- Reflect world cloud once; reflected wrist mapped back once; same plane; repeatable input seed.
- Right-hand candidate/three stages unchanged from incumbent inference for the same inputs; left stage transforms and 22 joints pass independent FK/landmark and frame tests.
- Generic candidate validates `required_arms` and side targets; old right-format input still accepted by compatibility layer; two-arm candidate is parsed but explicitly unsupported for evaluation.
- Left and right candidate identifiers remain unambiguous, even when raw indices match.

### Selector

- `[right]` only considers right, `[left]` only left, `[left,right]` considers both **and selects only one**.
- Robot `q56` includes non-selected side. Static contact checks happen in the actual object world and do not blanket-reject expected hand/object contact.
- Same six-CRX-joint effort weights/units across sides. Carefully crafted fixture verifies global minimum, deterministic tie handling and rejection of infeasible/NaN options.
- The original right selector fixture produces its **unchanged** selection and score; no legacy public input/output default breaks.

### Execution

- Headless full pipeline runs `pregrasp → grasp → squeeze → lift` against configured real mesh, records object trajectory and physical outcome; right/left/both-enabled scenarios are tested at least on deterministic small fixtures.
- Execution actuates **only selected** 28-DoF target; passive robot remains in the scene/contacts. No ROS modules required or launched; no hardware command published.
- If IK/path/contact/controller fails, no fake successful lift; report stage and cause. Produce replayable run artifacts.
- Clearly distinguish tests with a real DexLearn checkpoint/GPU from replayed fixture candidate tests; do not claim full inference acceptance based only on fixture injection.

### Regression & quality

- Run focused mirror, Phase 1 environment, right CRX/Sharpa, selector and pipeline tests, plus repo compile/lint as practical. Preserve existing golden/reference data.
- Separately run/document the existing Shadow quick classification check, knowing Phase 1 recorded **3 mismatched statuses**; do not attribute or silently “fix” it without evidence. Report all unrelated test failures.
- Confirm `git diff --name-status` shows no unapproved edits to ROS execution, protected right inference/benchmark, existing robot assets, external repos, submodules or source hyperparameters.

## 12. Final deliverables and Codex response format

Deliver actual source code and tests (not just a design document), plus:

- `script/run_dual_arm_grasp_pipeline.py` — one full offline pipeline CLI.
- Usable example scene YAML; document which asset paths/physics fields the user must supply and provide **working commands against an actually available local fixture**.
- `docs/phase2-dual-arm-grasp-pipeline.md` — exact command lines for complete mesh, simulated depth, `enabled_arms` variants, replay, outputs, validation and known limitations.
- Versioned generic candidate schema with `required_arms` + per-arm `targets`, plus backward-compatible right adapter.
- Evidence for mirroring, IK, collision/effort, real-scene camera/mesh consistency, grasp episode and outcome.

When done, summarize:

1. Files added/changed and public APIs.
2. **Actual executed commands** for end-to-end simulation; provide at least one run output path.
3. Which steps used the actual checkpoint and which were fixture/replay only.
4. Scene source, object pose/mass/collision assumptions, camera and point cloud diagnostics.
5. Selected side, candidate ID, effort, IK/collision diagnostics and grasp episode result for a reproducible example.
6. New tests (counts) and right-hand/Phase 1 regression results, including the known Shadow quick mismatch status.
7. Exact limitations: uncalibrated mounts, source mesh collisions/convex hull approximations, provisional actuation, partial-cloud inference distribution shift, static vs trajectory safety, no hardware execution, no bimanual cooperative grasps.
8. Confirmation original right pipeline works as before and all user work is preserved.

**Definition of done:** An operator can edit a scene YAML to set a real object's asset/pose/scale and camera, enable left and/or right, then run a new Python CLI that generates the chosen point cloud, gets real checkpoint candidates, selects **one** feasible lowest-CRX-joint-effort arm, executes the complete grasp episode in offline MuJoCo, and writes an honest success/failure report without modifying the established right-hand or ROS execution pipelines.
