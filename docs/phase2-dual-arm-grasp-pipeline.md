# Offline single-arm selection in the dual-arm scene

This independent experimental pipeline adds configurable real-mesh perception, original/right and
reflected/left DexLearn inference, minimum CRX joint-displacement selection, and a recorded MuJoCo
grasp episode. It does not change the maintained hand/method matrix or the historical right/ROS pipeline.

**Validation status, 2026-11-11:** after restoring historical Sharpa contact numerics, both the
original/right and moved/left bottle trials physically lift and retain the bottle approximately
18.5 cm. Their strict Phase 2 status is still **failed_grasp** because hold speed exceeds 0.02 m/s
at some samples. This is distinct from the preceding source-contact-profile runs, which lost the
bottle during squeeze. No success-rate or hardware-readiness claim is made. See the restored-profile
section below for exact settings, unchanged constraints, and current-run evidence.

## Run

From the checkout, using its existing environment:

```bash
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py \
  --environment simulation --scene config/simulation_scene.yaml \
  --output-root output/phase2/my_checkpoint_run
```

Use a new output directory for every invocation. The CLI refuses to overwrite existing runs.
Relative scene paths use Ada's data-root resolution; relative output paths use its output root.
They are not relative to the YAML file or the caller's working directory. Absolute paths work too.
The script selects EGL before importing MuJoCo; no display or ROS process is needed.

The checked-in example uses an actually available bundled DGN bottle and the final local Phase 1
`output/dual_arm/phase1_20261111/commit_snapshot`. Unlike the earlier `snapshot_complete`, this snapshot
contains `initial_pose.yaml`. Set `scene.robot_snapshot` to your own complete portable export on
another machine. `initial_robot_state` accepts `snapshot_initial_pose` or a mapping of all 56 canonical
joint names to radians. Native MuJoCo joint order is never assumed.

`inference.reference` reuses the checkpoint, training-config and metadata paths/checksums from the
existing right-hand inference report. Alternatively provide explicit `checkpoint`, `training_config`
and `metadata` paths. Also specify `python`, `dexlearn_root`, `device` and `samples_per_side`.
The virtualenv executable path is preserved without resolving its symlink to the base interpreter.
Both branches call the unchanged `fixed_camera_grasp` inference module, including its training-contract
checks and original three-stage decoder. No dependency installation or checkpoint conversion occurs.

For a perception-only check:

```bash
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py \
  --environment simulation --scene config/simulation_scene.yaml \
  --perception-only --output-root output/phase2/my_perception
```

To generate directly runnable YAML variants without changing the source example:

```bash
PYTHONPATH=src .conda/bin/python - <<'PY'
from copy import deepcopy
from pathlib import Path
import yaml
base = yaml.safe_load(Path('config/simulation_scene.yaml').read_text())
for name, arms, source in [
    ('left', ['left'], 'complete_mesh'),
    ('right', ['right'], 'complete_mesh'),
    ('both', ['left', 'right'], 'complete_mesh'),
    ('depth', ['left', 'right'], 'simulated_depth'),
]:
    scene = deepcopy(base)
    scene['selector']['enabled_arms'] = arms
    scene['point_cloud']['source'] = source
    Path('/tmp/ada_phase2_' + name + '.yaml').write_text(yaml.safe_dump(scene))
PY

PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py \
  --environment simulation --scene /tmp/ada_phase2_depth.yaml \
  --perception-only --output-root output/phase2/my_depth
```

Remove `--perception-only` to run actual inference and selection with depth input. The model was
trained on complete clouds; partial-depth inference has an unvalidated distribution shift.
Only enabled inference branches run. Enabling both selects one side, never a cooperative grasp.

## Interactive object placement

The existing Phase 1 Viser viewer can load the complete Phase 2 scene, including both canonical
CRX/Sharpa branches, the original table, and the scaled object mesh:

```bash
PYTHONPATH=src .conda/bin/python script/preview_dual_arm_web.py \
  --scene config/simulation_scene.yaml --port 8089
```

Open `http://localhost:8089`. The **Object placement** folder has only X/Y sliders in meters.
Height and orientation remain exactly as defined in the loaded scene. Load a prepared asset pose
on the table, move X/Y, then save. The pose uses the configured mesh origin; scale is unchanged.
Object edits leave all 56 robot joints unchanged. Robot joint sliders remain preview-only.
No physics stepping or gravity settling runs in the preview.

Click **Save Scene** to create a unique YAML under `output/phase2/scene_edits/`. The UI and terminal
(`SAVED_SCENE=...`) show its absolute path. Alternatively, choose a new filename at startup:

```bash
PYTHONPATH=src .conda/bin/python script/preview_dual_arm_web.py \
  --scene config/simulation_scene.yaml \
  --save-scene output/phase2/scene_edits/my_bottle.yaml --port 8089
```

Only `object.pose.position_m` and `object.pose.quaternion_wxyz` change. Robot state, mesh paths,
collision meshes, scale, camera, physics, point-cloud settings, inference, enabled arms, Selector,
execution and output settings are preserved. YAML comments/formatting are not retained.
The source YAML and any existing destination are never overwritten, including source symlink aliases.
An explicit destination can be saved once; default filenames are unique for every save.
Relative input/output paths follow the same data/output roots as the pipeline, not the YAML directory.

Pass the saved absolute path to the existing pipeline. First verify perception without inference:

```bash
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py \
  --environment simulation --scene /absolute/path/to/saved_scene.yaml \
  --perception-only --output-root output/phase2/edited_bottle_perception
```

Inspect `perception.json` and `object_cloud.npz`: both record the updated `T_world_object` and the
cloud uses that pose. Remove `--perception-only` and use another fresh `--output-root` to run the
unchanged inference/selection/execution pipeline. A source scene with `placement: settle_on_table`
retains that setting: the pipeline will settle from the saved pose, whereas the editor never settles.

Optional camera and cloud inspection uses the existing Phase 2 perception methods:

```bash
PYTHONPATH=src .conda/bin/python script/preview_dual_arm_web.py \
  --scene config/simulation_scene.yaml --camera-preview --port 8089
```

Click **Refresh camera / point cloud** after moving the object. It displays the configured simulated
camera RGB image and the configured `complete_mesh` or `simulated_depth` cloud. It never runs DexLearn.
These previews refresh on demand; they remain at the last refresh until the button is clicked again.
The existing camera visibility/FOV errors are reported in the UI. Headless rendering defaults to EGL.
Converted meshes and camera artifacts are temporary and cleaned up when the viewer exits normally.

Implementation is limited to `script/preview_dual_arm_web.py` and the new
`experiments/object_placement_preview.py` controller; the Phase 2 loader and algorithms are unchanged.
The original `--snapshot` and `--initial-pose` workflow remains available.

Placement validation (2026-11-11) uses the existing `commit_snapshot` and bundled bottle:

```bash
PYTHONPATH=src MUJOCO_GL=egl \
  ADA_DUAL_ARM_SNAPSHOT="$PWD/output/dual_arm/phase1_20261111/commit_snapshot" \
  .conda/bin/python -m unittest discover -s tests -p 'test_object_placement*.py' -v
```

The 10 tests cover actual compiled mesh scale/origin, free and fixed object poses, unchanged robot
state/camera/geometry/time, unique saves and overwrite refusal, preserved settings, reload equivalence,
and the existing perception-only CLI's transformed point cloud. Real HTTP/WebSocket tests verify
the original colored meshes and 56 joint controls, object render transforms, Save Scene, camera/cloud
refresh, and legacy snapshot reset. Viewer subprocesses reject ROS, inference and episode imports.
The existing 18 Phase 2, 11 Phase 1 and 56 CRX/Sharpa/Selector/mirroring tests also pass.
Repository compilation and Ruff lint pass. Changed Python files pass format checks; the whole-repository
format check still reports the same 8 pre-existing files described in the Phase 1 report.
No manual browser screenshot review, full release gate or physical hardware run is claimed.

## Scene and perception contract

- `object.mesh` is a real triangle mesh supported by installed trimesh. Scene transforms are retained;
  scale and the full world XYZ/WXYZ pose are applied once. Position means the **mesh frame origin**.
  Converted run-local STL geometry is used in MuJoCo, preserving that origin.
- `collision_meshes` explicitly lists the object's physical collision pieces. Each gets the same
  scale and world transform as the visible object. MuJoCo uses each piece's convex hull.
  Unsupported/degenerate/missing assets, nonpositive scale and non-unit quaternions fail explicitly.
- Dynamic evaluation requires `freejoint: true`, `physics.mass_kg`, three friction coefficients,
  `physics.assumptions`, and explicit collision meshes. Missing physics permits perception and
  selection diagnostics, then reports `physics_unavailable`. A missing collision declaration with
  supplied physics is an input error. Diagnostic-only visual-hull contact is clearly distinguished.
- The example's 0.1 kg mass and `[0.8, 0.02, 0.001]` friction are **explicit exploratory assumptions**,
  not measured values or changed benchmark settings. Mass is divided equally between the declared
  convex pieces and inertia is calculated from their uniform-density volumes. Compiled mass and
  inertia are recorded. Supply measured/source properties for a meaningful experiment.
- The example bottle uses scale `[0.08, 0.08, 0.08]`, origin `[0.08, -0.08, 0.072]` m and a 90-degree
  X rotation. The unmodified source table is 0.30 × 0.30 × 0.04 m. Source robot bases/mounts are retained.
- `explicit_pose` does not settle or relocate the object. `settle_on_table` steps the real scene before
  perception, requires 100 consecutive steps with free-joint velocity norm below 0.001, and fails
  after 5 simulated seconds. The final object pose and robot state are used thereafter. The example
  bottle did **not** meet this strict settling criterion; `implementation_settle_01` records the failure.
- `complete_mesh` deterministically samples the actual scaled surface with the configured seed.
  It is an oracle input, not a simulated RealSense observation.
- `simulated_depth` renders metric axial depth through MuJoCo 3.6's Renderer, whose API performs
  depth-buffer linearization. Object geom segmentation excludes robot, table and background.
  Collision hulls are hidden from rendering so they cannot create false visible surfaces. Unprojection
  uses pixel centers, camera +X right/+Y up/-Z forward, vertical FOV and actual camera transforms.
  All visible valid pixels are retained; downstream existing inference preprocessing samples its
  required count. Invisible/out-of-range/occluded objects with fewer than three pixels fail;
  no complete-mesh fallback exists.

The bundled object derives from the existing DGN 2k quick subset. See `examples/data/README.md`
and `examples/quick_manifest.json` for source identities and checksums. No object or robot assets,
attribution facts, permission records, or redistribution claims were changed. The repository has
no project-wide license; upstream asset restrictions remain applicable.

## Candidates and minimum effort

`grasp_selection.multi_arm.validate_candidate` defines schema 1: `candidate_id`, `required_arms`,
`targets[side][pregrasp|grasp|squeeze]`, and `provenance`. Top-level conventions are `frame_id: world`,
`unit: m`, `joint_unit: rad`; each stage requires the exact `{side}_hand_flange` frame, a proper
`T_world_hand`, and all 22 canonical named finger positions. Both-arm schemas parse but return
`unsupported_multi_arm_candidate`. `from_right_record` is the dedicated compatibility adapter for
existing decoded right records; it does not alter the incumbent selector API or outputs.

Left inference reflects points around a single recorded plane before random preprocessing, then
uses the **same right-full checkpoint**. Default plane normal is `[0,1,0]`; its point is the original
cloud centroid. `SharpaHandMirror` maps the predicted reflected-world wrist back to the original
world and maps all three 22-joint stages. There is no extra pregrasp extrapolation. The physical
object stays unreflected and the robot uses the real exported left URDF. Original raw indices,
probabilities, decoded records, preprocessing, checkpoint identities and plane are retained.

For each enabled side and bounded accepted IK solution, the cost is exactly

```text
sum_j W[j] * (q_grasp_CRX[j] - q_current_CRX[j])², j = 1…6
```

The default weights come from the existing `JointDisplacement` criterion: six ones. Both arms use
the same optional six-element `selector.weights`. There is no finger effort, angle wrapping,
distance-to-object score, likelihood weighting, or side preference. “Effort” means a joint-displacement
proxy in rad², not measured torque, power or energy. Selection minimizes across feasible candidates
and the evaluated finite seeds; it does not claim to enumerate all continuous IK branches.

The wrapper evaluates the current arm seed and the incumbent selector's five fixed fallback seeds.
Out-of-bounds seeds and rejected/malformed/nonfinite IK are recorded, never clipped into acceptance.
The grasp solution seeds pregrasp/squeeze checks, with the other 28 joints held at their current
state. Equal cost breaks by candidate ID, then seed index. The result contains the selected side,
source ID, six CRX targets, three 22-finger targets, full stage states, effort and rejection diagnostics.

Contact eligibility keeps every contact diagnostic. The **active hand/object** 10 mm penetration
gate applies only during initialization and pregrasp, including the approach path and live approach
checks. Grasp, squeeze, lift and hold retain these contacts without rejecting their penetration depth.
This policy is shared by endpoint selection, sampled paths and live checks; MuJoCo contact response
remains enabled. Same-hand self-contact and hand/table contact retain their 10 mm gates. Arm/object,
arm/table, passive hand/object, inter-arm and other environment contacts retain the 10 µm gate.
The existing adjacent base/J1 exclusions are unchanged. No collision geom is globally disabled.

## Execution, replay and outcomes

`dual_arm_grasp_episode.execute` keeps Phase 1 source gains and the 2 ms implicitfast timestep.
It sends only the selected side's 28-joint targets, retaining the passive side's prior command.
The explicit experimental `arm_bias_compensation` option supplies model-based feedforward to both
arms for gravity holding. Combined actuator/applied arm torque is bounded by source effort at each
command evaluation; object gravity is unchanged. This is a new simulation option, not a change to
the benchmark/hardware controller, and its physical accuracy has not been validated.

For legacy scenes with no `execution.protocol`, or with `protocol: position`, pregrasp, grasp,
squeeze and +world-Z lift use bounded IK and quintic position interpolation, with
the existing 2 s stage convention as the example default. All command segments are checked with
joint increments at most 0.02 rad before execution. During physics, numerical warnings are checked
every step and actual contacts/joint limits every 20 ms. Soft-limit overshoot is reported separately
with an explicit 0.01 rad diagnostic bound; commands remain strictly inside source limits. Stage
arrival checks arm angles and the actual flange pose; finger tracking errors are recorded because
intentional contact may prevent full finger closure. No state teleport is used to obtain an outcome.

Static selection alone has `execution_ready: false`. Sampled path validation permits simulation
launch; it does not certify all intermediate continuous motion or hardware safety. Motion/collision/
arrival failure aborts and saves the partial trajectory. A failed minimum-cost candidate is not
silently replaced with a more expensive candidate after execution starts.
Initial object/table or object/environment penetration beyond 2 mm also rejects execution before stepping.

The independent experimental success criterion requires the object origin to remain at least
`minimum_lift_m` above its **pre-lift actual pose** throughout the hold interval, with linear speed
below `maximum_object_speed_m_s` and active hand/object contact at every hold sample. Example values
are 0.03 m and 0.02 m/s for a commanded 0.05 m lift and 1 s hold. These are disclosed diagnostic
settings, not a reused or revised Shadow benchmark success-rate definition.

Reported statuses include `success`, `failed_grasp`, `invalid_initialization`, `motion_failed`,
`no_feasible_candidate`, `physics_unavailable`, `sim_numerical_failure`, `solver_degraded`, `perception_only`, and
`preflight_or_runtime_failed`. Exit codes are 0 for success or requested perception-only completion,
1 for a diagnosed unsuccessful selection/physical episode, and 2 for input/environment/runtime failure.

Explicit FK fixtures exercise motion without claiming checkpoint inference or an object grasp:

```bash
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py \
  --environment simulation --scene config/simulation_scene.yaml \
  --fixture-current-pose --output-root output/phase2/my_fixture

PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py \
  --environment simulation --scene /tmp/ada_phase2_left.yaml \
  --replay output/phase2/my_fixture/candidates.json --output-root output/phase2/my_replay

PYTHONPATH=src .conda/bin/python script/replay_dual_arm_grasp.py \
  --run output/phase2/my_fixture --output output/phase2/my_fixture/replay.gif --stride 10
```

Candidate replay validates schema and a hash binding the mesh, collision/physics, robot snapshot,
current state, camera, object pose and exact cloud. Enabled arms may change; geometry/state changes
require regenerating candidates. A replay retains the original candidate provenance, so replaying an
FK fixture never becomes real inference evidence. The render script also supports a final `.png`.

Runs save effective YAML, invocation/manifest, source and converted geometry, object/camera provenance,
world cloud, raw and preprocessed per-side inference, generic candidates, full selection, and episode
diagnostics. Depth runs also save RGB, depth, segmentation and mask. `scene.mjb` embeds the compiled
model/meshes; `initial_state.npz` and `trajectory.npz` retain actual qpos/qvel/ctrl/applied force, phase,
time and object transforms. MJB playback requires a compatible MuJoCo version. Playback displays
recorded states without rerunning inference or pretending to perform a new physical episode.

## Native CRX protocol extension (2026-11-11)

Set `execution.protocol: crx_overhead_native_ours` in a **new** scene to use the previously
successful right-CRX motion/control structure described in [crx-sharpa-right-smoke.md](crx-sharpa-right-smoke.md).
Selection is unchanged: rank feasible candidates by weighted squared displacement of the active six
CRX joints, then execute the single minimum-effort winner. No higher-effort fallback is substituted.
The implementation supports either selected side; the physical saved-scene run below selected right.

- Approach: keep configured open fingers, raise clearance, move the palm 30 cm above the object,
  orient above, then descend to the selected pregrasp. Reuse the historical pose interpolation;
  each segment takes at least 2 s, translation is at most 0.1 m/s, and rotation at most 0.5 rad/s.
  IK runs at 10 Hz with the existing acceptance tolerances, branch/velocity bounds and five static
  interpolation checks. Require measured palm arrival within 1 mm / 0.01 rad, waiting at most 3 s.
- Grasp/squeeze: use the unchanged shared native `ours` runner, controller objectives, 15 N target,
  2 s approach and 2 s squeeze reference paths, and native stop policy (at most 80 actions).
  A reduced Pinocchio model retains world mounts and active contact frames. Only this kinematics
  model locks inactive joints; both arms remain dynamic in the full MuJoCo simulation.
- Lift: start at the measured flange pose, retain finger commands, and solve 20 continuous-seed
  Cartesian Z targets over 2 s with the historical 0.35 rad branch bound. The new validation scene
  explicitly requests `lift_height_m: 0.2`. Existing `durations_s` and joint arrival settings belong
  to the legacy position protocol; native timing and palm arrival follow the historical settings above.
- Keep the loaded scene, object freejoint, gravity, object properties, 2 ms timestep, arm compensation,
  collision policies and retained-lift criterion. Unlike the historical smoke, global gravity is not
  disabled and no extra 1 N downward force is applied. The old smoke's success definition is not used.
  The object is never reset between inference, collision checking, approach and execution.
- Record full-state `trajectory.npz`, native `control.npy`, consumed `native_controller.yaml`, segment
  arrival errors, contacts, stage completion and solver degradation. Solver abort skips lift and saves
  partial records. Physical success cannot mask degraded command solves.

The input copy is
`output/phase2/scene_edits/my_bottle_right_native_crx_20261111_01.yaml`; the user's original scene
is unchanged. Run it with a fresh destination:

```bash
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py \
  --environment simulation \
  --scene output/phase2/scene_edits/my_bottle_right_native_crx_20261111_01.yaml \
  --output-root output/phase2/my_native_crx_run
```

Final-code actual-checkpoint evidence: `output/phase2/native_crx_20261111_02`.
The scene identity is `39671a73fd13cd6cfef1f1f77efd135fe25bee6a8a41b0190bc6f152898025e4`, matching
the earlier saved-scene run. The winner is `right:c0004`, effort `0.7466237874870858 rad²`.
All four overhead segments reach the historical arrival tolerances, with no hand/object contact
throughout pregrasp. Initial object XYZ is `(0.607, -0.575, 0.0709401)` m; at native control start
it is approximately `(0.606944, -0.574925, 0.0704711)` m: 0.478 mm total displacement.
This is observed resting behavior during approach, not a new settling operation or a reset.

Native control records hand/object contacts in 44 of 80 actions. Total sampled normal force peaks
at 14.962 N, below the native 15 N stopping target; the bottle then slips out during squeeze.
Contacts are absent from action 52 onward and during lift. Pregrasp, grasp, squeeze, the 20 cm
commanded lift and the 1 s hold all complete, but the bottle is not retained: minimum hold lift
relative to pre-lift is -0.000805 m, maximum hold speed 0.05114 m/s, and retained contact is false.
The result is `failed_grasp`, exit 1, with **no solver degradation**, no forbidden collision abort,
and unchanged passive commands. This isolates the remaining failure to physical grasp retention;
it does not establish a high success rate for this bottle or candidate.

Validation for this extension: five focused tests pass, covering both-side route semantics,
full/reduced-model FK and Jacobian equality, passive commands, no object reset, gravity, malformed
commands, and solver-abort partial records. The full suite runs 338 tests with nine skips and no
failures. `ruff check src tests script`, `python -m compileall -q src tests`, changed-file formatting,
and `git diff --check` pass. Whole-repository formatting still reports the same eight unrelated files.
The fresh quick gate at `output/phase2/native_crx_release_20261111_01` stops at the same three Shadow
classification differences recorded before this extension; later hands are not reached. This is not
a passing release gate. No golden was changed. Exact commands, differences, source hashes and logs
are preserved in the final run's `validation_report.json` and `validation_*.log`.

## Explicit pregrasp opening and Y-only placement comparison (2026-11-11)

A scene may opt into `candidate_preparation: {pregrasp_open_multiplier: 3.0}`. The default is 1
(exact legacy behavior). After unchanged inference and before collision selection, each side's
pregrasp finger vector becomes `clip(q_grasp + 3 * (q_pregrasp - q_grasp), source_joint_limits)`.
This is a multiplier of the joint-angle difference, not a claim of three times the physical aperture.
Only the 22 pregrasp finger targets change. All wrist poses, grasp/squeeze targets, checkpoint,
IK tolerances, collision thresholds, controller objectives and six-CRX-joint effort ranking remain
unchanged. Raw `candidates.json` remains replayable; `execution_candidates.json` contains the exact
adjusted targets consumed by both selection and execution. Provenance records original/requested
angles and every clipped joint. Applying preparation twice is rejected.

The 3x bottle candidate clips `right_thumb_CMC_AA` from -0.370947 to its source lower limit -0.3491 rad.
For the same `right:c0004` and original scene, static collision-geometry distance at pregrasp increases
from 2.341 mm to 18.320 mm. This is endpoint clearance, not a guarantee for the continuous approach.

Clipped pregrasp targets exposed two native-adapter integration issues. First, MuJoCo's measured
angle can slightly exceed a soft joint limit (observed 0.000852 rad), while the shared native runner
uses measured angles to initialize its command. Only that startup command is projected into source
limits, within the existing 0.01 rad soft-limit bound. Measurements and simulated positions are never
changed; larger excursions still fail. Second, SLSQP/interpolation can return a valid boundary angle
a few floating-point units outside the exact bound (observed 2.16e-15 rad). The adapter projects only
roundoff bounded by 64 machine epsilons times the joint-limit scale, records every occurrence, and
continues to reject substantive out-of-range commands. Solver policy and diagnostics are unchanged.

The comparison uses the user's X=0.607 m and starting Y=-0.575 m, then only changes Y to -0.300 and
0 m. The model's left base is at Y=0; right base Y=-0.911563. Both sides remain enabled. The bottle's
scale/orientation/Z are unchanged. Three already bundled assets use their original dataset orientation
and scale 0.081: `ddg_kit_HeringTin`, `ddg_kit_SoftCheese`, and `ddg_gd_box_poisson_020`, each from
`tabletop_ur10e/scale008_pose000_0.npy`. Their world heights are approximately 2.7, 3.3 and 2.9 cm.
For each new asset, Z places its collision-mesh bottom 0.1 mm above the existing table. Z remains
fixed during its Y sweep. All trials retain the explicitly experimental 100 g mass, friction and
gravity. No asset files, golden data, original scene, or earlier outputs are overwritten.

Actual-checkpoint candidates for the new objects/positions are saved in
`output/phase2/pregrasp3_flat_y_20261111_01`. The original bottle uses the exact earlier checkpoint
candidates to isolate pregrasp opening. Subsequent adapter corrections replay the identical
scene-bound candidates in fresh run roots, with `inference_mode: cached_candidates` explicitly
recorded. Final comparison: `output/phase2/pregrasp3_flat_y_20261111_03/comparison.json`.
Scene copies and source-pose/mesh checksums are in that run's `scenes/` and `experiment_manifest.json`.
These few chosen assets/poses are diagnostic examples, not an estimated grasp success rate.

Final minimum-effort side selection (both sides enabled throughout):

| Asset | Y=-0.575 m | Y=-0.300 m | Y=0 m |
| --- | --- | --- | --- |
| Bottle | right (0.746624) | left (0.522812) | left (1.855854) |
| Flat tin | left (0.517426) | left (0.692123) | left (0.629417) |
| Cheese box | right (0.761597) | left (0.547008) | left (0.573511) |
| Thin box | right (0.216630) | left (0.358866) | left (0.700728) |

Costs are rad². At Y=-0.300, the bottle's right-arm minimum is 1.913931 versus the left's 0.522812;
this is a cost-based side switch, not an imposed left-arm preference. Final outcomes: six completed
`failed_grasp` episodes (three bottle, flat tin at Y=0, thin box at original/middle), four command-limit
aborts during squeeze (flat tin original/middle, cheese original/left), one measured soft-limit abort
(cheese middle), and one overhead-IK failure (thin box left). No successful lift is claimed. Native
solver degradation is false for the eleven stepped episodes. Later command excursions beyond the
machine-roundoff projection remain rejected; the pregrasp clipping request does not retune optimizer
constraints or disable runtime gates. Pre-control object drift is approximately 0.48 mm for the bottle,
0.25 mm for the tin, 4.3 mm for cheese, and 16.2 mm for the thin box. The latter scenes are not evidence
of negligible placement drift, and no unstable pose is described as settled.

The candidate/pregrasp/adapter checks cover signed extrapolation and source clipping,
unchanged wrists/grasp/squeeze, raw replay, matching selection/execution targets, strict normal command
limits, and unchanged measured states. Eleven focused tests and the final full suite pass (344 tests,
nine skips). Lint, compile, changed-file format, and whitespace checks pass. The quick gate remains
nonpassing with the same three prior Shadow differences; it ran before the private adapter corrections
and no maintained controller implementation or golden changed. See `validation_report.json` for exact
commands and source hashes. `pregrasp_comparison.json` records the endpoint clearance measurement;
`preview/index.html` contains videos of the original/right and moved/left bottle and each flat asset.

## Restored historical Sharpa contact profile (2026-11-11)

The user authorized restoring the old numerical settings after auditing the actual successful
`g0000/ours/control.scene.xml` in
`output/regression/crx-sharpa-right-smoke/20261110T010645248343Z-21280`. New scene copies use:

```yaml
scene:
  # Other scene fields remain as saved.
  contact_profile: sharpa_native_v3
```

| Setting | Restored value |
| --- | --- |
| All geom friction (hand/object/table) | `[0.8, 0.02, 0.001]` |
| Contact dimension | `4`, including torsional friction |
| `solref` | `[0.005, 1]` |
| `solimp` | `[0.5, 0.99, 0.0001, 0.5, 2]` |
| Timestep / integrator | `0.004 s` / `implicitfast` |
| Friction cone / `impratio` | `elliptic` / `10` |
| `noslip_iterations` | `2` |

Both Sharpa hands already match the historical motor settings exactly: kp=7.5, kv=0, passive joint
damping=0.05, armature=1e-5, gear=1, and all 22 named force limits. They are not modified. The profile
sets contact numerics on the assembled scene before compilation, covering the object as well as
both hands and table. Configured object friction must match the historical triple or loading fails,
so YAML cannot silently declare a conflicting value. The resolved profile enters perception metadata
and scene identity; source-profile candidate caches cannot silently bypass that identity change.
An omitted profile or `source` preserves older scene behavior and existing regression fixtures.

Global gravity remains enabled, as explicitly requested earlier; no additional downward force is
applied. Existing arm compensation, geometry, mass, freejoint, object XYZ/WXYZ, 3x pregrasp with
source-limit clipping, six-joint minimum-effort ranking, IK/collision limits, native controller
objectives and strict Phase 2 success criterion are retained. Point clouds still use 2,048 samples
from `complete_mesh`, not camera depth. Numerical warnings remain checked each step; diagnostic
checks and trajectory samples remain every 20 ms (five steps at 4 ms). The retained 56-joint URDF
geometry is not replaced by the historical right-only benchmark geometry, so this is a restoration
of numerical settings, not an exact reproduction of the old full scene.

Fresh actual-checkpoint runs are in `output/phase2/restored_sharpa_20261111_01`:

| Scene | Selected candidate | Effort rad² | Physical outcome / strict status |
| --- | --- | ---: | --- |
| Bottle, original Y=-0.575 | right:c0004 | 0.746624 | Retained lift >=18.496 cm; speed gate fails |
| Bottle, Y=-0.300 | left:c0004 | 0.522812 | Retained lift >=18.441 cm; speed gate fails |
| Flat tin, original XY | See selection.json | See selection.json | Squeeze command-limit abort |
| Cheese box, original XY | See selection.json | See selection.json | Squeeze command-limit abort |
| Thin box, original XY | right:c0006 | 0.216630 | Completed motion without retained lift |

Both bottle runs complete pregrasp/grasp/squeeze/lift/hold and retain hand/object contact throughout
hold, without solver degradation. Maximum hold speeds are 0.118485 and 0.117656 m/s, respectively,
above the unchanged 0.02 m/s criterion. Final recorded speeds decrease to 0.017364 and 0.014949 m/s,
but the last half-second still contains faster samples; they are not classified as stable throughout
hold. Their official `failed_grasp` and exit code 1 are preserved. The three flat-object runs also
remain unsuccessful. Settings were restored as a bundle; this comparison does not isolate which
individual contact parameter caused the change.

Five focused tests verify effective 4D contact friction, exact compiled agreement with the actual
historical scene, unchanged actuators/geometry/mass/initial states, explicit conflict rejection,
and 20 ms sampling at both timesteps. The full suite completes 349 tests with nine skips and no
failures. Lint, compile and changed-file format checks pass. Validation and unchanged quick-gate
limitations are recorded in `validation_report.json`; no golden is promoted. The new scene YAMLs
are under `scenes/` and the recorded videos are under `preview/` in that run root.

## Visual-only selected-grasp inspection

`script/preview_selected_grasps.py` opens the selected thin-box and cheese-box targets from saved
runs. The default is `grasp`, with only the active hand and object visible. The object/stage dropdowns
switch between the two assets and `pregrasp`/`grasp`/`squeeze`; the context checkbox shows arms and
table. The object stays at the saved inference pose. These are static selected IK targets, not new
predictions or dynamically achieved poses. No inference, controller, physics steps or hardware run.

```bash
PYTHONPATH=src .conda/bin/python script/preview_selected_grasps.py \
  --thin-box-run output/phase2/restored_sharpa_20261111_01/thin_box_original \
  --cheese-box-run output/phase2/restored_sharpa_20261111_01/cheese_box_original \
  --port 8094
```

Collision render groups 0 (robot/table) and 3 (object) are hidden; collision response and all saved
models/trajectories remain intact. The same groups are now hidden by the headless Phase 2 replay
script. Clean versions of the five restored-profile videos are in
`output/phase2/restored_sharpa_20261111_01/preview_clean_01`; previous videos are preserved.
Two focused tests verify saved joint targets, unchanged object/physics/source files, and visual-only
context grouping. Actual HTTP/WebSocket checks cover both object switches, all three finger poses,
and hidden collision mesh handles. Lint, compile and changed-file format checks pass. This display-only
change does not rerun grasp simulations or alter scientific results.

## Executed validation and remaining acceptance items

The commands below were actually executed, with fresh output roots:

```bash
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py --environment simulation \
  --scene config/simulation_scene.yaml --output-root output/phase2/validated_checkpoint
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py --environment simulation \
  --scene /tmp/ada_phase2_depth.yaml --perception-only --output-root output/phase2/validated_depth
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py --environment simulation \
  --scene /tmp/ada_phase2_depth.yaml --output-root output/phase2/validated_depth_checkpoint
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py --environment simulation \
  --scene config/simulation_scene.yaml --fixture-current-pose --output-root output/phase2/validated_both
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py --environment simulation \
  --scene /tmp/ada_phase2_left.yaml --replay output/phase2/validated_both/candidates.json \
  --output-root output/phase2/validated_left
PYTHONPATH=src .conda/bin/python script/run_dual_arm_grasp_pipeline.py --environment simulation \
  --scene /tmp/ada_phase2_right.yaml --fixture-current-pose --output-root output/phase2/validated_right
```

| Evidence | Actual result |
| --- | --- |
| Real right-full checkpoint | `step_100000.pth`, SHA-256 `92fc78e580753a3659116e867e075e91d75a342176abd2434b9d98828b3083d6`; both GPU branches ran |
| `validated_checkpoint` | 8 predictions per side; no feasible candidate, no execution attempted |
| `implementation_checkpoint_32` | 32 predictions per side at the same pose; no feasible candidate |
| `implementation_table_corner` | 32 per side at `[-0.11,-0.11,0.072]` m; no feasible candidate |
| `validated_depth` | 624 visible object points; 0.8125% image coverage; reprojection error ≤5.7e-14 pixels |
| `validated_depth_checkpoint` | Actual checkpoint on partial depth, 8 predictions per side; no feasible candidate |
| `validated_both` / `validated_left` | `left:fk_fixture`, effort 0 rad²; all four phases completed; `failed_grasp` |
| `validated_right` | `right:fk_fixture`, effort 0 rad²; all four phases completed; `failed_grasp` |
| Passive robot | Prior command target retained; real passive motion remains part of the scene/contact diagnostics |
| Source gains without feedforward | Diagnostic fixture failed stage arrival under gravity; recorded in `implementation_fixture_02` |
| Settling | Example did not satisfy strict stability gate; explicit failure, no stale-pose inference |

The complete-cloud actual runs reject unaccepted IK, hand joint limits, inter-arm/base/hand collisions,
and predicted finger self-contact. Exact pairs, penetrations and stage residuals are in each selection
report. The minimum-cost fixture is intentionally trivial; separate analytical tests verify different
nonzero left/right costs, current-state dependence, weights, deterministic ties and rejection paths.

Checks executed:

```bash
PYTHONPATH=src ADA_DUAL_ARM_SNAPSHOT=output/dual_arm/phase1_20261111/commit_snapshot \
  .conda/bin/python -m unittest discover -s tests -p test_phase2_pipeline.py -v
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  ADA_DUAL_ARM_SNAPSHOT=output/dual_arm/phase1_20261111/commit_snapshot \
  .conda/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .conda/bin/python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
```

The available existing Ruff binary was `/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff`.
No tool/dependency installation was necessary. Final full suite: **310 tests, 8 skipped, no failures**,
including **18 new Phase 2 tests** and all **11 Phase 1 tests**. Headless depth tests use a fresh EGL
subprocess when earlier suite imports have already chosen GLFW. Lint/compile and new-file formatting
pass; the whole-tree format check still reports the same eight unchanged preexisting files.

A fresh Shadow quick run used `PYTHON_BIN=$PWD/.conda/bin/python`,
`ADA_GRASP_CTRL_RUN_ID=phase2_implementation_20261111`, and
`ADA_GRASP_CTRL_EXAMPLE_BASE=$PWD/output/phase2/legacy_example` with
`bash script/run_example.sh shadow quick`. `validate_quick_results.py verify` was then run against
`examples/quick_expected_status.json`. It **fails** with exactly the same three per-record differences
recorded in Phase 1: 68 success, 4 failure, 21 invalid initialization, 7 degraded, 0 execution errors.
`output/phase2/validation/shadow_comparison.json` records `same_as_phase1: true`. No baseline was changed;
this is not a passing release gate. The full three-hand release gate was not run.

Outstanding acceptance: a feasible real-checkpoint candidate completing physical execution, successful
lift/retention evidence, and real-checkpoint depth-mode physical evaluation. The uncalibrated source
mounts, conservative contact policy, convex collision approximations, provisional gains/feedforward,
assumed physics and complete-to-partial training shift remain material limitations. Joint-sampled and
20 ms contact checks are not continuous collision proof. There is no ROS/hardware execution or
two-hand cooperative planning. Existing right-hand regression tests pass and all preexisting tracked
code, assets, submodules, source hyperparameters and the user's specification were preserved.
