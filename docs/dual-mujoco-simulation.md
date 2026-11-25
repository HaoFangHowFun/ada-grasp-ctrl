# Dual grasp pipeline in MuJoCo

The existing dual pipeline now supports `--environment simulation` without ROS,
robot drivers, camera capture, or a MuJoCo desktop window. The same 8089 Viser
server owns scene editing, trajectory review, measured physics display and Reset.
Hardware and ROS mock retain their existing entry points and eight-stage behavior.

## Start

Run from the Ada checkout with its existing `.conda` environment and local assets:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
env -u LD_LIBRARY_PATH python3 script/run_dual_grasp_pipeline.py \
  --environment simulation --selector hierarchical --arms auto \
  --grasp-model native_multimode \
  --output-root "output/simulation/live_$(date +%Y%m%d_%H%M%S)"
```

Open <http://localhost:8089>. The launcher selects the maintained Ada Python and
isolates inherited ROS Python paths. Do not source ROS for this workflow. For a
remote machine, forward port 8089 to the browser host. An occupied port fails;
the application does not leave a server running on an alternate port.

For paired-only diagnostics, add `--grasp-mode both_only` to that command. This
requires both enabled arms and the native model. `--arms auto` permits both arms;
the selected source mode determines which arms actually move. Use `--arms left`
or `--arms right` to restrict eligibility. Native inference requires the
hierarchical selector. The existing `--grasp-model right_full` supports both
`--selector legacy` and `--selector hierarchical`.

Placement remains a separate development invocation:

```bash
env -u LD_LIBRARY_PATH python3 script/run_dual_grasp_pipeline.py \
  --environment simulation --placement-only \
  --output-root "output/simulation/placement_$(date +%Y%m%d_%H%M%S)"
```

Every command needs a fresh output root. `--preview-only` and `--export-only`
both retain the interactive scene builder and preview server in simulation;
they save the reference and disable Execute Grasp. They do not execute, nor do
they automatically exit after exporting. Simulation rejects `--ros-domain-id`,
`--roi`, `--cluster`, and captured `--object-root`. UOP path options require
`--placement-only`.

## Interaction

1. Choose one of the ten local meshes. Edit world X/Y in metres, Yaw in degrees
   around the measured table normal, and uniform Scale within the object's manifest
   range. Each object retains its fixed support orientation; Roll and Pitch are
   not editor controls. Switching objects restores the selected object's default scale.
   Z comes from the full rotated mesh support point on the effective table plane,
   plus 1 mm clearance. XY is clamped to keep the complete mesh 5 mm inside the
   table edges, and the sliders display the resulting position.
   These edits only update the visual mesh. Randomize changes object, XY and Yaw
   while preserving its fixed support and using its default size. It uses the displayed Seed;
   the same seed gives the same requested scene. Reset edit controls restores
   the selected object's default XY, scale and zero Yaw. Scaling recomputes Z and
   the allowed XY footprint immediately; edited sizes still require physical validation.
2. Confirm Scene compiles the physical scene and settles it under gravity. A
   valid scene produces exactly 8,192 deterministic area-weighted complete
   surface points, transformed by the actual settled mesh pose. Invalid table
   footprints, unstable settling, excessive penetration or invalid robot state
   produce a recoverable error. Reset Scene returns to editing.
   Settling now tolerates contact jitter: over a full 0.5 s window the position
   span must be at most 3 mm and orientation span at most 2 degrees, with table
   support throughout. It waits up to 10 s of simulation time. Unlike the earlier
   mixed linear/angular velocity test, this measures bounded pose motion.
   `settling_report.json` saves the actual spans and separate velocity diagnostics
   on success, timeout or cancellation. Timeout messages include the observed spans.
   A custom pipeline YAML can override `simulation.settling` fields `timeout_s`,
   `window_s`, `position_span_m`, and `orientation_span_deg`. These initialize the
   scene; they do not change grasp success or physical contact parameters.
3. Normal mode runs the existing model, then applies the simulation squeeze
   multiplier before IK and candidate selection: `grasp + 2 * (squeeze - grasp)`
   for each active wrist's world translation and 22 finger angles. Squeeze wrist
   rotation and the pregrasp/grasp targets are preserved. Raw `candidates.json`
   remains unchanged; `simulation_candidates.json` records adjusted targets and
   provenance. Joint limits and feasibility checks still apply, without clipping.
   The selector and planner consume these adjusted candidates.
   Review the selected mode, active arms, candidate, effort and candidate counts.
   The Phase dropdown and Playback slider show the same six-stage linear
   reference that execution will use. Scrubbing never advances physical state.
4. Execute Grasp verifies source, reference and settled-state identity, then
   starts actual position actuators and `mj_step`. The view changes to measured
   joints and the actual object pose. The sequence is
   `above_object → pregrasp → grasp → squeeze → lift → hold`.
   New plans use timed progression: each scheduled interpolation duration plus
   its fixed 0.3 s endpoint dwell, regardless of joint arrival. There is no
   additional arrival wait or arrival timeout. This preserves reviewed durations;
   it does not shorten every motion to Bench's one-second stages. Tracking errors
   and arrival diagnostics remain in the physical log.
5. Following scheduled Hold, physics continues with final joint targets until
   Reset or exit. Review scheduled hold success separately from current retention
   and later drop. No automatic release or return runs in simulation.
6. Reset Scene works during preparation, preview, execution, hold and errors.
   It invalidates old callbacks, cancels owned inference processes, joins the
   physics owner, clears derived state and restores initial robots. It preserves
   the last requested object edits and every prior attempt directory. Confirm
   again always performs fresh inference. Attempt IDs may skip numbers because
   cancellation also advances the generation counter.

In placement-only mode, Confirm Scene opens the reused placement panel with P00
Original Pose selected. Run UOP-Net once invokes the existing model once for that
scene. Switching candidates rotates the original cloud about its observed
centroid for display, then shifts world Z so the lowest rotated cloud point lies
on the calibrated tabletop plane, including tilted tables. XY stays unchanged;
P00 keeps the original observed pose. Switching always starts from the immutable
cloud, so rotations and height shifts do not accumulate. This is a geometric
point-cloud preview, not a stability or finite-table/collision check.
Confirm Placement saves the relative rotation and the display-only
`preview_translation_world_m`; execution translation remains
`translation_status: undetermined`. It starts neither grasp inference nor motion.
P00 remains available if UOP fails. Reset clears the panel and returns to the same
scene builder and server; an obsolete UOP worker may finish writing only to its
old attempt, and cannot update the replacement panel.

## Local prerequisites and explicit assumptions

Paths are in `src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml` and
`simulation_objects.yaml`. No dependencies or datasets are downloaded.

- Robot: `output/dual_arm/phase1_20261111/commit_snapshot`, copied with the current
  authoritative `config/calibration/dual_system_calibration.json` base and mounts.
  Initial arms use the snapshot's `initial_pose.yaml`; fingers start at zero.
  A custom pipeline YAML may supply `simulation.initial_robot_state` with exactly
  the 56 canonical named joint values in radians.
- Table top: the dual pipeline's configured latest plane. Its current source file
  still says `preview_unaccepted`; simulation preserves that status. The original
  30 cm square nominal table is disabled. The finite tabletop is an explicitly
  **unmeasured footprint assumption**: 1.2 × 1.4 m, centre XY `[0.35, -0.4]`,
  thickness 0.04 m. Override the complete `simulation.table` mapping in a custom
  pipeline YAML if measured boundaries become available. XY slider bounds derive
  from this geometry; all object vertices must fit the footprint.
- Objects: existing local `examples/assets/object/DGN_2k/processed_data` meshes and
  source convex pieces. The manifest records actual IDs, visual hashes, nominal
  dimensions, scale ranges and collision paths. Jar/container replace the
  suggested storage-container/pitcher categories. Mug is the default stable
  development scene. All ten fixed support orientations passed actual settling
  and the separate 2 mm penetration check at four cardinal yaw angles. Book's
  old support face produced about 14 mm of real collision penetration; its new
  preset uses another support face without changing mass, friction or contact
  parameters. Presets are table-relative WXYZ quaternions in the manifest.
  Final validation covered 30 scenes (ten objects × three XY/yaw combinations),
  each with five additional seconds of physics after settling. Container retains
  slow drift of several millimetres over five seconds with its source convex hull;
  supported initialization does not guarantee a motionless free body indefinitely.
  Legacy explicit RPY/scale requests remain readable by the scene builder;
  arbitrary legacy orientations and custom physical scenes still require validation.
- Mass: existing exploratory `config/simulation_scene.yaml` assumptions:
  100 g, fixed mass across scales, equal mass per
  convex collision piece with uniform-density inertia. These are not identified
  object properties. Actual compiled mass/inertia are saved per attempt.
- Contacts: `simulation.contact_profile: dexgraspbench` applies friction
  `[0.8, 0.02, 0.0001]`, `condim=4` (sliding plus torsional friction), elliptic
  cone, `impratio=10`, and `noslip_iterations=2`. Only robot geoms receive
  `solref=[0.005, 1]` and the first three `solimp` values `[0.5, 0.99, 0.0001]`;
  object/table retain their source stiffness. This follows DexGraspBench's
  `MjHO._add_hand` and `_set_friction`, unlike historical `sharpa_native_v3`,
  which hardens every geom. The profile and effective object friction are saved
  in scene configuration and physical evidence.
- Gravity: `simulation.gravity_mode: object_only` matches the historical successful
  single-arm benchmark: MuJoCo global gravity is disabled, and every physics step
  applies a world-Z force of `-10 * compiled_object_mass_kg` N at the object's COM.
  At the default 100 g mass this is 1 N downward. Arm bias compensation is disabled.
  The same force policy applies during settling, execution and continued Hold,
  and is recorded with the external forces in initial/settled snapshots and reports.
  Set `world` to restore global gravity with bounded arm bias compensation.
  Existing standalone ObjectScene configs without the field keep `world` behavior.
  Default open-hand initialization in object-only mode starts at least 0.0001 rad
  inside finger limits to avoid tiny soft-limit undershoot at exactly zero. Explicit
  initial states, physical limits and generated grasp targets are not altered.
- Dynamics: existing canonical position actuators with implicitfast integration.
  Timestep is 0.004 s; `simulation.reference_hz: 25` updates the shared linear
  reference every 40 ms, held for ten physics substeps, matching Bench sampling.
  The reviewed plan records the rate; old plans without it retain 100 Hz.
  Standalone ObjectScene configs default to the unchanged `source` profile.
  To restore previous interactive numerics, set `contact_profile: source` and
  `reference_hz: 100` together.
  All active side targets are set before the common step. Rendering is 30 Hz;
  UI metrics update at 10 Hz. Interactive stepping is paced near real time.
  CRX paths, stage durations, squeeze multiplier, gains, fixed mass policy and
  success criteria remain as before. This matches Bench contact/control sampling,
  not its dummy-arm evaluator or density-based object masses; its success rate
  does not transfer to this setup.
- Progression: `simulation.stage_completion: timed` is the default. The existing
  0.01 rad arm arrival threshold is diagnostic in this mode. Set `arrival` to
  restore the older arrival/settle/timeout behavior; older saved plans without
  this field retain that behavior. The reviewed plan records the resolved policy.
  `simulation.squeeze_extrapolate_ratio: 2.0` controls only simulation candidate
  preparation; set 1.0 for the original targets. Both settings are in the dual
  pipeline YAML. Hardware execution policies and model gains are unchanged.
  The existing 0.01 rad actual joint overshoot bound, collision checks, numerical
  checks and passive-side command preservation remain enforced.
- Grasp criterion: the prior Phase 2 0.03 m minimum lift and 0.02 m/s maximum
  speed, applied throughout the configured five-second scheduled Hold. Lift uses
  the actual mesh origin's world Z immediately before lift. Every active side
  must retain object contact at every hold sample; a paired grasp requires both.
  Completion alone is not success. Continued retention failure is distinct from
  a drop after previously retaining the object. `slip_from_peak_m` reports actual
  loss of lift height without declaring a new success threshold.
- Models: the existing native 200k and right-full 100k deployment paths and
  DexLearn `.venv` from the dual config. UOP uses `../uop-net/.venv/bin/python`
  and its existing `uop_net/partial_best.pth`; `--uop-root`/`--uop-python` can
  select existing local installations. UOP still samples 2,048 distinct points.

## Evidence and limits

Each attempt records requested edits, actual scene config, original/settled
physics states, compiled MJB, visual/collision provenance, cloud, canonical q56,
actual predictions, selection diagnostics, reference and checks. `physics.jsonl`
streams measured tracking, contacts, object pose and paired relative-wrist
reference/error. `execution_report.json` separates pipeline completion, scheduled
hold success and later retention. `cancellation.json` identifies resets and exit.
The session manifest and report identify the owning invocation and attempts.

Physics owns the full object collision solid. Inference sees the immutable cloud
and source hashes; selection and planning use a separate canonical `live_scene()`
with its unchanged robot/table collision policy and no object solid. Ground-truth
pose, inertia and collision meshes are kept in physics evidence. A simulation
reference is a distinct artifact, never a relabeled hardware/mock packet.

This implementation does not certify grasp success or physical hardware parity.
It adds no MPC, tactile control, camera simulation, placement-aware ranking,
placement translation, planning or execution. See
[implementation and validation evidence](dual-mujoco-simulation-handoff.md).
