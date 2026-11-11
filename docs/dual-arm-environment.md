# Offline dual-arm environment (Phase 1)

This independent package loads both real FANUC CRX-5iA arms and both real Sharpa Wave hands,
provides Pinocchio FK / bounded single-arm IK, and runs MuJoCo without ROS. It adds no pipeline,
selector, grasp-inference, ROS execution, or hardware integration. The public maintained hand matrix
and existing right-side behavior are unchanged.

## Export and run

Use the existing ROS Python only to export. Choose a **new** absolute snapshot directory each time;
the exporter refuses an existing directory. No ROS nodes or drivers are launched.

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
/usr/bin/python3 script/export_bimanual_model.py \
  --output-root /home/hao-fang/hugs-workspace/ada-grasp-ctrl/output/dual_arm/my_run/snapshot \
  --mesh-python /home/hao-fang/hugs-workspace/ada-grasp-ctrl/.conda/bin/python

PYTHONPATH=src .conda/bin/python script/preview_dual_arm.py \
  --snapshot output/dual_arm/my_run/snapshot \
  --output-root output/dual_arm/my_run/validation
```

`PYTHONPATH=src` deliberately replaces inherited ROS Python 3.12 paths when using Ada's Python 3.10.
The export subprocess clears `PYTHONPATH` for the same reason. No dependencies are installed by these
commands. Existing MuJoCo 3.6.0, Pinocchio 3.0.0, NumPy 1.26.4, SciPy, trimesh 4.11.5 and pycollada
were used. Only export needs ROS package lookup and xacro; subsequent loading/stepping can run with
the ROS environment absent. Runtime needs the complete snapshot directory, not the original workspace.

Add `--viewer` for an optional local MuJoCo viewer; closing the window ends the preview. The default is
headless. CLI exit codes are 0 for passing validation, 1 for a failed validation, and 2 for preflight
failure. Both export and validation require fresh output directories.

For the reviewed browser preview (static, no physics stepping), run:

```bash
PYTHONPATH=src .conda/bin/python script/preview_dual_arm_web.py \
  --snapshot output/dual_arm/my_run/snapshot --port 8089
```

Open `http://localhost:8089`; Ctrl+C closes the server. The web viewer reads the copied
`initial_pose.yaml`: left J1–J6 `[0, 30, -30, 0, 60, 0]` degrees, right J1–J6
`[-90, -30, 210, 0, -60, 0]` degrees, with both hands at zero as in the upstream bimanual
default-pose command. Sliders use radians; reset restores those imported angles. For earlier
snapshots without that YAML, supply `--initial-pose /absolute/path/to/initial_pose.yaml` explicitly.
The numerical model's `home_configuration()` remains the zero-angle inspection pose.

The browser displays the original DAE material parts (white, green, gray/black and red details)
from the snapshot and updates them with MuJoCo link poses. It hides the corresponding colorless STL
visual duplicates in its private display model. Collision geometry remains unchanged. This restores
the source colors in the web preview; the numerical STL snapshot and native viewer retain the
conversion limitations described below.

Snapshot contents: unchanged `source.urdf`, path-resolved `resolved.urdf`, portable `model.urdf`, copied
meshes, `initial_pose.yaml`, `sources/`, available upstream license files, `mesh_conversion.json`, and `manifest.json` with
source paths, source checksums, commits, dirty states and canonical names. Both numerical engines load
the same `model.urdf`. Runtime never edits the snapshot. Treat a completed snapshot as immutable;
generate another export to update it. There is no mandatory signing or hash-gate framework.

## Python API and joint/frame contract

```python
from ada_grasp_ctrl.robot_environment import DualArmModel, DualArmSimulation

robot = DualArmModel.load("output/dual_arm/my_run/snapshot")
sim = DualArmSimulation(robot)
q = robot.home_configuration()  # zero angles; not a calibrated physical home
target = robot.fk(q, side="left", frame="hand_flange")
candidate, diagnostic = robot.solve_arm_ik(q, "left", target)
contacts = sim.check_static_contacts(candidate)
if diagnostic["accepted"]:
    sim.select_side("left")
    sim.set_target(candidate[:28])
    sim.step(10)
```

Contact results are diagnostic, including intentional robot/object contact; this example does not
constitute a collision acceptance policy. IK acceptance checks solver success, finite bounds and
residuals (1 mm / 0.01 rad, matching the existing right-side preparation convention). IK changes only
six active arm joints and holds the other 50 exactly. An unreachable or exhausted solve returns a
bounded candidate with `accepted=False`; malformed/nonfinite solver output raises. No output clipping
or automatic application occurs. This local solver does not enumerate IK branches.

External vectors are radians in this explicit order:

| Indices | Joints |
| --- | --- |
| 0–5 | `left_J1` … `left_J6` |
| 6–27 | 22 left Sharpa joints |
| 28–33 | `right_J1` … `right_J6` |
| 34–55 | 22 right Sharpa joints |

Hand suffix order (each prefixed with `left_` or `right_`) is:

```text
thumb_CMC_FE thumb_CMC_AA thumb_MCP_FE thumb_MCP_AA thumb_IP
index_MCP_FE index_MCP_AA index_PIP index_DIP
middle_MCP_FE middle_MCP_AA middle_PIP middle_DIP
ring_MCP_FE ring_MCP_AA ring_PIP ring_DIP
pinky_CMC pinky_MCP_FE pinky_MCP_AA pinky_PIP pinky_DIP
```

`robot.joint_names` and `sim.mapping` expose the complete name order and explicit MuJoCo qpos/qvel/
actuator and Pinocchio q/v indices. Native engine ordering is never assumed. A free object can make
`nq=63`, while the robot subset remains 56. Supply a copy of `robot.spec` with an explicitly positioned
object and its root name through `DualArmSimulation(robot, scene_spec=spec, object_bodies=("object",))`.
Object descendants inherit contact classification. No default object is invented.

For each side the chain is:

```text
world -> {side}_base_link -> J1_link -> ... -> J6_link -> {side}_flange
  -> {side}_ee_mount -> {side}_hand_flange -> {side}_hand_wrist -> {side}_hand_C_MC
```

Aliases are explicit: `crx_flange` is `{side}_flange`, `ee_mount` is `{side}_ee_mount`,
`fanuc_flange` is the separately rotated `{side}_fanuc_flange`, `hand_flange` is the hand root,
`palm` is `{side}_hand_C_MC`, and `tcp` is `{side}_tcp` (a separate source branch with a 35 mm offset).
`fk` also accepts exact side-qualified frame names. IK accepts `hand_flange`, `crx_flange`, or `tcp`.
There is no silent flange/TCP substitution and no additional mirroring of left-hand assets.

`set_target` takes 28 values and preserves the other side's command. The passive side can move under
gravity/contact. `check_static_contacts` reuses a separate `MjData`, copies current object poses,
calls `mj_forward`, and leaves live state, time and targets unchanged. Categories are `left_right`,
`robot_table`, `within_robot`, `robot_object`, and `robot_environment`; rows include signed distance,
penetration, body/geom names, sides and direct source-link adjacency. It does not step or reject contacts.

## Source provenance and conversion limits

Recorded on 2026-11-11; Ada started clean on `main` at `c5d9f84d4611ab090ec05d92a48f79166adcf2ff`.
The installed package shares resolve the following source files:

| Source repository | Commit | Inputs under `/home/hao-fang/ws_fanuc/src/` |
| --- | --- | --- |
| `bimanual_manipulation` | `cb2734c60619d43e9109a2a6a3d1c646874ede6f` | `bimanual_manipulation/bimanual_manipulation/composite_description.py`, `bimanual_manipulation/config/hand_mounts.yaml` |
| `dual_crx_control` | `c0c25931b62f408eb0ed238d40ef305efd16387c` | `dual_crx_control/urdf/dual_crx.urdf.xacro`, `dual_crx_control/config/initial_pose.yaml` |
| `dual_sharpa_wave_ros2` | `85d02e0275e8bfb55a9402f21dba08cec7b3ab97` | `dual_sharpa_wave_ros2/urdf/{left,right}_sharpa_wave_with_flange.urdf` |
| `fanuc_description` | `fb40c9803a826ba68c7c8e28ba904a25efa7fcd2` | `fanuc_description/fanuc_crx_description/urdf/crx5ia_urdf_macro.xacro` |

FANUC meshes come from `fanuc_description/fanuc_crx_description/meshes/crx5ia/{visual,collision}`.
Sharpa meshes come from `dual_sharpa_wave_ros2/third_party/sharpa_models/{left,right}/meshes`.
Every exact resolved mesh path/checksum is in the snapshot manifest. The installed shares are under
`/home/hao-fang/ws_fanuc/install/{package}/share/{package}`; the hand package is `dual_sharpa_wave`.

Read-only GitHub HEAD checks matched the first three local commits above. Their worktrees contain
uncommitted changes outside these model inputs, recorded in the manifest and preserved. The FANUC
description worktree was clean; its live GitHub HEAD was not queried. Local installed model input files
resolve directly to the source files above. No external repository or dependency was changed.

Both bases and mounts are copied exactly from upstream. Left base XYZ/RPY is zero; right base XYZ is
`[0.03058936, -0.91156264, 0.00873476]`, RPY is `[0.00994493, -0.01029918, -1.54694055]`.
The hand mounts have zero translation, left RPY `[0, 1.570796, 0]`, right RPY
`[3.14159265359, -1.57079665359, 0]`. **Hand mounts are upstream mock preview placeholders and are
uncalibrated for physical motion.** No independent calibration claim is made for the bases or table.
The existing upstream table is a 0.30 × 0.30 × 0.04 m box at Z = -0.02 m; no measured table was fabricated.

DAE visuals are converted from actual source meshes into STL with their scene transforms (these DAE
files specify meters and Z-up). Textures/DAE material detail are not retained. Original DAE files are
also copied. STL geometry and source collision meshes retain their URDF scale/origin. MuJoCo uses
convex hulls for mesh contact, so concavities are approximated. `balanceinertia=true` follows the
existing import convention and can balance incompatible source inertias. Fixed bodies are retained
for frame diagnostics. Default MuJoCo parent/weld contact filtering remains; no explicit exclusions
or colliders were added/removed. In particular, the existing right pipeline's base/J1 exclusion is
not imported into this independent model.

Numerical actuation reuses existing conventions: arm kp/kv = 100/10, hand kp/kv = 7.5/0, with source
effort bounds. Hand damping 0.05 and armature 1e-5 follow
`HUGS-DexGraspBench/assets/hand/sharpa_wave/right.xml`; these are numerical settings, not identified
hardware parameters. Both hands use those settings with their own actual geometry. Without these
existing joint dynamics, the raw hand URDF's tiny inertias caused large short-step oscillations.
The new scene uses a 2 ms timestep and `implicitfast`. Gains are provisional; sustained control and
grasp performance are not validated. MuJoCo soft limits can be slightly exceeded during physics.

The zero pose has natural palm/thumb contacts (~2.17 mm penetration) and tiny base/J1 touching
contacts. These remain visible in diagnostics. Static checking proves nothing about intervening
trajectory collisions. No hardware-ready, trajectory-clear, or successful left-grasp claim is made.

## Validation artifacts

The final local run is under `output/dual_arm/phase1_20261111/`: use `snapshot_complete/` and
`validation_complete/report.json`. Earlier investigation snapshots/reports remain separate.
The snapshot includes the Sharpa model's original `LICENSE.txt` and `NOTICE.txt`; copying these
records makes no new redistribution claim. The focused tests use real exported assets; without the explicit snapshot variable,
integration tests are reported as skipped rather than substituting dummy geometry:

```bash
PYTHONPATH=src ADA_DUAL_ARM_SNAPSHOT="$PWD/output/dual_arm/my_run/snapshot" \
  .conda/bin/python -m unittest discover -s tests -p test_dual_arm_environment.py -v
PYTHONPATH=src .conda/bin/python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
```

The tests exercise every retained robot body at zero, 56 single-joint perturbations and 10 deterministic
random configurations. The CLI checks 22 representative frames at 77 configurations. Both use the
existing 1e-6 FK regression tolerance and log actual position/rotation errors. Tests also cover all
three IK frame aliases on both sides, malformed/rejected solutions, inactive targets, actual crafted
inter-arm/table/self contacts, a caller-specified free object, portable snapshot relocation, and a
subprocess that forbids ROS imports while loading and stepping.

Measured results from the final run (one local machine, not a hardware-independent performance claim):

| Check | Result |
| --- | --- |
| Robot joints / actuators | 56 / 56; 64 copied source mesh references |
| CLI FK maximum position / rotation error | 6.83e-16 m / 1.30e-15 rad |
| All-body test FK maximum position / rotation error | 7.41e-16 m / 1.39e-15 rad |
| Left hand-flange IK position / rotation error | 1.53e-15 m / 2.59e-15 rad |
| Right hand-flange IK position / rotation error | 1.47e-15 m / 3.24e-15 rad |
| 100 / 1,000 static checks | 0.0146 / 0.1454 seconds, excluding model compilation |
| Headless physics | 10 steps / 20 ms; finite state, no MuJoCo warnings |
| Physics maximum displacement / soft-limit overshoot | 0.0230 / 0.000542 rad |
| New integration/unit tests | 11 passed, 0 skipped |
| Existing CRX/Sharpa, selector, mirroring tests | 56 passed, 0 skipped |
| Whole-repository Ruff lint / compile checks | Passed |
| New-file format check | Passed (8 Python files, including the browser preview) |
| Whole-repository format check | Failed on 8 unchanged pre-existing files |

These FK errors measure two engines consuming the same URDF, not physical calibration accuracy.
Crafted contacts include actual inter-arm penetration (~8.43 mm) and arm/table penetration
(~112 mm) found with seed 42, plus self contacts. Exact poses and contacts are in the JSON report.
The free-object test separately confirms `nq=63` and intentional robot/object classification.
The browser preview was additionally checked over its actual WebSocket: 34 original colored mesh
parts across 14 arm bodies and 56 joint sliders were delivered, with upstream initial arm angles.
That check used a temporary local port and stopped the server afterward.

An additional existing Shadow quick run completed 100 records with no execution errors:
68 success, 4 failure, 21 invalid initialization, 7 solver degraded. Its exit code was 1, and
`validate_quick_results.py verify` **failed**: three classifications differ from
`examples/quick_expected_status.json` (expected totals: 66/4/21/9). The differences are recorded in
`output/dual_arm/phase1_20261111/quick_classification_diff.json`; root cause remains unestablished.
The new package is not imported by that existing pipeline, and `git diff --name-status` remains empty
for tracked files. No baseline was changed. This result is not reported as a passing release gate.
The full three-hand release gate, full unrelated unit suite, interactive viewer, and hardware checks
were not run. Existing tests were selected using `test_crx_sharpa*.py`, `test_grasp_selection_crx.py`,
`test_sharpa_mirror.py`, and `test_sharpa_regression.py`.

The smallest next step is read-only alignment of the exported frame chains against observed ROS TF
and measured mounts. That alignment and left-hand grasp integration remain outside this phase.
