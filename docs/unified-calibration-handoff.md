# Unified calibration integration — 2026-11-22

Implemented in the original Ada checkout based on `cf76617753ea2fdbd356c4e26838c992c874b309`.
Initial integration validation was offline or mocked. The later operator-requested table recalibration
used passive camera capture on domain 87, as described below. No robot motion commands, external
repository edits, commit or push were performed. ROS contract tests used domain 88 with localhost
discovery; the user's normal domain 87 configuration is unchanged.

## One calibration source

The user-supplied `config/calibration/dual_system_calibration_20261122.json` was moved to:

`src/ada_grasp_ctrl/config/calibration/dual_system_calibration.json`

Its bytes were preserved: SHA-256 `479b031ec8212166e707334b987a646d8470fe1d76d2430dedf05919b600deca`.
`ada_grasp_ctrl.calibration.DEFAULT_CALIBRATION` is the single default path. The inherited pipeline
setting `extrinsics: null` selects it; an explicit path selects another input. Dual Live requires a
complete master JSON because camera and robot geometry use the same input.

Only these geometry keys are consumed:

| JSON key under `authoritative` | Consumer |
| --- | --- |
| `dual_crx.T_leftbase_rightbase.xyz_m` and `.rpy_rad` | Right CRX base fixed origin |
| `camera.T_leftbase_color_optical.matrix_4x4` | Color-optical points to left-base world |
| `hand_mounts.left.xyz_m` and `.rpy_rad` | Left Sharpa mount fixed origin |
| `hand_mounts.right.xyz_m` and `.rpy_rad` | Right Sharpa mount fixed origin |

The loader also reads `conventions` for world/units validation and `calibration_id` for provenance.
It ignores `derived`, `non_authoritative_measurements`, and redundant base/mount matrices. XYZ is in
metres; RPY is URDF fixed-axis `Rz(yaw) @ Ry(pitch) @ Rx(roll)`.

`pipeline.runner.perceive` supplies the canonical default to capture. The existing `load_extrinsics`
recognizes the master JSON and preserves its path/hash and unverified status in capture metadata.
Legacy `T_world_cam` inputs remain readable for historical replay/tests; the 11/07 camera file is
no longer the active default. Old recorded captures, gallery references, table files and acceptance
records retain their historical references without migration.

The master calibration's intrinsics were measured at **1280x720**. The live driver remains
**640x480 at 30 FPS** and projection still uses live `CameraInfo`; the stored 1280x720 K is not substituted.

## Ada model integration

The installed source snapshot is `output/dual_arm/phase1_20261111/commit_snapshot`.
Inspection resolved the following exact mappings; no mount-frame ambiguity remains for this snapshot:

| Fixed joint | Parent → child | Applied value |
| --- | --- | --- |
| `left_base_joint` | `world` → `left_base_link` | Existing identity retained |
| `right_base_joint` | `world` → `right_base_link` | Authoritative base XYZ/RPY |
| `left_arm_to_hand` | `left_ee_mount` → `left_hand_flange` | Authoritative left mount XYZ/RPY |
| `right_arm_to_hand` | `right_ee_mount` → `right_hand_flange` | Authoritative right mount XYZ/RPY |

The source has identity `*_flange` → `*_ee_mount` joints. `*_fanuc_flange` is a separate rotated
branch and is not the mount parent. Unexpected mappings raise an input error rather than being guessed.

Before creating pipeline processes, `pipeline.dual_runner.main` creates `<run>/robot_snapshot` and
records it in `effective_config.yaml` and `run_manifest.yaml`. The helper copies source assets and
replaces only those three fixed origins in the new URDF. Its manifest records calibration provenance
and source snapshot/model/manifest hashes. Existing outputs are never overwritten. Applying the same
calibration to an already-calibrated URDF produces identical model bytes, with no accumulated offsets.

The effective snapshot feeds the existing `live_scene`, `DualArmModel`, collision geometry, viewer
and packet provenance. FK and IK therefore use the same geometry as MuJoCo and preview. The historical
source snapshot, meshes, licenses and recorded runs are preserved. Legacy/hierarchical selection,
standalone right Linear/MPC control and the ROS execution contract are unchanged.

## Table recalibration applied after operator review

The original `output/fixed_camera/table_recalibration_20261110_213933/fit/table_plane.json`
remains `preview_unaccepted` and bound to the old 11/07 extrinsics. That old plane correctly raises
**`Object capture and table must share camera and calibration`** with new captures, including when
`allow_unaccepted=True`. The operator subsequently requested recalibration after adjusting only
the table, confirmed an empty surface, and explicitly requested applying the resulting new table.

The active `table_plane` now points to:

`output/fixed_camera/table_recalibration_20261122T231227_977817Z/fit/table_plane.json`

Three independent 640x480 RGB/depth captures were taken with the existing camera-only launch on domain 87;
that camera process was stopped cleanly afterward. Main-fit RMS is 2.42 mm, inlier ratio 96.89%, and
tilt 0.47 degrees. Both distributed repeat fits passed the unchanged quality/consistency thresholds.
The calibration directory contains `table_review.png`, `calibration_review.json` and
`application_verification.json` with the exact geometry, capture metadata and checks.

The active right pipeline now uses
`src/ada_grasp_ctrl/config/experiments/fixed_camera_acceptance_2026-11-22.json`.
It records the operator's table-only confirmation and explicit request to apply this result,
with SHA-256 `1873ee8424a67ed67f4c95759bbcc303c81cc8620bca7b4715e3e551982c07cd` for the new plane
and the unchanged master JSON hash. No independent world-reference measurements were supplied or
claimed: the saved fit remains `preview_unaccepted`, and the existing operator-attestation path
provides automatic-mode acceptance without editing its geometry/status.

Both manual and automatic extraction passed the existing calibration checks using the real recorded
repeat capture, then correctly reported no object on the empty table. The calibration mismatch is
resolved for these configured paths; this does not claim a full live grasp/execution test. The old
plane, old operator record, camera extrinsics, base placement and hand mounts were preserved.

For future table changes, use the existing
[capture, fit and acceptance workflow](fixed-camera-table.md#capture-and-offline-table-fitting):

1. In the user's camera environment, clear the table and capture using
   `--extrinsics src/ada_grasp_ctrl/config/calibration/dual_system_calibration.json` into a fresh output directory.
2. Fit the current empty-table ROI; inspect diagnostics/overlays, repeat regions and independently measured
   world references using the existing review procedure. Do not repurpose old evidence or hashes.
3. Save the reviewed/accepted result in a new directory and update the pipeline's `table_plane` to it.
   If using the automatic path, supply the corresponding reviewed `calibration_acceptance` record, or
   use its existing accepted-plane path without the obsolete operator record.

## Validation

Evidence is retained under `output/calibration/unified_20261122_v1/`, including the derived and
reapplied snapshots, `run_report.json`, `shadow_classification_comparison.json`, and `test_logs/`.

| Check | Result |
| --- | --- |
| Focused calibration, camera, legacy/hierarchical and standalone pipeline tests | 50 run: 49 passed, 1 OpenCV skip |
| Full Ada unittest suite | 577 run: 491 passed, 86 skipped, no failures/errors |
| System OpenCV and sourced ROS receiver contracts | 30 run: 26 passed, 4 dynamic-lift fixture skips |
| Original recorded dual snapshot regression | 11 passed |
| Calibrated recorded model audit | Only 3 origins changed; source preserved; repeat application identical; both IK solves accepted |
| Ruff lint, compileall and `git diff --check` | Passed |
| Changed Python file formatting | All 7 passed; whole-tree check retains 8 pre-existing format differences |
| Maintained quick release gate | Failed on the existing 3 Shadow classification differences; Allegro/LEAP not reached |

The calibrated recorded model audit used zero plus four seeded configurations. Maximum all-body
Pinocchio/MuJoCo disagreement was `6.96e-16` m and `1.07e-15` rad. Fixture tests also vary the JSON
base and both mount placements and check FK, collision geometry, IK, camera transforms, live intrinsics,
idempotence and unchanged table rejection. These establish software geometry, not physical validation.

Shadow produced 68 successes, 4 failures, 21 invalid initializations, 7 degraded solves and zero
execution errors. The expected inventory has 66 successes and 9 degraded solves. All 100 classifications
match the retained pre-change `output/examples/shadow/environment-20261101-quick` run exactly; the same
three differences are documented in [the local environment report](local-environment-2026-11-01.md).
The historical drift remains unresolved. No baseline or comparison tolerance was changed, and this is
not a passing release gate.

Commands run from the checkout, with CPU threads limited to one and CUDA disabled:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export MPLCONFIGDIR=/tmp/ada_calibration_mpl
PYTHONPATH=src:tests nice -n 10 .conda/bin/python -m unittest \
  test_unified_calibration test_fixed_camera test_dual_live_pipeline \
  test_dual_hierarchical_pipeline test_grasp_pipeline -v
PYTHONPATH=src nice -n 10 .conda/bin/python -m unittest discover -s tests -v
PYTHONPATH=src:tests ADA_DUAL_ARM_SNAPSHOT="$PWD/output/dual_arm/phase1_20261111/commit_snapshot" \
  nice -n 10 .conda/bin/python -m unittest test_dual_arm_environment -v
PYTHONPATH=src PYTHON_BIN=.conda/bin/python \
  ADA_GRASP_CTRL_RELEASE_GATE_ROOT="$PWD/output/calibration/release_20261122_v1" \
  nice -n 10 bash script/run_release_gate.sh quick
.conda/bin/python -m compileall -q src tests script
ruff check src tests script
ruff format --check src tests script
git diff --check
```

The recorded quick-gate root already exists; choose a new empty root for any future invocation.
Ruff was provided by `/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff`.
In a separate system-Python shell, source `/opt/ros/jazzy/setup.bash` and
`/home/hao-fang/ws_fanuc/install/setup.bash`, then run the existing contract tests:

```bash
ROS_DOMAIN_ID=88 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}" \
  nice -n 10 /usr/bin/python3 -m unittest test_fixed_camera test_right_ros_receiver \
  test_dual_live_receiver test_mpc_p3_receiver test_dynamic_lift_ros -v
```

Changed files: `calibration.py`, the renamed canonical JSON, `config/experiments/grasp_pipeline.yaml`,
`experiments/fixed_camera_geometry.py`, `pipeline/{runner,dual_runner}.py`,
`tests/{test_unified_calibration,test_fixed_camera,dual_selector_fixtures}.py`, this handoff,
and the linked dual-live and fixed-camera guides.
