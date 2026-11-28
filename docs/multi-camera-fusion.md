# Stationary D415 + two D405 cameras

Opt-in implementation extending `ada-grasp-ctrl` baseline `261b7ef` and the
`bimanual_manipulation` camera integration. Paired fixed/wrist calibration capture
is available separately in `bimanual_manipulation` commit `ac9a860`.

The cameras are stationary. Historical `wrist_left` / `wrist_right` ROS names
are retained; they do not imply wrist mounting or live arm FK. The wrist-camera
Trigger/Save mode in the calibration repository does not make this fusion route
support moving cameras: this route still requires stationary optical-to-base
extrinsics.

## What is implemented

- `--multi-camera` on the existing dual pipeline, hardware only. Simulation and
  `--placement-only` combinations fail explicitly. Omitting the flag retains the
  D415 path; no D405 subscription or extrinsic is required by that path.
- Concurrent RGB, rectified depth and matching CameraInfo subscriptions for all
  cameras; a fresh global timestamp window (default 100 ms), bounded timeout,
  live serial checks and per-camera diagnostics. No camera driver is started by
  the capture code. ROS `16UC1` is millimetres, `32FC1` is metres; each depth
  image uses its own intrinsics. Nonzero depth distortion is rejected.
- Optical-to-left-base calibration plus verified internal optical TF, applied
  exactly once. Ada's snapshot explicitly checks `world == left_base_link`;
  a ROS frame merely named `world` is not assumed equivalent.
- Concatenation, 3 mm voxel representatives, existing table-height filtering,
  existing D415 ROI/cluster selection and unique seed-supported clustering.
  A selected cluster can extend outside the seed ROI. Competing associations or
  connections to another visible D415 component fail, without a largest-cluster
  fallback. Touching objects or occluded robot links cannot be guaranteed
  separable; keep a clear stationary workspace and review the result.
- There were no existing numeric 3D workspace bounds. The default uses the
  existing **full D415 workspace field of view**, depth range and table-height
  limits, not the selected object ROI. `workspace_bounds_m` and
  `exclude_bounds_m` optionally add left-base-frame boxes. No new robot masking
  or automatic table refitting is claimed.
- The fused world/metre cloud enters the existing inference boundary. The
  single-view depth-grid mesh exporter is bypassed for this opt-in route.
  HUGS normalization, mirroring, ranking, IK and execution remain unchanged.
  `points_color_m` is the fused XYZ expressed in D415 color coordinates for
  the existing round-trip contract, not fabricated RGB for unseen surfaces.
- Existing 8089 viewer gains individual D415, D405-left, D405-right, seed and
  fused-object toggles. No second server or continuous reconstruction.

Changed ownership: `experiments/multi_camera{,_geometry,_ros,_calibration,_preview}.py`,
`config/experiments/multi_camera{,_driver}.yaml`, `pipeline/dual_runner.py`,
`pipeline/dual_viewer.py`, additive calibration compatibility in `calibration.py`
and `experiments/fixed_camera.py`, and `tests/test_multi_camera.py`.

## A. Launch and inspect streams

The already installed launch accepts this profile. Check existing nodes first;
never start a duplicate driver for a serial.

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=87
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ros2 node list --no-daemon --spin-time 3

# Only when these camera drivers are not already running:
ros2 launch bimanual_manipulation cameras.launch.py \
  enable_fix_camera:=true enable_wrist_camera:=true \
  camera_config:="$PWD/src/ada_grasp_ctrl/config/experiments/multi_camera_driver.yaml"
```

The override preserves D415's existing pointcloud publisher for compatibility;
D405 pointcloud publishers stay off. All three publish internal optical TF.

```bash
# Another sourced terminal, while the scene remains stationary:
/usr/bin/python3 -m ada_grasp_ctrl.experiments.multi_camera probe \
  --observe-seconds 10 \
  --output-root "output/multi_camera/probe_$(date +%Y%m%d_%H%M%S)"
```

For each role, topics are `/camera/<role>/color/image_raw`,
`/camera/<role>/depth/image_rect_raw`, and corresponding `color/camera_info` and
`depth/camera_info`. Frames are `<role>_color_optical_frame` and
`<role>_depth_optical_frame`. Probe needs no external calibration and never
sends a robot command.

## B. Acquire and validate calibration

The initial fusion implementation did not include measured D405 extrinsics.
The 2026-11-27 installation below adds both measured results with explicit pending-validation status.
The authoritative file remains `src/ada_grasp_ctrl/config/calibration/dual_system_calibration.json`.
The existing collector supports topic/frame overrides. Use the existing
ROS-Industrial static-camera hand-eye solver, detector, measured ChArUco board
and robot pose workflow. Do not substitute guessed board dimensions or a
flange-to-board transform. A hand mounting transform is not a board mounting
transform. Source the restored pinned ROS-Industrial 1.2.0 workspace before
collection:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/hugs-workspace/third_party_build/fixed_camera_calibration_ws/install/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
```

See `/home/hao-fang/ws_fanuc/src/bimanual_manipulation/docs/fixed_camera_calibration.md`
for single-camera collection or `docs/dual_camera_calibration.md` in that repository
for paired Trigger/Save. Use `mounting_mode:=static` for this stationary fusion
route; a wrist-to-camera result cannot be used directly as optical-to-base geometry.

For each camera, print a collection command with the actual measured board file:

```bash
PYTHONPATH=src .conda/bin/python -m ada_grasp_ctrl.experiments.multi_camera_calibration \
  collection-command --role wrist_left \
  --target-config /absolute/path/to/measured_charuco.yaml \
  --target-mount-frame left_ee_mount \
  --output output/multi_camera/calibration_left
```

Repeat with `wrist_right` and a different output directory. Execute the printed
command in the sourced calibration environment. It uses
`camera_mount_frame:=left_base_link`; confirm that frame and the actual board
mount frame exist in the robot TF tree. The launch starts collection/visualization
nodes, not a robot or camera driver. Move the robot/board only manually or via
an already approved workflow; gather diverse positions/orientations in each
D405's working range and reserve independent observations. Reuse the existing
multi-pose solver if board mounting is unknown.

The importer deliberately consumes a normalized, explicit export of the
existing solver rather than guessing its YAML transform direction. Two JSON
inputs are required:

| Candidate field | Required meaning |
| --- | --- |
| `role`, `serial` | `wrist_left` / `260522275388` or `wrist_right` / `260522273536` |
| `calibrated_optical_frame` | The role's actual color or depth optical frame |
| `reference_frame`, `unit` | `left_base_link` (or calibrated `right_base_link`), `m` |
| `transform_direction` | `optical_to_reference` |
| `converged` | Actual solver convergence, boolean `true` |
| `T_reference_optical` | Measured finite proper 4x4 rigid transform |
| `fit_observation_ids` | Distinct IDs used for fitting, at least three; more diverse samples recommended |
| `provenance` | `solver_result`, `board_config`, `mount_registration`: each has an existing `path` and its `sha256` |

`mount_registration` identifies the measured or jointly solved board mounting
and frame registration, including the existing right-to-left relation if used.

The independent evidence JSON contains `observations`, each with a distinct
`id`, `T_leftbase_board`, `T_optical_board`, and measured `reprojection_rms_px`;
no validation ID may occur in the fit set. At least three diverse held-out
observations are required. `overlap_distances_m` contains measured independent
geometric overlap distances (at least three, preferably a dense correspondence
set). The importer computes base-frame translation/rotation residuals and
reports sample counts, pixel RMS and overlap RMS. It uses
`T_leftbase_optical @ T_optical_board` versus `T_leftbase_board`.

```bash
# Validate into a new candidate file; does not modify the authoritative file.
PYTHONPATH=src .conda/bin/python -m ada_grasp_ctrl.experiments.multi_camera_calibration \
  validate --role wrist_left \
  --candidate /absolute/path/to/left_solver_export.json \
  --evidence /absolute/path/to/left_independent_validation.json \
  --output output/multi_camera/calibration_left_validated.json

# After reviewing the measured evidence, explicitly save. Use a new output path.
PYTHONPATH=src .conda/bin/python -m ada_grasp_ctrl.experiments.multi_camera_calibration \
  validate --role wrist_left \
  --candidate /absolute/path/to/left_solver_export.json \
  --evidence /absolute/path/to/left_independent_validation.json \
  --output output/multi_camera/calibration_left_saved.json --save
```

Repeat for right. Defaults reject translation maxima above 10 mm, rotation
maxima above 2 degrees, reprojection maxima above 2 px or overlap RMS above
10 mm; explicit `--max-*` options are recorded in the result. These are practical
initial checks, not an accuracy guarantee. Verify physical side mapping and
cloud overlap visually. Moving a camera invalidates its extrinsic.

Schema extension: `authoritative.stationary_cameras.<role>` contains `serial`,
`mounting: stationary`, `calibrated_optical_frame`, `reference_frame: left_base_link`,
`transform_direction: optical_to_leftbase`, `unit: m`, `T_leftbase_optical`,
`status: validated`, `quality` and `provenance`. There are no identity placeholders.
All existing fields remain numerically unchanged. Each output retains a
`.previous` backup and hash-bound lineage. **Keep these backups:** the D415/table
hash compatibility path accepts an older file only after verifying that every
legacy field is identical and only stationary-camera entries differ. Changed
D415/base/mount data still invalidates the old table/acceptance record.

## C–D. Observe, infer and review

After valid calibration and passive robot feedback are available:

```bash
# In the sourced ROS terminal; perception only, no robot feedback needed:
/usr/bin/python3 -m ada_grasp_ctrl.experiments.multi_camera observe \
  --plane output/fixed_camera/table_recalibration_20261123T215945_898848Z/fit/table_plane.json \
  --output-root "output/multi_camera/observation_$(date +%Y%m%d_%H%M%S)"

# Existing full pipeline with passive robot-state snapshots, inference and export:
.conda/bin/python script/run_dual_grasp_pipeline.py \
  --environment hardware --ros-domain-id 87 \
  --selector hierarchical --arms auto --grasp-model native_multimode \
  --multi-camera --export-only \
  --output-root "output/dual_live/multi_camera_$(date +%Y%m%d_%H%M%S)"
```

Use `--roi x0 y0 x1 y1` when no desktop display is available. If multiple D415
clusters are detected, inspect the saved overlays and use `--cluster ID`; fused
association errors require correcting the scene/settings, not a silent fallback.
`--multi-camera-config PATH` changes capture/fusion limits.

As before, `--export-only` produces artifacts without starting the viewer.
After a successful export, inspect it without execution:

```bash
PYTHONPATH=src .conda/bin/python -m ada_grasp_ctrl.pipeline.dual_viewer \
  --run /absolute/path/to/the/exported/run
```

Use existing `http://localhost:8089`. A previously running preview must release
that port first. Alternatively `--preview-only` runs capture/inference/review
without execution. Real robot execution remains a separate user-approved step.
Test `--arms left`, `--arms right`, and `--arms auto --grasp-mode both_only` after
calibration; availability/feasibility can legitimately reject a mode.

Per-run `multi_camera/` contains a calibration snapshot, D415 `capture/snapshot.*`,
D405 arrays, `capture/cameras.json`, `capture/fused.npz`, `d415_seed/`, `object/`,
unchanged-geometry table copy and `run_report.json`. The outer pipeline freezes
one calibration before producing its robot snapshot. Reports separate capture,
transform/fusion, segmentation and total perception time; human ROI time is
reported separately and excluded from the total. Inference has its existing
separate timer. No cloud is normalized by the fusion module.

To revert: omit `--multi-camera` and `--multi-camera-config`. D405s need not be
present. If restarting camera drivers, use the original
`fixed_camera_driver.yaml` with only `enable_fix_camera:=true`.

## Validation evidence (2026-11-27)

Real hardware, no robot motion:

- Enumerated D415 `347322060270`, D405 `260522275388`, D405 `260522273536`, all
  firmware 5.17.0.10 and USB 3.2. Queried profiles support 640x480x30 RGB + depth.
- Ten-second concurrent capture: RGB received 29.46/29.69/29.59 Hz and depth
  29.66/29.79/29.99 Hz (fixed/left/right); selected global skew 43.88 ms.
  Finished probe CLI repeated successfully with 22.98 ms selected skew.
- Depth was `16UC1`, matching 640x480 depth CameraInfo with zero distortion.
  Backprojection compared with each official driver's same-timestamp cloud:
  maximum XYZ nearest-neighbor discrepancy 0.000229 mm across the three cameras.
  D405 pointcloud publishers were enabled only during this diagnostic and
  restored to disabled. This verifies depth geometry/units, **not external
  inter-camera alignment**.
- Driver logs contained startup/profile and hardware-clock warnings; no USB
  bandwidth timeout appeared during this observation. Ten seconds is not a
  sustained-load guarantee, and hardware clocks are not hardware-synchronized.
- The original D415-only command also completed against live hardware:
  `/usr/bin/python3 -m ada_grasp_ctrl.experiments.fixed_camera capture --extrinsics src/ada_grasp_ctrl/config/calibration/dual_system_calibration.json --output-root output/multi_camera/d415_regression_20261127`.
  Default camera config, driver profile and authoritative calibration remained unchanged.
- The actual multi-camera native export-only command exited 2 on missing D405
  calibration before any inference or robot-state process. The original
  authoritative JSON was not modified.

Evidence is local under `output/multi_camera/validation_20261127/`,
`output/multi_camera/cli_probe_20261127/` and
`output/multi_camera/missing_calibration_preflight_20261127/` (generated, not committed).

Synthetic/unit evidence: nontrivial rotated/translated optical composition,
invalid transforms/depth, row padding/endianness, stale/future/missing/global
skew rejection, voxel/workspace/table filtering, backside growth, neighboring
object veto, ambiguous/empty association, no premature normalization, independent
calibration validation/right-base conversion, save isolation, legacy hash
compatibility, default pipeline routing and layer toggle binding.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  .conda/bin/python -m unittest discover -s tests -p test_multi_camera.py -v
# Existing OpenCV lives in system Python; no dependency installation needed.
PYTHONPATH=src /usr/bin/python3 -m unittest discover -s tests -p test_multi_camera.py -v
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  .conda/bin/python -m unittest discover -s tests -v
.conda/bin/python -m compileall -q src tests
RUFF_BIN=/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff
"$RUFF_BIN" check src tests script
"$RUFF_BIN" format --check src tests script
```

The 19 new tests are covered across Ada and system Python (two environment-specific
skips in each, exercised by the other interpreter). Final focused dual/native and
multi-camera regressions: 53 tests, 51 passed, two image-environment skips.
Full canonical suite: 746 tests, 646 passed, 99 skipped, one failure because the
pre-existing simulation occupies 8089. That existing mocked launcher test passed
when only its port-availability check was mocked; no robot process was started.
The existing simulation was preserved. Global Ruff lint, changed-file formatting,
compilation and `git diff --check` pass. Global formatting reports eight pre-existing
unformatted files outside this change (the two CRX phase scripts,
`crx_sharpa_adaptive.py`, `crx_sharpa_demo.py`, `utils/viewer_session.py` and three
CRX/dummy-arm tests); they were not reformatted.

A 150,000-point synthetic sphere benchmark measured 0.2024 s for transform/fusion
and 0.0236 s for seed association (0.2260 s combined, 2,475 retained points).
It excludes capture, human ROI, model inference, serialization and the existing
D415 ROI extraction. It is not a real-scene total-perception guarantee.
`validation_20261127/synthetic_benchmark.json` explicitly labels this evidence
synthetic. The final probe with coverage thresholds also passed under
`output/multi_camera/final_probe_20261127/`.

Calibration collection, measured extrinsics, external overlap, real fused
inference/selection/export and physical 8089 fused-cloud review remain unverified.
The exact serial-to-physical-left/right mounting assignment still needs operator
confirmation. A preview cannot establish grasp success. Camera-only drivers
started during this task were left running; no robot was commanded, authoritative
calibration changed, commit made or push performed.


## Installed stationary D405 results — 2026-11-27

The operator requested installation before independent-pose validation. Both converged
ROS-Industrial results from `static_fit_20261127_183512/app_inputs_8ee3f0e62f` are now
normalized into `authoritative.stationary_cameras` in the existing master calibration.
They map the respective color optical frames into `left_base_link`; the board was attached
to `right_ee_mount`. D415 geometry, arm bases, hand mounts, and the existing table are unchanged.
The pre-installation master is retained and hash-verified for existing D415/table compatibility.

Each new entry has `status: pending_independent_validation`, `solver_converged: true`,
`independent_validation_passed: false`, 20 fit inputs, and zero independent validation samples.
The default `multi_camera.yaml` explicitly pins the SHA256 of these two complete entries in
`accepted_pending_calibrations`, per the operator's request. This is an exact-result policy,
not a general skip-validation flag. Unselected pending results still fail; mismatched serial,
frame, units, geometry, source hashes, or solver convergence still fail. Calibration status
is retained in capture reports and snapshot provenance. Independent validation thresholds
and the validated-entry import path are unchanged.

### Start the calibrated cameras

With the previous camera/calibration launch stopped, run in a desktop terminal:

```bash
ROS_DOMAIN_ID=87 bash /home/hao-fang/hugs-workspace/ada-grasp-ctrl/script/start_calibrated_cameras.sh
```

This starts the two D405 RGB-D drivers using the existing `multi_camera_driver.yaml` and
one extrinsics publisher. It does not start arm/hand drivers or issue motion commands.
To also start the D415 for three-camera fusion, append `enable_fix_camera:=true`.
If the camera drivers are already running with internal TF enabled, append
`start_cameras:=false` to reuse them. The old calibration capture profile disables depth
and internal TF; restart with the RGB-D profile above when transitioning from capture.
The same-domain duplicate-driver graph check is bounded, not a cross-domain device lock.
The launch shuts down its own processes if the extrinsics publisher exits.

The publisher checks the two live serial parameters, reads each RealSense internal
`<role>_link -> <role>_color_optical_frame` transform, and publishes only:

```
left_base_link -> wrist_left_link -> [RealSense internal TF] -> wrist_left_color_optical_frame
left_base_link -> wrist_right_link -> [RealSense internal TF] -> wrist_right_color_optical_frame
```

The root pose is `T_base_optical @ inverse(T_link_optical)`. No second parent is assigned
to an optical frame. If either root already has a parent, startup fails instead of
replacing an existing TF owner. Internal transforms must be available; none are guessed.
The existing fusion code reads the same master directly and composes color/depth TF once.
The hardware pipeline remains opt-in via its existing `--multi-camera` option; omitting it
continues the D415-only path. Starting this camera launch does not run grasp inference or motion.

Installation record and rollback copies:

`output/wrist_calibration_20261127_174104/static_fit_20261127_183512/system_install_20261127/`

Restore both `previous_calibration.json` to the master calibration path and
`previous_multi_camera.yaml` to its profile path if reverting this installation; stop the
extrinsics publisher first. Original solver and capture files are retained.

Initial focused regression: 46 tests, 43 passed and 3 optional-dependency skips. Actual ROS TF
verification used synthetic camera nodes on domain 198/localhost: both optical transforms
matched solver matrices to 1e-10, duplicate root owners were rejected, and wrong serials
were rejected. No physical driver was started for these checks. Physical alignment and
independent-pose accuracy remain unverified.

Final installation validation: full Ada unittest discovery completed with **773 tests: 683 passed,
90 optional-dependency skips, no failures/errors** (199.618 s). ROS/system-Python focused
fusion tests: 24 total, 22 passed, 2 optional skips. Full Ruff lint, compileall, changed-file
formatting, shell syntax, launch argument discovery, and diff whitespace checks passed.
The whole-tree formatting check retains 8 pre-existing unrelated differences. Logs and
`validation.json` are retained in `system_install_20261127/`. No domain-87 process or physical
hardware driver was started during installation.

Commit preparation curated the small source artifacts into
`config/calibration/stationary_d405_20261127/` under the package. Active provenance paths
now resolve through Ada's data root; original captures remain untracked. Follow
[the new-conversation handoff](handoff-2026-11-27.md) for the original dual pipeline's
actual deployment status, next live preview command, and separate commit scopes.
