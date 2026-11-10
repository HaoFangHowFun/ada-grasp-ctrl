# Fixed D415 to ROS to Ada: table, object and right-hand grasp previews

These experimental Task 04 tools do not change Ada's public four-stage control workflow or supported hand matrix.
Bimanual runs one camera driver; Ada passively subscribes to PointCloud2, RGB, CameraInfo and internal camera TF.
Table fitting, object extraction, DexLearn inference and previews belong to Ada. These tools create no arm or hand
command publishers.

## Environments and camera startup

Use system Python 3.12 for ROS Jazzy, Ada's existing `.conda` Python 3.10 for previews, and DexLearn's existing `.venv`
for inference. Do not inherit ROS Python paths into Ada or DexLearn. No dependencies were added: OpenCV comes from
the existing system environment; Viser and Trimesh come from Ada's environment.

Build bimanual with the optional camera settings, then start its mock/read-only system:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/hao-fang/ws_fanuc
colcon build --symlink-install --packages-select bimanual_manipulation
source install/setup.bash
export ROS_DOMAIN_ID=87
ros2 launch bimanual_manipulation bimanual_system.launch.py \
  crx_mock:=true sharpa_backend:=mock read_only:=true \
  rviz:=false viewer:=false enable_fix_camera:=true \
  camera_config:=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/src/ada_grasp_ctrl/config/experiments/fixed_camera_driver.yaml
```

The supplied configuration selects fixed D415 `347322060270`, 640x480 at 30 FPS, RGB/depth, pointcloud, internal TF and
synchronization. Upstream defaults remain RGB-only. Never start a second driver for the same serial.
For camera-only use, replace `bimanual_system.launch.py` with `cameras.launch.py` and retain only
`enable_fix_camera:=true camera_config:=...`. A desktop session can enable the system launch's RViz or viewer.

Prepare another ROS shell:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=87
export ADA_ROOT=/home/hao-fang/hugs-workspace/ada-grasp-ctrl
export PYTHONPATH="$ADA_ROOT/src:$PYTHONPATH"
cd "$ADA_ROOT"
```

Relative inputs resolve against Ada's data root, defaulting to the checkout; outputs use `--output-root` and the Ada
path helpers. Every invocation requires a new output directory, including retries after a diagnostic failure.

## Capture and offline table fitting

Clear a visible table patch manually, then capture and fit:

```bash
/usr/bin/python3 -m ada_grasp_ctrl.experiments.fixed_camera capture \
  --extrinsics /home/hao-fang/hugs-workspace/agent_tasks/camera_extrinsics_20261107_640x480.json \
  --output-root output/fixed_camera/my_empty_table

/usr/bin/python3 -m ada_grasp_ctrl.experiments.fixed_camera fit \
  --snapshot output/fixed_camera/my_empty_table \
  --roi 120 180 500 380 \
  --output-root output/fixed_camera/my_table_fit
```

ROI coordinates are `x0 y0 x1 y1`, excluding the right and bottom edges. Inspect `rgb.png` to choose an empty patch;
the example coordinates are specific to this session. With a display, omit `--roi` to select on the RGB image and
confirm with Enter. Cancellation rejects the fit. Fitting is offline, but projection and overlays require OpenCV.

Defaults live in `src/ada_grasp_ctrl/config/experiments/fixed_camera.yaml`; use `--config` for a separate configuration.
Thresholds cover depth, freshness, synchronization, valid fraction, point count, RANSAC, residuals, coverage, height,
tilt and clustering. Defaults are not a guarantee of D415 accuracy.

The capture path uses metric PointCloud2 XYZ directly, without reprojecting depth through intrinsics. It checks the
live serial, RGB/depth CameraInfo, image dimensions and frames; rejects stale, unsynchronized, low-coverage or invalid
data; transforms depth-optical points through internal TF to color optical, then through JSON `T_world_cam` once.
It does not additionally apply world TF. `T_flange_board` is unused. Color projection uses live CameraInfo and supports
plumb-bob/rational distortion.

Snapshots retain raw cloud/image/CameraInfo CDRs, unfiltered XYZ, filtered camera/world XYZ, RGB, transforms, timestamps,
frames, intrinsics, calibration hashes, serial and valid fraction. Loading verifies the NPZ hash and camera/world
round trip. Raw CDRs can be deserialized with `rclpy.serialization.deserialize_message` and the corresponding sensor type.

Fitting writes `table_plane.json`, `inlier_indices.npy` and `inlier_overlay.png`. The plane is `n·p+d=0`, with unit normal
oriented toward world +Z; `h(p)=n·p+d` is positive above it. RANSAC is followed by least squares on inliers. The full tilted
plane is retained; `z_table` is supplied only within the horizontal tolerance. Observed bounds and tangent spans describe
the measured patch, not the entire table. No table frame, yaw, thickness or collision box is inferred.

## World preview and acceptance

Set RViz's fixed frame to `world`, then add PointCloud2 `/ada/table/points` and Marker `/ada/table/plane`:

```bash
/usr/bin/python3 -m ada_grasp_ctrl.experiments.fixed_camera preview \
  --plane output/fixed_camera/my_table_fit/table_plane.json
```

This publishes visualization only. Stop with Ctrl+C; `--duration 5` limits a smoke run. Planes start as
`preview_unaccepted`. A poor fit still saves diagnostics, returns 1 and cannot be accepted.

Before acceptance, capture and fit at least three distributed regions, and verify camera-to-world using at least three
non-collinear known markers. Supply measured color-optical XYZ and independently measured world XYZ for each marker.
Point-to-plane distance alone cannot verify world XY origin or yaw: acceptance checks full XYZ correspondence errors,
repeat normals/heights and region separation. Counts, distances and angular tolerances are configurable.

Create a JSON evidence file with measured values, never references derived from the fitted plane:

```text
camera_base_unchanged: true
world_definition_confirmed: true
reference_points_color_m: [[...], [...], [...]]
reference_points_world_m: [[...], [...], [...]]
repeat_plane_files: ["/absolute/path/repeat1/table_plane.json", "/absolute/path/repeat2/table_plane.json"]
```

After inspecting the overlay, world preview and diagnostics, explicitly accept:

```bash
/usr/bin/python3 -m ada_grasp_ctrl.experiments.fixed_camera accept \
  --plane output/fixed_camera/my_table_fit/table_plane.json \
  --evidence /absolute/path/measured_references.json --reviewed \
  --output-root output/fixed_camera/my_table_accepted
```

Geometric acceptance does not authorize robot motion. Reload checks preserve representative signed heights. Revalidate
after changing the camera, table, base or world relationship. Camera extrinsics and Sharpa mounts are not overwritten.

## Object extraction

After accepting the empty table, place an object and capture it with the same camera, calibration and work area:

```bash
/usr/bin/python3 -m ada_grasp_ctrl.experiments.fixed_camera object \
  --snapshot output/fixed_camera/my_object_capture \
  --plane output/fixed_camera/my_table_accepted/table_plane.json \
  --roi 125 55 255 125 \
  --output-root output/fixed_camera/my_object
```

Extraction uses signed height above the full plane and voxel-assisted radius clustering. Multiple clusters produce
`clusters.json` and overlays; retry in a new directory with explicit `--cluster ID`. Outputs include `object.npz`,
`object.json` and `object_overlay.png`, source references/hashes, selected indices, color/world points, frames, units,
timestamps and settings. The accepted path requires the object capture to follow the accepted empty-table capture.

To publish the object at `/ada/object/points` for RViz:

```bash
/usr/bin/python3 -m ada_grasp_ctrl.experiments.fixed_camera preview \
  --plane output/fixed_camera/my_table_accepted/table_plane.json \
  --object-root output/fixed_camera/my_object
```

`--allow-unaccepted` is for diagnostic previews and propagates that status into extraction, inference and preview reports.
Thin objects may require a noise-aware `object_min_height_m` adjustment. This session used a separate 8 mm configuration;
the general default remains 15 mm.

## Checkpoint inference

Run without inherited ROS Python paths. These are the locally verified checkpoint, configuration and metadata:

```bash
export ADA_ROOT=/home/hao-fang/hugs-workspace/ada-grasp-ctrl
export DEX_ROOT=/home/hao-fang/hugs-workspace/HUGS-DexLearn
export TRAIN_ROOT=/home/hao-fang/hugs-dataset/BimanBODex_sharpa_1264k_v1/dexlearn_training/sharpaMulti_robotMultiHierar_sharpa_right_full_1m_100k_v1
export PYTHONPATH="$ADA_ROOT/src"
OMP_NUM_THREADS=4 "$DEX_ROOT/.venv/bin/python" -m ada_grasp_ctrl.experiments.fixed_camera_grasp \
  --object output/fixed_camera/my_object \
  --dexlearn-root "$DEX_ROOT" \
  --checkpoint "$TRAIN_ROOT/ckpts/step_100000.pth" \
  --training-config "$TRAIN_ROOT/.hydra/config.yaml" \
  --metadata /home/hao-fang/hugs-dataset/BimanBODex_sharpa_1264k_v1/sharpa_right/right_full/metadata.json \
  --samples 8 --seed 4 \
  --output-root output/fixed_camera/my_candidates
```

Add `--allow-unaccepted` for diagnostic runs. The implementation supports the verified `RobotHierarchicalModel` /
`DiffusionBiRT_MLPRTJ` contract and rejects incompatible models. It calls the actual model; no grasp or scene is fabricated.

Preprocessing follows training: 1,024 points sampled with replacement, centered on the sampled mean, in meters and world
XYZ axes. There is no rescaling or rotation. Outputs restore the center without applying camera-to-world again. All three
poses are also saved in color coordinates with round-trip checks. The 22 finger angles retain metadata names, radians
and WXYZ quaternions. Pregrasp, grasp and squeeze are preserved without extrapolation or phase substitution.

`candidates.json` records IDs, all three poses and named angles, raw candidate indices, log probabilities, checkpoint /
configuration / metadata hashes, DexLearn commit, calibration, table state and preprocessing. Raw predictions and
preprocessing NPZ/JSON files permit offline inspection. Training used complete object clouds; these inputs are single-view
visible surfaces, so grasp quality under this distribution difference remains unverified.

## Three-stage hand preview

```bash
export ADA_ROOT=/home/hao-fang/hugs-workspace/ada-grasp-ctrl
PYTHONPATH="$ADA_ROOT/src" "$ADA_ROOT/.conda/bin/python" \
  -m ada_grasp_ctrl.experiments.fixed_camera_preview \
  --candidates output/fixed_camera/my_candidates/candidates.json \
  --hand-urdf /home/hao-fang/hugs-workspace/HUGS-Sharpa-Assets/urdf/right_sharpa_wave_with_flange.urdf \
  --output-root output/fixed_camera/my_preview
```

Open `http://127.0.0.1:8087`. Select a candidate and one or all stages: blue pregrasp, green grasp, orange squeeze. The world
view contains the observed table patch, scene/object points and actual right-hand URDF visual meshes. Per-stage diagnostics
show minimum signed table distance and joint-limit violations. Angles are not clipped and candidates are not moved to
conceal table intersections. These static visual checks do not establish safe paths or successful physical grasps.

Outputs include `preview.glb` and `preview_report.json`, all candidate diagnostics and model/mesh hashes. `--export-only`
exports without serving; `--duration 2` limits a smoke run; `--candidate c0001` selects the initial candidate.
CRX reachability is unassessed without a verified physical mount. The hand is placed at the checkpoint's
`right_hand_flange`; the calibration board's `T_flange_board` is not used. Physical execution remains Task 02; this tool
has no execution sender and never automatically sends candidates.

## Validation and exit codes

- `0`: invocation completed; a preview may still be unaccepted.
- `1`: fitting completed but failed quality thresholds; inspect saved diagnostics and overlays.
- `2`: input, configuration, environment or inference failure; the new run directory retains its failure report.

Run geometry, asset and inference-contract tests in Ada, then cover OpenCV projection in system Python:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl .conda/bin/python -m unittest discover -s tests -p 'test_fixed_camera*.py' -v
PYTHONPATH=src /usr/bin/python3 -m unittest discover -s tests -p 'test_fixed_camera_grasp.py' -v
```

The full selection needs Ada's Trimesh for PLY round trips. Ada skips the system-OpenCV projection test, which is covered
by the second command. Coverage includes tilted/horizontal planes, signs, extrinsic direction, invalid/low-coverage depth,
timing, quality rejection, ROIs, clusters, file integrity, full-XYZ references, repeat evidence, acceptance reloads,
sampling round trips, all three stages and measured-surface topology.

See [validation results and outstanding acceptance work](fixed-camera-validation.md) and
[visible-surface asset export and previews](fixed-camera-assets.md).
