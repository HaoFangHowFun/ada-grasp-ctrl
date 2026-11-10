# Task 04 validation — 2026-11-09

Implementation: `fixed_camera`, `fixed_camera_geometry`, `fixed_camera_ros`, `fixed_camera_grasp`,
`fixed_camera_preview`, `fixed_camera_asset` under `src/ada_grasp_ctrl/experiments/`.
Bimanual changes are limited to optional per-role pointcloud/internal-TF/sync configuration, validation, tests and docs.
Existing RGB defaults and unrelated local Task 02 work are retained.

## Measured on the connected camera

- Fixed D415 serial `347322060270`, firmware 5.17.0.10; RealSense ROS/SDK 4.58.4/2.58.4.
- Started through **bimanual_system.launch.py**, with Ada's `fixed_camera_driver.yaml`, one USB driver.
- Arms: `mock_components/GenericSystem`; hands: `backend=mock`; `read_only=true` throughout.
- Ada subscribed to PointCloud2/RGB/CameraInfo and internal TF. No motor commands, Sharpa SDK, Task 02 sender or
  physical arm drivers were used.
- Example capture `output/fixed_camera/live_20261109_c`: 264,428 cloud points before distance filtering;
  valid source fraction 0.9790. RGB/cloud/color CameraInfo/depth CameraInfo timestamps matched exactly;
  receipt age 36.5 ms. RGB/depth calibration resolutions 640×480. Cloud frame `fixed_depth_optical_frame`.
- The tool recorded depth→color TF and applied the provided color→world JSON once. World TF was not used.
  Live color K agrees with the supplied file. This does **not** establish that the physical extrinsic calibration is valid.
- Later captures `live_20261109_d` and `live_20261109_e` verified the added depth-pixel-coverage rejection path on real data.
  Raw PointCloud2, images, CameraInfo, NPZ and calibration identifiers are retained in each run.

## Table previews and repeated regions

All results remain **preview_unaccepted**. The camera saw a table with a phone and other items; the selected ROIs were
clear patches. These were implementation tests, not the prescribed accepted empty-table / object-insertion procedure.

| Region / run | Inliers | RMS | p95 | Tilt from world +Z |
| --- | ---: | ---: | ---: | ---: |
| Central `table_preview_20261109_b` | 76,407 | 2.070 mm | 4.048 mm | 1.210° |
| Left `table_repeat_left` | 44,464 | 1.147 mm | 2.313 mm | 1.210° |
| Right `table_repeat_right` | 50,357 | 1.155 mm | 2.202 mm | 0.813° |

Central plane: `normal=[-0.00514222, 0.02048409, 0.99977695]`, `d=0.17695013 m`.
Its observed centroid is `[0.46454248, -0.37678390, -0.16688049] m` in the supplied world definition.
No extrinsic/world/base corrections were made to level this plane.

Compared with the central fit, left/right normals differ by 0.214° / 1.971°;
maximum signed-height disagreement at the pair of observed centroids is 0.304 / 5.464 mm.
The right comparison is close to the configured 2° repeat tolerance. This is a partial repeat diagnostic,
not the reference-point acceptance check. Exact values: `output/fixed_camera/repeat_comparison.json`.

Re-fitting the same central snapshot produced identical plane coefficients and quality statistics.
The task's `[0,0,0.5] color → [0.359774,-0.383605,0.003431] world` reference is also covered by a numerical test.

## Real point cloud → real checkpoint → hand preview

- ROI `[125,55,255,125]` captured the visible phone. At the default 15 mm height threshold, no qualifying cluster
  was found and the tool rejected it. The measured phone surface was mostly about 13 mm above the plane.
- A separately saved diagnostic config used 8 mm (the default file remains 15 mm). It selected 2,410 measured points;
  `phone_preview_20261109_b/object_overlay.png` was inspected and overlays the phone surface.
- Used `--allow-unaccepted` explicitly. This status propagates into object, candidate and preview records.
- Loaded the actual `step_100000.pth` right-full Sharpa checkpoint, hash
  `92fc78e580753a3659116e867e075e91d75a342176abd2434b9d98828b3083d6`, with its `.hydra/config.yaml` and
  `sharpa_right/right_full/metadata.json`. No synthetic scene configuration was created.
- Ran `RobotHierarchicalModel.sample` in the existing DexLearn GPU environment (PyTorch 2.11.0+cu128), generated
  **8 right_full candidates**. Training contract: 1,024 metric points, replacement sampling, centering, world XYZ,
  scale 1, three 58D outputs with the right hand in the first 29 entries. The right pose is the flange; 22 named
  angles are radians and quaternion order is WXYZ. No pregrasp extrapolation, clipping, phase substitution or
  repeated camera→world transformation occurred.
- A second invocation from the same object snapshot and seed reproduced outputs within the repository's existing
  `rtol=1e-5, atol=1e-6`. They were **not bitwise identical**: max pose-element difference `1.0803e-6`, log-probability
  difference `1.5259e-5`; type IDs and type probabilities were exact. No tolerance was changed.
  See `inference_replay_comparison.json`.
- World/color pose round trips, centered-point inverse mapping and raw outputs are saved in
  `phone_grasps_20261109_a` and `phone_grasps_20261109_replay`.
- Loaded the actual flange-rooted right Sharpa URDF, checked its named joints, and generated all three stage meshes.
  Saved GLB and per-candidate static distances. Browser preview on loopback port 8087 was smoke-tested; RViz table
  and object PointCloud2/plane Marker publication also completed.
- Candidate `c0000` (highest likelihood) has minimum visual-mesh table distances approximately
  `[-12.2, -14.4, -12.3] mm`: **intersects the table**. `c0001` is about `[19.8, 15.3, 11.3] mm`; `c0006` is about
  `[5.6, 1.1, 0.7] mm`. None is declared executable. No joint-limit violations were observed in these 8 candidates.
  Exact results and source mesh hashes are in `phone_visual_20261109_a/preview_report.json`.

## Software checks

- New geometry/contract tests: **16 passed** in system Python. Ada Python 3.10 skips only the OpenCV projection test,
  which is covered in system Python.
- Full Ada unit suite: **197 tests run, OK, 1 skipped** (the system-OpenCV test), at the final geometry/contract revision.
  Command: `PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl .conda/bin/python -m unittest discover -s tests -v`.
- Bimanual camera, integration, single-side and gain launch selection: **49 passed, 10 skipped**.
  Camera tests alone: **24 passed**, including unchanged defaults, flag validation, depth prerequisite and per-role isolation.
- `ruff check src tests script`: passed. `python -m compileall -q src tests`: passed.
- Formatting passes for all new files. Repository-wide `ruff format --check src tests script` identifies 8 unrelated
  existing files; they were not reformatted as part of this task.
- Both worktrees passed `git diff --check`.
- Original `shadow quick` was also exercised. An initial run inherited ROS Python 3.12 paths into Python 3.10 and failed
  to import Pinocchio. Re-running with `PYTHONPATH=<Ada>/src` and `CUDA_VISIBLE_DEVICES=''` completed all 100 outputs:
  68 success, 4 failure, 21 invalid_initialization, 7 solver_degraded, 0 execution_error.
  **This does not match the stored Shadow inventory** (66/4/21/9/0). The existing inventory verifier rejected the run.
  This broader classification discrepancy remains unresolved; no golden records, tolerances, controller hyperparameters
  or control semantics were changed. Task 04 does not route through this simulation pipeline.
  Artifacts: `output/examples/shadow/20261110T011021Z-25098` and `/tmp/task04-ada-quick-clean.log`.

## Still required for Task 04 physical acceptance

- User confirmation of unchanged camera/base relation and current world definition.
- Independent measured color/world correspondences at distributed known markers; none were supplied during this run.
- User inspection and explicit `accept --reviewed` after reference/repeat checks. No accepted production table file was made.
- Accepted empty-table capture followed by a separate object capture, using the intended object and work area.
- Review suitability of single-view partial clouds for the complete-point-cloud training distribution.
- Physical right CRX / Sharpa mount verification before any claim of reachability; currently shown as unassessed.
- Longer camera stability assessment if needed. Startup has previously emitted a depth hardware warning despite later
  successful data; this implementation does not conceal that warning or claim continuous reliability.

Real arm grasp execution remains Task 02 and was not attempted.

## Five-object capture session and shutdown

After the initial phone trial, five object groups were captured through the bimanual camera-only launch. The same
unaccepted table and separately saved 8 mm extraction threshold were reused with explicit `--allow-unaccepted`.
Each group produced a measured surface asset and eight real-checkpoint candidates, for 40 candidates in total.

| Group | Object | Measured points | Surface triangles | Candidate diagnostics |
| --- | --- | ---: | ---: | --- |
| 01 | Green pear-shaped object | 2,848 | 5,293 | `c0001`: right pinky CMC limit violation |
| 02 | Green object, second capture | 3,244 | 5,986 | No static table intersections or joint-limit violations |
| 03 | Rubik's cube | 3,607 | 6,727 | No static table intersections or joint-limit violations |
| 04 | Transparent bottle | 5,935 | 10,669 | No static table intersections or joint-limit violations |
| 05 | USB adapter body | 1,463 | 2,760 | Seven candidates intersect the table; only `c0001` has no such flag |

The transparent bottle remains subject to depth errors from its material. The USB adapter produced two clusters;
cluster 0 was explicitly selected after overlay inspection. Its cable and USB plug are excluded from the asset.
These are single-view open surfaces, not completed solids or certified executable grasps.

The local gallery at port 8089 embedded five independent candidate viewers. Its manifest and source thumbnails are
retained in `output/fixed_camera/five_objects_gallery`; generated captures, assets and gallery files are not committed.
All five child viewers, thumbnails and the gallery returned HTTP 200 before review. The user finished review and
requested shutdown: all ten preview servers and the ROS camera launch/node exited, and their listening ports were
verified closed. No camera, ROS node or preview server was restarted for commit checks. All local artifacts remain saved.

## Final commit checks

- Ada camera selection: 21 tests run, OK with one system-OpenCV skip.
- System Python grasp-contract selection: all five tests passed, including OpenCV projection.
- The entire camera selection is not suitable for system Python: its PLY round-trip test requires Trimesh, available
  in Ada's environment. That test passed in Ada; no dependency was installed to merge the environments.
- Bimanual camera selection: 24 passed after sourcing both Jazzy and the existing workspace for `dual_sharpa_wave`.
- Repository Ruff lint and Python compilation passed. The eight previously documented unrelated formatting findings
  remain outside this change. The earlier Shadow inventory discrepancy remains unresolved and was not rerun.
