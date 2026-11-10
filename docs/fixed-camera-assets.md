# Measured visible-surface assets

`fixed_camera_asset` extends the passive camera pipeline with point-cloud and open-surface exports.
It does not complete unseen geometry or create a simulation/collision solid. No dependencies or existing public
hand/converter contracts were changed.

## Build from a captured object

Use system Python with the existing OpenCV installation. Set `PYTHONPATH` to the Ada checkout's `src`.
Input is an `object` run from `fixed_camera`, including its raw capture provenance.

```bash
PYTHONPATH=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/src \
  /usr/bin/python3 -m ada_grasp_ctrl.experiments.fixed_camera_asset build \
  --object output/fixed_camera/green_object_20261109_1727 \
  --name green_pear_visible_surface \
  --output-root output/fixed_camera/assets/my_new_asset
```

Every output root must be new. Relative paths use the existing Ada root helpers, independent of shell CWD.
The source cloud must be in the original depth optical frame, with zero-distortion rectified depth CameraInfo.
Points must project back to unique original depth pixel centers; unsupported inputs are rejected rather than meshed
with guessed topology. Each triangle connects neighbors in a 2×2 depth-pixel cell. Missing pixels remain holes.
Triangles with an edge or depth jump exceeding the configured limit are rejected. Defaults are 8 mm for each,
configurable with `--max-edge-m` and `--max-depth-jump-m`.

Vertices remain measured coordinates. Normals are oriented toward the observing camera. RGB is sampled from the
synchronized color frame using the recorded depth→color transform and live color CameraInfo. There is no smoothing,
Poisson reconstruction, convex hull, bottom cap, scaling or hidden-surface completion.

Files:

- `points.npy`, `points.ply`: all selected points, centered at their observed centroid, meters.
- `surface.ply`, `surface.obj`: open measured surface in the same local frame. PLY preserves per-vertex RGB;
  OBJ also includes the common vertex-color extension (support depends on the reader).
- `points_world.npy`, `points_world.ply`, `surface_world.ply`: original calibrated world coordinates, meters.
- `surface.npz`: local vertices, triangle indices, RGB, `T_world_asset`, `T_world_color`.
- `asset.json`: dimensions of the **observed** support, mesh diagnostics, calibration/table state, hashes and provenance.
- `object.npz`, `object.json`: preserved inference input. The existing `fixed_camera_grasp --object <asset directory>`
  can consume it. Use `--allow-unaccepted` only for diagnostics while the table remains unaccepted.
- `rgb.png`, `object_overlay.png`, `capture_metadata.json`, `table_plane.json`: inspection and frame context.
- `run_manifest.yaml`, `run_report.json`: current invocation and outcome.

`T_world_asset` translates local vertices back into world. Local axes remain parallel to world; the origin is the
centroid of **visible samples**, not an inferred object center of mass. Reported extents do not prove full object size.
Backsides, bottoms, occlusions and points rejected by the table-height threshold remain unobserved.

## Interactive preview

Use Ada's Python environment with a clean PYTHONPATH (do not inherit ROS Python 3.12 packages into Python 3.10):

```bash
PYTHONPATH=/home/hao-fang/hugs-workspace/ada-grasp-ctrl/src \
  /home/hao-fang/hugs-workspace/ada-grasp-ctrl/.conda/bin/python \
  -m ada_grasp_ctrl.experiments.fixed_camera_asset preview \
  --asset output/fixed_camera/assets/my_new_asset \
  --output-root output/fixed_camera/my_new_asset_preview --port 8088
```

Open `http://127.0.0.1:8088`. Display modes are surface, point cloud, and both; additional controls show wireframe,
local axes, and the observed table patch. The sidebar includes original RGB and the exact extraction overlay.
Coordinates in the viewer are object-local; source world placement is retained in the asset metadata.
The preview exports a colored `surface.glb`. Use `--export-only` for GLB without a server, or `--duration 2` for a smoke run.
Double-sided display aids inspection of the intentionally open surface; it does not create a back surface.

## Connected-camera result, 2026-11-09

The user placed a new green pear-shaped object. It was captured with bimanual's camera launch from fixed D415
`347322060270`; the previous unaccepted table plane was reused as requested.

- Capture: `output/fixed_camera/object_capture_20261109_1727`.
- Extraction: `output/fixed_camera/green_object_20261109_1727`, ROI `[245,100,340,180]`.
- Separate extraction config uses an 8 mm above-table cutoff; the general default remains 15 mm.
- Asset: `output/fixed_camera/assets/green_pear_20261109_1727`.
- Preview: `output/fixed_camera/green_pear_preview_20261109_1727`, port 8088.
- 2,848 colored measured points; 5,293 triangles, with 121 candidate triangles rejected at discontinuities.
- 2,779 vertices participate in faces; all 2,848 observations remain in the point cloud.
- 263 open boundary edges, zero nonmanifold edges; intentionally not watertight.
- Observed world-axis extents: 104.57 × 105.97 × 61.41 mm.
- PLY reload: identical face indices and RGB; maximum coordinate error below `5e-14 m`.
- Five new unit tests verify face orientation, neighbor topology, missing-pixel holes, depth-step rejection,
  invalid grids and PLY color/metric round trip. The fixed-camera test selection ran 21 tests, OK with one skip
  (OpenCV projection runs in the separate system environment). Ruff and compilation passed.

The initial asset-only pass did not run grasp inference. A later pass generated eight candidates for this object and
eight each for four additional groups; see [the final session results](fixed-camera-validation.md#five-object-capture-session-and-shutdown).
No physical motion was run. Calibration/table status remains `preview_unaccepted`; all exports preserve that state.
