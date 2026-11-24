# Phase 1: interactive placement selection

The existing Dual Live entry now accepts `--placement-only`. It reuses the current
camera capture and ROI/cluster object selection, then opens the existing **8089**
preview in placement mode. Grasp model setup, HUGS inference, robot state capture,
planning and execution are bypassed. The normal Dual Live command is unchanged.

## Run

With the existing fixed camera publisher available, use a fresh output directory:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
env -u LD_LIBRARY_PATH python3 script/run_dual_grasp_pipeline.py \
  --placement-only --environment hardware --ros-domain-id 87 \
  --output-root "output/placement_selection/live_$(date +%Y%m%d_%H%M%S)"
```

This command does not start or configure robot hardware. It needs the camera's
existing RGB/depth/CameraInfo topics for a new capture. Use the existing ROI window,
or supply `--roi x0 y0 x1 y1` and, when needed, `--cluster ID`. The ROI window retains
its existing image-selection confirmation; there is no additional terminal Enter gate.

Open <http://localhost:8089>. **P00 — Original Pose** is selected immediately.
Click **Run UOP-Net once**, choose a placement candidate, then **Confirm Placement**.
The panel shows both the selected candidate and the most recently confirmed one.
Changing the dropdown after confirmation does not overwrite the saved selection;
click Confirm Placement again to save another selection. The viewer stays open for
inspection; Ctrl+C closes it. Confirmation never advances into a grasp or execution stage.

To review an explicitly chosen saved camera object without any ROS capture:

```bash
env -u LD_LIBRARY_PATH python3 script/run_dual_grasp_pipeline.py \
  --placement-only --environment mock --ros-domain-id 88 \
  --object-root output/dual_live/native_both_live_20261123_173430/object \
  --output-root "output/placement_selection/replay_$(date +%Y%m%d_%H%M%S)"
```

The object path above is a local validation example, not a default object. Reopen
an existing placement run using its cached candidates, with no new inference:

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src .conda/bin/python \
  -m ada_grasp_ctrl.pipeline.dual_viewer --placement-only \
  --run /absolute/path/to/placement_run
```

Only one viewer owns 8089; an occupied port is an error. No alternate viewer or
port is started. `env -u LD_LIBRARY_PATH` isolates inherited ROS native libraries
from Ada; camera subprocesses still source the configured ROS environments.

## Model and geometry

Default UOP installation: the source checkout's sibling `../uop-net`, overridable
with `--uop-root` or `UOP_NET_ROOT`. `--uop-python` overrides its existing
`.venv/bin/python`. No dependencies, checkpoints or upstream files are changed.

The adapter calls the installed `UOPModule.inference_gpu()` once, then fits every
positive instance with at least 50 points using the installed plane fitter. It
keeps all valid planes, preserving instance IDs, normals, plane equations,
rotations and inlier counts/ratios/indices. It adds no ratio threshold, fixed
candidate count or deduplication. The original Top-1 API remains unchanged.
P01 onward are ordered by instance ID within that saved result. RANSAC reruns may
produce different planes; the saved plane/rotation pair is authoritative.

Only 2048 distinct observed points are deterministically sampled for inference;
fewer than 2048, missing model/environment, inference failure or zero planes leave
P00 available. The complete selected partial cloud is retained for visualization.
An inference attempt is one-shot, including failures; start a fresh run to retry.

Points and relative rotations are in the captured `world` frame, with the source
calibration transform/hash retained. Active column-vector rotations use
`v_new = R_relative @ v`; row points use `@ R_relative.T`. The installed UOP
rotation maps the outward support normal to world `-Z`; the adapter composes a
rotation from `+Z` to the measured table normal. A deterministic parallel/antiparallel
guard fixes the known singularity of the installed rotation helper without
changing ordinary-angle behavior or upstream code.

Every display update uses `(original_points - centroid) @ R_relative.T + centroid`.
P00 returns an exact copy of the original points. The table/environment stays
fixed; the selected object's original pixels are excluded from the background
to avoid displaying a duplicate. The object uses float32 Viser transmission.
The centroid is a display pivot, not a planned translation or inferred contact
height. No shape completion, stability certification or collision certification
is performed; P00 is not automatically stable.

## Saved artifacts

- `snapshot.npz`: original object cloud, captured environment, object point indices,
  table normal and camera-to-world transform. It is never rebuilt during switching.
- `source_object.json`, `source_capture.json`, `table_plane.json`: source provenance.
- `candidates.json`: P00 plus cached valid UOP candidates, frame conventions and status.
- `uop_invocation.json`, `uop.log`, `uop_result.json`, `inference.npz`: actual invocation,
  checkpoint/source hashes, seed, normalized sampled inputs, logits, embeddings,
  labels and same-inference plane diagnostics.
- `selection.json`: latest confirmed candidate ID/source/relative rotation/snapshot
  ID/frame conventions, explicitly `translation_status: undetermined`.
- `selection_<confirmation_id>.json` and `candidates_<sha256>.json`: retained
  confirmation history and its exact candidate set, including confirmation of P00
  before inference. No full 6D placement pose is represented.
- `run_manifest.yaml`, `run_report.json`, `viewer_state.json`: run configuration and mode.

## Implementation and validation (2026-11-24)

New modules: `placement/session.py` (data and confirmation), `uop_adapter.py`
(installed-UOP compatibility adapter), `workflow.py` (capture/replay orchestration),
`ui.py` (single-viewport controls). Integration changes are limited to
`pipeline/dual_runner.py`, `dual_viewer.py` and an optional asset-build bypass in
`runner.perceive`. The default asset/grasp path is preserved.

`tests/test_placement_selection.py` covers P00, zero/one/multiple planes, failures,
ordering, no repeated inference, original-snapshot transforms, tilted-table
conventions, serialization, confirmation history and the execution boundary.
`tests/test_placement_web.py` drives the real 8089 server through its installed
Viser WebSocket protocol, including P01 → P02 → P00 and confirmation. Its optional
integration case launches the real Placement Only CLI, calls local UOP once and
checks the resulting displayed geometry/artifact; ROS imports are forbidden in
that offline subprocess.

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src:tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ADA_PLACEMENT_OBJECT_ROOT="$PWD/output/dual_live/native_both_live_20261123_173430/object" \
  .conda/bin/python -m unittest test_placement_selection test_placement_web -v
```

All **15 focused tests passed**, including the real UOP/WebSocket integration.
Separate retained smoke evidence is under
`output/placement_selection/validation_20261124T214200Z/`: both the saved real
camera capture and the previously tested milk-carton synthetic partial cloud
produced P00 plus two valid candidates. The two UOP inlier counts were **49/119**
for the real capture and **28/626** for the prior partial cloud. The 50-point rule
applies to the instance before fitting, not its inlier count. These are software
integration results, not physical placement outcomes.

The full suite passed: **619 tests, 88 skipped, no failures/errors**. Ruff lint,
`compileall`, changed-file formatting and `git diff --check` passed. Whole-tree
formatting has eight pre-existing differences outside this change. An initial test
imported Pinocchio after numerical operations and crashed in its native extension;
loading the existing pipeline dependencies at test-module import time, as the real
CLI does, resolved the test import-order issue without replacing dependencies.

No fresh live camera capture or hardware motion was performed during validation.
