# Minimal Sharpa right-to-left mirroring

This isolated utility reuses `HUGS-Sharpa-Assets/urdf/{right,left}_sharpa_wave_with_flange.urdf`.
It does not change the existing right-hand inference, selector, robot assets or execution pipeline.
The initial repository worktree was clean. Existing standalone assets already retain all fixed
flange, wrist, palm and fingertip frames, so MuJoCo FK can be reused directly without a new
URDF converter or Pinocchio dependency.

## Result and conventions

For the explicit 22-joint order in `src/ada_grasp_ctrl/config/hand/sharpa_mirror.yaml`,
`q_left = q_right`: identity permutation, all positive signs, zero offsets. Angles are radians.
The order is thumb (5), index (4), middle (4), ring (4), pinky (5).

The local right-to-left reflection is `S_hand = diag(1, -1, 1)`.
Flange origins coincide; flange-to-wrist translation is `(0, 0, 0.0005)` meters and
wrist-to-palm translation is `(0, 0, 0.029)` meters, with identity rotations on both sides.
The palm landmark means the URDF palm-frame origin, not an estimated mesh centroid.
For an arbitrary world plane, the API computes `R_left = S_world @ R_right @ S_hand.T`
and reflects the position about the supplied plane point. The result is a proper rotation.

The pose API uses 4x4 homogeneous **world-from-flange** or **world-from-wrist** matrices.
Existing DexLearn flange poses use XYZ/WXYZ; convert WXYZ explicitly before using this API.
Arm mounting transforms and arbitrary TCP offsets are not calibrated or included: convert
an arm/TCP pose to the named flange frame first, and apply the left mounting transform later.
Do not send the output directly to hardware.

## Minimal derivation and validation

The tool checks all 24 improper signed-axis frame candidates using paired anatomical link
origins. It scores every topology-depth-compatible joint and both signs against independent
right-FK reflection at valid ±0.1/±0.25 rad perturbations. A global assignment establishes the
permutation, followed by joint-ancestry and full-range limit checks. Names label anatomical
landmarks; the joint signs and assignment are established by motion, not name replacement.
The report includes each winning score, its next-best alternative and one-sided probes.
No optimizer or offset fitting is needed. Random validation poses are held out from derivation.

Final run: `output/sharpa_mirror/minimal_v2/` (seed 42):

| Check | Result |
| --- | --- |
| Total cases | 220 |
| Individual-joint probes | 63, covering all 22 joints |
| Individual-joint boundary cases | 44 |
| Random held-out configurations | 100 |
| Real prepared DexLearn candidates | 4 × pregrasp/grasp/squeeze = 12 |
| Invalid joint configurations | 0 |
| Maximum link-origin / fingertip landmark error | 0.001526 mm |
| Maximum error under 32 arbitrary world poses/planes | 0.001444 mm |
| Pose round-trip maximum orientation error | 6.6e-15 degrees |
| Held-out mean nearest-sample surface distance | 1.578 mm |
| Held-out maximum nearest-sample surface distance | 15.162 mm |

The kinematic result is approximate equivalence, with residuals consistent with rounded URDF
angles. These tiny errors do not justify adding optimization. There is no invented pass
threshold for declaring exact mesh equivalence.

Surface diagnostics use only 240 independently sampled points per visual mesh, fixed across
poses. Their distances have a substantial sampling floor, especially on the large palm mesh;
the maximum is not a measured mechanical offset. Some left/right source mesh hashes are
identical and are listed in the report. The two hands are still sampled and transformed
independently; no reused reference cloud or assumed vertex correspondence produces artificial
zero errors. Exact nearest-triangle distances and detailed mesh asymmetry analysis are outside
this minimal implementation.

The 19 MuJoCo screenshots use surface points: canonical right/reference/left/overlay,
five finger examples, two random poses, five worst held-out poses and all three phases of
one real DexLearn candidate. Reference points are cyan; actual left points are orange.
Larger markers identify wrist, palm and fingertips. Point rendering avoids improper body
quaternions and reflected triangle winding issues. It is an offscreen diagnostic, not an
interactive mesh viewer.

`report.json` contains every configuration and fingertip error, worst-five configurations,
probe evidence, sampling limitations and screenshot paths. CSV gives compact per-case metrics.
`run_manifest.yaml` records source hashes, arguments, asset hashes and candidate hashes.
The stored mapping contains the actual URDF/visual asset hashes and explicit joint limits.
The inference API does not load assets or rerun derivation; rederive if those assets change.

## Reproduce

From the repository root, use a fresh output directory each time:

```bash
MUJOCO_GL=egl PYTHONPATH=src .conda/bin/python script/validate_sharpa_mirroring.py \
  --models ../HUGS-Sharpa-Assets/urdf \
  --output-root output/sharpa_mirror/my_new_run \
  --candidates output/regression/sharpa-right-100k-tabletop-smoke-v1/prepared_grasps

PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl .conda/bin/python \
  -m unittest discover -s tests -p test_sharpa_mirror.py -v
```

Use `--no-render` if EGL is unavailable; omit `--candidates` if real prepared records are absent.
Candidate files must be trusted local NPY dictionaries with explicit `joint_names`; pickle
loading is deliberately limited to that explicitly supplied directory. Inputs are prepared
DexLearn records, including the existing pregrasp extrapolation, not fresh inference.

```python
import numpy as np
from ada_grasp_ctrl.sharpa_mirror import SharpaHandMirror

mirror = SharpaHandMirror()
q_left = mirror.right_to_left(np.zeros(22))
left_pose = mirror.mirror_wrist_pose(np.eye(4), normal=[1, 0, 0], point=[0, 0, 0])
```

Four API regression tests cover explicit order, range boundaries, malformed joints, arbitrary
world-plane geometry, proper rotations, round trips and malformed poses. Compile checks and
whole-repository Ruff lint pass. Formatting checks pass for the new files; the whole-repository
format check reports eight pre-existing files, which were left unchanged.

The mapping is ready for an **offline DexLearn integration step**. Object/cloud reflection,
world-scene integration, left-arm mounting calibration, collision checking and physical grasp
success remain unvalidated. No existing selector or robot execution code was changed.
