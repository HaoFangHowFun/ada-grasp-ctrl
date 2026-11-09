# Right CRX source for the placement preview

`arm_source.urdf` is the right-only expansion of FANUC's CRX-5iA macro, with the
right-base origin read from the local `dual_crx_control/urdf/dual_crx.urdf.xacro`.
It was expanded offline with the installed xacro package; no ROS node or hardware
connection was started. Source paths, SHA-256 hashes and actual Git revisions are
recorded in `source_manifest.json`. The FANUC macro records
SPDX-FileCopyrightText 2025-2026 FANUC America Corporation / FANUC CORPORATION and
SPDX-License-Identifier Apache-2.0; the existing license text is retained at
`../crx5ia_sharpa_wave_left/licenses/`.

The reference right base is XYZ `[0.03058936, -0.91156264, 0.00873476]` meters and
RPY `[0.00994493, -0.01029918, -1.54694055]` radians. The hand mount is from
`bimanual_manipulation/config/hand_mounts.yaml`: right ee_mount to right hand
flange, XYZ zero, RPY `[3.14159265359, -1.57079665359, 0]`. This is explicitly a
mock preview placeholder, not a measured hand calibration.

The runtime builder uses the independently generated right arm kinematics. It
reuses the existing FANUC visual/collision mesh files from the left asset only
after checking each mesh origin; those files represent the same arm mechanics.
Their conversion and provenance remain documented in that asset's `SOURCE.md`.
The right hand comes exclusively from the frozen right-hand benchmark's MJCF and
URDF, including all 22 named axes, limits, geometry, damping and hand actuators.
No left-hand geometry or axes are renamed into a right hand.

The generated full MJCF/URDF and mesh hashes remain in each new preview output.
The source metadata also pins the inspected DexLearn dataset and export code as
evidence that predictions are source-world flange poses, including restoration
of point-cloud centering before saving. These files are checked for drift before
preparation. Source code inspection does not independently establish a training
split or prove checkpoint ancestry beyond the existing suite contract.

The combined asset's existing redistribution limitations are unchanged. In
particular, `dual_crx_control` has an unresolved license declaration in the
original provenance record. This local simulation derivative does not clear it.
