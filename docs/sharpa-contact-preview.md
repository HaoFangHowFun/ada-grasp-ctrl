# Right Sharpa contact point, normal and F6 preview

This extends the existing `ada_grasp_ctrl.tactile_preview` entrypoint on
`http://localhost:8089`. It does not start a driver, connect an SDK device, issue
commands, tare, or change any ROS topics/QoS. The existing URDF, FK, force readouts,
axes, freshness checks and Python 3.12/3.10 process separation are retained.

The reusable [tactile adapter](sharpa-tactile-adapter.md) now owns the derived
contact observations and shared hand FK. The viewer consumes those results;
its diagnostic sign flip affects rendering only.

## What appears

- Red marker: the first current contact row's mapped position.
- Blue arrow: the map's normalized surface normal.
- Green arrow: the fingertip-level F6 resultant, using the selected force channels.

Both arrows start exactly at the red marker. F6 is not allocated among contacts,
and moving its graphical origin does not transfer a wrench or establish a physical
point of application. The operator confirmed that the displayed force direction
is correct and that Newton (N) units are reasonable in the tested conditions,
supporting the retained working indices `0,1,2`. This is a manual observation;
calibrated magnitude and full per-axis sign/alignment remain unverified. See the
[operator follow-up](sharpa-contact-preview-validation-2026-11-12.md#operator-follow-up-force-direction-and-newton-units).
Extra F6 channels
and every column of CONTACT_POINT row 0 remain raw diagnostics. In particular,
CONTACT_POINT's third field is not labeled depth, pressure or force.

Missing, empty, invalid or stale contact hides **all three**. Valid contact with
near-zero F6 retains red/blue and hides green. Invalid/stale F6, transport or pose
also hides the world objects. There is no fallback to an elastomer-origin arrow
or a contact from an earlier frame. Numeric F6 diagnostics remain available even
when no contact is rendered. Multiple contacts report N but select row 0 only.

Local mode displays the selected finger's `p_E`, `n_E`, `f_E` together, offset by a
labeled local triad's display origin. It does not claim a world pose. World mode
uses the existing hand-flange-fixed viewer world W, not a laboratory/arm world.

## Geometry verification gate

**The installed HB1 maps currently provide an UNVERIFIED diagnostic candidate.**
The operator's force feedback does not verify contact position, geometry-map XYZ
units, frame alignment or the blue surface normal.
Their default display is disabled. The checkbox
`Show UNVERIFIED geometry (diagnostic only)` enables explicitly labeled candidates;
a prominent warning and labels beside the contact stay visible. Enabling this switch
does not change verification status. The equivalent explicit CLI opt-in is
`--allow-unverified-geometry`.

The active profile, raw units, `meters_per_unit`, UV mapping, `T_E_S` and evidence
are visible in the Geometry source folder. All three contact graphics require
either documented verification evidence supplied in configuration or diagnostic
opt-in. An unavailable map remains an error even with diagnostic opt-in.

### Actual source and offline access

The local SDK is `/opt/sharpa-wave-sdk`, version 5.0.10, build 5.0.10.6. Header
`include/SharpaWaveSDK.h` documents `deform_map_uv(channel, row, col, tactile_pn)`
as returning `x,y,z,nx,ny,nz` in a finger coordinate system, but does not establish
the XYZ unit or equivalence to the URDF elastomer frame. Its no-hand default can
select HA4, so calling it without an explicit hardware/product decision would be
unsafe for interpretation. The SDK constructors' offline behavior was not proven;
**this implementation never constructs or imports an SDK object**.

Instead, inspection found named static maps embedded in the installed library:

```text
/opt/sharpa-wave-sdk/lib/libsharpa-wave-sdk.so
sha256: 841951806d74dc2adbcbe443f4a21d8668fb2a19ba315b5af36708a26ae7e5dc
tactile_map_general_hb1_point_begin / _end
tactile_map_general_hb1_normal_begin / _end
tactile_map_thumb_hb1_point_begin / _end
tactile_map_thumb_hb1_normal_begin / _end
```

Each byte extent is 691200 bytes: 240 × 240 × 3 little-endian float32 values.
`geometry.sdk_hb1_arrays()` reads the file's ELF symbol and section tables as bytes,
without `dlopen`, native execution or network access. The entire binary hash must
match this audited build. A different/stripped SDK fails closed instead of guessing
offsets. This is a local static-asset adapter, not a stable public SDK API. No SDK
maps or replacement URDF assets are added to Git; the original installation and
its license/provenance remain unchanged.

The general map is shared by right channels 0–3 (pinky/ring/middle/index), and the
thumb map is used by channel 4. This agrees with the SDK channel table and the
inspected lookup's thumb-channel selection (4/9). The right-hand URDF references
HB1 four-finger and thumb meshes, but the connected hardware's product variant
has not been independently established by this preview; that remains a gate.

### Pixel, scale and frame evidence

Official `sample/python/tactile_sample_common.py` draws CONTACT_POINT entries 0/1
as image x/y. Its normal/resultant/distributed display uses scale 1 on a 240-square
tile; SE contacts are scaled by 4 from a 60-square native grid. `include/touch.h`
documents lookup row/col in `[0,240)` for non-SE and `[0,60)` for SE. The installed
lookup selects tensor entries by row then column. A 60×60 DIST_FORCE block alone
does not identify the contact coordinate domain or prove SE hardware.

The current **candidate**, stated explicitly rather than silently assumed, is:

```text
(u,v) = contact row's image (x,y)
col = u, row = v in a 240×240 HB1 lookup
SDK XYZ candidate unit: mm; meters_per_unit = 0.001
T_E_S candidate: identity 4×4, after converting position to meters
normal: stored map direction, normalized; no automatic outward sign flip
```

Only coordinates inside pixel-center bounds `[0,239]` are accepted. The nearest
integer is `floor(value + 0.5)` (ties upward). Bounds are checked **before** rounding;
there is no clamping, mirroring, axis swap or rescaling unless explicitly supplied
by a profile. The selected pixel must pass the finite/nonzero-normal validity mask.
Holes are rejected rather than searching across the pad for another point.

A static comparison with official elastomer meshes supports the candidate scale
and orientation: sampled median nearest-surface distances are about 0.052 mm for
the general map and 0.059 mm for thumb, and median normal/mesh-normal dot products
exceed 0.999. Some edge distances and normal signs disagree, so these statistics
do not establish calibrated alignment or outward normals everywhere. Full evidence
and limits are in [the validation report](sharpa-contact-preview-validation-2026-11-12.md).

## Atomic contact adapter and transforms

The installed Phase 1A `dual_sharpa_wave/msg/TactileFrame` contains `f6` and
`contact_points` from one callback. No legacy readers or per-block caches are used.
One snapshot carries their common SDK ID, session, receipt sequence/stamps, F6,
contact status, original shape, row count and all raw columns of row 0.

`first_contact()` checks status, nonnegative integral dimensions, bounded product,
flattened length and finiteness of the **full** contact payload. The last dimension
is columns; leading dimensions enumerate rows in C order, and a one-dimensional
K-vector is one row. Nonempty rows need at least two columns. Zero-size blocks
become EMPTY with zero rows. Shape is retained without squeezing away count.
MISSING does not physically prove no contact, but supplies no current point to draw.

Maps load once at startup and are shared per profile. DDS callbacks never process
DIST_FORCE grids. Viser allocates markers/arrows once and updates their geometry
atomically at the existing 20 FPS loop; no per-frame URDF rebuild occurs.

```text
p_E = R_E_S (meters_per_unit * p_S) + t_E_S
n_E = normalize(R_E_S n_S)
p_W = R_W_E p_E + t_W_E
n_W = normalize(R_W_E n_E)
f_W = R_W_E f_E
```

Only points are translated. Force magnitude is preserved. All transforms must be
finite, proper rigid transforms. Existing stale/skew thresholds and clock choices
are unchanged; SDK timestamps never become wall-clock freshness timestamps.

## Launch and manual check

Keep the existing operator-started `read_only:=true` driver running. Use the same
ROS domain/discovery environment; do not start another driver. For full driver/SDK
setup, see [the existing startup instructions](sharpa-tactile-preview.md).

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"

ros2 param get /sharpa/right_hand/hand_node read_only
ros2 topic info -v /sharpa/right_hand/tactile/index/frame
.conda/bin/python -m ada_grasp_ctrl.tactile_preview
```

Open <http://localhost:8089>. Geometry is hidden until verified/configured or the
diagnostic checkbox is enabled. Observe idle with no contact objects; then gently
touch right index at center and off-center locations. Record whether the red marker
tracks position, whether blue is outward, active force indices/sign-flip state, and
whether all objects disappear on release. Test other fingers only when the operator
chooses. A live readout or static mesh comparison alone is not a passed visual test.

For an opt-in compact right-index capture, in another sourced terminal:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
/usr/bin/python3 -m ada_grasp_ctrl.tactile_preview.capture \
  --finger index --duration 30 --output /tmp/right-index-contact-new.jsonl
```

This recorder subscribes only, refuses an existing output path, records raw F6 and
contact row 0 plus original identity/timestamps and the latest joint stamp, and
reports diagnostic `p_S/n_S/p_E/n_E` or a geometry error. It does not fabricate a
world pose or claim the records' candidate geometry is verified. Use another
filename for each run. Ctrl-C stops capture or preview, not the hardware driver.

## Explicit external geometry profiles

`--geometry-config /absolute/path/geometry.json` accepts independently prepared
NPY XYZ, normal and boolean-mask arrays. `allow_pickle=False` is always used. Paths
are relative to the JSON directory; each profile loads once and each finger maps
explicitly to a profile. A failed/missing profile disables only its assigned fingers.

```json
{
  "schema_version": 1,
  "fingers": {"index": "index-profile"},
  "profiles": {
    "index-profile": {
      "xyz": "xyz.npy",
      "normals": "normals.npy",
      "valid_mask": "valid_mask.npy",
      "meters_per_unit": 0.001,
      "T_E_S": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "uv_to_col_row": [[1,0,0],[0,1,0]],
      "verified": false,
      "evidence": {},
      "source": "Replace with the actual asset provenance",
      "units": "candidate mm; verify before marking verified"
    }
  }
}
```

The filenames above illustrate a config schema; those files are not supplied or
claimed to exist. `T_E_S` translation is in meters. For `verified: true`, nonempty
documented evidence is required for `pixel_mapping`, `units`, `alignment`,
`normal_convention` and `finger_model`. This is a user-supplied assertion, labeled
`VERIFIED BY CONFIG EVIDENCE`; setting a boolean is not independent calibration.

## Tests

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ADA_TACTILE_DDS_TEST=1 .conda/bin/python -m unittest discover -s tests -p 'test_tactile*.py' -v
```

The DDS fixture alone uses localhost domain 198 with synthetic telemetry. Tests
cover first-row selection, unknown extra columns, multiple contacts, malformed
shapes, NaN/Inf, contact disappearance, same-frame replacement, rounding/bounds,
mask holes, finger types, unverified/missing maps, two nonidentity transforms,
co-origin arrows, zero force, stale pose/data, initial assembly and existing F6
behavior. Detailed command results and manual observations are recorded separately.
