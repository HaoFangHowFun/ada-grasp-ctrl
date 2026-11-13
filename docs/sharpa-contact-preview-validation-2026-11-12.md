# Contact / normal / F6 preview verification — 2026-11-12

## Outcome and handoff

The existing right-hand preview has been extended with a red first-contact marker,
blue normal and green F6 resultant at the same origin. The software and synthetic
tests pass. Live operator-assisted right-index captures confirm current contact
delivery, changing UV, one/two-contact frames, successful candidate lookup and
return to MISSING after release. **Physical geometry alignment and outward-normal
direction remain UNVERIFIED**: the operator initially deferred browser observation
and has not subsequently confirmed marker placement or surface-normal direction.
No other finger's physical contact placement is claimed.

At the operator's final request, the `ada-contact-preview` tmux viewer/subscriber
was stopped and **8089 was confirmed closed**. The existing ROS driver was not
stopped or modified. No commit/push, dependency installation, upstream changes,
hardware commands or automatic calibration were performed.

Local checkout: `/home/hao-fang/hugs-workspace/ada-grasp-ctrl`, branch `main`, HEAD
`539b165`. Existing unrelated worktree changes were preserved. See the
[operation/geometry guide](sharpa-contact-preview.md) for exact launch commands,
profile schema, diagnostic opt-in, source details and unresolved assumptions.

## Operator follow-up: force direction and Newton units

After the handoff, the operator reported: 「力的方向沒問題 單位是牛頓也合理」
(the force direction is correct, and Newton units are reasonable). Record this as
an operator-observed pass for force direction in the tested conditions and support
for interpreting the displayed force in newtons (N). Retain the previously agreed
mapping `Fx=F6[0], Fy=F6[1], Fz=F6[2]`.

This follow-up supplies no calibrated force reference or per-finger/per-axis
measurements. It does not verify contact-marker placement, map XYZ units,
`T_E_S`, or the blue surface normal. The geometry profile remains UNVERIFIED;
its diagnostic opt-in and existing software labels are unchanged.

## Implementation paths

- Updated `src/ada_grasp_ctrl/tactile_preview/data.py`: same-frame first contact,
  original shape/status/count, finite/length validation, all raw columns of row 0.
- Updated `src/ada_grasp_ctrl/tactile_preview/viewer.py`: co-origin graphics,
  visibility gates, local/world transforms, profile diagnostics and atomic updates.
- Updated `src/ada_grasp_ctrl/tactile_preview/__main__.py`: geometry configuration,
  diagnostic opt-in, and port preflight tolerant of TIME_WAIT after shutdown.
- Added `src/ada_grasp_ctrl/tactile_preview/geometry.py`: immutable static profiles,
  explicit transforms, nearest-pixel lookup, external NPY/config support and an
  exact-hash-gated reader for the installed SDK's embedded HB1 arrays.
- Added `src/ada_grasp_ctrl/tactile_preview/capture.py`: opt-in bounded compact ROS
  recorder, exclusive output creation and file I/O outside subscriber callbacks.
- Updated `tests/test_tactile_preview.py`, `tests/tactile_ros_fixture.py`; added
  `tests/test_tactile_contact_geometry.py`.
- Added `docs/sharpa-contact-preview.md` and this report; updated the existing
  preview guide and historical session summary to link to current behavior.

The ROS reader's topics/QoS and process separation are unchanged. The extended
snapshot passes through its existing adapter. No legacy-topic synchronization,
deformation processing, SDK device object or control pipeline was introduced.

## Live environment and captures

Actual discovery: ROS Jazzy, domain 87, SUBNET discovery; right-index frame publisher
count 1, BEST_EFFORT/VOLATILE. `ros2 param get /sharpa/right_hand/hand_node read_only`
returned True, including the final read-only audit. The running driver executable
came from `/tmp/sharpa_phase1a_impl/install`; the normally sourced interface came
from `/home/hao-fang/ws_fanuc/install`. The observed topic type/hash matched
`dual_sharpa_wave/msg/TactileFrame`. Installed interfaces and Phase 1A contract were
inspected before coding. No driver restart or rebuild was performed.

Baseline eight-second capture: 1,058 right-hand frames, zero reported validation
errors, valid F6 throughout, all CONTACT_POINT MISSING. Measured rates:

| Finger | Frames | Observed Hz |
| --- | ---: | ---: |
| pinky | 211 | 28.13 |
| ring | 210 | 27.99 |
| middle | 209 | 27.73 |
| index | 213 | 28.39 |
| thumb | 215 | 28.66 |

Baseline artifact: `/tmp/ada_contact_initial_capture_20261112.json`. The installed
`ros2 run ... capture_tactile_frames.py` alias was unavailable, so the existing
source script was run directly with sourced system Python. Subsequent captures
used the new, documented `ada_grasp_ctrl.tactile_preview.capture` module.

The operator then announced touching the right index and later releasing it:

| Capture | Total | VALID contact | MISSING contact | Geometry errors |
| --- | ---: | ---: | ---: | ---: |
| 180 s initial manual window | 5085 | 694 | 4391 | 0 |
| 90 s window started at touch announcement | 2524 | 181 | 2343 | 0 |

**These windows overlap; do not add their counts.** All recorded F6 blocks were
VALID. Both captures included CONTACT_POINT shapes `[1,3]` and `[2,3]`; only row 0
was selected, with total count retained. First-window UV ranged from 62.578125 to
202.96875 in u, and 39.814453125 to 191.1328125 in v. This is evidence of changing
image coordinates, not evidence of physically correct 3D placement.

After release, the final runs contained respectively 377 MISSING frames over
13.16 s and 1776 MISSING frames over 63.24 s. Software tests verify that this state
hides all contact graphics. The browser was inspected afterward: correctly
assembled hand, LIVE hand FK, five MISSING contact statuses, and
`UNVERIFIED GEOMETRY — HIDDEN`; no contact objects were displayed. Screenshot:
`/tmp/ada-contact-idle-preview.png`; browser text: `/tmp/ada-contact-idle-browser.txt`.

Actual same-frame example, SDK frame ID 41024:

```text
CONTACT_POINT row 0 = [146.8359375, 134.1796875, 14.9921875]
selected pixel      = row 134, col 147
F6 raw              = [-0.02685546875, 0.90478515625, 10.84375,
                       -0.179931640625, -0.017791748046875, -0.0017414093017578125]
candidate p_E (m)   = [0.0015695039, -0.0166018314, -0.0060232024]
candidate n_E       = [0.1525959183, -0.0953974431, -0.9836736316]
geometry status     = UNVERIFIED DIAGNOSTIC
```

The third contact value and last three F6 channels are raw unnamed quantities.
No depth/pressure/torque interpretation is assigned. The capture's geometry uses
explicit candidate HB1 240×240 maps, `col=u,row=v`, 0.001 meters per native unit,
identity `T_E_S`, and unchanged map-normal sign. No world-pose claim is made by
the capture module. Later recorder metadata also includes the full active profile.

Evidence files (local `/tmp`, not committed):

- `/tmp/ada_contact_index_manual_20261112.jsonl`, SHA256
  `35bd64d26ffcacf0c1141fdd00d2f3dea8b3a9f28eacf12c1a8f7b935627559a`.
- `/tmp/ada_contact_index_manual_touch_20261112.jsonl`, SHA256
  `9698450ac5ac2b62a98cce9de9394e888213898958f465af7ea05ebc2c841354`.
- `/tmp/ada_contact_manual_summary.json` contains counts, ranges and examples.

## Static geometry evidence and limits

The SDK library SHA256 is
`841951806d74dc2adbcbe443f4a21d8668fb2a19ba315b5af36708a26ae7e5dc`.
Its general/thumb point and normal symbol extents are each 691200 bytes, interpreted
as 240×240×3 float32 arrays only for that exact audited build. The byte reader never
executes SDK code. Valid masks contain 46,119 general and 45,670 thumb pixels based
on finite data and nonzero normals. The raw source files are not copied into Git.

Comparing every 32nd valid point, with candidate mm-to-m conversion and identity
alignment, to the official URDF elastomer meshes yielded:

| Profile | Sampled points | Median surface distance | 95th percentile | Maximum | Median normal dot mesh normal |
| --- | ---: | ---: | ---: | ---: | ---: |
| general | 1442 | 0.052 mm | 1.360 mm | 2.796 mm | 0.99950 |
| thumb | 1428 | 0.059 mm | 0.645 mm | 1.275 mm | 0.99980 |

Some normal dot products were negative at edges (minima about -0.256/-0.535).
The comparison supports a plausible candidate, but does not prove exact SDK/URDF
origin equivalence, outward direction everywhere, connected hardware variant or
contact pixel convention. No rigid fit, hidden sign correction or asset change was
used to improve agreement. The detailed bounds, mesh hashes and metrics are in
`/tmp/ada_contact_geometry_audit.json`. User-supplied verified profiles require
explicit evidence for pixel mapping, units, alignment, normals and finger model.

## Automated results

| Check | Actual result |
| --- | --- |
| Focused suite including synthetic DDS and 8089 start/stop | 27 passed, 0 skipped; 5.548 s |
| Final geometry validation checks | 9 passed after numeric/mask type guards |
| Full canonical unittest discovery | 406 total: 341 passed, 65 skipped; 31.322 s |
| Compile `src tests` | Passed |
| Canonical Ruff check `src tests script` | Passed |
| Formatting of changed Python files | Passed |
| Whole-repo format check | Same eight pre-existing files need formatting; untouched |
| LEAP Tac3D headless quick example | Exit 0; 100 outputs, 93 completed, 7 invalid initialization, 0 execution errors/degraded |
| Live HTTP/node inspection before requested stop | HTTP 200; six intended subscriptions, only standard parameter-events publisher, no command publishers/action clients |

The full suite skips environment-dependent existing tests and the opt-in DDS test;
its 8089 smoke was also skipped while the port was temporarily unavailable. The
separate final focused run passed the 8089 check. Port preflight now uses
SO_REUSEADDR so shutdown TIME_WAIT does not incorrectly look like a live listener.
An actual occupied listener is still rejected.

Reproduction:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ADA_TACTILE_DDS_TEST=1 .conda/bin/python -m unittest discover -s tests -p 'test_tactile*.py' -v

PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  .conda/bin/python -m unittest discover -s tests -v
PYTHONPATH=src PYTHON_BIN=.conda/bin/python \
  ADA_GRASP_CTRL_EXAMPLE_BASE=/tmp/ada_contact_quick_example \
  bash script/run_example.sh leap_tac3d quick
```

The quick example's fresh root was
`/tmp/ada_contact_quick_example/leap_tac3d/20261113T003358Z-403434`:
87 successful lifts, 6 failures, 7 invalid initializations. This is a simulation
smoke result, not a new golden baseline. Logs:
`/tmp/ada_contact_focused_tests_final.log`, `/tmp/ada_contact_full_tests.log`,
`/tmp/ada_contact_ruff_check.log`, `/tmp/ada_contact_ruff_format.log`,
`/tmp/ada_contact_quick_example.log`.

## Remaining manual checks

The initial operator request was to defer browser observation and keep geometry
unverified. The later force-direction/Newton-unit feedback above updates the force
observation only. The next optional manual check is to enable diagnostic geometry and
compare right-index center/off-center marker location, outward blue normal and
hide-on-release against physical touch. Other fingers, all directions, SE/model
selection, calibrated S-to-E alignment, calibrated force magnitude, full per-axis
force-sign certification and any control
integration remain outside the verified results of this session.
