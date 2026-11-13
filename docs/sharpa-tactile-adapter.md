# Right Sharpa tactile observation adapter

`TactileAdapter` converts the existing atomic-frame snapshots into immutable
`TactileContactObservation` values. The 8089 preview consumes these same results;
it no longer implements its own UV lookup or contact/force transformation.
The sensing API imports neither Viser, ROS nor a simulator, opens no connection,
and never owns an SDK device. The optional CLI reuses the existing ROS worker.

## Architecture and existing contracts

```text
Existing read-only driver -> existing ros_reader / LatestFrames -> TactileAdapter
  TactileFrame (same-callback F6 + contact row 0) + latest JointState
  -> existing GeometryCatalog + shared yourdfpy hand FK
  -> immutable per-finger observations and statuses
      -> Viser preview
      -> read-only consumer (no control integration)
```

The audit found no reusable hardware tactile adapter. The former conversion lived
in `tactile_preview/viewer.py`; it now lives in `tactile_preview/adapter.py`.
URDF loading and joint validation moved to `tactile_preview/kinematics.py`, with
the original viewer import names retained for compatibility. Preview and adapter
share the same hand model. The adapter validates and updates its FK; Viser renders
that configuration. Consumers must not concurrently mutate the shared model.

There is already a simulation/controller `ada_grasp_ctrl.contact.ContactObservation`.
It requires full object-oriented six-component wrenches, object identities and
contact frames. The new `TactileContactObservation` is deliberately a separate
sensing type: no zero torques, object identities or controller-ready contacts are
invented. Existing controller contracts and MuJoCo adapters remain unchanged.

The ROS reader still subscribes to five
`/sharpa/right_hand/tactile/{finger}/frame` topics and
`/sharpa/right_hand/joint_states`, with BEST_EFFORT/VOLATILE/KEEP_LAST depth 1.
Frame identity is checked by the existing `normalize_frame()` ingress. This API
accepts its normalized `Frame` values or the reader's JSON-compatible dictionaries,
not arbitrary ROS messages. It rechecks normalized shapes, finite values and
finger keys. No second subscriber path or legacy-topic synchronization is added.

## Importable API

```python
from ada_grasp_ctrl.tactile_preview import TactileAdapter, TactileContactObservation
from ada_grasp_ctrl.tactile_preview.geometry import GeometryCatalog
from ada_grasp_ctrl.tactile_preview.kinematics import load_hand

# urdf_path and package_share are explicit pathlib.Path values for the installed model.
hand = load_hand(urdf_path, package_share, load_meshes=False)
adapter = TactileAdapter(
    hand,
    GeometryCatalog.installed_candidate(),
    stale_after=0.5,
    pose_skew=0.15,
    receipt_clock_synchronized=False,
)

# packet and connected come from the existing reader bridge, not a new SDK connection.
adapter.update(
    tactile_frames=packet.get("frames", {}),
    joint_state=packet.get("joints", {}),
    connected=connected,
    endpoints=packet.get("endpoints", {}),
    errors=packet.get("errors", {}),
    allow_unverified_geometry=False,
)
contacts = adapter.get_contacts()  # all five fingers: observation or None
statuses = adapter.get_statuses()  # separate explicit status for each finger

# Prefer one snapshot when contacts and statuses must share an evaluation instant.
snapshot = adapter.get_snapshot()
for finger, result in snapshot.fingers:
    print(finger, result.status, result.observation)
```

`update()` owns a copy and **replaces the entire latest-per-finger snapshot**.
Omitted fingers clear older contacts; this is not an incremental merge API.
State access is lock-protected and retains only five current frames and one joint
snapshot. `get_snapshot()` and both convenience getters re-evaluate freshness
using the current clock, even when no new callback arrives. Separate getter calls
can straddle the stale threshold; use one snapshot for an atomic decision.
Set `connected=False` immediately when the reader exits or its heartbeat fails.
The existing bridge uses a 0.5-second same-host worker heartbeat.

With live reader packets, pass `endpoints` to retain the one-publisher/type check.
Offline normalized data may omit endpoints (`None`). An empty endpoint dictionary
is an observed lack of publishers and rejects observations.

## Observation semantics

All primary `position`, `normal` and `force` vectors use the observation's explicit
`frame_id`. Public `get_contacts()` yields only **`right_hand_flange`** observations,
never laboratory/world coordinates. Position is in meters; normal is unit length;
force uses the operator-supported Newton interpretation. No force offset or
wrench relocation is inferred from drawing the resultant at contact row 0.

The default force mapping remains `[0,1,2]`, with raw signs preserved and
`force_convention="object_on_finger"` explicitly **provisional**. Other selected
channels remain diagnostic and are recorded in `force_indices`. The preview's
sign-flip control changes rendering only; it never changes the observation's
force or convention. A future force-on-object conversion would require an explicit
negation and convention verification, and is not a controller integration here.

Observation fields retain original row 0 (including unnamed extra columns),
shape/count, raw six-channel F6, source frame, session/receipt sequence, SDK ID/raw
representation, SDK timestamp/unit/epoch, receipt times and warnings. A base-frame
observation also records the joint timestamps used. Geometry provenance includes
profile name, source (including the audited SDK hash for installed candidates),
units/scale, UV affine, `T_E_S`, evidence and verification status. Vectors and nested
provenance use tuples inside frozen dataclasses; there are no mutable NumPy views.

`FingerObservation.local` is an explicitly **elastomer-frame diagnostic**, produced
from the same geometry lookup and force selection. It preserves local-mode viewing
when joints are absent/stale/skewed. It is never returned by `get_contacts()` in
place of a missing hand-base observation. Numeric F6 diagnostics remain available
without a contact; near-zero force remains a valid observation with zero force,
while the UI hides only its green arrow.

## Timing, availability and identity

`timestamp` means subscriber **host monotonic receipt time**, copied from
`Frame.received`. It is neither SDK acquisition time nor a Unix timestamp.
The Python 3.12 reader and Python 3.10 adapter run on the same host/boot.
The SDK timestamp is preserved for diagnostics and never used for freshness.

By default, freshness and tactile/joint skew use local subscriber arrivals.
`receipt_clock_synchronized=True` additionally checks driver-to-subscriber ROS
receipt age, and measures skew between tactile ROS header receipt time and the
joint message header stamp. This requires the operator's explicit clock assertion;
the joint header is not relabeled as sensor acquisition time. Latest joint feedback
is used without interpolation, extrapolation or a historical synchronizer.
Default age is 0.5 seconds and allowed skew is 0.15 seconds; both are configurable.
Missing/incomplete names, nonfinite values, URDF limit violations and malformed
timestamps reject base-frame observations.

| Status | Meaning |
| --- | --- |
| `VALID` | Fresh allowed first contact with usable hand-base FK |
| `NO_DATA` | Finger absent from current snapshot |
| `MISSING` | Source contact block missing; not proof of no physical contact |
| `EMPTY` | Explicit zero-size contact block; kept distinct from missing |
| `INVALID` | Rejected identity, malformed frame or invalid contact block |
| `INVALID_FORCE` | Invalid F6 values/shape or duplicate/out-of-range selected channels |
| `F6/TRANSPORT ...` | F6 unavailable or publisher/type check failed |
| `STALE`, `CLOCK_ERROR`, `DISCONNECTED` | Age/clock/reader availability failure |
| `POSE_MISSING`, `POSE_STALE`, `POSE_INVALID`, `POSE_SKEWED` | No usable base-frame FK; local diagnostic may exist |
| `GEOMETRY_UNAVAILABLE`, `GEOMETRY_UNVERIFIED`, `GEOMETRY_INVALID` | Missing map, closed diagnostic gate or failed lookup |

There is no fabricated `NO_CONTACT` physical conclusion for missing transport data.
The original block status remains in `result.frame` even when a freshness or
transport failure takes precedence. `result.detail` explains geometry/pose errors.

Arrival order follows the Phase 1A contract: SDK IDs may repeat or regress without
guessing resets/wrap. Entire records replace entire records; contact and F6 are
never paired across callbacks. Reading/replaying an identical reader snapshot
does not refresh its original arrival time. Driver duplicate suppression and
identity warnings remain upstream responsibilities.

## Geometry gates and operator evidence

The installed HB1 candidate remains **UNVERIFIED**, hidden by default. Diagnostic
opt-in is explicit in the API/CLI/UI. All existing pixel rounding, bounds and mask
hole rejection are retained. External profiles must name units beginning with
`m` or `mm` (optional annotations and `candidate ` prefix are accepted), with a
matching scale of 1 or 0.001. Unknown units, inconsistent scales, missing source
and malformed evidence are rejected. A configured profile name is retained; a
missing profile disables its finger. No hardware variant is inferred from a grid.

The operator confirmed **force direction looks correct and N units are reasonable**.
Contact-marker placement, blue normal, S-to-E alignment, geometry-map unit scale
and all-finger/per-axis calibration remain unverified. `object_on_finger` is the
requested provisional interpretation; an independently measured push/reaction
sign relationship was not provided. See [geometry and source evidence](sharpa-contact-preview.md).

## Standalone read-only example

No browser is needed. Keep the operator-started read-only driver and matching
ROS domain/discovery environment; this command starts only the existing subscriber
worker and prints five-finger contacts/statuses once per second:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
.conda/bin/python -m ada_grasp_ctrl.tactile_preview.observe \
  --run-seconds 10 --allow-unverified-geometry
```

Omit the diagnostic opt-in to keep unverified contact geometry disabled.
`--geometry-config` accepts the same external profile file as the preview.
An offline smoke command creates no ROS node or HTTP listener:

```bash
PYTHONPATH=src .conda/bin/python -m ada_grasp_ctrl.tactile_preview.observe \
  --offline --run-seconds 1
```

The existing preview launch command and controls are unchanged. A currently
running Python preview must be restarted by its owner to load the new code.
See [adapter validation](sharpa-tactile-adapter-validation-2026-11-12.md) for actual
checks and limits. No MPC, motor, tare or calibration calls are added.
