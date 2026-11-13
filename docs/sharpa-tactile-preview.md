# Right Sharpa tactile live preview (experimental, read only)

Session overview and operator feedback (Traditional Chinese):
[2026-11-12 implementation summary](sharpa-tactile-session-summary-2026-11-12.md).

Current contact-point extension: [contact/normal/F6 guide](sharpa-contact-preview.md).
The display now gates the red contact point, blue normal and green force on a
current valid contact and an explicitly verified or diagnostic geometry profile.
F6 numerics remain available without a contact; no origin fallback arrow is drawn.

This standalone browser tool displays the original right-hand URDF and five F6
force arrows. It tests the hypothesis that selected F6 channels have the axes of
the corresponding `right_<finger>_elastomer` frames. It is not calibration evidence
or part of the grasp/control pipeline. No robot commands, SDK connection, motor
configuration, tare, deformation rendering, or physics
stepping occur in the preview.

## Start and stop

Use the existing local Python 3.10 `.conda` environment for Viser. The entrypoint
starts a separate `/usr/bin/python3` ROS Jazzy/Python 3.12 subscriber and transports
only compact F6/pose snapshots through an owned pipe. No new dependencies are needed.

The operator starts the existing sensing driver separately, with its documented SDK
environment and `read_only:=true`. Do not start a second driver if one is already running:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="/opt/sharpa-wave-sdk/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/sharpa-wave-sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Keep the actual driver's ROS_DOMAIN_ID and network/discovery settings.
ros2 launch dual_sharpa_wave dual_sharpa.launch.py \
  backend:=sharpa_sdk read_only:=true use_rviz:=false
```

In another terminal with the same ROS environment:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
# Prepend the source tree; do not replace ROS's existing PYTHONPATH.
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ros2 topic list -t --no-daemon --spin-time 3
ros2 topic info -v /sharpa/right_hand/tactile/index/frame
ros2 param get /sharpa/right_hand/hand_node read_only
ros2 interface show dual_sharpa_wave/msg/TactileFrame
ros2 interface show dual_sharpa_wave/msg/TactileBlock
.conda/bin/python -m ada_grasp_ctrl.tactile_preview
```

Open <http://localhost:8089>. Ctrl-C stops the viewer and its subscriber child only;
it leaves the driver's lifecycle unchanged. If the ROS environment is not sourced,
the terminal reports an import failure and the browser reports DISCONNECTED.

Keep the terminal running while using the viewer. If port 8089 refuses connections,
check `ss -ltnp 'sport = :8089'`: a stopped preview must be started again. During
this session a detached tmux session was used for persistence, then stopped at the
operator's request; it is not an automatically installed background service.

If `ros2` reports `PackageNotFoundError: ros2cli` after editing `PYTHONPATH`, check
that the `$` characters in `$PWD` and `$PYTHONPATH` were retained. The literal value
`PWD/src:PYTHONPATH` overwrites ROS's import paths. In the Ada preview terminal:

```bash
unset PYTHONPATH
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

This repairs the environment without reinstalling ROS. The driver terminal also
needs the SDK exports above if its environment has been reset.

For a model/HTTP smoke test without any ROS connection:

```bash
PYTHONPATH=src .conda/bin/python -m ada_grasp_ctrl.tactile_preview --offline --run-seconds 5
```

`--host` defaults to `127.0.0.1`; `--port` defaults to `8089`. An occupied port is an
error, rather than an automatic switch. For a remote viewer:

```bash
ssh -N -L 8089:127.0.0.1:8089 hao-fang@128.32.164.23
```

Then open <http://localhost:8089> on the client machine. Substitute the actual SSH
host if needed. The preview itself does not open remote connections.

## Actual data/model contract

The inspected installed type is `dual_sharpa_wave/msg/TactileFrame`, from the
`dual_sharpa_wave` package in `/home/hao-fang/ws_fanuc/install`. Source contract:
`/home/hao-fang/ws_fanuc/src/dual_sharpa_wave_ros2/docs/tactile_phase1a_contract.md`.

| Finger / channel | Atomic topic | URDF force frame |
| --- | --- | --- |
| pinky / 0 | `/sharpa/right_hand/tactile/pinky/frame` | `right_pinky_elastomer` |
| ring / 1 | `/sharpa/right_hand/tactile/ring/frame` | `right_ring_elastomer` |
| middle / 2 | `/sharpa/right_hand/tactile/middle/frame` | `right_middle_elastomer` |
| index / 3 | `/sharpa/right_hand/tactile/index/frame` | `right_index_elastomer` |
| thumb / 4 | `/sharpa/right_hand/tactile/thumb/frame` | `right_thumb_elastomer` |

The only additional subscription is `sensor_msgs/msg/JointState` on
`/sharpa/right_hand/joint_states` (configurable with `--joint-topic`). Full sets of
22 unique official joint names are mapped by name, checked for finite values and
URDF limits, and used in `yourdfpy` FK through the existing Viser stack.

Default model:
`/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave/urdf/right_sharpa_wave_with_flange.urdf`.
Original package mesh URIs resolve inside that package's share directory; no asset
is copied or edited. Override `--package-share` / `--urdf` for another installation
of the same right-hand model. Startup checks all five elastomer frames and the
22-joint/root contract. This does not expand the public supported hand matrix.

Viewer world **W is fixed at `right_hand_flange`**, not the laboratory/CRX world.
The same URDF FK drives both meshes and `p_WE, R_WE`. The force is rotated as
`f_W = R_WE @ f_E`. The contact extension maps the first same-frame contact to `p_E`,
computes `p_W = R_WE @ p_E + p_WE`, and draws force from `p_W` to `p_W + scale * f_W`.
Translation never acts on force direction. URDF coordinates are right-handed,
meters, with renderer Z up and no hidden axis conversion. Arm/world tracking and
TF extrapolation are outside this hand-only preview. Keep the robot stationary
for manual checks; this tool cannot validate force directions in laboratory world.

## Interpretation and controls

Force component order remains unverified. The installed SDK sample labels the
values only `F0` through `F5`; it does not establish `Fx/Fy/Fz/Tx/Ty/Tz`. Default
indices `0,1,2` are explicitly **exploratory**, with a persistent warning. Set
`--force-indices 2,0,1` or use the three browser dropdowns to test another mapping.
Duplicate indices hide arrows and report INVALID mapping. All six values remain
visible; unselected channels are raw values without torque units assigned.

After the 2026-11-12 implementation, the operator reported correct directions in
the read-only test and explicitly confirmed keeping mapping `0,1,2`. This is
operator-observed support for that mapping in the tested conditions, not vendor
certification. Per-finger/per-direction raw records and the sign-flip state were
not supplied; see the [manual record](sharpa-tactile-validation.md). The software
continues to label the mapping/alignment hypothesis explicitly.

After the contact-preview handoff, the operator again confirmed correct force
direction and reported that Newton (N) units are reasonable. The working mapping
remains `0,1,2`; see the [follow-up report](sharpa-contact-preview-validation-2026-11-12.md#operator-follow-up-force-direction-and-newton-units).
The prior pad-center origin report is unchanged. This documentation update records
the manual observation; the norm/length controls retain **presumed N** because no
calibrated reference load was supplied. Geometry alignment and the blue normal
remain unverified. There is no implicit
negation. The sign-flip checkbox is visibly labeled as a visualization-only
diagnostic. The force-on-fingertip interpretation must be compared against the
observed push/reaction relationship, not assumed to have a particular sign.

The contact legend is red point / blue surface normal / green F6 resultant.
A selected elastomer displays local XYZ axes (red/green/blue); the world has its
own triad. World mode draws each valid contact and its two arrows on the hand.
Local mode draws the selected `p_E`, `n_E`, `f_E` together in a separate labeled
triad. Force scale is meters per presumed N; normal length is display-only.
Near-zero force hides green without hiding an otherwise valid point/normal.

## Availability and freshness

Subscriptions use BEST_EFFORT / VOLATILE / KEEP_LAST depth 1, compatible with the
documented Phase 1A publisher. Each finger displays live discovered publisher
QoS/type/node/count; missing, multiple, or wrong-type publishers hide its arrow.
Hardware QoS must still be checked with `ros2 topic info -v` when available.

The adapter checks schema/source/side/finger/channel and F6 status, shape, length,
and finite values. MISSING, EMPTY, INVALID and STALE remain distinct; malformed
topic identity immediately invalidates its cached arrow. CONTACT_POINT is now
validated in the same frame and row 0 is copied with original shape/count/status;
its absence hides contact graphics but preserves valid F6 numeric readouts.
DIST_FORCE is not consulted or copied. ROS must deserialize the atomic message,
including its grid, but the preview does not decode/render/copy that grid in its callback hot path.

The browser shows raw F6, selected `f_E`, `f_W`, norm, SDK ID and raw representation,
original SDK timestamp/unit/epoch, session/sequence, ROS receipt stamp, source frame,
measured recent arrival frequency, availability and block/identity errors. ROS
callbacks update bounded latest snapshots; a separate loop emits compact data at
20 Hz, and the viewer refreshes at 20 FPS. Neither is a guaranteed sensor rate.

By default, freshness is **local monotonic time since subscriber arrival**. The
browser explicitly states that source age is unverified. It cannot detect an old
measurement delivered for the first time using this clock alone. The SDK timestamp
is never interpreted as wall time. If driver/subscriber ROS clocks are verified
synchronized (e.g. both use system ROS time on the same host), start with
`--receipt-clock-synchronized` to also reject excessive ROS receipt age, zero
stamps, or negative clock deltas. This measures host-callback freshness, not
sensor-acquisition latency. Worker/parent share one host's monotonic clock.

The default stale limit is 0.5 s (`--stale-after`); maximum force/pose stamp skew is
0.15 s (`--pose-skew`). With unverified source clocks, skew uses local arrivals and
cannot establish acquisition synchrony. Missing/stale/invalid/skewed joint data
retains a labeled static/uncertain model and hides world arrows. Local mode still
permits inspecting a fresh mapped contact and `f_E` without claiming world validation. Disconnected or
stale tactile data never retains a live arrow. No joint extrapolation is performed.

## Tests and manual validation

```bash
# Existing viewer environment, no ROS needed (DDS test skipped):
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  .conda/bin/python -m unittest discover -s tests -p test_tactile_preview.py -v

# Optional full DDS -> adapter -> viewer synthetic integration:
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ADA_TACTILE_DDS_TEST=1 .conda/bin/python -m unittest discover -s tests -p test_tactile_preview.py -v
```

The synthetic fixture/reader children alone use isolated localhost domain 198; the
hardware driver's domain/configuration is never changed. The fixture creates only
tactile and joint-state telemetry publishers, never command publishers. It is
gated against accidental use outside that test environment. Tests cover rotations,
selection/sign/norm/zero, malformed/missing/stale data, identity, FK, actual HTTP
start/stop on 8089, five atomic DDS streams with joint states, and disconnection.

Before touching fingertips, verify the driver reports `read_only=true`, inspect
preview node endpoints (`ros2 node info /ada_tactile_preview`), confirm no command
publisher exists, and check idle values/finger labels. For each finger apply an
approximately normal push, then two tangential pushes while the robot remains
stationary. Record raw channels/signs and arrow direction without changing signs
to make the hypothesis pass. CONTACT_POINT availability is irrelevant to this test.

Use [the manual validation form](sharpa-tactile-validation.md). Software tests do
not establish units, component order, axis alignment, contact thresholds, or
five-finger hardware validation.
