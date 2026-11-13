# Tactile adapter validation — 2026-11-12 session

## Outcome

The existing read-only tactile path now exposes `TactileAdapter` and immutable
`TactileContactObservation` results. The 8089 viewer uses the same derived
observations for its marker, normal and force. No CoorGrasp/MPC or robot control
integration is added. See [the API and usage guide](sharpa-tactile-adapter.md).

The audit found an existing **simulation** `contact.ContactObservation`, but no
reusable hardware tactile adapter. The simulation contract requires object
identities and six-component wrenches that this sensor path cannot establish.
It remains intact; `TactileContactObservation` is a separate sensing type, not a
fake controller-compatible wrench with zero-filled unknown fields.

## Changed paths

- `src/ada_grasp_ctrl/contact/tactile_observation.py`: frozen sensing observation
  and geometry provenance; finite normalized vectors and owned immutable tuples.
- `src/ada_grasp_ctrl/contact/__init__.py`: export the additional sensing type
  without changing existing `ContactObservation` or `ContactSource`.
- `src/ada_grasp_ctrl/tactile_preview/adapter.py`: one shared frame/contact/FK
  conversion layer, five-finger statuses, full-snapshot replacement, freshness on
  every read, lock-protected access, provenance and explicit local diagnostics.
- `src/ada_grasp_ctrl/tactile_preview/kinematics.py`: extracted existing URDF
  loading/joint validation, with finite-clock checks and optional mesh-free load.
- `src/ada_grasp_ctrl/tactile_preview/geometry.py`: preserve profile names, require
  explicit supported units/consistent scale, nonempty source and textual evidence.
- `src/ada_grasp_ctrl/tactile_preview/viewer.py`: consume adapter observations;
  keep controls, mesh assembly, F6 readouts, local/base display and visual gates.
  Diagnostic sign reversal affects rendered arrows only.
- `src/ada_grasp_ctrl/tactile_preview/__init__.py`: lazy public adapter export so
  the system-Python ROS worker does not load FK/rendering dependencies.
- `src/ada_grasp_ctrl/tactile_preview/observe.py`: standalone five-finger read-only
  example using the existing ROS worker, with offline and bounded-duration modes.
- `tests/test_tactile_adapter.py`, `tests/test_tactile_preview.py`: adapter
  contract/transform/provenance/freshness tests and exact UI-observation comparison.
- API guide, this report, contact-preview guide link and task-baseline corrections.

The existing `data.py`, `ros_reader.py`, message definitions, subscriptions, QoS,
SDK byte loader, static maps, URDF assets and hardware driver were not changed.
The existing compact capture remains a raw evidence recorder with explicitly
diagnostic local geometry; it does not claim to provide synchronized base contacts.
Other pipeline/worktree changes were preserved separately.

## Actual validation

| Check | Result |
| --- | --- |
| Focused adapter + geometry + UI suite, with isolated DDS enabled | 41 tests: 40 passed, 1 skipped |
| Full canonical unittest discovery | 420 tests: 355 passed, 65 skipped; 33.553 s |
| Canonical Ruff check `src tests script` | Passed |
| Compile `src tests` | Passed |
| Changed Python files format check | Passed |
| Whole-repository format check | Eight existing unrelated files need formatting; preserved |
| Offline observation CLI | Exit 0; all five fingers `NO_DATA`, no ROS or HTTP listener |
| Public import dependency test | No `viser`, `rclpy` or `mujoco` imported |
| Installed ROS frame definition | Matched the inspected checkout definition byte-for-byte |

The focused skip was **8089 startup/shutdown** because an existing user preview
was listening on that port. Its process was preserved. The suite instead created
an independent Viser server on a free loopback port, checked HTTP 200 and the
actual scene handles against adapter observations, and stopped its own server.
The existing 8089 process runs the older loaded code until its owner restarts it.

The isolated DDS fixture used localhost domain **198** and the same existing
ROS reader bridge. Five synthetic tactile frames and joint feedback reached the
shared adapter and Viser; stopping the synthetic publisher caused contact graphics
to disappear on age expiry. No hardware driver was started/stopped, and no hardware
domain was changed. The full suite skips opt-in DDS and environment-dependent
checks; the separate focused run enabled and passed DDS.

The full-suite run preceded a final diagnostic-channel quality-label refinement;
the final focused run rechecked that refinement. Existing eight formatting
differences were already present before this work and were not fixed here.

Tests exercise all five identities, first-row selection/count, same-frame raw F6
and SDK metadata, finite tuple ownership, nonidentity S-to-E and E-to-base
transforms, default sign/order, alternative channel diagnostic labeling, zero
force, invalid/missing/empty/stale blocks, missing/stale/invalid/skewed joints,
publisher/disconnect gates, map bounds/holes, unavailable/unverified profiles,
unknown/inconsistent units, repeated/regressed SDK IDs, and age expiry without
another update. An unchanged reader snapshot cannot refresh its sample timestamp.

A regression uses sanitized recorded SDK frame **41024** from the earlier contact
session. With the audited installed static SDK maps, UV `[146.8359375,134.1796875]`
selects row 134, column 147; the resulting local point/normal matches the previous
preview's recorded numbers (1e-9 absolute tolerance). The test applies a known
nonidentity hand transform and verifies force rotation without negation. This
proves software consistency with the earlier candidate, not physical calibration.

Local evidence logs, not committed:

- `/tmp/ada_adapter_dds_tests_final.log`
- `/tmp/ada_adapter_full_tests.log`
- `/tmp/ada_adapter_format.log`

Reproduction in the existing workstation environment:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ADA_TACTILE_DDS_TEST=1 .conda/bin/python -m unittest discover -s tests -p 'test_tactile*.py' -v

PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
  .conda/bin/python -m unittest discover -s tests -v
.conda/bin/python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
```

Ruff was invoked from the workstation's existing UV tool cache, without installing
dependencies. Synthetic test children isolate their own domain; these instructions
do not reconfigure the operator's ROS driver.

## Verified, observed and unresolved

**Software verified:** shared conversion, immutable vectors/provenance, one explicit
frame per observation, same-frame contact/F6, validity/freshness/status handling,
UI/adapter agreement and isolated ROS delivery. There is no measured end-to-end
hardware latency or new real-hand contact capture in this refactor.

**Operator observed in the preceding session:** force direction was correct and
Newton units were reasonable; selected channels remain `0,1,2`. No additional
physical tests were performed or claimed in this task.

**Still unverified:** contact-marker position, blue surface normal, geometry XYZ
units/calibration, exact S-to-E alignment, connected sensor variant, full
per-finger/per-axis force signs, calibrated force magnitude and push/reaction
convention. `object_on_finger` is explicitly provisional; unverified geometry
still needs diagnostic opt-in. Normal and force are not flipped to agree.
The task document's stronger initial claim of observed marker/normal agreement
was corrected to match the actual operator feedback.

No new physical observation becomes valid when joints are missing: only the
explicit local elastomer diagnostic may remain available. No observation is
promoted to the existing control contract, and no command publisher, SDK device,
calibration, tare, motor operation, asset modification or upstream edit is added.
