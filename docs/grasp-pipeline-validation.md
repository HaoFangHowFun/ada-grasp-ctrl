# CRX + Sharpa pipeline validation — 2026-11-10

The complete entry point is `script/run_grasp_pipeline.py`. All grasp inference,
selection, preview and execution tests use the **right CRX and the 22-joint right
Sharpa hand**. Robot execution was mock-only; no physical actuator was commanded.

## Five real recorded clouds

`output/grasp_pipeline/five_raw_v1/run_report.json` records a fresh run through
object extraction, surface construction, the actual right-full DexLearn checkpoint,
selection from newly measured mock joint states, six-slot contact sheets and
selected-target export. The run did not reuse saved inference.

| Group | Generated | Feasible | Selected |
| --- | ---: | ---: | --- |
| 01 | 8 | 2 | c0003 |
| 02 | 8 | 3 | c0004 |
| 03 | 8 | 3 | c0004 |
| 04 | 8 | 4 | c0000 |
| 05 | 8 | 3 | c0000 |

These choices match the preceding selector validation. Each group has a contact
sheet under `groupNN/images/candidates_001.png`, with unused cells left blank.
A seven-feasible-candidate unit fixture separately verifies two-page pagination.
The fast saved-candidate mode also passed all five groups in `five_cached_v2/`.

## 8089 and Enter-to-ROS execution

Accepted full-entry audit: `output/grasp_pipeline/enter_audit_v4/run_report.json`.
The audit launches the entry point with owned right-only mock drivers on domain 88,
connects to the actual Viser WebSocket, switches all five datasets, tests playback,
and writes one Enter to the pipeline terminal while group 01 is displayed.

| Check | Result |
| --- | --- |
| Before Enter | Zero arm, hand, left-arm and left-hand commands; no arm/hand command publishers |
| Preview | HTTP 200, all five dataset controls and timed playback work |
| Approved execution | Only group 01 / c0003 |
| Three phases | pregrasp, grasp and squeeze all succeeded |
| Target/timing identity | All sent joint arrays and planned durations exactly match the reviewed plan |
| Commands | 1,246 right-arm and 1,246 right-hand messages |
| Left-side commands | Zero |
| Cleanup | Only this run's owned processes stopped |

The preview uses the full measured 28-joint start, the receiver's shared quintic
coefficient and exported phase durations, including nominal settling periods.
The sender rechecks the reviewed start before each phase. This is nominal command
agreement, not a claim that ROS timing, tracking or velocity limiting produces
pixel-for-pixel playback of an actual physical trajectory.

The original three-phase preview-only server displayed `five_raw_v1/integration_summary.json`
on 8089, including the recorded environment cloud. Its HTTP/WebSocket smoke evidence
is `output/grasp_pipeline/preview_8089_final/ui_smoke_full_scene.json`. It has no Enter gate or
ROS command process attached.

## ROS capture branch

`output/grasp_pipeline/live_transport_v2/run_report.json` verifies the `--mode live`
branch with original PointCloud2/RGB/CameraInfo messages from capture 01 republished
with fresh timestamps on isolated mock domain 88. The pipeline itself uses the
normal ROS subscriptions, serial parameter check, internal TF, calibration, object
extraction, checkpoint inference, selection and export. It produced eight candidates,
two feasible candidates and selected c0003. Camera/world round-trip maximum error
was approximately `6.66e-16` m.

This validates ROS sensor transport with recorded real measurements. It is **not**
a new physical-camera capture or physical calibration acceptance. The test publisher
is explicitly named `script/replay_camera_for_pipeline_test.py` and refuses domains
other than 88. It publishes no robot command.

## Regression and failure handling

- Full Ada suite: **254 run, 247 passed, 7 skipped**. Six skips require ROS Python;
  the existing OpenCV-dependent test is skipped in Ada's environment.
- Sourced ROS receiver suite: **6 passed**, covering goal validation, original speed/
  acceleration defaults, stale feedback behavior and absence of bimanual task imports.
- Pipeline unit tests cover Enter versus EOF/cancel, snapshot freshness, arm/hand
  drift, exact quintic endpoints and settling, six-slot pagination, no-feasible export
  refusal, source/target/timing changes, and output overwrite refusal.
- Full Ruff lint and compileall pass; all changed files pass format and diff checks.
  Repository-wide formatting retains the eight previously documented unrelated failures.

Retained diagnostic runs:

- `five_cached_v1`: inherited ROS Python paths contaminated Ada imports. The checkout
  entry point now re-execs in Ada's maintained environment with an isolated Python path.
- `enter_audit_v1`: the test WebSocket client omitted Viser's required version subprotocol.
  The client was fixed; the viewer was already serving HTTP.
- `enter_audit_v2`: checking availability with a bare socket bind incorrectly rejected a
  recently closed port. The entry now checks for an active listener, while the viewer
  still refuses any port other than 8089.
- `live_transport_v1`: a newly created read-only snapshot received stale queued feedback.
  The snapshot collector now discards old samples while waiting for fresh ones within the
  original bounded timeout. Discarded samples never refresh state; execution watchdogs
  and the 0.5-second threshold were not relaxed. Failures now record their reason in the
  snapshot diagnostic log.

The repo's separate maintained Shadow quick regression was also run; it is unrelated
to the CRX/Sharpa grasp data path. All 100 classifications exactly match the preceding
pre-task run: 68 success, 4 failure, 21 invalid initialization, 7 solver degraded and
0 execution errors. Its exit 1 and the same three pre-existing golden differences
remain recorded in `validation_final/quick_comparison.json`. No golden inventory or
scientific thresholds were changed, and this is not reported as a passing release gate.

Physical calibration, mount acceptance, full environment geometry and continuous-path
validation remain pending. The inherited 10 mm hand penetration gate is unchanged.
Both selector and exported plans retain their non-hardware-ready status. The original
validation above predates the optional lift described below. No automatic fallback
candidate, retry of a grasp, release or return motion was added.

## Post-selection lift validation

The default pipeline now appends a 5 cm world +Z lift after squeeze. Lift target
conversion is excluded from candidate scoring/selection, and lift collision/path
checks are deliberately absent. A bounded local IK solve converts the chosen target
to joint angles; a failed conversion retains the selected grasp but blocks that
group's execution. All evidence below is mock-only.

`output/grasp_pipeline/lift_validation_v1/five_comparison.json` compares against
`five_cached_v2`, which uses the identical saved inference inputs. All five selected
IDs, costs and original three-stage joint arrays match exactly. Comparison with the
separately rerun inference in `five_raw_v1` shows small input-dependent numerical
differences, so that run is not used to claim exact equality.

| Group | Selected | Lift result |
| --- | --- | --- |
| 01 | c0003 | +Z 0.05 m; 2.2 s plus 0.3 s settling |
| 02 | c0004 | +Z 0.05 m; 2.2 s plus 0.3 s settling |
| 03 | c0004 | +Z 0.05 m; 2.2 s plus 0.3 s settling |
| 04 | c0000 | +Z 0.05 m; 2.2 s plus 0.3 s settling |
| 05 | c0000 | Target conversion failed; no execution packet |

For groups 01–04, direct FK verifies the displacement and unchanged orientation;
all 22 hand joints exactly retain squeeze. Group 05's solve terminated with position
error approximately 0.677 mm and orientation error 0.04733 rad, exceeding the unchanged
0.01 rad orientation tolerance. This diagnoses failure of this conversion, not a
proof that every possible IK branch is unreachable. The candidate is not reselected,
the height is not silently reduced, and no approximate failed target is sent.

`output/grasp_pipeline/lift_enter_audit_v1/run_report.json` records a full entry-point
audit: zero commands before Enter; group 01 alone executes pregrasp, grasp, squeeze
and lift; all four sent targets and durations exactly match the reviewed preview.
There were 1,485 right-arm and 1,485 right-hand messages, with zero left commands.
The pipeline reports the group 05 preparation error and exits 1 even though group
01's requested execution completed. `lift_motion_completed` is true;
`object_lifted` remains null. The mock test does not establish a physical grasp.

The current preview-only server uses
`lift_enter_audit_v1/pipeline/integration_summary.json` on **8089**. Its final
WebSocket evidence is `lift_preview_8089_v2/ui_smoke.json`: HTTP 200, five datasets,
four working lift phase controls, a blocked-execution status for group 05, environment
cloud visibility and timed playback. It has no execution gate or attached ROS sender.

Validation logs under `lift_validation_v1/` record **259 Ada tests, 252 passed and
7 skipped**, plus **6 ROS receiver tests passed**. Focused tests cover world/model
frame rotation, hand/orientation retention, absence of candidate/collision calls,
failed/nonfinite target refusal, four-stage timing and transport binding, and legacy
three-stage compatibility. Ruff lint and compileall pass; changed-file formatting
and diff checks pass. The same eight unrelated repository-wide formatting failures
remain. No golden data, candidate tolerances or speed/acceleration settings changed.

## Current live grasp / hold / release / return cycle

The entry now defaults to `--mode live`. Its seven declared phases are pregrasp,
grasp, squeeze, lift, hold, release and return. Hold streams the unchanged lift
target for 5 seconds only after lift arrival succeeds. Release keeps the lifted
arm pose and opens all 22 hand joints to zero. Return restores the measured
six-joint start while keeping the hand open. The nominal preview uses the same
targets, durations and quintic interpolation as execution.

The operator explicitly approved leaving the new return path unchecked, in
addition to the previously exempted lift. This is recorded as unchecked motion,
not a verified continuous path. Hardware is an explicit backend option and is
never launched automatically. The receiver accepts only the next exact target
and duration from the approved packet. Its hardware exception is scoped to that
reviewed complete cycle; unreviewed joint commands retain the original hardware
path guard. Backend/model checks, MIT/read-only checks, original limits,
fresh-feedback watchdogs and cancellation remain enforced.

New evidence (all robot commands were mock-only):

| Artifact under `output/grasp_pipeline/` | Result |
| --- | --- |
| `cycle_enter_audit_v2/run_report.json` | All seven actions completed; exact preview/command target and timing identity; zero commands before Enter |
| `cycle_enter_audit_v2/ui.json` | HTTP 200; all five datasets; working lift/hold/release/return controls for groups 01–04; group 05 conversion failure retained |
| `cycle_cancel_audit_v2/run_report.json` | Cancellation during hold acknowledged; no release or return goal sent |
| `cycle_live_transport_v2/run_report.json` | Default live branch received fresh ROS messages, reran actual inference, selected c0003 from 2/8 feasible candidates and exported seven phases |
| `cycle_validation_v1/unit_tests.log` | 264 tests run, 256 passed, 8 skipped (7 ROS-only and 1 existing OpenCV-dependent test) |
| `cycle_validation_v1/ros_tests.log` | All 7 receiver tests passed, including retained hardware guards and MIT/read-only/backend refusal |

The complete execution audit recorded 3,360 right-arm and 3,360 right-hand
commands, zero left-side commands, and return arrival within the original
tolerances. The hold goal was sent after the lift result; release began 5.364 s
later, including the nominal 5 s hold, 0.3 s settling and transport overhead.
`object_lifted` and `object_held` remain unknown.

The live transport audit republishes original recorded real camera messages with
fresh stamps on isolated domain 88. It does not claim a new physical-camera
capture. Physical camera/robot validation is left to the operator. Audit-owned
ROS processes and the 8089 viewer were closed when testing completed.

Retained diagnostic attempts:

- `cycle_enter_audit_v1`: receiver logging could not serialize the newly added
  packet Path argument. The CLI now normalizes it to a string for logs.
- `cycle_cancel_audit_v1`: a stale arm sample stopped execution before pregrasp
  while the full unit suite was running concurrently. The 0.5 s watchdog was
  retained; the cancellation audit was rerun without that competing workload.
- `cycle_live_transport_v1`: the test sensor publisher lost its ROS Python path.
  The test harness was corrected; the application's environment isolation did
  not change.

Canonical Ruff lint, compileall and diff checks pass. Changed source files are
formatted; the same eight unrelated repository-wide format failures remain.
No golden data or maintained controller behavior was changed.
