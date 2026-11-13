# MPC Phase 2 offline validation, 2026-11-12

Initial and final HEAD: `7907b4925cc4c84d76e2898c5f39016f66716eff`.
No commit or push was performed. Existing dirty/untracked dual-arm, ROS and tactile
work was preserved. Task-only changes are recorded relative to the initial file
hashes, rather than incorrectly attributing the entire `git diff` to this task.

Fresh evidence root:
`output/mpc-phase2-20261113T021739Z/` (UTC; local date November 12).

The offline results below describe the original Phase 2 delivery. Later operator-run
physical baseline recording and compact-recorder checks are documented in the
[follow-up](#operator-baseline-recording-follow-up); they do not qualify physical MPC.

## Status

| Task | Implemented and tested offline | Incomplete / unverified |
| --- | --- | --- |
| A | Explicit modes, unchanged baseline default, shared linear reference/preview, guarded linear mock receiver, ordered passive/active stage boundary, structured logs; known seven-stage baseline and passive cycles complete | Active known-cycle fixture aborts in grasp at native timeout before reviewed arrival; no automatic transition to squeeze/lift was permitted |
| B | Original native solver wrapper, named measured/commanded state separation, bounded 100 Hz stream, exclusive ROS mock ownership, JSON Python ABI boundary, real DDS timing, failure and shutdown checks | Complete active grasp-cycle qualification remains open; timing experiment used synthetic valid-empty contact and does not establish a physical/contact grasp |
| C | Strict point-contact compatibility, provenance, signed force and unchanged normal, versioned replay, MuJoCo/real-model Jacobian/Ks/optimizer equivalence | Replay fixture is explicitly synthetic; formal force/geometry/mount/acquisition calibration and physical recordings were not validated |

The original Ours mathematics and `TactileAdapter` were not modified. Unknown F6
trailing channels do not block point-contact compatibility and are never labeled
measured zero torque. During the original offline delivery, physical hardware was
not launched, connected, commanded, tared or reconfigured. All hardware enablement
remains disabled for new modes.

## Checks and results

Tests used the maintained `.conda/bin/python` (3.10), with an explicit clean
`PYTHONPATH=$PWD/src`, `MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl`, `MUJOCO_GL=egl`, and
`OMP_NUM_THREADS=4`. ROS-only checks used sourced Jazzy `/usr/bin/python3` (3.12).

| Check | Exact entrypoint | Result | Evidence |
| --- | --- | --- | --- |
| Before-change unit baseline | `.conda/bin/python -m unittest discover -s tests -v` | 420 total; 355 passed, 65 skipped, 0 failed | `baseline-unittest.log` |
| Final full suite, explicit real-CRX fixture | Same command, `ADA_RIGHT_FULL_PREVIEW=$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1` | 444 total; 387 passed, 57 skipped, 0 failed | `full-unittest-final.log` |
| New focused integration tests after final diagnostic/provenance edits | `.conda/bin/python -m unittest discover -s tests -p 'test_mpc*.py' -v`, same explicit fixture | 24 passed, 0 skipped/failed | `mpc-focused-final.log` |
| Original right-full linear/ours/native parity | `.conda/bin/python -m unittest discover -s tests -p test_right_full_control.py -v`, absolute fixture path | 20 passed, 0 skipped/failed | `right-full-absolute-path.log` |
| Existing ROS receiver safeguards | `/usr/bin/python3 -m unittest discover -s tests -p test_right_ros_receiver.py -v` | 7 passed, 0 skipped/failed | `ros-receiver-tests.log` |
| ROS native-worker + streaming + ownership/stall/silence | `/usr/bin/python3 script/test_mpc_ros_mock.py --output-root <root>/ros-mock --model-report <absolute-preview>/preview_report.json --mpc-python $PWD/.conda/bin/python` | Passed; 30 proposals, 289 paired command publishes, left count zero | `ros-mock/report.json`, `ros-mock/trace.json`, `ros-mock/runtime/` |
| Unchanged CRX quintic retiming | `/usr/bin/python3 script/test_mpc_retiming.py --trace <root>/ros-mock/trace.json --output-root <root>/quintic-retiming` | Passed replay; 39 extended horizons | `quintic-retiming/report.json`, `trace.json` |
| Lint | `ruff check src tests script` | Passed | `ruff-check-final.log` |
| Compile | `.conda/bin/python -m compileall -q src tests` | Passed | `compile-final.log` |
| Whitespace | `git diff --check` | Passed | Final working-tree check |
| Whole-repo format | `ruff format --check src tests script` | 8 pre-existing unrelated files need formatting; task files pass | `ruff-format-final.log` |

Ruff was the already available cached executable
`/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff`.
No dependency was added. The eight format failures are `script/run_crx_sharpa_phase1.py`,
`script/run_crx_sharpa_phase2.py`, `src/ada_grasp_ctrl/crx_sharpa_adaptive.py`,
`src/ada_grasp_ctrl/crx_sharpa_demo.py`, `src/ada_grasp_ctrl/utils/viewer_session.py`,
`tests/test_crx_sharpa_demo.py`, `tests/test_crx_sharpa_preview.py`, and
`tests/test_dummy_arm_ik.py`; their original hashes were unchanged.

The initial optional right-full invocation used a relative fixture path; its repeat
test hit existing process-global data-root resolution (19 passed, 1 error). The
absolute-path run passed all 20. The new factory restores the caller's effective
runtime roots after native model/config construction.

## Baseline, passive, active and contact artifacts

The tested CLI pattern was:

```bash
.conda/bin/python script/run_grasp_pipeline.py \
  --mode replay --environment mock --controller-mode MODE \
  --offline-packet "$PWD/output/grasp_pipeline/cycle_enter_audit_v1/pipeline/group01/export/targets.json" \
  --output-root FRESH_OUTPUT
```

`baseline-known-cycle/` and `passive-known-cycle-v2/` completed pregrasp, grasp,
squeeze, lift, hold, release and return. Both emitted 3,537 **mock** baseline commands;
their `mpc_offline/commands.jsonl` files are byte-identical. Baseline had no proposals.
Passive had 51 unavailable proposals with explicit reasons and zero MPC commands.

Adding `--mpc-scene <absolute-preview>/g0000/scene.xml` produced
`passive-mujoco-cycle/`: 51 valid native proposals, **zero MPC commands**, zero ROS
publishers, and the same byte-identical baseline command trace. This is a kinematic
contact fixture, not a claim that the camera-selected object matches that frozen
scene or that a dynamic grasp succeeded.

Adding `--tactile-replay $PWD/tests/fixtures/mpc_tactile_replay_v1.json` produced
`passive-tactile-cycle/`: all seven stages completed on the mock baseline; 51 proposals
were unavailable/rejected and none executed. The short synthetic recording is not
coherent/fresh for that long selected trajectory. The dedicated fixture tests verify
eligible contact → valid empty → eligible contact → invalid → reordered → aged gap,
without inventing contacts for the pipeline run. Different replay clock origins can
shift a boundary by one 10 ms tick; no physical timing equivalence is claimed.

`active-mujoco-cycle/` and the diagnostic rerun `active-mujoco-cycle-v2/` preserve
the unsuccessful active result. Pregrasp completed; 88 accepted proposals produced
880 guarded mock commands during grasp. The next native stop/timeout rejection
aborted before reviewed grasp arrival. The candidate was not applied, and squeeze,
lift and subsequent stages were not run. This full active fixture is **not accepted**;
qualifying its contact scene/reference/arrival interaction is remaining work, without
changing the original objective or silently relaxing arrival checks.
At the final tick, the reference-minus-measured errors were approximately
`right_J1=-0.011121 rad` and `right_J4=0.012267 rad`, exceeding the retained 0.01 rad
arm arrival threshold. Contact count was zero. The native controller optimizes the
hand-base pose/force objective; this result must not be relabeled joint arrival.

Each offline run has `run_manifest.yaml`, `run_report.json`, `proposals.jsonl` and
`commands.jsonl`. Solver rejection reasons remain in proposal diagnostics even in
earlier evidence whose top-level guard error was generic.

## Scientific regression gate: existing differences remain visible

The canonical command was run with a unique root:

```bash
ADA_GRASP_CTRL_RELEASE_GATE_ROOT=FRESH_ROOT \
PYTHON_BIN="$PWD/.conda/bin/python" bash script/run_release_gate.sh quick
```

An initial attempt inherited ROS 3.12 `PYTHONPATH` and failed to load the correct
Pinocchio extension. Its artifacts remain in `quick-gate/`; it is **not** a passing
test. Re-running with clean `PYTHONPATH=$PWD/src` produced complete current-run data.
The gate stopped at Shadow inventory mismatch, so the remaining hands were explicitly
run with `script/run_example.sh HAND quick` and independently checked using:

```bash
.conda/bin/python script/validate_quick_results.py verify \
  --manifest examples/quick_manifest.json --expected examples/quick_expected_status.json \
  --hand HAND --output-root FRESH_HAND_OUTPUT
```

| Hand | Current classification counts | Golden comparison |
| --- | --- | --- |
| Shadow | 68 success, 4 failure, 21 invalid, 7 degraded, 0 execution error | 3 sample differences; expected 66 success, 4 failure, 21 invalid, 9 degraded |
| Allegro | 80 success, 5 failure, 14 invalid, 1 degraded, 0 execution error | All 100 match |
| LEAP Tac3D | 87 success, 6 failure, 7 invalid, 0 degraded/error | 1 sample difference; expected 88 success, 5 failure, 7 invalid |

To establish causality, the initial tracked source was reconstructed in
`original-workspace/` using HEAD plus `original.diff`, with identical asset/fixture
inputs. Its Shadow and LEAP quick runs reproduced the **exact same 100 per-sample
classifications** as current code, not just the same totals. See
`quick-original-comparison.json`, `original-shadow-quick.log`, and
`original-leap-quick.log`. Golden files and numerical tolerances were not changed.
The overall release gate remains **not green** despite original/current parity.

## Timing, no-hardware evidence and future work

| Measured quantity | p50 | p95 | p99 | maximum |
| --- | ---: | ---: | ---: | ---: |
| MPC request period | 100.377 ms | 100.722 ms | 100.839 ms | 100.869 ms |
| Guarded publish period | 10.315 ms | 10.768 ms | 10.923 ms | 10.999 ms |
| Solver duration, valid-empty contact | 1.492 ms | 2.520 ms | 2.779 ms | 2.873 ms |
| Feedback receipt age | 4.566 ms | 9.637 ms | 10.594 ms | 18.824 ms |

The DDS observer received 1,600 lower-level messages, but callback arrival intervals
include queueing and the intentional shutdown/stall test. They do not measure physical
servo rate or prove exact 500 Hz scheduling. Offline replay of the actual bounded
targets through the unchanged quintic implementation had horizon p50/p95/max
30/61.2/80 ms, 39 horizon extensions, maximum reference tracking difference
0.00043625 rad and zero terminal difference after settling. These are tiny-target
mock results, not general contact-motion performance bounds.

`ros-launch-command.txt` records the exact launch:
`ros2 launch bimanual_manipulation single_arm_hand.launch.py prefix:=right_ crx_mock:=true sharpa_backend:=mock rviz:=false viewer:=false`.
The domain probe required empty localhost domain 199. Runtime independently recorded
the mock arm plugin, mock hand backend, unchanged quintic/100 Hz input settings and
one owner on each right command topic. Both left topic publisher inventories and
message counts were zero; both right publishers disappeared on shutdown. Only the
owned mock process group was stopped. No 8089 listener was required or killed.

`task-change-manifest.json` records SHA-256 equality with the initial source for
`grasp_controller.py`, `optimized_runner.py`, `optimization.py`, `mujoco_adapter.py`
and `tactile_preview/adapter.py`. Driver repos, SDK, calibration, gains and safety
thresholds were not edited.

Future work is separate: first resolve/qualify the active mock stage-arrival fixture
and existing release-inventory drift. Then review physical **passive** recording of
the approved baseline to measure force sign/units, geometry/mount alignment,
acquisition synchronization, stale/gap behavior and tracking lag. Only after those
criteria and ownership/stop readiness pass should physical active grasp/squeeze be
reviewed. No physical commissioning was performed here.

## Operator baseline recording follow-up

The operator subsequently executed the existing hardware **baseline** on ROS domain
87 and recorded `/home/hao-fang/ws_fanuc/output/physical_mpc_observe_20261112_194701/`.
The assistant inspected it offline without replaying commands or operating hardware.
This was the original full recording, including `dist_force`, not a compact recording.

Evidence:

- Bag metadata and MCAP in the absolute directory above.
- Decode/timing audit: `output/physical_bag_inspection_20261112_194701/report.json`.
- Baseline result: `output/grasp_pipeline/physical_baseline_20261112_194720/run_report.json`.

All 75,626 stored messages decoded successfully; counts match metadata. Duration was
92.385 seconds and file size approximately 1.1 GB. All nine requested topics contained
data. Arm/hand commands each had 3,950 messages, spanning bag offsets 46.507–88.414 s.
Joint feedback had 45,969 arm messages and 9,219 hand messages. Feedback and tactile
streams covered the command interval.

| Finger | Frames | Contact VALID | Contact MISSING |
| --- | ---: | ---: | ---: |
| Thumb | 2,506 | 448 | 2,058 |
| Index | 2,498 | 440 | 2,058 |
| Middle | 2,504 | 365 | 2,139 |
| Ring | 2,508 | 297 | 2,211 |
| Pinky | 2,522 | 184 | 2,338 |

All recorded F6 blocks were source-status VALID. No nonfinite tactile numeric values
or joint positions were found. This establishes structural readability, not force
calibration. No contact EMPTY was reported. Index had two two-point frames and middle
had four; resultant-force allocation for such frames remains rejected by compatibility.
Source validation errors were empty; a few SDK forward-gap warnings were retained.

The recorder terminal reported **53 transport-lost messages**. The available bag does
not identify all affected messages or establish their cause. The largest observed
receipt gaps (approximately 264–300 ms across feedback/tactile streams) occurred near
offset 39.6 s, before commands began. Within the command interval, maximum arm/hand
feedback gaps were 3.27/11.89 ms, tactile gaps 67.38–99.85 ms, and command gaps about
64.37 ms. These are bag receipt intervals, not hardware acquisition timing or proof
that loss had no effect. Per-finger receipt sequence skips include driver cache
overwrites and must not be equated with those 53 transport losses.

The baseline report marked all eight motion phases completed, with finger error
recording but no finger-arrival gate. `object_lifted` and `object_held` remained null;
physical object success was not independently established. Widespread contact MISSING
is still unresolved and must not be silently interpreted as valid no-contact.

### Compact recording follow-up

The [compact recorder and usage guide](tactile-compact-recording.md) were added after
this inspection. It preserves the nine topics but removes distribution arrays before
writing MCAP, marking the omitted block MISSING with explicit original-metadata
provenance. The 12,538 original distribution arrays alone occupy 1,083,283,200 bytes.
Original driver, live messages, force conventions and MPC math are unchanged.

Validation used sourced ROS Jazzy `/usr/bin/python3` and `PYTHONPATH=$PWD/src:$PYTHONPATH`:

```bash
/usr/bin/python3 -m unittest discover -s tests -p test_tactile_compact_recording.py -v
```

Both tests passed: source/non-distribution field preservation across all four contact
statuses, and nine-topic MCAP round-trip with preserved joint bytes and bag timestamps.
An additional synthetic localhost domain 199 run recorded all nine topics for three
seconds and closed cleanly (97 tactile frames per finger, 99–100 joint messages per
topic). This live check used temporary files and is a session-observed result, not a
retained physical evidence bag. Focused Ruff, compile and whitespace checks passed.

Remaining work: resolve MISSING/EMPTY source semantics, implement complete synchronized
bag conversion and physical passive orchestration, and qualify the active mock
grasp-stage arrival/timeout behavior before physical active MPC integration. Neither
the baseline recording nor the new recorder clears force/geometry calibration gates.
