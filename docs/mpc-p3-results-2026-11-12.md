# P3 software integration and offline qualification

> Subsequent operator-run hardware trial: see [first hardware trial report](mpc-p3-first-hardware-trial-2026-11-12.md). MPC commands were transmitted, but a streaming joint-limit fault stopped the run before native completion or lift. The offline-only statements below describe the earlier software handoff.

P3 is implemented through the hardware operation boundary. No real driver was started or commanded, no physical trial was performed, and no MuJoCo grasp was rerun. Existing unit tests that check contact/Jacobian equivalence were retained.

Operator commands: [P3 runbook](mpc-p3-operator-2026-11-12.md).
Evidence root: `output/mpc-p3-20261112/`.

## Audit and implementation

Starting HEAD: `7907b4925cc4c84d76e2898c5f39016f66716eff`. The worktree already contained extensive P1/P2 and pipeline changes; these were preserved. No commit or push. Starting hashes and copies are under `initial.json` and `original/`; final comparison is in `readiness-summary.json`.

P3 changed/added these files:

- `src/ada_grasp_ctrl/mpc/live.py`: default read-only readiness and explicit operator execution entrypoint.
- `src/ada_grasp_ctrl/mpc/live_execution.py`: existing receiver ownership, live sensing, passive worker and combined active episode.
- `src/ada_grasp_ctrl/mpc/observation.py`: selected-packet reference and receipt-aligned original adapter/estimator.
- `src/ada_grasp_ctrl/mpc/worker.py`, `ipc.py`: normalized tactile requests across existing Python ABI boundary.
- `src/ada_grasp_ctrl/mpc/streaming.py`: explicit hardware opt-in to existing bounded stream.
- `src/ada_grasp_ctrl/mpc/mcap_replay.py`: opt-in `--p3` receipt/reference/contact policy and native completion accounting.
- `src/ada_grasp_ctrl/contact/compatibility.py`: opt-in measured-cone diagnostic and ambiguous-finger omission policies; strict default preserved.
- `src/ada_grasp_ctrl/ros_execution/bridge.py`, `experiments/right_ros_sender.py`: default baseline preserved; active completion acknowledgment skips the combined squeeze phase only after native completion.
- `tests/test_mpc_p3.py`, `tests/test_mpc_p3_receiver.py`: focused integration regressions.
- `script/test_mpc_ros_mock.py`: explicitly inject the intended 60 ms stall; discovery latency is no longer assumed to exceed 50 ms.
- This report and `docs/mpc-p3-operator-2026-11-12.md`.

Original TactileAdapter, Ours optimized policy, grasp controller, native wrapper, pipeline runner, Runtime and settings retain their starting hashes. No driver, SDK, force-limiter or successful Linear pipeline modification was made in P3.

## Evidence

| Scope | Result | Evidence |
| --- | --- | --- |
| Implemented | Selected reference; live receipt bridge; passive/active receiver paths; readiness CLI | Sources above |
| MPC regressions | 36 passed; 3 ROS-only tests skipped in native Python and separately passed in ROS Python | `mpc-tests.log`, `p3-receiver-tests.log` |
| Tactile regressions | 45 passed, 3 ROS-dependent skips | `tactile-tests.log` |
| Existing receiver / transport / pipeline tests | 7 / 11 / 16 passed | `receiver-tests.log`, `transport-tests.log`, `pipeline-tests.log` |
| P3 receiver, synthetic | Passive failure leaves baseline serialized commands byte-identical; native completion advances two phases; timeout/solver/duplicate-owner/cancel/stale prevents lift | `p3-receiver-tests.log` |
| Real recording, counterfactual | 52 contact-eligible ticks; 51 eligible observations; 44 accepted proposals; one native completion; six later ticks marked after completion | `replay-final/run_report.json`, `replay-final/ticks.jsonl` |
| DDS mock transport, domain 199 | 30 requests/proposals; 288 paired arm/hand commands; no left commands; one publisher per command topic; zero after shutdown | `ros-mock-final/report.json`, `trace.json` |
| Live read-only | **Not performed** | Operator step |
| Physical active | **Not performed** | Operator step |

Mock MPC period p50/p99: 100.61/100.84 ms. Bounded stream period p50/p99: 10.39/10.98 ms. These are observed mock scheduling measurements, not a real-time guarantee. New P3 receiver branches were tested with synthetic I/O; DDS transport was qualified through the existing mock stream script. A complete P3 physical session remains unverified.

Ruff for changed Python files, `python -m compileall -q src tests`, and `git diff --check` passed.
The first mock fault injection exposed a test assumption: DDS ownership detection completed too quickly to create a stall. Explicitly injecting 60 ms resolved the fixture without changing runtime limits. Receiver and transport suites use ROS system Python and native conda Python respectively; a combined system-Python attempt lacked MuJoCo, so they were rerun separately with the appropriate environments.

## Recorded contact and completion diagnostics

The replay uses the original 75,626 decoded messages from `physical_mpc_observe_20261112_194701`, receipt ordering, the recorded baseline phase intervals, and selected candidate `grouplive_c0003`. It never publishes recorded commands or runs a simulator. Baseline measured states remain independent of counterfactual MPC proposals.

Only one tick remains unavailable because it lies at the uncertain recorded phase-clock boundary. Compatibility no longer blocks on measured friction-cone violation or on the unrelated frozen scene. All 44 command proposals were accepted; native completion was recorded once at receipt offset 65.0522 s, index 44, normal-force sum 16.1829 N. This is an offline evaluation of recorded forces, **not evidence of physical MPC completion**.

| Finger | Single-point samples | Negative normal | Excess tangential / cone FAIL | Cone PASS | Ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: |
| Index | 51 | 0 | 51 | 0 | 1 |
| Ring | 13 | 0 | 2 | 11 | 0 |

At the nearest sampled tick to +64.5 s (+64.4522 s), Index Fn=3.0119 N, Ft=2.0267 N, required mu=0.6729; Ring Fn=0.2994 N, Ft=0.09084 N, required mu=0.3034. Both exceed mu=0.3 because of tangential force, not negative normal. Different sampling time explains differences from the earlier exact +64.5 s preview. Full summary: `contact-summary.json`.

Remaining commissioning issues: provisional force/geometry calibration, unknown acquisition clock, live freshness and ownership, actual pregrasp-to-MPC timing, and native endpoint alignment with the reviewed lift. Existing checks remain in effect; no live-ready or physical-success claim is made.

## Reproduction

Native regressions:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl OMP_NUM_THREADS=4 \
ADA_RIGHT_FULL_PREVIEW="$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1" \
ADA_P3_PACKET="$PWD/output/grasp_pipeline/physical_baseline_20261112_194720/grouplive/export/targets.json" \
.conda/bin/python -m unittest discover -s tests -p 'test_mpc*.py' -q
```

Read-only replay, including the existing decoder:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 .conda/bin/python -m ada_grasp_ctrl.mpc.mcap_replay --p3 \
  --bag /home/hao-fang/ws_fanuc/output/physical_mpc_observe_20261112_194701 \
  --baseline-run output/grasp_pipeline/physical_baseline_20261112_194720 \
  --model-report output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json \
  --output-root "output/mpc_p3/replay_$(date +%Y%m%d_%H%M%S)"
```

Mock only, in a dedicated terminal:

```bash
source /opt/ros/jazzy/setup.bash
source /home/hao-fang/ws_fanuc/install/setup.bash
export ROS_DOMAIN_ID=199 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export PYTHONPATH="$PWD/src:$PYTHONPATH"
/usr/bin/python3 -m ada_grasp_ctrl.pipeline.ros_probe --empty-domain
ros2 launch bimanual_manipulation single_arm_hand.launch.py \
  prefix:=right_ crx_mock:=true sharpa_backend:=mock rviz:=false viewer:=false
```

In a second terminal with the same setup and domain:

```bash
/usr/bin/python3 -m unittest discover -s tests -p 'test_mpc_p3_receiver.py' -v
/usr/bin/python3 script/test_mpc_ros_mock.py \
  --output-root "$PWD/output/mpc_p3/mock_$(date +%Y%m%d_%H%M%S)" \
  --model-report "$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json" \
  --mpc-python "$PWD/.conda/bin/python" --duration 3
```

Stop only the mock launch you started with Ctrl+C. Hardware commands are separate in the operator runbook.
