# MPC P1 / P2a minimal offline implementation — 2026-11-12

Evidence: `output/mpc-p1-p2a-20261112/`. HEAD remains
`7907b4925cc4c84d76e2898c5f39016f66716eff`. Existing dirty/untracked work was
preserved; no commit or push. `initial.json` and `original/` retain the starting
MPC source inventory. Validation was deliberately limited to focused tests and
offline replay, as requested; no full release gate or live DDS test was run.

## P1: implemented and checked offline

`NativeMPC` reports `native_complete` separately from `native_timeout`, after input
validity checks. Diagnostics include native policy stage, reference index, force,
threshold, `should_stop`, maximum/remaining steps and solver outcomes. Original
Ours mathematics, policy thresholds and solver handling were not modified.

The offline active runner treats grasp/squeeze as two reference segments within
one native episode. The grasp reference boundary is not reported as completed
control until Ours completes the episode. Linear arrival remains authoritative for
baseline stages. Guard checks still run before native completion is acknowledged.
The ROS mock stream recognizes completion and closes its owned runtime; this
branch was updated but was not exercised against live DDS in this delivery.
Passive computation/contact-source exceptions become diagnostics. Passive command
interpolation now follows the supplied baseline plan, including a linear plan.

The previous `active-mujoco-cycle-v2` failure came from `NativeMPC.tick` at index 88,
which returned a rejected proposal consumed by `GuardedStream.accept`. Its zero
contacts imply zero summed force and false native completion; native maximum steps
were exhausted. The final J1/J4 errors were additional diagnostics, not the first
abort gate. Historical logs lack all native state fields, so they cannot support a
complete retrospective per-tick force-policy audit; the new run supplies them.

| Fresh run | Result |
| --- | --- |
| `final-cycle-baseline`, `final-cycle-mpc_passive` | Seven stages complete; 3,537 commands each, byte-identical; passive publishes zero MPC commands |
| `final-cycle-mpc_active` | 88 valid proposals / 880 guarded mock commands; native stage 2, path length 44, force 0 N, threshold 15 N, `should_stop=false`, max steps 88; native timeout; only pregrasp completed, no lift |
| `fake-native-complete` | Explicitly patched `should_stop` becomes true at index 44; normal completion and seven-stage cycle succeed without fixed-target arrival authority; this is a fake termination fixture, not physical force success |
| `final-linear-linear`, `final-linear-mpc_passive` | 2,734 commands each, byte-identical; both stop in release at the retained joint-limit guard; full linear cycle is **not** qualified by this converted fixture |

Cycle reproduction:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl OMP_NUM_THREADS=4 \
  .conda/bin/python output/mpc-p1-p2a-20261112/check_cycles.py
```

The script uses fresh output directories; change its evidence root before rerunning.
`cycle-comparison.json` contains counts and the identical linear command SHA-256.
The original legacy baseline interpolation and the guard implementation are unchanged.

## P2a: real bag replay implemented; valid proposals blocked by source data

`mpc.mcap_decode` uses `rosbag2_py.SequentialReader` and CDR deserialization only;
it does not initialize a ROS context or create nodes, subscriptions or publishers.
`mpc.mcap_replay` uses the existing `TactileAdapter`, candidate geometry catalog,
`TactileContactEstimator`, kinematic model and native wrapper. No SDK library is
loaded into a device runtime; the existing candidate geometry loader reads bytes.
The native proposal branch is present, but no real-bag tick qualifies to exercise it.

A single command reads the real full bag and writes a manifest, normalized messages,
per-tick trace and report. Use an unused absolute output directory:

```bash
PYTHONPATH="$PWD/src" MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl OMP_NUM_THREADS=4 \
  .conda/bin/python -m ada_grasp_ctrl.mpc.mcap_replay \
  --bag /home/hao-fang/ws_fanuc/output/physical_mpc_observe_20261112_194701 \
  --baseline-run "$PWD/output/grasp_pipeline/physical_baseline_20261112_194720" \
  --model-report "$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json" \
  --output-root "$PWD/output/mpc-p1-p2a-replay-new"
```

The CLI invokes the ROS system Python decoder with domain 199 / localhost
configuration, without starting DDS. It then computes in the maintained conda
interpreter. It never imports the streaming publisher or launches drivers.

Fresh evidence: `real-replay-final/`.

- All 75,626 messages from nine topics decoded.
- 52 ticks cover recorded Grasp/Squeeze at 100 ms intervals.
- 52 unavailable, zero accepted/rejected solver proposals, zero eligible observations.
- All 52 ticks have at least one `MISSING` contact, unknown acquisition-clock mapping,
  and unverified frozen-reference alignment. One tick has a multi-point resultant;
  one lies within stage-clock mapping uncertainty. Reason counts overlap.
- Example: tick `1794541681997558784` in grasp; exact contact rows, named measured
  joints, recorded commands, receipt/source clocks and adapter reasons are in
  `ticks.jsonl`. `examples` in the report identifies each blocker separately.
- Receipt-stage clock mapping uses the first paired header/host-monotonic receipt;
  maximum observed residual is about 36 microseconds. This is **not** sensor
  acquisition synchronization: SDK unit and epoch explicitly remain `unknown`.
- Latest-delivered samples are selected without future sample interpolation.
  Named joints are reordered explicitly, and missing/duplicate/nonfinite positions
  are rejected. Original statuses, all raw contact rows and source identity remain
  in decoded output; multi-contact force is never divided among points.
- Timing percentiles in the report measure parser/adapter processing, not solver
  speed. Sequence statistics are explicitly scoped to bag start through the last
  Grasp/Squeeze tick; they do not measure DDS loss. The historical recorder reported
  53 transport-lost messages.

Geometry and force conventions remain provisional. The frozen model is checked
against the packet's kinematic contract, but its grasp reference does not match the
recorded selected path. No reference or contact was fabricated to obtain proposals.
This is counterfactual replay of feedback produced by baseline commands, not MPC
closed-loop hardware performance. P2b physical passive commissioning is not done.

## Focused validation

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl OMP_NUM_THREADS=4 \
ADA_RIGHT_FULL_PREVIEW="$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1" \
  .conda/bin/python -m unittest discover -s tests -p 'test_mpc*.py' -q
```

28 tests passed, including native completion/timeout/safety authority, passive
exception isolation, recorded joint mapping and clock/status gates, and the existing
contact-sign/Jacobian/Ks/optimizer tests. Changed-file Ruff, compile and whitespace
checks passed. No hardware I/O, live ROS publisher test, full unit suite or release
gate was run in this minimal-effort delivery.
