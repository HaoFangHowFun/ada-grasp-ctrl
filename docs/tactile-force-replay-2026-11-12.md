# Recorded contact-force preview on 8089

The read-only preview is running at http://localhost:8089, paused at bag receipt
+64.5 seconds. Select **Play**, or move **Replay receipt seconds**. Speed is fixed
at 1×; pausing freezes the replay clock and does not age the displayed snapshot.
Only one server may own the port; a second launch fails instead of choosing another
port or stopping somebody else's server.

## Launch

After stopping the previous preview, run from the repository:

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
PYTHONPATH="$PWD/src" MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl OMP_NUM_THREADS=4 \
  .conda/bin/python -m ada_grasp_ctrl.tactile_preview.replay \
  --bag /home/hao-fang/ws_fanuc/output/physical_mpc_observe_20261112_194701 \
  --start-s 64.5 --port 8089
```

Output defaults to a fresh `output/tactile_force_replay_<UTC>/` directory. The CLI
prints its report path and URL. `--decoded <existing decoded.jsonl>` reuses the
existing MCAP decoder output; `--audit-only` writes diagnostics without a server.
The live preview command remains `python -m ada_grasp_ctrl.tactile_preview`.

## Display and scope

Orange is the original object-on-finger force transformed into hand-flange
coordinates; blue is its exact negative. Green is the unchanged outward normal.
All three start at the same contact position. Both force arrows use the same
m/N scale; normal length is a separate display scale. Failed cones remain visible.
The diagnostic table shows finger, Fn, Ft, angle, required mu and PASS/FAIL.
Fn <= 0 displays `NEGATIVE_NORMAL_FORCE` and no friction ratio.

The coefficient is read from the existing `control_eval.yaml` controller setting,
mu=0.3, inherited unchanged by the right-full MPC config factory. Cone slack uses
the existing `friction_cone_slack` function and the existing contact-frame convention.
It does not use the different simulated-surface friction coefficient.

Healthy MISSING/EMPTY hides arrows; invalid, stale, disconnected or abnormal-source
frames show UNAVAILABLE. Multiple points sharing one F6 resultant show AMBIGUOUS,
with no force arrows. Original statuses, F6 and source metadata remain in the
existing per-finger detail panels and decoder output.

Only the existing **hand mesh** is animated. CRX joint feedback is read on the same
receipt timeline and displayed numerically; a full CRX mesh/world animation was not
added. All rendered coordinates are `right_hand_flange`, not laboratory world.
No object mesh is inferred. Geometry/alignment and force calibration remain
provisional. Bag receipt time is explicitly not sensor acquisition time.

## Visual and numeric evidence

Evidence root: `output/tactile-force-replay-20261112/`.

- `replay-64p5s-loaded.png`: actual Chrome/WebGL screenshot after mesh loading.
- `ui-verification.json`: HTTP 200; Play advanced 1.412 s, hand pose changed,
  Pause preserved timestamp, and seeking selected 64.5 s.
- `preview/viewer_state.json`: current named CRX feedback and exact rendered force
  vectors, contact positions, normals and diagnostics.
- `preview/run_report.json`: whole-recording 10 Hz diagnostic summary.
- `preview/run_manifest.yaml`: input paths, receipt semantics, config hash and mu.

At receipt +64.5 s:

| Finger | Fn (N) | Ft (N) | Angle | Required mu | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Index | 3.086 | 2.061 | 33.746° | 0.668 | FAIL |
| Ring | 0.274 | 0.106 | 21.103° | 0.386 | FAIL |

Across 924 latest-past samples covering the 92.385 s recording:

| Finger | PASS | Excess tangential force | Nonpositive Fn | Ambiguous | No contact | Unavailable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Index | 5 | 156 | 0 | 1 | 750 | 12 |
| Ring | 17 | 93 | 0 | 0 | 801 | 13 |

These are sampled diagnostics, not counts of unique sensor frames or physical grasp
successes. Index/Ring valid single-contact failures are due to excess tangential
force relative to mu=0.3, not negative normal force. Median required mu is 0.528
for Index and 0.468 for Ring. No sign, normal, friction or calibration repair was made.

## Changes and tests

Files:

- `src/ada_grasp_ctrl/tactile_preview/viewer.py`: expose existing header/legend
  handles and allow offline replay metadata without pretending there are live DDS
  publishers; live behavior remains unchanged.
- `src/ada_grasp_ctrl/tactile_preview/force_viewer.py`: small subclass adding the
  blue arrow and consistent-frame diagnostics using existing Viser primitives.
- `src/ada_grasp_ctrl/tactile_preview/replay.py`: receipt-indexed decoder playback,
  existing adapter/FK/geometry, minimal controls and diagnostic report.
- `tests/test_tactile_force_replay.py`: numeric, status, receipt-order and real
  renderer/FK regression tests.
- This document.

Focused checks:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl OMP_NUM_THREADS=4 \
  .conda/bin/python -m unittest discover -s tests -p 'test_tactile*.py' -q
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl OMP_NUM_THREADS=4 \
ADA_RIGHT_FULL_PREVIEW="$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1" \
  .conda/bin/python -m unittest discover -s tests -p 'test_mpc*.py' -q
```

Tactile: 48 total, 44 passed, 4 optional ROS tests skipped. MPC: 32 passed.
Changed-file Ruff, compile and whitespace checks passed. Renderer tests confirm
common origins, opposite orange/blue vectors, unchanged normals, colors, no fake
no-contact arrows, ambiguity handling and joint-driven hand FK.

No robot publishers, ROS context, hardware startup, MuJoCo grasp simulation or
Linear Pipeline change. Original TactileAdapter and Ours source hashes are unchanged.
No dependencies were added; screenshot capture used the already installed Chrome.
