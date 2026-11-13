# P2a contact-status update — 2026-11-12

The operator confirmed that Sharpa normally omits contact points when there is no
contact. This updates the earlier handoff's assumption that all `MISSING` contact
blocks must reject the tick.

Only MPC compatibility and replay handling changed. Original `TactileAdapter`, ROS
drivers and Ours controller were not modified. Raw source status is never rewritten.

| Input | Compatibility behavior |
| --- | --- |
| VALID point contact | Existing geometry, identity and force checks; add the accepted point to the contact set |
| Fresh MISSING, valid F6/transport, consistent source | NO_CONTACT; no point added |
| Healthy EMPTY | NO_CONTACT; no point added |
| INVALID, stale, disconnected, missing stream, inconsistent source | UNAVAILABLE; no fabricated no-contact observation |
| Mixed contacts and healthy no-contact fingers | Use the valid contacts; five contacts are not required |
| All fingers healthy with MISSING/EMPTY | Valid empty contact set |

`ContactEstimate.provenance.fingers` preserves every original frame, including
fingers after an early rejection. `finger_statuses` separately records raw status,
adapter/transport status and CONTACT / NO_CONTACT / UNAVAILABLE interpretation.
An unexamined finger after an earlier failure explicitly has `detail: not evaluated`.
The existing aggregate API retains `insufficient_data` for unavailable observations.

## Real MCAP rerun

Fresh output: `output/mpc-p2a-contact-status-20261112/real-replay/`.

| Metric | Result |
| --- | ---: |
| Decoded messages | 75,626 |
| Grasp/Squeeze 10 Hz ticks | 52 |
| Contact-eligible ticks | 0 |
| Fully eligible MPC ticks | 0 |
| Native MPC proposals / accepted proposals | 0 / 0 |
| Unavailable ticks | 52 |
| MISSING-contact blockers | 0 |

Remaining contact blockers are **51 friction-cone failures** (49 index, 2 ring)
and **1 index multi-contact resultant** that cannot be assigned to individual points.
For example, pinky is correctly NO_CONTACT at all 52 ticks. The previous MISSING
rejection had prevented inspection from reaching these later contact checks.

Independent blockers remain at all 52 ticks: SDK acquisition units/epoch are unknown,
and the recorded selected reference is not aligned with the frozen native reference.
One tick lies within phase-clock boundary uncertainty. Counts overlap.
The force convention, candidate geometry, friction coefficient and solver were not
changed to bypass these failures. No real-bag observation reached the solver.

`run_report.json` records `contact_eligible_ticks`, `eligible_observations`,
`native_proposals`, `compatibility_blockers` and examples. `ticks.jsonl` contains the
per-finger interpretations alongside the unchanged adapter output and raw source
metadata. Replay is counterfactual and creates no ROS nodes or robot publishers.

Reproduce with an unused output directory:

```bash
PYTHONPATH="$PWD/src" MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl OMP_NUM_THREADS=4 \
  .conda/bin/python -m ada_grasp_ctrl.mpc.mcap_replay \
  --bag /home/hao-fang/ws_fanuc/output/physical_mpc_observe_20261112_194701 \
  --baseline-run "$PWD/output/grasp_pipeline/physical_baseline_20261112_194720" \
  --model-report "$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json" \
  --output-root "$PWD/output/mpc-p2a-contact-status-rerun"
```

## Focused regression

32 MPC tests passed, including four new mixed-contact tests: Thumb/Index VALID with
other fingers MISSING, all-no-contact MISSING/EMPTY, sensing failures, and source
regression. Fixtures preserve source snapshots and verify that unavailable sensing
never becomes NO_CONTACT. No MuJoCo grasp episode was rerun.

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl OMP_NUM_THREADS=4 \
ADA_RIGHT_FULL_PREVIEW="$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1" \
  .conda/bin/python -m unittest discover -s tests -p 'test_mpc*.py' -q
```

Changed-file Ruff, compile and whitespace checks passed. Input/output source hashes
are saved in `output/mpc-p2a-contact-status-20261112/`. No commit or push.
