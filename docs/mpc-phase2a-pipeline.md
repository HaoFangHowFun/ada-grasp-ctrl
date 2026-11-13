# Phase 2A: pipeline modes and offline orchestration

Implemented against initial HEAD `7907b4925cc4c84d76e2898c5f39016f66716eff`.
The working tree already contained dual-arm and receiver edits; these were retained.
This delivery operates offline/mock only. It does not commission physical MPC.

## Modes and compatibility

`script/run_grasp_pipeline.py --controller-mode` accepts `baseline`, `linear`,
`mpc_passive`, and `mpc_active`. The existing `--mode live|replay` continues to select
perception inputs. The default controller remains `baseline`, including existing
hardware safeguards, reviewed packet hashes, endpoint targets and stage sequence.
No checkpoint, ranking, pose mirroring, camera or inference change was made.

| Controller mode | Execution in this delivery |
| --- | --- |
| baseline | Existing reviewed action receiver, legacy quintic Ada reference |
| linear | Explicit right/mock receiver setting and matching reviewed preview; linear Ada reference followed by bounded command supervision |
| mpc_passive | Offline mock plant follows baseline references; only grasp/squeeze compute proposals; the passive object has no publisher/action capability |
| mpc_active | Offline mock plant accepts guarded native MPC targets only in grasp/squeeze; other stages retain baseline references |

The new modes reject hardware before execution. An offline packet also rejects
hardware regardless of controller mode. `controller.hardware_enable` must be false.
Physical passive observation while a baseline executes is a **future integration**,
not an enabled mode of this delivery. `mpc_passive` does not refer to a Shadow hand.

## One reference and explicit ownership

`mpc/reference.py` implements exactly
`q0 + clip((t - t0) / T, 0, 1) * (qf - q0)`. `LinearReference` also checks endpoints,
limits and monotonic sampling. Pipeline preview, the new linear receiver and native
right-full reference helper share this sampler. Existing legacy samples retain
their smoothstep behavior. A linear plan requires a matching linear receiver and
cannot silently reinterpret an existing quintic review.

The preview depicts `q_ref`. The safety supervisor can delay `q_100Hz` to meet speed,
step and acceleration limits; it is logged separately. Neither is a claim that the
downstream CRX quintic trajectory or physical motion is linear.

`StageOrchestrator` requires ordered stages and arrival acknowledgment before advancing.
Only grasp/squeeze can call MPC. Cancellation, stale feedback and invalid stage
sequence latch failure. Invalid MPC proposals remain diagnostics in passive mode;
they never select an active owner. Offline execution preserves velocity state across
ownership transitions, checks arrival and settle duration, and fails on arrival timeout.

Logs distinguish named measured `q`, last command `q_cmd`, linear `q_ref`, proposal
validity and timestamps, baseline/MPC owner, and actual mock `q_100Hz`. A proposal row
always has `proposal_executed=false`; command rows separately identify actual output.

## Reproducible offline use

Use the maintained Python 3.10 interpreter with `PYTHONPATH` set only to this repo's
`src`, not inherited ROS Python 3.12 packages. All output directories must be new.

```bash
export PYTHONPATH="$PWD/src" MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl MUJOCO_GL=egl
.conda/bin/python script/run_grasp_pipeline.py \
  --mode replay --environment mock --controller-mode mpc_passive \
  --offline-packet "$PWD/output/grasp_pipeline/cycle_enter_audit_v1/pipeline/group01/export/targets.json" \
  --mpc-scene "$PWD/output/regression/crx-sharpa-right-preview-x080-y070-config-v1/g0000/scene.xml" \
  --output-root /tmp/ada-phase2-passive-new
```

`--offline-packet` skips capture, inference, viewer and live ROS execution and verifies
the existing hash-bound packet first. Packet model kinematics and reviewed target
identity are checked again by the offline executor. `--mpc-scene` is an **explicit
kinematic MuJoCo contact fixture** at measured mock q, not a dynamically validated
camera-object grasp. Dynamic regression is provided separately by right-full tests.

Alternatively use `--tactile-replay tests/fixtures/mpc_tactile_replay_v1.json` (prefer an
absolute path). This portable fixture is explicitly synthetic. Recorded contacts
must agree with the mock state; a mismatched or stale recording yields no valid
proposal. Omitting a source also produces explicit unavailability, never valid empty
contacts. Supplying multiple sources is rejected.

The ordinary pipeline still exports selected targets and uses the Enter review gate.
For MPC modes its post-review branch runs the offline executor, rather than starting
an actuator receiver. ROS streaming is a separate mock-only interface described in
[Phase 2B](mpc-phase2b-executor.md).

## Evidence and limits

See [the validation report](mpc-phase2-validation-2026-11-12.md) for exact commands,
counts, original/current quick comparisons and fresh artifact paths. Tests cover
28-joint midpoint/endpoints, preview/reference parity, legacy samples, stage/cancel/
stale failures, and passive zero-publish ownership. The known seven-stage mock cycle
completed with identical baseline/passive command arrays. Physical arrival, force
response and successful closed-loop MPC grasping remain untested.
