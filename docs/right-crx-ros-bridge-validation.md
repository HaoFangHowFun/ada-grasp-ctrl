# Ada-owned receiver migration — 2026-11-10

The passive receiver now lives in `src/ada_grasp_ctrl/ros_execution/`.
The bimanual executable is a compatibility import only. Ada imports driver
interfaces, but no bimanual task/runtime modules. The action name stays unchanged.
Speeds, accelerations, tolerances and timeout values remain at their original
defaults; no slowdown was applied.

Current evidence: `output/ros_bridge/ada-receiver-migration-v3/run_report.json`.
This fresh domain-88 audit used the existing `g0000-export-v1/targets.json` and
started only right-side mock drivers. It did not rerun the five-cloud selector
or execute its selected grasps on hardware.

| Check | Result |
| --- | --- |
| Idle / invalid goal | Zero commands before a valid goal; wrong names rejected |
| Three source targets | Exact 28-axis arrays preserved; all three reached |
| Planned duration | pregrasp 25.153947 s; grasp 2.2 s; squeeze 2.2 s |
| Complete sequence | 2,887 arm and 2,887 hand command messages |
| Separate canceled goal | Sender exited 130; cancel acknowledged and result received |
| Receiver outcomes | DONE, DONE, DONE, ABORTED; controller deactivation acknowledged |
| Left-side commands | Zero |
| Core migration comparison | 23 AST comparisons equal after removing type annotations |
| Ada full unit suite | 243 run, 236 passed, 7 skipped (6 ROS-only, 1 OpenCV) |
| ROS receiver unit suite | 6 passed under sourced Jazzy system Python |
| Bimanual unit suite / compatibility entry | 68 passed, 10 opt-in launch tests skipped |
| Static checks | Ruff lint, compileall and diff checks passed; changed files formatted |

The AST comparison covers goal parsing, duration calculation, acceptance and
execution, model checks, feedback handling, preflight, watchdog and abort logic.
Runtime node names and source-log ownership changed. Source hashes, resolved
configuration, actual trajectories and controller-stop results are retained per
goal. `migration_validation.json` records the comparison and original-config check.

The sender's 2.0-second floor plus existing margin produces a 2.2-second minimum
planned duration. Each phase also requires actual arrival and 0.3 seconds of
stable feedback. The 8-second receiver settling timeout is an extra deadline,
not a fixed pause. The above durations apply to this sample and mock initial pose.

Diagnostic runs `ada-receiver-migration-v1` and `v2` completed all three phases,
but their separately launched cancellation senders rejected stale startup
feedback before sending a goal (hand, then arm). These runs remain failed and
retained. The sender now logs rejected timestamps and ages for diagnosis. No
freshness threshold or execution behavior was relaxed; v3 passed, but the
intermittent startup issue is not claimed resolved. No automatic retry was added.

Repository-wide formatting still reports the eight pre-existing files listed
below. This migration does not alter the MuJoCo control pipeline or scientific
trajectories; no release golden or hardware validation was performed.

## Historical validation — 2026-11-09

The corrected integration verifies **Ada-originated** pregrasp, grasp and squeeze
commands, rather than a standalone task's synthetic YAML targets. All ROS work
used mock drivers on domain **88**. No hardware test was performed.

Accepted evidence:

`output/ros_bridge/integration-v3/run_report.json`

The prepared source was explicitly selected as `g0000` from
`output/regression/crx-sharpa-right-preview-x080-y070-config-v1`. The exporter used
Ada Python 3.10, verified the existing preview and dependency hashes, and preserved
`pregrasp_qpos`, `grasp_qpos`, `squeeze_qpos` exactly. No replacement poses or new
registration were generated. The system-Python sender checked the source model's
28 joints and world/base/mount/flange/palm kinematics against the live ROS model.

| Check | Result |
| --- | --- |
| Passive receiver idle | 0 arm/hand commands and no command publishers before a client goal |
| Wrong joint names | Goal rejected before command output |
| Target origin | All three action goal position arrays exactly equal the exported Ada arrays |
| pregrasp | Action SUCCEEDED, error code 0, actual target reached |
| grasp | Action SUCCEEDED, error code 0, actual target reached |
| squeeze | Action SUCCEEDED, error code 0, actual target reached |
| Complete sequence commands | 2,829 right-arm and 2,829 right-hand messages observed |
| Left-side commands | 0 arm and 0 hand messages |
| Ctrl+C during a separate goal | Ada exited 130, cancellation acknowledged, final cancellation result received; no next phase |
| Receiver stop | Canceled goal recorded ABORTED and arm-controller deactivation acknowledgment |
| Cleanup | Audit stopped only the mock/bridge processes it created |

The sender initiated exactly three goals. The receiver handled each as an independent
28-axis target, with no pregrasp/grasp/squeeze policy, no generated lift, and no
automatic cycle. Each goal's actual trajectories and results are in `receiver/`;
client feedback/results and input packet are in `sender/`. Cancellation evidence
is in `cancel_sender/`. `validation.json` records code hashes and receiver outcomes.

Validation commands/results:

- Ada: `PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl .conda/bin/python -m unittest discover -s tests -p test_right_ros_transport.py -v` — 9 passed.
- ROS: `/usr/bin/python3 -m pytest -q /home/hao-fang/ws_fanuc/src/bimanual_manipulation/test` — 44 passed, 10 existing opt-in launch tests skipped.
- `ruff check src tests script` — passed.
- `ruff format --check` on all five new Ada Python files — passed.
- `.conda/bin/python -m compileall -q src tests` — passed.
- `git diff --check` in both repositories — passed.

The repository-wide `ruff format --check src tests script` reports eight pre-existing
files outside this change that would be reformatted. Those files were left intact:
`script/run_crx_sharpa_phase1.py`, `script/run_crx_sharpa_phase2.py`,
`src/ada_grasp_ctrl/crx_sharpa_adaptive.py`, `src/ada_grasp_ctrl/crx_sharpa_demo.py`,
`src/ada_grasp_ctrl/utils/viewer_session.py`, `tests/test_crx_sharpa_demo.py`,
`tests/test_crx_sharpa_preview.py`, and `tests/test_dummy_arm_ik.py`.

Earlier diagnostic runs are retained, not counted as final acceptance:
`integration-v1` reached all three targets but exposed a SIGINT/ROS C-extension
cancellation acknowledgment issue. The client now sets an interruption flag and
cancels outside the C-extension call. `integration-v2` failed during mock startup
because the linter tool environment's Python was first on PATH; the audit now pins
its child processes to the invoking system Python directory. `integration-v3`
passed the complete sequence and cancellation checks.

The bridge currently rejects non-mock configuration. Target arrival does not
prove object contact, grasp stability, collision-free transit, or physical stopping.
`object_lifted` and `object_held` remain null. No online Ada contact adaptation,
MuJoCo control episode, hardware acceptance, or 2,000-pair evaluation was run here.

The old `right_arm_hand_task` YAML demonstration is not part of this chain. It is
retained because existing preparation references its initial configuration and
hashes; those existing user artifacts were not changed.
