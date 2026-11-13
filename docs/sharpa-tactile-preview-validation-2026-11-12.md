# Local tactile preview verification — 2026-11-12

See the [session summary](sharpa-tactile-session-summary-2026-11-12.md) for the
consolidated operator handoff, startup commands and troubleshooting.

## Latest operator feedback

After the software checks and startup troubleshooting, the operator reported that
the read-only direction check was correct and explicitly confirmed retaining raw
channel indices `0,1,2` for Fx/Fy/Fz. This is an operator-reported manual result,
superseding the earlier statement that no manual test had been reported. No
per-finger/per-direction measurements, final live QoS/rate capture, or explicit
sign-flip state were provided. Do not infer a completed fifteen-row validation
matrix or certified axes/signs from this feedback.

The [manual record](sharpa-tactile-validation.md) now captures that aggregate
observation; its detailed rows remain unrecorded. Historical zero-publisher
observations below describe the initial audit, not the later operator session.

## Follow-up: initial model assembly fix

The operator reported overlapping/misassembled hand meshes on the idle viewer.
Browser inspection reproduced it: `ViserUrdf` creates joint scene nodes at identity
and does not initialize their URDF transforms until `update_cfg()` is called.
The preview previously called it only after receiving valid live joint feedback.
Consequently, fixed and moving joint offsets were absent in the initial browser
scene while the separately calculated fingertip labels used correct FK.

The viewer now applies `hand.cfg` immediately after constructing `ViserUrdf`, so
the static neutral model is assembled before telemetry. No URDF asset, force
mapping, or live/static status semantics were changed. The new regression captures
all initial displayed joint transforms before any feedback and compares them to
the original URDF. It failed before the fix and passed afterward. The focused
suite, including isolated synthetic DDS, now passes **15/15** tests (6.229 s);
changed-file lint, format, compile and diff checks also pass. The initial checks
below remain historical results, not evidence of this later regression check.

The corrected service was restarted on 8089. Real-browser captures
`/tmp/ada-tactile-before.png` and `/tmp/ada-tactile-after.png` show the overlapping
initial meshes and the corrected assembled hand, respectively. Hardware telemetry
is still unnecessary for displaying the static model correctly.

## Scope and checkout

Local checkout: `/home/hao-fang/hugs-workspace/ada-grasp-ctrl`, branch `main`, HEAD
`539b165` (`Integrate right_full Linear and original MPC simulation`). At audit,
HEAD and the locally cached `origin/main` had zero divergence; no fetch was used
to infer the current remote. Numerous existing pipeline/ROS-execution modifications
and untracked dual-live files were present and were preserved. No commit or push
was performed. No dependency, upstream asset, driver, hardware or domain setting
was changed by the preview implementation.

Added implementation paths:

- `src/ada_grasp_ctrl/tactile_preview/{__init__,__main__,data,ros_reader,viewer}.py`
- `tests/test_tactile_preview.py`, `tests/tactile_ros_fixture.py`
- `docs/sharpa-tactile-preview.md`, `docs/sharpa-tactile-validation.md`
- This verification report.

## Interface and live environment audit

The Phase 1A contract and actual installed `ros2 interface show` outputs for
`dual_sharpa_wave/msg/TactileFrame` and `TactileBlock` were read before coding field
access. Installed block statuses are MISSING=0, VALID=1, EMPTY=2, INVALID=3; F6
uses `shape` and `raw_values`, with receipt time in `header.stamp` and explicit
SDK identity/timestamp validity fields. No substitute message was created.

At the initial audit, the shell had ROS Jazzy, `ROS_DOMAIN_ID=87`, and
`ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`. Initial direct discovery with a five-second
spin found only `/parameter_events` and `/rosout`; process inspection found no
running Sharpa driver. Driver host/domain clarification was requested from the
operator; hardware power alone does not establish a publishing ROS driver.

The viewer was started with the documented sourced environment and returned
**HTTP 200 on `http://127.0.0.1:8089`**. Direct `ros2 node info` showed:

- Five right-hand atomic tactile subscriptions.
- `/sharpa/right_hand/joint_states` subscription.
- Only ROS's standard `/parameter_events` publisher.
- No action clients/servers or command publishers; no service clients.

`ros2 topic info -v /sharpa/right_hand/tactile/index/frame` showed **0 publishers**
and **1 subscriber**, the preview using BEST_EFFORT / VOLATILE. Thus, subsequent
topic-list entries can be created by the subscriber itself and are not hardware
communication evidence. That initial audit did not verify live hardware publisher
QoS, `read_only`, joint feedback, per-finger rates or force measurements. The later
operator feedback above is separate evidence and includes no numeric capture.

No live left-hand stream was discoverable at that audit either. The requested application is
right-hand only; no left-hand display or control feature was added.

## Tests actually run

| Check | Result |
| --- | --- |
| Focused preview suite with isolated synthetic DDS enabled | 14 passed, 0 skipped; 6.174 s |
| Canonical full unittest discovery in `.conda` | 393 discovered: 329 passed, 64 skipped; 31.546 s |
| `.conda/bin/python -m compileall -q src tests` | Passed |
| `ruff check src tests script` | Passed |
| `ruff format --check` on all seven new Python files | Passed |
| `ruff format --check src tests script` | Eight pre-existing files would be reformatted; none modified here |
| `git diff --check` | Passed |
| Canonical LEAP Tac3D headless quick example, clean Python 3.10 path | Exit 0; 100 outputs, 93 completed, 7 invalid initialization, 0 execution errors/degraded |

Ruff was the already cached executable at
`/home/hao-fang/.cache/uv/archive-v0/ybMZaBS9Gm8eIiW3/bin/ruff`; it was not installed
or added as a dependency. Full-suite skips include tests requiring explicit real
robot snapshots/ROS environments and the opt-in DDS test, which was run separately.

The focused suite checks explicit channel selection and signs, zero/norm behavior,
known 90-degree rotations around all three axes, invalid rotation rejection,
five-finger metadata mapping, missing/empty/invalid/stale F6, nonfinite/malformed
payloads, preserved SDK identity, bounded owned caches, stale/future ROS stamps,
named current-joint FK and force endpoints, joint limits, pose skew, duplicate
publishers, local/world mode, disconnect/reconnect, and actual HTTP start/stop on
8089. The DDS fixture publishes only synthetic tactile and joint telemetry in
localhost domain 198. Its five received messages and simulated pose drive the
same real viewer adapter; stopping the fixture removes all live arrows. Hardware
driver configuration is unaffected by this test-only child environment.

Raw verification logs are in `/tmp/ada_tactile_focused_tests.log`,
`/tmp/ada_tactile_full_tests.log`, `/tmp/ada_tactile_ruff_check.log`, and
`/tmp/ada_tactile_ruff_format.log`.

The eight existing format differences are in `script/run_crx_sharpa_phase1.py`,
`script/run_crx_sharpa_phase2.py`, `src/ada_grasp_ctrl/crx_sharpa_adaptive.py`,
`src/ada_grasp_ctrl/crx_sharpa_demo.py`, `src/ada_grasp_ctrl/utils/viewer_session.py`,
`tests/test_crx_sharpa_demo.py`, `tests/test_crx_sharpa_preview.py`, and
`tests/test_dummy_arm_ik.py`.

The quick example was run as:

```bash
PYTHONPATH=src PYTHON_BIN=.conda/bin/python \
  ADA_GRASP_CTRL_EXAMPLE_BASE=/tmp/ada_tactile_quick_example \
  bash script/run_example.sh leap_tac3d quick
```

Its fresh result root is
`/tmp/ada_tactile_quick_example/leap_tac3d/20261112T231108Z-393157` and log is
`/tmp/ada_tactile_quick_example_clean.log`. The verified report lists 100 outputs:
87 successful lifts, 6 failures, 7 invalid initializations, and no solver degradation
or execution errors. This is a smoke result, not a new golden baseline. A preceding
attempt inherited ROS Python 3.12 paths into Python 3.10 and failed to import
`pinocchio.pinocchio_pywrap_default`; isolating `PYTHONPATH=src` for the simulation
resolved it without changing dependencies. The actual ROS preview intentionally
preserves the sourced ROS path for its Python 3.12 child and does not import Pinocchio.

## Limits and operator follow-up

The software implementation and synthetic path are verified, and the operator
subsequently reported correct manual-test directions with mapping `0,1,2`. The
full five-finger, multi-direction hardware completion criterion is not documented
at that granularity. The [manual form](sharpa-tactile-validation.md) records the
aggregate feedback while leaving specific measurements unfilled.

F6 indexing/axes/sign are not certified. The inspected SDK Python sample only
labels `F0` through `F5`; default mapping 0/1/2 remains prominently exploratory.
N and pad-center origin are operator-reported inputs. The preview world is the
fixed hand flange, not the laboratory or moving arm world. Default freshness is
subscriber-arrival freshness; source ROS receipt age requires an explicit
synchronized-clock assertion. No SDK sensor latency is claimed.
