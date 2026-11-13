# Agent Implementation Task — Integrate ContactObservation into Existing Tactile Adapter

**Project:** `/home/hao-fang/hugs-workspace/ada-grasp-ctrl`
**Date:** 2026-11-12
**Scope:** Extend existing tactile integration for right Sharpa Wave; prepare a reusable sensing contract for future CoorGrasp-style control. **Read-only; no robot motion.**

## 0. Context and verified baseline

Read the repository's current implementation and local documents **before modifying any files**:

- `docs/sharpa-contact-preview.md`
- `docs/sharpa-contact-preview-validation-2026-11-12.md`
- `docs/sharpa-tactile-preview.md`
- `src/ada_grasp_ctrl/tactile_preview/{data.py,geometry.py,viewer.py,ros_reader.py,__main__.py,capture.py}` (where present)
- Existing tactile adapter(s), if any, in the current checkout; locate by repository search rather than assuming a particular class/file exists.
- `dual_sharpa_wave_ros2` installed interface and `docs/tactile_phase1a_contract.md` in its checkout.

Established from earlier handoff reports:

- Right-hand atomic ROS topics: `/sharpa/right_hand/tactile/{pinky,ring,middle,index,thumb}/frame` of type `dual_sharpa_wave/msg/TactileFrame`.
- Right-hand joints: `/sharpa/right_hand/joint_states`. Reported tactile QoS: `BEST_EFFORT / VOLATILE / KEEP_LAST depth 1`.
- Phase 1A transport preserves same-frame `F6`, `CONTACT_POINT`, `DIST_FORCE`, availability/status, timestamps and source frame identity.
- The existing 8089 viewer already renders a first-contact marker, surface normal and F6 force arrow using UV geometry lookup and URDF FK. It uses `right_hand_flange` as its fixed preview frame, **not lab world**.
- Operator confirmed force direction is correct and Newton units are reasonable, retaining F6 channels `[0, 1, 2]` as Fx/Fy/Fz. Contact-marker placement and the surface normal remain unverified. This is **operator-observed prototype evidence, not calibrated metrology**.
- Geometry candidate uses 240×240 SDK HB1 static point/normal maps, `col=u`, `row=v`, candidate mm→m factor 0.001, and candidate identity `T_E_S`. A mesh comparison gave strong agreement, but formal pixel mapping/frame equivalence and sensor variant were not certified. Preserve this provenance and diagnostic/verification gates; do not silently claim full calibration.
- A `CONTACT_POINT` frame may have shape `[1,3]` or `[2,3]`. Select only row 0 for this initial adapter API. `MISSING`, `EMPTY`, `INVALID`, stale and absent-joint states are distinct.

**Critical:** The previous conversation uses “Tactile Adapter” as the desired architectural owner. It is not evidence that a reusable class with this exact name currently exists. Audit first; if absent, refactor the smallest existing reusable data path rather than creating a parallel SDK/ROS ingestion implementation.

## 1. Goal

Create **one reusable tactile conversion layer** in `ada-grasp-ctrl` which transforms existing atomic tactile frames plus synchronized hand joint states into standardized contact observations. The existing 8089 preview and future CoorGrasp controller must consume the **same derived contact data** (not separately reimplement UV lookup, force rotation or FK).

Target data flow:

```text
Existing Sharpa read-only ROS driver
  -> atomic TactileFrame per finger + right-hand JointState
  -> existing ROS subscriber/bridge (reuse; no extra SDK device connection)
  -> TactileAdapter (one shared conversion/validation layer)
       - block/status and source-frame checks
       - first CONTACT_POINT UV selection
       - static geometry lookup + explicit S->E transform
       - F6[0:3] selection with explicit force convention
       - elastomer -> hand-base FK with joint timestamp checks
       - freshness/diagnostics
  -> ContactObservation(s) + per-finger status
       -> existing 8089 Viser preview
       -> future CoorGrasp controller (NOT implemented in this task)
```

## 2. Scope and constraints

### In scope

1. Audit and **extend the existing tactile adapter/data path**; avoid a second competing `SharpaContactAdapter` and duplicate geometry/FK code.
2. Add a typed, immutable `ContactObservation` result with `position`, `normal`, `force`, frame metadata and timestamps.
3. Provide simple `update(...)` and `get_contacts(...)` style API (adapt signatures to actual project architecture), supporting all five right fingers and optional no-contact values.
4. Reuse the existing static geometry map loader, explicit profile, transform utilities, and hand FK.
5. Make existing 8089 preview consume the same derived observation for marker, normal and force arrow; preserve its existing functionality, user controls, safety gates and CLI.
6. Add unit/integration tests, documentation and a read-only usage example.

### Out of scope

- MPC, grasp control, motor commands, PID changes, calibration/tare, SDK device ownership, or any robot motion.
- Altering the `dual_sharpa_wave_ros2` driver or the atomic ROS contract.
- Inferring contact force at every pixel or splitting aggregate F6 between multiple simultaneous contacts.
- Claiming formally calibrated force, verified physical geometry alignment, or verified F6 torque channels.
- Changing the first-contact selection policy beyond providing a clear future extension point.
- Commit/push unless explicitly instructed.

## 3. ContactObservation contract

Prefer an immutable frozen dataclass or equivalent with validated 3-vector shapes and finite values. Final field names can follow existing project conventions. Suggested conceptual contract:

```python
@dataclass(frozen=True)
class ContactObservation:
    finger: str
    position: np.ndarray          # (3,), meters in frame_id
    normal: np.ndarray            # (3,), normalized, in frame_id
    force: np.ndarray             # (3,), Newtons in frame_id
    frame_id: str                 # explicitly: right_hand_flange or documented hand base
    timestamp: float             # source/receipt provenance documented; no invented epoch
    sdk_frame_id: int
    contact_uv: tuple[float, float]
    contact_count: int
    geometry_profile: str
    force_convention: str         # explicitly object_on_finger (provisional)
    # Optionally: side, source status, receipt stamp, joint stamp,
    # sample age, mapping/calibration quality, raw F6[6]
```

The above is **illustrative**. Make semantics explicit and keep all dependent sample data traceable to the same SDK frame. Avoid raw mutable NumPy arrays inside a nominally immutable object; either copy and set them read-only, use immutable tuples, or equivalent.

**Force convention:** Use the requested provisional `object_on_finger` interpretation; inward reaction force and an outward surface normal can be physically consistent, but this specific push/reaction relationship was not independently recorded by the operator. Retain the raw F6 values, label the provisional nature, and **do not silently negate**. If a future controller needs force on the object, provide an explicit separate conversion `f_finger_on_object = -f_object_on_finger` after conventions are verified. Do not relabel the measurement as object force without that conversion.

### Contact availability API

The intended call shape is simple:

```python
adapter.update(tactile_frames=frames, joint_state=joint_state)
contacts = adapter.get_contacts()
# {"thumb": None, "index": ContactObservation(...), ...}
statuses = adapter.get_statuses()  # distinguish NO_CONTACT, MISSING, INVALID, STALE, etc.
```

Adapt to the existing pipeline (e.g. pass immutable snapshots instead of ROS messages into a ROS-independent core). **`None` must never ambiguously stand for a stale/invalid measurement without exposing its diagnostic status separately.**

## 4. Geometry and coordinate convention

Let `S` be the SDK/static UV geometry frame, `E` the corresponding finger's URDF `elastomer` frame and `B` the hand-base preview frame (currently `right_hand_flange`).

- Select only valid row 0 of `CONTACT_POINT`; use its first two values as `(u,v)` **under the active, explicit candidate pixel-profile**.
- Use exactly the existing UV-to-map convention (candidate `col=u`, `row=v`) and existing nearest-pixel and invalid-mask behavior.
- Geometry lookup returns candidate `p_S`, `n_S`. Apply explicitly configured `T_E_S` and unit scale; do not hardcode identity beyond the existing opt-in candidate profile.
- For joint configuration `q`, use the **existing URDF FK** to compute `T_B_E(q)`.

Equations:

```text
p_E = R_E_S p_S + t_E_S
n_E = normalize(R_E_S n_S)
p_B = R_B_E(q) p_E + t_B_E(q)
n_B = normalize(R_B_E(q) n_E)
f_B = R_B_E(q) f_E,  f_E := F6[0:3] under existing candidate alignment
```

- All three vectors in `ContactObservation` must share the same `frame_id`.
- For current prototype, `B` refers to `right_hand_flange`; **never label it lab/world frame**. Future CRX-to-world is a separate transform step.
- Do not flip the normal or force merely to make arrows visually align. Normal outward and sensor reaction force inward can be physically consistent.
- Only produce `ContactObservation` when source contact block, F6, geometry, and required FK are valid/fresh and allowed by current geometry-verification configuration. Otherwise return `None` with a precise status.
- Preserve diagnostics regarding profile/source variant, physical alignment confidence, geometry-map hash, assumptions and possible limitation of one aggregate force applied at first selected point.

## 5. Timing and lifecycle

- Match `CONTACT_POINT` and `F6` from the **same atomic `TactileFrame`**, never latest separately sampled topics.
- Joint states are a distinct stream. Use existing receipt timestamps / monotonic freshness logic and the best available joint-state synchronization. Document clock domains explicitly. Reject or mark invalid when joints are unavailable or too old.
- No stale contact may persist in `get_contacts()` after contact goes `MISSING`, `EMPTY`, `INVALID`, ages out, or ROS disconnects. Make stale-age policy configurable; no hard-coded assumption that 30 Hz source means perfect 30 Hz delivery.
- Preserve SDK frame ID, receipt/source timestamps and diagnostic counters where feasible; avoid treating SDK `ts` as synchronized wall-clock unless proven.
- Make the update/reader path bounded and thread-safe as required by the existing process separation. No duplicate ROS subscriber, device SDK object or hardware commands.

## 6. Existing 8089 preview migration

- Refactor 8089 viewer so it takes standardized derived data from the **same adapter**, instead of keeping a second independent `CONTACT_POINT -> geometry -> FK` implementation.
- Preserve right-hand mesh assembly, five finger labels, controls, local/world viewing, selected F6 channels, scale, zero threshold, opt-in diagnostic sign flip, connection/freshness state, and invalid/missing visual gating.
- A valid first contact renders the red point and blue normal and green F6 vector sharing that point as origin.
- No contact or invalid/stale sample hides all three contact graphics promptly; don't freeze the last contact position.
- Preserve diagnostic markings such as `UNVERIFIED GEOMETRY` when profile evidence is preliminary; operator-observed plausible agreement must not silently promote profile to certified calibration.
- Confirm the user-visible result in `http://localhost:8089` via existing non-hardware tests; do not assume a physical server is running.

## 7. Tests and acceptance criteria

Add tests using synthetic frames / isolated DDS fixtures; do not command a real robot.

1. All five right-finger mappings and expected output-frame labels; unsupported channels rejected.
2. One valid contact (`[1,3]`) yields one observation with exact UV, `contact_count=1`, normalized normal and correctly transformed position/force.
3. Multi-contact (`[2,3]`) selects only first row and preserves count/provenance.
4. F6 `[0:3]` order and sign unchanged; explicit `object_on_finger` convention; force magnitude/rotation correctness.
5. Nonidentity `T_E_S` and nontrivial `T_B_E(q)` transform both position and directions correctly; zero/invalid normals rejected.
6. Geometry lookup outside map, invalid map masks, wrong source profile, unsupported/unknown units and unavailable verified geometry are surfaced as statuses, **never fake valid contacts**.
7. `MISSING` vs `EMPTY` vs `INVALID` contact blocks; invalid F6; no contact; stale tactile; stale/unavailable joints; no reuse of previous contact.
8. Atomic same-frame association and SDK identity preserved, including reordered/duplicate callback scenarios as supported by current reader.
9. Existing 8089 UI still shows matching point/normal/force from exactly the same `ContactObservation` data and hides them on release.
10. Existing unrelated tests, existing command topics/QoS, read-only behavior and other modalities unchanged; no command publisher, SDK connection, tare or motor call introduced.
11. If possible, a deterministic regression comparing the new adapter output against the existing pre-refactor 8089 geometry math for a fixed valid frame (including actual captured sanitized values).

Record **actual** tests run (pass/fail/skip) and any skipped live checks; do not claim hardware certification based on synthetic tests.

## 8. Deliverables

- Reusable `TactileAdapter` (or appropriately named extension of the existing tactile class/path) and immutable `ContactObservation` type.
- Shared geometry/FK pipeline used by both ContactObservation and the existing 8089 viewer.
- Public, importable API and a short standalone **read-only** example showing five-finger `get_contacts()` plus `get_statuses()`.
- Updated documentation explaining source topics, candidate geometry profile, coordinate frames, force convention, units, freshness and exact no-contact behavior.
- Tests, outcomes and concise validation report, with changed paths and a clear list of **verified vs operator-observed vs still unverified** properties.
- Do **not** commit or push.

## 9. Definition of Done

A consumer inside Ada can obtain per-finger `(position, normal, force)` in the **same declared hand-base frame** from existing raw tactile + joint input, with explicit status and provenance, without importing Viser or connecting directly to the SDK. The existing 8089 preview uses **that same computed observation** with no duplicated physical conversion. The path is safe/read-only, tested, supports first-contact-only semantics, rejects stale or invalid input, and truthfully labels candidate geometry/force conventions. **No CoorGrasp MPC is implemented yet.**
