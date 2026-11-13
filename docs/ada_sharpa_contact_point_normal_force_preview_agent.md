# Agent Implementation Task — Ada-Grasp Sharpa Contact Point + Normal + F6 Live Preview

**Date:** 2026-11-12

**Target repository:** `/home/hao-fang/hugs-workspace/ada-grasp-ctrl`

**Scope:** Extend the **existing** right-hand, read-only Sharpa tactile preview at `http://localhost:8089`.

**Do not replace or rebuild the working preview. Do not change hardware control.**

## 1. Goal / Definition of Done

For each fingertip of the **right Sharpa Wave**, show in the existing Viser 3D hand scene:

1. **Red marker:** 3D contact point obtained from the **first** valid SDK `CONTACT_POINT` row and the verified UV/geometry lookup.
2. **Blue arrow:** geometric contact **surface normal**, rooted at that red marker.
3. **Green arrow:** the **3D measured F6 force** (`F6[0], F6[1], F6[2]`, per the operator's preliminary direction test), also rooted at the same red marker.

The marker and both arrows appear **only for a current, valid contact**. If CONTACT_POINT is missing, explicitly empty, invalid, or stale, hide all three immediately; do **not** retain the last point, substitute the elastomer origin, invent a contact, or reuse a contact from a previous frame. If multiple valid contacts are present, **use row 0 only for now** and show the total count in diagnostics; do not average them or allocate F6 among contacts.

Retain the existing per-finger F6 numeric readouts, URDF, elastomer axes, local/world view, scaling/freshness diagnostics, and read-only operation. This is **visual verification**, not a physical calibration certificate or a control-ready contact estimator.

## 2. Existing implementation to extend (read before editing)

The 2026-11-12 session summary reports an implemented and operator-tested preview:

- Entry point: `.conda/bin/python -m ada_grasp_ctrl.tactile_preview`
- Browser: `http://localhost:8089`, bound by default to `127.0.0.1`
- Source files:
  - `src/ada_grasp_ctrl/tactile_preview/__main__.py` — CLI/server/subprocess
  - `src/ada_grasp_ctrl/tactile_preview/ros_reader.py` — ROS 2 Jazzy/Python 3.12 read-only subscription
  - `src/ada_grasp_ctrl/tactile_preview/data.py` — data checking, freshness and coordinate handling
  - `src/ada_grasp_ctrl/tactile_preview/viewer.py` — Viser/Python 3.10 and URDF FK
  - `tests/test_tactile_preview.py`, `tests/tactile_ros_fixture.py`
- ROS interface: `dual_sharpa_wave/msg/TactileFrame` (+ `TactileBlock`), already implemented in `dual_sharpa_wave_ros2` Phase 1A.
- Right-hand frame topics: `/sharpa/right_hand/tactile/{pinky,ring,middle,index,thumb}/frame`
- Joint states: `/sharpa/right_hand/joint_states`
- Subscriber QoS: `BEST_EFFORT`, `VOLATILE`, `KEEP_LAST` depth 1.
- Installed URDF: `/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave/urdf/right_sharpa_wave_with_flange.urdf`
- Current arrow origin: `right_<finger>_elastomer` origin. Current preview world `W` is anchored to `right_hand_flange` (not the lab/global robot world).
- Preliminary operator observation: direction looked correct with `Fx=F6[0]`, `Fy=F6[1]`, `Fz=F6[2]`; full per-axis sign/alignment certification was not documented. Keep a visible assumption/diagnostic flag.
- Focused preview tests were reported 15/15 passed after the assembly transform fix; inspect current checkout rather than assuming unchanged status.

**Phase 1A validation facts:** `TactileFrame` preserves SDK frame identity/timestamps, F6, `CONTACT_POINT` and `DIST_FORCE` with per-block validity/availability. A real right-index press yielded `CONTACT_POINT` shape `[1,3]` and first row approximately `[124.8047, 134.8828, 11.5781]`; the SDK sample uses the first two entries as image X/Y, but the third field's meaning is **unknown**. Hardware contact data may be MISSING on non-contact frames. Actual message field names, block status enums, shape flattening, SDK-to-finger mapping, and diagnostics **must** be read from the installed `.msg` and Phase 1A contract, not guessed.

**Read these first on the workstation:**

- `docs/sharpa-tactile-session-summary-2026-11-12.md` or the supplied session summary.
- `docs/sharpa-tactile-preview.md`, `docs/sharpa-tactile-preview-validation-2026-11-12.md` if present.
- `dual_sharpa_wave_ros2/docs/tactile_phase1a_contract.md` and `msg/TactileFrame.msg`, `msg/TactileBlock.msg` in the actual checkout.
- Official Sharpa tactile sample/SDK docs implementing `deform_map_uv(channel, row, col, tactile_pn=...)`, plus the actual URDF and available geometry map assets.

Preserve user modifications: inspect `git status`, current branch, and HEAD first; never reset, clean, commit, or push unless separately requested.

## 3. Data and coordinate contract — do not assume missing facts

We want each contact in the same **elastomer frame** `E` used by the existing F6 display, and then to render it in preview world `W` through the existing URDF FK.

- `S`: native SDK **geometry map** frame, origin/axes/units not yet confirmed.
- `E`: `right_<finger>_elastomer` URDF frame.
- `W`: existing preview `right_hand_flange`-fixed world.

For a first contact pixel `(u,v)`:

```text
same atomic TactileFrame
  CONTACT_POINT row[0] -> (u, v)  [SDK image coordinates, likely pixels]
                         -> validated pixel/grid mapping
                         -> static geometry lookup -> p_S, n_S
                         -> verified T_E_S -> p_E, n_E
  F6[0:3] ---------------------------------------> f_E  [operator-validated working assumption]
  joint_states + URDF FK ------------------------> T_W_E
  p_W = R_W_E p_E + t_W_E
  n_W = normalize(R_W_E n_E)
  f_W = R_W_E f_E
```

Equations:

\[
{}^E\mathbf p_c = {}^E R_S\,{}^S\mathbf p_c + {}^E\mathbf t_S,\qquad
{}^E\mathbf n_c = \operatorname{normalize}({}^E R_S\,{}^S\mathbf n_c)
\]

\[
{}^W\mathbf p_c = {}^W R_E\,{}^E\mathbf p_c + {}^W\mathbf t_E,\quad
{}^W\mathbf n_c = {}^W R_E\,{}^E\mathbf n_c,\quad
{}^W\mathbf f_c = {}^W R_E\begin{bmatrix}F6[0]\\F6[1]\\F6[2]\end{bmatrix}
\]

**Crucial unknowns / verification gates:**

1. `CONTACT_POINT[0:2]` are rendered as image X/Y in an SDK sample, but we have **not proven** that they are unmodified `(col,row)` of a 240×240 geometry lookup. Verify image dimensions, origin, indexing, axis swap, rotation, mirroring, scaling, and whether the pixels refer to a 240×240 rather than 60×60 map. No unvalidated implicit `round(v), round(u)` pipeline.
2. Find the **actual static** per-finger XYZ/normal maps available in the workstation/SDK/official assets, or confirm that `deform_map_uv()` is a read-only offline geometry lookup that can run without connecting to hardware. Verify whether maps are 240×240, which assets are for thumb vs four fingers, and channel mapping. Never invent filenames or silently use the wrong map.
3. Identify geometry XYZ units (mm vs m etc), normal convention, reference origin/frame, and how `S` relates to URDF `E`. Do **not** assume `T_E_S = I` just because the URDF has an `elastomer` frame or F6 seems aligned. Use actual SDK source, asset metadata, URDF, or a documented controlled geometry test to establish transform. If evidence is insufficient, expose the candidate mapping only as **UNVERIFIED/diagnostic**, with explicit toggles/config and a visually distinct warning; do not mark it calibrated or integrate into control. Prefer a safe refusal to render a claimed physically aligned point over silently rendering a false position.
4. The third `CONTACT_POINT` component is **unknown**. Preserve it as a raw diagnostic; never label it Z, depth in mm, pressure, or force.
5. F6 is **one fingertip-level resultant**, not per-pixel force. Its first three channels are the current operator-tested working mapping; do not distribute among multiple contacts. Rendering at row 0 is a **visualization approximation** and does **not** prove the force truly acts at that first point or that any F6 torque can be ignored. Keep the existing F6 remaining channels as raw diagnostics.
6. The existing operator supplied Newton units and finger-pad-center origin for F6; do not infer torque units, reference point alignment, or full sign conventions. Do not create a physical wrench transfer just to move the arrow's graphical origin.

## 4. Implementation tasks

### A. Inspect the current code and contracts

- Audit current preview's ROS-to-Viser data path, F6 freshness, joint-state handling, FK, and all Viser object update/remove paths.
- Inspect installed message/interface definitions and actual availability/status semantics. Check whether preview's compact serialized snapshot currently omits `CONTACT_POINT`; extend it minimally so it carries same-frame contact payload, shape, availability and frame identity alongside F6.
- Confirm transport process separation: ROS 2 Jazzy system Python 3.12 and Ada Python 3.10. Avoid importing ROS generated types directly into Ada's Python 3.10 interpreter.
- Inspect valid geometry APIs/files first; document uncertainties rather than guessing.

### B. Reconstruct the first contact row faithfully

- Accept only fresh, valid atomic frames for each finger, with F6 and contact block tied to the **same frame ID**.
- Distinguish `MISSING`, `EMPTY`, `INVALID`, `VALID`, plus stale/unavailable transport.
- Verify declared shape and flattened payload length **before reshaping**. Allow legitimate one-row or multiple-row payloads; check at least two columns, finite first-row `(u,v)`, bounds against the verified lookup domain. Preserve all raw columns; do not squeeze away point count.
- For VALID with `N>=1`, select **first row index 0**. Do not infer order stability, sort by strength or compute centroid. Show `N` and `selected_index=0` in diagnostics.
- For no-current-contact, explicitly publish a **no-contact** state to viewer; remove the red point and both arrows for that finger immediately (not the entire Viser hand), including when frame freshness expires or joint-state transform is invalid.

### C. Static geometry lookup

- Implement an isolated, testable geometry adapter or loader, independent of ROS commands/hardware motion; load static maps **once** per finger type or application startup, not every tactile frame.
- If using `deform_map_uv()`, verify that API's exact argument ordering, output shape, units, frame and hardware connection behavior. If using `.npy`, verify shape/dtype/valid mask and mapping from contact X/Y to map row/col.
- Use nearest valid pixel initially, with clear, documented rounding and bounds policy; leave optional interpolation for a separate enhancement. Normalize normal after finite/nonzero checks.
- Apply a **verified** fixed `T_E_S` for point and normal. If not verified, keep the diagnostic stage visibly unverified and do not claim physically located points.
- Separate geometry lookup/mapping errors from no-contact states in diagnostics. Do not crash the preview when some fingertip lacks a valid map.

### D. Update existing Viser preview (port 8089)

For each right-hand finger (pinky, ring, middle, index, thumb):

- Draw **red contact marker** at selected contact point `p_W`.
- Draw **blue contact normal arrow** with tail exactly at `p_W` and direction `n_W`; adjustable **display-only** length, preferably fixed physical-looking length with clear arbitrary visualization scale.
- Draw **green F6 force arrow** with tail exactly at `p_W`, direction `f_W`; length proportional to measured norm using existing/adjustable display-only scale; do not normalize away force magnitude. Near-zero F6 may hide only the green shaft while preserving the red point/blue normal when contact exists.
- Update markers and arrows at incoming tactile-frame cadence, using existing FK and viewer update facilities; avoid reallocating heavy URDF nodes per frame.
- Make color labels/legend explicit. Show the numeric selected `(u,v)`, raw `CONTACT_POINT` row, point count, geometry XYZ/normal (with confirmed units/frame), transformed coordinates, F6 force and magnitude, source frame ID, contact status, and geometry alignment verification status.
- Keep the original elastomer axes and the existing F6 readout. Avoid displaying a **duplicate green arrow** at the elastomer origin when the contact-rooted arrow is present; if desired, add an explicitly named toggle for legacy-origin debugging, default off.
- Local/world mode should display point and both vectors consistently with the current frame convention. `W` is `right_hand_flange`-fixed. When FK/joint state is stale or unavailable, do not draw a misleading world-space contact point; show a clear status, and allow local-frame diagnostics only if correctly anchored.
- Do not change port 8089, serving address by default, command pathway, or ROS topic/QoS behavior.

### E. Testing (synthetic; hardware not required)

Add focused tests to `tests/test_tactile_preview.py` or adjacent tests:

1. One valid first contact; correct row 0 with input `shape=[1,3]` and float `(u,v)`, preserving unknown third value.
2. Multiple contacts `shape=[N,K]`, `N>1`: select row 0 only; report N; do not average or allocate force.
3. `MISSING`, `EMPTY`, `INVALID`, zero points, malformed flat length, NaNs/Infs, out-of-range pixels: no stale marker/arrows; clear diagnostics.
4. Same-frame atomic coherence: F6 and CONTACT_POINT cannot be accidentally combined from different SDK frames.
5. Synthetic static XYZ/normal lookup with known pixel mapping, row/col orientation, boundary rounding, thumb/four-finger mapping and valid mask.
6. Known rigid transform `T_E_S` and FK `T_W_E`: point rotates **and translates**, normal/force **rotate only**, normal stays unit length; test non-identity transforms (including sign/orientation).
7. Display starts at contact point for both arrows, not the elastomer origin; proper hide/reappear on contact disappearance and stale state.
8. Zero/near-zero F6 contact retains point and normal, hides zero-length force arrow as needed.
9. Missing geometry assets/alignment information causes an explicit warning or safe diagnostic fallback; never an unlabeled identity assumption.
10. Existing no-ROS preview, initial URDF assembly (`update_cfg(hand.cfg)` fix), read-only behavior, and existing F6 display remain passing.
11. ROS subscriber-only path: no command publishers, SDK device connections, `calib_tactile()`, joint commands, motor control, or physical motion.

Use a small recorded/synthetic frame with values comparable to the earlier right-index sample (e.g. `[124.8046875,134.8828125,11.578125]`) only as a **pixel-mapping fixture**, not as evidence of an SDK geometry frame. Do not invent expected XYZ results.

### F. Safe operator validation (only after automated tests)

- Operator starts the **existing** driver with `read_only:=true`; if already running, do not start another. Run existing Ada preview at `http://localhost:8089`.
- Confirm actual topics, message types, freshness and QoS. Observe idle: contact absent => no red/blue/green markers. Then gently touch **right index** at pad center and off-center positions, checking 2D pixel marker (if available), 3D red point, normal and F6 arrow; later other fingers when operator chooses.
- Explicitly capture which geometry mapping and `T_E_S` were used and whether each was verified. Check whether the blue normal points physically outward, and the contact point follows touch position across the pad. Force sign should be reported from observed data and active sign-flip setting; do not claim per-axis calibration without records.
- Record an optional short **read-only** summary (per-finger contact/absence counts, first point raw values, pixel coords, geometry coordinates, normal, force, frame ID and stale/invalid count). Do not need large image buffers in Git.
- Hardware manual touch is user/operator initiated. No robot motion or auto-tare.

## 5. Acceptance criteria

- Existing read-only preview still launches at port **8089** with correct assembled right-hand URDF.
- Every fingertip can show **one red contact point and two co-origin arrows** (blue normal, green F6 force) whenever a fresh valid first contact can be mapped; no-contact yields no marker/arrows.
- Valid multiple-point frames use only first contact; malformed or missing blocks never get reused or silently treated as valid `(0,0)`.
- Spatial coordinates are explicitly labeled as verified or unverified. A verified `S->E` transform is preferred; unknown alignment cannot be described as correct world-position visualization.
- F6 remains the existing three-channel working force vector. No commands, control, calibration, upstream repository edits, hidden topic changes, or implicit force/torque computation.
- Automated tests pass, and a concise actual validation report lists pass/fail/skip plus any remaining geometry-frame ambiguity.

## 6. Deliverables

1. Minimal code changes **within `ada-grasp-ctrl`**, reusing existing tactile preview.
2. Documentation of SDK first-contact UV convention, geometry asset/API, any fixed frame transform and unit evidence, and how to launch/debug.
3. Focused tests and actual command outputs/results.
4. A short Markdown validation report with screenshots optional, explicit unresolved items, and a clearly stated **read-only** hardware check procedure.
5. List changed files and exact launch command. **Do not commit or push.**

## 7. Explicit non-goals

No full contact segmentation, multi-contact force decomposition, normal averaging, contact wrench/torque relocation, slip inference, MPC, closed-loop tactile correction, physically calibrated force control, bimanual support, driver-side SDK changes, or automatic tactile tare. These can follow after the geometry/normal visualization is validated.
