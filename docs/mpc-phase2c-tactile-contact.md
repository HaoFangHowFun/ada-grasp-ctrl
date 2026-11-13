# Phase 2C: tactile point-contact compatibility

The original MPC needs **3D contact force, not sensed local contact torque**.
This supersedes the handoff's possible full-wrench-blocker hypothesis. No original
MPC equation, centroid, grasp matrix, `Ks`, friction coefficient or cm scaling changed.

## Downstream audit before conversion

| Consumer | Fields used | Consequence |
| --- | --- | --- |
| `GraspController.compute_grasp_matrix` | World positions and contact column frames | Centroid is mean contact position; moment rows use `(p-centroid)*100 × force` in N cm |
| `GraspController.Ks` | Local contact point/frame, body identity, measured q and q_cmd | Body-local point Jacobian `[I,-skew(p_local)] J_body`; no contact moment input |
| `ctrl_opt` / problem construction | `contact_force[:3]` | Local force optimization only; no sensed local torque dependency |
| `_CoordinatedWrenchPolicy`, `Controller`, episode runner | First three force components / summed normal | Original force schedule and native solver behavior preserved |
| `control_stat` | `G @ concatenated contact_force[:3]` | Object resultant torque comes from grasp geometry |
| Contact/schema validation and recording | Six-vector shape | Zero trailing values require explicit model-assumption provenance |
| Legacy generic `BaseEval` initialization diagnostic | Norm of the full six-vector | Remains a simulation diagnostic; tactile compatibility does **not** route through it or claim measured six-axis magnitude |
| `hand_util` / MuJoCo adapter | Full simulated wrench, including torsion | Unchanged; simulation moments remain simulated measurements and are not erased |
| Right-full diagnostics | Records six components, force maximum uses first three | Preserve source provenance when reading new contact records |

Local moments of an ideal point contact are **assumed zero**, not measured zero.
The compatibility layer appends three model zeros solely for the six-component
legacy dictionary shape. It records `moment=point_contact_assumed_zero_not_measured`,
the original raw F6 values, and `hardware_eligible=false`. F6 channels 3:6 are never
interpreted as measured torques and their calibration is not a necessary blocker.

## Field and frame compatibility matrix

| MPC quantity | Tactile source / reconstruction | Status |
| --- | --- | --- |
| Hand body identity | `finger` → `right_<finger>_elastomer` | Explicit named association |
| Object body identity | Exactly one supplied task object | External assumption, never sensor observation; ambiguous sets rejected |
| World contact position | `T_world_hand_flange * p_B` | Current full 28-joint model FK, including model mount chain |
| Body-local point | `R_bodyᵀ (p_world-p_body)` | Same convention as MuJoCo adapter |
| Normal / frame first column | `R_world_B * n_B` | Hand → object; normal is not flipped |
| Other frame columns | MuJoCo default isotropic tangent gauge | Declared coordinate choice, not tactile measurement |
| `contact_force[0]` | `n_worldᵀ (-R_world_B f_B)` | Compression must be nonnegative; negative projection rejected |
| `contact_force[1:3]` | Remaining columns of `R_contactᵀ (-R_world_B f_B)` | Signed tangential force retained; circular friction cone checked |
| `contact_force[3:6]` | Point-contact local-moment model zeros | Explicitly **not sensed** |
| Contact Jacobian | Native full-arm body Jacobian at actual contact point | Same body-local transport as `Ks`; FD and `mj_jac` verified |
| Native snapshot timestamp | Oldest contributing monotonic receipt | No freshness renewal; original per-finger times retained |
| Acquisition timestamp / SDK ID | Original adapter metadata | Unknown SDK units/epochs stay unknown; no invented synchronization |
| Contact count | Original reported count, first point only | Multiple contacts rejected because resultant force cannot be allocated |
| Penetration/contact distance | Not sensed | Not invented; original MPC does not consume it |

Specifically `n_Ada=n_Sharpa`, `f_Ada=-f_Sharpa`, then `f_c=R_c.T@f_Ada`.
`object_on_finger` remains a provisional operator observation, not a formally
calibrated force sign. No force/normal changes occur in `TactileAdapter`; visualization
does not determine the canonical normal. The native circular cone is
`mu*f_cx - hypot(f_cy,f_cz) >= 0`. Neither projection nor cone failures trigger sign
repair or force projection.

The default tangent gauge uses Y projected perpendicular to n when `abs(n_y)<0.5`,
otherwise projected Z, and third axis `n × tangent`. MuJoCo contacts with explicitly
oriented anisotropic tangent directions require matching external orientation;
normal-only tactile cannot measure that direction. Native isotropic point-contact
equivalence is tested for both geom orders and multiple non-axis-aligned normals.

## Strict eligibility and replay

`ContactEstimate` distinguishes `eligible`, `valid_empty`, and `insufficient_data`.
No unavailable finger is silently treated as no contact. Following operator
confirmation of Sharpa source semantics, fresh `MISSING` contact points with valid
F6/transport mean `NO_CONTACT`, as does healthy `EMPTY`. These fingers contribute no
contacts; only valid contact points enter the contact set. Five contacts are not
required, but loss of a finger's sensing stream still makes the tick unavailable.
Raw per-finger statuses and source metadata are retained separately from this
compatibility interpretation. State/configuration association,
frame consistency, 200 ms receipt age and 150 ms skew bounds are checked. Repeated
identical reads preserve age; changed payloads with duplicate identity, session
changes and sequence/SDK regressions are rejected. Reinitialization must be explicit.

`TactileAdapter` remains the sole UV→3D and sensor/elastomer→hand-base geometry/FK
implementation. The compatibility layer only consumes its immutable results.
Full-arm FK is reused from the existing validated model; there is no second sensor
geometry map or SDK parser. Unknown calibration profile values are not promoted to
hardware readiness. In particular identity SDK alignment and 0.001 m/native-unit
remain candidate assumptions requiring explicit offline opt-in.

Replay schema version 1 (`ada_tactile_adapter_replay`) stores `origin_s`, provenance,
explicit tactile joint order and arrival-ordered events containing `delivered_at`,
full named feedback and serialized adapter snapshots. It retains source/receive/SDK
timestamps, raw F6, contact count, geometry evidence and invalid/empty statuses.
It does not sort away SDK regressions or reset old timestamps at every subscriber read.
Replay uses the original monotonic timeline; unknown acquisition epochs cannot be
used to certify end-to-end sensor latency.

`tests/fixtures/mpc_tactile_replay_v1.json` is a portable **synthetic equivalent-observation
recording**, not a physical sensor recording. Its contact-on/off, invalid, reordered
and aging events are regression-tested. The original offline delivery did not
include a real-machine recording. An operator subsequently recorded a physical
baseline ROS bag; its decode/status audit is documented in the
[validation follow-up](mpc-phase2-validation-2026-11-12.md#operator-baseline-recording-follow-up).
The real bag now has a separate synchronized MCAP replay entrypoint. Its
[updated contact-status results](mpc-p2a-contact-status-results-2026-11-12.md)
do not certify acquisition clock mapping or calibration. Future baseline captures can use the
[compact recorder](tactile-compact-recording.md) to omit distribution arrays while
preserving raw F6, contact points and source identities.

## Equivalence evidence and go/no-go

Tests compare MuJoCo and equivalent tactile contact frames, world/local point and
3D force, grasp matrix and resultant wrench (including cm scaling). A full real-CRX
model test compares contact Jacobians against finite differences and `mj_jac`, and
compares native `Ks` plus original optimizer outputs with nonzero unknown F6 trailing
channels versus explicit point-contact moments. Tangential forces, negative projection,
friction violation, unavailable/empty/partial/skewed/stale data and ambiguous objects
are covered independently. See [validation](mpc-phase2-validation-2026-11-12.md).

**GO:** eligible offline point-contact inputs, deterministic replay, passive MPC and
guarded active mock tests. Missing local torque calibration is not a blocker.

**NO-GO for physical active MPC:** force direction/units, geometry scale/alignment,
full mount calibration, acquisition synchronization, multi-contact resultant allocation,
tracking/retiming behavior and physical safety commissioning remain unverified.
First perform separately reviewed read-only physical passive observation during the
approved baseline, measure lag/sign/skew/contact validity, then review qualified active
grasp/squeeze only. Physical passive orchestration remains unimplemented; the
passive CLI is currently offline/mock only. The synchronized MCAP replay is offline,
and the no-contact active mock now reports a precise native timeout; see the
[P1/P2a report](mpc-p1-p2a-results-2026-11-12.md).
Neither the operator's baseline recording nor the compact recorder enables physical MPC.
