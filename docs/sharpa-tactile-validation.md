# Manual right-hand force direction validation

## Recorded operator feedback — 2026-11-12

The operator reported that the read-only direction check was correct and then
confirmed keeping channel indices `0,1,2`: Fx = F6[0], Fy = F6[1], Fz = F6[2].
This is the recorded aggregate manual result for the conditions actually tested.
Per-finger/per-direction channel changes and signs were not supplied. The sign-flip
checkbox defaults to off, but its actual setting during the reported test was not
explicitly confirmed. No numeric capture, measured rate or final joint-state stamp
was provided. The rows below are **not recorded**, rather than assumed passed or
assumed never tested. See the [session summary](sharpa-tactile-session-summary-2026-11-12.md).

## Contact-preview follow-up

The feedback above describes the original F6 session. After the contact-preview
handoff, the operator additionally reported: 「力的方向沒問題 單位是牛頓也合理」
(force direction is correct; Newton units are reasonable). The working mapping
remains `0,1,2`. This supports force direction and the N interpretation for the
observed conditions; no calibrated reference load, per-axis results or sign-flip
setting were supplied with this follow-up. Contact-marker position and the blue
normal remain unverified. Later live contact captures and the full follow-up are
recorded in the [contact-preview report](sharpa-contact-preview-validation-2026-11-12.md).

## Detailed form

Operator/date: __________  Driver host/domain: __________  read_only verified: __________

URDF path: __________  Preview command: __________  Force indices: **0,1,2 (operator confirmed)**

Clock relationship: __________  Pose status/stamp: __________  Sign flip (normally off): __________

Record the physical direction relative to the pad/elastomer triad. “Uncertain” is
appropriate when component order, pose freshness, applied direction, or sign
interpretation is unresolved. Do not infer calibration from software test success.

| Finger | Applied direction | Changed raw F6 channel(s) / sign | Preview arrow direction / sign | Pass / uncertain / fail |
| --- | --- | --- | --- | --- |
| thumb | normal to pad | | | not recorded |
| thumb | tangent A | | | not recorded |
| thumb | tangent B | | | not recorded |
| index | normal to pad | | | not recorded |
| index | tangent A | | | not recorded |
| index | tangent B | | | not recorded |
| middle | normal to pad | | | not recorded |
| middle | tangent A | | | not recorded |
| middle | tangent B | | | not recorded |
| ring | normal to pad | | | not recorded |
| ring | tangent A | | | not recorded |
| ring | tangent B | | | not recorded |
| pinky | normal to pad | | | not recorded |
| pinky | tangent A | | | not recorded |
| pinky | tangent B | | | not recorded |

Idle baseline / finger identification: __________

Telemetry disconnect: arrows disappear and status changes: __________

Telemetry reconnect: only new valid samples restore arrows: __________

Pose missing/stale: static/uncertain label and world arrows hidden: __________

Observed push-versus-force sign relationship: __________

Conclusions and remaining ambiguities: __________
