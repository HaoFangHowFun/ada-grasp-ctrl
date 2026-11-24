# Switchable Native Sharpa models in the dual live pipeline

The existing `script/run_dual_grasp_pipeline.py` now accepts `--grasp-model
right_full|native_multimode`. Omission retains the stable 100k Right-Full checkpoint,
checksums, per-side inference, IDs and selector behavior. The selector still defaults
to `legacy`; explicitly select `hierarchical` for the native model. An incompatible
native/legacy combination fails before capture or subprocess startup.

The integrated paired implementation and current validation are documented in
[native-bimanual-hardware.md](native-bimanual-hardware.md). `--grasp-mode auto` is
the default; `both_only` filters model-available modes after inference.

## Hardware-connected validation commands

These commands are for the operator. Implementation validation did not capture a new
camera image, connect to ROS, or command a physical robot. `--export-only` captures,
reads robot state and writes the existing motion packet without starting execution.

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
.conda/bin/python script/run_dual_grasp_pipeline.py \
  --environment hardware --ros-domain-id 87 \
  --selector hierarchical --arms auto \
  --grasp-model right_full --export-only \
  --output-root "output/dual_live/right_full_$(date +%Y%m%d_%H%M%S)"
```

```bash
cd /home/hao-fang/hugs-workspace/ada-grasp-ctrl
.conda/bin/python script/run_dual_grasp_pipeline.py \
  --environment hardware --ros-domain-id 87 \
  --selector hierarchical --arms auto \
  --grasp-model native_multimode --export-only \
  --output-root "output/dual_live/native_200k_$(date +%Y%m%d_%H%M%S)"
```

Use `--arms left` or `--arms right` to restrict execution candidates. To inspect the
8089 viewer without execution, replace `--export-only` with `--preview-only` in a
fresh run. Removing both flags retains the existing viewer and terminal Enter review
before motion; the operator decides when to do so.

## Verified model and selection contracts

`dual_grasp_pipeline.yaml` holds both model references; the native run is
`sharpaDualMulti_robotMultiHierar_native_sharpa_all5_50k_v1/ckpts/step_200000.pth`.
Its real Hydra configuration specifies independent sigmoid availability, threshold
0.9, minimum one available type, score prior enabled, 100 samples/type and per-type
Top-10. The adapter calls the actual model availability mask and DexLearn
`_sample_save_mask`, including the model's top-score fallback. It does not normalize
the five scores or fill in missing modes. Equal availability scores use lexical mode
order, as defined by the existing hierarchical selector.

The checked head is `DiffusionBiRT_MLPRTJ`, with three stages and 44 finger joints.
Actual output is `(500, 3, 58)`: right XYZ/WXYZ/22 fingers, then left XYZ/WXYZ/22
fingers. The stages are pregrasp, grasp, squeeze. The dataset rotates points and
poses only during training; inference samples and centers metric world-axis points
without rotation or scale changes. The existing decoder restores the sampled cloud
centroid once and does not extrapolate pregrasp. All 22 joints remain present for
two- and three-finger contact modes. Five real metadata files are checked against the
stored Sharpa joint order and flange names; the bimanual metadata's 56 simulator
joint names are not mistaken for learned pose values.

Original-cloud Auto-Mode scores rank **source modes**. Right candidates use the
original cloud; left candidates use the established reflected cloud, forced
corresponding right-hand mode, and `SharpaHandMirror` conversion. Both candidates
retain `mode=right_three`, for example, while their `execution_mode` values are
`right_three` and `left_three`. IDs include side, source mode and raw sample index.
The original scene's score stays in both branches' provenance.

The existing hierarchical selector evaluates both allowed sides in each source
mode, selects minimum squared CRX displacement within the first feasible mode,
and does not run IK on lower-ranked modes after selection. `both_three` and
`both_full` now decode paired original-cloud targets and use independent arm IK
with combined-state collision checking. Their effort is the sum of both six-joint
squared displacements. Only model-available modes receive a priority.
`both_only` restricts feasibility to available paired modes without changing scores
or sampling. Bimanual candidates require both enabled arms.

One DexLearn subprocess loads one checkpoint per capture. It eagerly samples the
available mirrored single-hand modes before returning to Ada. This avoids adding
a persistent process or request protocol. The current DexLearn `sample()` generates
all five modes even for a forced type, then masks them; those extra raw predictions
are retained. Only Top-K selected records are decoded for each enabled side and sent
to the existing CRX checker. `--arms left` runs only the original backbone/availability
head for ranking, then mirrored sampling. Robot conversion stays in Ada's environment;
no MuJoCo or Pinocchio dependencies were added to DexLearn.

## Original single-arm integration evidence on 2026-11-23

This section records the earlier single-arm implementation. Current paired evidence
is in [native-bimanual-hardware.md](native-bimanual-hardware.md).

`candidates.json` records the checkpoint, all five scores, ranked available and eligible
modes, conversion rejections, reflection plane and branch timings. Each native branch
retains all 500 raw samples and the actual Top-K/availability masks in
`inference_native/*.npz`. `hierarchical_selection.json` records source-mode skips and
the selected execution mode. `run_report.json` adds model, source/execution mode,
side, effort and capture/inference/selection/export durations. The existing 8089
markdown panel displays the same model/mode/score information.

Real GPU runs on the RTX 5090 loaded the 200k checkpoint strictly and used saved
clouds. No candidate was padded, clamped or repaired:

| Saved cloud | Auto-Mode priority | Canonical executable candidates |
| --- | --- | ---: |
| `dual_live/auto_min_effort_20261122_164058/object` | right_two, right_full | 39 |
| `fixed_camera/bottle_group04_object` | right_full, right_three, both_full, both_three | 40 |
| `fixed_camera/cube_group03_object` | right_three, right_full, right_two | 59 |
| `fixed_camera/adapter_group05_body_object` | right_two | 20 |

This covers actual original-right and reflected-left predictions for all three
single-hand modes and raw predictions for available bimanual modes. The first and
cube runs each rejected one left conversion through the existing Sharpa joint-limit
mapping. Separate real left-only and right-only runs each produced 20 candidates
on the first cloud and retained the original-cloud availability ranking.
An independent raw-array audit checked all 198 resulting candidates against their
actual Top-K masks, original scores, centroid restoration, WXYZ rotations and stored
22-joint reflection mapping. Its report is
`/tmp/ada_native_real_contract_validation_20261123.json`.

Using that first cloud's **saved** calibrated dual scene and saved 56-joint robot
state, real IK/joint-limit/collision selection chose `right_two` executed on the left,
with effort **0.3082390589 rad²**. The existing exporter and packet loader accepted
the unchanged eight-stage packet. Evidence is at
`/tmp/ada_native_200k_offline_20261123_v2/`, including `export/targets.json` and
`offline_selection_report.json`. These are offline artifacts, not commands to reuse
on current hardware.

One-run wall times on the same saved cloud:

| Measurement | Seconds |
| --- | ---: |
| Stable 100k: two inference processes, 8 samples/side, including conversion | 4.813 |
| Native 200k: one inference subprocess, 100 samples/type, Top-10 | 3.705 |
| Native worker setup/model load/hash/imports | 2.154 |
| Native original Auto-Mode sampling and raw save | 0.370 |
| Native mirrored right_two sampling and raw save | 0.066 |
| Native mirrored right_full sampling and raw save | 0.067 |
| Native real saved-scene selection | 2.917 |
| Native existing packet export/validation | 0.109 |

Native subprocess timing excludes parent canonical conversion, while the legacy
aggregate includes conversion and reference verification. Sampling counts differ;
these are observed runs, not a hardware-independent speedup claim. Capture and live
robot-state latency were not measured. Further saved-cloud native subprocess runs
took 2.6–2.8 seconds with warmed filesystem caches.

Focused tests cover legacy defaults/checksums, all three mode conversions, WXYZ and
world-flange transforms, all 22 joints, deterministic IDs/ties, source versus execution
labels, both-side effort comparison, fallback, early stop, enabled arms, unsupported
dual modes, the existing packet builder/lineage and viewer text. Run:

```bash
PYTHONPATH=src:tests MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl .conda/bin/python -m unittest \
  test_native_multimode_pipeline test_dual_hierarchical_pipeline \
  test_dual_live_pipeline test_hierarchical_selection test_grasp_contracts \
  test_fixed_camera_grasp -v
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl .conda/bin/python -m unittest discover -s tests -v
.conda/bin/python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
```

The full suite ran 591 tests successfully (86 existing optional/environment skips).
The final focused native suite, including the added viewer-text check, passed all
nine tests. Full Ruff lint and compile checks passed. Repository-wide formatting reports eight pre-existing
unformatted files outside this change; all changed Python files pass formatting.
The unchanged upstream availability/forced-mode/fallback tests also passed:
`cd ../HUGS-DexLearn && PYTHONPATH=. OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests/test_robot_hierarchical.py -q`
(six tests).
The initial implementation checks did not test physical grasp success, fresh-camera
integration or browser rendering. The viewer's displayed text is covered offline.
The subsequent camera-only recalibration is recorded below.

## Table update after operator testing

Review of the operator's first native runs found that the configured table plane
was about 91 mm below the observed tabletop. Object extraction therefore retained
large patches of table, shifting the input centroid and changing mode availability.
Model-to-FK-to-export pose checks agreed to numerical precision. Offline extraction
with a plane fitted to each saved capture improved alignment and restored executable
single-hand candidates for both large-object captures; no model thresholds changed.

At the operator's request, three fresh passive camera captures were then taken on
domain 87. The table had a different position from the review captures, about
52.75 mm above the former configured plane. The dual-live YAML now overrides
`table_plane` with:

`output/fixed_camera/table_recalibration_20261123T215945_898848Z/fit/table_plane.json`

Primary fit RMS was 1.77 mm; distributed repeat fits disagreed by at most 4.67 mm,
passing the existing thresholds. The current cube extraction retained 2,468 points
instead of 6,548 with the old plane; a separate empty region returned no object.
The calibration directory contains `table_review.png`, `calibration_review.json`,
`application_verification.json` and `RECALIBRATION.md`. The plane SHA-256 is
`a4f1409f7fee2e4cd2a891baf7e9047c13037d296008c61995a30054351adcdf`.

The camera extrinsics and historical planes remain unchanged. No robot commands
were sent during recalibration. No independent world reference points were supplied,
so the fit retains `preview_unaccepted`, supported by the existing dual manual path;
no new acceptance attestation was fabricated. The standalone MPC acceptance record
is unchanged. All 23 dual-live/hierarchical/native regression tests and 12 system-Python
camera tests passed after the update. Calibration captures, checkpoint files and
other generated artifacts remain local runtime dependencies outside Git.

## Changed files and remaining limitations

- `config/experiments/dual_grasp_pipeline.yaml` and `pipeline/grasp_models.py`: model references and early CLI compatibility.
- `experiments/native_multimode_grasp.py`: isolated real model inference and raw diagnostics.
- `pipeline/dual_inference.py`: one-process native orchestration and existing canonical mirror conversion.
- `pipeline/dual_selection.py`: source-mode grouping, scored priorities and compatible export envelope.
- `pipeline/dual_runner.py`: CLI, reporting and lightweight timing.
- `pipeline/dual_artifacts.py`: dispatch by required arms, retaining the existing single-side exporter.
- `pipeline/dual_viewer.py`: model, score and execution-mode text in the existing viewer.
- `tests/test_native_multimode_pipeline.py`: focused non-hardware regression coverage.

Paths above are relative to `src/ada_grasp_ctrl/`, except the test file. The paired
implementation and physical-validation limitations are in the linked runbook.
Mirrored sampling is eager in `auto` (omitted in `both_only`), and availability
is not a calibrated physical success probability. The trained input source is a
complete object cloud; live single-view partial-cloud grasp performance remains for
the operator to validate. No checkpoint/tensor-contract mismatch was found.
