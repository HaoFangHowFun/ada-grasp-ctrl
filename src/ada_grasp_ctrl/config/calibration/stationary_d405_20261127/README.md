# Stationary D405 calibration sources — 2026-11-27

Curated small source artifacts from the operator's 20-pair static-camera session.
Both cameras were fixed on the table; the board was mounted at `right_ee_mount`.
The official ROS-Industrial solver converged for both sides. Independent physical
validation is **not complete**. These are not benchmark or golden trajectory assets.

`left_result.yaml` and `right_result.yaml` contain the original solver output.
The app configs, measured board dimensions, capture/session metadata, serial mapping,
and original app manifest preserve the source record. `previous_calibration.json`
is the byte-preserved master before this additive installation. All runtime source
references in the active master resolve through Ada's data root to this directory.

Historical absolute paths inside the copied session/app manifest identify archived
raw capture locations; they are diagnostic records, not runtime dependencies.
Images, point clouds, per-pose raw captures, test logs, and generated app sessions
remain under ignored `output/` and are not part of the commit.

The active master and `multi_camera.yaml` select the exact pending entries by hash.
Changing a measurement or its provenance invalidates that selection. The D415,
base transforms, hand mounts, and existing independent-validation thresholds are
preserved. A separate original-pipeline three-camera live preview is still needed.
