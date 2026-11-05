"""Frozen-input identity, matched pregrasp and regression accounting contracts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.shadow_regression import (
    _timed_eval_one,
    baseline_delta,
    load_suite,
    matched_pregrasp,
    run_suite,
    select_samples,
)


class ShadowRegressionTest(unittest.TestCase):
    def test_worker_wrapper_preserves_full_suite_index_and_native_result(self):
        params = ("sample.npy", object(), None, 731)
        native_result = object()
        with patch(
            "ada_grasp_ctrl.experiments.shadow_regression.safe_eval_one", return_value=native_result
        ) as evaluate:
            result, elapsed = _timed_eval_one(params)
        evaluate.assert_called_once_with(params)
        self.assertIs(result, native_result)
        self.assertGreaterEqual(elapsed, 0)

    def test_invalid_worker_count_fails_before_reading_or_creating_outputs(self):
        with patch("ada_grasp_ctrl.experiments.shadow_regression.load_suite") as load:
            for workers in (0, -1, True, 1.5):
                with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "positive integer"):
                    run_suite(Path("missing.json"), Path("unused"), workers=workers)
            load.assert_not_called()

    def test_pregrasp_extrapolates_world_translation_not_euler_and_preserves_squeeze(self):
        names = [f"finger_{i}" for i in range(22)]
        quat = Rotation.from_euler("xyz", [0.4, 0.2, -0.8]).as_quat()[[3, 0, 1, 2]]
        raw = {
            "pregrasp_qpos": np.r_[[0.2, 0.1, 0.3], quat, np.arange(22) * 0.01, np.zeros(29)],
            "grasp_qpos": np.r_[[0.1, 0.2, 0.1], quat, np.arange(22) * 0.02, np.zeros(29)],
        }
        prepared = {
            "pregrasp_qpos": np.arange(28, dtype=float),
            "grasp_qpos": np.ones(28),
            "squeeze_qpos": np.full(28, 2.0),
            "joint_names": [f"arm_{i}" for i in range(6)] + names[::-1],
        }
        before = copy.deepcopy(prepared)
        result = matched_pregrasp(prepared, raw, {"joint_names": names})
        palm = result["pregrasp_qpos"][:3] + Rotation.from_quat(quat[[1, 2, 3, 0]]).apply([0, 0, 0.034])
        np.testing.assert_allclose(palm, [0.3, 0.0, 0.5])
        np.testing.assert_allclose(result["pregrasp_qpos"][6:], 0.0)
        np.testing.assert_array_equal(result["pregrasp_qpos"][3:6], before["pregrasp_qpos"][3:6])
        for field in ("grasp_qpos", "squeeze_qpos"):
            np.testing.assert_array_equal(result[field], before[field])
        np.testing.assert_array_equal(prepared["pregrasp_qpos"], before["pregrasp_qpos"])

    def test_smoke_preserves_global_indices(self):
        suite = {
            "samples": [
                {"sample_id": str(i), "sample_index": i, "scale": s}
                for i, s in enumerate([0.06, 0.06, 0.08, 0.08, 0.1, 0.1, 0.12])
            ]
        }
        chosen = select_samples(suite, True)
        self.assertEqual([e["sample_index"] for e in chosen["samples"]], [0, 2, 4, 6])
        self.assertEqual(len(suite["samples"]), 7)

    def test_baseline_comparison_detects_lost_success_even_when_totals_match(self):
        baseline = {
            "suite_sha256": "same",
            "protocol": {"id": "p"},
            "sample_ids": ["a", "b"],
            "comparison": {
                "pairs": [
                    {"sample_id": "a", "phase1": "success", "phase2": "success"},
                    {"sample_id": "b", "phase1": "failure", "phase2": "solver_degraded"},
                ]
            },
        }
        current = copy.deepcopy(baseline)
        current["comparison"]["pairs"][0]["phase2"] = "solver_degraded"
        current["comparison"]["pairs"][1]["phase2"] = "success"
        delta = baseline_delta(current, baseline)
        self.assertEqual([r["sample_id"] for r in delta["lost_successes"]], ["a"])
        self.assertEqual([r["sample_id"] for r in delta["new_solver_degradation"]], ["a"])
        self.assertEqual(baseline_delta(baseline, baseline)["changes"], [])
        current["suite_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "suite_sha256"):
            baseline_delta(current, baseline)

    def test_baseline_rejects_different_subset(self):
        a = {"suite_sha256": "same", "protocol": {}, "sample_ids": ["a"]}
        b = {"suite_sha256": "same", "protocol": {}, "sample_ids": ["b"]}
        with self.assertRaisesRegex(ValueError, "sample_ids"):
            baseline_delta(a, b)

    def test_changed_manifest_is_not_silently_regenerated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "suite_manifest.json"
            path.write_text(json.dumps({"samples": []}))
            path.with_suffix(".sha256").write_text("wrong-digest\n")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_suite(path)


if __name__ == "__main__":
    unittest.main()
