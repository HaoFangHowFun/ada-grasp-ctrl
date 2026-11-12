"""Explicit threefold pregrasp extrapolation, clipping, and unchanged grasp semantics."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from ada_grasp_ctrl.experiments.pregrasp_opening import opening_multiplier, prepare_candidates
from ada_grasp_ctrl.experiments.dual_arm_grasp_pipeline import run
from ada_grasp_ctrl.grasp_selection.multi_arm import STAGES, hand_names
from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.robot_environment.model import side_slice


def candidate(side: str) -> dict:
    targets = {}
    for stage, value in zip(STAGES, (0.1, 0.3, 0.5)):
        targets[stage] = dict(
            frame_id="world",
            pose_frame=f"{side}_hand_flange",
            T_world_hand=np.eye(4).tolist(),
            joint_positions_rad=dict.fromkeys(hand_names(side), value),
        )
    return dict(
        schema_version=1,
        frame_id="world",
        unit="m",
        joint_unit="rad",
        candidate_id=side + ":test",
        required_arms=[side],
        targets={side: targets},
        provenance={},
    )


class OpeningTests(unittest.TestCase):
    def test_threefold_signed_delta_and_both_limits_on_both_sides(self) -> None:
        limits = np.tile([-1.0, 1.0], (56, 1))
        for side in ("left", "right"):
            raw = candidate(side)
            names = hand_names(side)
            raw["targets"][side]["pregrasp"]["joint_positions_rad"][names[0]] = 0.9
            limits[side_slice(side)][7] = [-0.2, 0.2]
            before = deepcopy(raw)
            result = prepare_candidates([raw], limits, 3)[0]
            applied = result["targets"][side]["pregrasp"]["joint_positions_rad"]
            self.assertEqual(applied[names[0]], 1.0)
            self.assertEqual(applied[names[1]], -0.2)
            self.assertAlmostEqual(applied[names[2]], -0.3)
            self.assertEqual(raw, before)
            for stage in ("grasp", "squeeze"):
                self.assertEqual(result["targets"][side][stage], raw["targets"][side][stage])
            self.assertEqual(
                result["targets"][side]["pregrasp"]["T_world_hand"], raw["targets"][side]["pregrasp"]["T_world_hand"]
            )
            clipped = result["provenance"]["pregrasp_opening"]["sides"][side]["clipped_joints"]
            self.assertEqual([v["name"] for v in clipped], list(names[:2]))

    def test_default_is_independent_exact_legacy_copy(self) -> None:
        raw = [candidate("right")]
        self.assertEqual(opening_multiplier({}), 1.0)
        result = prepare_candidates(raw, np.tile([-1.0, 1.0], (56, 1)), 1)
        self.assertEqual(result, raw)
        result[0]["candidate_id"] = "changed"
        self.assertEqual(raw[0]["candidate_id"], "right:test")

    def test_rejects_invalid_settings_and_double_application(self) -> None:
        for value in (True, False, None, "3", 0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                opening_multiplier(dict(candidate_preparation=dict(pregrasp_open_multiplier=value)))
        for settings in ([], None, dict(unknown=3)):
            with self.assertRaises(ValueError):
                opening_multiplier(dict(candidate_preparation=settings))
        limits = np.tile([-1.0, 1.0], (56, 1))
        result = prepare_candidates([candidate("right")], limits, 3)
        with self.assertRaisesRegex(ValueError, "twice"):
            prepare_candidates(result, limits, 3)
        with self.assertRaises(ValueError):
            prepare_candidates([candidate("right")], np.zeros((22, 2)), 3)

    @unittest.skipUnless(os.environ.get("ADA_DUAL_ARM_SNAPSHOT"), "Requires explicit real Phase 1 snapshot")
    def test_pipeline_keeps_raw_replay_and_uses_prepared_targets_for_selection_and_execution(self) -> None:
        config = yaml.safe_load((project_root() / "config/simulation_scene.yaml").read_text())
        config["scene"]["robot_snapshot"] = os.environ["ADA_DUAL_ARM_SNAPSHOT"]
        config["candidate_preparation"] = {"pregrasp_open_multiplier": 3}
        raw = candidate("right")
        captured = {}

        def select(robot: object, sim: object, rows: list, *args: object) -> dict:
            captured["candidate"] = deepcopy(rows[0])
            return dict(
                status="selected",
                selected=dict(selected_side="right", candidate_id="right:test", effort=0.0, candidate=rows[0]),
            )

        def execute(scene: object, selection: dict, *args: object) -> dict:
            self.assertEqual(selection["selected"]["candidate"], captured["candidate"])
            return dict(status="failed_grasp", execution_ready=True)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("ada_grasp_ctrl.experiments.dual_arm_grasp_pipeline.infer_sides", return_value=([raw], [])),
            patch("ada_grasp_ctrl.experiments.dual_arm_grasp_pipeline.select_candidates", side_effect=select),
            patch("ada_grasp_ctrl.experiments.dual_arm_grasp_pipeline.execute", side_effect=execute),
        ):
            out = Path(tmp)
            run(config, out)
            self.assertEqual(json.loads((out / "candidates.json").read_text())["candidates"], [raw])
            self.assertEqual(
                json.loads((out / "execution_candidates.json").read_text())["candidates"], [captured["candidate"]]
            )
            self.assertNotEqual(
                captured["candidate"]["targets"]["right"]["pregrasp"], raw["targets"]["right"]["pregrasp"]
            )
            self.assertEqual(captured["candidate"]["targets"]["right"]["grasp"], raw["targets"]["right"]["grasp"])


if __name__ == "__main__":
    unittest.main()
