"""Regression coverage for the display-only CRX-5iA and Sharpa asset."""

import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from script import preview_crx_sharpa_asset


class StaticAssetPreviewTest(unittest.TestCase):
    def test_fanuc_uses_original_visual_frames_and_separate_collision_meshes(self) -> None:
        robot = ET.parse(preview_crx_sharpa_asset.MODEL_PATH).getroot()
        visual_counts = {"base": 5, "j1": 2, "j2": 1, "j3": 2, "j4": 3, "j5": 2, "j6": 2}
        for part in ("base", "j1", "j2", "j3", "j4", "j5", "j6"):
            link_name = "left_base_link" if part == "base" else f"left_{part.upper()}_link"
            link = robot.find(f"link[@name='{link_name}']")
            self.assertIsNotNone(link)
            visuals = link.findall("visual")
            collision = link.find("collision")
            self.assertEqual(len(visuals), visual_counts[part])
            for index, visual in enumerate(visuals):
                self.assertEqual(
                    visual.find("geometry/mesh").get("filename"),
                    f"meshes/fanuc/crx5ia/visual_materials/{part}_{index:02d}.stl",
                )
                self.assertIsNone(visual.find("geometry/mesh").get("scale"))
                self.assertEqual(visual.find("origin").get("rpy"), "0 0 0")
                self.assertIsNotNone(visual.find("material/color").get("rgba"))
            self.assertEqual(
                collision.find("geometry/mesh").get("filename"), f"meshes/fanuc/crx5ia/collision/{part}.stl"
            )
        self.assertEqual(
            robot.find("link[@name='left_J6_link']/visual/origin").get("xyz"), "-0.43 0.13 -0.595"
        )
        j4_colors = [
            visual.find("material/color").get("rgba")
            for visual in robot.findall("link[@name='left_J4_link']/visual")
        ]
        self.assertIn("1.000000 0.000000 0.000000 1.000000", j4_colors)

    def test_viewer_keeps_the_initial_pose(self) -> None:
        with (
            patch.object(preview_crx_sharpa_asset.mujoco.viewer, "launch_passive") as launch,
            patch.object(preview_crx_sharpa_asset.mujoco, "mj_step", side_effect=AssertionError("physics advanced")),
            patch.object(preview_crx_sharpa_asset.time, "sleep"),
        ):
            viewer = launch.return_value.__enter__.return_value
            viewer.opt.geomgroup = preview_crx_sharpa_asset.mujoco.MjvOption().geomgroup
            viewer.is_running.side_effect = [True, False]

            preview_crx_sharpa_asset.main()

            model, data = launch.call_args.args
            self.assertEqual(data.qpos.tolist(), model.qpos0.tolist())
            self.assertEqual(viewer.opt.geomgroup.tolist()[:2], [0, 1])
            viewer.sync.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
