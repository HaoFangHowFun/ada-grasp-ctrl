"""Selection-only support-face coverage, small-contact tolerance and FK reuse."""

from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.placement.face_coverage import (
    FaceCoverage,
    MIN_AREA_M2,
    MIN_SPAN_M,
    contact_patch,
    hull,
    overlap,
    plane_axes,
)
from ada_grasp_ctrl.placement.feasibility import PlacementGate
from ada_grasp_ctrl.grasp_selection.multi_arm import select_candidates
from test_phase2_pipeline import FakeRobot, FakeSim, candidate


def box(size, center=(0, 0, 0)):
    return np.asarray(list(product((-1, 1), repeat=3))) * np.asarray(size) / 2 + center


class FaceCoverageTests(unittest.TestCase):
    def test_broad_finger_rejected_but_tip_edge_and_outside_face_allowed(self):
        plane = np.array([0.0, 0, 1, 0])
        axes = plane_axes(plane[:3])
        patch = hull(np.asarray(contact_patch(box((0.04, 0.04, 0.02), (0, 0, 0.01)), plane)) @ axes)
        for size, center, blocked in (
            ((0.025, 0.01, 0.01), (0, 0, 0), True),
            ((0.005, 0.005, 0.01), (0, 0, 0), False),
            ((0.025, 0.01, 0.01), (0.0315, 0, 0), False),
            ((0.025, 0.01, 0.01), (0.05, 0, 0), False),
            ((0.025, 0.01, 0.01), (0, 0, 0.008), False),
        ):
            with self.subTest(size=size, center=center):
                area, span = overlap(box(size, center), plane, axes, patch)
                self.assertEqual(area >= MIN_AREA_M2 and span >= MIN_SPAN_M, blocked)
        self.assertEqual(contact_patch(np.array([[0.0, 0, 0], [0, 0, 0.1]]), plane), [])

    def test_rotation_translation_and_tilted_table_preserve_coverage(self):
        plane = np.array([0.0, 0, 1, 0])
        support = box((0.04, 0.04, 0.02), (0, 0, 0.01))
        finger = box((0.025, 0.01, 0.01))
        axes = plane_axes(plane[:3])
        expected = overlap(finger, plane, axes, hull(np.asarray(contact_patch(support, plane)) @ axes))
        R = Rotation.from_euler("xyz", [0.4, -0.3, 1.2]).as_matrix()
        t = np.array([0.7, -0.4, 0.23])
        n = R[:, 2]
        plane = np.r_[n, -n @ t]
        axes = plane_axes(n)
        patch = np.asarray(contact_patch(support @ R.T + t, plane))
        np.testing.assert_allclose(patch @ n + plane[3], 0, atol=1e-12)
        actual = overlap(finger @ R.T + t, plane, axes, hull(patch @ axes))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_compiled_mesh_both_sides_no_forward_step_or_live_mutation(self):
        vertices = " ".join(str(x) for x in box((0.025, 0.01, 0.01)).ravel())
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
          <mesh name="finger" vertex="{vertices}"/></asset><worldbody>
          <body name="left_hand_flange" pos="0.3 0.1 0.2"><body name="left_finger">
            <geom name="left_finger" type="mesh" mesh="finger"/></body></body>
          <body name="right_hand_flange" pos="-0.3 0.1 0.2"><body name="right_finger">
            <geom name="right_finger" type="mesh" mesh="finger"/></body></body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        labels = {model.body(side + "_finger").id: side for side in ("left", "right")}
        static = SimpleNamespace(
            model=model, static_data=data, qpos=np.array([], dtype=int), hand_body_ids=set(labels), body_labels=labels
        )
        plane = np.array([0.0, 0, 1, 0])
        target = dict(
            table_plane_world=plane, contact_patch_world_m=contact_patch(box((0.04, 0.04, 0.02), (0, 0, 0.01)), plane)
        )
        checker = FaceCoverage(static, target)
        before = data.geom_xpos.copy(), data.qpos.copy(), data.time
        with (
            patch.object(mujoco, "mj_forward", side_effect=AssertionError("extra FK")),
            patch.object(mujoco, "mj_step", side_effect=AssertionError("physics")),
        ):
            for side in ("left", "right"):
                result = checker.check(side, np.array([]), np.eye(4))
                self.assertFalse(result["accepted"])
                self.assertEqual(result["blocked_geoms"][0]["geom"], side + "_finger")
                self.assertAlmostEqual(result["blocked_geoms"][0]["overlap_area_m2"], 250e-6, places=10)
                with patch("ada_grasp_ctrl.placement.face_coverage.overlap", side_effect=AssertionError("recomputed")):
                    self.assertEqual(checker.check(side, np.array([]), np.eye(4)), result)
                raised = np.eye(4)
                raised[2, 3] = 0.1
                self.assertTrue(checker.check(side, np.array([]), raised)["accepted"])
        np.testing.assert_array_equal(data.geom_xpos, before[0])
        np.testing.assert_array_equal(data.qpos, before[1])
        self.assertEqual(data.time, before[2])

    def test_covered_candidate_rejected_before_place_ik_and_next_candidate_selected(self):
        robot, sim = FakeRobot(), FakeSim()
        gate = PlacementGate(robot, sim, dict(Delta_T=np.eye(4).tolist(), target_revision=1, target_sha256="test"))
        gate.face_coverage = Mock()
        gate.face_coverage.check.side_effect = lambda side, q, pose: dict(accepted=pose[0, 3] > 0.15)
        with patch.object(robot, "solve_arm_ik", side_effect=AssertionError("unnecessary place IK")):
            result = gate(candidate("right", 0.1), "right", np.zeros(56))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "placement_face_covered")
        selected = select_candidates(
            robot,
            sim,
            [candidate("right", 0.1, "covered"), candidate("right", 0.2, "clear")],
            ["right"],
            np.zeros(56),
            endpoint_gate=gate,
        )
        self.assertEqual(selected["selected"]["candidate_id"], "right:clear")
        self.assertTrue(all(a["reasons"] == ["placement_face_covered"] for a in selected["evaluations"][0]["attempts"]))


if __name__ == "__main__":
    unittest.main()
