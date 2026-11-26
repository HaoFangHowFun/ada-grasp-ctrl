"""Resultants, exact sensing regions, native contact signs and display lifecycle."""

from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np

from ada_grasp_ctrl.contact.force_display import (
    FINGERS,
    SIDES,
    FingerForce,
    ForceOverlay,
    ForceSnapshot,
    normal_load_totals,
)
from ada_grasp_ctrl.contact.force_hud import force_overlay
from ada_grasp_ctrl.contact.types import ContactObservation
from ada_grasp_ctrl.pipeline.fingertip_forces import FingertipForceSource, aggregate
from ada_grasp_ctrl.pipeline.simulation_ui import SimulationApp


def contact(force=(2, 1, 0), point=(0, 0, 0), frame=None):
    frame = np.eye(3) if frame is None else np.asarray(frame, dtype=float)
    return dict(
        body1_name="left_index_elastomer",
        body2_name="object",
        contact_pos=np.array(point, dtype=float),
        contact_pos_local=np.array(point, dtype=float).reshape(3, 1),
        contact_frame=frame,
        contact_frame_local=frame,
        contact_force=np.r_[force, np.zeros(3)],
        normal_force=force[0],
        timestamp=1.0,
        source="mujoco",
        hand_side="left",
        valid=True,
        geom_ids=[1, 2],
    )


def resultant(*rows):
    observation = ContactObservation(1.0, tuple(rows), "mujoco")
    return aggregate(observation, "left", {1: "index"})[1]


class ResultantTests(unittest.TestCase):
    def test_single_and_weighted_multiple_contacts(self):
        row = resultant(contact())
        self.assertEqual(row.status, "VALID")
        np.testing.assert_allclose(row.force, [2, 1, 0])
        rotated = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        row = resultant(contact(), contact((6, 2, 0), (4, 0, 0), rotated))
        np.testing.assert_allclose(row.force, [0, 7, 0])
        np.testing.assert_allclose(row.position, [3, 0, 0])
        np.testing.assert_allclose(row.normal, np.array([2, 6, 0]) / np.sqrt(40))
        self.assertEqual(row.normal_load, 8.0)
        self.assertNotEqual(row.normal_load, np.linalg.norm(row.force))

    def test_zero_load_is_not_invalid_and_tiny_positive_load_is_degenerate(self):
        for row in (resultant(), resultant(contact((0, 0, 0)))):
            self.assertEqual(row.status, "NO_CONTACT")
            self.assertIsNone(row.force)
            self.assertEqual(row.normal_load, 0.0)
        row = resultant(contact((1e-14, 0, 0)))
        self.assertEqual(row.status, "DEGENERATE")
        self.assertIsNone(row.force)

    def test_cancellation_and_normal_only_degeneracy(self):
        reverse = np.diag([-1, -1, 1])
        row = resultant(contact((2, 0, 0)), contact((2, 0, 0), frame=reverse))
        self.assertEqual(row.status, "NORMAL_DEGENERATE")
        np.testing.assert_array_equal(row.force, [0, 0, 0])
        self.assertEqual(row.normal_load, 4.0)
        self.assertIsNone(row.normal)
        row = resultant(contact((2, 1, 0)), contact((2, 0, 0), frame=reverse))
        np.testing.assert_array_equal(row.force, [0, 1, 0])
        self.assertIsNone(row.normal)

    def test_invalid_contact_never_produces_partial_sum(self):
        for field, value in (
            ("contact_force", [np.nan, 0, 0, 0, 0, 0]),
            ("valid", False),
            ("contact_pos", [0, np.inf, 0]),
            ("contact_frame", np.zeros((3, 3))),
        ):
            bad = contact()
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resultant(contact(), bad)

    def test_geom_filter_does_not_include_other_geoms_on_the_same_body(self):
        excluded = contact((100, 0, 0))
        excluded["geom_ids"] = [3, 2]
        np.testing.assert_allclose(resultant(contact(), excluded).force, [2, 1, 0])


def native_scene(object_first=False, missing_pad=False):
    """Ten pad contacts and three excluded contacts on a single free object."""
    pads, object_geoms = [], []
    for index, (side, finger) in enumerate((s, f) for s in SIDES for f in FINGERS):
        name = f"{side}_{finger}_elastomer"
        geom_name = "unknown_pad" if missing_pad and side == "left" and finger == "index" else name + "_collision_0"
        x = index * 0.3
        pads.append(f'<body name="{name}" pos="{x} 0 0"><geom name="{geom_name}" size=".05"/></body>')
        object_geoms.append(f'<geom pos="{x} 0 0" size=".05"/>')
    for index, name in enumerate(("left_hand_C_MC", "right_index_DP", "simulation_table"), 10):
        x = index * 0.3
        pads.append(f'<body name="{name}" pos="{x} 0 0"><geom size=".05"/></body>')
        object_geoms.append(f'<geom pos="{x} 0 0" size=".05"/>')
    obj = '<body name="object" pos="0 0 .09"><freejoint/>' + "".join(object_geoms) + "</body>"
    bodies = obj + "".join(pads) if object_first else "".join(pads) + obj
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody>" + bodies + "</worldbody></mujoco>")
    data = mujoco.MjData(model)
    data.qvel[:3] = [0.02, 0.01, 0]
    mujoco.mj_forward(model, data)
    return model, data


class SourceTests(unittest.TestCase):
    def test_native_sign_both_orders_all_fingers_and_excluded_contacts(self):
        orders = set()
        for object_first in (False, True):
            model, data = native_scene(object_first)
            source = FingertipForceSource(model, data)
            before = data.qpos.copy(), data.qvel.copy(), data.time
            snapshot = source.sample()
            self.assertFalse(snapshot.errors)
            self.assertEqual(len(snapshot.fingers), 10)
            self.assertEqual(data.ncon, 13)
            for row in snapshot.fingers:
                self.assertEqual(row.status, "VALID")
                geom = model.geom(f"{row.side}_{row.finger}_elastomer_collision_0").id
                index, con = next((i, c) for i, c in enumerate(data.contact) if geom in c.geom)
                raw = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, raw)
                direct = int(con.geom[0]) == geom
                orders.add(direct)
                expected = con.frame.reshape(3, 3).T @ raw[:3] * (1 if direct else -1)
                np.testing.assert_allclose(row.force, expected)
                self.assertAlmostEqual(row.normal_load, raw[0])
                self.assertGreater(row.force[2], 0)
                np.testing.assert_allclose(row.normal, [0, 0, 1])
            np.testing.assert_array_equal(data.qpos, before[0])
            np.testing.assert_array_equal(data.qvel, before[1])
            self.assertEqual(data.time, before[2])
        self.assertEqual(orders, {False, True})

    def test_one_side_failure_clears_all_five_other_side_survives(self):
        model, data = native_scene()
        source = FingertipForceSource(model, data)
        with patch.object(source.adapters["left"], "observe", side_effect=ValueError("broken contact")):
            snapshot = source.sample()
        self.assertEqual([r.status for r in snapshot.fingers], ["UNAVAILABLE"] * 5 + ["VALID"] * 5)
        self.assertIn("broken contact", snapshot.errors[0])
        self.assertTrue(all(r.force is None for r in snapshot.fingers[:5]))
        totals = normal_load_totals(snapshot)
        self.assertIsNone(totals[0])
        self.assertGreater(totals[1], 0)
        self.assertFalse(source.sample().errors)

    def test_missing_pad_mapping_is_explicit_side_failure(self):
        model, data = native_scene(missing_pad=True)
        source = FingertipForceSource(model, data)
        snapshot = source.sample()
        self.assertEqual([r.status for r in snapshot.fingers[:5]], ["UNAVAILABLE"] * 5)
        self.assertTrue(snapshot.errors)
        self.assertEqual([r.status for r in snapshot.fingers[5:]], ["VALID"] * 5)

    def test_compiled_sharpa_snapshot_has_distinct_pad_and_shell(self):
        path = Path("output/dual_arm/phase1_20261111/commit_snapshot/model.urdf")
        if not path.exists():
            self.skipTest("Local canonical snapshot unavailable")
        spec = mujoco.MjSpec.from_file(str(path))
        obj = spec.worldbody.add_body(name="object", pos=[10, 0, 0])
        obj.add_freejoint()
        obj.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=0.1)
        model = spec.compile()
        source = FingertipForceSource(model, mujoco.MjData(model))
        self.assertFalse(source.errors)
        for side in SIDES:
            self.assertEqual(len(source.geom_fingers[side]), 5)
            for finger in FINGERS:
                name = f"{side}_{finger}"
                pad = model.geom(name + "_elastomer_collision_0").id
                self.assertEqual(source.geom_fingers[side][pad], finger)
                for suffix in ("_DP_collision_0", "_elastomer_visual_0"):
                    self.assertNotIn(model.geom(name + suffix).id, source.geom_fingers[side])


def fake_server():
    server = MagicMock()
    server.gui.add_checkbox.side_effect = lambda label, initial_value: NS(value=initial_value)
    server.gui.add_slider.side_effect = lambda label, **kwargs: NS(value=kwargs["initial_value"])
    server.gui.add_markdown.side_effect = lambda content: NS(content=content)
    server.gui.add_html.side_effect = lambda content, **kwargs: NS(content=content, **kwargs)
    for name in ("add_frame", "add_icosphere", "add_arrows"):
        getattr(server.scene, name).side_effect = lambda *args, **kwargs: NS(**kwargs)
    return server


class DisplayTests(unittest.TestCase):
    def setUp(self):
        self.server = fake_server()
        self.overlay = ForceOverlay(self.server)
        self.row = FingerForce("left", "index", "VALID", (1, 2, 3), (2, 0, 0), (1, 0, 0))
        self.snapshot = ForceSnapshot(1.0, (self.row,))

    def test_ten_groups_sign_scale_toggles_disappearance_and_zero(self):
        self.assertEqual(len(self.overlay.groups), 10)
        self.assertEqual(self.server.scene.add_arrows.call_count, 30)
        self.overlay.update(self.snapshot)
        marker, reaction, obj, normal = self.overlay.groups["left", "index"]
        np.testing.assert_allclose(reaction.points, [[[1, 2, 3], [0.98, 2, 3]]])
        np.testing.assert_allclose(obj.points, [[[1, 2, 3], [1.02, 2, 3]]])
        np.testing.assert_allclose(normal.points, [[[1, 2, 3], [1.025, 2, 3]]])
        self.overlay.object_force.value = False
        self.overlay.update(self.snapshot)
        self.assertFalse(obj.visible)
        self.assertTrue(reaction.visible)
        self.overlay.update(
            ForceSnapshot(2.0, (replace(self.row, force=(0, 0, 0), normal=None, status="NORMAL_DEGENERATE"),))
        )
        self.assertTrue(marker.visible)
        self.assertFalse(any(h.visible for h in (reaction, obj, normal)))
        self.overlay.update(ForceSnapshot(3.0, ()))
        self.assertFalse(marker.visible)
        self.overlay.sides["left"].value = False
        self.overlay.update(self.snapshot)
        self.assertFalse(marker.visible)
        self.overlay.enabled.value = False
        self.overlay.update(self.snapshot)
        self.assertFalse(self.overlay.root.visible)
        self.assertFalse(self.overlay.hud.visible)
        self.assertEqual(self.server.scene.add_arrows.call_count, 30)

    def app(self):
        app = SimulationApp.__new__(SimulationApp)
        app.state, app.generation, app.force_generation = "EXECUTING", 1, 1
        app.scene = object()
        app.executor = NS(
            scene=app.scene, cancel=threading.Event(), result={"status": "executing"}, force_snapshot=self.snapshot
        )
        app.force_overlay, app.force_diagnostic = self.overlay, NS(content="")
        return app

    def test_preview_error_cancel_generation_and_persistent_placement(self):
        app = self.app()
        for state, status in (("EXECUTING", "executing"), ("HOLDING", "holding"), ("COMPLETED", "completed")):
            app.state, app.executor.result["status"] = state, status
            app.render_forces()
            self.assertTrue(self.overlay.root.visible)
            self.assertTrue(self.overlay.hud.visible)
        for state in ("EDITING", "PREVIEW_READY", "RESETTING", "ERROR"):
            app.state = state
            app.render_forces()
            self.assertFalse(self.overlay.root.visible)
            self.assertFalse(self.overlay.hud.visible)
            self.assertEqual(self.overlay.hud.content, "")
        app.state = "EXECUTING"
        app.executor.cancel.set()
        app.render_forces()
        self.assertFalse(self.overlay.root.visible)
        app.executor.cancel.clear()
        app.generation += 1
        app.render_forces()
        self.assertFalse(self.overlay.root.visible)
        app.force_generation = app.generation
        app.executor.result["status"] = "motion_failed"
        app.render_forces()
        self.assertFalse(self.overlay.root.visible)

    def test_render_failure_hides_existing_arrows_and_exposes_diagnostic(self):
        app = self.app()
        app.render_forces()
        with patch.object(self.overlay, "update", side_effect=RuntimeError("broken renderer")):
            app.render_forces()
        self.assertFalse(self.overlay.root.visible)
        self.assertIn("broken renderer", app.force_diagnostic.content)
        self.assertFalse(self.overlay.hud.visible)
        self.assertEqual(app.executor.result["status"], "executing")

    def test_invalid_force_overflow_hides_the_entire_overlay(self):
        app = self.app()
        app.executor.force_snapshot = ForceSnapshot(1.0, (replace(self.row, force=(1e308, 0, 0)),))
        app.render_forces()
        self.assertFalse(self.overlay.root.visible)
        self.assertIn("Contact forces unavailable", app.force_diagnostic.content)

    def test_hud_sums_raw_normal_loads_and_ignores_arrow_visibility_and_scale(self):
        rows = [FingerForce(s, f, "NO_CONTACT", normal_load=0.0) for s in SIDES for f in FINGERS]
        # Two opposed contacts cancel the vector, but their compressive loads still add.
        rows[1] = resultant(contact((2, 0, 0)), contact((2, 0, 0), frame=np.diag([-1, -1, 1])))
        rows[6] = replace(self.row, side="right", normal_load=6.0)
        snapshot = ForceSnapshot(1.0, tuple(rows))
        self.overlay.update(snapshot)
        self.assertEqual(normal_load_totals(snapshot), (4.0, 6.0))
        content = self.overlay.hud.content
        for text in ("Total fingertip ΣFn", "10.00 N", "Left ΣFn: 4.00 N", "Right ΣFn: 6.00 N"):
            self.assertIn(text, content)
        self.assertNotIn("Grasp complete", content)
        self.overlay.sides["left"].value = False
        self.overlay.scale.value = 0.05
        self.overlay.update(snapshot)
        self.assertEqual(self.overlay.hud.content, content)
        self.overlay.update(
            ForceSnapshot(2.0, tuple(FingerForce(s, f, "NO_CONTACT", normal_load=0.0) for s in SIDES for f in FINGERS))
        )
        self.assertIn("Left ΣFn: 0.00 N", self.overlay.hud.content)
        self.assertIn("Right ΣFn: 0.00 N", self.overlay.hud.content)

    def test_hud_bad_or_missing_finger_withholds_side_and_combined_total(self):
        healthy = [replace(self.row, side=s, finger=f, normal_load=1.0) for s in SIDES for f in FINGERS]
        for status, load in (("UNAVAILABLE", None), ("DEGENERATE", None), ("VALID", float("nan")), ("VALID", -1)):
            rows = list(healthy)
            rows[0] = replace(rows[0], status=status, normal_load=load)
            snapshot = ForceSnapshot(1.0, tuple(rows))
            self.assertEqual(normal_load_totals(snapshot), (None, 5.0))
            self.overlay.update(snapshot)
            self.assertIn("Left ΣFn: —", self.overlay.hud.content)
            self.assertIn("Right ΣFn: 5.00 N", self.overlay.hud.content)
            self.assertIn("UNAVAILABLE", self.overlay.hud.content)
            self.assertNotIn("10.00 N", self.overlay.hud.content)
        self.assertEqual(normal_load_totals(ForceSnapshot(1.0, tuple(healthy[1:]))), (None, 5.0))

    def test_shared_hud_preserves_single_arm_completion_and_escapes_status(self):
        from ada_grasp_ctrl.pipeline.dashboard import force_overlay as legacy_overlay

        self.assertIs(legacy_overlay, force_overlay)
        content = legacy_overlay(2.5, 15.25, "<fault>")
        for text in ("Live contact ΣFn", "2.50 N", "Grasp complete ΣFn", "15.25 N", "&lt;fault&gt;"):
            self.assertIn(text, content)
        self.assertIn("top:16px;left:16px", content)
        self.assertNotIn("Right ΣFn", content)
