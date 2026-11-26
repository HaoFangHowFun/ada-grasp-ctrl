"""Real Viser/MuJoCo lifecycle with explicitly synthetic endpoint results."""

from contextlib import contextmanager
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ada_grasp_ctrl.calibration import calibrated_snapshot
from ada_grasp_ctrl.pipeline import dual_runner
from ada_grasp_ctrl.pipeline.dual_viewer import create_server
from ada_grasp_ctrl.pipeline.simulation_scene import SceneBuilder
from ada_grasp_ctrl.pipeline.simulation_ui import SimulationApp
from ada_grasp_ctrl.pipeline.placement_preview import PlacementPreview
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.placement.target import transform_points
import test_dual_mujoco_web
from test_placement_selection import candidate

import msgspec
import numpy as np
import viser
from websockets.sync.client import connect


class PlacementAwareWebTests(unittest.TestCase):
    wait_for = test_dual_mujoco_web.SimulationWebTests.wait_for
    update = test_dual_mujoco_web.SimulationWebTests.update
    state = test_dual_mujoco_web.SimulationWebTests.state
    confirm_scene = test_dual_mujoco_web.SimulationWebTests.confirm_scene
    reset_scene = test_dual_mujoco_web.SimulationWebTests.reset_scene

    @contextmanager
    def viewer(self, root):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8089)) == 0:
                self.skipTest("User port 8089 occupied")
        from types import SimpleNamespace

        config = dual_runner.configuration("src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml")
        config["robot_snapshot"] = str(calibrated_snapshot(config["robot_snapshot"], root / "robot"))
        config["grasp_model"] = "right_full"
        builder = SceneBuilder(config)
        server = create_server()
        app = SimulationApp(
            server,
            builder,
            SimpleNamespace(
                placement_only=False,
                placement_aware=True,
                placement_execute=getattr(self, "placement_execute", False),
                preview_only=False,
                export_only=False,
                uop_root=None,
                uop_python=None,
                selector="legacy",
            ),
            root,
        )
        self.app = app
        app.transition("EDITING")
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.fail_next = False

        def prepare(engine, target, destination, canceled):
            self.calls.append(target["target_revision"])
            self.entered.set()
            self.release.wait(15)
            if canceled():
                return None
            if self.fail_next:
                write_json(destination / "placement_selection_report.json", dict(counts={"place_feasible": 0}))
                return None
            q = engine.state["full_positions_rad"]
            result = dict(
                target_revision=target["target_revision"],
                arm="right",
                mode="fixture",
                candidate_id="synthetic_ui_only",
                effort=0.0,
                grasp_q56=q,
                place_q56=q,
                placement=dict(ik={"accepted": True}),
                target=target,
            )
            write_json(destination / "placement_endpoint_preview.json", result)
            return result

        try:
            with (
                patch.object(PlacementPreview, "prepare", prepare),
                patch("ada_grasp_ctrl.pipeline.simulation_ui.run_uop", return_value=[candidate(1, [1, 0, 0])]) as uop,
            ):
                self.uop = uop
                with connect("ws://127.0.0.1:8089", subprotocols=[f"viser-v{viser.__version__}"], max_size=None) as ws:
                    self.ws, self.gui, self.nodes = ws, {}, {}
                    ws.send(
                        msgspec.msgpack.encode(
                            dict(
                                type="ViewerCameraMessage",
                                wxyz=[1, 0, 0, 0],
                                position=[2, -2, 2],
                                fov=1.0,
                                near=0.01,
                                far=100.0,
                                image_height=600,
                                image_width=800,
                                look_at=[0, 0, 0],
                                up_direction=[0, 0, 1],
                            )
                        )
                    )
                    self.wait_for(lambda: "Confirm Scene" in self.gui)
                    yield app
        finally:
            self.release.set()
            app.close()
            server.stop()

    def test_drag_confirm_edit_stale_result_reset_and_static_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.viewer(root) as app:
                attempt = self.confirm_scene(root, "PLACEMENT_EDITING")
                self.wait_for(lambda: "Confirm Placement Target" in self.gui)
                self.assertTrue(self.gui["Placement candidate"]["props"]["value"].startswith("P00"))
                self.assertFalse(self.gui["Execute Grasp"]["props"]["visible"])
                settled_qpos = app.scene.sim.data.qpos.copy()
                settled_time = app.scene.sim.data.time
                self.update("Destination X (m)", 0.42)
                self.wait_for(
                    lambda: abs(self.nodes.get("/placement_destination", {}).get("position", [0])[0] - 0.42) < 1e-6
                )
                self.ws.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="TransformControlsUpdateMessage",
                            name="/placement_destination",
                            wxyz=[1, 0, 0, 0],
                            position=[0.43, -0.4, 99],
                        )
                    )
                )
                self.wait_for(lambda: abs(self.gui["Destination X (m)"]["props"]["value"] - 0.43) < 1e-6)
                marker = np.array(app.panel.marker.position)
                self.assertAlmostEqual(marker @ app.panel.geometry.plane[:3] + app.panel.geometry.plane[3], 0)
                self.assertEqual(self.calls, [])
                self.update("Run UOP-Net once", True)
                self.wait_for(lambda: len(self.gui["Placement candidate"]["props"]["options"]) == 2)
                self.update("Confirm Placement Target", True)
                self.wait_for(lambda: self.entered.is_set())
                first = app.panel.revision
                self.update("Edit Placement Target", True)
                self.wait_for(lambda: app.panel.revision != first)
                self.release.set()
                time.sleep(0.1)
                self.assertIsNone(app.panel.endpoint)
                self.assertEqual(app.state, "PLACEMENT_EDITING")
                self.update("Confirm Placement Target", True)
                self.wait_for(lambda: app.state in ("PLACEMENT_PREVIEW_READY", "PLACEMENT_ERROR"))
                self.assertEqual(app.state, "PLACEMENT_PREVIEW_READY", app.status.content)
                target = app.panel.endpoint["target"]
                np.testing.assert_allclose(
                    app.panel.target_cloud.points, transform_points(app.panel.session.points, target), atol=1e-7
                )
                for endpoint in ("Grasp", "Place"):
                    self.update("Endpoint", endpoint)
                    self.wait_for(lambda: app.panel.endpoint_choice.value == endpoint)
                    np.testing.assert_array_equal(app.scene.sim.data.qpos, settled_qpos)
                    self.assertEqual(app.scene.sim.data.time, settled_time)
                self.assertEqual(self.uop.call_count, 1)
                app.execute()
                self.assertIsNone(app.executor)
                self.assertFalse((attempt / "simulation_plan.json").exists())
                self.assertFalse((attempt / "export").exists())
                self.fail_next = True
                self.update("Edit Placement Target", True)
                self.wait_for(lambda: app.state == "PLACEMENT_EDITING")
                self.update("Confirm Placement Target", True)
                self.wait_for(lambda: app.state == "PLACEMENT_NO_FEASIBLE_GRASP")
                self.assertIsNone(app.panel.endpoint)
                self.assertFalse(app.panel.target_cloud.visible)
                self.update("Destination X (m)", 0.44)
                self.wait_for(lambda: app.state == "PLACEMENT_EDITING")
                self.assertEqual(app.attempt, attempt)
                self.release.clear()
                self.entered.clear()
                self.update("Confirm Placement Target", True)
                self.wait_for(lambda: self.entered.is_set())
                old = app.panel
                self.reset_scene(root)
                self.release.set()
                self.assertFalse(old.active)
                self.assertEqual(json.loads((old.session.root / "target_state.json").read_text())["status"], "canceled")
                self.assertEqual(app.state, "EDITING")

    def test_execution_modifier_failure_retry_scrub_edit_execute_and_reset(self):
        from test_dual_mujoco_simulation import idle_reference
        from ada_grasp_ctrl.pipeline.placement_schedule import PHASES, build_schedule
        from ada_grasp_ctrl.pipeline.paired_artifacts import named_state

        self.placement_execute = True
        fail = True

        def preparation(engine, endpoint, destination, canceled):
            if fail:
                raise ValueError("retreat: synthetic preparation failure; selected candidate retained")
            plan = idle_reference(self.app.scene, ["right"])
            q = self.app.scene.sim.configuration()
            plan.update(
                workflow="grasp_to_place",
                execution_phases=list(PHASES),
                accepted_place_q56=q.tolist(),
                target_revision=endpoint["target_revision"],
                target=endpoint["target"],
                preparation_s=0.1,
                T_world_hand_source_held=engine.robot.fk(q, "right").tolist(),
                schedule=build_schedule(
                    plan["schedule"]["plan"],
                    named_state(q),
                    {p: {"right": named_state(q)["right"]} for p in PHASES},
                    {},
                ),
            )
            write_json(destination / "simulation_plan.json", plan)
            return plan

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("ada_grasp_ctrl.placement.target_ui.prepare_execution", side_effect=preparation),
        ):
            root = Path(directory)
            with self.viewer(root) as app:
                self.confirm_scene(root, "PLACEMENT_EDITING")
                self.wait_for(lambda: "Confirm Placement Target" in self.gui)
                self.release.set()
                self.update("Confirm Placement Target", True)
                self.wait_for(lambda: app.state == "PLACEMENT_ERROR")
                self.assertIsNone(app.plan)
                self.assertTrue(app.execute_button.disabled)
                self.assertEqual(len(self.calls), 1)
                fail = False
                self.update("Edit Placement Target", True)
                self.wait_for(lambda: app.state == "PLACEMENT_EDITING")
                self.update("Confirm Placement Target", True)
                self.wait_for(lambda: app.state == "PREVIEW_READY")
                initial_time = app.scene.sim.data.time
                initial_qpos = app.scene.sim.data.qpos.copy()
                for phase in PHASES:
                    self.update("Phase", phase)
                    self.wait_for(lambda: app.phase.value == phase)
                    np.testing.assert_array_equal(app.scene.sim.data.qpos, initial_qpos)
                    self.assertEqual(app.scene.sim.data.time, initial_time)
                reviewed = app.plan_root
                self.update("Destination X (m)", 0.43)
                self.wait_for(lambda: app.state == "PLACEMENT_EDITING")
                self.assertIsNone(app.plan)
                app.execute()
                self.assertIsNone(app.executor)
                self.update("Confirm Placement Target", True)
                self.wait_for(lambda: app.state == "PREVIEW_READY")
                self.assertNotEqual(app.plan_root, reviewed)
                self.assertTrue((reviewed / "simulation_plan.json").exists())
                self.update("Execute Grasp-to-Place", True)
                self.wait_for(lambda: app.state == "EXECUTING")
                self.wait_for(lambda: app.scene.sim.data.time > initial_time)
                revision = app.panel.revision
                app.panel.edit_target()
                self.assertEqual(app.panel.revision, revision)
                self.assertTrue(app.object_handle.visible)
                self.assertFalse(app.panel.cloud.visible)
                owner = app.executor
                self.reset_scene(root)
                self.assertTrue(owner.cancel.is_set())
                self.assertEqual(app.state, "EDITING")


if __name__ == "__main__":
    unittest.main()
