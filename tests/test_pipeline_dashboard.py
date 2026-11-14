"""Deterministic monitoring tests: actual feedback, no control interfaces or hardware I/O."""

from dataclasses import asdict, replace
import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from ada_grasp_ctrl.experiments.right_ros_contract import ARM_NAMES, HAND_NAMES
from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.pipeline.dashboard import Dashboard, PhaseObserver, RobotScene, serve_dashboard
from ada_grasp_ctrl.pipeline.telemetry import RobotReader, full_feedback, measured_positions
from ada_grasp_ctrl.pipeline.runner import Processes
from ada_grasp_ctrl.tactile_preview.adapter import TactileAdapter
from ada_grasp_ctrl.tactile_preview.data import FINGERS, normalize_frame
from test_tactile_adapter import KnownHand
from test_tactile_preview import message, synthetic_geometry


def fixture(now=10.0):
    def joints(names):
        return dict(
            names=list(names),
            positions=[0.0] * len(names),
            received=now,
            receipt_ros=100.0,
            subscriber_ros=100.0,
            error="",
        )

    frames = {}
    for finger in FINGERS:
        frame = normalize_frame(message(finger), finger, now, 100.0)
        frames[finger] = asdict(replace(frame, raw=(0.0, 0.0, -4.0, 0.0, 0.0, 0.0)))
    return dict(
        arm=joints(ARM_NAMES),
        joints=joints(HAND_NAMES),
        frames=frames,
        errors={},
        model_valid=True,
        model_sha256="model",
        world_T_right_base=np.eye(4).tolist(),
        joint_sources=dict(arm=1, hand=1, description=1),
        worker_monotonic=now,
        endpoints={f: [dict(type="dual_sharpa_wave/msg/TactileFrame")] for f in FINGERS},
    )


def server():
    def handle(*args, **kw):
        return NS(content=args[0] if args else "", value=kw.get("initial_value"), **kw)

    result = Mock()
    result.gui.add_html.side_effect = handle
    result.gui.add_markdown.side_effect = handle
    result.gui.add_slider.side_effect = handle
    for name in ("add_icosphere", "add_arrows", "add_point_cloud", "add_mesh_simple"):
        getattr(result.scene, name).side_effect = handle
    return result


class FeedbackTests(unittest.TestCase):
    def test_named_actual_joints_reorder_and_validity(self):
        packet = fixture()
        packet["arm"]["names"].reverse()
        packet["arm"]["positions"] = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        self.assertEqual(full_feedback(packet, 10.0)[:6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        for mutation in (
            lambda p: p.update(model_valid=False),
            lambda p: p["joint_sources"].update(arm=2),
            lambda p: p["joints"]["names"].__setitem__(0, p["joints"]["names"][1]),
            lambda p: p["arm"]["positions"].__setitem__(0, float("nan")),
            lambda p: p["arm"].update(received=9.0),
            lambda p: p["arm"].update(received=10.1),
            lambda p: p["joints"].update(receipt_ros=98.0),
            lambda p: p["joints"].update(received=9.8),
            lambda p: p.update(joints={}),
        ):
            bad = fixture()
            mutation(bad)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                full_feedback(bad, 10.0)
        with self.assertRaises(ValueError):
            measured_positions(fixture()["arm"], ARM_NAMES, 10.6)

    def test_monitor_exit_does_not_abort_control_service_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            owned = Processes(dict(ros_domain_id=88), Path(tmp))
            viewer, receiver = Mock(), Mock()
            viewer.poll.return_value = 1
            receiver.poll.return_value = None
            owned.children = [viewer, receiver]
            owned.monitors = [viewer]
            owned.services = [receiver]
            owned.check_services()
            receiver.poll.return_value = 1
            with self.assertRaises(RuntimeError):
                owned.check_services()


class ReaderLifecycleTests(unittest.TestCase):
    def test_subscriptions_only_actual_callbacks_and_latched_model_change(self):
        node = Mock(spec=["create_subscription", "get_clock"])
        node.get_clock.return_value.now.return_value.nanoseconds = 100_000_000_000
        ros = NS(create_node=Mock(return_value=node))
        qos = NS(
            QoSProfile=Mock(),
            ReliabilityPolicy=NS(BEST_EFFORT=1),
            DurabilityPolicy=NS(VOLATILE=0, TRANSIENT_LOCAL=1),
            qos_profile_sensor_data=object(),
        )
        modules = {
            "rclpy": ros,
            "rclpy.qos": qos,
            "sensor_msgs.msg": NS(JointState=object),
            "std_msgs.msg": NS(String=object),
            "dual_sharpa_wave.msg": NS(TactileFrame=object),
        }
        joints = ['<joint name="mount" type="fixed"><parent link="world"/><child link="right_base_link"/></joint>']
        for name in ARM_NAMES + HAND_NAMES:
            joints.append(
                f'<joint name="{name}" type="revolute"><parent link="right_base_link"/>'
                f'<child link="{name}_link"/><axis xyz="1 0 0"/>'
                '<limit lower="-3" upper="3" velocity="1"/></joint>'
            )
        for link in ("right_hand_flange", "right_hand_C_MC"):
            joints.append(
                f'<joint name="{link}_mount" type="fixed"><parent link="right_base_link"/>'
                f'<child link="{link}"/></joint>'
            )
        description = '<robot name="fixture">' + "".join(joints) + "</robot>"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            urdf = Path(tmp) / "model.urdf"
            urdf.write_text(description)
            reader = RobotReader(urdf)
            self.assertEqual(node.create_subscription.call_count, 8)
            ros.create_node.assert_called_once_with(
                "ada_tactile_preview", enable_rosout=False, start_parameter_services=False
            )
            reader.receive_model(NS(data=description))
            header = NS(stamp=NS(sec=100, nanosec=0), frame_id="base_link")
            with patch("time.monotonic", return_value=10.0):
                reader.receive_arm(NS(name=ARM_NAMES, position=[0.1] * 6, header=header))
                reader.receive_joints(NS(name=HAND_NAMES, position=[0.2] * 22, header=header))
            packet = reader.snapshot()
            packet["joint_sources"] = dict(arm=1, hand=1, description=1)
            self.assertEqual(full_feedback(packet, 10.0), [0.1] * 6 + [0.2] * 22)
            reader.receive_model(NS(data=description + " "))
            reader.receive_model(NS(data=description))
            self.assertFalse(reader.snapshot()["model_valid"])

    def test_occupied_port_never_starts_server_or_reader(self):
        with patch("ada_grasp_ctrl.pipeline.dashboard.socket.socket") as sock, patch("viser.ViserServer") as server:
            sock.return_value.__enter__.return_value.bind.side_effect = OSError("occupied")
            with self.assertRaisesRegex(RuntimeError, "8089 is occupied"):
                serve_dashboard(Path("status"), Path("ready"), Path("config"))
            server.assert_not_called()

    def test_stopped_worker_reports_disconnect_and_closes_owned_pipe(self):
        from ada_grasp_ctrl.tactile_preview.__main__ import RosProcess

        reader = RosProcess(
            sys.executable,
            "unused",
            command=[sys.executable, "-c", 'print(\'{"kind":"snapshot","worker_monotonic":0}\')'],
        )
        try:
            reader.process.wait(timeout=5)
            reader.thread.join(timeout=2)
            packet, connected, error = reader.snapshot()
            self.assertEqual(packet["kind"], "snapshot")
            self.assertFalse(connected)
            self.assertIn("exited", error)
        finally:
            reader.close()
        self.assertTrue(reader.process.stdout.closed)


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = 10.0
        self.adapter = TactileAdapter(
            KnownHand(), synthetic_geometry(), clock=lambda: self.now, stale_after=0.2, receipt_clock_synchronized=True
        )
        self.server = server()
        self.ui = Dashboard(self.server, {}, self.root, adapter=self.adapter, clock=lambda: self.now)
        # Deliberately nonidentity flange/world transform catches translation of directions and sign bugs.
        self.world = np.array([[0.0, -1, 0, 0.4], [1, 0, 0, -0.2], [0, 0, 1, 0.7], [0, 0, 0, 1]])
        self.ui.robot = NS(
            model_sha256="model",
            world_base=np.eye(4),
            table_hash="table",
            update=Mock(return_value=self.world),
            last_q=None,
        )
        self.phase = dict(phase="MPC grasp", mpc_active=True, native_force_n=16.0, native_force_live=True)

    def test_world_points_directions_sign_scale_and_over_threshold(self):
        state = self.ui.update(fixture(), True, phase=self.phase)
        snapshot = self.adapter.get_snapshot()
        self.assertEqual(state["total_n"], 20.0)
        self.assertIn("20.00", self.ui.forces.content)
        self.assertIn("20.00 N", self.ui.force_hud.content)
        self.assertIn("not a force limit", self.ui.native.content)
        for finger, result in snapshot.fingers:
            row = state["fingers"][finger]
            obs = result.observation
            np.testing.assert_allclose(row["position"], self.world[:3, :3] @ obs.position + self.world[:3, 3])
            np.testing.assert_allclose(row["normal"], self.world[:3, :3] @ obs.normal)
            np.testing.assert_allclose(row["f_object"], -self.world[:3, :3] @ obs.force)
            arrow = self.ui.arrows[finger]
            np.testing.assert_allclose(arrow.points[0, 1] - arrow.points[0, 0], 0.01 * np.array(row["f_object"]))
            self.assertEqual(arrow.colors, (255, 140, 20))
            self.assertEqual(self.ui.normals[finger].colors, (30, 210, 60))
        self.ui.scale.value = 0.02
        self.ui.update(fixture(), True, phase=self.phase)
        np.testing.assert_allclose(
            self.ui.arrows["index"].points[0, 1] - self.ui.arrows["index"].points[0, 0],
            0.02 * np.array(state["fingers"]["index"]["f_object"]),
        )

    def test_missing_invalid_multi_contact_stale_disconnect_clear_arrows(self):
        for fault in ("missing", "ambiguous", "invalid", "stale", "disconnected", "pose", "model"):
            self.ui.update(fixture(), True, phase=self.phase)
            p = fixture()
            connected = True
            if fault == "missing":
                del p["frames"]["index"]
            if fault == "ambiguous":
                p["frames"]["index"].update(contact_count=2, contact_shape=[2, 3])
            if fault == "invalid":
                p["frames"]["index"]["contact_row"] = [-100.0, 0.0, 0.0]
            if fault == "stale":
                p["frames"]["index"]["received"] = 9.0
            if fault == "disconnected":
                connected = False
            if fault == "pose":
                p["joints"] = {}
            if fault == "model":
                p["model_sha256"] = "changed"
            state = self.ui.update(p, connected, phase=self.phase)
            self.assertFalse(self.ui.arrows["index"].visible, fault)
            self.assertIsNone(state["total_n"], fault)
            self.assertNotEqual(state["fingers"]["index"].get("fn_n"), 0.0, fault)
        p = fixture()
        p["frames"]["index"].update(raw=[0, 0, 4, 0, 0, 0])
        self.assertEqual(self.ui.update(p, True, phase=self.phase)["fingers"]["index"]["status"], "INVALID")

    def test_static_cloud_loads_once_and_stays_fixed_during_all_phases(self):
        p = self.root / "grouplive/object"
        p.mkdir(parents=True)
        (p / "object.json").write_text("{}")
        (p / "run_report.json").write_text('{"status":"completed"}')
        points = np.array([[0.4, -0.3, 0.1], [0.41, -0.3, 0.1], [0.4, -0.29, 0.1]])
        with (
            patch("ada_grasp_ctrl.pipeline.dashboard.object_cloud", return_value=(points, dict(cluster_id=0))) as load,
            patch(
                "ada_grasp_ctrl.pipeline.dashboard.object_surface",
                return_value=(points, np.array([[0, 1, 2]]), np.full((3, 3), 128)),
            ) as surface,
        ):
            for i, phase in enumerate(
                ("pregrasp", "MPC grasp", "MPC squeeze", "dynamic lift planning", "lift", "hold", "release", "return")
            ):
                if i == 1:
                    asset = self.root / "grouplive/asset"
                    asset.mkdir()
                    (asset / "asset.json").write_text("{}")
                    (asset / "run_report.json").write_text("{}")
                self.now = 10.0 + i * 0.1
                packet = fixture(self.now)
                packet["arm"]["positions"][0] = i * 0.01
                result = self.ui.update(packet, True, phase=dict(phase=phase, mpc_active=phase.startswith("MPC")))
                self.assertEqual(result["phase"]["phase"], phase)
                self.assertEqual(result["total_n"], 20.0)
                self.assertEqual(result["static_surface_loaded"], i >= 1)
                np.testing.assert_array_equal(self.ui.cloud.points, points)
            load.assert_called_once()
            surface.assert_called_once()
            self.server.scene.add_mesh_trimesh.assert_called_once()
        self.assertEqual(self.ui.robot.update.call_args.args[0][0], 0.07)
        self.assertIn("MPC inactive", self.ui.banner.content)
        self.assertIn("Not tracking", self.ui.static.content)
        # No browser connection check participates in update, and no simulated commands exist here.
        self.server.get_clients.assert_not_called()

    def test_healthy_no_contact_is_zero_but_unknown_finger_is_partial(self):
        packet = fixture()
        packet["frames"]["index"].update(contact_status="EMPTY", contact_shape=[], contact_row=[], contact_count=0)
        result = self.ui.update(packet, True, phase=self.phase)
        self.assertEqual(result["total_n"], 16.0)
        self.assertEqual(result["fingers"]["index"]["fn_n"], 0.0)
        packet["frames"].pop("thumb")
        result = self.ui.update(packet, True, phase=self.phase)
        self.assertEqual(result["total_status"], "PARTIAL")
        self.assertEqual(result["partial_n"], 12.0)


class PhaseTests(unittest.TestCase):
    def test_native_force_phase_and_terminal_status_without_control_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sender").mkdir()
            (root / "receiver/action").mkdir(parents=True)
            status = root / "pipeline_status.json"
            progress = root / "sender/progress.json"
            status.write_text(json.dumps(dict(stage="sender", state="running")))
            progress.write_text(json.dumps(dict(kind="goal_sent", phase="grasp")))
            log = root / "receiver/action/events.jsonl"
            log.write_text(
                json.dumps(
                    dict(time=10.0, kind="mpc_proposal", diagnostics=dict(native_stage=2, current_sum_force=16.0))
                )
                + "\n"
            )
            observer = PhaseObserver(root)
            state = observer.update(10.1)
            self.assertEqual(state["phase"], "MPC squeeze")
            self.assertTrue(state["native_force_live"])
            self.assertEqual(state["native_force_n"], 16.0)
            self.assertIsNone(state["completed_force_n"])
            with log.open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            time=10.15,
                            kind="native_complete",
                            diagnostics=dict(current_sum_force=16.4, final_sum_force=15.0),
                        )
                    )
                    + "\n"
                )
                stream.write(
                    json.dumps(dict(time=10.16, kind="mpc_proposal", diagnostics=dict(current_sum_force=2.0))) + "\n"
                )
            for phase in ("lift", "hold", "release", "return"):
                progress.write_text(json.dumps(dict(kind="goal_sent", phase=phase)))
                state = observer.update(10.2)
                self.assertEqual(state["phase"], phase)
                self.assertEqual(state["completed_force_n"], 16.4)
                self.assertFalse(state["native_force_live"])
            status.write_text(json.dumps(dict(stage="pipeline", state="failed", reason="fixture error")))
            self.assertEqual(observer.update(11.0)["phase"], "failed")
            self.assertEqual(observer.update(11.0)["reason"], "fixture error")


class RecordedModelTests(unittest.TestCase):
    @unittest.skipUnless(
        (
            project_root() / "output/regression/crx-sharpa-right-preview-x080-y070-config-v1/preview_report.json"
        ).exists(),
        "local model artifacts required",
    )
    def test_all_actual_joints_world_transform_and_no_dynamics(self):
        import mujoco
        from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import RightKinematics

        config = yaml.safe_load(
            (project_root() / "src/ada_grasp_ctrl/config/experiments/grasp_pipeline.yaml").read_text()
        )
        config.update(config["active_hardware"])
        source = project_root() / "output/mpc_p3/active_20261113_132624/sender/dynamic-lift-request.json"
        q = np.array(json.loads(source.read_text())["positions"])
        report = json.loads((project_root() / config["model_report"]).read_text())
        # Deliberately different from model world to exercise root-body composition.
        base = np.array(report["model"]["world_T_base"])
        base[:3, 3] += [0.1, -0.2, 0.3]
        with (
            patch("mjviser.ViserMujocoScene"),
            patch.object(mujoco, "mj_step", side_effect=AssertionError("physics step")),
        ):
            scene = RobotScene(server(), config, base.tolist())
            self.assertTrue(np.all(scene.model.geom_rgba[scene.model.geom_group != 1, 3] == 0))
            self.assertGreater(np.count_nonzero(scene.model.geom_rgba[:, 3]), 0)
            self.assertFalse(np.any(scene.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
            first = scene.update(q)
            kin = RightKinematics(report["model"])
            expected = base @ np.linalg.inv(np.array(report["model"]["world_T_base"])) @ kin.forward(q)
            np.testing.assert_allclose(first, expected, atol=1e-6)
            changed = q + 0.001
            scene.update(changed)
            np.testing.assert_allclose(scene.data.qpos[scene.addresses], changed)
            self.assertFalse(np.array_equal(first, scene.update(changed)))
        np.testing.assert_array_equal(q, np.array(json.loads(source.read_text())["positions"]))


if __name__ == "__main__":
    unittest.main()
