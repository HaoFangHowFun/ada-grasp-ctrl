"""Phase 2 contracts: minimum effort, mirror recovery, real mesh and visible camera depth."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import trimesh
import yaml

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.grasp_selection.multi_arm import (
    enabled_set,
    forbidden_contacts,
    from_right_record,
    hand_names,
    select_candidates,
    validate_candidate,
    STAGES,
)
from ada_grasp_ctrl.experiments.mirrored_left_grasp import infer_sides, reflect_points, reflection_plane
from ada_grasp_ctrl.experiments.simulated_object_cloud import (
    ObjectScene,
    apply_arm_bias_compensation,
    enlarge_table_footprint,
    exclude_adjacent_base_contacts,
)
from ada_grasp_ctrl.experiments.dual_arm_grasp_pipeline import run, fixture_candidates


def candidate(side: str, target: float, name: str = "c0") -> dict:
    """Build a deterministic schema fixture with an independently calculable six-joint score."""
    pose = np.eye(4)
    pose[0, 3] = target
    stages = {
        s: dict(
            frame_id="world",
            pose_frame=f"{side}_hand_flange",
            T_world_hand=pose.tolist(),
            joint_positions_rad=dict.fromkeys(hand_names(side), 0.0),
        )
        for s in STAGES
    }
    return dict(
        schema_version=1,
        frame_id="world",
        unit="m",
        joint_unit="rad",
        candidate_id=f"{side}:{name}",
        required_arms=[side],
        targets={side: stages},
        provenance=dict(source_candidate_id=name),
    )


class FakeRobot:
    """Analytical fixture IK: target X maps to six equal angles, never fingers or passive side."""

    limits = np.tile([-10.0, 10.0], (56, 1))

    def configuration(self, q: object) -> np.ndarray:
        q = np.array(q, dtype=float)
        if q.shape != (56,) or not np.isfinite(q).all():
            raise ValueError("bad robot state")
        return q

    def solve_arm_ik(self, q: np.ndarray, side: str, pose: object, **kwargs: object) -> tuple[np.ndarray, dict]:
        q = q.copy()
        q[side_slice(side, True)] = np.asarray(pose)[0, 3]
        return q, dict(accepted=abs(q[side_slice(side, True)][0]) < 5)


class FakeSim:
    def __init__(self) -> None:
        self.checked = []

    def check_static_contacts(self, q: np.ndarray) -> list[dict]:
        self.checked.append(q.copy())
        return []


class BearingContactTests(unittest.TestCase):
    def spec(self, *, missing: str | None = None, detached: str | None = None) -> mujoco.MjSpec:
        """Build a tiny tree to exercise exact pair validation without robot assets."""
        spec = mujoco.MjSpec()
        for side in ("left", "right"):
            base = spec.worldbody.add_body(name=f"{side}_base_link")
            if side != missing:
                parent = spec.worldbody if side == detached else base
                parent.add_body(name=f"{side}_J1_link")
            base.add_body(name=f"{side}_J2_link")
        return spec

    def test_only_existing_adjacent_policy_pairs_are_excluded_idempotently(self) -> None:
        spec = self.spec()
        for _ in range(2):
            exclude_adjacent_base_contacts(spec)
            self.assertEqual(
                {(e.bodyname1, e.bodyname2) for e in spec.excludes},
                {(f"{s}_base_link", f"{s}_J1_link") for s in ("left", "right")},
            )

    def test_invalid_adjacency_fails_before_mutating_either_side(self) -> None:
        for side in ("left", "right"):
            for problem in ("missing", "detached"):
                with self.subTest(side=side, problem=problem):
                    spec = self.spec(**{problem: side})
                    with self.assertRaisesRegex(ValueError, f"adjacent {side}_base_link"):
                        exclude_adjacent_base_contacts(spec)
                    self.assertEqual(list(spec.excludes), [])


class TableFootprintTests(unittest.TestCase):
    def test_enlarges_only_xy_and_keeps_collision_flags_height_and_friction(self) -> None:
        spec = mujoco.MjSpec()
        for kind in ("visual", "collision"):
            spec.worldbody.add_geom(
                name=f"calibration_table_surface_{kind}_0",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.15, 0.15, 0.02],
                pos=[0, 0, -0.02],
                contype=0 if kind == "visual" else 1,
                conaffinity=0 if kind == "visual" else 1,
                friction=[0.8, 0.02, 0.001],
            )
        for bad in ([1], [1, 1, 1], [float("nan"), 1], [0, 1], [-1, 1], [0.1, 1]):
            with self.assertRaises(ValueError):
                enlarge_table_footprint(spec, bad)
            for geom in spec.geoms:
                np.testing.assert_array_equal(geom.size, [0.15, 0.15, 0.02])
        metadata = enlarge_table_footprint(spec, [1, 1])
        self.assertEqual(metadata["half_extents_xy_m"], [1, 1])
        for geom in spec.geoms:
            np.testing.assert_array_equal(geom.size, [1, 1, 0.02])
            np.testing.assert_array_equal(geom.pos, [0, 0, -0.02])
            np.testing.assert_array_equal(geom.friction, [0.8, 0.02, 0.001])
            self.assertEqual(geom.contype, 0 if "visual" in geom.name else 1)
            self.assertEqual(geom.conaffinity, geom.contype)


class SelectionTests(unittest.TestCase):
    def test_inference_filters_enabled_side_and_preserves_virtualenv_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "venv_python"
            python.symlink_to("/usr/bin/python3")
            points = np.random.default_rng(5).normal(size=(20, 3))
            config = dict(
                selector=dict(enabled_arms=["left"]),
                point_cloud=dict(seed=3, source="complete_mesh"),
                object=dict(name="fixture"),
                inference=dict(
                    python=str(python),
                    samples_per_side=2,
                    checkpoint="checkpoint",
                    training_config="training",
                    metadata="metadata",
                    dexlearn_root="dex",
                ),
            )
            observed = []

            def fake_infer(cmd: list, **kwargs: object) -> SimpleNamespace:
                observed.append(cmd)
                self.assertEqual(cmd[0], str(python))
                destination = Path(cmd[cmd.index("--output-root") + 1])
                destination.mkdir()
                obj = Path(cmd[cmd.index("--object") + 1])
                cloud = np.load(obj / "object.npz")["points_world_m"]
                plane = reflection_plane(points, {})
                np.testing.assert_allclose(cloud, reflect_points(points, plane))
                row = dict(candidate_id="c0", stages=candidate("right", 0.1)["targets"]["right"])
                payload = dict(
                    candidates=[row],
                    checkpoint={},
                    metadata={},
                    training_config={},
                    dexlearn_commit="fixture",
                    preprocessing={},
                )
                (destination / "candidates.json").write_text(json.dumps(payload))
                return SimpleNamespace(returncode=0)

            with patch("ada_grasp_ctrl.experiments.mirrored_left_grasp.subprocess.run", side_effect=fake_infer):
                rows, rejected = infer_sides(points, config, root, "fixture")
            self.assertEqual(len(observed), 1)
            self.assertEqual(rows[0]["required_arms"], ["left"])
            self.assertEqual(rejected, [])

    def test_minimum_effort_from_current_and_global_single_winner(self) -> None:
        q = np.zeros(56)
        q[:6], q[28:34] = 1.0, 2.0
        rows = [candidate("left", 1.2), candidate("right", 2.1)]
        result = select_candidates(FakeRobot(), FakeSim(), rows, ["left", "right"], q)
        self.assertEqual(result["selected"]["selected_side"], "right")
        self.assertAlmostEqual(result["selected"]["effort"], 6 * 0.1**2)
        self.assertFalse(result["execution_ready"])

    def test_enabled_filter_and_passive_state_in_every_stage(self) -> None:
        for side in ("left", "right"):
            q = np.zeros(56)
            passive = side_slice("left" if side == "right" else "right")
            q[passive] = 0.37
            sim = FakeSim()
            result = select_candidates(FakeRobot(), sim, [candidate("left", 0.2), candidate("right", 0.1)], [side], q)
            self.assertEqual(result["selected"]["selected_side"], side)
            for checked in sim.checked:
                np.testing.assert_array_equal(checked[passive], q[passive])
            self.assertTrue(any(e["reasons"] == ["disabled_arm"] for e in result["evaluations"]))

    def test_weights_ties_and_no_finger_effort(self) -> None:
        rows = [candidate("right", 0.2), candidate("left", 0.2)]
        for t in rows[1]["targets"]["left"].values():
            t["joint_positions_rad"] = dict.fromkeys(hand_names("left"), 2.0)
        result = select_candidates(FakeRobot(), FakeSim(), rows, ["left", "right"], np.zeros(56), [1, 2, 3, 4, 5, 6])
        self.assertEqual(result["selected"]["candidate_id"], "left:c0")
        self.assertAlmostEqual(result["selected"]["effort"], 0.04 * 21)

    def test_multi_arm_parsed_but_explicitly_unsupported(self) -> None:
        row = candidate("left", 0.1)
        row["required_arms"].append("right")
        row["targets"].update(candidate("right", 0.1)["targets"])
        validate_candidate(row)
        result = select_candidates(FakeRobot(), FakeSim(), [row], ["left"], np.zeros(56))
        self.assertEqual(result["evaluations"][0]["reasons"], ["unsupported_multi_arm_candidate"])

    def test_nan_infeasible_and_invalid_permissions(self) -> None:
        for permission in ([], ["both"], ["left", "left"], "left"):
            with self.assertRaises(ValueError):
                enabled_set(permission)
        rows = [candidate("left", float("nan")), candidate("right", 6)]
        result = select_candidates(FakeRobot(), FakeSim(), rows, ["left", "right"], np.zeros(56))
        self.assertIsNone(result["selected"])
        self.assertEqual(result["evaluations"][0]["reasons"], ["invalid_candidate"])
        with self.assertRaises(ValueError):
            select_candidates(FakeRobot(), FakeSim(), [], ["left"], np.zeros(56), [1, 1, 1, 1, 1, float("nan")])

    def test_legacy_contact_boundaries_both_sides_and_geom_orders(self) -> None:
        from ada_grasp_ctrl.experiments.crx_sharpa_right_simulation import invalid_initialization

        fields = ("arm", "hand_self", "hand_object", "hand_table")
        cases = (
            ("within_robot", ("J2_link", "J5_link"), (), "arm"),
            ("within_robot", ("J6_link", "hand_C_MC"), ("hand_C_MC",), "arm"),
            ("robot_object", ("J6_link", "object"), (), "arm"),
            ("robot_table", ("base_link", "table"), (), "arm"),
            ("within_robot", ("hand_C_MC", "thumb_MC"), ("hand_C_MC", "thumb_MC"), "hand_self"),
            ("within_robot", ("ring_PP", "pinky_MP"), ("ring_PP", "pinky_MP"), "hand_self"),
            ("robot_object", ("index_DP", "object"), ("index_DP",), "hand_object"),
            ("robot_table", ("hand_C_MC", "table"), ("hand_C_MC",), "hand_table"),
        )
        for side in ("left", "right"):
            for category, suffixes, hand_suffixes, field in cases:
                threshold = 1e-5 if field == "arm" else 0.01
                for depth in (0.0, np.nextafter(threshold, 0), threshold, np.nextafter(threshold, np.inf)):
                    for reverse in (False, True):
                        with self.subTest(side=side, category=category, field=field, depth=depth, reverse=reverse):
                            bodies = [s if s in ("object", "table") else f"{side}_{s}" for s in suffixes]
                            row = dict(
                                category=category,
                                bodies=bodies[::-1] if reverse else bodies,
                                hand_bodies=[f"{side}_{s}" for s in hand_suffixes],
                                sides=[side],
                                penetration_m=float(depth),
                                geoms=["collision_a", "collision_b"],
                            )
                            diagnostic = {f"{f}_penetration_m": depth if f == field else 0.0 for f in fields}
                            before = deepcopy(row)
                            rejected = forbidden_contacts([row], side)
                            self.assertEqual(bool(rejected), invalid_initialization(diagnostic))
                            self.assertEqual(bool(rejected), depth > threshold)
                            self.assertEqual(row, before)
                            if rejected:
                                self.assertIs(rejected[0], row)
                            if field != "hand_object":
                                other = "right" if side == "left" else "left"
                                self.assertEqual(forbidden_contacts([row], other), rejected)

    def test_dual_arm_and_unknown_contacts_keep_strict_gate(self) -> None:
        for side, other in (("left", "right"), ("right", "left")):
            hand = f"{side}_index_DP"
            cases = (
                ("left_right", [hand, f"{other}_thumb_DP"], [hand, f"{other}_thumb_DP"], ["left", "right"]),
                ("left_right", [f"{side}_J6_link", f"{other}_thumb_DP"], [f"{other}_thumb_DP"], ["left", "right"]),
                ("robot_object", [hand, "object"], [hand], [side]),
                ("robot_environment", [hand, "obstacle"], [hand], [side]),
                ("new_category", [hand, "obstacle"], [hand], [side]),
                ("within_robot", [hand, f"{side}_ee_mount"], [hand], [side]),
                ("within_robot", [hand, f"{other}_thumb_DP"], [hand, f"{other}_thumb_DP"], ["left", "right"]),
                ("robot_table", [f"{side}_hand_unknown", "table"], [], [side]),
            )
            for category, bodies, hands, sides in cases:
                for depth in (np.nextafter(1e-5, 0), 1e-5, np.nextafter(1e-5, np.inf), 0.01):
                    with self.subTest(side=side, category=category, bodies=bodies, depth=depth):
                        row = dict(
                            category=category, bodies=bodies, hand_bodies=hands, sides=sides, penetration_m=depth
                        )
                        self.assertEqual(forbidden_contacts([row], other), [row] if depth > 1e-5 else [])

    def test_base_j1_exclusion_requires_exact_adjacent_same_side_pair(self) -> None:
        for side in ("left", "right"):
            for reverse in (False, True):
                bodies = [f"{side}_base_link", f"{side}_J1_link"]
                row = dict(
                    category="within_robot",
                    bodies=bodies[::-1] if reverse else bodies,
                    sides=[side],
                    penetration_m=0.02,
                    neighboring_source_links=True,
                )
                before = deepcopy(row)
                self.assertEqual(forbidden_contacts([row], side), [])
                self.assertEqual(row, before)
                for changes in (
                    dict(neighboring_source_links=False),
                    dict(bodies=[f"{side}_base_link", f"{side}_J2_link"]),
                    dict(category="left_right", sides=["left", "right"]),
                    dict(bodies=["left_base_link", "right_J1_link"]),
                    dict(category="robot_table", bodies=[f"{side}_base_link", "table"]),
                ):
                    bad = {**row, **changes}
                    self.assertEqual(forbidden_contacts([bad], side), [bad])
                del row["neighboring_source_links"]
                self.assertEqual(forbidden_contacts([row], side), [row])

    def test_selection_checks_all_stages_with_legacy_hand_gate(self) -> None:
        for side in ("left", "right"):
            for failing_stage in (*STAGES, None):
                contacts = dict(
                    category="within_robot",
                    sides=[side],
                    bodies=[f"{side}_ring_PP", f"{side}_pinky_MP"],
                    hand_bodies=[f"{side}_ring_PP", f"{side}_pinky_MP"],
                )
                rows = [candidate(side, 0.1)]
                for index, stage in enumerate(STAGES):
                    rows[0]["targets"][side][stage]["joint_positions_rad"][hand_names(side)[0]] = index * 0.1

                def check(q: np.ndarray) -> list[dict]:
                    stage = STAGES[round(q[side_slice(side).start + 6] * 10)]
                    depth = np.nextafter(0.01, np.inf) if stage == failing_stage else 0.01
                    return [{**contacts, "penetration_m": float(depth)}]

                with patch.object(FakeSim, "check_static_contacts", side_effect=check):
                    result = select_candidates(FakeRobot(), FakeSim(), rows, [side], np.zeros(56))
                self.assertEqual(result["selected"] is None, failing_stage is not None)
                if failing_stage is None:
                    self.assertAlmostEqual(result["selected"]["effort"], 0.06)
                else:
                    for attempt in result["evaluations"][0]["attempts"]:
                        self.assertEqual(attempt["reasons"], [f"{failing_stage}:collision"])

    def test_only_active_hand_object_gate_is_limited_to_pregrasp(self) -> None:
        for side in ("left", "right"):
            other = "right" if side == "left" else "left"
            row = dict(
                category="robot_object",
                sides=[side],
                bodies=[f"{side}_thumb_DP", "object"],
                hand_bodies=[f"{side}_thumb_DP"],
                penetration_m=0.02,
            )
            for phase in ("initial", *STAGES, "lift", "hold"):
                for reverse in (False, True):
                    contact = {**row, "bodies": row["bodies"][::-1] if reverse else row["bodies"]}
                    self.assertEqual(
                        bool(forbidden_contacts([contact], side, phase=phase)), phase in ("initial", "pregrasp")
                    )
                    self.assertEqual(forbidden_contacts([contact], other, phase=phase), [contact])
                for changes in (
                    dict(bodies=[f"{side}_J6_link", "object"], hand_bodies=[]),
                    dict(category="robot_table", bodies=[f"{side}_thumb_DP", "table"]),
                    dict(
                        category="within_robot",
                        bodies=[f"{side}_thumb_DP", f"{side}_thumb_PP"],
                        hand_bodies=[f"{side}_thumb_DP", f"{side}_thumb_PP"],
                    ),
                ):
                    protected = {**row, **changes}
                    self.assertEqual(forbidden_contacts([protected], side, phase=phase), [protected])
        with self.assertRaisesRegex(ValueError, "Unknown contact-check phase"):
            forbidden_contacts([], "right", phase="typo")

    def test_selector_keeps_grasp_squeeze_contacts_but_only_gates_pregrasp(self) -> None:
        for side in ("left", "right"):
            for deep_stage in STAGES:
                rows = [candidate(side, 0.1)]
                for index, stage in enumerate(STAGES):
                    rows[0]["targets"][side][stage]["joint_positions_rad"][hand_names(side)[0]] = index * 0.1

                def check(q: np.ndarray) -> list[dict]:
                    stage = STAGES[round(q[side_slice(side).start + 6] * 10)]
                    return [
                        dict(
                            category="robot_object",
                            sides=[side],
                            bodies=[f"{side}_thumb_DP", "object"],
                            hand_bodies=[f"{side}_thumb_DP"],
                            penetration_m=0.02 if stage == deep_stage else 0.0,
                        )
                    ]

                with patch.object(FakeSim, "check_static_contacts", side_effect=check):
                    result = select_candidates(FakeRobot(), FakeSim(), rows, [side], np.zeros(56))
                self.assertEqual(result["selected"] is None, deep_stage == "pregrasp")
                if deep_stage != "pregrasp":
                    stage = result["selected"]["stages"][deep_stage]
                    self.assertEqual(stage["contacts"][0]["penetration_m"], 0.02)
                    self.assertEqual(stage["forbidden_contacts"], [])
                    self.assertAlmostEqual(result["selected"]["effort"], 0.06)

    def test_reflection_and_legacy_stage_preservation(self) -> None:
        points = np.random.default_rng(4).normal(size=(30, 3))
        plane = reflection_plane(points, dict(plane_normal_world=[1, 2, 3]))
        np.testing.assert_allclose(reflect_points(reflect_points(points, plane), plane), points, atol=1e-14)
        original = candidate("right", 0.3)["targets"]["right"]
        original["pregrasp"]["T_world_hand"][1][3] = 0.17
        raw = dict(candidate_id="c2", stages=original, log_probability=3.0, raw_candidate_index=7)
        before = deepcopy(raw)
        right = from_right_record(raw, {})
        self.assertEqual(right["targets"]["right"], before["stages"])
        self.assertEqual(raw, before)
        left = from_right_record(raw, {}, side="left", plane=plane)
        from ada_grasp_ctrl.sharpa_mirror import SharpaHandMirror

        mirror = SharpaHandMirror()
        for stage in STAGES:
            pose = left["targets"]["left"][stage]["T_world_hand"]
            recovered = mirror.mirror_wrist_pose(pose, plane["normal_world"], plane["point_world_m"])
            np.testing.assert_allclose(recovered, original[stage]["T_world_hand"], atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(np.asarray(pose)[:3, :3]), 1.0)
        self.assertEqual(left["provenance"]["source_record"]["raw_candidate_index"], 7)
        raw["stages"]["grasp"]["quaternion_wxyz"] = [0, 1, 0, 0]
        with self.assertRaisesRegex(ValueError, "WXYZ and matrix"):
            from_right_record(raw, {})


@unittest.skipUnless(os.environ.get("ADA_DUAL_ARM_SNAPSHOT"), "Requires explicit real Phase 1 snapshot")
class RealSceneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = yaml.safe_load((project_root() / "config/simulation_scene.yaml").read_text())
        self.config["scene"]["robot_snapshot"] = os.environ["ADA_DUAL_ARM_SNAPSHOT"]

    def scene(self, config: dict | None = None) -> ObjectScene:
        path = self.root / str(len(list(self.root.iterdir())))
        path.mkdir()
        return ObjectScene(config or self.config, path)

    def render_in_fresh_process_if_needed(self) -> bool:
        """Isolate EGL when earlier suite modules already imported MuJoCo's GLFW backend."""
        if mujoco.GLContext.__module__ == "mujoco.egl":
            return False
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["PYTHONPATH"] = os.pathsep.join(str(project_root() / part) for part in ("src", "tests"))
        command = [
            sys.executable,
            "-m",
            "unittest",
            f"test_phase2_pipeline.RealSceneTests.{self._testMethodName}",
            "-v",
        ]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return True

    def test_mesh_pose_scale_determinism_and_real_surface(self) -> None:
        scene = self.scene()
        p, info = scene.cloud()
        again, _ = scene.cloud()
        np.testing.assert_array_equal(p, again)
        pose = np.array(info["T_world_object"])
        local = (p[:20] - pose[:3, 3]) @ pose[:3, :3]
        _, distance, _ = trimesh.proximity.closest_point_naive(scene.mesh, local)
        self.assertLess(max(distance), 1e-10)
        self.assertEqual(scene.sim.model.nq, 63)

    def test_physical_bearing_exclusion_preserves_source_and_other_collision_geoms(self) -> None:
        scene = self.scene()
        self.assertEqual(scene.robot.mj_model.nexclude, 0)
        self.assertEqual(scene.sim.model.nexclude, 2)
        expected = []
        for side in ("left", "right"):
            base = scene.sim.model.body(f"{side}_base_link").id
            child = scene.sim.model.body(f"{side}_J1_link").id
            expected.append((min(base, child) << 16) + max(base, child))
        self.assertEqual(sorted(scene.sim.model.exclude_signature.tolist()), sorted(expected))
        for index in range(scene.robot.mj_model.ngeom):
            original = scene.robot.mj_model.geom(index)
            actual = scene.sim.model.geom(original.name)
            np.testing.assert_array_equal(actual.contype, original.contype)
            np.testing.assert_array_equal(actual.conaffinity, original.conaffinity)
            np.testing.assert_array_equal(actual.friction, original.friction)
        np.testing.assert_array_equal(scene.sim.model.actuator_gainprm, scene.robot.mj_model.actuator_gainprm)
        np.testing.assert_array_equal(scene.sim.model.actuator_biasprm, scene.robot.mj_model.actuator_biasprm)
        np.testing.assert_array_equal(scene.sim.model.actuator_forcerange, scene.robot.mj_model.actuator_forcerange)

    def test_both_j1_joints_track_without_bearing_friction_or_relaxed_limits(self) -> None:
        from ada_grasp_ctrl.experiments.dual_arm_grasp_episode import live_contacts, smoothstep

        for side in ("left", "right"):
            scene = self.scene()
            sim = scene.sim
            initial = sim.configuration()
            target = initial.copy()
            arm = side_slice(side, True)
            block = side_slice(side)
            target[arm.start] -= 0.126
            steps = round(self.config["execution"]["durations_s"]["pregrasp"] / sim.model.opt.timestep)
            for index in range(1, steps + 2):
                q = initial + smoothstep(min(1.0, index / steps)) * (target - initial)
                sim.set_target(q[block], side=side)
                apply_arm_bias_compensation(sim)
                sim.step()
                if index % 10 == 0:
                    self.assertEqual(forbidden_contacts(live_contacts(sim), side), [])
            self.assertLessEqual(
                np.max(np.abs(sim.configuration()[arm] - target[arm])),
                self.config["execution"]["arm_arrival_tolerance_rad"],
            )
            self.assertFalse(np.any(sim.data.warning.number))

    def test_static_and_live_contact_policy_agree(self) -> None:
        from ada_grasp_ctrl.experiments.dual_arm_grasp_episode import live_contacts

        scene = self.scene()
        q = scene.sim.configuration()
        for side in ("left", "right"):
            # Include articulated fingers, not only the natural palm/thumb contact.
            q[side_slice(side).start + 6 : side_slice(side).stop] = scene.robot.limits[side_slice(side)][6:, 1] * 0.5
            scene.sim.reset(q)
            static = scene.sim.check_static_contacts(q)
            live = live_contacts(scene.sim)
            self.assertTrue(static)
            for original, actual in zip(static, live):
                self.assertEqual(actual, {key: original[key] for key in actual})
            self.assertEqual(len(static), len(live))
            self.assertEqual(
                [r["bodies"] for r in forbidden_contacts(static, side)],
                [r["bodies"] for r in forbidden_contacts(live, side)],
            )

    def test_depth_is_visible_real_surface_and_reprojects(self) -> None:
        if self.render_in_fresh_process_if_needed():
            return
        c = deepcopy(self.config)
        c["point_cloud"]["source"] = "simulated_depth"
        scene = self.scene(c)
        p, info = scene.cloud()
        self.assertGreater(len(p), 20)
        self.assertLess(info["reprojection_max_pixels"], 1e-8)
        pose = np.array(info["T_world_object"])
        local = (p[:: max(1, len(p) // 20)] - pose[:3, 3]) @ pose[:3, :3]
        _, distance, _ = trimesh.proximity.closest_point_naive(scene.mesh, local)
        self.assertLess(max(distance), 0.002)
        self.assertNotEqual(len(p), c["point_cloud"]["num_points"])
        archive = np.load(scene.output / "depth.npz")
        self.assertEqual(int(archive["mask"].sum()), len(p))

    def test_invisible_depth_fails_without_oracle_fallback(self) -> None:
        if self.render_in_fresh_process_if_needed():
            return
        c = deepcopy(self.config)
        c["point_cloud"]["source"] = "simulated_depth"
        c["camera"]["look_at_world_m"] = [3, 3, 2]
        with self.assertRaisesRegex(ValueError, "visible object"):
            self.scene(c).cloud()

    def test_missing_physics_is_diagnostic_only(self) -> None:
        c = deepcopy(self.config)
        del c["object"]["physics"]
        scene = self.scene(c)
        self.assertFalse(scene.physics_available)
        self.assertEqual(scene.sim.model.nq, 56)
        self.assertGreater(len(scene.cloud()[0]), 0)

    def test_invalid_assets_scale_camera_quaternion(self) -> None:
        for section, key, value in (
            ("object", "mesh", "/missing/bottle.obj"),
            ("object", "scale", [0, 1, 1]),
            ("camera", "far_m", 0.001),
        ):
            c = deepcopy(self.config)
            c[section][key] = value
            with self.assertRaises((ValueError, OSError)):
                self.scene(c)
        c = deepcopy(self.config)
        c["object"]["pose"]["quaternion_wxyz"] = [2, 0, 0, 0]
        with self.assertRaises(ValueError):
            self.scene(c)

    def test_real_selector_all_permissions_and_fk_frames(self) -> None:
        scene = self.scene()
        rows = fixture_candidates(scene, "test")
        for enabled in (["left"], ["right"], ["left", "right"]):
            result = select_candidates(scene.robot, scene.sim, rows, enabled, scene.sim.configuration())
            self.assertIn(result["selected"]["selected_side"], enabled)
            self.assertAlmostEqual(result["selected"]["effort"], 0, places=10)

    def test_scene_bound_replay_rejects_changed_geometry(self) -> None:
        first, second = self.root / "first", self.root / "second"
        first.mkdir()
        second.mkdir()
        # Missing physics ends after selection while preserving reusable candidates.
        c = deepcopy(self.config)
        del c["object"]["physics"]
        result = run(c, first, fixture=True)
        self.assertEqual(result["status"], "physics_unavailable")
        c["object"]["scale"] = [0.09, 0.09, 0.09]
        with self.assertRaisesRegex(ValueError, "Replay scene identity"):
            run(c, second, replay=first / "candidates.json")
        c = deepcopy(self.config)
        del c["object"]["physics"]
        c["scene"]["table_half_extents_xy_m"] = [1, 1]
        third = self.root / "third"
        third.mkdir()
        with self.assertRaisesRegex(ValueError, "Replay scene identity"):
            run(c, third, replay=first / "candidates.json")

    def test_wide_table_supports_right_bottle_without_moving_its_initial_pose(self) -> None:
        config = deepcopy(self.config)
        config["scene"]["table_half_extents_xy_m"] = [1, 1]
        config["object"]["pose"]["position_m"] = [0.77, -0.657, 0.072]
        scene = self.scene(config)
        sim = scene.sim
        np.testing.assert_array_equal(sim.data.body("object").xpos, config["object"]["pose"]["position_m"])
        table = sim.model.geom("calibration_table_surface_collision_0")
        np.testing.assert_array_equal(table.size, [1, 1, 0.02])
        np.testing.assert_array_equal(scene.robot.mj_model.geom(table.name).size, [0.15, 0.15, 0.02])
        for _ in range(500):
            apply_arm_bias_compensation(sim)
            sim.step()
        self.assertGreater(sim.data.body("object").xpos[2], 0)
        self.assertTrue(
            any(
                {sim.body_labels[int(sim.model.geom_bodyid[g])] for g in contact.geom} == {"object", "table"}
                for contact in sim.data.contact
            )
        )

    def test_full_dynamic_fixture_all_permissions_no_fake_lift(self) -> None:
        from ada_grasp_ctrl.experiments.dual_arm_grasp_episode import execute

        for enabled in (["left"], ["right"], ["left", "right"]):
            scene = self.scene()
            rows = fixture_candidates(scene, "test")
            selection = select_candidates(scene.robot, scene.sim, rows, enabled, scene.sim.configuration())
            result = execute(scene, selection, self.config["execution"], scene.output)
            self.assertEqual(result["status"], "failed_grasp")
            self.assertTrue(result["lift_completed"])
            self.assertTrue(result["passive_target_unchanged"])
            self.assertEqual([e["phase"] for e in result["phase_events"]], [*STAGES, "lift"])
            self.assertFalse(result["retained_contact_every_hold_sample"])
            data = np.load(scene.output / "trajectory.npz")
            self.assertTrue(np.all(np.diff(data["time_s"]) >= 0))
            self.assertEqual(data["qpos"].shape[1], scene.sim.model.nq)
            self.assertFalse(np.array_equal(data["object_pose"][0], data["object_pose"][-1]))

    def test_invalid_object_initialization_does_not_step(self) -> None:
        from ada_grasp_ctrl.experiments.dual_arm_grasp_episode import execute

        c = deepcopy(self.config)
        c["object"]["pose"]["position_m"][2] = 0.01
        scene = self.scene(c)
        rows = fixture_candidates(scene, "test")
        selection = select_candidates(scene.robot, scene.sim, rows, ["left"], scene.sim.configuration())
        result = execute(scene, selection, c["execution"], scene.output)
        self.assertEqual(result["status"], "invalid_initialization")
        self.assertEqual(scene.sim.data.time, 0)
        self.assertFalse((scene.output / "trajectory.npz").exists())

    def test_dynamic_hand_object_penetration_after_pregrasp_is_recorded_not_aborted(self) -> None:
        from ada_grasp_ctrl.experiments.dual_arm_grasp_episode import execute, live_contacts

        scene = self.scene()
        rows = fixture_candidates(scene, "test")
        selection = select_candidates(scene.robot, scene.sim, rows, ["left"], scene.sim.configuration())

        def contacts(sim: object) -> list[dict]:
            actual = live_contacts(sim)
            if sim.data.time > 2.01:
                actual.append(
                    dict(
                        category="robot_object",
                        sides=["left"],
                        bodies=["left_thumb_DP", "object"],
                        hand_bodies=["left_thumb_DP"],
                        penetration_m=0.02,
                    )
                )
            return actual

        with patch("ada_grasp_ctrl.experiments.dual_arm_grasp_episode.live_contacts", side_effect=contacts):
            result = execute(scene, selection, self.config["execution"], scene.output)
        self.assertTrue(result["lift_completed"])
        self.assertEqual(result["status"], "failed_grasp")
        self.assertTrue(any(c["penetration_m"] == 0.02 for r in result["contacts"] for c in r["contacts"]))

    def test_intermediate_collision_aborts_before_any_motion(self) -> None:
        from ada_grasp_ctrl.experiments.dual_arm_grasp_episode import execute

        scene = self.scene()
        rows = fixture_candidates(scene, "test")
        selection = select_candidates(scene.robot, scene.sim, rows, ["left"], scene.sim.configuration())
        original = scene.sim.data.qpos.copy()
        bad = dict(category="robot_table", sides=["left"], bodies=["left_J6_link", "table"], penetration_m=0.01)
        with patch.object(scene.sim, "check_static_contacts", side_effect=[[], [bad]]):
            result = execute(scene, selection, self.config["execution"], scene.output)
        self.assertEqual(result["status"], "motion_failed")
        self.assertFalse(result["execution_ready"])
        np.testing.assert_array_equal(scene.sim.data.qpos, original)
        self.assertEqual(scene.sim.data.time, 0)


if __name__ == "__main__":
    unittest.main()
