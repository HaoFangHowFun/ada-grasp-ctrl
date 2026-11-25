"""Finite calibrated tabletop, cached editing meshes and settled simulation observations."""

from copy import deepcopy
from pathlib import Path
import threading

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ada_grasp_ctrl.experiments.sharpa_contact_profile import BENCH_PROFILE, BENCH_FRICTION
from ada_grasp_ctrl.experiments.simulated_object_cloud import (
    ObjectScene,
    body_pose,
    initial_configuration,
    load_mesh,
    vector,
)
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json
from ada_grasp_ctrl.grasp_selection.multi_arm import forbidden_contacts
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.placement.session import write_json
from ada_grasp_ctrl.robot_environment import DualArmModel, DualArmSimulation
from .dual_scene import live_scene

DEFAULT_LIBRARY = "src/ada_grasp_ctrl/config/experiments/simulation_objects.yaml"


def table_frame(plane: dict, table: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a tabletop frame from the measured plane and explicit finite footprint."""
    n = vector(plane["normal_xyz"], 3, "table normal")
    if not np.isclose(np.linalg.norm(n), 1, atol=1e-6) or n[2] <= 0 or not np.isfinite(plane["d"]):
        raise ValueError("Expected finite upward unit table plane")
    xy = vector(table["center_xy_m"], 2, "table center")
    size = vector(table["half_extents_xy_m"], 2, "table half extents")
    if np.any(size <= 0):
        raise ValueError("Table dimensions must be positive")
    center = np.r_[xy, (-plane["d"] - n[:2] @ xy) / n[2]]
    rotation, _ = Rotation.align_vectors(n[None], [[0, 0, 1]])
    return center, rotation.as_matrix(), size


def install_table(spec: mujoco.MjSpec, plane: dict, table: dict) -> dict:
    """Replace obsolete nominal table geoms in an independent physics specification."""
    center, rotation, size = table_frame(plane, table)
    for geom in spec.geoms:
        if geom.name.startswith("calibration_table_surface"):
            geom.contype = geom.conaffinity = 0
            geom.rgba[3] = 0
    thickness = float(table["thickness_m"])
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("Table thickness must be positive")
    body = spec.worldbody.add_body(
        name="simulation_table",
        pos=center,
        quat=Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]],
    )
    body.add_geom(
        name="simulation_table_surface",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, -thickness / 2],
        size=[*size, thickness / 2],
        group=2,
        rgba=[0.55, 0.45, 0.3, 0.7],
    )
    return dict(**table, plane=plane, top_center_world_m=center.tolist())


class SceneBuilder:
    """Load visual assets once; edits are cheap mesh transforms with no physics or inference."""

    def __init__(self, config: dict, library: Path | None = None) -> None:
        self.config = config
        self.library = yaml.safe_load((library or resolve_from_root(DEFAULT_LIBRARY)).read_text())
        self.robot = DualArmModel.load(resolve_from_root(config["robot_snapshot"]))
        self.plane = read_json(resolve_from_root(config["table_plane"]))
        self.table = deepcopy(config.get("simulation", {}).get("table", self.library["table"]))
        self.center, self.rotation, self.half_extents = table_frame(self.plane, self.table)
        self.entries = {entry["id"]: entry for entry in self.library["objects"]}
        self.meshes = {}
        for key, entry in self.entries.items():
            if file_hash(resolve_from_root(entry["mesh"])) != entry["mesh_sha256"]:
                raise ValueError(f"Manifest visual mesh changed: {key}")
            self.meshes[key] = load_mesh(resolve_from_root(entry["mesh"]), np.ones(3))
            for path in entry["collision_meshes"]:
                hull = load_mesh(resolve_from_root(path), np.ones(3)).convex_hull
                if not np.isfinite(hull.volume) or hull.volume <= 0:
                    raise ValueError(f"Invalid collision volume: {path}")
        self.initial = initial_configuration(self.robot, config.get("simulation", {}))
        simulation = config.get("simulation", {})
        if (
            simulation.get("gravity_mode", "object_only") == "object_only"
            and simulation.get("initial_robot_state", "snapshot_initial_pose") == "snapshot_initial_pose"
        ):
            # Zero-gravity fingers at exactly zero can settle ~1e-7 rad below a soft
            # limit. Start the default open hand just inside bounds; never clip
            # observed physics or user-specified initial states/grasp targets.
            fingers = np.r_[6:28, 34:56]
            self.initial[fingers] = np.clip(
                self.initial[fingers], self.robot.limits[fingers, 0] + 1e-4, self.robot.limits[fingers, 1] - 1e-4
            )
        # Planning uses exactly the existing live policy, including its infinite table.
        _, self.static = live_scene(self.robot.directory, self.state(self.initial), self.plane)
        bad = forbidden_contacts(self.static.check_static_contacts(self.initial), ["left", "right"], phase="initial")
        if bad:
            raise ValueError(f"Initial robot collision: {bad}")
        spec = self.robot.spec.copy()
        install_table(spec, self.plane, self.table)
        self.display = DualArmSimulation(self.robot, scene_spec=spec, table_bodies=("simulation_table",))
        self.display.reset(self.initial)

    def state(self, q: np.ndarray) -> dict:
        """Produce genuine canonical simulation state without ROS feedback fields."""
        return dict(
            schema_version=1,
            environment="simulation",
            frame_id="world",
            joint_unit="rad",
            full_joint_names=list(self.robot.joint_names),
            full_positions_rad=q.tolist(),
            model_sha256=file_hash(self.robot.urdf_path),
        )

    def defaults(self, object_id: str | None = None) -> dict:
        """Return reproducible requested scene values, distinct from a settled pose."""
        key = object_id or next(iter(self.entries))
        return self.tabletop_request(key, self.library["default_xy_m"], 0.0, int(self.config.get("seed", 4)))

    def tabletop_request(
        self, object_id: str, xy_m: list[float], yaw_deg: float, seed: int, scale: float | None = None
    ) -> dict:
        """Build a fixed-orientation support pose; yaw rotates around the measured table normal.

        XY is constrained so the complete mesh stays at least 5 mm inside the table.
        Explicit legacy RPY/scale requests remain supported by ``pose``.
        """
        entry = self.entries[object_id]
        quaternion = vector(entry["support_quaternion_wxyz"], 4, "support quaternion")
        if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6) or not np.isfinite(yaw_deg):
            raise ValueError("Expected a unit support quaternion and finite yaw")
        rotation = (
            Rotation.from_matrix(self.rotation)
            * Rotation.from_euler("z", yaw_deg, degrees=True)
            * Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        )
        requested = dict(
            object_id=object_id,
            xy_m=vector(xy_m, 2, "XY").tolist(),
            yaw_deg=float(yaw_deg),
            rpy_deg=rotation.as_euler("xyz", degrees=True).tolist(),
            scale=entry["default_scale"] if scale is None else float(scale),
            seed=int(seed),
            pose_policy="fixed_tabletop_support",
        )
        _, vertices = self.pose(requested)
        local = (vertices - self.center) @ self.rotation
        limit = self.half_extents - 0.005
        low, high = -limit - local[:, :2].min(axis=0), limit - local[:, :2].max(axis=0)
        if np.any(low > high):
            raise ValueError("Object cannot fit within the finite tabletop")
        shift = np.clip(np.zeros(2), low, high)
        if np.any(np.abs(shift) > 1e-12):
            # Translate along the tilted plane; pose() then recomputes exact support Z.
            requested["xy_m"] = (np.asarray(requested["xy_m"]) + (self.rotation @ np.r_[shift, 0])[:2]).tolist()
        return requested

    def random_tabletop_request(self, seed: int) -> dict:
        """Reproduce an object, on-table XY and yaw without changing support or scale."""
        rng = np.random.default_rng(seed)
        key = str(rng.choice(list(self.entries)))
        yaw = float(rng.uniform(-180, 180))
        local = np.r_[rng.uniform(-0.5, 0.5, 2) * self.half_extents, 0]
        xy = (self.center + self.rotation @ local)[:2].tolist()
        return self.tabletop_request(key, xy, yaw, seed)

    def pose(self, requested: dict) -> tuple[np.ndarray, np.ndarray]:
        """Solve support height using the full rotated visual mesh and measured plane."""
        key = requested["object_id"]
        entry = self.entries[key]
        scale = float(requested["scale"])
        if not np.isfinite(scale) or not entry["scale_range"][0] <= scale <= entry["scale_range"][1]:
            raise ValueError("Object scale outside manifest range")
        rotation = Rotation.from_euler("xyz", vector(requested["rpy_deg"], 3, "RPY"), degrees=True).as_matrix()
        xy = vector(requested["xy_m"], 2, "XY")
        vertices = self.meshes[key].vertices * scale @ rotation.T
        n, d = np.asarray(self.plane["normal_xyz"]), self.plane["d"]
        z = (self.library["clearance_m"] - d - n[:2] @ xy - np.min(vertices @ n)) / n[2]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, [*xy, z]
        return pose, vertices + pose[:3, 3]

    def check_footprint(self, vertices: np.ndarray) -> None:
        """Reject any mesh vertex outside the finite tabletop's projected footprint."""
        local = (vertices - self.center) @ self.rotation
        if np.any(np.abs(local[:, :2]) > self.half_extents + 1e-6):
            raise ValueError("Object extends beyond the finite tabletop; move or resize it")

    def confirm(
        self, requested: dict, root: Path, cancel: threading.Event | None = None
    ) -> tuple[ObjectScene, Path, dict]:
        """Compile and settle once, then persist immutable observation and physical evidence."""
        pose, vertices = self.pose(requested)
        self.check_footprint(vertices)
        entry = self.entries[requested["object_id"]]
        profile = self.config.get("simulation", {}).get("contact_profile", "source")
        physics = deepcopy(entry["physics"])
        if profile == BENCH_PROFILE:
            physics["friction"] = list(BENCH_FRICTION)
            physics["assumptions"] += " Friction overridden by explicit DexGraspBench contact profile."
        config = dict(
            scene=dict(
                contact_profile=profile,
                robot_snapshot=str(self.robot.directory),
                initial_robot_state=dict(zip(self.robot.joint_names, self.initial.tolist())),
                table_plane=self.plane,
                table=self.table,
                gravity_mode=self.config.get("simulation", {}).get("gravity_mode", "object_only"),
                settling=self.config.get("simulation", {}).get("settling", self.library.get("settling", {})),
            ),
            object=dict(
                name=entry["id"],
                mesh=entry["mesh"],
                collision_meshes=entry["collision_meshes"],
                scale=[requested["scale"]] * 3,
                freejoint=True,
                placement="settle_on_table",
                physics=physics,
                pose=dict(
                    frame_id="world",
                    position_m=pose[:3, 3].tolist(),
                    quaternion_wxyz=Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist(),
                ),
            ),
            point_cloud=dict(source="complete_mesh", num_points=8192, object_only=True, seed=requested["seed"]),
            execution=dict(
                arm_bias_compensation=self.config.get("simulation", {}).get("gravity_mode", "object_only") == "world"
            ),
        )
        write_json(root / "requested_scene.json", requested)
        (root / "scene_config.yaml").write_text(yaml.safe_dump(config))
        scene = ObjectScene(config, root)
        initial = scene.sim.data
        np.savez_compressed(
            root / "initial_state.npz",
            qpos=initial.qpos,
            qvel=initial.qvel,
            ctrl=initial.ctrl,
            qfrc_applied=initial.qfrc_applied,
            xfrc_applied=initial.xfrc_applied,
        )
        bad = forbidden_contacts(scene.sim.check_static_contacts(self.initial), ["left", "right"], phase="initial")
        if bad:
            raise ValueError(f"Invalid initial physical scene: {bad}")
        scene.settle(canceled=cancel.is_set if cancel is not None else None)
        mujoco.mj_forward(scene.sim.model, scene.sim.data)
        settled = body_pose(scene.sim)
        actual_vertices = scene.mesh.vertices @ settled[:3, :3].T + settled[:3, 3]
        self.check_footprint(actual_vertices)
        if np.min(actual_vertices @ np.asarray(self.plane["normal_xyz"]) + self.plane["d"]) < -0.002:
            raise ValueError("Settled mesh penetrates the table by more than 2 mm")
        q = scene.sim.configuration()
        if np.max(np.abs((q - self.initial)[np.r_[0:6, 28:34]])) > 0.01:
            raise ValueError("Robot drift during settling exceeds existing 0.01 rad tolerance")
        scene.robot.configuration(q)
        points, cloud_info = scene.cloud()
        points.setflags(write=False)
        write_json(
            root / "physics_scene.json",
            dict(
                metadata=scene.metadata, requested_pose=pose.tolist(), settled_pose=settled.tolist(), cloud=cloud_info
            ),
        )
        mujoco.mj_saveModel(scene.sim.model, str(root / "scene.mjb"))
        np.savez_compressed(
            root / "settled_state.npz",
            qpos=scene.sim.data.qpos,
            qvel=scene.sim.data.qvel,
            ctrl=scene.sim.data.ctrl,
            time=scene.sim.data.time,
            qfrc_applied=scene.sim.data.qfrc_applied,
            xfrc_applied=scene.sim.data.xfrc_applied,
        )
        obj = root / "object"
        obj.mkdir()
        np.savez_compressed(obj / "object.npz", points_world_m=points)
        source = dict(
            scene_sha256=file_hash(root / "scene.mjb"),
            model_sha256=file_hash(scene.robot.urdf_path),
            mesh_sha256=scene.metadata["mesh_sha256"],
            seed=requested["seed"],
        )
        info = dict(
            schema_version=1,
            frame_id="world",
            unit="m",
            source_type="simulation_complete_mesh",
            simulation_source=source,
            object_sha256=file_hash(obj / "object.npz"),
            table_status=self.plane["status"],
            plane_path=str(resolve_from_root(self.config["table_plane"])),
            plane_sha256=file_hash(resolve_from_root(self.config["table_plane"])),
        )
        write_json(obj / "object.json", info)
        state = self.state(q)
        write_json(root / "robot_state.json", state)
        return scene, obj, state
