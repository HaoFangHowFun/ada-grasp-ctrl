"""One real-mesh MuJoCo world and metric complete/depth object observations."""

from __future__ import annotations

from pathlib import Path
import shutil

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.robot_environment import DualArmModel, DualArmSimulation
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash

from .sharpa_contact_profile import apply_contact_profile


def vector(value: object, size: int, label: str) -> np.ndarray:
    """Require finite numeric vectors without repairing external data."""
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected {size} finite values")
    return result


def positive(value: object, label: str) -> float:
    """Require a finite strictly positive scalar."""
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def load_mesh(path: Path, scale: np.ndarray) -> trimesh.Trimesh:
    """Load genuine triangles, preserving scene transforms and the mesh origin."""
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Invalid or unsupported triangle mesh: {path}")
    mesh.apply_scale(scale)
    if not np.isfinite(mesh.area) or mesh.area <= 0 or np.any(mesh.extents <= 0):
        raise ValueError(f"Degenerate mesh: {path}")
    return mesh


def initial_configuration(robot: DualArmModel, config: dict) -> np.ndarray:
    """Read copied upstream degrees or an explicit complete named 56-joint radian state."""
    initial = config.get("initial_robot_state", "snapshot_initial_pose")
    if initial == "snapshot_initial_pose":
        path = robot.directory / "initial_pose.yaml"
        if not path.is_file():
            raise ValueError(f"Snapshot lacks {path.name}; supply an explicit named 56-joint initial_robot_state")
        values = yaml.safe_load(path.read_text())
        q = np.zeros(56)
        for side in ("left", "right"):
            q[side_slice(side, True)] = np.deg2rad(vector(values[side], 6, side + " initial degrees"))
    elif isinstance(initial, dict) and set(initial) == set(robot.joint_names):
        q = np.array([initial[n] for n in robot.joint_names], dtype=float)
    else:
        raise ValueError("initial_robot_state requires snapshot_initial_pose or exactly 56 named radians")
    return robot.configuration(q)


def body_pose(sim: DualArmSimulation) -> np.ndarray:
    """Read the actual object mesh-frame world transform."""
    result = np.eye(4)
    result[:3, :3] = sim.data.body("object").xmat.reshape(3, 3)
    result[:3, 3] = sim.data.body("object").xpos
    return result


def apply_arm_bias_compensation(sim: DualArmSimulation) -> None:
    """Hold arm targets with bounded model-based feedforward; leave object gravity untouched.

    Combined actuator and applied torques are bounded at each command evaluation by
    the URDF effort values. This is a provisional simulation controller, not identification.
    """
    mujoco.mj_forward(sim.model, sim.data)
    indices = np.r_[0:6, 28:34]
    dofs = sim.qvel[indices]
    actuator = sim.data.qfrc_actuator[dofs]
    total = np.clip(sim.data.qfrc_bias[dofs] + actuator, -sim.robot.effort[indices], sim.robot.effort[indices])
    sim.data.qfrc_applied[dofs] = total - actuator


def exclude_adjacent_base_contacts(spec: mujoco.MjSpec) -> None:
    """Apply the existing legacy base/J1 exclusion to both physical robot branches.

    MuJoCo does not filter a joint's contact with its world-welded parent. The
    imported bearing surfaces consequently generate artificial friction opposing
    J1, even though this exact adjacent pair is already excluded by the selector.
    Validate both parent relations before editing; retain all other contact pairs.

    Raises:
        ValueError: If either named base/J1 pair is missing or not adjacent.
    """
    pairs = []
    for side in ("left", "right"):
        base, child = (spec.body(f"{side}_{suffix}") for suffix in ("base_link", "J1_link"))
        if base is None or child is None or child.parent.name != base.name:
            raise ValueError(f"Expected adjacent {side}_base_link/{side}_J1_link bodies")
        pairs.append((base.name, child.name))
    for base, child in pairs:
        if not any({e.bodyname1, e.bodyname2} == {base, child} for e in spec.excludes):
            spec.add_exclude(bodyname1=base, bodyname2=child)


def enlarge_table_footprint(spec: mujoco.MjSpec, half_extents: object) -> dict:
    """Enlarge the named table's visual and collision boxes without changing its height.

    Args:
        spec: Independent scene specification, never the source robot model.
        half_extents: Two positive table-local X/Y half lengths in meters.

    Returns:
        Explicit geometry metadata used to bind cached candidates to this scene.

    Raises:
        ValueError: On invalid dimensions, a smaller footprint or unexpected geometry.
    """
    size = vector(half_extents, 2, "scene.table_half_extents_xy_m")
    if np.any(size <= 0):
        raise ValueError("scene.table_half_extents_xy_m must be positive")
    geoms = [spec.geom(f"calibration_table_surface_{kind}_0") for kind in ("visual", "collision")]
    if any(g is None or g.type != mujoco.mjtGeom.mjGEOM_BOX for g in geoms):
        raise ValueError("Expected named table visual/collision boxes")
    if any(np.any(size < g.size[:2]) for g in geoms):
        raise ValueError("Table footprint may only be enlarged from the source geometry")
    for geom in geoms:
        geom.size[:2] = size
    return dict(half_extents_xy_m=size.tolist(), geoms=[g.name for g in geoms], height_changed=False)


class ObjectScene:
    """Compile the same physical geometry for rendering, checking and execution.

    Args:
        config: Valid scene/object/camera mappings. Paths follow ada_grasp_ctrl.paths.
        output: Fresh run directory for converted geometry and source snapshots.

    Physics is enabled only with explicit mass, friction, provenance and collision meshes.
    Inertia is calculated from declared collision volume at uniform density, an assumption.
    """

    def __init__(self, config: dict, output: Path) -> None:
        self.config, self.output = config, output
        self.robot = DualArmModel.load(resolve_from_root(config["scene"]["robot_snapshot"]))
        obj, camera = config["object"], config["camera"]
        scale = vector(obj["scale"], 3, "object.scale")
        if np.any(scale <= 0):
            raise ValueError("object.scale must be positive")
        pose = obj["pose"]
        if pose["frame_id"] != "world":
            raise ValueError("Object pose must be in world")
        position = vector(pose["position_m"], 3, "object.position_m")
        quat = vector(pose["quaternion_wxyz"], 4, "object.quaternion_wxyz")
        if not np.isclose(np.linalg.norm(quat), 1, atol=1e-8, rtol=0):
            raise ValueError("Object quaternion must be normalized WXYZ")
        source = resolve_from_root(obj["mesh"]).resolve(strict=True)
        self.mesh = load_mesh(source, scale)
        geometry = output / "geometry"
        geometry.mkdir()
        shutil.copyfile(source, geometry / ("source" + source.suffix))
        self.mesh.export(geometry / "visual.stl")
        physics = obj.get("physics")
        self.physics_available = physics is not None and obj.get("freejoint") is True
        if physics is not None:
            mass = positive(physics["mass_kg"], "mass_kg")
            friction = vector(physics["friction"], 3, "friction")
            if np.any(friction < 0) or not physics.get("assumptions"):
                raise ValueError("Physics requires nonnegative friction and explicit assumptions/provenance")
            if not obj.get("collision_meshes"):
                raise ValueError("Physics requires explicit collision_meshes (visual mesh may be listed explicitly)")
        else:
            mass, friction = 0.0, np.zeros(3)
        spec = self.robot.spec.copy()
        exclude_adjacent_base_contacts(spec)
        table = None
        if "table_half_extents_xy_m" in config["scene"]:
            table = enlarge_table_footprint(spec, config["scene"]["table_half_extents_xy_m"])
        spec.add_mesh(name="phase2_visual", file=str(geometry / "visual.stl"))
        body = spec.worldbody.add_body(name="object", pos=position, quat=quat)
        if self.physics_available:
            body.add_freejoint(name="object_free")
        body.add_geom(
            name="object_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="phase2_visual",
            contype=0,
            conaffinity=0,
            mass=0,
            group=2,
            rgba=[0.3, 0.7, 0.3, 1],
        )
        collision_sources = sorted(obj.get("collision_meshes", []))
        self.collision_hashes = []
        for index, raw in enumerate(collision_sources):
            path = resolve_from_root(raw).resolve(strict=True)
            mesh = load_mesh(path, scale)
            converted = geometry / f"collision_{index:03d}.stl"
            mesh.export(converted)
            name = f"object_collision_{index:03d}"
            spec.add_mesh(name=name, file=str(converted))
            body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                group=3,
                mass=mass / len(collision_sources),
                friction=friction,
            )
            self.collision_hashes.append(dict(path=str(path), sha256=file_hash(path)))
        if not collision_sources:
            # Diagnostic collision hull only; this never authorizes dynamic evaluation.
            body.add_geom(
                name="object_diagnostic_hull",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="phase2_visual",
                group=3,
                mass=0,
            )
        eye = vector(camera["position_world_m"], 3, "camera.position")
        center = vector(camera["look_at_world_m"], 3, "camera.look_at")
        z = eye - center
        if np.linalg.norm(z) < 1e-8:
            raise ValueError("Camera position and look_at coincide")
        z /= np.linalg.norm(z)
        up = vector(camera.get("up_world", [0, 0, 1]), 3, "camera.up")
        x = np.cross(up, z)
        if np.linalg.norm(x) < 1e-8:
            raise ValueError("Camera up is parallel to viewing direction")
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rotation = np.column_stack((x, y, z))
        self.width, self.height = int(camera["width"]), int(camera["height"])
        if self.width != camera["width"] or self.height != camera["height"] or min(self.width, self.height) < 2:
            raise ValueError("Camera dimensions must be integers >= 2")
        fovy = positive(camera["fovy_deg"], "camera.fovy")
        if fovy >= 179:
            raise ValueError("Camera fovy must be < 179 degrees")
        self.near, self.far = positive(camera["near_m"], "near"), positive(camera["far_m"], "far")
        if self.near >= self.far:
            raise ValueError("camera.near_m must be below far_m")
        self.camera_name = camera["name"]
        spec.worldbody.add_camera(
            name=self.camera_name, pos=eye, quat=Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]], fovy=fovy
        )
        profile = apply_contact_profile(spec, config["scene"].get("contact_profile", "source"), friction)
        self.sim = DualArmSimulation(self.robot, scene_spec=spec, object_bodies=("object",))
        self.sim.model.vis.global_.offwidth = max(640, self.width)
        self.sim.model.vis.global_.offheight = max(480, self.height)
        self.sim.model.vis.map.znear = self.near / self.sim.model.stat.extent
        self.sim.model.vis.map.zfar = self.far / self.sim.model.stat.extent
        self.sim.reset(initial_configuration(self.robot, config["scene"]))
        self.initial_q = self.sim.configuration()
        self.metadata = dict(
            mesh_path=str(source),
            mesh_sha256=file_hash(source),
            scale=scale.tolist(),
            collision_sources=self.collision_hashes,
            physics=physics,
            physics_available=self.physics_available,
            inertia_assumption="equal mass per collision piece, uniform-density convex volume",
            collision_assumption="MuJoCo convex hull per explicitly listed mesh",
            robot_snapshot=str(self.robot.directory),
            robot_manifest_sha256=file_hash(self.robot.directory / "manifest.json"),
            compiled_object_mass_kg=float(self.sim.model.body("object").mass[0]),
            compiled_object_inertia_kg_m2=self.sim.model.body("object").inertia.tolist(),
        )
        if table is not None:
            self.metadata["table_footprint"] = table
        if profile is not None:
            self.metadata["contact_profile"] = profile

    def settle(self) -> None:
        """Advance the actual object before perception; reject unstable or moved robot states."""
        placement = self.config["object"].get("placement", "explicit_pose")
        if placement == "explicit_pose":
            return
        if placement != "settle_on_table" or not self.physics_available:
            raise ValueError("settle_on_table requires declared free-object physics")
        stable = 0
        for _ in range(2500):
            if self.config.get("execution", {}).get("arm_bias_compensation", False):
                apply_arm_bias_compensation(self.sim)
            self.sim.step()
            if np.any(self.sim.data.warning.number):
                raise RuntimeError("MuJoCo warning during settling")
            stable = stable + 1 if np.linalg.norm(self.sim.data.joint("object_free").qvel) < 1e-3 else 0
            if stable >= 100:
                self.robot.configuration(self.sim.configuration())
                return
        raise ValueError("Object did not settle within 5 s; check pose/table/physics")

    def cloud(self) -> tuple[np.ndarray, dict]:
        """Return a deterministic world-frame cloud from actual mesh or visible depth."""
        pc = self.config["point_cloud"]
        count, seed = int(pc["num_points"]), int(pc["seed"])
        if count < 3 or count != pc["num_points"] or seed < 0 or seed != pc["seed"]:
            raise ValueError("Point count >= 3 and nonnegative integer seed required")
        if pc.get("object_only") is not True:
            raise ValueError("Phase 2 requires object_only=true")
        pose = body_pose(self.sim)
        if pc["source"] == "complete_mesh":
            points, _ = trimesh.sample.sample_surface(self.mesh, count, seed=seed)
            points = points @ pose[:3, :3].T + pose[:3, 3]
            info = dict(input_kind="oracle complete mesh; not a depth observation")
        elif pc["source"] == "simulated_depth":
            points, info = self.depth_cloud()
        else:
            raise ValueError("point_cloud.source must be complete_mesh or simulated_depth")
        info.update(self.metadata)
        info.update(
            source=pc["source"],
            seed=seed,
            requested_count=count,
            point_count=len(points),
            frame_id="world",
            unit="m",
            T_world_object=pose.tolist(),
            camera=self.config["camera"],
        )
        np.savez_compressed(self.output / "object_cloud.npz", points_world_m=points, T_world_object=pose)
        return points, info

    def depth_cloud(self) -> tuple[np.ndarray, dict]:
        """Unproject MuJoCo Renderer metric axial depth using object geom segmentation.

        Renderer performs depth-buffer linearization internally in MuJoCo 3.6.
        Its camera uses +X right, +Y up, -Z forward. Pixel centers are (u+.5,v+.5).
        """
        sim = self.sim
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0  # Hide convex collision hulls, retain actual visible surface.
        with mujoco.Renderer(sim.model, height=self.height, width=self.width) as renderer:
            renderer.update_scene(sim.data, camera=self.camera_name, scene_option=option)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render().copy()
        object_id = sim.model.geom("object_visual").id
        mask = (
            (segmentation[:, :, 0] == object_id)
            & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            & np.isfinite(depth)
            & (depth > self.near)
            & (depth < self.far)
        )
        v, u = np.nonzero(mask)
        if len(u) < 3:
            raise ValueError("simulated_depth: fewer than 3 visible object pixels; check FOV, occlusion and clipping")
        z = depth[v, u]
        f = 0.5 * self.height / np.tan(np.deg2rad(sim.model.camera(self.camera_name).fovy[0]) / 2)
        local = np.column_stack(((u + 0.5 - self.width / 2) * z / f, -(v + 0.5 - self.height / 2) * z / f, -z))
        camera = sim.data.camera(self.camera_name)
        rotation = camera.xmat.reshape(3, 3)
        points = local @ rotation.T + camera.xpos
        recovered = (points - camera.xpos) @ rotation
        projected = np.column_stack(
            (
                f * recovered[:, 0] / -recovered[:, 2] + self.width / 2 - 0.5,
                -f * recovered[:, 1] / -recovered[:, 2] + self.height / 2 - 0.5,
            )
        )
        error = float(np.max(np.abs(projected - np.column_stack((u, v)))))
        np.savez_compressed(
            self.output / "depth.npz",
            depth_m=depth,
            mask=mask,
            rgb=rgb,
            segmentation=segmentation,
            camera_rotation=rotation,
            camera_position=camera.xpos,
        )
        from imageio.v3 import imwrite

        imwrite(self.output / "camera.png", rgb)
        return points, dict(
            input_kind="partial visible metric depth; complete-training domain shift",
            visible_pixel_count=len(u),
            masking_coverage=float(mask.mean()),
            focal_pixels=f,
            reprojection_max_pixels=error,
            depth_convention="metric axial -cameraZ (Renderer API)",
        )
