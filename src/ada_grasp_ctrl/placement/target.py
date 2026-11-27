"""Simulation-only support geometry and immutable nominal placement targets."""

from copy import deepcopy
import hashlib
import json

import mujoco
import numpy as np

from ada_grasp_ctrl.robot_environment.kinematics import rigid
from .session import rotation
from .face_coverage import contact_patch


def digest(value: dict) -> str:
    """Hash the canonical finite JSON representation of a target."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def collision_support(scene: object) -> np.ndarray:
    """Read compiled convex-mesh vertices in the settled world without changing physics.

    Linear support extrema of each convex hull equal those of its mesh vertices.
    MuJoCo's compiled mesh centering/scaling is included through geom_xmat/xpos.
    """
    model, data = scene.sim.model, scene.sim.data
    vertices = []
    for geom in range(model.ngeom):
        if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith("object_collision_"):
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Placement support requires the existing object collision meshes")
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices.append(
            model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        )
    if not vertices:
        raise ValueError("No compiled object collision support geometry")
    return np.concatenate(vertices)


class TargetGeometry:
    """Resolve centroid height from collision support and reject off-table footprints."""

    def __init__(self, builder: object, scene: object, session: object) -> None:
        self.center = session.center.copy()
        self.support = collision_support(scene)
        body = scene.sim.data.body("object")
        visual = scene.mesh.vertices @ body.xmat.reshape(3, 3).T + body.xpos
        self.footprint = np.concatenate([self.support, visual])
        self.plane = np.r_[builder.plane["normal_xyz"], builder.plane["d"]]
        self.table_center = builder.center.copy()
        self.table_rotation = builder.rotation.copy()
        self.half_extents = builder.half_extents.copy()
        self.source = dict(
            kind="compiled_mujoco_collision_mesh_vertices",
            collision_sources=deepcopy(scene.metadata["collision_sources"]),
            support_sha256=hashlib.sha256(self.support.tobytes()).hexdigest(),
            footprint="union of collision vertices and full visual vertices",
        )
        self.table = deepcopy(builder.table)
        # Nominal contact endpoint, unlike the 1 mm initialization drop clearance.
        self.clearance = 0.0

    def marker(self, xy: object) -> np.ndarray:
        """Clamp a marker to the existing finite tilted tabletop, returning world XYZ."""
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Destination XY must contain two finite metres")
        n, d = self.plane[:3], self.plane[3]
        point = np.r_[xy, (-d - n[:2] @ xy) / n[2]]
        local = (point - self.table_center) @ self.table_rotation
        local[:2] = np.clip(local[:2], -self.half_extents, self.half_extents)
        local[2] = 0
        return self.table_center + self.table_rotation @ local

    def resolve(self, session: object, xy: object, revision: int) -> dict:
        """Freeze a complete target; reject invalid footprints without moving the goal."""
        xy = np.asarray(xy, dtype=float)
        marker = self.marker(xy)
        if not np.allclose(marker[:2], xy, atol=1e-9, rtol=0):
            raise ValueError("Destination marker lies outside the tabletop")
        R = rotation(session.selected.relative_rotation)
        n, d = self.plane[:3], self.plane[3]
        offsets = (self.support - self.center) @ R.T
        z = (self.clearance - d - n[:2] @ xy - np.min(offsets @ n)) / n[2]
        target_center = np.r_[xy, z]
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = R, target_center - R @ self.center
        rigid(delta)
        footprint = (self.footprint - self.center) @ R.T + target_center
        local = (footprint - self.table_center) @ self.table_rotation
        if np.any(np.abs(local[:, :2]) > self.half_extents + 1e-6):
            raise ValueError("Target footprint extends beyond the tabletop; move the marker or change orientation")
        value = dict(
            schema_version=1,
            kind="simulation_placement_target",
            preview_only=True,
            frame_id="world",
            unit="m",
            rotation_convention="active column vectors; row clouds multiply R.T",
            target_revision=revision,
            candidate_id=session.selected_id,
            source=session.selected.source,
            source_snapshot_id=session.snapshot_id,
            source_sha256=session.metadata["snapshot_sha256"],
            relative_rotation=R.tolist(),
            original_centroid_world_m=self.center.tolist(),
            requested_xy_m=xy.tolist(),
            realized_xy_m=xy.tolist(),
            target_centroid_world_m=target_center.tolist(),
            marker_world_m=marker.tolist(),
            Delta_T=delta.tolist(),
            table_plane_world=self.plane.tolist(),
            table=self.table,
            table_sha256=digest(dict(plane=self.plane.tolist(), table=self.table)),
            clearance_m=self.clearance,
            clearance_policy="nominal collision support contact; no destination settling",
            support_geometry=self.source,
            contact_patch_world_m=contact_patch(self.support @ R.T + delta[:3, 3], self.plane),
            stability_validated=False,
        )
        return {**value, "target_sha256": digest(value)}


def transform_points(points: np.ndarray, target: dict) -> np.ndarray:
    """Apply the confirmed full rigid transform to the immutable observation."""
    delta = rigid(target["Delta_T"])
    return points @ delta[:3, :3].T + delta[:3, 3]
