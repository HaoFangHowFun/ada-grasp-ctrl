"""Cheap, selection-only estimates of hand coverage over the intended support face."""

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ada_grasp_ctrl.robot_environment.model import side_slice


FACE_BAND_M = 0.002
MIN_AREA_M2 = 50e-6
MIN_SPAN_M = 0.010


def plane_axes(normal: np.ndarray) -> np.ndarray:
    """Return two orthonormal columns tangent to a unit plane normal."""
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(axis, normal)
    u /= np.linalg.norm(u)
    return np.column_stack((u, np.cross(normal, u)))


def hull(points: np.ndarray) -> np.ndarray:
    """Return a counterclockwise 2D boundary, or no area for degenerate points."""
    if len(points) < 3:
        return np.empty((0, 2))
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return np.empty((0, 2))


def contact_patch(vertices: np.ndarray, plane: np.ndarray) -> list:
    """Project near-support object vertices to one compact world-space boundary."""
    normal, offset = plane[:3], plane[3]
    distance = vertices @ normal + offset
    points = vertices[distance <= FACE_BAND_M + 1e-9]
    axes = plane_axes(normal)
    origin = -offset * normal
    boundary = hull((points - origin) @ axes)
    return (boundary @ axes.T + origin).tolist()


def intersection(subject: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Clip one convex polygon against another without mesh collision queries."""
    for a, b in zip(boundary, np.roll(boundary, -1, axis=0)):
        if len(subject) < 3:
            return np.empty((0, 2))
        edge = b - a
        inward = np.array([-edge[1], edge[0]])
        distance = (subject - a) @ inward
        output = []
        for i in range(len(subject)):
            previous, current = subject[i - 1], subject[i]
            d0, d1 = distance[i - 1], distance[i]
            inside0, inside1 = d0 >= -1e-12, d1 >= -1e-12
            if inside0 != inside1:
                output.append(previous + (current - previous) * d0 / (d0 - d1))
            if inside1:
                output.append(current)
        subject = np.asarray(output).reshape(-1, 2)
    return subject


def overlap(vertices: np.ndarray, plane: np.ndarray, axes: np.ndarray, patch: np.ndarray) -> tuple[float, float]:
    """Estimate area/span occupied by the low part of one hand collision mesh."""
    low = vertices[vertices @ plane[:3] + plane[3] <= FACE_BAND_M + 1e-9]
    if len(low) < 3:
        return 0.0, 0.0
    projected = low @ axes
    if np.any(projected.max(0) < patch.min(0)) or np.any(projected.min(0) > patch.max(0)):
        return 0.0, 0.0
    polygon = intersection(hull(projected), patch)
    if len(polygon) < 3:
        return 0.0, 0.0
    polygon = polygon - polygon[0]
    following = np.roll(polygon, -1, axis=0)
    area = abs(float(np.sum(polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0]))) / 2
    span = float(np.max(np.linalg.norm(polygon[:, None] - polygon[None, :], axis=2)))
    return area, span


class FaceCoverage:
    """Reuse cached hand meshes and the selector's already-computed Squeeze FK.

    No extra forward dynamics, IK, path search or live-state mutation is needed.
    The convex footprint is an inexpensive heuristic, not a release guarantee.
    """

    def __init__(self, static: object, target: dict) -> None:
        self.static = static
        self.plane = np.asarray(target["table_plane_world"], dtype=float)
        self.axes = plane_axes(self.plane[:3])
        self.patch = hull(np.asarray(target["contact_patch_world_m"]) @ self.axes)
        self.meshes = {side: [] for side in ("left", "right")}
        self.cache = {}
        model = static.model
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            if body not in static.hand_body_ids or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("Placement face coverage requires compiled hand collision meshes")
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            self.meshes[static.body_labels[body]].append((geom, model.mesh_vert[start : start + count].copy()))

    def check(self, side: str, held: np.ndarray, destination: np.ndarray) -> dict:
        """Test the rigidly carried Squeeze hand once, before placement IK seeds."""
        model, data = self.static.model, self.static.static_data
        if not np.array_equal(data.qpos[self.static.qpos], held):
            raise ValueError("Placement face check requires the current Squeeze static snapshot")
        # Hand geometry relative to its flange depends only on finger angles.
        # Different arm IK seeds for the same grasp need no repeat projection.
        key = (side, held[side_slice(side)][6:].tobytes(), destination.tobytes())
        if key in self.cache:
            return self.cache[key]
        wrist = data.body(side + "_hand_flange")
        rotation = destination[:3, :3] @ wrist.xmat.reshape(3, 3).T
        translation = destination[:3, 3] - rotation @ wrist.xpos
        blocked = []
        if len(self.patch) >= 3:
            for geom, vertices in self.meshes[side]:
                R = rotation @ data.geom_xmat[geom].reshape(3, 3)
                t = rotation @ data.geom_xpos[geom] + translation
                area, span = overlap(vertices @ R.T + t, self.plane, self.axes, self.patch)
                if area >= MIN_AREA_M2 and span >= MIN_SPAN_M:
                    blocked.append(dict(geom=model.geom(geom).name, overlap_area_m2=area, overlap_span_m=span))
        result = dict(
            accepted=not blocked,
            blocked_geoms=blocked,
            face_band_m=FACE_BAND_M,
            minimum_overlap_area_m2=MIN_AREA_M2,
            minimum_overlap_span_m=MIN_SPAN_M,
            method="projected_low_hand_mesh_convex_overlap",
        )
        self.cache[key] = result
        return result
