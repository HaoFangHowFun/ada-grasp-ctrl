"""Small geometric derivation and diagnostics for the existing Sharpa assets."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


class HandGeometry:
    """Load real hand FK and independently sampled visual surfaces in flange coordinates."""

    def __init__(self, path: Path, side: str) -> None:
        self.path, self.side = path.resolve(), side
        self.tree = ET.parse(path).getroot()
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.names = [j.get("name") for j in self.tree.findall("joint") if j.get("type") == "revolute"]
        if len(self.names) != 22 or self.model.nq != 22:
            raise ValueError("Expected a standalone hand with 22 revolute joints")
        self.jids = np.array([self.model.joint(n).id for n in self.names])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.limits = self.model.jnt_range[self.jids].copy()
        self.links = sorted(link.get("name").removeprefix(side + "_") for link in self.tree.findall("link"))
        self.bids = np.array([self.model.body(side + "_" + n).id for n in self.links])
        self.parents = {j.find("child").get("link"): j.find("parent").get("link") for j in self.tree.findall("joint")}
        self.joint_children = [self.tree.find(f"joint[@name='{n}']/child").get("link") for n in self.names]
        self.dependencies = {self.path}
        self.surfaces = []
        self.mesh_hashes = set()
        # Each hand has its own deterministic samples; no shared vertex correspondence.
        for i, link in enumerate(self.tree.findall("link")):
            for k, visual in enumerate(link.findall("visual")):
                element = visual.find("geometry/mesh")
                if element is None:
                    raise ValueError("Expected mesh visual geometry")
                mesh_path = (path.parent / element.get("filename")).resolve(strict=True)
                self.dependencies.add(mesh_path)
                self.mesh_hashes.add(hashlib.sha256(mesh_path.read_bytes()).hexdigest())
                mesh = trimesh.load_mesh(mesh_path, process=False)
                mesh.apply_scale(np.fromstring(element.get("scale", "1 1 1"), sep=" "))
                points, _ = trimesh.sample.sample_surface(mesh, 240, seed=1000 * (1 + (side == "left")) + i * 10 + k)
                origin = visual.find("origin")
                if origin is not None:
                    rotation = Rotation.from_euler(
                        "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                    ).as_matrix()
                    points = points @ rotation.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                self.surfaces.append((self.model.body(link.get("name")).id, points))

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return corresponding link origins without applying any mirror mapping."""
        self.data.qpos[self.qids] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.bids].copy()

    def cloud(self, q: np.ndarray) -> np.ndarray:
        """Return visual surface samples from this hand's original assets."""
        self.fk(q)
        return np.concatenate([p @ self.data.xmat[b].reshape(3, 3).T + self.data.xpos[b] for b, p in self.surfaces])

    def ancestors(self, child: str) -> list[str]:
        """Return the kinematic ancestry, used to constrain joint assignment."""
        result = []
        while child in self.parents:
            child = self.parents[child]
            result.append(child)
        return result

    def provenance(self) -> dict:
        """Hash the exact URDF and every visual dependency."""
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.dependencies)}


def derive_mapping(right: HandGeometry, left: HandGeometry) -> tuple[dict, dict]:
    """Fit a fixed signed-axis reflection and match joints by independent FK motion.

    Joint names only label the exported order. Geometry establishes reflection;
    all candidate joints/signs are scored and topology is checked afterwards.
    Zero offsets remain zero; small source rounding errors are reported.
    """
    if right.links != left.links:
        raise ValueError("Anatomical landmark inventory differs")
    zero = np.zeros(22)
    pr, pl = right.fk(zero), left.fk(zero)
    candidates = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            reflection = np.eye(3)[list(perm)] * np.array(signs)[:, None]
            if np.linalg.det(reflection) < 0:
                candidates.append((float(np.linalg.norm(pr @ reflection.T - pl)), reflection))
    candidates.sort(key=lambda item: item[0])
    reflection = candidates[0][1]
    costs = np.full((22, 22, 2), np.inf)
    probes = []
    for r in range(22):
        deltas = [d for d in [-0.25, -0.1, 0.1, 0.25] if right.limits[r, 0] <= d <= right.limits[r, 1]]
        probes.append(deltas)
        references = []
        for d in deltas:
            q = zero.copy()
            q[r] = d
            references.append(right.fk(q) @ reflection.T)
        for left_index in range(22):
            if len(right.ancestors(right.joint_children[r])) != len(left.ancestors(left.joint_children[left_index])):
                continue
            for si, sign in enumerate([-1, 1]):
                if any(not left.limits[left_index, 0] <= sign * d <= left.limits[left_index, 1] for d in deltas):
                    continue
                errors = []
                for d, reference in zip(deltas, references):
                    q = zero.copy()
                    q[left_index] = sign * d
                    errors.append(np.mean((left.fk(q) - reference) ** 2))
                costs[r, left_index, si] = np.mean(errors)
    rows, cols = linear_sum_assignment(costs.min(axis=2))
    permutation, signs = np.empty(22, dtype=int), np.empty(22, dtype=int)
    evidence = []
    for r, left_index in zip(rows, cols):
        si = int(np.argmin(costs[r, left_index]))
        permutation[left_index], signs[left_index] = r, [-1, 1][si]
        alternatives = np.delete(costs[r].reshape(-1), left_index * 2 + si)
        evidence.append(
            {
                "right": right.names[r],
                "left": left.names[left_index],
                "sign": int(signs[left_index]),
                "probe_rms_mm": float(np.sqrt(costs[r, left_index, si]) * 1000),
                "next_best_rms_mm": float(np.sqrt(np.min(alternatives)) * 1000),
                "deltas_rad": probes[r],
                "asymmetric": not any(d < 0 for d in probes[r]),
            }
        )
    # Matched joint ancestry must form the same tree, not just a cheap assignment.
    for left_index, r in enumerate(permutation):
        ra = {i for i, child in enumerate(right.joint_children) if child in right.ancestors(right.joint_children[r])}
        la = {
            int(permutation[i])
            for i, child in enumerate(left.joint_children)
            if child in left.ancestors(left.joint_children[left_index])
        }
        if ra != la:
            raise ValueError("Joint assignment does not preserve kinematic topology")
    mapped_limits = np.sort(right.limits[permutation] * signs[:, None], axis=1)
    if np.any(mapped_limits < left.limits[:, :1] - 1e-8) or np.any(mapped_limits > left.limits[:, 1:] + 1e-8):
        raise ValueError("Mapping does not preserve full joint ranges")
    config = {
        "schema_version": 1,
        "units": "radians/meters",
        "right_joint_names": right.names,
        "left_joint_names": left.names,
        "permutation": permutation.tolist(),
        "sign": signs.tolist(),
        "offset": [0.0] * 22,
        "local_reflection": reflection.tolist(),
        "right_limits": right.limits.tolist(),
        "left_limits": left.limits.tolist(),
        "pose_frame": "hand_flange or hand_wrist; homogeneous world-from-frame",
        "flange_to_wrist_xyz": [0, 0, 0.0005],
        "wrist_to_palm_xyz": [0, 0, 0.029],
        "tcp": "No arbitrary TCP or arm mount included; convert to flange before mirroring",
        "assets": {"right": right.provenance(), "left": left.provenance()},
    }
    report = {
        "canonical_max_landmark_error_mm": float(np.linalg.norm(pr @ reflection.T - pl, axis=1).max() * 1000),
        "reflection_runner_up_norm_mm": candidates[1][0] * 1000,
        "joints": evidence,
        "shared_visual_mesh_hashes": sorted(right.mesh_hashes & left.mesh_hashes),
        "surface_metric": (
            "Independent fixed visual surface samples, symmetric nearest-sample distances; "
            "sampling floor, not exact mesh distance"
        ),
    }
    return config, report


def diagnose(right: HandGeometry, left: HandGeometry, mirror: object, q: np.ndarray, case_id: str) -> dict:
    """Compare independent reflected-right and actual-left FK and surface samples."""
    ql = q[mirror.permutation] * mirror.sign + mirror.offset
    violations = [left.names[i] for i in range(22) if not left.limits[i, 0] - 1e-8 <= ql[i] <= left.limits[i, 1] + 1e-8]
    right_violations = [
        right.names[i] for i in range(22) if not right.limits[i, 0] - 1e-8 <= q[i] <= right.limits[i, 1] + 1e-8
    ]
    ref = right.fk(q) @ mirror.local_reflection.T
    actual = left.fk(ql)
    errors = np.linalg.norm(ref - actual, axis=1) * 1000
    a, b = right.cloud(q) @ mirror.local_reflection.T, left.cloud(ql)
    ab, ba = cKDTree(b).query(a)[0] * 1000, cKDTree(a).query(b)[0] * 1000
    distances = np.r_[ab, ba]
    return {
        "case_id": case_id,
        "valid": not (violations or right_violations),
        "q_right": q.tolist(),
        "q_left": ql.tolist(),
        "wrist_position_error_mm": float(errors[right.links.index("hand_wrist")]),
        "palm_error_mm": float(errors[right.links.index("hand_C_MC")]),
        "wrist_orientation_error_deg": float(
            np.rad2deg(
                Rotation.from_matrix(
                    left.data.xmat[left.model.body("left_hand_wrist").id].reshape(3, 3).T
                    @ mirror.local_reflection
                    @ right.data.xmat[right.model.body("right_hand_wrist").id].reshape(3, 3)
                    @ mirror.local_reflection.T
                ).magnitude()
            )
        ),
        "fingertip_errors_mm": {
            f: float(errors[right.links.index(f + "_fingertip")]) for f in ("thumb", "index", "middle", "ring", "pinky")
        },
        "max_landmark_error_mm": float(errors.max()),
        "mean_surface_error_mm": float(distances.mean()),
        "max_surface_error_mm": float(distances.max()),
        "p95_surface_error_mm": float(np.percentile(distances, 95)),
        "symmetric_chamfer_mm": float((ab.mean() + ba.mean()) / 2),
        "joint_limit_violations": violations,
        "right_joint_limit_violations": right_violations,
    }


def render_case(right: HandGeometry, left: HandGeometry, mirror: object, case: dict, output: Path, mode: str) -> None:
    """Render geometrically reflected surface points in MuJoCo, without improper quaternions."""
    import imageio.v3 as iio

    q, ql = np.array(case["q_right"]), np.array(case["q_left"])
    clouds = {
        "right": (right.cloud(q), [0.6, 0.65, 0.85, 1]),
        "reference": (right.cloud(q) @ mirror.local_reflection.T, [0, 0.9, 0.9, 0.55]),
        "left": (left.cloud(ql), [1, 0.45, 0.08, 0.8]),
    }
    model = mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="800" offheight="800"/></visual></mujoco>')
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.12]
    camera.distance, camera.azimuth, camera.elevation = 0.43, 160, -15
    with mujoco.Renderer(model, 800, 800, max_geom=20000) as renderer:
        renderer.update_scene(data, camera=camera)
        scene = renderer.scene
        for key in ["reference", "left"] if mode == "overlay" else [mode]:
            points, color = clouds[key]
            for p in points[::2]:
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.0007] * 3),
                    p,
                    np.eye(3).reshape(-1),
                    np.array(color, dtype=np.float32),
                )
                scene.ngeom += 1
        for key, points in [("reference", right.fk(q) @ mirror.local_reflection.T), ("left", left.fk(ql))]:
            if mode not in (key, "overlay"):
                continue
            for i, name in enumerate(right.links):
                if name.endswith("fingertip") or name in ("hand_wrist", "hand_C_MC"):
                    color = [0, 1, 1, 1] if key == "reference" else [1, 0.4, 0, 1]
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.002] * 3),
                        points[i],
                        np.eye(3).reshape(-1),
                        np.array(color, dtype=np.float32),
                    )
                    scene.ngeom += 1
        iio.imwrite(output, renderer.render())
