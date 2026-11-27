"""Actual MuJoCo geometry, isolated planning data and directed contact boundaries."""

from dataclasses import dataclass
import time
from types import SimpleNamespace
from typing import Callable

import mujoco
import numpy as np

from ada_grasp_ctrl.experiments.dual_arm_grasp_episode import live_contacts
from ada_grasp_ctrl.grasp_selection.multi_arm import forbidden_contacts
from ada_grasp_ctrl.placement.target import collision_support
from ada_grasp_ctrl.robot_environment.kinematics import rigid
from ada_grasp_ctrl.robot_environment.model import side_slice

# Existing scene initialization support tolerance, and the existing arm gate.
SUPPORT_TOLERANCE = 0.002
ROBOT_TOLERANCE = 1e-5
NUMERICAL_EPS = 1e-9


@dataclass
class Observation:
    """Minimum signed distances per actual geometry pair, plus policy violations."""

    support: dict[tuple[int, int], float]
    hand: dict[tuple[int, int], float]
    forbidden: list[dict]


def contacts(values: dict) -> dict:
    """Retain touching/penetrating pairs; positive signed distance means separated."""
    return {pair: distance for pair, distance in values.items() if distance <= NUMERICAL_EPS}


def separating(previous: dict, current: dict) -> bool:
    """Require only previously touching pairs, without increased penetration."""
    return all(pair in previous and distance >= previous[pair] - NUMERICAL_EPS for pair, distance in current.items())


class PlanningScene:
    """Reuse a compiled physics model with independent MjData; never step live data."""

    def __init__(self, sim: object, side: str, static: object | None = None) -> None:
        self.sim, self.model, self.robot, self.side, self.static = sim, sim.model, sim.robot, side, static
        self.data = mujoco.MjData(self.model)
        self.proxy = SimpleNamespace(
            model=self.model,
            data=self.data,
            robot=self.robot,
            body_labels=sim.body_labels,
            hand_body_ids=sim.hand_body_ids,
        )
        self.object_joint = int(self.model.joint("object_free").qposadr[0])
        self.object_body = self.model.body("object").id
        self.wrist_body = self.model.body(side + "_hand_flange").id
        self.arm = side_slice(side, True)
        self.queries = 0
        self.fixed = sim.configuration()
        self.pose = self.body_pose(sim.data, self.object_body)
        self.attachment = None
        self.mode = "approach"
        self.allowed_pairs: set[tuple[int, int]] = set()
        self.measured_hand = None
        self.table_plane = None
        self.table_clearance = 0.0
        labels = [sim.body_labels[int(b)] for b in self.model.geom_bodyid]
        active = [g for g in range(self.model.ngeom) if self.model.geom_contype[g] or self.model.geom_conaffinity[g]]
        self.clearance_geoms = [g for g in active if labels[g] in ("object", "left", "right")]
        self.collision_meshes = {}
        for geom in self.clearance_geoms:
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[geom]
                start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                self.collision_meshes[geom] = self.model.mesh_vert[start : start + count]
        self.table_pairs = [
            tuple(sorted((a, b))) for a in active if labels[a] == "object" for b in active if labels[b] == "table"
        ]
        self.support_geometry = {}
        for pair in self.table_pairs:
            obj = next(g for g in pair if labels[g] == "object")
            table = next(g for g in pair if labels[g] == "table")
            if (
                self.model.geom_type[obj] == mujoco.mjtGeom.mjGEOM_MESH
                and self.model.geom_type[table] == mujoco.mjtGeom.mjGEOM_BOX
            ):
                mesh = self.model.geom_dataid[obj]
                start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                self.support_geometry[pair] = (obj, table, self.model.mesh_vert[start : start + count])
        if not self.table_pairs:
            raise ValueError("Planning requires actual object and table collision geometry")
        self.snapshot()

    @staticmethod
    def body_pose(data: object, body: int) -> np.ndarray:
        """Read a finite proper world body transform."""
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = data.xmat[body].reshape(3, 3), data.xpos[body]
        return rigid(pose)

    def snapshot(self) -> None:
        """Copy state while the caller's sole physics owner is paused."""
        mujoco.mj_copyData(self.data, self.model, self.sim.data)

    def configure(
        self,
        fixed: np.ndarray,
        mode: str,
        pose: np.ndarray,
        *,
        attachment: np.ndarray | None = None,
        allowed_pairs: set | None = None,
        measured_hand: np.ndarray | None = None,
        table_plane: np.ndarray | None = None,
        table_clearance: float = 0.0,
    ) -> None:
        """Freeze all nonsearched joints and the static pose or measured attachment."""
        if mode not in ("approach", "pregrasp", "grasp", "squeeze", "carry", "settle", "release", "retreat"):
            raise ValueError("Unknown planning collision mode")
        self.fixed = self.robot.configuration(fixed)
        self.mode, self.pose = mode, rigid(pose)
        self.attachment = None if attachment is None else rigid(attachment)
        if (mode == "carry") != (attachment is not None):
            raise ValueError("Carry requires measured attachment; other phases require static object")
        self.allowed_pairs = set(allowed_pairs or ())
        self.measured_hand = None if measured_hand is None else self.robot.configuration(measured_hand)
        if not np.isfinite(table_clearance) or table_clearance < 0:
            raise ValueError("Invalid planning table clearance")
        if table_clearance and (mode != "carry" or table_plane is None):
            raise ValueError("Raised table clearance requires a carry phase and calibrated plane")
        self.table_plane = None if table_plane is None else self.valid_plane(table_plane)
        self.table_clearance = table_clearance

    @staticmethod
    def valid_plane(plane: np.ndarray) -> np.ndarray:
        """Validate the calibrated signed-distance convention without modifying it."""
        plane = np.asarray(plane, dtype=float)
        if plane.shape != (4,) or not np.isfinite(plane).all() or not np.isclose(np.linalg.norm(plane[:3]), 1):
            raise ValueError("Expected a finite unit-normal table plane")
        return plane.copy()

    def clearance(self, data: object, plane: np.ndarray) -> float:
        """Measure the lowest actual collision vertex against the calibrated plane."""
        plane = self.valid_plane(plane)
        vertices = collision_support(SimpleNamespace(sim=SimpleNamespace(model=self.model, data=data)))
        return float(np.min(vertices @ plane[:3] + plane[3]))

    def clearance_pose(
        self, data: object, plane: np.ndarray, height: float, *, pose: np.ndarray | None = None
    ) -> np.ndarray:
        """Raise the measured or requested object pose to a collision-mesh clearance."""
        distance = self.clearance(data, plane)
        actual = self.body_pose(data, self.object_body)
        if pose is None:
            pose = actual
        else:
            pose = rigid(pose)
            vertices = collision_support(SimpleNamespace(sim=SimpleNamespace(model=self.model, data=data)))
            vertices = (vertices - actual[:3, 3]) @ actual[:3, :3] @ pose[:3, :3].T + pose[:3, 3]
            distance = float(np.min(vertices @ np.asarray(plane)[:3] + plane[3]))
        pose[:3, 3] += max(0.0, height - distance) * np.asarray(plane)[:3]
        return pose

    def table_clearance_violations(self, data: object) -> list[dict]:
        """Check exact mesh support against a virtual plane in planning data only.

        The physical table and live data are never moved. The 2 mm physical
        support tolerance is not subtracted from this positive planning margin.
        Unsupported collision geometry fails closed instead of being skipped.
        """
        if not self.table_clearance:
            return []
        normal, offset = self.table_plane[:3], self.table_plane[3]
        bad = []
        for geom in self.clearance_geoms:
            if geom not in self.collision_meshes:
                raise ValueError(
                    f"Raised table clearance requires compiled collision mesh: {self.model.geom(geom).name}"
                )
            local_normal = data.geom_xmat[geom].reshape(3, 3).T @ normal
            distance = float(
                np.min(self.collision_meshes[geom] @ local_normal) + data.geom_xpos[geom] @ normal + offset
            )
            if distance < self.table_clearance - NUMERICAL_EPS:
                bad.append(
                    dict(
                        category="planning_table_clearance",
                        geom=self.model.geom(geom).name,
                        body=self.model.body(int(self.model.geom_bodyid[geom])).name,
                        distance_m=distance,
                        required_clearance_m=self.table_clearance,
                    )
                )
        return bad

    def full(self, arm: np.ndarray) -> np.ndarray:
        """Insert only the active six arm joints into the frozen 56-vector."""
        q = self.fixed.copy()
        q[self.arm] = arm
        return q

    def support_distance(self, data: object, pair: tuple[int, int]) -> float:
        """Disambiguate zero mesh/box distance using a provable separating plane.

        MuJoCo 3.6 can return zero for disjoint mesh/box pairs with no contacts.
        A positive mesh-to-top-plane gap bounds distance to the entire table box
        from below. Only this geometric proof can override an ambiguous zero;
        negative penetration distances and nonseparated states remain unchanged.
        """
        distance = float(mujoco.mj_geomDistance(self.model, data, *pair, 0.01, None))
        if distance == 0.0 and pair in self.support_geometry:
            obj, table, vertices = self.support_geometry[pair]
            normal = data.geom_xmat[table].reshape(3, 3)[:, 2]
            top = data.geom_xpos[table] + normal * self.model.geom_size[table, 2]
            local_normal = data.geom_xmat[obj].reshape(3, 3).T @ normal
            lower_bound = float(np.min(vertices @ local_normal) + (data.geom_xpos[obj] - top) @ normal)
            distance = max(distance, lower_bound)
        return distance

    def observe(self, data: object) -> Observation:
        """Apply legacy robot gates and explicit object policy to measured geometry."""
        proxy = self.proxy if data is self.data else self.sim
        rows = live_contacts(proxy)
        raw_robot = []
        support, hand, bad = {}, {}, []
        for contact in data.contact:
            pair = tuple(sorted(int(g) for g in contact.geom))
            labels = {self.sim.body_labels[int(self.model.geom_bodyid[g])] for g in pair}
            distance = float(contact.dist)
            if labels & {"left", "right"}:
                raw_robot.append(pair)
            elif labels == {"object", "table"} and distance <= NUMERICAL_EPS:
                geoms = [int(g) for g in contact.geom]
                table_index = next(
                    i for i, g in enumerate(geoms) if self.sim.body_labels[int(self.model.geom_bodyid[g])] == "table"
                )
                upward = data.geom_xmat[geoms[table_index]].reshape(3, 3)[:, 2]
                outward = np.asarray(contact.frame[:3]) * (1 if table_index == 0 else -1)
                if float(upward @ outward) < 1 - ROBOT_TOLERANCE:
                    bad.append(dict(category="object_table_non_support", pair=pair, distance_m=distance))
            elif "object" in labels and "table" not in labels and labels != {"object"}:
                if distance < -ROBOT_TOLERANCE:
                    bad.append(dict(category="object_environment", pair=pair, distance_m=distance))
        # Query table signed distances even at zero contact and at separation;
        # contact-list disappearance alone is not used as a penetration measure.
        for pair in self.table_pairs:
            distance = self.support_distance(data, pair)
            support[pair] = distance
            if distance < -SUPPORT_TOLERANCE:
                bad.append(dict(category="object_table", pair=pair, distance_m=distance))
        for row, pair in zip(rows, raw_robot):
            if row["category"] != "robot_object":
                bad.extend(forbidden_contacts([row], self.side, phase="squeeze"))
                continue
            intended = bool(
                row["sides"] == [self.side]
                and row["hand_bodies"]
                and not forbidden_contacts([{**row, "penetration_m": 1.0}], self.side, phase="squeeze")
            )
            if intended:
                hand[pair] = min(hand.get(pair, float("inf")), row["distance_m"])
            if self.mode in ("pregrasp", "grasp", "squeeze", "release"):
                # Use the same phase policy as candidate selection and live execution.
                # Release retains the legacy Squeeze hand/object policy, including
                # new intended contact pairs while the fingers open.
                # The arm-only approach retains its separate strict object gate.
                phase = "squeeze" if self.mode == "release" else self.mode
                bad.extend(forbidden_contacts([row], self.side, phase=phase))
                continue
            allowed = intended and self.mode in ("carry", "settle", "retreat") and pair in self.allowed_pairs
            if not allowed and row["penetration_m"] > ROBOT_TOLERANCE:
                bad.append({**row, "pair": pair})
        return Observation(support, hand, bad)

    def query(self, q: np.ndarray) -> Observation:
        """Check commanded and measured held hand geometry at every arm sample."""
        result = self._query(q)
        if self.measured_hand is not None:
            measured = np.array(q, copy=True)
            measured[side_slice(self.side)][6:] = self.measured_hand[side_slice(self.side)][6:]
            if not np.array_equal(measured, q):
                other = self._query(measured)
                result.forbidden.extend(other.forbidden)
                for pair, distance in other.hand.items():
                    result.hand[pair] = min(result.hand.get(pair, float("inf")), distance)
        return result

    def _query(self, q: np.ndarray) -> Observation:
        """Run full mj_forward on isolated buffers with actual collision assets."""
        q = self.robot.configuration(q)
        self.queries += 1
        self.data.qpos[self.sim.qpos] = q
        self.data.qvel[:] = 0
        pose = self.pose
        if self.attachment is not None:
            # MuJoCo wrist FK ensures the attachment and object use identical frames.
            mujoco.mj_forward(self.model, self.data)
            pose = self.body_pose(self.data, self.wrist_body) @ self.attachment
        adr = self.object_joint
        self.data.qpos[adr : adr + 3] = pose[:3, 3]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(pose[:3, :3]).ravel())
        self.data.qpos[adr + 3 : adr + 7] = quat
        mujoco.mj_forward(self.model, self.data)
        result = self.observe(self.data)
        result.forbidden.extend(self.table_clearance_violations(self.data))
        if self.static is not None:
            result.forbidden.extend(
                forbidden_contacts(self.static.check_static_contacts(q), self.side, phase="squeeze")
            )
        return result


class EdgePolicy:
    """Sample every edge with bounded table support and directed retreat separation.

    Object/table support may persist or recur at any sample within the scene's
    2 mm penetration cap. Release uses the legacy active-hand contact policy;
    retreat must separate monotonically within its initial connector radius.
    """

    def __init__(
        self,
        scene: PlanningScene,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        resolution: float = 0.02,
        connector_radius: float = 0.20,
        deadline: float = float("inf"),
        canceled: Callable[[], bool] = lambda: False,
        full: bool = False,
    ) -> None:
        if resolution <= 0 or connector_radius <= 0 or not np.isfinite([resolution, connector_radius]).all():
            raise ValueError("Invalid edge sampling settings")
        self.scene, self.start, self.goal = scene, np.array(start), np.array(goal)
        self.resolution, self.radius, self.deadline = resolution, connector_radius, deadline
        self.canceled, self.full = canceled, full
        self.last_failure = None
        self.hand_pairs: set[tuple[int, int]] = set()

    def guard(self) -> None:
        if self.canceled():
            raise InterruptedError("Planning canceled")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Planning deadline")

    def reject(self, reason: str) -> bool:
        """Retain the reason for an endpoint-connector rejection."""
        self.last_failure = reason
        return False

    def __call__(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Validate every sample in forward execution order, including endpoints."""
        self.guard()
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / self.resolution)))
        qs = [a + (b - a) * (i / count) for i in range(count + 1)]
        qs[0], qs[-1] = a, b
        observations = []
        for q in qs:
            self.guard()
            try:
                observation = self.scene.query(q if self.full else self.scene.full(q))
            except ValueError as error:
                self.last_failure = str(error)
                return False
            self.guard()
            if observation.forbidden:
                self.last_failure = observation.forbidden
                return False
            self.hand_pairs.update(contacts(observation.hand))
            observations.append(observation)
        mode = self.scene.mode
        initial = np.array_equal(a, self.start)
        if mode == "retreat":
            values = [contacts(o.hand) for o in observations]
            if np.array_equal(a, b):
                return not values[0] or (initial and not np.array_equal(self.start, self.goal))
            if values[0] and not initial:
                return self.reject("Hand/object contact outside initial separation connector")
            previous = values[0]
            for i, current in enumerate(values[1:], 1):
                if not separating(previous, current):
                    return self.reject("Hand/object penetration increased or recontacted")
                if current and np.linalg.norm(qs[i] - a) > self.radius + NUMERICAL_EPS:
                    return self.reject("Hand/object contact outside separation radius")
                previous = current
            if values[-1]:
                return self.reject("Retreat connector did not separate hand from object")
        return True
