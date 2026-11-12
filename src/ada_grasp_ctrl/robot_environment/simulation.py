"""Cached MuJoCo stepping and static contact diagnostics for both real robot branches."""

from __future__ import annotations

import mujoco
import numpy as np

from .model import DualArmModel, side_slice


class DualArmSimulation:
    """One compiled scene, physics data and separate reusable static-check data.

    Args:
        robot: Exported canonical model.
        scene_spec: Optional copy of robot.spec with explicitly configured scene additions.
        object_bodies: Exact names of caller-supplied object roots; descendants inherit the label.
        table_bodies: Exact table root names. The canonical upstream table is included by default.

    The passive side retains its command target, but contact can still move it physically.
    """

    def __init__(
        self,
        robot: DualArmModel,
        *,
        scene_spec: mujoco.MjSpec | None = None,
        object_bodies: tuple[str, ...] = (),
        table_bodies: tuple[str, ...] = ("calibration_table_surface",),
    ) -> None:
        self.robot = robot
        self.model = robot.mj_model if scene_spec is None else scene_spec.compile()
        self.mapping = robot.mapping_for(self.model)
        self.qpos = np.array([self.mapping[n]["qpos"] for n in robot.joint_names])
        self.qvel = np.array([self.mapping[n]["qvel"] for n in robot.joint_names])
        self.actuators = np.array([self.mapping[n]["actuator"] for n in robot.joint_names])
        self.data = mujoco.MjData(self.model)
        self.static_data = mujoco.MjData(self.model)
        self._targets = robot.home_configuration()
        self.active_side = "left"
        roots = {name: "object" for name in object_bodies}
        if set(object_bodies) & set(table_bodies):
            raise ValueError("Object and table body roots must be disjoint")
        roots.update({name: "table" for name in table_bodies})
        for name in roots:
            if self.model.body(name).id < 0:
                raise ValueError(f"Missing scene body: {name}")
        self.body_labels = {0: "environment"}
        # Match the legacy selector's hand-flange subtree, not a body-name prefix.
        self.hand_body_ids = {self.model.body(f"{side}_hand_flange").id for side in ("left", "right")}
        for i in range(1, self.model.nbody):
            name = self.model.body(i).name
            if int(self.model.body_parentid[i]) in self.hand_body_ids:
                self.hand_body_ids.add(i)
            parent = self.body_labels[int(self.model.body_parentid[i])]
            self.body_labels[i] = roots.get(
                name, "left" if name.startswith("left_") else "right" if name.startswith("right_") else parent
            )
        self.reset(self._targets)

    @property
    def targets(self) -> np.ndarray:
        """Return a copy of canonical targets so callers cannot bypass validation."""
        return self._targets.copy()

    def select_side(self, side: str) -> None:
        """Select which side receives subsequent implicit-side target commands."""
        side_slice(side)
        self.active_side = side

    def set_target(self, target: object, *, side: str | None = None) -> None:
        """Set 28 arm/hand targets while preserving the opposite side's command."""
        block = side_slice(self.active_side if side is None else side)
        value = np.asarray(target, dtype=float)
        if value.shape != (28,) or not np.isfinite(value).all():
            raise ValueError("Expected a finite 28-joint side target")
        candidate = self._targets.copy()
        candidate[block] = value
        self._targets = self.robot.configuration(candidate)
        self.data.ctrl[self.actuators] = self._targets

    def reset(self, q: object) -> None:
        """Reset the whole scene and initialize both robot command targets."""
        value = self.robot.configuration(q)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos] = value
        self._targets = value.copy()
        self.data.ctrl[self.actuators] = value
        mujoco.mj_forward(self.model, self.data)

    def configuration(self) -> np.ndarray:
        """Read the robot subset; physics joint limits are soft and may be exceeded."""
        return self.robot.configuration(self.data.qpos[self.qpos], check_limits=False)

    def step(self, count: int = 1) -> None:
        """Advance headless MuJoCo physics, rejecting invalid step counts or nonfinite state."""
        if not isinstance(count, int) or count < 1:
            raise ValueError("Step count must be a positive integer")
        mujoco.mj_step(self.model, self.data, nstep=count)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo produced nonfinite state")

    def check_static_contacts(self, q_all: object) -> list[dict]:
        """Report static contacts without modifying live physics state or command targets.

        Scene-object poses are copied from live data; robot positions come from q_all.
        No contact is automatically rejected and no trajectory claim is made.
        """
        q = self.robot.configuration(q_all)
        mujoco.mj_copyData(self.static_data, self.model, self.data)
        self.static_data.qpos[self.qpos] = q
        self.static_data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.static_data)
        rows = []
        for contact in self.static_data.contact:
            geoms = [int(g) for g in contact.geom]
            bodies = [int(self.model.geom_bodyid[g]) for g in geoms]
            names = [self.model.body(b).name for b in bodies]
            labels = [self.body_labels[b] for b in bodies]
            robot_sides = set(labels) & {"left", "right"}
            if not robot_sides:
                continue
            if len(robot_sides) == 2:
                category = "left_right"
            elif "table" in labels:
                category = "robot_table"
            elif "object" in labels:
                category = "robot_object"
            elif labels[0] == labels[1]:
                category = "within_robot"
            else:
                category = "robot_environment"
            neighboring = (
                self.robot.parent_links.get(names[0]) == names[1] or self.robot.parent_links.get(names[1]) == names[0]
            )
            rows.append(
                {
                    "category": category,
                    "sides": sorted(robot_sides),
                    "bodies": names,
                    "hand_bodies": [name for body, name in zip(bodies, names) if body in self.hand_body_ids],
                    "geoms": [self.model.geom(g).name for g in geoms],
                    "distance_m": float(contact.dist),
                    "penetration_m": max(0.0, float(-contact.dist)),
                    "neighboring_source_links": neighboring,
                    "position_world_m": contact.pos.tolist(),
                }
            )
        return rows
