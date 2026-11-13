"""Static dual-robot checks and side-local geometry for measured live scenes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.robot_environment import DualArmModel, DualArmSimulation
from ada_grasp_ctrl.robot_environment.model import side_slice
from ada_grasp_ctrl.experiments.simulated_object_cloud import exclude_adjacent_base_contacts


def live_scene(snapshot: Path, state: dict, plane: dict) -> tuple[DualArmModel, DualArmSimulation]:
    """Use measured 56-axis state and table plane; a partial cloud is not a collision solid."""
    robot = DualArmModel.load(snapshot)
    if state.get("full_joint_names") != list(robot.joint_names):
        raise ValueError("Snapshot must contain the canonical 56 named joints")
    q = robot.configuration(state["full_positions_rad"])
    spec = robot.spec.copy()
    exclude_adjacent_base_contacts(spec)
    # The measured table replaces only the nominal exported table collision surface.
    for geom in spec.geoms:
        if geom.name.startswith("calibration_table_surface"):
            geom.contype = geom.conaffinity = 0
    n = np.asarray(plane["normal_xyz"], dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n), 1, atol=1e-6) or n[2] <= 0:
        raise ValueError("Expected a finite upward unit table normal")
    d = float(plane["d"])
    if not np.isfinite(d):
        raise ValueError("Expected finite table offset")
    rotation, _ = Rotation.align_vectors(n[None], np.array([[0.0, 0.0, 1.0]]))
    quat = rotation.as_quat()[[3, 0, 1, 2]]
    table = spec.worldbody.add_body(name="live_table", pos=-d * n, quat=quat)
    table.add_geom(
        name="live_table_surface",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2, 2, 0.01],
        rgba=[0.55, 0.45, 0.3, 0.3],
        group=3,
    )
    sim = DualArmSimulation(robot, scene_spec=spec, table_bodies=("live_table", "calibration_table_surface"))
    sim.reset(q)
    return robot, sim


class SideAdapter:
    """Expose a selected 28-axis block to the existing overhead and lift endpoint helpers."""

    def __init__(self, robot: DualArmModel, sim: DualArmSimulation, q56: object, side: str) -> None:
        self.robot, self.side = robot, side
        self.current = robot.configuration(q56)
        self.block = side_slice(side)
        self.model, self.data = sim.model, mujoco.MjData(sim.model)
        self.addresses = sim.qpos[self.block]
        self.data.qpos[sim.qpos] = self.current
        self.limits = robot.limits[self.block]
        self.model_T_world = np.eye(4)
        self.hand_bodies = {
            sim.model.body(i).name for i in sim.hand_body_ids if sim.model.body(i).name.startswith(side + "_")
        }
        self.kin = SimpleNamespace(bounds=self.limits[:6].T, forward=self.forward)

    def forward(self, q: np.ndarray) -> np.ndarray:
        full = self.current.copy()
        full[self.block] = q
        return self.robot.fk(full, self.side)
