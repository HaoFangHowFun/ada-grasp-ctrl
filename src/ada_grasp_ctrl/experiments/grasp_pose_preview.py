"""Read-only static preview of saved selected grasp targets and original object poses."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.robot_environment.model import JOINT_NAMES


class GraspPosePreview:
    """Load the exact compiled run model without inference, control, or physics stepping."""

    def __init__(self, root: Path) -> None:
        self.root = resolve_from_root(root)
        self.config = yaml.safe_load((self.root / "effective_scene.yaml").read_text())
        self.selected = json.loads((self.root / "selection.json").read_text())["selected"]
        if self.selected is None:
            raise ValueError(f"No selected grasp in {self.root}")
        self.model = mujoco.MjModel.from_binary_path(str(self.root / "scene.mjb"))
        self.data = mujoco.MjData(self.model)
        with np.load(self.root / "initial_state.npz", allow_pickle=False) as archive:
            self.initial_qpos = archive["qpos"].copy()
        if self.initial_qpos.shape != (self.model.nq,) or not np.isfinite(self.initial_qpos).all():
            raise ValueError("Malformed saved initial qpos")
        self.indices = np.array([int(self.model.joint(n).qposadr[0]) for n in JOINT_NAMES])
        self.side = self.selected["selected_side"]
        self.hand_ids = {self.model.body(f"{self.side}_hand_flange").id}
        for body in range(self.model.nbody):
            if self.model.body_parentid[body] in self.hand_ids:
                self.hand_ids.add(body)
        self.object_id = self.model.body("object").id
        self.original_groups = self.model.geom_group.copy()
        self.set_stage("grasp")

    def set_stage(self, stage: str) -> None:
        """Apply a selected IK target, keeping the object at the inference pose and time zero."""
        if stage not in ("pregrasp", "grasp", "squeeze"):
            raise ValueError("Expected pregrasp, grasp or squeeze")
        q = np.asarray(self.selected["stages"][stage]["q56"], dtype=float)
        if q.shape != (56,) or not np.isfinite(q).all():
            raise ValueError(f"Malformed {stage} target")
        self.data.qpos[:] = self.initial_qpos
        self.data.qpos[self.indices] = q
        self.data.qvel[:] = 0
        self.data.time = 0
        mujoco.mj_forward(self.model, self.data)
        self.stage = stage

    def assign_context_group(self) -> None:
        """Put only visual geoms outside the active hand/object in display group four.

        Collision geoms retain groups zero/three and all physical parameters.
        Group five may already hide visual duplicates replaced with source colors.
        """
        keep = self.hand_ids | {self.object_id}
        for geom in range(self.model.ngeom):
            if self.model.geom_group[geom] not in (0, 3, 5) and self.model.geom_bodyid[geom] not in keep:
                self.model.geom_group[geom] = 4
