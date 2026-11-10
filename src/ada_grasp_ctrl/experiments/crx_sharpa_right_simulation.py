"""Simulation adapters for the private right CRX smoke experiment."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ada_grasp_ctrl.utils.hand_util import MjHO
from ada_grasp_ctrl.utils.robots.base import Arm, ArmHand
from .crx_sharpa_right_model import ACTUATORS, ARM_JOINTS, JOINTS
from .sharpa_model import FLANGE, SharpaRight


BASE_J1_EXCLUSION = ("right_base_link", "right_J1_link")


def exclude_base_joint_contact(spec: mujoco.MjSpec) -> None:
    """Exclude the adjacent fixed base/J1 bearing surfaces, retaining environment contacts.

    MuJoCo does not apply its usual parent collision filter to a parent welded to
    the world. The imported base/J1 meshes touch at the bearing and otherwise
    generate constraint friction opposing J1. Require the exact parent relation
    and never disable either body's collisions with other bodies.
    """
    base, child = (spec.body(name) for name in BASE_J1_EXCLUSION)
    if base is None or child is None or child.parent.name != base.name:
        raise ValueError("Expected adjacent right_base_link/right_J1_link bodies")
    if not any({e.bodyname1, e.bodyname2} == set(BASE_J1_EXCLUSION) for e in spec.excludes):
        spec.add_exclude(bodyname1=base.name, bodyname2=child.name)


class RightCrxArm(Arm):
    """Canonical arm metadata using the approved model's actuator gains and limits."""

    def __init__(self, info: dict) -> None:
        super().__init__("")
        self._dof_names = list(ARM_JOINTS)
        self._doa_names = list(ACTUATORS[:6])
        self._doa2dof_matrix = np.eye(6)
        self._doa_kp = [info["arm_gains"]["kp"]] * 6
        self._doa_max_vel = [float(info["arm_limits"][name]["velocity"]) for name in ARM_JOINTS]
        self._body_names = [f"right_J{i}_link" for i in range(1, 7)]
        self._base_name = "right_base_link"


class RightCrxSharpa(ArmHand):
    """Private robot object; importing this experiment does not change public dispatch."""

    def __init__(self, info: dict) -> None:
        super().__init__("")
        self.arm = RightCrxArm(info)
        self.hand = SharpaRight()
        self._urdf_path = info["urdf"]
        self._mjcf_path = info["mjcf"]


class ApprovedScene(MjHO):
    """Use the reviewed placement with explicit benchmark and adjacent-contact adaptations.

    The native benchmark disables global gravity and applies a 1 N downward object
    force. This numerical setting is applied here without rebuilding the scene or
    suppressing the reviewed arm/table contacts. Only the fixed base/J1 bearing pair
    is excluded. Robot state is always mapped by name.
    """

    hand_prefix = ""

    def __init__(self, scene_path: Path) -> None:
        self.spec = mujoco.MjSpec.from_file(str(scene_path))
        exclude_base_joint_contact(self.spec)
        self.spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.hand_mocap = False
        self.debug_viewer = self.debug_render = None
        self._owns_debug_viewer_session = False
        self.ext_force_on_obj = None
        self.target_qpos_a = np.zeros(28)
        self.debug_images = []
        self.object_id = self.model.body("object").id
        if self.object_id != self.model.nbody - 1:
            raise ValueError("Native contact adapter requires the named object to be the last body")
        self.q_addresses = [int(self.model.joint(n).qposadr[0]) for n in JOINTS]
        self.a_addresses = [self.model.actuator(n).id for n in ACTUATORS]
        self.object_address = int(self.model.joint("object_freejoint").qposadr[0])
        hand_ids = {self.model.body(FLANGE).id}
        for body in range(self.model.nbody):
            if self.model.body_parentid[body] in hand_ids:
                hand_ids.add(body)
        self.hand_bodies = {self.model.body(i).name for i in hand_ids}
        # Native canonicalization is retained; only supported hand bodies reach the controller.
        self.hand_contact_prefixes = tuple(sorted(self.hand_bodies))
        self.collision_history: list[dict] = []
        self.physics_steps = 0
        self.segment = "initial"
        for name, actuator in zip(JOINTS, self.a_addresses):
            if self.model.actuator_trnid[actuator, 0] != self.model.joint(name).id:
                raise ValueError(f"Actuator mapping mismatch: {name}")

    def reset_pose_qpos(self, hand_qpos: np.ndarray, obj_pose: np.ndarray) -> None:
        """Reset every integration state and set both poses with explicit named addresses."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.q_addresses] = hand_qpos
        self.data.qpos[self.object_address : self.object_address + 7] = obj_pose
        self.data.ctrl[self.a_addresses] = hand_qpos
        self.target_qpos_a = np.asarray(hand_qpos).copy()
        mujoco.mj_forward(self.model, self.data)

    def get_obj_pose(self) -> np.ndarray:
        """Return the named object's XYZ/WXYZ pose."""
        return self.data.qpos[self.object_address : self.object_address + 7].copy()

    def get_contact_info(self, hand_qpos: np.ndarray, obj_pose: np.ndarray, obj_margin: float = 0) -> tuple:
        """Keep only hand/object and hand/hand pairs in native initialization checks."""
        ho, hh = super().get_contact_info(hand_qpos, obj_pose, obj_margin)
        return (
            [c for c in ho if c["body1_name"] in self.hand_bodies],
            [c for c in hh if c["body1_name"] in self.hand_bodies and c["body2_name"] in self.hand_bodies],
        )

    def get_curr_contact_info(self) -> list[dict]:
        """Return native canonical six-dimensional contact wrenches for the hand only."""
        return [c for c in super().get_curr_contact_info() if c["body1_name"] in self.hand_bodies]

    def set_ext_force_on_obj_single_step(self, ext_force: np.ndarray) -> None:
        """Apply the benchmark's emulated object gravity to the named body."""
        self.data.xfrc_applied[self.object_id] = ext_force

    def contact_diagnostics(self) -> dict:
        """Separate forbidden arm/environment contacts from hand grasp penetration."""
        result = {
            "arm_penetration_m": 0.0,
            "hand_object_penetration_m": 0.0,
            "hand_self_penetration_m": 0.0,
            "hand_table_penetration_m": 0.0,
            "arm_pairs": [],
        }
        for contact in self.data.contact:
            bodies = [self.model.body(self.model.geom_bodyid[g]).name for g in contact.geom]
            depth = max(0.0, -float(contact.dist))
            if depth <= 0:
                continue
            is_arm = [b.startswith("right_J") or b == "right_base_link" for b in bodies]
            if any(is_arm):
                # The fixed base/J1 exception is excluded when compiling the scene.
                result["arm_penetration_m"] = max(result["arm_penetration_m"], depth)
                if depth > 1e-5:
                    result["arm_pairs"].append({"bodies": bodies, "depth_m": depth})
            elif "object" in bodies and any(b in self.hand_bodies for b in bodies):
                result["hand_object_penetration_m"] = max(result["hand_object_penetration_m"], depth)
            elif all(b in self.hand_bodies for b in bodies):
                result["hand_self_penetration_m"] = max(result["hand_self_penetration_m"], depth)
            elif "table" in bodies and any(b in self.hand_bodies for b in bodies):
                result["hand_table_penetration_m"] = max(result["hand_table_penetration_m"], depth)
        return result

    def control_hand_step(self, step_inner: int) -> None:
        """Step native dynamics while retaining collision events and invalid-state failures."""
        for _ in range(step_inner):
            if self.ext_force_on_obj is not None:
                self.set_ext_force_on_obj_single_step(self.ext_force_on_obj)
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("Nonfinite MuJoCo state")
            if any(self.data.warning[i].number for i in range(len(self.data.warning))):
                raise RuntimeError("MuJoCo issued a numerical/constraint warning")
            diagnostic = self.contact_diagnostics()
            if diagnostic["arm_penetration_m"] > 1e-5 or diagnostic["hand_table_penetration_m"] > 0.01:
                self.collision_history.append({"time": float(self.data.time), "segment": self.segment, **diagnostic})


def invalid_initialization(diagnostic: dict) -> bool:
    """Preserve the 10 mm hand gate and add a separate 10 micrometer arm clearance gate."""
    return bool(
        diagnostic["arm_penetration_m"] > 1e-5
        or max(
            diagnostic[k] for k in ("hand_object_penetration_m", "hand_self_penetration_m", "hand_table_penetration_m")
        )
        > 0.01
    )
