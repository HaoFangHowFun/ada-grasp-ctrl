"""Load one exported canonical URDF into Pinocchio and MuJoCo."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pinocchio as pin

from ada_grasp_ctrl.paths import resolve_from_root

HAND_SUFFIXES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
JOINT_NAMES = tuple(
    name
    for side in ("left", "right")
    for name in (*[f"{side}_J{i}" for i in range(1, 7)], *[f"{side}_{suffix}" for suffix in HAND_SUFFIXES])
)


def side_slice(side: str, arm_only: bool = False) -> slice:
    """Return the explicit side block in Ada's canonical 56-vector."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    start = 0 if side == "left" else 28
    return slice(start, start + (6 if arm_only else 28))


class DualArmModel:
    """Canonical named model; source geometry and mounts are never independently edited.

    Args:
        snapshot_directory: Completed snapshot produced by export_bimanual_model.py.

    Raises:
        ValueError: If the joint, world-root, snapshot, or index contracts differ.
    """

    joint_names = JOINT_NAMES

    @classmethod
    def load(cls, snapshot_directory: str | Path) -> DualArmModel:
        """Load a portable exported snapshot without any ROS package lookup."""
        return cls(snapshot_directory)

    def __init__(self, snapshot_directory: str | Path) -> None:
        self.directory = resolve_from_root(snapshot_directory).resolve(strict=True)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if (
            self.manifest.get("schema_version") != 1
            or self.manifest.get("status") != "complete"
            or self.manifest.get("joint_names") != list(JOINT_NAMES)
        ):
            raise ValueError("Expected a complete canonical 56-joint snapshot")
        self.urdf_path = self.directory / "model.urdf"
        root = ET.parse(self.urdf_path).getroot()
        links = [link.attrib["name"] for link in root.findall("link")]
        joints = root.findall("joint")
        names = [joint.attrib["name"] for joint in joints]
        if len(set(links)) != len(links) or len(set(names)) != len(names):
            raise ValueError("Duplicate source link or joint names")
        children = [joint.find("child").attrib["link"] for joint in joints]
        if set(links) - set(children) != {"world"} or len(children) != len(set(children)):
            raise ValueError("Expected the canonical fixed world-rooted tree")
        movable = {joint.attrib["name"]: joint for joint in joints if joint.attrib["type"] != "fixed"}
        if set(movable) != set(JOINT_NAMES) or any(j.attrib["type"] != "revolute" for j in movable.values()):
            raise ValueError("Expected exactly 56 bounded revolute robot joints")
        if any(j.find("mimic") is not None for j in movable.values()):
            raise ValueError("Mimic joints are not supported by the independent 56-DoF contract")
        self.limits = np.array(
            [[float(movable[n].find("limit").attrib[k]) for k in ("lower", "upper")] for n in JOINT_NAMES]
        )
        self.effort = np.array([float(movable[n].find("limit").attrib["effort"]) for n in JOINT_NAMES])
        if (
            not np.isfinite(self.limits).all()
            or np.any(self.limits[:, 0] >= self.limits[:, 1])
            or not np.isfinite(self.effort).all()
            or np.any(self.effort <= 0)
        ):
            raise ValueError("Invalid source joint limits/effort")
        for mesh in root.findall(".//mesh"):
            path = (self.directory / mesh.attrib["filename"]).resolve(strict=True)
            if not path.is_relative_to(self.directory):
                raise ValueError("Snapshot mesh references must remain inside the snapshot")
        self.parent_links = {j.find("child").attrib["link"]: j.find("parent").attrib["link"] for j in joints}
        self.pin_model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.pin_data = self.pin_model.createData()
        pin_names = set(self.pin_model.names[1:])
        if pin_names != set(JOINT_NAMES):
            raise ValueError("Pinocchio joint names differ from the canonical model")
        self.pin_q = np.array([self.pin_model.joints[self.pin_model.getJointId(n)].idx_q for n in JOINT_NAMES])
        self.pin_v = np.array([self.pin_model.joints[self.pin_model.getJointId(n)].idx_v for n in JOINT_NAMES])
        if any(j.nq != 1 or j.nv != 1 for j in list(self.pin_model.joints)[1:]):
            raise ValueError("Expected scalar Pinocchio joint coordinates")
        if not np.allclose(self.pin_model.lowerPositionLimit[self.pin_q], self.limits[:, 0], atol=1e-12, rtol=0):
            raise ValueError("Pinocchio lower limits differ from URDF")
        if not np.allclose(self.pin_model.upperPositionLimit[self.pin_q], self.limits[:, 1], atol=1e-12, rtol=0):
            raise ValueError("Pinocchio upper limits differ from URDF")
        self.spec = mujoco.MjSpec.from_file(str(self.urdf_path))
        self.spec.option.timestep = 0.002
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for i, name in enumerate(JOINT_NAMES):
            actuator = self.spec.add_actuator(name=f"{name}_position", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            arm = i % 28 < 6
            actuator.set_to_position(kp=100 if arm else 7.5, kv=10 if arm else 0)
            if not arm:
                # Existing Sharpa right.xml uses these numerical joint dynamics with kp=7.5.
                # They stabilize the tiny hand inertias without changing canonical geometry.
                joint = self.spec.joint(name)
                joint.damping = 0.05
                joint.armature = 1e-5
            actuator.ctrllimited = True
            actuator.ctrlrange = self.limits[i]
            actuator.forcelimited = True
            actuator.forcerange = [-self.effort[i], self.effort[i]]
        self.mj_model = self.spec.compile()
        self.mapping = self.mapping_for(self.mj_model)
        self.frames = {
            side: {
                "base": f"{side}_base_link",
                "crx_flange": f"{side}_flange",
                "ee_mount": f"{side}_ee_mount",
                "fanuc_flange": f"{side}_fanuc_flange",
                "hand_flange": f"{side}_hand_flange",
                "palm": f"{side}_hand_C_MC",
                "tcp": f"{side}_tcp",
            }
            for side in ("left", "right")
        }
        for frames in self.frames.values():
            for name in frames.values():
                if not self.pin_model.existFrame(name) or self.mj_model.body(name).id < 0:
                    raise ValueError(f"Missing canonical frame: {name}")

    def mapping_for(self, model: mujoco.MjModel) -> dict[str, dict[str, int]]:
        """Validate explicit robot indices even when a scene includes a free object."""
        actual = {model.joint(i).name for i in range(model.njnt) if model.joint(i).name.startswith(("left_", "right_"))}
        if actual != set(JOINT_NAMES):
            raise ValueError("MuJoCo robot subset differs from canonical joints")
        mapping = {}
        for i, name in enumerate(JOINT_NAMES):
            joint, actuator = model.joint(name), model.actuator(f"{name}_position")
            if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Expected scalar revolute joint: {name}")
            if not joint.limited or not np.allclose(joint.range, self.limits[i], atol=1e-12, rtol=0):
                raise ValueError(f"MuJoCo source-limit mismatch: {name}")
            if model.actuator_trnid[actuator.id, 0] != joint.id:
                raise ValueError(f"Actuator transmission mismatch: {name}")
            mapping[name] = {
                "qpos": int(joint.qposadr[0]),
                "qvel": int(joint.dofadr[0]),
                "actuator": actuator.id,
                "pin_q": int(self.pin_q[i]),
                "pin_v": int(self.pin_v[i]),
            }
        for field in ("qpos", "qvel", "actuator", "pin_q", "pin_v"):
            if len({row[field] for row in mapping.values()}) != 56:
                raise ValueError(f"Non-unique joint mapping: {field}")
        return mapping

    def configuration(self, value: object, *, check_limits: bool = True) -> np.ndarray:
        """Validate a finite canonical 56-vector, optionally enforcing source limits."""
        q = np.asarray(value, dtype=float)
        if q.shape != (56,) or not np.isfinite(q).all():
            raise ValueError("Expected a finite canonical configuration of shape (56,)")
        if check_limits and (np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1])):
            raise ValueError("Configuration exceeds source joint limits")
        return q.copy()

    def home_configuration(self) -> np.ndarray:
        """Return bounded zero angles for offline inspection, not a calibrated robot home."""
        return self.configuration(np.zeros(56))

    def frame_name(self, side: str, frame: str) -> str:
        """Resolve an explicit semantic alias or exact frame on the selected side."""
        side_slice(side)
        name = self.frames[side].get(frame, frame)
        if not name.startswith(side + "_") or not self.pin_model.existFrame(name):
            raise ValueError(f"Unknown {side} frame: {frame}")
        return name

    def fk(self, q: object, side: str, frame: str = "hand_flange") -> np.ndarray:
        """Return world_T_frame from the exported URDF using Pinocchio."""
        state = pin.neutral(self.pin_model)
        state[self.pin_q] = self.configuration(q)
        pin.framesForwardKinematics(self.pin_model, self.pin_data, state)
        return self.pin_data.oMf[self.pin_model.getFrameId(self.frame_name(side, frame))].homogeneous.copy()

    def solve_arm_ik(
        self,
        q_current: object,
        side: str,
        target_world_T_flange: object,
        *,
        frame: str = "hand_flange",
        max_nfev: int = 200,
    ) -> tuple[np.ndarray, dict]:
        """Solve only the six selected arm joints and return explicit acceptance diagnostics."""
        from .kinematics import solve_arm_ik

        return solve_arm_ik(self, q_current, side, target_world_T_flange, frame=frame, max_nfev=max_nfev)
