"""Private right Sharpa model; generated assets stay inside the frozen suite."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ada_grasp_ctrl.utils.robots.base import ArmHand, Hand, register_robot
from ada_grasp_ctrl.utils.robots.dummy_arm import DummyArm

HAND_JOINTS = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)
ARM = DummyArm("")
JOINTS = tuple(ARM.dof_names) + HAND_JOINTS
ACTUATORS = tuple(ARM.doa_names) + tuple(f"{j}_position" for j in HAND_JOINTS)
FLANGE = "right_hand_flange"
PALM = "right_hand_C_MC"
ROBOT = "dummy_arm_sharpa_right"


class SharpaRight(Hand):
    """Metadata for the explicit provisional right-hand gain profile."""

    def __init__(self, prefix: str = "") -> None:
        super().__init__(prefix)
        self._dof_names = list(HAND_JOINTS)
        self._doa_names = list(ACTUATORS[6:])
        self._doa2dof_matrix = np.eye(22)
        self._doa_kp = [7.5] * 22
        self._doa_max_vel = [2.0] * 22
        self._base_name = PALM
        self._body_names = [PALM]


@register_robot(ROBOT)
class DummyArmSharpaRight(ArmHand):
    """Private registration; the public control_eval allowlist is unchanged."""

    def __init__(self, prefix: str = "") -> None:
        if prefix:
            raise ValueError("Sharpa experiment requires unprefixed canonical names")
        super().__init__(prefix)
        self.arm = DummyArm("")
        self.hand = SharpaRight()
        self._urdf_path = "sharpa_right.urdf"
        self._mjcf_path = "sharpa_right.xml"


def build_models(source_xml: Path, source_urdf: Path, destination: Path) -> list[Path]:
    """Wrap existing right-hand geometry with six named dummy joints.

    Args:
        source_xml: Bench right-hand MJCF with the kp=7.5 profile.
        source_urdf: Matching flange-rooted right-hand URDF.
        destination: New suite-local model directory.

    Returns:
        Source and mesh dependencies to hash in the suite.

    Raises:
        ValueError: If the source joint, gain, or actuator contract differs.
    """
    root = ET.parse(source_xml).getroot()
    compiler = root.find("compiler")
    meshdir = (source_xml.parent / compiler.get("meshdir", "")).resolve()
    dependencies = [source_xml, source_urdf]
    for mesh in root.findall("./asset/mesh"):
        path = (meshdir / mesh.attrib["file"]).resolve(strict=True)
        dependencies.append(path)
        mesh.set("file", str(path))
    compiler.set("meshdir", "")
    joints = {j.attrib["name"]: j for j in root.findall(".//worldbody//joint")}
    if set(joints) != set(HAND_JOINTS):
        raise ValueError("Expected exactly the 22 right Sharpa joints in Bench XML")
    actuators = root.find("actuator")
    if [a.get("joint") for a in actuators] != list(HAND_JOINTS):
        raise ValueError("Unexpected source actuator order")
    for a in actuators:
        if float(a.get("kp", "0")) != 7.5 or float(a.get("kv", "0")) != 0:
            raise ValueError("Expected explicit right-hand kp=7.5, kv=0 profile")
    world = root.find("worldbody")
    flange = world.find(f"body[@name='{FLANGE}']")
    if flange is None or len(world) != 1:
        raise ValueError("Expected one fixed flange root")
    world.remove(flange)
    parent = ET.SubElement(world, "body", name="dummy_base", gravcomp="1")
    for i, name in enumerate(ARM.dof_names):
        parent = ET.SubElement(parent, "body", name=ARM.body_names[i], gravcomp="1")
        ET.SubElement(parent, "inertial", pos="0 0 0", mass="0.001", diaginertia="0.0001 0.0001 0.0001")
        axis = ["1 0 0", "0 1 0", "0 0 1"][i % 3]
        bounds = "-1 1" if i < 3 else "-6.283 6.283"
        ET.SubElement(
            parent,
            "joint",
            name=name,
            type="slide" if i < 3 else "hinge",
            axis=axis,
            range=bounds,
            limited="true",
            damping="100" if i < 3 else "10",
            armature="0.01",
            actuatorfrcrange="-10000000 10000000",
        )
        actuator = ET.Element(
            "position",
            name=ARM.doa_names[i],
            joint=name,
            kp=str(ARM.doa_kp[i]),
            ctrlrange=bounds,
            ctrllimited="true",
            forcerange="-100000000 100000000",
            forcelimited="true",
        )
        actuators.insert(i, actuator)
    parent.append(flange)

    # URDF is a kinematic derivative, retaining source link frames and joint axes.
    urdf = ET.parse(source_urdf).getroot()
    for extension in list(urdf.findall("mujoco")):
        urdf.remove(extension)
    for link in urdf.findall("link"):
        for child in list(link):
            if child.tag in ("visual", "collision"):
                link.remove(child)
    for joint in urdf.findall("joint"):
        if joint.get("name") in joints:
            lo, hi = joints[joint.get("name")].attrib["range"].split()
            joint.find("limit").set("lower", lo)
            joint.find("limit").set("upper", hi)
    ET.SubElement(urdf, "link", name="dummy_base")
    previous = "dummy_base"
    for i, name in enumerate(ARM.dof_names):
        link_name = ARM.body_names[i]
        ET.SubElement(urdf, "link", name=link_name)
        joint = ET.SubElement(urdf, "joint", name=name, type="prismatic" if i < 3 else "revolute")
        ET.SubElement(joint, "parent", link=previous)
        ET.SubElement(joint, "child", link=link_name)
        ET.SubElement(joint, "axis", xyz=["1 0 0", "0 1 0", "0 0 1"][i % 3])
        bound = "1" if i < 3 else "6.283"
        ET.SubElement(
            joint, "limit", lower=f"-{bound}", upper=bound, effort="10000000", velocity=str(ARM.doa_max_vel[i])
        )
        previous = link_name
    mount = ET.SubElement(urdf, "joint", name="dummy_to_flange", type="fixed")
    ET.SubElement(mount, "parent", link=previous)
    ET.SubElement(mount, "child", link=FLANGE)
    destination.mkdir(parents=True, exist_ok=False)
    ET.indent(root)
    ET.indent(urdf)
    ET.ElementTree(root).write(destination / "sharpa_right.xml", encoding="unicode")
    ET.ElementTree(urdf).write(destination / "sharpa_right.urdf", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(destination / "sharpa_right.xml"))
    if model.nq != 28 or model.nu != 28:
        raise ValueError("Expected 28 joints and 28 actuators")
    if [model.joint(i).name for i in range(28)] != list(JOINTS):
        raise ValueError("MuJoCo joint order does not match the data contract")
    if [model.actuator(i).name for i in range(28)] != list(ACTUATORS):
        raise ValueError("MuJoCo actuator order does not match the data contract")
    return sorted(set(dependencies))
