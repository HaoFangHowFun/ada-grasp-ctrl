"""Named right CRX assembly using the frozen benchmark's canonical right hand."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.paths import project_root
from .sharpa_model import FLANGE, HAND_JOINTS

ARM_JOINTS = tuple(f"right_J{i}" for i in range(1, 7))
JOINTS = ARM_JOINTS + HAND_JOINTS
ACTUATORS = tuple(f"{name}_position" for name in JOINTS)
MOUNT_RPY = (3.14159265359, -1.57079665359, 0.0)


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a finite XYZ/WXYZ pose to a rigid transform, rejecting malformed input."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 1e-4:
        raise ValueError("Expected finite XYZ/WXYZ pose with a unit quaternion")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def matrix_pose(matrix: np.ndarray) -> np.ndarray:
    """Return XYZ/WXYZ from a homogeneous rigid transform."""
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[matrix[:3, 3], quat[[3, 0, 1, 2]]]


def origin_matrix(origin: ET.Element | None) -> np.ndarray:
    """Interpret a URDF origin with extrinsic XYZ roll, pitch and yaw."""
    result = np.eye(4)
    if origin is not None:
        result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


def write_xml(root: ET.Element, path: Path) -> None:
    """Write a deterministic readable XML document."""
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def build_model(suite: dict, destination: Path) -> dict:
    """Assemble source arm kinematics with the exact frozen hand geometry and gains.

    Args:
        suite: Already hash-verified right Sharpa frozen suite.
        destination: New model directory under the current preview run.

    Returns:
        Model paths, named permutations and source dependencies.

    Raises:
        ValueError: If arm geometry origins or canonical joint contracts differ.
    """
    destination.mkdir(parents=True, exist_ok=False)
    repo = project_root()
    source = repo / "assets/robot/crx5ia_sharpa_wave_right/arm_source.urdf"
    visual_source = repo / "assets/robot/crx5ia_sharpa_wave_left/model.urdf"
    arm = ET.parse(source).getroot()
    visuals = ET.parse(visual_source).getroot()
    dependencies = {source, visual_source}
    # FANUC's shared mechanical links are checked before reusing converted DAE visuals.
    # Hand links are never mirrored or renamed from the left-hand asset.
    for link in arm.findall("link"):
        if link.find("visual") is None:
            continue
        previous = visuals.find(f"link[@name='{link.get('name').replace('right_', 'left_', 1)}']")
        if previous is None:
            raise ValueError(f"Missing FANUC visual source: {link.get('name')}")
        for tag in ("visual", "collision"):
            if not np.allclose(
                origin_matrix(link.find(f"{tag}/origin")), origin_matrix(previous.find(f"{tag}/origin")), atol=1e-12
            ):
                raise ValueError(f"FANUC {tag} origin drift: {link.get('name')}")
            for element in list(link.findall(tag)):
                link.remove(element)
            for element in previous.findall(tag):
                element = deepcopy(element)
                for mesh in element.findall(".//mesh"):
                    path = (visual_source.parent / mesh.get("filename")).resolve(strict=True)
                    dependencies.add(path)
                    mesh.set("filename", str(path))
                link.append(element)
    extension = ET.SubElement(arm, "mujoco")
    ET.SubElement(
        extension, "compiler", strippath="false", balanceinertia="true", discardvisual="false", fusestatic="false"
    )
    arm_path = destination / "arm.urdf"
    write_xml(arm, arm_path)
    spec = mujoco.MjSpec.from_file(str(arm_path))
    root = ET.fromstring(spec.to_xml())
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str((destination / mesh.get("file")).resolve(strict=True)))
    hand_dir = Path(suite["asset_root"])
    hand = ET.parse(hand_dir / "sharpa_right.xml").getroot()
    dependencies.update([hand_dir / "sharpa_right.xml", hand_dir / "sharpa_right.urdf"])
    for mesh in hand.findall("./asset/mesh"):
        mesh = deepcopy(mesh)
        path = (hand_dir / mesh.get("file")).resolve(strict=True)
        mesh.set("file", str(path))
        dependencies.add(path)
        root.find("asset").append(mesh)
    flange = deepcopy(hand.find(f".//body[@name='{FLANGE}']"))
    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_euler("xyz", MOUNT_RPY).as_matrix()
    flange.set("pos", "0 0 0")
    flange.set("quat", " ".join(map(str, matrix_pose(mount)[3:])))
    root.find(".//body[@name='right_ee_mount']").append(flange)
    actuators = ET.SubElement(root, "actuator")
    limits = {}
    for name in ARM_JOINTS:
        joint = arm.find(f"joint[@name='{name}']")
        limit = joint.find("limit")
        limits[name] = dict(limit.attrib)
        ET.SubElement(
            actuators,
            "position",
            name=f"{name}_position",
            joint=name,
            kp="100",
            kv="10",
            ctrlrange=f"{limit.get('lower')} {limit.get('upper')}",
            ctrllimited="true",
            forcerange=f"-{limit.get('effort')} {limit.get('effort')}",
            forcelimited="true",
        )
    for name in HAND_JOINTS:
        actuators.append(deepcopy(hand.find(f"./actuator/*[@joint='{name}']")))
    # Keep every hand collision exclusion from the benchmark, without dummy bodies.
    if hand.find("contact") is not None:
        root.append(deepcopy(hand.find("contact")))
    model_path = destination / "model.xml"
    write_xml(root, model_path)
    # Independent URDF kinematics retain the same canonical hand frames and axes.
    kin = deepcopy(arm)
    hand_kin = ET.parse(hand_dir / "sharpa_right.urdf").getroot()
    for element in hand_kin:
        if element.tag == "link" and element.get("name", "").startswith("right_"):
            kin.append(deepcopy(element))
        if element.tag == "joint" and element.find("parent").get("link", "").startswith("right_"):
            kin.append(deepcopy(element))
    joint = ET.SubElement(kin, "joint", name="right_arm_to_hand", type="fixed")
    ET.SubElement(joint, "parent", link="right_ee_mount")
    ET.SubElement(joint, "child", link=FLANGE)
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy=" ".join(map(str, MOUNT_RPY)))
    kinematics = destination / "model.urdf"
    write_xml(kin, kinematics)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != 28 or model.nu != 28 or {model.joint(i).name for i in range(28)} != set(JOINTS):
        raise ValueError("Expected exactly 28 named right-side joints and actuators")
    mapping = {
        name: {
            "qpos": int(model.joint(name).qposadr[0]),
            "dof": int(model.joint(name).dofadr[0]),
            "actuator": model.actuator(f"{name}_position").id,
        }
        for name in JOINTS
    }
    for name, indices in mapping.items():
        if model.actuator_trnid[indices["actuator"], 0] != model.joint(name).id:
            raise ValueError(f"Actuator transmission mismatch: {name}")
    return {
        "mjcf": str(model_path),
        "urdf": str(kinematics),
        "joint_names": list(JOINTS),
        "mapping": mapping,
        "dependencies": sorted(map(str, dependencies)),
        "arm_limits": limits,
        "world_T_base": origin_matrix(arm.find("joint[@name='right_base_joint']/origin")).tolist(),
        "ee_mount_T_hand_flange": mount.tolist(),
        "mount_calibrated": False,
        "arm_gains": {"kp": 100, "kv": 10, "status": "provisional; no control evaluation yet"},
    }


def build_scene(model_path: Path, record: dict, registration: np.ndarray, path: Path) -> None:
    """Build the evaluation scene with the original convex meshes, scale and table plane.

    Args:
        model_path: Right arm/hand model used for IK and evaluation.
        record: Registered common grasp record.
        registration: Evaluation-world from source-world rigid transform.
        path: New scene XML output path.
    """
    from ada_grasp_ctrl.utils.hand_util import collision_mesh_paths

    root = ET.parse(model_path).getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.004")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("noslip_iterations", "2")
    option.set("impratio", "10")
    assets, world = root.find("asset"), root.find("worldbody")
    table_pose = matrix_pose(registration)
    table = ET.SubElement(
        world, "body", name="table", pos=" ".join(map(str, table_pose[:3])), quat=" ".join(map(str, table_pose[3:]))
    )
    ET.SubElement(
        table, "geom", name="table_surface", type="plane", size="0.75 0.75 0.02", rgba="0.40 0.46 0.53 1", group="1"
    )
    # A display-only pedestal makes the right base position easy to inspect.
    base = root.find(".//body[@name='right_base_link']")
    ET.SubElement(
        base,
        "geom",
        name="base_marker",
        type="cylinder",
        size="0.12 0.025",
        pos="0 0 -0.025",
        rgba="0.2 0.24 0.28 1",
        contype="0",
        conaffinity="0",
        group="1",
    )
    obj = ET.SubElement(world, "body", name="object")
    ET.SubElement(obj, "freejoint", name="object_freejoint")
    scale = str(record["obj_scale"])
    paths = collision_mesh_paths(record["obj_path"])
    if not paths:
        raise ValueError("Object has no convex collision meshes")
    info = json.loads((Path(record["obj_path"]) / "info/simplified.json").read_text())
    coefficient = info["mass"] / (info["density"] * info["scale"] ** 3)
    density = 0.1 / (coefficient * float(record["obj_scale"]) ** 3)
    for i, mesh in enumerate(paths):
        ET.SubElement(assets, "mesh", name=f"object_mesh_{i}", file=str(mesh), scale=" ".join([scale] * 3))
        ET.SubElement(
            obj,
            "geom",
            name=f"object_collision_{i}",
            type="mesh",
            mesh=f"object_mesh_{i}",
            rgba="0.93 0.65 0.28 1",
            group="1",
            density=str(density),
        )
    for geom in root.findall(".//geom"):
        geom.set("friction", "0.8 0.02 0.001")
        geom.set("condim", "4")
        geom.set("solimp", "0.5 0.99 0.0001")
        geom.set("solref", "0.005 1")
    ET.SubElement(world, "light", pos="0 -0.4 2", dir="0 0 -1", directional="true", diffuse="0.8 0.8 0.8")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    write_xml(root, path)
