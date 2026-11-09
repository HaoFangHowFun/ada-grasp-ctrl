"""Static preparation for the right CRX preview milestone; never advances physics."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record
from ada_grasp_ctrl.paths import project_root
from .crx_sharpa_right_model import (
    ARM_JOINTS,
    JOINTS,
    build_model,
    build_scene,
    matrix_pose,
    pose_matrix,
)
from .hugs_sharpa_input import FIELDS, SCALES
from .phase_comparison import sha256
from .sharpa_model import FLANGE, HAND_JOINTS, PALM, JOINTS as SOURCE_JOINTS
from .sharpa_regression import load_suite

# One fixed registration for every object/scale/controller, chosen before any outcomes.
# Keep the tabletop level and move the source origin to the right arm's forward side.
REGISTRATION = np.eye(4)
REGISTRATION[:3, 3] = [0.8439450985964946, -0.7016988483249088, 0.0]
POSITION_TOLERANCE = 1e-3
ANGLE_TOLERANCE = 1e-2


def json_write(path: Path, value: Any) -> None:
    """Write finite JSON, converting NumPy values without hiding nonfinite diagnostics."""

    def encode(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    path.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + "\n")


def body_pose(data: mujoco.MjData, name: str) -> np.ndarray:
    """Return a named body's world transform after forward kinematics."""
    result = np.eye(4)
    result[:3, :3] = data.body(name).xmat.reshape(3, 3)
    result[:3, 3] = data.body(name).xpos
    return result


def residual(actual: np.ndarray, target: np.ndarray) -> dict:
    """Measure translation in meters and shortest rotation angle in radians."""
    return {
        "position_m": float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        "orientation_rad": float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()),
    }


def register_record(source: dict, registration: np.ndarray) -> tuple[dict, list[dict]]:
    """Apply one rigid transform to source-world object and flange poses exactly once."""
    if source.get("source_frame") != FLANGE or source.get("pregrasp_extrapolation") != 2:
        raise ValueError("Expected frozen world-flange input with pregrasp extrapolation already applied")
    registration = np.asarray(registration, dtype=float)
    if (
        registration.shape != (4, 4)
        or not np.isfinite(registration).all()
        or not np.allclose(registration[3], [0, 0, 0, 1], atol=1e-12)
        or not np.allclose(registration[:3, :3].T @ registration[:3, :3], np.eye(3), atol=1e-12)
        or not np.isclose(np.linalg.det(registration[:3, :3]), 1, atol=1e-12)
    ):
        raise ValueError("Registration must be a finite proper rigid transform")
    original_object = pose_matrix(source["obj_pose"])
    registered_object = registration @ original_object
    result = dict(source)
    result["obj_pose"] = matrix_pose(registered_object)
    result["joint_names"] = list(JOINTS)
    result["quaternion_order"] = "WXYZ"
    result["source_pose_convention"] = "source-world; not object-local"
    targets, checks = [], []
    for target in source["flange_targets_wxyz"]:
        original = pose_matrix(target)
        registered = registration @ original
        before = np.linalg.solve(original_object, original)
        after = np.linalg.solve(registered_object, registered)
        check = residual(after, before)
        check["passed"] = check["position_m"] < 1e-10 and check["orientation_rad"] < 1e-10
        if not check["passed"]:
            raise ValueError("Scene registration changed object-relative grasp pose")
        checks.append(check)
        targets.append(matrix_pose(registered))
    result["flange_targets_wxyz"] = np.asarray(targets)
    return result, checks


def load_initial_configuration(task_path: Path, reference_path: Path) -> np.ndarray:
    """Read URDF-radian offline_seed and named open hand; cross-check the degree reference.

    Args:
        task_path: Bimanual right-arm task YAML containing offline_seed and open_hand.
        reference_path: Shared initial_pose YAML whose right entry is in URDF degrees.

    Returns:
        Canonical 28-joint initial configuration in radians.

    Raises:
        ValueError: If names, dimensions, finite values or the two arm references disagree.
    """
    task = yaml.safe_load(task_path.read_text())
    reference = yaml.safe_load(reference_path.read_text())
    arm = np.asarray(task["offline_seed"], dtype=float)
    degrees = np.asarray(reference["right"], dtype=float)
    hand = task["open_hand"]
    if arm.shape != (6,) or degrees.shape != (6,) or set(hand) != set(HAND_JOINTS):
        raise ValueError("Initial config requires six arm angles and all 22 canonical right-hand joints")
    q = np.r_[arm, [hand[name] for name in HAND_JOINTS]]
    if not np.isfinite(q).all() or not np.isfinite(degrees).all():
        raise ValueError("Initial joint values must be finite")
    if not np.allclose(arm, np.deg2rad(degrees), atol=1e-10, rtol=0):
        raise ValueError("offline_seed radians disagree with shared initial_pose URDF degrees")
    return q


class RightKinematics:
    """Bounded CRX IK and independent MuJoCo/Pinocchio frame/Jacobian audit."""

    def __init__(self, info: dict) -> None:
        from ada_grasp_ctrl.utils.pin_helper import PinocchioHelper

        spec = mujoco.MjSpec.from_file(info["mjcf"])
        spec.worldbody.add_geom(
            name="ik_table", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=REGISTRATION[:3, 3], size=[1, 1, 0.01]
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.addresses = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.dofs = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.bounds = np.array([self.model.joint(n).range for n in ARM_JOINTS]).T
        self.pin = PinocchioHelper(info["urdf"], "urdf")
        if set(self.pin.dof_joint_names) != set(JOINTS):
            raise ValueError("Independent kinematics has different joints")
        self.to_pin = [JOINTS.index(n) for n in self.pin.dof_joint_names]
        self.to_user = [self.pin.dof_joint_names.index(n) for n in JOINTS]

    def forward(self, q: np.ndarray) -> np.ndarray:
        """Compute named flange FK without running contact dynamics."""
        self.data.qpos[self.addresses] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return body_pose(self.data, FLANGE)

    def solve(self, target: np.ndarray, hand: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, dict]:
        """Use continuous seeds, fixed fallback seeds and explicit residual acceptance."""

        def error(arm: np.ndarray) -> np.ndarray:
            actual = self.forward(np.r_[arm, hand])
            return np.r_[
                10 * (actual[:3, 3] - target[:3, 3]),
                Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).as_rotvec(),
            ]

        seeds = [
            seed,
            np.deg2rad([0, -30, 130, 0, -60, 0]),
            np.deg2rad([0, 60, 150, 180, 60, 0]),
            np.deg2rad([160, 30, 50, 0, 60, 0]),
            np.deg2rad([160, -30, 130, 0, -60, 0]),
            np.deg2rad([-160, 30, 50, 0, 60, 0]),
        ]
        best, best_rank, attempts = None, (True, True, float("inf")), []
        for initial in seeds:
            fit = least_squares(
                error,
                np.clip(initial, self.bounds[0] + 1e-8, self.bounds[1] - 1e-8),
                bounds=self.bounds,
                max_nfev=160,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            actual = self.forward(np.r_[fit.x, hand])
            errors = residual(actual, target)
            reachable = bool(
                fit.success
                and errors["position_m"] <= POSITION_TOLERANCE
                and errors["orientation_rad"] <= ANGLE_TOLERANCE
            )
            mujoco.mj_forward(self.model, self.data)
            table_id = self.model.geom("ik_table").id
            penetration = 0.0
            for contact in self.data.contact:
                if table_id not in contact.geom:
                    continue
                other = next(i for i in contact.geom if i != table_id)
                body = self.model.body(self.model.geom_bodyid[other]).name
                if body.startswith("right_J") or body == "right_base_link":
                    penetration = max(penetration, float(-contact.dist))
            clear = penetration <= 1e-5
            cost = float(np.linalg.norm(error(fit.x)))
            attempts.append({"seed_rad": initial, "reachable": reachable, "arm_table_penetration_m": penetration})
            rank = (not reachable, not clear, cost)
            if rank < best_rank:
                best, best_rank = fit, rank
            if reachable and clear:
                break
        accepted = not best_rank[0]
        q = np.r_[best.x, hand]
        actual = self.forward(q)
        return q, {
            **residual(actual, target),
            "reachable": accepted,
            "attempts": attempts,
            "arm_table_clear": not best_rank[1],
            "solver_success": bool(best.success),
            "solver_message": best.message,
            "actual_flange_wxyz": matrix_pose(actual),
            "actual_palm_wxyz": matrix_pose(body_pose(self.data, PALM)),
            "reason": None if accepted else "bounded IK residual exceeds tolerance or solver did not converge",
        }

    def audit(self, q: np.ndarray) -> dict:
        """Compare every retained right-side body and its world-space Jacobian."""
        self.forward(q)
        self.pin.compute_jacobians(q[self.to_pin])
        errors = {"position_m": 0.0, "orientation_rad": 0.0, "jacobian": 0.0}
        for i in range(1, self.model.nbody):
            name = self.model.body(i).name
            if not name.startswith("right_"):
                continue
            diff = residual(body_pose(self.data, name), self.pin.get_frame_pose(name))
            for key, value in diff.items():
                errors[key] = max(errors[key], value)
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jp, jr, i)
            reference = self.pin.get_frame_space_jacobian(name)[:, self.to_user]
            errors["jacobian"] = max(
                errors["jacobian"], float(np.max(np.abs(reference - np.vstack([jp, jr])[:, self.dofs])))
            )
        if any(value > 1e-6 for value in errors.values()):
            raise ValueError(f"Independent right-side FK/Jacobian mismatch: {errors}")
        return errors


def set_state(model: mujoco.MjModel, data: mujoco.MjData, record: dict, field: str) -> None:
    """Set joints and object by name, then calculate static contacts without stepping."""
    data.qpos[[model.joint(n).qposadr[0] for n in JOINTS]] = record[field]
    address = model.joint("object_freejoint").qposadr[0]
    data.qpos[address : address + 7] = record["obj_pose"]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Report all penetrating pairs, separating arm, hand, object and table contacts."""
    rows = []
    for contact in data.contact:
        if contact.dist >= 0:
            continue
        bodies = [model.body(model.geom_bodyid[i]).name for i in contact.geom]
        rows.append(
            {
                "bodies": bodies,
                "geoms": [model.geom(i).name for i in contact.geom],
                "penetration_m": float(-contact.dist),
                "involves_arm": any(b.startswith("right_J") or b == "right_base_link" for b in bodies),
                "involves_table": "table" in bodies,
            }
        )
    return {
        "penetrating_pairs": rows,
        "max_penetration_m": max([r["penetration_m"] for r in rows], default=0),
        "arm_table_penetration": any(r["involves_arm"] and r["involves_table"] for r in rows),
    }


def object_bounds(record: dict) -> dict:
    """Apply mesh scale once, then object pose, to the exact collider vertices."""
    import trimesh
    from ada_grasp_ctrl.utils.hand_util import collision_mesh_paths

    vertices = np.concatenate(
        [trimesh.load_mesh(p, process=False).vertices for p in collision_mesh_paths(record["obj_path"])]
    )
    local = vertices * record["obj_scale"]
    transform = pose_matrix(record["obj_pose"])
    world = local @ transform[:3, :3].T + transform[:3, 3]
    return {
        "local_dimensions_m": np.ptp(local, axis=0),
        "world_aabb_m": [world.min(axis=0), world.max(axis=0)],
        "world_dimensions_m": np.ptp(world, axis=0),
        "mesh_offset_applied_m": [0, 0, 0],
        "table_height_m": float(REGISTRATION[2, 3]),
        "object_bottom_to_table_m": float(world[:, 2].min() - REGISTRATION[2, 3]),
    }


def prepare_preview(suite_path: Path, output: Path, sample_id: str | None = None) -> dict:
    """Verify the full suite and prepare deterministic four-scale preview cases.

    Args:
        suite_path: Existing frozen suite manifest; never changed by this operation.
        output: New output directory.
        sample_id: Optional additional case from the same suite; no outcome filtering.

    Returns:
        Complete preview report, including model, inputs, frames and static diagnostics.
    """
    suite = load_suite(suite_path)
    if suite["smoke"] or len(suite["samples"]) != 1000:
        raise ValueError("Preview requires the full frozen 1000-grasp suite")
    selected = [next(r for r in suite["samples"] if r["scale"] == scale) for scale in SCALES]
    if sample_id:
        extra = next((r for r in suite["samples"] if r["sample_id"] == sample_id), None)
        if extra is None:
            raise ValueError(f"Unknown sample ID: {sample_id}")
        if extra not in selected:
            selected.append(extra)
    output.mkdir(parents=True, exist_ok=False)
    provenance_path = project_root() / "assets/robot/crx5ia_sharpa_wave_right/source_manifest.json"
    provenance = json.loads(provenance_path.read_text())
    for entry in provenance["sources"].values():
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Reference source drift: {entry['path']}")
    arm_path = provenance_path.parent / "arm_source.urdf"
    if sha256(arm_path) != provenance["generated_arm_sha256"]:
        raise ValueError("Generated arm source drift")
    info = build_model(suite, output / "model")
    kin = RightKinematics(info)
    initial_q = load_initial_configuration(
        Path(provenance["sources"]["initial_task_config"]["path"]),
        Path(provenance["sources"]["initial_pose_reference"]["path"]),
    )
    for name, value in zip(JOINTS, initial_q):
        lower, upper = kin.model.joint(name).range
        if not lower <= value <= upper:
            raise ValueError(f"Initial config exceeds {name} limits")
    initial_audit = kin.audit(initial_q)
    zero_audit = kin.audit(np.zeros(28))
    kin.forward(np.zeros(28))
    flange_T_palm = np.linalg.solve(body_pose(kin.data, FLANGE), body_pose(kin.data, PALM))
    report = {
        "schema_version": 1,
        "milestone": "preview awaiting user placement confirmation",
        "provenance": provenance,
        "physics_steps": 0,
        "evaluation_episodes": 0,
        "quaternion_order": "WXYZ",
        "units": {"position": "m", "joint": "rad", "angle_residual": "rad"},
        "suite_manifest": str(suite_path),
        "suite_sha256": sha256(suite_path),
        "suite_count": len(suite["samples"]),
        "scales": dict(Counter(str(r["scale"]) for r in suite["samples"])),
        "model": info,
        "registration": REGISTRATION,
        "registration_rule": (
            "fixed translation: place g0000 object center at world [0.8,-0.7,z]; same transform for every sample"
        ),
        "transform_chain": (
            "eval_T_hand = eval_T_source @ source_T_hand; eval_T_object = eval_T_source @ source_T_object"
        ),
        "source_pose_convention": "source-world flange poses; frozen adapter pregrasp extrapolation already applied",
        "flange_T_palm": flange_T_palm,
        "zero_pose_audit": zero_audit,
        "initial_configuration": {
            "joint_names": list(JOINTS),
            "qpos_rad": initial_q,
            "arm_degrees": np.rad2deg(initial_q[:6]),
            "independent_kinematics": initial_audit,
            "source": provenance["sources"]["initial_task_config"],
            "semantics": "configured initial pose and first IK seed; distinct from pregrasp target",
        },
        "tolerances": {
            "ik_position_m": POSITION_TOLERANCE,
            "ik_angle_rad": ANGLE_TOLERANCE,
            "relative_pose": 1e-10,
            "independent_fk_jacobian": 1e-6,
        },
        "samples": [],
    }
    for row in selected:
        source = load_npy_record(row["prepared_path"])
        validate_grasp_record(
            source,
            row["sample_id"],
            expected_joint_dim=28,
            expected_joint_names=SOURCE_JOINTS,
            require_joint_names=True,
            require_assets=True,
        )
        scene = np.load(source["scene_path"], allow_pickle=True).item()
        table = scene["scene"]["table"]
        if table["type"] != "plane" or not np.allclose(pose_matrix(table["pose"]), np.eye(4), atol=1e-12):
            raise ValueError(f"{row['sample_id']}: expected source table plane at source-world z=0")
        record, checks = register_record(source, REGISTRATION)
        record["initial_qpos"] = initial_q.copy()
        seed = initial_q[:6].copy()
        targets = []
        for index, field in enumerate(FIELDS):
            permutation = [source["joint_names"].index(n) for n in HAND_JOINTS]
            q, diagnostic = kin.solve(
                pose_matrix(record["flange_targets_wxyz"][index]), source[field][permutation], seed
            )
            seed = q[:6]
            record[field] = q
            diagnostic["independent_kinematics"] = kin.audit(q)
            diagnostic["joint_limit_violations"] = [
                n
                for n, value in zip(JOINTS, q)
                if value < kin.model.joint(n).range[0] - 1e-8 or value > kin.model.joint(n).range[1] + 1e-8
            ]
            targets.append(diagnostic)
        sample_dir = output / row["sample_id"]
        sample_dir.mkdir()
        prepared_path = sample_dir / "prepared.npy"
        record["registration"] = REGISTRATION.copy()
        record["sample_id"] = row["sample_id"]
        np.save(prepared_path, record)
        scene_path = sample_dir / "scene.xml"
        build_scene(Path(info["mjcf"]), record, REGISTRATION, scene_path)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        set_state(model, data, record, "initial_qpos")
        initial_state = {
            "static_contacts": contacts(model, data),
            "joint_limit_violations": [],
            "actual_flange_wxyz": matrix_pose(body_pose(data, FLANGE)),
            "actual_palm_wxyz": matrix_pose(body_pose(data, PALM)),
        }
        for field, diagnostic in zip(FIELDS, targets):
            set_state(model, data, record, field)
            diagnostic["static_contacts"] = contacts(model, data)
        set_state(model, data, record, FIELDS[0])
        data.qpos[[model.joint(n).qposadr[0] for n in JOINTS]] = 0
        mujoco.mj_forward(model, data)
        item = {
            "sample_id": row["sample_id"],
            "sample_index": row["sample_index"],
            "object_id": row["object_id"],
            "scale": row["scale"],
            "prepared_path": str(prepared_path),
            "scene_path": str(scene_path),
            "source_object_wxyz": source["obj_pose"],
            "evaluation_object_wxyz": record["obj_pose"],
            "source_flange_targets_wxyz": source["flange_targets_wxyz"],
            "evaluation_flange_targets_wxyz": record["flange_targets_wxyz"],
            "relative_pose_checks": checks,
            "geometry": object_bounds(record),
            "targets": targets,
            "initial_state": initial_state,
            "zero_pose_contacts": contacts(model, data),
            "all_targets_reachable": all(d["reachable"] for d in targets),
        }
        report["samples"].append(item)
        print(f"PREPARED {row['sample_id']} scale={row['scale']} reachable={item['all_targets_reachable']}", flush=True)
    paths = [Path(p) for p in info["dependencies"]] + list((output / "model").glob("*"))
    paths.append(provenance_path)
    paths.extend(Path(entry["path"]) for entry in provenance["sources"].values())
    paths.extend(sorted((project_root() / "src/ada_grasp_ctrl/experiments").glob("crx_sharpa_right_*.py")))
    paths.extend(
        project_root() / "script" / name for name in ("preview_crx_sharpa_right.py", "preview_crx_sharpa_right.sh")
    )
    for row in report["samples"]:
        record = np.load(row["prepared_path"], allow_pickle=True).item()
        paths.append(Path(record["obj_path"]) / "info/simplified.json")
        paths.extend([Path(row["prepared_path"]), Path(row["scene_path"])])
    report["dependencies"] = [{"path": str(p), "sha256": sha256(p)} for p in sorted(set(paths))]
    json_write(output / "preview_report.json", report)
    (output / "preview_report.sha256").write_text(sha256(output / "preview_report.json") + "\n")
    (output / "run_manifest.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "static_preview",
                "suite_manifest": str(suite_path),
                "suite_sha256": report["suite_sha256"],
                "output_root": str(output),
                "sample_ids": [r["sample_id"] for r in selected],
                "registration": REGISTRATION.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    return report


def load_preview(output: Path) -> dict:
    """Reject changed report/model/scene/input content before reopening a preview."""
    path = output / "preview_report.json"
    if sha256(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Preview report changed; create a fresh preview")
    report = json.loads(path.read_text())
    suite_path = Path(report["suite_manifest"])
    if sha256(suite_path) != report["suite_sha256"]:
        raise ValueError("Frozen suite changed since preview preparation")
    load_suite(suite_path)
    for item in report["dependencies"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Preview dependency changed: {item['path']}")
    return report
