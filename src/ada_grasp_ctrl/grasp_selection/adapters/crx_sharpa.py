"""Reuse real right CRX preparation, with the measured table and no dynamics."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from ada_grasp_ctrl.experiments.crx_sharpa_right_preparation import (
    ANGLE_TOLERANCE,
    POSITION_TOLERANCE,
    RightKinematics,
    body_pose,
    residual,
)
from ada_grasp_ctrl.experiments.crx_sharpa_right_model import JOINTS, matrix_pose
from ada_grasp_ctrl.experiments.crx_sharpa_right_simulation import (
    ApprovedScene,
    exclude_base_joint_contact,
    invalid_initialization,
)
from ada_grasp_ctrl.paths import resolve_from_root
from .dexlearn import file_hash, read_json, verify_reference
from ..types import (
    ARM_NAMES,
    HAND_NAMES,
    STAGES,
    CandidateEvaluation,
    GraspCandidate,
    IKOption,
    SelectionContext,
    rigid,
    vector,
)

# Same fixed fallback seeds as RightKinematics.solve, invoked separately to retain branches.
FALLBACK_SEEDS = np.deg2rad(
    [
        [0, -30, 130, 0, -60, 0],
        [0, 60, 150, 180, 60, 0],
        [160, 30, 50, 0, 60, 0],
        [160, -30, 130, 0, -60, 0],
        [-160, 30, 50, 0, 60, 0],
    ]
)


class CrxSharpaAdapter:
    """One cached model/table scene; incomplete object geometry remains unverified.

    Args:
        model_report: Existing right-side preview report, used only for model assets.
        table: Measured world plane (normal dot x + d = 0).
        world_T_right_base: Explicit world-from-base transform with external provenance.

    No benchmark registration, initial pose or simulation result is used for ranking.
    """

    def __init__(self, model_report: Path, table: dict, world_T_right_base: object) -> None:
        path = resolve_from_root(model_report)
        report = read_json(path)
        sidecar = path.with_suffix(".sha256")
        if not sidecar.exists() or sidecar.read_text().strip() != file_hash(path):
            raise ValueError("Missing or changed model report checksum")
        self.info = report["model"]
        if self.info["joint_names"] != list(ARM_NAMES + HAND_NAMES) or tuple(JOINTS) != ARM_NAMES + HAND_NAMES:
            raise ValueError("Expected real right CRX + canonical Sharpa, not dummy arm")
        dependencies = {r["path"]: r for r in report["dependencies"]}
        verified = []
        for name in sorted(set(self.info["dependencies"] + [self.info["mjcf"], self.info["urdf"]])):
            if name not in dependencies:
                raise ValueError("Unhashed model dependency: " + name)
            verify_reference(dependencies[name])
            verified.append(dependencies[name])
        self.model_id = file_hash(Path(self.info["urdf"]))
        self.world_T_base = rigid(world_T_right_base, "world_T_right_base")
        self.model_T_base = rigid(self.info["world_T_base"], "model.world_T_base")
        self.model_T_world = self.model_T_base @ np.linalg.inv(self.world_T_base)
        self.kin = RightKinematics(self.info)
        if self.kin.model.nq != 28 or not all(self.kin.model.joint(n).limited for n in JOINTS):
            raise ValueError("Expected 28 limited revolute joints")
        self.kin.forward(np.zeros(28))
        # Existing exported MJCF rounds origins; use the existing FK audit tolerance (1e-6).
        if not np.allclose(body_pose(self.kin.data, "right_base_link"), self.model_T_base, atol=1e-6, rtol=0):
            raise ValueError("Model base transform differs from report")
        if table.get("frame_id") != "world" or table.get("unit") != "m":
            raise ValueError("Table requires world frame and meters")
        normal = vector(table["normal_xyz"], 3, "table.normal_xyz")
        if abs(np.linalg.norm(normal) - 1) > 1e-6 or not np.isfinite(table["d"]):
            raise ValueError("Invalid normalized table plane")
        table_world = np.eye(4)
        table_world[:3, 3] = -float(table["d"]) * normal
        axis = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = axis - np.dot(axis, normal) * normal
        x /= np.linalg.norm(x)
        table_world[:3, :3] = np.column_stack([x, np.cross(normal, x), normal])
        self.table_model = self.model_T_world @ table_world
        pose = matrix_pose(self.table_model)
        # The existing solver consults ik_table when choosing among its fallback attempts.
        self.kin.model.geom("ik_table").pos[:] = pose[:3]
        self.kin.model.geom("ik_table").quat[:] = pose[3:]
        spec = mujoco.MjSpec.from_file(self.info["mjcf"])
        exclude_base_joint_contact(spec)
        body = spec.worldbody.add_body(name="table", pos=pose[:3], quat=pose[3:])
        body.add_geom(name="selector_table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.addresses = [self.model.joint(n).qposadr[0] for n in JOINTS]
        self.limits = np.array([self.model.joint(n).range for n in JOINTS])
        ids = {self.model.body("right_hand_flange").id}
        for i in range(1, self.model.nbody):
            if self.model.body_parentid[i] in ids:
                ids.add(i)
        self.hand_bodies = {self.model.body(i).name for i in ids}
        self.provenance = {
            "model_report": str(path),
            "model_report_sha256": file_hash(path),
            "model_id": self.model_id,
            "dependencies": verified,
            "model_T_base": self.model_T_base.tolist(),
            "world_T_right_base": self.world_T_base.tolist(),
            "model_T_world": self.model_T_world.tolist(),
            "table": table,
            "mount_calibrated": self.info["mount_calibrated"],
            "ik_position_tolerance_m": POSITION_TOLERANCE,
            "ik_angle_tolerance_rad": ANGLE_TOLERANCE,
            "arm_penetration_tolerance_m": 1e-5,
            "hand_penetration_tolerance_m": 0.01,
            "exclusion": ["right_base_link", "right_J1_link"],
            "physics_steps": 0,
            "object_collision": "unverified: no complete collision solid supplied",
            "scene_complete": False,
            "branch_subset": "q_current plus five fixed seeds; max six returned grasp branches",
            "stage_policy": "all three endpoints; no branch-continuous or continuous-path claim",
            "transform_chain": (
                "base_T_flange=inv(world_T_base)@world_T_flange; model_T_flange=model_T_base@base_T_flange"
            ),
        }

    def validate_context(self, context: SelectionContext) -> None:
        """Reject stale/mismatched model coordinates and invalid current joint limits."""
        if context.robot_state.model_id != self.model_id:
            raise ValueError("State/model identity mismatch")
        if not np.allclose(context.world_T_right_base, self.world_T_base, atol=1e-10, rtol=0):
            raise ValueError("Context/adapter base transform mismatch")
        q = context.robot_state.positions_rad
        if np.any(q < self.limits[:6, 0]) or np.any(q > self.limits[:6, 1]):
            raise ValueError("Current right-arm state exceeds model limits")
        if context.scene_complete:
            raise ValueError("This measured-surface adapter has no complete object/environment scene")

    def static_check(self, q: np.ndarray) -> dict:
        """Reuse right-side contact classification and penetration gates without stepping."""
        self.data.qpos[self.addresses] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        view = SimpleNamespace(model=self.model, data=self.data, hand_bodies=self.hand_bodies)
        diagnostic = ApprovedScene.contact_diagnostics(view)
        return {
            **diagnostic,
            "known_geometry_passed": not invalid_initialization(diagnostic),
            "object_environment": "unverified",
            "complete": False,
        }

    def check_solution(self, q: np.ndarray, target: np.ndarray, diagnostic: dict) -> tuple[dict, list[str]]:
        """Independently check returned FK, limits, and static contacts against exact targets."""
        q = vector(q, 28, "IK full configuration")
        actual = self.kin.forward(q)
        errors = residual(actual, target)
        limits = [
            name for name, v, bounds in zip(JOINTS, q, self.limits) if v < bounds[0] - 1e-8 or v > bounds[1] + 1e-8
        ]
        collision = self.static_check(q)
        reasons = []
        if not diagnostic["reachable"] or not diagnostic["solver_success"]:
            reasons.append("ik_failed")
        if errors["position_m"] > POSITION_TOLERANCE or errors["orientation_rad"] > ANGLE_TOLERANCE:
            reasons.append("ik_residual")
        if limits:
            reasons.append("joint_limits")
        if not collision["known_geometry_passed"]:
            reasons.append("static_collision")
        return {
            "q_rad": q,
            "ik": diagnostic,
            "residual": errors,
            "joint_limit_violations": limits,
            "collision": collision,
            "passed": not reasons,
            "reasons": reasons,
        }, reasons

    def evaluate(self, candidate: GraspCandidate, context: SelectionContext, evaluation: CandidateEvaluation) -> None:
        """Retain bounded distinct grasp branches and validate each original stage."""
        evaluation.warnings.extend(["object_environment_collision_unverified", "endpoint_checks_only"])
        targets = {s: self.model_T_world @ candidate.stages[s].transform() for s in STAGES}
        hand = {s: candidate.stages[s].joint_positions_rad for s in STAGES}
        violations = {
            s: [n for n, v, b in zip(HAND_NAMES, hand[s], self.limits[6:]) if v < b[0] - 1e-8 or v > b[1] + 1e-8]
            for s in STAGES
        }
        evaluation.diagnostics["hand_joint_limits"] = violations
        if any(violations.values()):
            evaluation.reasons.append("hand_joint_limits")
            evaluation.diagnostics["stages"] = {
                s: {"status": "not_checked", "reason": "hand_joint_limits", "violations": violations[s]} for s in STAGES
            }
            return
        seeds = [context.robot_state.positions_rad, *FALLBACK_SEEDS]
        evaluation.diagnostics["outer_seeds_rad"] = seeds
        evaluation.diagnostics["grasp_attempts"] = []
        retained = []
        for index, seed in enumerate(seeds):
            q, diag = self.kin.solve(targets["grasp"], hand["grasp"], seed)
            check, reasons = self.check_solution(q, targets["grasp"], diag)
            evaluation.diagnostics["grasp_attempts"].append({"solution_index": index, **check})
            if any(np.allclose(q[:6], prev, atol=1e-6, rtol=0) for prev in retained):
                continue
            retained.append(q[:6].copy())
            option = IKOption(index, q[:6], {"grasp": check}, ["grasp:" + r for r in reasons])
            evaluation.options.append(option)
            if reasons:
                for stage in ("pregrasp", "squeeze"):
                    option.stages[stage] = {"status": "not_checked", "reason": "grasp_endpoint_failed"}
                continue
            for stage in ("pregrasp", "squeeze"):
                stage_q, stage_diag = self.kin.solve(targets[stage], hand[stage], q[:6])
                stage_check, stage_reasons = self.check_solution(stage_q, targets[stage], stage_diag)
                option.stages[stage] = stage_check
                option.reasons.extend(stage + ":" + r for r in stage_reasons)
            if not option.reasons:
                option.stages["grasp"]["independent_fk_audit"] = self.kin.audit(q)
