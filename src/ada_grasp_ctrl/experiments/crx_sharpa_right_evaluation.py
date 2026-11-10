"""Paired native op/ours smoke evaluation of the approved right CRX preview."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import numpy as np
from omegaconf import OmegaConf, open_dict

from ada_grasp_ctrl.errors import ControlSolveEpisodeAbort
from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.runtime import seed_sample
from ada_grasp_ctrl.schema import load_npy_record, validate_grasp_record
from ada_grasp_ctrl.tasks.control_eval_func.base import BaseEval
from ada_grasp_ctrl.tasks.control_eval_func.optimized_runner import WrenchControlPolicy, run_wrench_control_episode
from ada_grasp_ctrl.tasks.control_eval_func.tabletop_dummy_arm_op import tabletopDummyArmOpEval
from .crx_sharpa_right_approach import (
    HEIGHT_M,
    INITIALIZATION_SEGMENTS,
    actual_palm_error,
    prepare_approach,
)
from .crx_sharpa_right_model import JOINTS, matrix_pose
from .crx_sharpa_right_preparation import RightKinematics, json_write, load_preview
from .crx_sharpa_right_simulation import ApprovedScene, RightCrxSharpa, invalid_initialization
from .episode_recording import EpisodeRecorder, load_episode
from .hugs_sharpa_input import FIELDS, SCALES
from .phase_comparison import sha256
from .sharpa_regression import phase_config

STATUSES = ("success", "physical_failure", "ik_invalid", "initialization_invalid", "solver_degraded", "execution_error")
PROTOCOL = {
    "name": "crx-sharpa-right-smoke-v3-overhead-30cm",
    "collision_exclusions_added": [["right_base_link", "right_J1_link"]],
    "collision_fix": "Exclude adjacent bearing surfaces; retain all arm/environment contacts",
    "evaluation_seed": 12,
    "quaternion_order": "WXYZ",
    "friction": [0.8, 0.02],
    "object_mass_setting_kg": 0.1,
    "object_downward_force_n": 1.0,
    "global_gravity": "disabled; native benchmark emulation",
    "hand_max_penetration_m": 0.01,
    "arm_clearance_tolerance_m": 1e-5,
    "initial_force_gate": False,
    "final_sum_force_n": 15,
    "control_frequency_hz": 10,
    "physics_timestep_s": 0.004,
    "offsets_m": [0],
    "initialization": "config open hand; raise, move above object, align, then descend to frozen pregrasp",
    "overhead_reference": "right_hand_C_MC (palm) above registered object pose center",
    "overhead_height_m": HEIGHT_M,
    "approach_linear_speed_m_s": 0.1,
    "approach_angular_speed_rad_s": 0.5,
    "approach_settle_timeout_s": 3.0,
    "approach_arrival_position_tolerance_m": 0.001,
    "approach_arrival_angle_tolerance_rad": 0.01,
    "home_path_precheck": "Cartesian IK action targets plus 5 joint interpolation substeps; sampled collision checks",
    "control": "native op and coordinated ours; native 2 s approach/2 s squeeze references and ours stop policy",
    "lift": "20 Cartesian Z targets over 2 s, total 0.2 m, bounded continuous-seed flange IK",
    "lift_branch_step_limit_rad": 0.35,
    "solver_failure_policy": "apply_candidate",
    "success_rule": "abs((object_z_at_control_start + 0.2) - terminal_object_z) < 0.1",
    "differences_from_dummy": [
        "real CRX joints, limits, gains and bounded IK",
        "config-to-overhead-to-pregrasp motion with actual arrival checks",
        "arm/table collisions retained and recorded",
        "Cartesian lift trajectory",
    ],
    "status_precedence": [
        "execution_error",
        "initialization_invalid",
        "solver_degraded",
        "ik_invalid",
        "success or physical_failure after completed lift",
    ],
}


class ApproachInvalidError(RuntimeError):
    """The shared initialization path collided or failed to reach a waypoint."""


class LiftIKError(RuntimeError):
    """A Cartesian lift waypoint could not be executed within the joint/pose contract."""


class RightEvaluator(BaseEval):
    """Use native policy/controller hooks with the exact scene and explicit experiment lifecycle."""

    def __init__(self, row: dict, report: dict, config: Any, method: str) -> None:
        self.input_npy_path = row["prepared_path"]
        self.grasp_data = validate_grasp_record(
            load_npy_record(row["prepared_path"]),
            row["sample_id"],
            expected_joint_dim=28,
            expected_joint_names=JOINTS,
            require_joint_names=True,
            require_assets=True,
        )
        self.configs = config
        self.robot = RightCrxSharpa(report["model"])
        self.mj_ho = ApprovedScene(Path(row["scene_path"]))
        self._initialize_controller(method)
        self.ctrl_freq, self.action_dt, self.sim_step_per_action = 10, 0.1, 25
        self.episode_recorder = EpisodeRecorder(
            self.mj_ho,
            {
                **{field: self.grasp_data[field].copy() for field in FIELDS},
                "initial_qpos": self.grasp_data["initial_qpos"].copy(),
            },
        )
        self.lift_diagnostics: list[dict] = []
        self.approach_diagnostics: list[dict] = []

    def motion(self, path: np.ndarray, segment: str) -> None:
        """Execute reference positions through the native actuator interpolation API."""
        self.mj_ho.segment = segment
        for target in path:
            self.mj_ho.ctrl_qpos_a_with_interp(
                self.mj_ho.get_qpos_a(), target, self.robot.doa_names, step_outer=5, step_inner=5
            )
            self.episode_recorder.capture(segment, reference=target, command=target)

    def overhead_approach(self, plan: dict) -> None:
        """Execute the shared overhead route, requiring measured arrival before each handoff."""
        sim = self.mj_ho
        for segment in plan["segments"]:
            name = segment["name"]
            allow_contact = name == "descend_to_pregrasp"

            def check_contacts() -> None:
                contacts = sim.contact_diagnostics()
                if (
                    invalid_initialization(contacts)
                    or sim.collision_history
                    or (not allow_contact and sim.get_curr_contact_info())
                ):
                    raise ApproachInvalidError(f"Unexpected contact during {name}: {contacts}")

            for target in segment["path"]:
                self.motion(np.asarray([target]), name)
                check_contacts()
            diagnostic = actual_palm_error(sim, segment["target_palm_wxyz"])
            settled = 0
            while (diagnostic["position_m"] > 0.001 or diagnostic["orientation_rad"] > 0.01) and settled < 30:
                self.motion(np.asarray([segment["path"][-1]]), f"settle_{name}")
                check_contacts()
                settled += 1
                diagnostic = actual_palm_error(sim, segment["target_palm_wxyz"])
            diagnostic.update(segment=name, target_palm_wxyz=segment["target_palm_wxyz"], settle_steps=settled)
            self.approach_diagnostics.append(diagnostic)
            if diagnostic["position_m"] > 0.001 or diagnostic["orientation_rad"] > 0.01:
                raise ApproachInvalidError(f"Did not reach {name} within 3 s settling: {diagnostic}")

    def native_grasp(self) -> None:
        """Dispatch only the policy difference; both methods use the native shared runner."""
        self.mj_ho.segment = "control"
        values = [self.grasp_data[field] for field in FIELDS]
        if self.method_name == "op":
            tabletopDummyArmOpEval._simulate_under_extforce_details(self, *values)
        else:
            run_wrench_control_episode(self, *values, WrenchControlPolicy(use_approaching_arm_motion=True))

    def cartesian_lift(self, kinematics: RightKinematics) -> None:
        """Raise actual flange in Cartesian Z, retaining fingers and recording all IK decisions."""
        actual = self.mj_ho.get_qpos_f(JOINTS)
        start = kinematics.forward(actual)
        seed = actual[:6].copy()
        hand_command = self.mj_ho.get_qpos_a()[6:].copy()
        for step in range(1, 21):
            target = start.copy()
            target[2, 3] += 0.2 * step / 20
            q, diagnostic = kinematics.solve(target, hand_command, seed)
            jump = float(np.max(np.abs(q[:6] - seed)))
            diagnostic.update(target_wxyz=matrix_pose(target), joint_step_rad=jump)
            self.lift_diagnostics.append(diagnostic)
            if not diagnostic["reachable"] or not diagnostic["arm_table_clear"] or jump > 0.35:
                raise LiftIKError(
                    f"Lift waypoint {step}: reachable={diagnostic['reachable']}, "
                    f"arm_table_clear={diagnostic['arm_table_clear']}, joint_step={jump:.6g}"
                )
            self.grasp_ctrl.r_data["obj_pose"].append(self.mj_ho.get_obj_pose())
            self.motion(np.asarray([q]), "lift")
            seed = q[:6].copy()


def make_config(report: dict, row: dict, output: Path, method: str) -> Any:
    """Reuse the frozen Sharpa scientific settings without registering a new public hand."""
    suite = json.loads(Path(report["suite_manifest"]).read_text())
    config = phase_config(suite, "phase1" if method == "op" else "phase2", output, 1)
    with open_dict(config):
        config.hand_name = "crx_sharpa_right_experiment"
        config.hand.xml_path = report["model"]["mjcf"]
        config.hand.dummy_arm = False
        config.grasp_dir = str(Path(row["prepared_path"]).parent)
    return config


def sample_preflight(row: dict, record: dict, model_info: dict) -> dict:
    """Retain all sample IDs and check the shared overhead route before physics."""
    if not row["all_targets_reachable"]:
        return {"status": "ik_invalid", "reason": "One or more frozen grasp targets failed IK residual tolerance"}
    if any(t["joint_limit_violations"] for t in row["targets"]):
        return {"status": "initialization_invalid", "reason": "A target violates named joint limits"}
    if any(not t["arm_table_clear"] for t in row["targets"]):
        return {"status": "initialization_invalid", "reason": "A frozen target has arm/table penetration"}
    return prepare_approach(row, record, model_info)


def classify(
    *,
    error: str | None,
    invalid: bool,
    degraded: bool,
    lift_error: bool,
    lift_completed: bool,
    initial_z: float | None,
    terminal_z: float | None,
) -> str:
    """Apply mutually exclusive precedence while retaining physical outcome separately."""
    if error:
        return "execution_error"
    if invalid:
        return "initialization_invalid"
    if degraded:
        return "solver_degraded"
    if lift_error:
        return "ik_invalid"
    if not lift_completed or initial_z is None or terminal_z is None:
        return "execution_error"
    return "success" if abs(initial_z + 0.2 - terminal_z) < 0.1 else "physical_failure"


def run_episode(row: dict, report: dict, output: Path, method: str, preflight: dict) -> dict:
    """Execute one attributable method/sample pair, saving partial records on every runtime failure."""
    episode_dir = output / row["sample_id"] / method
    episode_dir.mkdir(parents=True, exist_ok=False)
    seed = seed_sample(12, row["sample_index"])
    result = {
        "sample_id": row["sample_id"],
        "sample_index": row["sample_index"],
        "method": method,
        "scale": row["scale"],
        "object_id": row["object_id"],
        "seed": seed,
        "preflight": preflight,
        "status": preflight["status"],
        "physics_steps": 0,
        "joint_names": list(JOINTS),
        "quaternion_order": "WXYZ",
        "control_steps": 0,
    }
    if preflight["status"] != "ready":
        result["reason"] = preflight["reason"]
        json_write(episode_dir / "result.json", result)
        return result
    evaluator = None
    started = time.monotonic()
    error = None
    invalid = lift_error = lift_completed = False
    initial_z = terminal_z = None
    try:
        config = make_config(report, row, episode_dir, method)
        OmegaConf.save(config, episode_dir / "run_manifest.yaml", resolve=True)
        evaluator = RightEvaluator(row, report, config, method)
        seed_sample(12, row["sample_index"])
        sim, record = evaluator.mj_ho, evaluator.grasp_data
        sim.reset_pose_qpos(record["initial_qpos"], record["obj_pose"])
        sim.set_ext_force_on_obj(np.array([0, 0, -1.0, 0, 0, 0]))
        evaluator.episode_recorder.capture("initial_config")
        evaluator.episode_recorder.metadata.update(
            sample_id=row["sample_id"], seed=seed, method=method, reference_joint_names=list(JOINTS), protocol=PROTOCOL
        )
        evaluator.episode_recorder.targets["overhead_palm_wxyz"] = np.asarray(preflight["overhead_palm_wxyz"])
        evaluator.overhead_approach(preflight)
        diag = sim.contact_diagnostics()
        invalid = invalid_initialization(diag) or bool(sim.collision_history)
        result["post_approach_contacts"] = diag
        evaluator.episode_recorder.capture("control_initial")
        if invalid:
            result["reason"] = "Home-to-pregrasp dynamic initialization exceeded collision/penetration tolerance"
        else:
            initial_z = float(sim.get_obj_pose()[2])
            evaluator.native_grasp()
            evaluator.cartesian_lift(RightKinematics(report["model"]))
            lift_completed = True
    except ApproachInvalidError as exc:
        invalid = True
        result["reason"] = str(exc)
    except ControlSolveEpisodeAbort as exc:
        result["reason"] = str(exc)
    except LiftIKError as exc:
        lift_error = True
        result["reason"] = str(exc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result.update(reason=error, traceback=traceback.format_exc())
    finally:
        if evaluator is not None:
            try:
                sim = evaluator.mj_ho
                terminal_z = float(sim.get_obj_pose()[2])
                degraded = bool(evaluator.grasp_ctrl.solver_degraded)
                result["status"] = classify(
                    error=error,
                    invalid=invalid,
                    degraded=degraded,
                    lift_error=lift_error,
                    lift_completed=lift_completed,
                    initial_z=initial_z,
                    terminal_z=terminal_z,
                )
                if not invalid:
                    evaluator.grasp_ctrl.r_data["obj_pose"].append(sim.get_obj_pose())
                native_status = {
                    "initialization_invalid": "invalid_initialization",
                    "solver_degraded": "solver_degraded",
                    "success": "completed",
                    "physical_failure": "completed",
                }.get(result["status"], "execution_error")
                control_path = episode_dir / "control.npy"
                evaluator.grasp_ctrl.save_recorded_data(str(control_path), episode_status=native_status)
                evaluator.episode_recorder.capture("terminal")
                evaluator.episode_recorder.metadata.update(lift_completed=lift_completed, ik_lift_error=lift_error)
                evaluator.episode_recorder.save(control_path, result["status"])
                r_data = evaluator.grasp_ctrl.r_data
                adaptive_steps = sum(
                    bool(c) and np.linalg.norm(np.asarray(opt["q_a"]) - target) > 1e-4
                    for c, opt, target in zip(r_data["contacts"], r_data["opt_res"], r_data["planned_dof"])
                )
                result.update(
                    physics_steps=sim.physics_steps,
                    control_steps=len(r_data["dof"]),
                    contact_steps=sum(bool(c) for c in r_data["contacts"]),
                    adaptive_contact_steps=int(adaptive_steps),
                    lift_completed=lift_completed,
                    lift_diagnostics=evaluator.lift_diagnostics,
                    approach_diagnostics=evaluator.approach_diagnostics,
                    object_z_at_control_start=initial_z,
                    terminal_object_z=terminal_z,
                    object_lift_delta_m=None if initial_z is None else terminal_z - initial_z,
                    physical_lift_pass=None if not lift_completed else abs(initial_z + 0.2 - terminal_z) < 0.1,
                    solver_degraded=degraded,
                    collision_events=len(sim.collision_history),
                    control_path=str(control_path),
                    replay_path=str(control_path.with_suffix(".replay.npz")),
                    evaluated_scene_sha256=sha256(control_path.with_suffix(".scene.xml")),
                )
                json_write(episode_dir / "collisions.json", sim.collision_history)
                result["arm_tracking"] = tracking_diagnostics(evaluator.episode_recorder.frames, sim.q_addresses[:6])
            except Exception as exc:
                result.update(
                    status="execution_error",
                    reason=f"Recording failed: {type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                    physics_steps=evaluator.mj_ho.physics_steps,
                )
            finally:
                evaluator.mj_ho.close_view_and_render()
        else:
            result["status"] = "execution_error"
        result["elapsed_seconds"] = time.monotonic() - started
        json_write(episode_dir / "result.json", result)
    return result


def tracking_diagnostics(frames: list[dict], addresses: list[int]) -> dict:
    """Report measured arm tracking without converting tracking errors into grasp success."""
    diagnostics = {}
    for segment in ("home_to_pregrasp", *sorted(INITIALIZATION_SEGMENTS), "approach", "squeeze", "lift"):
        selected = [f for f in frames if f["segment"] == segment and f["command"] is not None]
        if selected:
            errors = [np.max(np.abs(f["qpos"][addresses] - f["command"][:6])) for f in selected]
            diagnostics[segment] = {"maximum_error_rad": float(max(errors)), "terminal_error_rad": float(errors[-1])}
    return diagnostics


def verify_pairs(results: list[dict]) -> dict:
    """Check matching seeds and full initial/control-start states, including home approach frames."""
    by_sample = defaultdict(dict)
    for row in results:
        by_sample[row["sample_id"]][row["method"]] = row
    checked, invalid_pairs = [], []
    for sample_id, pair in by_sample.items():
        a, b = pair["op"], pair["ours"]
        if a["seed"] != b["seed"] or a["preflight"] != b["preflight"]:
            raise ValueError(f"Pair seed/preflight mismatch: {sample_id}")
        if "replay_path" not in a or "replay_path" not in b:
            invalid_pairs.append(sample_id)
            continue
        records = [load_episode(Path(r["replay_path"])) for r in (a, b)]
        frames = [
            [
                f
                for f in r["frames"]
                if f["segment"] in {"initial_config", "home_to_pregrasp", "control_initial"} | INITIALIZATION_SEGMENTS
            ]
            for r in records
        ]
        if len(frames[0]) != len(frames[1]):
            raise ValueError(f"Pair initialization length mismatch: {sample_id}")
        for first, second in zip(*frames):
            if first["segment"] != second["segment"] or first["time"] != second["time"]:
                raise ValueError(f"Pair initialization stage/time mismatch: {sample_id}")
            for field in ("qpos", "qvel", "ctrl", "act", "object_pose"):
                np.testing.assert_array_equal(first[field], second[field], err_msg=f"{sample_id}: {field}")
        for field in (*FIELDS, "initial_qpos", "overhead_palm_wxyz"):
            np.testing.assert_array_equal(records[0]["targets"][field], records[1]["targets"][field])
        checked.append(sample_id)
    return {"identical_initial_state_pairs": checked, "pairs_rejected_before_physics": invalid_pairs}


def run_smoke(preview_root: Path, output: Path, *, prepare_only: bool = False) -> int:
    """Validate the approved preview and produce all eight expected sample/method statuses."""
    report = load_preview(preview_root)
    rows = [next(row for row in report["samples"] if row["scale"] == scale) for scale in SCALES]
    if len({r["sample_id"] for r in rows}) != 4:
        raise ValueError("Expected four distinct fixed-scale samples")
    # Fail shared controller/environment setup before any episode output is created.
    for method in ("op", "ours"):
        probe = RightEvaluator(rows[0], report, make_config(report, rows[0], output, method), method)
        probe.mj_ho.close_view_and_render()
    # Verify shared kinematics and sample inputs before creating episode outputs.
    RightKinematics(report["model"])
    records = {
        row["sample_id"]: validate_grasp_record(
            load_npy_record(row["prepared_path"]),
            row["sample_id"],
            expected_joint_dim=28,
            expected_joint_names=JOINTS,
            require_joint_names=True,
            require_assets=True,
        )
        for row in rows
    }
    preflight = {row["sample_id"]: sample_preflight(row, records[row["sample_id"]], report["model"]) for row in rows}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "mode": "prepare_only" if prepare_only else "smoke",
        "protocol": PROTOCOL,
        "preview_root": str(preview_root),
        "preview_sha256": sha256(preview_root / "preview_report.json"),
        "suite_sha256": report["suite_sha256"],
        "sample_ids": [r["sample_id"] for r in rows],
        "expected_pairs": 8,
        "workers": 1,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root(), text=True).strip(),
        "model": report["model"],
        "registration": report["registration"],
        "initial_configuration": report["initial_configuration"],
    }
    json_write(output / "run_manifest.yaml", manifest)
    json_write(output / "preflight.json", preflight)
    code_files = [
        project_root() / "script/test_crx_sharpa_right_controller.sh",
        project_root() / "script/test_crx_sharpa_right_controller.py",
    ]
    code_files += sorted((project_root() / "src/ada_grasp_ctrl").rglob("*.py"))
    json_write(output / "code_state.json", {str(p.relative_to(project_root())): sha256(p) for p in code_files})
    if prepare_only:
        json_write(
            output / "run_report.json",
            {"mode": "prepare_only", "expected_pairs": 8, "executed_pairs": 0, "preflight": preflight},
        )
        return 0
    results = []
    for row in rows:
        for method in ("op", "ours"):
            result = run_episode(row, report, output, method, preflight[row["sample_id"]])
            results.append(result)
            json_write(output / "results.json", results)
            print(
                f"{row['sample_id']} {method}: {result['status']} ({result['physics_steps']} physics steps)", flush=True
            )
    pair_check = verify_pairs(results)
    counts = {method: dict(Counter(r["status"] for r in results if r["method"] == method)) for method in ("op", "ours")}
    summary = {
        "schema_version": 1,
        "mode": "smoke",
        "expected_pairs": 8,
        "recorded_pairs": len(results),
        "missing": 8 - len(results),
        "executed_pairs": sum(r["physics_steps"] > 0 for r in results),
        "counts": counts,
        "paired_initialization": pair_check,
        "adaptive_contact_response_verified": any(
            r["method"] == "ours" and r.get("adaptive_contact_steps", 0) > 0 for r in results
        ),
        "ik_reachable_samples": sum(row["all_targets_reachable"] for row in rows),
        "preflight_ready_samples": sum(p["status"] == "ready" for p in preflight.values()),
        "all_physics_completed": all(r.get("lift_completed", False) for r in results),
        "results": results,
    }
    summary["smoke_acceptance_passed"] = (
        summary["all_physics_completed"]
        and summary["adaptive_contact_response_verified"]
        and not any(r["status"] in ("execution_error", "solver_degraded") for r in results)
    )
    for method in counts:
        success = counts[method].get("success", 0)
        valid = success + counts[method].get("physical_failure", 0)
        summary.setdefault("rates", {})[method] = {
            "success_over_selected": success / 4,
            "success_over_success_plus_physical_failure": success / valid if valid else None,
        }
    json_write(output / "run_report.json", summary)
    with (output / "failures.jsonl").open("w") as stream:
        for row in results:
            if row["status"] in ("execution_error", "solver_degraded"):
                stream.write(
                    json.dumps({k: row[k] for k in ("sample_id", "method", "status", "reason") if k in row}) + "\n"
                )
    (output / "summary.md").write_text(
        "# Right CRX / right Sharpa smoke\n\n"
        + f"Expected/recorded pairs: 8/{len(results)}; physics executed: {summary['executed_pairs']}.\n\n"
        + f"Counts: `{counts}`.\n\n"
        + f"Adaptive contact response verified: {summary['adaptive_contact_response_verified']}.\n\n"
        + "This four-scale feasibility test does not estimate full-suite performance. "
        + "See run_report.json for every status.\n"
    )
    return 1 if any(r["status"] in ("execution_error", "solver_degraded") for r in results) else 0
