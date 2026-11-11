"""Measured-start, mesh-based tabletop clearance evaluation without grasp inference."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import select
import sys
import time

import numpy as np
import yaml

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, kinematic_contract
from ada_grasp_ctrl.paths import resolve_from_root
from ada_grasp_ctrl.ros_execution.motion import smoothstep
from ada_grasp_ctrl.ros_execution.settings import bridge_config, validate_domain


def save(path: Path, value: dict) -> None:
    """Persist finite diagnostics and plans."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    """Bind execution to the exact reviewed files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vertical_gaps(points: np.ndarray, plane: dict) -> np.ndarray:
    """Measure world-Z gaps to a tilted plane at each mesh point's XY."""
    normal = np.asarray(plane["normal_xyz"], dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or abs(np.linalg.norm(normal) - 1) > 1e-6:
        raise ValueError("Expected a finite unit table normal")
    if normal[2] < 0.9 or not np.isfinite(plane["d"]):
        raise ValueError("Table must face world +Z and be nearly horizontal")
    return (np.asarray(points) @ normal + plane["d"]) / normal[2]


class HandGeometry:
    """Use every visual and collision mesh on the hand-flange descendant bodies."""

    def __init__(self, adapter: object, plane: dict) -> None:
        import mujoco

        self.adapter, self.plane = adapter, plane
        self.world_T_model = np.linalg.inv(adapter.model_T_world)
        self.meshes = []
        model = adapter.model
        for geom in range(model.ngeom):
            if model.body(int(model.geom_bodyid[geom])).name not in adapter.hand_bodies:
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("Unsupported hand geometry; cannot establish whole-hand clearance")
            mesh = model.geom_dataid[geom]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            self.meshes.append((geom, model.mesh_vert[first : first + count].copy()))
        if not self.meshes:
            raise ValueError("No hand meshes found")

    def measure(self, q: np.ndarray) -> dict:
        """Return the lowest vertical mesh-to-table gap and its world point."""
        import mujoco

        a = self.adapter
        a.data.qpos[a.addresses] = q
        mujoco.mj_kinematics(a.model, a.data)
        result = None
        for geom, local in self.meshes:
            points = local @ a.data.geom_xmat[geom].reshape(3, 3).T + a.data.geom_xpos[geom]
            points = points @ self.world_T_model[:3, :3].T + self.world_T_model[:3, 3]
            gaps = vertical_gaps(points, self.plane)
            i = int(np.argmin(gaps))
            if result is None or gaps[i] < result["gap_m"]:
                result = dict(
                    gap_m=float(gaps[i]),
                    point_world_m=points[i].tolist(),
                    body=a.model.body(int(a.model.geom_bodyid[geom])).name,
                )
        return result


def make_plan(adapter: object, state: dict, plane: dict, clearance: float, environment: str) -> dict:
    """Solve local vertical waypoints and check sampled interpolation against known geometry."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    if not np.isfinite(clearance) or not 0.02 <= clearance <= 0.30:
        raise ValueError("Clearance must be 0.02..0.30 meters")
    start = np.asarray(state["full_positions_rad"], dtype=float)
    if state["full_joint_names"] != list(JOINT_NAMES) or start.shape != (28,) or not np.isfinite(start).all():
        raise ValueError("Invalid measured joint state")
    if state["model_id"] != adapter.model_id or state["environment"] != environment:
        raise ValueError("State model/environment mismatch")
    if np.any(start < adapter.limits[:, 0]) or np.any(start > adapter.limits[:, 1]):
        raise ValueError("Measured joints outside model limits")
    geometry = HandGeometry(adapter, plane)
    initial = geometry.measure(start)
    if initial["gap_m"] < 0.02:
        raise ValueError("Current hand is within 2 cm of/below the table; this tool is not a recovery motion")
    displacement = clearance - initial["gap_m"]
    if abs(displacement) > 0.30:
        raise ValueError("Requested vertical move exceeds the evaluator's 30 cm bound")
    flange = adapter.kin.forward(start).copy()
    keyframes = [start]
    count = max(1, int(np.ceil(abs(displacement) / 0.01)))
    for z in np.linspace(0, displacement, count + 1)[1:]:
        target = flange.copy()
        target[:3, 3] += adapter.model_T_world[:3, :3] @ [0, 0, z]

        def error(arm: np.ndarray) -> np.ndarray:
            actual = adapter.kin.forward(np.r_[arm, start[6:]])
            return np.r_[
                10 * (actual[:3, 3] - target[:3, 3]),
                Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).as_rotvec(),
            ]

        fit = least_squares(
            error, keyframes[-1][:6], bounds=adapter.kin.bounds, max_nfev=160, ftol=1e-11, xtol=1e-11, gtol=1e-11
        )
        e = error(fit.x)
        if not fit.success or np.linalg.norm(e[:3]) > 0.001 or np.linalg.norm(e[3:]) > 0.001:
            raise ValueError("Local vertical IK did not converge; no alternative branch will be executed")
        q = np.r_[fit.x, start[6:]]
        if np.max(np.abs(q[:6] - keyframes[-1][:6])) > 0.15:
            raise ValueError("IK branch jump exceeds 0.15 rad per waypoint")
        keyframes.append(q)
    settings = bridge_config({"environment": environment})
    durations, minimum, samples = [], initial["gap_m"], 0
    inverse = np.linalg.inv(adapter.model_T_world)
    flange_world = inverse @ flange
    for a, b in zip(keyframes, keyframes[1:]):
        delta = float(np.max(np.abs(b[:6] - a[:6])))
        # Lower than the existing receiver limits; the limits themselves remain unchanged.
        durations.append(max(0.5, 1.875 * delta / 0.05, float(np.sqrt(5.774 * delta / 0.1))))
        for u in np.linspace(0, 1, max(11, int(np.ceil(delta / 0.002)) + 1)):
            q = a + smoothstep(float(u)) * (b - a)
            if not adapter.static_check(q)["known_geometry_passed"]:
                raise ValueError("Sampled path collides with known robot/table geometry")
            measurement = geometry.measure(q)
            minimum = min(minimum, measurement["gap_m"])
            actual = inverse @ adapter.kin.forward(q)
            if np.linalg.norm(actual[:2, 3] - flange_world[:2, 3]) > 0.001:
                raise ValueError("Sampled path departs from vertical by more than 1 mm")
            if Rotation.from_matrix(actual[:3, :3] @ flange_world[:3, :3].T).magnitude() > 0.005:
                raise ValueError("Sampled path changes wrist orientation")
            if measurement["gap_m"] < min(initial["gap_m"], clearance) - 0.001:
                raise ValueError("Sampled path undershoots requested clearance")
            samples += 1
    final = geometry.measure(keyframes[-1])
    if abs(final["gap_m"] - clearance) > 0.001:
        raise ValueError("Final whole-hand clearance misses target by more than 1 mm")
    return dict(
        schema_version=1,
        operation="table_clearance",
        environment=environment,
        joint_names=list(JOINT_NAMES),
        keyframes=[q.tolist() for q in keyframes],
        durations_s=durations,
        clearance_m=clearance,
        clearance_definition="minimum world-Z mesh-to-plane gap at each vertex XY",
        vertical_displacement_m=displacement,
        initial=initial,
        target=final,
        sampled_path_min_gap_m=minimum,
        path_samples=samples,
        continuous_path_verified=False,
        environment_collision_verified=False,
        table_status=plane["status"],
        hand_policy="retain_measured_finger_angles",
        settings=settings,
        kinematics=kinematic_contract(state["observed_description"]),
    )


def preview(adapter: object, plan: dict, port: int, execute: bool) -> bool:
    """Review actual mesh interpolation; terminal confirmation alone authorizes this plan."""
    import mujoco
    import viser
    from mjviser import ViserMujocoScene
    from ada_grasp_ctrl.experiments.fixed_camera_preview import plane_mesh

    server = viser.ViserServer(host="127.0.0.1", port=port, label="桌面净空评估")
    try:
        if server.get_port() != port:
            raise RuntimeError(f"Port {port} is occupied")
        scene = ViserMujocoScene(server, adapter.model, num_envs=1)
        scene.camera_tracking_enabled = False
        scene.geom_groups_visible[0] = False
        scene._sync_visibilities()
        vertices, faces = plane_mesh(adapter.provenance["table"])
        vertices = vertices @ adapter.model_T_world[:3, :3].T + adapter.model_T_world[:3, 3]
        server.scene.add_mesh_simple(
            "/measured_table", vertices=vertices, faces=faces, color=(155, 145, 125), side="double", opacity=0.7
        )
        slider = server.gui.add_slider("路径进度 %", min=0, max=100, step=0.1, initial_value=100)
        server.gui.add_markdown(
            f"整只手最低净空：{plan['target']['gap_m'] * 100:.2f} cm\n\n"
            "保持当前手指姿态和手腕方向。桌面及外参未独立验收；仅检查已知模型的离散路径。\n\n"
            + (
                "最低点位于标定采样区域外，桌面高度由平面外推；请核对实际桌边。\n\n"
                if plan.get("target_uses_plane_extrapolation")
                else ""
            )
            + ("终端输入 MOVE 才执行；其他输入取消。" if execute else "只预览，不发送运动指令。Ctrl+C 退出。")
        )
        center = (adapter.model_T_world @ np.r_[plan["target"]["point_world_m"], 1])[:3]
        server.initial_camera.look_at = tuple(center)
        server.initial_camera.position = tuple(center + [0.65, -0.65, 0.45])
        server.initial_camera.up = (0, 0, 1)
        print(f"PREVIEW_URL=http://localhost:{port}", flush=True)
        if execute:
            print("检查预览后输入 MOVE 并回车执行；其他输入取消。", flush=True)
        frames = np.asarray(plan["keyframes"])
        while True:
            progress = slider.value / 100 * sum(plan["durations_s"])
            for i, duration in enumerate(plan["durations_s"]):
                if progress <= duration or i == len(frames) - 2:
                    q = frames[i] + smoothstep(float(np.clip(progress / duration, 0, 1))) * (frames[i + 1] - frames[i])
                    break
                progress -= duration
            adapter.data.qpos[adapter.addresses] = q
            mujoco.mj_forward(adapter.model, adapter.data)
            scene.update_from_mjdata(adapter.data)
            scene.request_update()
            if execute and select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.readline().strip() == "MOVE"
            time.sleep(0.05)
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> int:
    """Capture feedback, prepare a new reviewable plan, and optionally execute it once."""
    from ada_grasp_ctrl.experiments.fixed_camera import load_plane
    from ada_grasp_ctrl.grasp_selection.adapters.crx_sharpa import CrxSharpaAdapter
    from ada_grasp_ctrl.pipeline.runner import Processes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/ada_grasp_ctrl/config/experiments/grasp_pipeline.yaml")
    parser.add_argument("--environment", choices=["hardware", "mock"], default="hardware")
    parser.add_argument("--ros-domain-id", type=int, required=True)
    parser.add_argument("--clearance-m", type=float, default=0.05)
    parser.add_argument("--output-root", default="output/table_clearance/" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--port", type=int, default=8090)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--prepare-only", action="store_true", help="Write plan and diagnostics, without opening preview")
    args = parser.parse_args(argv)
    root = resolve_from_root(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(resolve_from_root(args.config).read_text())
    config.update(environment=args.environment, ros_domain_id=validate_domain(args.environment, args.ros_domain_id))
    processes = Processes(config, root)
    report = {"status": "preparing", "motion_execution_requested": False}
    save(root / "run_manifest.json", {"arguments": vars(args), "configuration": config})
    try:
        model_report = resolve_from_root(config["model_report"])
        model = json.loads(model_report.read_text())["model"]["urdf"]
        processes.run(
            [
                config["ros_python"],
                "-m",
                "ada_grasp_ctrl.ros_execution.snapshot",
                "--model-urdf",
                model,
                "--environment",
                args.environment,
                "--output-root",
                str(root / "state"),
            ],
            "state",
            ros=True,
        )
        state = json.loads((root / "state/robot_state.json").read_text())
        plane_path = resolve_from_root(config["table_plane"])
        plane = load_plane(plane_path)
        adapter = CrxSharpaAdapter(model_report, plane, state["world_T_right_base"])
        plan = make_plan(adapter, state, plane, args.clearance_m, args.environment)
        projected = np.asarray(plan["target"]["point_world_m"]) - [0, 0, plan["target"]["gap_m"]]
        patch_coordinates = np.asarray(plane["observed_tangent_axes"]) @ (projected - plane["centroid_world_m"])
        plan["target_uses_plane_extrapolation"] = bool(
            np.any(np.abs(patch_coordinates) > np.asarray(plane["observed_span_m"]) / 2)
        )
        dependencies = [
            model_report,
            model_report.with_suffix(".sha256"),
            plane_path,
            resolve_from_root(plane["extrinsics"]["path"]),
            Path(model),
        ]
        dependencies.extend(Path(row["path"]) for row in adapter.provenance["dependencies"])
        plan["file_sha256"] = {str(p): digest(p) for p in dependencies}
        plan["ros_domain_id"] = args.ros_domain_id
        save(root / "plan.json", plan)
        plan_hash = digest(root / "plan.json")
        print(
            f"Current gap: {plan['initial']['gap_m'] * 100:.2f} cm; target: {plan['target']['gap_m'] * 100:.2f} cm; "
            f"vertical move: {plan['vertical_displacement_m'] * 100:+.2f} cm",
            flush=True,
        )
        report.update(status="prepared", plan=str(root / "plan.json"))
        if args.prepare_only:
            return 0
        if not preview(adapter, plan, args.port, args.execute):
            report["status"] = "canceled"
            return 0
        report["motion_execution_requested"] = True
        processes.run(
            [
                config["ros_python"],
                "-m",
                "ada_grasp_ctrl.ros_execution.table_clearance",
                "--plan",
                str(root / "plan.json"),
                "--sha256",
                plan_hash,
                "--output-root",
                str(root / "execution"),
            ],
            "execution",
            ros=True,
            timeout=900,
        )
        result = json.loads((root / "execution/result.json").read_text())
        report.update(
            status="completed",
            measured_model_clearance=HandGeometry(adapter, plane).measure(np.asarray(result["final_positions"])),
            physical_clearance_independently_measured=False,
        )
        print(json.dumps(report, indent=2), flush=True)
        return 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        return 130
    except Exception as error:
        report.update(status="failed", reason=str(error))
        print(str(error), file=sys.stderr)
        return 1
    finally:
        processes.close()
        save(root / "run_report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
