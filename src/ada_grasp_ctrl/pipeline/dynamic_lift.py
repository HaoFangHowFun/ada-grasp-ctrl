"""Measured-state MPC handoff; native geometry work stays outside ROS Python."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from ada_grasp_ctrl.experiments.right_ros_contract import JOINT_NAMES, finite_vector
from ada_grasp_ctrl.pipeline.plan import check_start

PREFIX = "ada_dynamic_lift_v1:"
HEIGHT_M = 0.15


def fingerprint(value: dict) -> str:
    """Bind a derived goal to its exact original packet and snapshot."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def eligible(mode: str, packet: dict, native_completed: bool) -> bool:
    """Limit dynamic handoff to successful right hardware MPC episodes."""
    return mode == "mpc_active" and packet.get("environment") == "hardware" and native_completed


def validate_snapshot(snapshot: dict, timeout: float, *, after: float = 0.0) -> list[float]:
    """Check freshness at capture, named order and existing 150 ms receipt coherence.

    Subsequent planning time does not renew these timestamps. Both endpoints must
    separately compare fresh actual feedback with this captured start before motion.
    """
    if snapshot.get("joint_names") != list(JOINT_NAMES) or snapshot.get("source") != "same_host_receipt_aligned":
        raise ValueError("Dynamic lift requires named actual feedback provenance")
    q = finite_vector(snapshot.get("positions"), 28, "dynamic measured start")
    captured = snapshot["captured_at"]
    receipts = [snapshot["received_at"][g] for g in ("arm", "hand")]
    stamps = [snapshot["header_stamps"][g] for g in ("arm", "hand")]
    if not all(math.isfinite(v) for v in [captured, *receipts, *stamps]):
        raise ValueError("Invalid handoff timestamps")
    if min(stamps) <= 0 or min(receipts) <= after or not all(0 <= captured - v <= timeout for v in receipts):
        raise ValueError("Stale handoff snapshot or feedback predates native completion")
    if max(receipts) - min(receipts) > 0.15:
        raise ValueError("Incoherent handoff joint receipts")
    if snapshot.get("topics") != {"arm": "/crx5ia/right/joint_states", "hand": "/sharpa/right_hand/joint_states"}:
        raise ValueError("Missing handoff feedback sources")
    return q


def check_handoff_start(actual: list[float], expected: list[float], event: Callable[..., None]) -> None:
    """Log actual race deltas before applying the unchanged strict start check."""
    event("dynamic_start_check", actual=actual, expected=expected, delta_rad=[a - b for a, b in zip(actual, expected)])
    check_start(actual, expected)


def build_handoff(packet: dict, snapshot: dict) -> dict:
    """Reuse selected calibrated geometry, local IK and sampled known-geometry gates.

    No physics steps, grasp inference or controller solves are performed. The
    sampled joint path does not certify continuous or unknown-environment collision
    clearance; it uses the existing selector's robot/table collision policy.
    """
    import numpy as np
    from ada_grasp_ctrl.experiments.right_ros_sender import motion_duration
    from ada_grasp_ctrl.experiments.right_ros_contract import check_kinematics, kinematic_contract
    from ada_grasp_ctrl.experiments.table_clearance import HandGeometry
    from ada_grasp_ctrl.grasp_selection.adapters.crx_sharpa import CrxSharpaAdapter
    from ada_grasp_ctrl.pipeline.lift import lift_target

    if packet.get("environment") != "hardware" or packet.get("side", "right") != "right":
        raise ValueError("Dynamic lift requires right hardware packet")
    plan = packet["reviewed_plan"]
    if "passive_positions_rad" in plan:
        raise ValueError("Dynamic lift dual-scene validation is unavailable")
    start = validate_snapshot(snapshot, plan["receiver_settings"]["feedback_timeout"])
    configuration = json.loads(Path(packet["source"]["selection_path"]).read_text())["configuration"]
    adapter = CrxSharpaAdapter(
        Path(configuration["robot"]["model_report"]),
        configuration["robot"]["table"],
        configuration["context"]["world_T_right_base"],
    )
    check_kinematics(packet["kinematics"], kinematic_contract(Path(adapter.info["urdf"]).read_text()))
    q = np.asarray(start)
    if np.any(q < adapter.limits[:, 0] - 1e-8) or np.any(q > adapter.limits[:, 1] + 1e-8):
        raise ValueError("Dynamic lift measured start exceeds model limits")
    lift = lift_target(adapter, start, HEIGHT_M)
    target = np.asarray(lift["positions"])
    geometry = HandGeometry(adapter, configuration["robot"]["table"])
    initial_gap = geometry.measure(q)["gap_m"]
    minimum_gap = initial_gap
    count = max(11, int(np.ceil(np.max(np.abs(target - q)) / 0.002)) + 1)
    # Both existing quintic and linear timing traverse this same joint-space segment.
    for u in np.linspace(0, 1, count):
        sample = (1 - u) * q + u * target
        if np.any(sample < adapter.limits[:, 0] - 1e-8) or np.any(sample > adapter.limits[:, 1] + 1e-8):
            raise ValueError("Dynamic lift path exceeds model limits")
        if not adapter.static_check(sample)["known_geometry_passed"]:
            raise ValueError(f"Dynamic lift known robot/table collision at path fraction {u:g}")
        gap = geometry.measure(sample)["gap_m"]
        minimum_gap = min(minimum_gap, gap)
        if gap < 0 or gap < initial_gap - 0.001:
            raise ValueError("Dynamic lift sampled hand path intersects/undershoots table clearance")
    lift.update(hand_policy="retain_measured_completion", orientation_policy="retain_measured_fk_orientation")
    index = plan["phases"].index("lift")
    args = argparse.Namespace(minimum_duration=2.0, **plan["receiver_settings"])
    segments = {}
    previous = start
    for phase in plan["phases"][index:]:
        if phase in ("lift", "hold"):
            positions = lift["positions"]
        elif phase == "release":
            positions = lift["positions"][:6] + [0.0] * 22
        elif phase == "return":
            positions = plan["keyframes"][-1]
        else:
            raise ValueError("Unsupported post-lift phase")
        duration = plan["cycle"]["hold_s"] if phase == "hold" else motion_duration(previous, positions, args)
        if not math.isfinite(duration) or not 0 < duration <= 300:
            raise ValueError("Dynamic goal duration outside existing contract")
        segments[phase] = dict(start=previous, positions=positions, duration_s=duration)
        previous = positions
    result = dict(
        schema_version=1,
        packet_sha256=fingerprint(packet),
        snapshot=snapshot,
        previous_preview_start=plan["keyframes"][index],
        preview_delta_rad=(q - np.asarray(plan["keyframes"][index])).tolist(),
        lift=lift,
        ik_seed=start[:6],
        ik_solution=lift["positions"][:6],
        geometry_provenance=adapter.provenance,
        validation=dict(
            joint_limits=True,
            sampled_known_geometry=True,
            path_samples=count,
            initial_hand_table_gap_m=initial_gap,
            minimum_hand_table_gap_m=minimum_gap,
            continuous_path_verified=False,
            object_environment_collision_verified=False,
            scope="lift only; existing hold/release/return policies retained",
        ),
        segments=segments,
    )
    result["review_id"] = fingerprint(result)
    return result


def decode(frame: str) -> dict | None:
    """Decode the explicit handoff extension; ordinary empty headers stay unchanged."""
    if not frame:
        return None
    if not frame.startswith(PREFIX) or len(frame) > 200_000:
        raise ValueError("Unsupported trajectory frame/metadata")
    value = json.loads(frame[len(PREFIX) :])
    if not isinstance(value, dict):
        raise ValueError("Malformed dynamic lift metadata")
    return value


def check_goal(packet: dict, handoff: dict, index: int, positions: list, duration: float, config: dict) -> dict:
    """Retain original packet/settings validation and bind the derived phase exactly."""
    from ada_grasp_ctrl.ros_execution.review import check_reviewed_goal

    original = packet["targets"][index]
    check_reviewed_goal(packet, index, original["positions"], original["duration_s"], config)
    unsigned = {k: v for k, v in handoff.items() if k != "review_id"}
    if handoff.get("packet_sha256") != fingerprint(packet) or handoff.get("review_id") != fingerprint(unsigned):
        raise ValueError("Dynamic lift review/source identity mismatch")
    if handoff.get("schema_version") != 1 or handoff["lift"]["height_m"] != HEIGHT_M:
        raise ValueError("Dynamic lift policy mismatch")
    segment = handoff["segments"].get(original["phase"])
    if segment is None or positions != segment["positions"] or abs(duration - segment["duration_s"]) > 1e-8:
        raise ValueError("Goal differs from dynamic reviewed phase")
    return segment


def check_trusted_plan(handoff: dict, path: str | Path | None, now: float) -> None:
    """Accept only the controlled sender's persisted plan, never a self-asserted hash.

    The runner chooses this same-run path before starting either process. This is
    a local trusted-planner boundary, not authentication against a compromised host.
    No FK, IK or geometry sampling is performed here.
    """
    if path is None:
        raise ValueError("Dynamic lift requires the runner's --dynamic-lift-plan path")
    reply = json.loads(Path(path).read_text())
    if reply.get("ok") is not True or reply.get("handoff") != handoff:
        raise ValueError("Dynamic lift differs from controlled sender plan")
    age = now - handoff["snapshot"]["captured_at"]
    if not math.isfinite(age) or not 0 <= age <= 20.0:
        raise ValueError("Dynamic lift plan expired or has future capture time")
    validation = handoff.get("validation", {})
    if validation.get("joint_limits") is not True or validation.get("sampled_known_geometry") is not True:
        raise ValueError("Dynamic lift lacks successful planner feasibility checks")


def compute(
    python: str,
    packet_path: Path,
    snapshot: dict,
    root: Path,
    spin: Callable[[], None],
    *,
    expected: dict | None = None,
) -> dict:
    """Run read-only native geometry with bounded lifetime while ROS keeps receiving.

    Only the sender invokes this planner. The receiver checks the persisted result
    and fresh actual start. Cancellation terminates only this owned process.
    """
    from ada_grasp_ctrl.paths import project_root

    request = root / "dynamic-lift-request.json"
    output = root / "dynamic-lift-result.json"
    if output.exists() or request.exists():
        raise ValueError("Dynamic lift output must belong to a fresh planning invocation")
    request.write_text(json.dumps(snapshot, allow_nan=False))
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(PYTHONPATH=str(project_root() / "src"), OMP_NUM_THREADS="1")
    with (root / "dynamic-lift-worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                python,
                "-m",
                "ada_grasp_ctrl.pipeline.dynamic_lift",
                "--targets",
                str(packet_path),
                "--snapshot",
                str(request),
                "--output",
                str(output),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while process.poll() is None:
                spin()
                if time.monotonic() > deadline:
                    raise TimeoutError("Dynamic lift planning timeout")
            if process.returncode or not output.exists():
                raise ValueError("Dynamic lift planner failed; see " + str(root / "dynamic-lift-worker.log"))
            reply = json.loads(output.read_text())
            if not reply["ok"]:
                raise ValueError("Dynamic lift: " + reply["reason"])
            if expected is not None and reply["handoff"] != expected:
                raise ValueError("Dynamic review differs from expected offline fixture")
            return reply["handoff"]
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)


def main() -> None:
    """Offline-only geometry entrypoint; never imports ROS or creates publishers."""
    from ada_grasp_ctrl.experiments.right_ros_sender import load_packet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        began = time.monotonic()
        with redirect_stdout(sys.stderr):
            result = build_handoff(load_packet(args.targets), json.loads(args.snapshot.read_text()))
        reply: dict[str, Any] = dict(ok=True, handoff=result, geometry_s=time.monotonic() - began)
    except Exception as error:
        reply = dict(ok=False, reason=str(error))
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(reply, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
