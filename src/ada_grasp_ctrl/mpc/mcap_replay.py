"""Counterfactual, read-only real MCAP replay through existing tactile/Ours adapters."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ada_grasp_ctrl.experiments.right_ros_contract import ARM_NAMES, HAND_NAMES, JOINT_NAMES

JOINT_TOPICS = ("/crx5ia/right/joint_states", "/sharpa/right_hand/joint_states")
COMMAND_TOPICS = ("/crx5ia/joint_targets", "/sharpa/right_hand/joint_command")


def ordered(row: dict, names: tuple[str, ...]) -> tuple[float, ...]:
    """Map named recorded joints, rejecting missing/duplicate/nonfinite values."""
    keys, values = row["names"], row["positions"]
    if len(keys) != len(values) or len(set(keys)) != len(keys) or set(keys) != set(names):
        raise ValueError("joint_names_missing_duplicate_or_unexpected")
    lookup = dict(zip(keys, values))
    result = tuple(float(lookup[n]) for n in names)
    if not all(map(math.isfinite, result)):
        raise ValueError("joint_nonfinite")
    return result


def contact_blockers(frame: dict, now: float) -> list[str]:
    """Keep source status, attribution and acquisition-clock failures independent."""
    reasons = []
    status = frame["contact_status"]
    if status not in ("VALID", "EMPTY", "MISSING"):
        reasons.append("contact_" + status.lower())
    if frame["status"] != "VALID":
        reasons.append("force_" + frame["status"].lower())
    if frame["contact_count"] > 1:
        reasons.append("multi_contact_resultant")
    if now - frame["received"] > 0.2:
        reasons.append("tactile_stale")
    if frame["received"] > now:
        reasons.append("future_receipt")
    if frame["sdk_timestamp_unit"] not in ("s", "seconds") or frame["sdk_timestamp_epoch"] not in ("unix", "UNIX"):
        reasons.append("acquisition_clock_unmapped")
    elif frame["sdk_timestamp"] is None or not 0 <= now - frame["sdk_timestamp"] <= 0.2:
        reasons.append("acquisition_stale_or_future")
    return reasons


def native_reference_matches(report: dict, plan: dict) -> bool:
    """Only associate a frozen reference when its named joint endpoints agree."""
    sample = np.load(report["samples"][0]["prepared_path"], allow_pickle=True).item()
    i, j = plan["phases"].index("grasp"), plan["phases"].index("squeeze")
    return all(
        np.allclose(plan["keyframes"][k], sample[field], atol=1e-6, rtol=0)
        for k, field in ((i, "pregrasp_qpos"), (i + 1, "grasp_qpos"), (j + 1, "squeeze_qpos"))
    )


def replay(decoded: Path, baseline: Path, report_path: Path, output: Path, package: Path, *, p3: bool = False) -> dict:
    """Sample only already-delivered data at 10 Hz; never replay robot commands."""
    import torch  # noqa: F401
    from ada_grasp_ctrl.contact.compatibility import TactileContactEstimator
    from ada_grasp_ctrl.mpc.contracts import JointFeedback
    from ada_grasp_ctrl.mpc.native import NativeMPC
    from ada_grasp_ctrl.mpc.offline import json_value, make_runner
    from ada_grasp_ctrl.tactile_preview.adapter import TactileAdapter
    from ada_grasp_ctrl.tactile_preview.geometry import GeometryCatalog
    from ada_grasp_ctrl.tactile_preview.kinematics import load_hand

    events = [json.loads(line) for line in decoded.open()]
    if any(a["bag_ns"] > b["bag_ns"] for a, b in zip(events, events[1:])):
        raise ValueError("Bag delivery timestamps are out of order")
    goals = [json.loads(line) for line in (baseline / "sender/events.jsonl").open()]
    goals = [row for row in goals if row["kind"] == "goal_sent"]
    packet = json.loads((baseline / "sender/targets.json").read_text())
    plan = packet["reviewed_plan"]
    report = json.loads(report_path.read_text())
    from ada_grasp_ctrl.experiments.right_ros_contract import check_kinematics, kinematic_contract, validate_packet
    from ada_grasp_ctrl.pipeline.plan import validate_plan

    validate_packet(packet)
    validate_plan(plan)
    check_kinematics(packet["kinematics"], kinematic_contract(Path(report["model"]["urdf"]).read_text()))
    if p3:
        from .observation import selected_runner

        runner = selected_runner(report, packet, output)
    else:
        runner = make_runner(report, plan, output)
    native = NativeMPC(runner)
    reference_ok = p3 or native_reference_matches(report, plan)
    # Header and receive_monotonic_ns are a paired host-receipt observation.
    # This maps stage logs only, never the SDK acquisition clock.
    first = next(e for e in events if "frame" in e and e["receive_monotonic_ns"] > 0)
    offset = first["frame"]["receipt_ros"] - first["receive_monotonic_ns"] * 1e-9
    offsets = [
        e["frame"]["receipt_ros"] - e["receive_monotonic_ns"] * 1e-9
        for e in events
        if "frame" in e and e["receive_monotonic_ns"] > 0
    ]
    uncertainty = max(abs(x - offset) for x in offsets)
    intervals = [(a["phase"], a["time"] + offset, b["time"] + offset) for a, b in zip(goals, goals[1:])]
    active = [v for v in intervals if v[0] in ("grasp", "squeeze")]
    if len(active) != 2:
        raise ValueError("Missing recorded grasp/squeeze phase interval")
    hand = load_hand(package / "urdf/right_sharpa_wave_with_flange.urdf", package, load_meshes=False)
    now = 0.0
    adapter = TactileAdapter(
        hand,
        GeometryCatalog.installed_candidate(),
        clock=lambda: now,
        stale_after=0.2,
        receipt_clock_synchronized=not p3,
    )
    estimator = TactileContactEstimator(
        runner.robot_adaptor,
        object_bodies=("object",),
        tactile_joint_names=tuple(hand.actuated_joint_names),
        friction_mu=runner.grasp_ctrl.mu,
        allow_candidate_geometry=True,
        measured_force_policy="diagnostic" if p3 else "strict",
        ambiguous_contact_policy="omit" if p3 else "reject",
    )
    latest, previous, issues = {}, {}, {}
    counts, blockers, statuses, topic_counts = Counter(), Counter(), Counter(), Counter()
    gaps, durations, examples = Counter(), [], {}
    cursor = 0
    start_ns, end_ns = int(active[0][1] * 1e9), int(active[-1][2] * 1e9)
    tick_ns = start_ns
    with (output / "ticks.jsonl").open("x") as trace:
        while tick_ns < end_ns:
            receipt_now = tick_ns * 1e-9
            now = (tick_ns - events[0]["bag_ns"]) * 1e-9 if p3 else receipt_now
            started = time.perf_counter()
            while cursor < len(events) and events[cursor]["bag_ns"] <= tick_ns:
                event = events[cursor]
                topic = event["topic"]
                topic_counts[topic] += 1
                problems = []
                prior = previous.get(topic)
                if "frame" in event:
                    f = event["frame"]
                    if prior:
                        old = prior["frame"]
                        if f["session_id"] != old["session_id"] or f["sequence"] <= old["sequence"]:
                            problems.append("session_change_or_sequence_regression")
                        if f["sdk_frame_id"] is not None and old["sdk_frame_id"] is not None:
                            delta = f["sdk_frame_id"] - old["sdk_frame_id"]
                            if delta <= 0:
                                problems.append("sdk_out_of_order")
                            elif delta > 1:
                                gaps["sdk_forward_gap_events"] += 1
                        gaps["receipt_sequence_skipped"] += max(0, f["sequence"] - old["sequence"] - 1)
                    gaps["source_sdk_gap_warnings"] += sum("forward_gap" in w for w in f["warnings"])
                elif prior and event["source_ros_s"] <= prior["source_ros_s"]:
                    problems.append("joint_source_out_of_order")
                latest[topic], previous[topic], issues[topic] = event, event, problems
                cursor += 1
            phase = next((p for p, lo, hi in active if lo <= receipt_now < hi), None)
            if phase is None:
                tick_ns += 100_000_000
                continue
            reasons = []
            if not reference_ok:
                reasons.append("frozen_reference_alignment_unverified")
            if uncertainty > 0.01 or any(
                abs(receipt_now - edge) <= uncertainty for _, lo, hi in active for edge in (lo, hi)
            ):
                reasons.append("phase_clock_boundary_uncertain")
            row = dict(
                schema_version=1,
                tick_ns=tick_ns,
                replay_s=(tick_ns - events[0]["bag_ns"]) * 1e-9,
                clock_domain="relative_bag_receipt" if p3 else "bag_ROS_receipt",
                phase=phase,
                proposal_status="unavailable",
                proposal=None,
                native=None,
                contacts={},
                joints={},
                baseline_commands={},
                reference_provenance=runner.reference_provenance
                if p3
                else "recorded_reviewed_plan; frozen_model_scene_association_unverified",
            )
            feedback = None
            try:
                states = [latest[t] for t in JOINT_TOPICS]
                commands = [latest[t] for t in COMMAND_TOPICS]
                q = ordered(states[0], ARM_NAMES) + ordered(states[1], HAND_NAMES)
                cmd = ordered(commands[0], ARM_NAMES) + ordered(commands[1], HAND_NAMES)
                row["joints"] = dict(zip(JOINT_NAMES, q))
                row["baseline_commands"] = dict(zip(JOINT_NAMES, cmd))
                row["joint_clocks"] = [
                    {
                        "topic": t,
                        "source_ROS_s": e["source_ros_s"],
                        "bag_ns": e["bag_ns"],
                        "source_age_s": receipt_now - e["source_ros_s"],
                    }
                    for t, e in zip(JOINT_TOPICS, states)
                ]
                for e in states:
                    if not 0 <= receipt_now - (e["bag_ns"] * 1e-9 if p3 else e["source_ros_s"]) <= 0.2:
                        reasons.append("joint_stale_or_future")
                feedback = JointFeedback(
                    tuple(JOINT_NAMES),
                    q,
                    min((e["bag_ns"] - events[0]["bag_ns"]) * 1e-9 for e in states)
                    if p3
                    else min(e["source_ros_s"] for e in states),
                    (max(e["bag_ns"] for e in states) - (events[0]["bag_ns"] if p3 else 0)) * 1e-9,
                    "receipt_aligned" if p3 else "recorded_joint_header_ROS",
                )
                frames = {e["frame"]["finger"]: e["frame"] for e in latest.values() if "frame" in e}
                if p3:
                    frames = {
                        e["frame"]["finger"]: dict(e["frame"], received=(e["bag_ns"] - events[0]["bag_ns"]) * 1e-9)
                        for e in latest.values()
                        if "frame" in e
                    }
                h = states[1]
                snapshot = adapter.update(
                    frames,
                    dict(
                        names=h["names"],
                        positions=h["positions"],
                        received=(h["bag_ns"] - (events[0]["bag_ns"] if p3 else 0)) * 1e-9,
                        receipt_ros=h["source_ros_s"],
                        subscriber_ros=h["bag_ns"] * 1e-9,
                    ),
                    allow_unverified_geometry=True,
                )
                estimate = estimator.estimate(snapshot, feedback, now)
                row["compatibility"] = dict(
                    status=estimate.status, reasons=estimate.reasons, provenance=estimate.provenance
                )
                counts["contact_eligible_ticks"] += estimate.observation is not None
                counts["contact_valid_empty_ticks"] += estimate.status == "valid_empty"
                for reason in estimate.reasons:
                    counts["compatibility_reason:" + reason] += 1
                for finger, result in snapshot.fingers:
                    raw = frames.get(finger)
                    per_finger = contact_blockers(raw, now) if raw else ["contact_no_data"]
                    if p3:
                        per_finger = [
                            r
                            for r in per_finger
                            if r
                            not in (
                                "acquisition_clock_unmapped",
                                "acquisition_stale_or_future",
                                "multi_contact_resultant",
                            )
                        ]
                    if raw and abs((raw["received"] if p3 else raw["receipt_ros"]) - feedback.source_timestamp) > 0.15:
                        per_finger.append("joint_tactile_clock_skew")
                    source = latest.get(f"/sharpa/right_hand/tactile/{finger}/frame", {})
                    row["contacts"][finger] = dict(
                        adapter=asdict(result),
                        compatibility=estimate.provenance["finger_statuses"][finger],
                        reasons=per_finger,
                        all_contact_rows=source.get("contact_raw"),
                        receive_monotonic_ns=source.get("receive_monotonic_ns"),
                        receipt_age_s=receipt_now - raw["receipt_ros"] if raw else None,
                        solver_accepted=False,
                        acquisition_age_s=None
                        if "acquisition_clock_unmapped" in per_finger
                        else receipt_now - raw["sdk_timestamp"]
                        if raw and raw["sdk_timestamp"] is not None
                        else None,
                    )
                    reasons.extend(per_finger)
                if estimate.observation is None:
                    reasons.append("contact_compatibility_unavailable")
                for problem in issues.values():
                    reasons.extend(problem)
                if not reasons:
                    counts["eligible_observations"] += 1
                if not reasons and p3 and native.completed:
                    row["proposal_status"] = "after_native_complete"
                elif not reasons:
                    q_ref = tuple(runner.qpos_path[min(native.index, len(runner.qpos_path) - 1)])
                    proposed = native.tick(now, feedback, cmd, q_ref, estimate.observation, stage=phase)
                    row["native"] = asdict(proposed)
                    row["proposal_status"] = "accepted" if proposed.valid else "rejected"
                    if proposed.valid:
                        row["proposal"] = dict(zip(JOINT_NAMES, proposed.positions_rad))
                        for contact in row["contacts"].values():
                            contact["solver_accepted"] = True
                    elif p3 and proposed.diagnostics.get("termination") == "native_complete":
                        row["proposal_status"] = "native_complete"
                    else:
                        reasons.append(proposed.diagnostics.get("termination", "native_rejected"))
            except (KeyError, ValueError) as error:
                reasons.append("joint_or_adapter_invalid")
                row["error"] = str(error)
            row["reasons"] = sorted(set(reasons))
            for reason in row["reasons"]:
                blockers[reason] += 1
                examples.setdefault(reason, dict(tick_ns=tick_ns, phase=phase))
            statuses[row["proposal_status"]] += 1
            duration = time.perf_counter() - started
            durations.append(duration)
            row["processing_duration_s"] = duration
            trace.write(json.dumps(json_value(row), allow_nan=False) + "\n")
            tick_ns += 100_000_000
    summary = dict(
        schema_version=1,
        status="completed",
        counterfactual=True,
        messages=len(events),
        topic_counts=dict(Counter(e["topic"] for e in events)),
        ticks=sum(statuses.values()),
        proposal_statuses=dict(statuses),
        eligible_observations=counts["eligible_observations"],
        contact_eligible_ticks=counts["contact_eligible_ticks"],
        contact_valid_empty_ticks=counts["contact_valid_empty_ticks"],
        compatibility_blockers={
            k.removeprefix("compatibility_reason:"): v
            for k, v in counts.items()
            if k.startswith("compatibility_reason:")
        },
        native_proposals=sum(statuses[k] for k in ("accepted", "rejected")),
        native_completion_events=statuses["native_complete"],
        blockers=dict(blockers),
        examples=examples,
        timing_s=dict(zip(("p50", "p95", "p99"), np.percentile(durations, [50, 95, 99]).tolist())),
        timing_semantics="parser_adapter_processing; not solver latency or closed-loop performance",
        phase_intervals=intervals,
        clock_mapping=dict(
            source="first paired header/host receipt",
            offset_s=offset,
            observed_max_residual_s=uncertainty,
            acquisition_mapping=None,
        ),
        sequence_diagnostics=dict(gaps),
        sequence_scope="bag start through last grasp/squeeze tick",
        frozen_reference_matches=native_reference_matches(report, plan),
        selected_reference_verified=p3,
        compatibility_policy="p3_receipt_aligned_diagnostic_cone" if p3 else "p2a_strict",
        acquisition_synchronized=False,
        hardware_io=False,
        ros_context_initialized=False,
        robot_command_publishers=0,
        no_hardware_evidence="decoder uses SequentialReader only; replay imports no ROS runtime",
        limitations=[
            "Unknown SDK units/epoch are not acquisition timestamps",
            "Candidate geometry and force convention remain provisional",
            "53 reported transport losses; sequence gaps are not DDS loss counts",
            "Recorded feedback follows baseline commands, not MPC proposals",
        ],
    )
    (output / "run_report.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Decode and replay in one invocation, with an isolated ROS serialization ABI."""
    from ada_grasp_ctrl.paths import project_root, resolve_from_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p3", action="store_true", help="Selected reference, receipt alignment and diagnostic measured cone"
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--model-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--package-share", default="/home/hao-fang/ws_fanuc/install/dual_sharpa_wave/share/dual_sharpa_wave"
    )
    parser.add_argument("--ros-setup", default="/home/hao-fang/ws_fanuc/install/setup.bash")
    args = parser.parse_args()
    bag, baseline, report = (resolve_from_root(v) for v in (args.bag, args.baseline_run, args.model_report))
    output = resolve_from_root(args.output_root, "output")
    output.mkdir(parents=True, exist_ok=False)
    decoded = output / "decoded.jsonl"
    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID="199", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
    shell = (
        'source /opt/ros/jazzy/setup.bash; source "$1"; export PYTHONPATH="$2:$PYTHONPATH"; '
        'exec /usr/bin/python3 -m ada_grasp_ctrl.mpc.mcap_decode --bag "$3" --output "$4"'
    )
    subprocess.run(
        ["bash", "-c", shell, "offline-decode", args.ros_setup, str(project_root() / "src"), str(bag), str(decoded)],
        env=env,
        check=True,
    )
    manifest = dict(
        schema_version=1,
        argv=sys.argv,
        bag=str(bag),
        baseline=str(baseline),
        model_report=str(report),
        command_publishers=0,
        ros_init=False,
        input_hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (baseline / "sender/events.jsonl", baseline / "sender/targets.json", report)
        },
        bag_files=[dict(path=str(p), size=p.stat().st_size) for p in sorted(bag.glob("*.mcap"))],
    )
    (output / "run_manifest.yaml").write_text(json.dumps(manifest, indent=2))
    summary = replay(decoded, baseline, report, output, resolve_from_root(args.package_share), p3=args.p3)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
