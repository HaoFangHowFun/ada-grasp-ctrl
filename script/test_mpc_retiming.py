"""Replay guarded arm targets through the unchanged CRX quintic implementation."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


def main() -> None:
    """Measure deterministic reference lag; this never initializes ROS or a device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    from dual_crx_control.interpolation.quintic import QuinticInterpolation
    from dual_crx_control.interpolation.ruckig import MAX_VELOCITY, MAX_ACCELERATION, MAX_JERK

    trace = json.loads(args.trace.read_text())
    commands = trace["commands"]
    if not commands:
        raise ValueError("Expected current-run guarded ROS commands")
    args.output_root.mkdir(parents=True, exist_ok=False)
    planner = QuinticInterpolation(
        0.002, input_period=0.01, max_velocity=MAX_VELOCITY, max_acceleration=MAX_ACCELERATION, max_jerk=MAX_JERK
    )
    initial = np.asarray(trace["measurements"][0]["positions"][:6])
    target, position = initial.copy(), initial.copy()
    start = commands[0]["time"]
    end = commands[-1]["time"] + 1.0
    index, rows, horizons = 0, [], []
    for step in range(int(np.ceil((end - start) / 0.002))):
        now = start + step * 0.002
        while index < len(commands) and commands[index]["time"] <= now:
            target = np.asarray(commands[index]["positions"][:6])
            planner.target({"right": target}, {"right": position}, {"right": np.zeros(6)}, timestamp=now)
            horizons.append(planner.duration)
            index += 1
        output = planner.step(timestamp=now)
        if output:
            position = output["right"]
        rows.append(
            {
                "time": now,
                "q_100Hz": target.tolist(),
                "q_500Hz": position.tolist(),
                "max_tracking_error_rad": float(np.max(np.abs(target - position))),
                "horizon_s": planner.duration,
            }
        )
    source = Path(inspect.getfile(QuinticInterpolation))
    report = {
        "hardware_used": False,
        "clock": "deterministic_500Hz_replay_of_observed_100Hz_targets",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_rate_hz": 100,
        "output_rate_hz": 500,
        "method": "quintic",
        "commands": len(commands),
        "extended_horizons": planner.extended_count,
        "horizon_s_p50_p95_max": np.percentile(horizons, [50, 95, 100]).tolist(),
        "tracking_error_rad_p50_p95_max": np.percentile(
            [r["max_tracking_error_rad"] for r in rows], [50, 95, 100]
        ).tolist(),
        "terminal_error_rad": rows[-1]["max_tracking_error_rad"],
        "note": "500 Hz reference generation, not physical servo timing; real DDS arrival times are separate",
    }
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output_root / "trace.json").write_text(json.dumps(rows, allow_nan=False))


if __name__ == "__main__":
    main()
