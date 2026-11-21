"""Record deterministic shared-stage fixtures twice and verify exact equality; no ROS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    """Write fixed-clock synchronization evidence to a new output root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(checkout / "src"), str(checkout / "tests")]
    from ada_grasp_ctrl.paths import resolve_from_root
    from shared_executor_fixtures import packet
    from test_shared_executor import simulate

    root = resolve_from_root(args.output_root, "output")
    root.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, active, options in (
        ("both", ("left", "right"), dict(full=True)),
        ("right", ("right",), {}),
        ("left", ("left",), {}),
        ("late_hand_error", ("left", "right"), dict(late=True, hand_error=True)),
    ):
        digests = []
        for run in (1, 2):
            stream, events, frames = simulate(active, **options)
            if not stream.done or stream.fault:
                raise RuntimeError(f"{name}: incomplete recording")
            data = dict(
                schema_version=1,
                environment="offline_mock",
                hardware_used=False,
                clock="fixed 100 Hz starting at zero",
                packet=packet(active, full=options.get("full", False)),
                events=events,
                frames=frames,
            )
            raw = (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()
            (root / f"{name}_{run}.json").write_bytes(raw)
            digests.append(hashlib.sha256(raw).hexdigest())
        if digests[0] != digests[1]:
            raise RuntimeError(f"{name}: recordings differ")
        reports[name] = dict(sha256=digests[0], identical_runs=2, frames=len(frames), completed_stages=stream.index)
    manifest = dict(
        schema_version=1,
        status="passed",
        cases=reports,
        source_sha256={
            str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                checkout / "tests/test_shared_executor.py",
                checkout / "tests/shared_executor_fixtures.py",
                checkout / "src/ada_grasp_ctrl/ros_execution/shared_stream.py",
                checkout / "src/ada_grasp_ctrl/ros_execution/shared_packet.py",
                checkout / "src/ada_grasp_ctrl/mpc/baseline_stream.py",
            )
        },
    )
    (root / "run_manifest.yaml").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
