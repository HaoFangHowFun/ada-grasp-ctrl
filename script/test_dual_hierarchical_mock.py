"""Record synthetic dual-live selection/export audits without ROS, model inference or devices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    """Run deterministic analytical fixtures and preserve inspectable single-side packets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(checkout / "src"), str(checkout / "tests")]
    from ada_grasp_ctrl.paths import resolve_from_root
    from ada_grasp_ctrl.experiments.right_ros_sender import load_packet
    from dual_selector_fixtures import LiveFixture

    root = resolve_from_root(args.output_root, "output")
    root.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, options, distances, first in (
        ("legacy_default", [], None, "right_full"),
        ("right_ranked", ["--selector", "hierarchical"], None, "right_full"),
        ("left_ranked", ["--selector", "hierarchical"], None, "left_full"),
        ("fallback", ["--selector", "hierarchical"], dict(right=6, left=0.2), "right_full"),
        ("infeasible", ["--selector", "hierarchical"], dict(right=6, left=6), "right_full"),
        ("left_only", ["--selector", "hierarchical", "--arms", "left"], None, "right_full"),
    ):
        fixture = LiveFixture(root / name, distances=distances)
        fixture.config["hierarchical_selector"]["prior"]["ranking"] = [
            first,
            "left_full" if first == "right_full" else "right_full",
        ]
        code = fixture.invoke(options)
        expected = 1 if name == "infeasible" else 0
        if code != expected:
            raise RuntimeError(f"{name}: unexpected exit code {code}")
        report = fixture.artifact("run_report.json")
        path = fixture.root / "run/export/targets.json"
        if name == "infeasible":
            assert not path.exists()
            assert fixture.artifact("hierarchical_selection.json")["status"] == "NO_FEASIBLE_GRASP"
        else:
            packet = load_packet(path)
            selected = fixture.artifact("selection.json")["selected"]
            assert packet["sample_id"] == selected["candidate_id"]
            assert packet["side"] == selected["selected_side"]
            report["validated_packet_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert all(
            label in ("initial_state", "state", "inference_left", "inference_right") for label, _ in fixture.calls
        )
        report.update(exit_code=code, synthetic=True, hardware_used=False, ros_started=False)
        reports[name] = report
    sources = [
        "src/ada_grasp_ctrl/pipeline/dual_runner.py",
        "src/ada_grasp_ctrl/pipeline/dual_selection.py",
        "src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml",
        "tests/dual_selector_fixtures.py",
        "script/test_dual_hierarchical_mock.py",
    ]
    manifest = dict(
        schema_version=1,
        status="passed",
        fixture_kind="synthetic analytical capture/network/IK/geometry",
        ros_started=False,
        hardware_used=False,
        cases=reports,
        source_sha256={name: hashlib.sha256((checkout / name).read_bytes()).hexdigest() for name in sources},
    )
    (root / "run_manifest.yaml").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
