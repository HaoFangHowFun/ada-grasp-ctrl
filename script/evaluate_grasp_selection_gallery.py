"""Evaluate the saved five-capture gallery twice without ROS, inference or motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ada_grasp_ctrl.grasp_selection.cli import main as select_main, write_json
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json
from ada_grasp_ctrl.paths import resolve_from_root


def main(argv: list[str] | None = None) -> int:
    """Use independent fresh run directories and compare deterministic report bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--model-report", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    output = resolve_from_root(args.output_root, "output")
    try:
        groups = read_json(resolve_from_root(args.gallery))["groups"]
        if len(groups) != 5 or len({g["group"] for g in groups}) != 5:
            raise ValueError("Expected five uniquely named capture groups")
        output.mkdir(parents=True, exist_ok=False)
        summary = []
        for group in groups:
            group_id = str(group["group"])
            if not group_id.isdecimal():
                raise ValueError("Group IDs must be numeric directory labels")
            runs = []
            for repeat in (1, 2):
                run = output / f"group{group_id}_repeat{repeat}"
                code = select_main(
                    [
                        "--candidates",
                        group["candidates_path"],
                        "--state",
                        args.state,
                        "--model-report",
                        args.model_report,
                        "--output-root",
                        str(run),
                        "--replay",
                    ]
                )
                if code:
                    raise ValueError(f"Group {group_id} repeat {repeat} failed; inspect {run}")
                runs.append(run)
            comparisons = {
                name: (runs[0] / name).read_bytes() == (runs[1] / name).read_bytes()
                for name in ("selection_result.json", "candidate_evaluations.json")
            }
            report = read_json(runs[0] / "run_report.json")
            summary.append(
                {
                    "group": group_id,
                    "label": group["label"],
                    **report,
                    "reproducible": all(comparisons.values()),
                    "comparisons": comparisons,
                    "runs": list(map(str, runs)),
                }
            )
            write_json(output / "integration_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if all(r["reproducible"] for r in summary) else 1
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "integration_error", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
