"""Analytical dual-live CLI fixture: synthetic capture/network/robot I/O, real selection and export."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json
from ada_grasp_ctrl.grasp_selection.cli import write_json
from ada_grasp_ctrl.pipeline import dual_runner, dual_artifacts, approach
from ada_grasp_ctrl.robot_environment.model import side_slice
from shared_executor_fixtures import description
from test_phase2_pipeline import FakeRobot, FakeSim, candidate


class AnalyticalSide:
    """Simple translational FK for endpoint tests; not a model of physical CRX reachability."""

    def __init__(self, robot: object, sim: object, current: object, side: str) -> None:
        self.limits = robot.limits[side_slice(side)]
        self.model_T_world = np.eye(4)
        self.kin = SimpleNamespace(forward=self.forward, bounds=self.limits[:6].T)

    def forward(self, q: object) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q)[:3]
        return pose


class AnalyticalGeometry:
    def __init__(self, adapter: object, plane: dict) -> None:
        pass

    def measure(self, q: object) -> dict:
        return dict(gap_m=float(q[2]))


class LiveFixture:
    """Run the real dual CLI with strictly synthetic boundary replacements and no ROS startup."""

    def __init__(self, root: Path, *, distances: dict | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        inputs = root / "inputs"
        inputs.mkdir()
        self.robot, self.sim = FakeRobot(), FakeSim()
        self.robot.directory = inputs
        self.robot.urdf_path = inputs / "model.urdf"
        self.robot.urdf_path.write_text(description())
        write_json(inputs / "manifest.json", dict(source="synthetic fixture; not physical robot geometry"))
        write_json(inputs / "table.json", dict(normal_xyz=[0, 0, 1], d=0))
        self.config = dual_runner.configuration("src/ada_grasp_ctrl/config/experiments/dual_grasp_pipeline.yaml")
        refs = {}
        for key in ("checkpoint", "training_config", "metadata"):
            path = inputs / (key + ".json")
            write_json(path, dict(fixture=True, mode="right_full", name=key))
            refs[key] = dict(path=str(path), sha256=file_hash(path))
        self.config.update(
            environment="mock",
            robot_snapshot=str(inputs),
            table_plane=str(inputs / "table.json"),
            inference_model=refs,
            device="cpu",
        )
        self.distances = distances or dict(left=0.1, right=0.4)
        self.calls = []
        self.closed = False
        self.q = np.zeros(56)

    def perceive(self, group: dict, args: object, config: dict, processes: object, output: Path, **kw: object) -> Path:
        obj = output / "object"
        obj.mkdir()
        points = np.array([[0.1, 0.2, 0.01], [0.2, 0.4, 0.03], [0.3, 0.1, 0.02]])
        np.savez(obj / "object.npz", points_world_m=points, points_color_m=points, T_world_color=np.eye(4))
        write_json(obj / "object.json", dict(object_sha256=file_hash(obj / "object.npz"), extrinsics={}))
        return obj

    def run(self, command: list[str], label: str, **kw: object) -> None:
        self.calls.append((label, command))
        output = Path(command[command.index("--output-root") + 1])
        output.mkdir()
        if label in ("initial_state", "state", "before_execute"):
            write_json(
                output / "robot_state.json",
                dict(
                    environment="mock",
                    full_positions_rad=self.q.tolist(),
                    sides={s: dict(positions_rad=self.q[side_slice(s)].tolist()) for s in ("left", "right")},
                ),
            )
        elif label.startswith("inference_"):
            side = label.removeprefix("inference_")
            rows = []
            for i, distance in enumerate(
                self.distances[side] if isinstance(self.distances[side], list) else [self.distances[side]]
            ):
                legacy = candidate("right", distance, f"c{i:04}")
                rows.append(dict(candidate_id=f"c{i:04}", stages=legacy["targets"]["right"], raw_index=i))
            write_json(
                output / "candidates.json",
                dict(
                    candidates=rows,
                    checkpoint=self.config["inference_model"]["checkpoint"],
                    metadata=self.config["inference_model"]["metadata"],
                    preprocessing=dict(fixture=True),
                ),
            )
        else:
            raise AssertionError(f"Unexpected process launch: {label}")

    def start(self, command: list[str], label: str, **kw: object) -> None:
        self.calls.append((label, command))
        if label != "viewer":
            raise AssertionError(f"Motion service launch forbidden in fixture: {label}")
        write_json(self.root / "run/viewer_state.json", dict(fixture=True))

    def close(self) -> None:
        self.closed = True

    def invoke(self, options: list[str] | None = None, *, export_only: bool = True) -> int:
        """Exercise argparse, inference adaptation, selector, exporter and cancel gate without devices."""
        with ExitStack() as stack:
            for module, name, value in (
                (dual_runner, "configuration", lambda path: deepcopy(self.config)),
                (dual_runner, "Processes", lambda *args: self),
                (dual_runner, "perceive", self.perceive),
                (dual_runner, "live_scene", lambda *args: (self.robot, self.sim)),
                (dual_runner, "wait_for_enter", lambda processes: False),
                (dual_artifacts, "SideAdapter", AnalyticalSide),
                (approach, "HandGeometry", AnalyticalGeometry),
            ):
                stack.enter_context(patch.object(module, name, value))
            if not export_only:
                # Do not probe or start the user's actual 8089 service in fixture tests.
                probe = stack.enter_context(patch.object(dual_runner.socket, "socket"))
                probe.return_value.__enter__.return_value.connect_ex.return_value = 1
            return dual_runner.main(
                [
                    "--environment",
                    "mock",
                    "--ros-domain-id",
                    "88",
                    "--output-root",
                    str(self.root / "run"),
                    *(["--export-only"] if export_only else []),
                    *(options or []),
                ]
            )

    def artifact(self, name: str) -> dict:
        return read_json(self.root / "run" / name)
