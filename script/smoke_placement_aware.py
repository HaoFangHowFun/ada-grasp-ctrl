"""Drive the actual simulation CLI, UOP and native HUGS through the 8089 wire protocol."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import msgspec
import numpy as np
import viser
from websockets.sync.client import connect

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.placement.session import write_json, sha256


def main() -> None:
    """Retain fresh actual-model evidence; never stop another owner's viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8089)) == 0:
            raise SystemExit("8089 occupied; no server stopped")
    root.mkdir(parents=True, exist_ok=False)
    run = root / "session"
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["PYTHONPATH"] = str(project_root() / "src")
    # Forbid ROS imports in the real CLI, including preparation and model setup.
    bootstrap = """
import sys
class NoROS:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rclpy', 'rospy'):
            raise AssertionError('Placement-aware imported ROS')
sys.meta_path.insert(0, NoROS())
from ada_grasp_ctrl.pipeline.dual_runner import main
raise SystemExit(main())
"""
    command = [
        sys.executable,
        "-c",
        bootstrap,
        "--environment",
        "simulation",
        "--placement-aware",
        "--selector",
        "hierarchical",
        "--grasp-model",
        "native_multimode",
        "--arms",
        "auto",
        "--output-root",
        str(run),
    ]
    write_json(root / "invocation.json", dict(command=command))
    report = dict(schema_version=1, actual_models=True, preview_only=True, targets=[])
    with (root / "cli.log").open("w") as log:
        child = subprocess.Popen(command, cwd=project_root(), env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 90
            while not (run / "viewer_state.json").exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Viewer startup failed; see cli.log")
                time.sleep(0.1)
            # Use the same decoder as the existing real-web tests (no mocked runtime).
            sys.path.insert(0, str(project_root() / "tests"))
            from test_dual_mujoco_web import SimulationWebTests
            from test_object_placement_web import receive

            client = SimulationWebTests()
            with connect("ws://127.0.0.1:8089", subprotocols=[f"viser-v{viser.__version__}"], max_size=None) as ws:
                client.ws, client.gui, client.nodes = ws, {}, {}
                ws.send(
                    msgspec.msgpack.encode(
                        dict(
                            type="ViewerCameraMessage",
                            wxyz=[1, 0, 0, 0],
                            position=[2, -2, 2],
                            fov=1.0,
                            near=0.01,
                            far=100.0,
                            image_height=600,
                            image_width=800,
                            look_at=[0, 0, 0],
                            up_direction=[0, 0, 1],
                        )
                    )
                )
                client.wait_for(lambda: "Confirm Scene" in client.gui)
                attempt = client.confirm_scene(run, "PLACEMENT_EDITING")
                client.wait_for(lambda: "Confirm Placement Target" in client.gui)
                client.update("Run UOP-Net once", True)
                candidates_path = attempt / "placement/candidates.json"

                def wait(predicate, timeout=600):
                    limit = time.monotonic() + timeout
                    while not predicate():
                        if child.poll() is not None or time.monotonic() > limit:
                            raise RuntimeError("Integration timeout; see cli.log and artifacts")
                        try:
                            receive(ws)
                        except TimeoutError:
                            pass

                wait(lambda: json.loads(candidates_path.read_text())["inference_status"] not in ("not_run", "running"))
                candidates = json.loads(candidates_path.read_text())
                report["uop_status"] = candidates["inference_status"]
                report["uop_candidate_count"] = len(candidates["candidates"])
                # GUI handles retain their stable UUIDs while UOP updates the options.
                choices = [
                    "P00",
                    candidates["candidates"][1]["candidate_id"] if len(candidates["candidates"]) > 1 else "P00",
                ]
                original = np.load(attempt / "object/object.npz")["points_world_m"]
                xy = original.mean(0)[:2]
                source_hash = sha256(attempt / "object/object.npz")
                settled_hash = sha256(attempt / "settled_state.npz")
                for index, candidate_id in enumerate(choices):
                    if index:
                        client.update("Edit Placement Target", True)
                        wait(lambda: client.state(run)["state"] == "PLACEMENT_EDITING")
                        selected = next(c for c in candidates["candidates"] if c["candidate_id"] == candidate_id)
                        label = (
                            f"{candidate_id} — UOP-Net instance {selected['instance_id']}"
                            if candidate_id != "P00"
                            else "P00 — Original Pose"
                        )
                        client.update("Placement candidate", label)
                    for j, axis in enumerate(("X", "Y")):
                        client.update("Destination " + axis + " (m)", float(xy[j] + (0.04 if index and j == 0 else 0)))
                    time.sleep(0.3)
                    client.update("Confirm Placement Target", True)
                    wait(
                        lambda: (
                            client.state(run)["state"]
                            in ("PLACEMENT_PREVIEW_READY", "PLACEMENT_NO_FEASIBLE_GRASP", "PLACEMENT_ERROR")
                        )
                    )
                    state = client.state(run)
                    current = json.loads((attempt / "placement/target_state.json").read_text())
                    entry = dict(candidate_id=candidate_id, state=state, target_state=current)
                    if current.get("target"):
                        target_root = Path(current["target"]).parent
                        selection = target_root / "placement_selection_report.json"
                        if selection.exists():
                            result = json.loads(selection.read_text())
                            entry["counts"] = result["counts"]
                            entry["selected"] = result["selected"]
                    report["targets"].append(entry)
                    write_json(root / "smoke_report.json", report)
                    print(candidate_id, state["state"], entry.get("counts"), flush=True)
                    if state["state"] == "PLACEMENT_PREVIEW_READY":
                        for endpoint in ("Grasp", "Place"):
                            client.update("Endpoint", endpoint)
                            time.sleep(0.2)
                    if index == 0:
                        chrome = "/usr/bin/google-chrome"
                        if Path(chrome).exists():
                            with (root / "chrome.log").open("w") as browser_log:
                                subprocess.run(
                                    [
                                        chrome,
                                        "--headless",
                                        "--no-sandbox",
                                        "--disable-dev-shm-usage",
                                        "--use-angle=swiftshader",
                                        "--enable-unsafe-swiftshader",
                                        "--user-data-dir=" + str(root / "chrome_profile"),
                                        "--window-size=1600,1100",
                                        "--virtual-time-budget=5000",
                                        "--screenshot=" + str(root / "preview.png"),
                                        "http://127.0.0.1:8089",
                                    ],
                                    stdout=browser_log,
                                    stderr=browser_log,
                                    timeout=45,
                                    check=True,
                                )
                if len(candidates["candidates"]) == 1:
                    report["limitation"] = "UOP supplied no candidate; only P00 reviewed"
                assert sha256(attempt / "object/object.npz") == source_hash
                assert sha256(attempt / "settled_state.npz") == settled_hash
                assert not (attempt / "physics.jsonl").exists()
                assert not (attempt / "simulation_plan.json").exists()
                assert not (attempt / "export").exists()
                report.update(source_unchanged=True, settled_artifact_unchanged=True, execution_artifacts_absent=True)
                write_json(root / "smoke_report.json", report)
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)


if __name__ == "__main__":
    main()
