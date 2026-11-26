"""Drive the actual simulation CLI, UOP and native HUGS through the 8089 wire protocol."""

import argparse
import base64
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import msgspec
import numpy as np
import viser
from websockets.sync.client import connect

from ada_grasp_ctrl.paths import project_root
from ada_grasp_ctrl.placement.session import write_json, sha256


def capture_browser(root: Path, label: str) -> None:
    """Wait for the real page to receive its controls before capturing local Chrome."""
    chrome = "/usr/bin/google-chrome"
    if not Path(chrome).exists():
        return
    profile = root / (label + "_chrome_profile")
    with (root / (label + "_chrome.log")).open("w") as log:
        browser = subprocess.Popen(
            [
                chrome,
                "--headless",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--remote-debugging-port=0",
                "--user-data-dir=" + str(profile),
                "--window-size=1600,1100",
                "http://127.0.0.1:8089",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 60
            port_file = profile / "DevToolsActivePort"
            while not port_file.exists():
                if browser.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Chrome debug endpoint unavailable")
                time.sleep(0.1)
            port = int(port_file.read_text().splitlines()[0])
            with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
                pages = json.load(response)
            page = next(p for p in pages if p["type"] == "page")
            with connect(page["webSocketDebuggerUrl"], max_size=None) as devtools:
                serial = 0

                def call(method: str, params: dict) -> dict:
                    nonlocal serial
                    serial += 1
                    devtools.send(json.dumps(dict(id=serial, method=method, params=params)))
                    while True:
                        result = json.loads(devtools.recv(timeout=10))
                        if result.get("id") == serial:
                            if "error" in result:
                                raise RuntimeError(str(result["error"]))
                            return result["result"]

                while True:
                    content = call("Runtime.evaluate", dict(expression="document.body.innerText", returnByValue=True))
                    if "Confirm Placement Target" in content["result"].get("value", ""):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("Chrome page controls did not load")
                    time.sleep(0.2)
                time.sleep(2)
                png = call("Page.captureScreenshot", dict(format="png"))
                (root / (label + ".png")).write_bytes(base64.b64decode(png["data"]))
        finally:
            browser.terminate()
            try:
                browser.wait(timeout=10)
            except subprocess.TimeoutExpired:
                browser.kill()
                browser.wait(timeout=5)


def main() -> None:
    """Retain fresh actual-model evidence; never stop another owner's viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--placement-execute", action="store_true", help="Review full schedules and execute the final target"
    )
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
    if args.placement_execute:
        command.append("--placement-execute")
        write_json(root / "invocation.json", dict(command=command))
    report = dict(schema_version=1, actual_models=True, preview_only=not args.placement_execute, targets=[])
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
            with connect(
                "ws://127.0.0.1:8089",
                subprotocols=[f"viser-v{viser.__version__}"],
                max_size=None,
                max_queue=None,
                ping_timeout=None,
            ) as ws:
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
                    if index or args.placement_execute:
                        selected = next(c for c in candidates["candidates"] if c["candidate_id"] == candidate_id)
                        label = (
                            f"{candidate_id} — UOP-Net instance {selected['instance_id']}"
                            if candidate_id != "P00"
                            else "P00 — Original Pose"
                        )
                        client.update("Placement candidate", label)
                    for j, axis in enumerate(("X", "Y")):
                        offset = 0.02 if args.placement_execute else 0.04 if index else 0
                        client.update("Destination " + axis + " (m)", float(xy[j] + (offset if j == 0 else 0)))
                    time.sleep(0.3)
                    client.update("Confirm Placement Target", True)
                    wait(
                        lambda: (
                            client.state(run)["state"]
                            in (
                                "PREVIEW_READY",
                                "PLACEMENT_PREVIEW_READY",
                                "PLACEMENT_NO_FEASIBLE_GRASP",
                                "PLACEMENT_ERROR",
                            )
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
                    if state["state"] == "PREVIEW_READY":
                        plan = json.loads((target_root / "simulation_plan.json").read_text())
                        entry["preparation_s"] = plan["preparation_s"]
                        for phase in plan["execution_phases"]:
                            client.update("Phase", phase)
                            time.sleep(0.1)
                        for progress in (0, 37, 80, 100, 37):
                            client.update("Playback (%)", progress)
                            time.sleep(0.1)
                        assert not (target_root / "physics.jsonl").exists()
                    if index == 0 or args.placement_execute and state["state"] == "PREVIEW_READY":
                        capture_browser(root, candidate_id + "_preview")
                if len(candidates["candidates"]) == 1:
                    report["limitation"] = "UOP supplied no candidate; only P00 reviewed"
                assert sha256(attempt / "object/object.npz") == source_hash
                assert sha256(attempt / "settled_state.npz") == settled_hash
                if args.placement_execute and client.state(run)["state"] == "PREVIEW_READY":
                    client.update("Execute Grasp-to-Place", True)
                    wait(lambda: client.state(run)["state"] in ("COMPLETED", "ERROR"), timeout=600)
                    report["execution_state"] = client.state(run)
                    capture_browser(root, "execution_end")
                    wait(lambda: (target_root / "execution_report.json").exists())
                    report["execution"] = json.loads((target_root / "execution_report.json").read_text())
                    time.sleep(0.5)
                    report["physics_log_bytes"] = (target_root / "physics.jsonl").stat().st_size
                    client.update("Reset Scene", True)
                    wait(lambda: client.state(run)["state"] == "EDITING")
                    report["reset_after_execution"] = True
                elif args.placement_execute:
                    report["execution_blocked"] = client.state(run)
                assert not (attempt / "physics.jsonl").exists()
                assert not (attempt / "simulation_plan.json").exists()
                assert not (attempt / "export").exists()
                report.update(source_unchanged=True, settled_artifact_unchanged=True)
                if not args.placement_execute:
                    report["execution_artifacts_absent"] = True
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
