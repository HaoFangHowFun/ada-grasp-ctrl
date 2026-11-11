"""Exercise the actual Viser dataset controls and timed playback on port 8089."""

import argparse
import json
from pathlib import Path
import time
import urllib.request

import msgspec.msgpack
import zstandard
import viser
from websockets.sync.client import connect


def main() -> int:
    """Switch every dataset, play the reviewed plan and leave group 01 selected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "http_status": urllib.request.urlopen("http://127.0.0.1:8089", timeout=5).status,
        "groups": [],
        "environment_point_cloud_seen": False,
        "lift_controls": [],
        "execution_blocked_groups": [],
        "cycle_controls": [],
    }
    controls, values = {}, {}
    with connect(
        "ws://127.0.0.1:8089", max_size=None, compression=None, subprotocols=["viser-v" + viser.__version__]
    ) as ws:
        ws.send(
            msgspec.msgpack.encode(
                {
                    "type": "ViewerCameraMessage",
                    "wxyz": [1, 0, 0, 0],
                    "position": [1.45, -1.75, 1.0],
                    "fov": 1.0,
                    "near": 0.01,
                    "far": 1000.0,
                    "image_height": 800,
                    "image_width": 1200,
                    "look_at": [0.35, -0.65, 0.2],
                    "up_direction": [0, 0, 1],
                }
            )
        )

        def pump() -> None:
            try:
                raw = ws.recv(timeout=0.2)
            except TimeoutError:
                return
            size = int.from_bytes(raw[8:16], "little")
            payload = msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(raw[16 : 16 + size]))
            for message in payload["messages"]:
                if message.get("name") == "/environment_points":
                    report["environment_point_cloud_seen"] = True
                props = message.get("props", {})
                if "label" in props:
                    controls[props["label"]] = message
                if message.get("type") == "GuiUpdateMessage":
                    values.setdefault(message["uuid"], {}).update(message["updates"])

        def wait(predicate, timeout: float = 15) -> None:
            deadline = time.monotonic() + timeout
            while not predicate():
                if time.monotonic() > deadline:
                    raise TimeoutError("Viser UI condition timed out")
                pump()

        def update(label: str, value: object) -> None:
            ws.send(
                msgspec.msgpack.encode(
                    {"type": "GuiUpdateMessage", "uuid": controls[label]["uuid"], "updates": {"value": value}}
                )
            )

        wait(lambda: all(name in controls for name in ("測資", "姿態", "播放姿態插值", "進度 %")))
        for label in controls["測資"]["props"]["options"]:
            update("測資", label)
            group = label.split(" · ")[0]
            wait(lambda: json.loads(args.status.read_text())["group"] == group)
            report["groups"].append(group)
            status = json.loads(args.status.read_text())
            phase_id = controls["姿態"]["uuid"]
            expected = ["目前手臂", *status.get("phases", ["pregrasp", "grasp", "squeeze"])]
            wait(lambda: values.get(phase_id, {}).get("options", controls["姿態"]["props"]["options"]) == expected)
            if status.get("execution_error"):
                report["execution_blocked_groups"].append(group)
            if "lift" in expected:
                for phase in [p for p in ("lift", "hold", "release", "return") if p in expected]:
                    update("姿態", phase)
                    progress_id = controls["進度 %"]["uuid"]
                    target_progress = status["phase_progress"][phase]
                    wait(lambda: abs(values.get(progress_id, {}).get("value", -100) - target_progress) < 0.11)
                report["lift_controls"].append(group)
                if "return" in expected:
                    report["cycle_controls"].append(group)
        update("測資", controls["測資"]["props"]["options"][0])
        wait(lambda: json.loads(args.status.read_text())["group"] == report["groups"][0])
        update("播放姿態插值", True)
        progress_id = controls["進度 %"]["uuid"]
        seen = []
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            pump()
            value = values.get(progress_id, {}).get("value")
            if value is not None:
                seen.append(value)
        assert len(set(seen)) > 2, seen
        update("播放姿態插值", False)
        report["playback_progress_changed"] = True
        assert report["environment_point_cloud_seen"], "Missing captured environment point cloud"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
