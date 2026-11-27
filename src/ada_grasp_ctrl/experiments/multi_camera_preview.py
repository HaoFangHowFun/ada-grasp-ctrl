"""Optional diagnostic point-cloud layers in the existing 8089 viewer."""

from pathlib import Path

import numpy as np

from .fixed_camera_geometry import file_hash
from .fixed_camera import read_json


def add_layers(server: object, root: Path) -> None:
    """Add source/seed/fused toggles without starting another server or stream."""
    info = read_json(root / "object/object.json")
    evidence = info["multi_camera"]
    for name in ("fused", "cameras"):
        if file_hash(Path(evidence[name + "_path"])) != evidence[name + "_sha256"]:
            raise ValueError(f"Multi-camera {name} artifact changed")
    layers = [
        ("D415", root / "capture/snapshot.npz", [70, 160, 255]),
        ("D405 left", root / "capture/wrist_left.npz", [255, 150, 60]),
        ("D405 right", root / "capture/wrist_right.npz", [190, 90, 245]),
        ("D415 seed", root / "d415_seed/object.npz", [255, 230, 60]),
        ("Fused object", root / "object/object.npz", [80, 220, 100]),
    ]
    with server.gui.add_folder("Camera layers"):
        for name, path, color in layers:
            with np.load(path, allow_pickle=False) as archive:
                points = archive["points_world_m"]
            points = points[:: max(1, len(points) // 50000)]
            handle = server.scene.add_point_cloud(
                "/multi_camera/" + name,
                points=points,
                colors=np.tile(color, (len(points), 1)),
                point_size=0.002,
                visible=name == "Fused object",
            )
            checkbox = server.gui.add_checkbox(name, initial_value=name == "Fused object")
            checkbox.on_update(
                lambda event, handle=handle, checkbox=checkbox: setattr(handle, "visible", checkbox.value)
            )
