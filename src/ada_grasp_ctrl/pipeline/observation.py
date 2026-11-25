"""Source-specific cloud validation shared by the existing inference workers."""

from pathlib import Path
import numpy as np

from ada_grasp_ctrl.experiments.fixed_camera_geometry import rigid, transform
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import file_hash, read_json


def read_observation(root: Path) -> tuple[dict, np.ndarray, dict]:
    """Validate world/metre data and retain camera evidence only for real captures."""
    info = read_json(root / "object.json")
    if (info.get("schema_version"), info.get("frame_id"), info.get("unit")) != (1, "world", "m"):
        raise ValueError("Unsupported object schema/frame/units")
    if file_hash(root / "object.npz") != info["object_sha256"]:
        raise ValueError("Object snapshot hash mismatch")
    with np.load(root / "object.npz", allow_pickle=False) as data:
        points = data["points_world_m"].copy()
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Object must contain finite Nx3 world points")
        if info.get("source_type") == "simulation_complete_mesh":
            if points.shape != (8192, 3) or set(data.files) != {"points_world_m"}:
                raise ValueError("Simulation observation requires exactly 8192 points and no oracle state")
            source = info["simulation_source"]
            for field in ("scene_sha256", "model_sha256", "mesh_sha256"):
                if not isinstance(source.get(field), str) or len(source[field]) != 64:
                    raise ValueError(f"Missing simulation provenance: {field}")
            return info, points, {}
        camera = rigid(data["T_world_color"])
        if not np.allclose(transform(data["points_color_m"], camera), points, atol=1e-7, rtol=0):
            raise ValueError("Object camera/world round-trip failed")
    return info, points, dict(T_world_color=camera)


def source_provenance(info: dict) -> dict:
    """Keep hardware lineage unchanged; never invent a simulation camera transform."""
    if info.get("source_type") == "simulation_complete_mesh":
        return dict(simulation_source=info["simulation_source"], input_point_source=info["source_type"])
    return dict(calibration=info["extrinsics"], input_point_source="live_measured_object")
