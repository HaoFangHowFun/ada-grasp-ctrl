"""Kinematic object editing for the existing Phase 2 scene and Phase 1 viewer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ada_grasp_ctrl.paths import resolve_from_root
from .simulated_object_cloud import ObjectScene, vector


class ObjectPlacementPreview:
    """Keep editable mesh-frame XYZ/WXYZ separate from robot joint controls.

    Args:
        source: Source scene YAML, resolved with the pipeline's data-root convention.
        workspace: Empty directory for the existing scene loader's converted meshes.
        destination: Optional new YAML path, resolved through the output root.

    Raises:
        ValueError: If the schema or destination is invalid.
        FileExistsError: If the explicit destination already exists.
    """

    def __init__(self, source: Path, workspace: Path, destination: Path | None = None) -> None:
        self.source = resolve_from_root(source)
        self.config = yaml.safe_load(self.source.read_text())
        if not isinstance(self.config, dict) or self.config.get("schema_version") != 1:
            raise ValueError("Expected scene schema_version: 1")
        self.destination = None if destination is None else resolve_from_root(destination, "output")
        if self.destination is not None:
            self._check_destination(self.destination)
        self.scene = ObjectScene(deepcopy(self.config), workspace)
        pose = self.config["object"]["pose"]
        self.position = np.array(pose["position_m"], dtype=float)
        self.wxyz = np.array(pose["quaternion_wxyz"], dtype=float)
        self.rpy_degrees = Rotation.from_quat(self.wxyz[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
        self._geom_groups = self.scene.sim.model.geom_group.copy()

    def _check_destination(self, path: Path) -> None:
        """Reject the source, aliases of it, and all existing output files."""
        if path.resolve() == self.source.resolve():
            raise ValueError("Save Scene must not overwrite the source YAML")
        if path.exists():
            raise FileExistsError(f"Save Scene requires a new file: {path}")

    def set_pose(self, position: object, rpy_degrees: object) -> None:
        """Set world XYZ in meters and fixed-axis XYZ roll/pitch/yaw in degrees.

        Rotation is Rz(yaw) @ Ry(pitch) @ Rx(roll). Translation never rotates about
        a mesh centroid. Preserve the original quaternion until rotation changes.
        """
        position = vector(position, 3, "object position")
        angles = vector(rpy_degrees, 3, "object roll/pitch/yaw")
        if not np.array_equal(angles, self.rpy_degrees):
            self.wxyz = Rotation.from_euler("xyz", angles, degrees=True).as_quat()[[3, 0, 1, 2]]
        self.position, self.rpy_degrees = position.copy(), angles.copy()
        self.apply_pose()

    def apply_pose(self) -> None:
        """Apply only the object transform after a robot reset, without stepping physics."""
        sim = self.scene.sim
        if self.scene.physics_available:
            joint = sim.data.joint("object_free")
            joint.qpos[:] = np.r_[self.position, self.wxyz]
            joint.qvel[:] = 0
        else:
            sim.model.body("object").pos[:] = self.position
            sim.model.body("object").quat[:] = self.wxyz
        mujoco.mj_forward(sim.model, sim.data)

    def save(self) -> Path:
        """Save only the object pose into a new YAML, retaining all other settings.

        Returns:
            Absolute saved filename. Default saves use a fresh UUID each time.

        Raises:
            ValueError: If the destination is the source YAML.
            OSError: If the file exists or cannot be written.
        """
        path = self.destination or resolve_from_root(
            Path("phase2/scene_edits") / f"{self.source.stem}_{uuid4().hex}.yaml", "output"
        )
        self._check_destination(path)
        config = deepcopy(self.config)
        # Replace mappings instead of mutating possible YAML aliases of unrelated fields.
        config["object"] = dict(config["object"])
        config["object"]["pose"] = {
            **config["object"]["pose"],
            "frame_id": "world",
            "position_m": self.position.tolist(),
            "quaternion_wxyz": self.wxyz.tolist(),
        }
        payload = yaml.safe_dump(config, sort_keys=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(payload)
        return path

    def perception_preview(self) -> tuple[np.ndarray, np.ndarray]:
        """Reuse Phase 2 camera rendering and configured cloud generation, without inference.

        Returns:
            RGB camera image and world-frame object points. No physics is advanced.

        Raises:
            ValueError: If the object has too few visible camera pixels.
        """
        sim = self.scene.sim
        display_groups = sim.model.geom_group.copy()
        try:
            # Phase 1 hides STL duplicates when it adds DAE colors in Viser. The
            # MuJoCo camera needs the original visual groups from the scene loader.
            sim.model.geom_group[:] = self._geom_groups
            if self.config["point_cloud"]["source"] == "simulated_depth":
                points, _ = self.scene.cloud()
            else:
                self.scene.depth_cloud()
                points, _ = self.scene.cloud()
            with np.load(self.scene.output / "depth.npz") as archive:
                rgb = archive["rgb"].copy()
        finally:
            sim.model.geom_group[:] = display_groups
        return rgb, points
