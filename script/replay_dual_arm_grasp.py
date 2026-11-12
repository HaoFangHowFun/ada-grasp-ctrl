"""Render saved Phase 2 physical states headlessly; this is playback, not a new grasp episode."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from imageio.v2 import get_writer, imwrite  # noqa: E402

from ada_grasp_ctrl.paths import resolve_from_root  # noqa: E402


def main() -> int:
    """Render a GIF or final PNG from the exact compiled model and recorded qpos trajectory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True, help="Fresh .gif or .png path")
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    root, output = resolve_from_root(args.run), resolve_from_root(args.output, "output")
    if output.exists() or output.suffix.lower() not in (".gif", ".png") or args.stride < 1:
        parser.error("Use a new .gif/.png output and a positive stride")
    config = yaml.safe_load((root / "effective_scene.yaml").read_text())
    camera = config["camera"]
    model = mujoco.MjModel.from_binary_path(str(root / "scene.mjb"))
    data = mujoco.MjData(model)
    with np.load(root / "trajectory.npz", allow_pickle=False) as archive:
        qpos, times = archive["qpos"].copy(), archive["time_s"].copy()
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError("Invalid recorded trajectory")
    options = mujoco.MjvOption()
    options.geomgroup[0] = 0
    options.geomgroup[3] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=camera["height"], width=camera["width"]) as renderer:

        def frame(index: int) -> np.ndarray:
            data.qpos[:] = qpos[index]
            data.time = times[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera["name"], scene_option=options)
            return renderer.render().copy()

        if output.suffix.lower() == ".png":
            imwrite(output, frame(-1))
        else:
            interval = float(np.median(np.diff(times))) if len(times) > 1 else 0.02
            with get_writer(output, mode="I", duration=max(interval * args.stride, 0.001) * 1000, loop=0) as writer:
                for index in range(0, len(qpos), args.stride):
                    writer.append_data(frame(index))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
