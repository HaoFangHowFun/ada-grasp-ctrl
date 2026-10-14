"""Show the CRX-5iA and left Sharpa Wave asset without advancing physics."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).resolve().parents[1] / "assets/robot/crx5ia_sharpa_wave_left/model.urdf"


def main() -> None:
    """Open a static viewer for the imported arm and hand."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[0] = 0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.45]
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 125
        viewer.cam.elevation = -22
        print("Static asset preview: close the viewer window to exit.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
