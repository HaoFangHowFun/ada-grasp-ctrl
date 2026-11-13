"""Native Python worker; stdin/stdout carry JSON only, with no ROS imports."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import sys


def main() -> None:
    """Initialize one original controller, then solve externally clocked requests."""
    # Preserve the maintained Torch-before-Pinocchio ABI initialization order.
    import torch  # noqa: F401
    import numpy as np
    from ada_grasp_ctrl.contact import ContactObservation
    from .contracts import JointFeedback
    from .native import NativeMPC
    from .offline import make_runner, json_value

    init = json.loads(sys.stdin.readline())
    if init.get("schema_version") != 1 or init.get("environment") not in ("mock", "hardware"):
        raise ValueError("Worker requires schema-1 initialization; this process has no actuator interface")
    with redirect_stdout(sys.stderr):
        sensing = None
        if init.get("packet") is not None:
            from .observation import ReceiptObservation, selected_runner

            runner = selected_runner(init["model_report"], init["packet"], Path(init["output"]))
            sensing = ReceiptObservation(runner, Path(init["package_share"]))
        else:
            runner = make_runner(init["model_report"], init["plan"], Path(init["output"]))
        native = NativeMPC(runner)
    print(json.dumps({"ready": True, "schema_version": 1}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("schema_version") != 1:
            raise ValueError("Unsupported MPC IPC schema")
        feedback = JointFeedback(**request["feedback"])
        with redirect_stdout(sys.stderr):
            provenance = {}
            if "tactile" in request:
                if sensing is None:
                    raise ValueError("Tactile requests require a selected packet")
                observation, provenance = sensing.estimate(request["now"], feedback, request["tactile"])
                q_ref = tuple(runner.qpos_path[min(native.index, len(runner.qpos_path) - 1)])
            else:
                raw = request["observation"]
                for contact in raw["contacts"]:
                    for field in (
                        "contact_pos",
                        "contact_pos_local",
                        "contact_frame",
                        "contact_frame_local",
                        "contact_force",
                    ):
                        contact[field] = np.asarray(contact[field], dtype=float)
                observation, q_ref = ContactObservation(**raw), request["q_ref"]
            proposal = native.tick(
                request["now"], feedback, request["q_cmd"], q_ref, observation, stage=request["stage"]
            )
            proposal.diagnostics["contact_provenance"] = provenance
            if sensing is not None:
                proposal.diagnostics["reference_provenance"] = runner.reference_provenance
        print(json.dumps(json_value(asdict(proposal)), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
