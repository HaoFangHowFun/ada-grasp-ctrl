"""Read-only dual tactile display projection using calibrated measured wrist FK."""

import numpy as np

from ada_grasp_ctrl.contact.force_display import FINGERS, SIDES, FingerForce, ForceSnapshot
from ada_grasp_ctrl.robot_environment.kinematics import rigid
from .geometry import transform_contact


def world_forces(adapters: dict, robot: object, measured: dict, *, now: float) -> ForceSnapshot:
    """Project fresh observations into Left CRX Base; clear unavailable fingers.

    Args:
        adapters: One existing TactileAdapter per available side.
        robot: Model from the same calibrated run snapshot as the joint feedback.
        measured: Canonical joint_names/positions_rad and same-host received timestamp.
        now: Same-host monotonic evaluation time (not ROS or simulation time).

    Returns:
        Ten-finger display snapshot. Aggregate F6 plus first-contact geometry cannot
        reconstruct the sum of per-contact compressive normal loads; these are None.
        Only fresh explicitly empty contact blocks yield a zero NO_CONTACT load.
    """
    rows, errors = [], []
    try:
        if measured["joint_names"] != list(robot.joint_names):
            raise ValueError("Measured full joint order differs")
        q = robot.configuration(measured["positions_rad"])
        received = float(measured["received"])
        if not np.isfinite([received, now]).all():
            raise ValueError("Invalid feedback clock")
    except (KeyError, TypeError, ValueError) as error:
        q, received = None, float("nan")
        errors.append(str(error))
    for side in SIDES:
        adapter = adapters.get(side)
        observations = {}
        wrist = None
        if adapter is not None:
            try:
                if adapter.side != side:
                    raise ValueError("Tactile adapter side differs")
                observations = dict(adapter.get_snapshot().fingers)
                if q is not None and 0 <= now - received <= adapter.stale_after:
                    wrist = rigid(robot.fk(q, side))
            except (ValueError, TypeError, KeyError) as error:
                errors.append(f"{side}: {error}")
        for finger in FINGERS:
            result = observations.get(finger)
            status = result.status if result is not None else "NO_DATA"
            frame = result.frame if result is not None else None
            valid_pose = bool(
                wrist is not None
                and frame is not None
                and 0 <= now - frame.received <= adapter.stale_after
                and abs(frame.received - received) <= adapter.pose_skew
            )
            if not valid_pose:
                rows.append(FingerForce(side, finger, "POSE_UNAVAILABLE" if result else status))
                continue
            if status == "EMPTY" and result.transport_status == "VALID" and frame.contact_count == 0:
                rows.append(FingerForce(side, finger, "NO_CONTACT", normal_load=0.0))
                continue
            observation = result.observation
            if status != "VALID" or observation is None:
                rows.append(FingerForce(side, finger, status))
                continue
            if observation.frame_id != f"{side}_hand_flange" or observation.force_convention != "object_on_finger":
                rows.append(FingerForce(side, finger, "FRAME_OR_CONVENTION_UNAVAILABLE"))
                continue
            point, normal, reaction = transform_contact(
                wrist,
                observation.position,
                observation.normal,
                observation.force,
            )
            # Existing sensing convention is object-on-finger. The overlay expects
            # the opposite object-directed resultant; normals are rotated, not mirrored.
            rows.append(
                FingerForce(side, finger, "VALID", tuple(point), tuple(-np.asarray(reaction)), tuple(normal), None)
            )
    return ForceSnapshot(now, tuple(rows), tuple(errors), "Hardware receipt monotonic")
