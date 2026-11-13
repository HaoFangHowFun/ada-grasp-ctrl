"""Print five-finger contact observations using the existing read-only ROS worker."""

from __future__ import annotations

import json
import time

from .__main__ import RosProcess, parse_args
from .adapter import TactileAdapter
from .geometry import GeometryCatalog
from .kinematics import load_hand


def main(argv: list[str] | None = None) -> None:
    """Run without Viser, SDK devices or command publishers; Ctrl-C closes the reader."""
    args = parse_args(argv)
    hand = load_hand(args.urdf, args.package_share, load_meshes=False)
    geometry = (
        GeometryCatalog.from_config(args.geometry_config.expanduser().resolve())
        if args.geometry_config
        else GeometryCatalog.installed_candidate()
    )
    adapter = TactileAdapter(
        hand,
        geometry,
        stale_after=args.stale_after,
        pose_skew=args.pose_skew,
        receipt_clock_synchronized=args.receipt_clock_synchronized,
    )
    reader = None
    try:
        if not args.offline:
            reader = RosProcess(args.ros_python, args.joint_topic)
        deadline = time.monotonic() + args.run_seconds if args.run_seconds else float("inf")
        printed = 0.0
        while time.monotonic() < deadline:
            packet, connected, error = reader.snapshot() if reader else ({}, False, "offline")
            snapshot = adapter.update(
                packet.get("frames", {}),
                packet.get("joints", {}),
                connected=connected,
                endpoints=packet.get("endpoints", {}),
                errors=packet.get("errors", {}),
                force_indices=args.force_indices,
                allow_unverified_geometry=args.allow_unverified_geometry,
            )
            # A single snapshot keeps contacts and statuses on exactly the same evaluation time.
            if snapshot.evaluated_at - printed >= 1:
                output = {}
                for finger, result in snapshot.fingers:
                    contact = result.observation
                    output[finger] = dict(
                        status=result.status,
                        detail=result.detail,
                        contact=None
                        if contact is None
                        else dict(
                            position=contact.position,
                            normal=contact.normal,
                            force=contact.force,
                            frame_id=contact.frame_id,
                            timestamp=contact.timestamp,
                            session_id=contact.session_id,
                            sequence=contact.sequence,
                            sdk_frame_id=contact.sdk_frame_id,
                            force_convention=contact.force_convention,
                            geometry=contact.geometry.verification,
                        ),
                    )
                print(json.dumps(dict(connected=connected, error=error, fingers=output), allow_nan=False), flush=True)
                printed = snapshot.evaluated_at
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.close()


if __name__ == "__main__":
    main()
