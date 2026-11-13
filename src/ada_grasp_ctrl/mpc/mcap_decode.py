"""ROS ABI-only MCAP decoder. No ROS context, nodes, discovery or publishers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path


def decode(bag: Path, output: Path) -> None:
    """Read CDR messages and preserve source clocks separately from bag receipt."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    from ada_grasp_ctrl.tactile_preview.data import TOPICS, normalize_frame, stamp_seconds

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    fingers = {topic: finger for finger, topic in TOPICS.items()}
    with output.open("x") as stream:
        while reader.has_next():
            topic, data, stamp = reader.read_next()
            msg = deserialize_message(data, types[topic])
            row = {"topic": topic, "bag_ns": stamp}
            if topic in fingers:
                frame = normalize_frame(msg, fingers[topic], stamp * 1e-9, stamp * 1e-9)
                row.update(
                    frame=asdict(frame),
                    receive_monotonic_ns=msg.receive_monotonic_ns,
                    contact_raw=list(msg.contact_points.raw_values),
                    contact_source_status=int(msg.contact_points.status),
                )
            else:
                row.update(
                    names=list(msg.name),
                    positions=list(msg.position),
                    source_ros_s=stamp_seconds(msg.header.stamp),
                    frame_id=msg.header.frame_id,
                )
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decode(args.bag, args.output)


if __name__ == "__main__":
    main()
