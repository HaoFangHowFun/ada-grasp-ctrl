"""Camera drivers and measured stationary extrinsics; no robot drivers."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from ada_grasp_ctrl.calibration import DEFAULT_CALIBRATION
from ada_grasp_ctrl.experiments.multi_camera import DEFAULT_CONFIG, load_config
from ada_grasp_ctrl.experiments.multi_camera_calibration import load_cameras
from ada_grasp_ctrl.experiments.stationary_camera_tf import require_camera_drivers_stopped
from ada_grasp_ctrl.paths import resolve_from_root
from bimanual_manipulation.cameras import load_camera_config


def bringup(context: object) -> list:
    """Validate the selected calibration and serial mapping before starting processes."""

    def value(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    calibration = resolve_from_root(value("calibration"))
    config_path = resolve_from_root(value("fusion_config"))
    config = load_config(config_path)
    load_cameras(calibration, config)
    roles = ["wrist_left", "wrist_right"]
    if value("enable_fix_camera") == "true":
        roles.append("fixed")
    driver_path = resolve_from_root(value("camera_config"))
    drivers = load_camera_config(driver_path, roles)
    for role, settings in drivers.items():
        if settings["serial_no"] != config["cameras"][role]["serial"]:
            raise ValueError(f"{role}: driver serial differs from calibrated serial")
        if not settings["publish_tf"]:
            raise ValueError(f"{role}: internal RealSense TF must be enabled")
    actions = []
    if value("start_cameras") == "true":
        require_camera_drivers_stopped(roles)
        share = Path(get_package_share_directory("bimanual_manipulation"))
        get_package_share_directory("realsense2_camera")
        actions.append(
            GroupAction(
                scoped=True,
                forwarding=False,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(str(share / "launch/cameras.launch.py")),
                        launch_arguments={
                            "enable_wrist_camera": "true",
                            "enable_fix_camera": value("enable_fix_camera"),
                            "camera_config": str(driver_path),
                        }.items(),
                    )
                ],
            )
        )
    publisher = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            "-m",
            "ada_grasp_ctrl.experiments.stationary_camera_tf",
            "--calibration",
            str(calibration),
            "--config",
            str(config_path),
        ],
        output="screen",
    )
    actions.extend(
        [
            RegisterEventHandler(
                OnProcessExit(
                    target_action=publisher,
                    on_exit=[EmitEvent(event=Shutdown(reason="Camera extrinsics publisher exited"))],
                )
            ),
            publisher,
        ]
    )
    return actions


def generate_launch_description() -> LaunchDescription:
    """Expose explicit reuse and calibration-file choices for the camera-only entry."""
    return LaunchDescription(
        [
            DeclareLaunchArgument("start_cameras", default_value="true", choices=["true", "false"]),
            DeclareLaunchArgument("enable_fix_camera", default_value="false", choices=["true", "false"]),
            DeclareLaunchArgument("calibration", default_value=DEFAULT_CALIBRATION),
            DeclareLaunchArgument("fusion_config", default_value=DEFAULT_CONFIG),
            DeclareLaunchArgument(
                "camera_config", default_value="src/ada_grasp_ctrl/config/experiments/multi_camera_driver.yaml"
            ),
            OpaqueFunction(function=bringup),
        ]
    )
