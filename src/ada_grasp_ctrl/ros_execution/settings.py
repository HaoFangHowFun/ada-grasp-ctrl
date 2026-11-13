"""Receiver defaults owned by Ada; importing them does not require ROS."""

from __future__ import annotations

import math
from typing import Any


DEFAULTS = {
    "environment": "mock",
    "reference_frame": "world",
    "tip_frame": "right_hand_flange",
    "force_limiter": "required",
    "rate_hz": 100.0,
    "feedback_timeout": 0.5,
    "startup_timeout": 20.0,
    "settle_timeout": 8.0,
    "settle_time": 0.3,
    "arm_speed": 0.15,
    "hand_speed": 0.4,
    "arm_acceleration": 0.3,
    "hand_acceleration": 0.8,
    "arm_step": 0.01,
    "hand_step": 0.02,
    "arm_tolerance": 0.01,
    "hand_tolerance": 0.02,
    "hand_arrival_required": True,
}


def bridge_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve limits and explicit backend selection; hardware still needs a reviewed goal.

    Returns:
        A new validated configuration, including internal preflight markers.

    Raises:
        ValueError: If an override has unsupported or invalid settings.
    """
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Receiver config must be a mapping")
    config = dict(DEFAULTS)
    if overrides is not None:
        unknown = set(overrides) - set(DEFAULTS) - {"side", "active_arms", "interpolation"}
        if unknown:
            raise ValueError(f"Unknown receiver settings: {sorted(unknown)}")
        config.update(overrides)
    side = config.get("side", "right")
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    active = config.get("active_arms", [side])
    if active not in (["left"], ["right"], ["left", "right"]) or side not in active:
        raise ValueError("active_arms must include the selected side")
    if overrides and "side" in overrides and "tip_frame" not in overrides:
        config["tip_frame"] = f"{side}_hand_flange"
    if config["environment"] not in ("mock", "hardware"):
        raise ValueError("environment must be mock or hardware")
    if "interpolation" in config and (
        config["interpolation"] != "linear" or config["environment"] != "mock" or side != "right"
    ):
        raise ValueError("New linear interpolation requires the right mock backend")
    if not isinstance(config["hand_arrival_required"], bool):
        raise ValueError("hand_arrival_required must be boolean")
    for key, default in DEFAULTS.items():
        if not isinstance(default, float):
            continue
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["rate_hz"] > 100 or config["force_limiter"] not in ("required", "disabled"):
        raise ValueError("Invalid command rate or limiter policy")
    if config["reference_frame"] not in ("world", f"{side}_base_link"):
        raise ValueError("reference_frame must be world or right_base_link")
    if config["tip_frame"] not in (f"{side}_hand_C_MC", f"{side}_hand_flange", f"{side}_fanuc_flange"):
        raise ValueError("tip_frame must be the right palm or a right flange")
    config.update(mode="joint", path_verified=False)
    return config


def validate_domain(environment: str, value: str | int | None) -> int:
    """Require an explicit domain; reserve domain 87 from mock execution."""
    if environment not in ("mock", "hardware") or value is None or isinstance(value, bool):
        raise ValueError("An explicit environment and ROS_DOMAIN_ID are required")
    try:
        domain = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError("ROS_DOMAIN_ID must be an integer") from error
    if str(domain) != str(value) or not 0 <= domain <= 232 or (environment == "mock" and domain == 87):
        raise ValueError("Invalid ROS_DOMAIN_ID; mock execution cannot use domain 87")
    return domain
