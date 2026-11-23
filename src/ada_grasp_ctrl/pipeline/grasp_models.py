"""Small deployment registry; the legacy reference remains in the live YAML."""

from pathlib import Path

from ada_grasp_ctrl.paths import resolve_from_root

SOURCE_MODES = ("right_two", "right_three", "right_full", "both_three", "both_full")
SINGLE_MODES = SOURCE_MODES[:3]


def configure_model(config: dict, name: str, selector: str) -> None:
    """Resolve the selected model before capture without changing legacy references."""
    if name not in ("right_full", "native_multimode"):
        raise ValueError(f"Unknown grasp model: {name}")
    if name == "native_multimode":
        if selector != "hierarchical":
            raise ValueError("native_multimode requires --selector hierarchical (Ranked Feasibility)")
        model = config["grasp_models"][name]
        for key in ("checkpoint", "training_config"):
            if not resolve_from_root(model[key]).is_file():
                raise ValueError(f"Missing native model {key}: {model[key]}")
        for mode in SOURCE_MODES:
            path = resolve_from_root(model["metadata_root"]) / mode / "metadata.json"
            if not Path(path).is_file():
                raise ValueError(f"Missing native mode metadata: {path}")
    config["grasp_model"] = name
