"""Lightweight joint and fixture contract for the experimental CRX-Sharpa example."""

ASSET = "assets/robot/crx5ia_sharpa_wave_left/model.urdf"
OBJECT = "examples/assets/object/DGN_2k/processed_data/core_bottle_15787789482f045d8add95bf56d3d2fa"
ARM_JOINTS = tuple(f"left_J{index}" for index in range(1, 7))
HAND_SUFFIXES = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
HAND_JOINTS = tuple(f"left_{suffix}" for suffix in HAND_SUFFIXES)
JOINTS = ARM_JOINTS + HAND_JOINTS
PALM_FRAME = "left_hand_C_MC"
OBJECT_SCALE = 0.06
OBJECT_MASS = 0.1
