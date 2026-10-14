"""Experimental CRX-5iA and left Sharpa Wave control metadata."""

import numpy as np

from ada_grasp_ctrl.crx_sharpa_contract import ARM_JOINTS, ASSET, HAND_JOINTS, PALM_FRAME

from .base import Arm, ArmHand, Hand, register_robot


@register_robot("crx5ia_left")
class Crx5iaLeft(Arm):
    """Six-axis FANUC arm with one position actuator per joint."""

    def __init__(self, prefix: str = "") -> None:
        super().__init__(prefix)
        self.name = "crx5ia_left"
        self._dof_names = list(ARM_JOINTS)
        self._doa_names = list(ARM_JOINTS)
        self._doa2dof_matrix = np.eye(6)
        self._doa_kp = [100.0] * 6
        self._doa_max_vel = [2.61799, 2.61799, 3.14159, 3.92699, 3.92699, 3.92699]
        self._body_names = [f"left_J{index}_link" for index in range(1, 7)]
        self._base_name = "left_base_link"
        self._base_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self._urdf_path = ASSET


@register_robot("sharpa_wave_left")
class SharpaWaveLeft(Hand):
    """Twenty-two independently commanded Sharpa hand joints."""

    def __init__(self, prefix: str = "") -> None:
        super().__init__(prefix)
        self.name = "sharpa_wave_left"
        self._dof_names = list(HAND_JOINTS)
        self._doa_names = list(HAND_JOINTS)
        self._doa2dof_matrix = np.eye(22)
        self._doa_kp = [5.0] * 22
        self._doa_max_vel = [2.0] * 22
        self._body_names = [PALM_FRAME]
        self._base_name = PALM_FRAME
        self._base_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self._urdf_path = ASSET


@register_robot("crx_sharpa_left")
class CrxSharpaLeft(ArmHand):
    """Single left arm and hand used by the experimental adaptive example."""

    def __init__(self, prefix: str = "") -> None:
        super().__init__(prefix)
        self.name = "crx_sharpa_left"
        self.arm = Crx5iaLeft(prefix)
        self.hand = SharpaWaveLeft(prefix)
        self._base_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self._urdf_path = ASSET
