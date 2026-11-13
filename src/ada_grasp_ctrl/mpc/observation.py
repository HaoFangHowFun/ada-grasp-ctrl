"""Selected-reference, receipt-aligned tactile bridge to the unchanged Ours policy."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path


from ada_grasp_ctrl.contact import ContactObservation
from ada_grasp_ctrl.contact.compatibility import TactileContactEstimator
from ada_grasp_ctrl.experiments.right_ros_contract import (
    HAND_NAMES,
    check_kinematics,
    kinematic_contract,
    validate_packet,
)
from ada_grasp_ctrl.pipeline.plan import validate_plan
from ada_grasp_ctrl.tactile_preview.adapter import TactileAdapter
from ada_grasp_ctrl.tactile_preview.geometry import GeometryCatalog
from ada_grasp_ctrl.tactile_preview.kinematics import load_hand
from .contracts import JointFeedback
from .offline import make_runner


def selected_runner(report: dict, packet: dict, output: Path):
    """Use the selected packet trajectory; frozen report supplies robot/config only."""
    validate_packet(packet)
    validate_plan(packet["reviewed_plan"])
    if packet["source"].get("kind") != "grasp_selection":
        raise ValueError("P3 requires an actual selected grasp packet")
    check_kinematics(packet["kinematics"], kinematic_contract(Path(report["model"]["urdf"]).read_text()))
    runner = make_runner(report, packet["reviewed_plan"], output)
    runner.reference_provenance = dict(
        kind="selected_pipeline_packet",
        sample_id=packet["sample_id"],
        candidate_id=packet["reviewed_plan"]["candidate_id"],
        source=packet["source"],
        object_pose="not measured or required by native policy",
    )
    return runner


class ReceiptObservation:
    """One existing adapter + estimator; same-host receipt clocks, no acquisition claim."""

    def __init__(self, runner, package: Path) -> None:
        self.now = 0.0
        hand = load_hand(package / "urdf/right_sharpa_wave_with_flange.urdf", package, load_meshes=False)
        self.adapter = TactileAdapter(
            hand, GeometryCatalog.installed_candidate(), stale_after=0.2, pose_skew=0.15, clock=lambda: self.now
        )
        self.estimator = TactileContactEstimator(
            runner.robot_adaptor,
            object_bodies=("object",),
            tactile_joint_names=tuple(hand.actuated_joint_names),
            friction_mu=runner.grasp_ctrl.mu,
            allow_candidate_geometry=True,
            measured_force_policy="diagnostic",
            ambiguous_contact_policy="omit",
        )

    def estimate(self, now: float, feedback: JointFeedback, tactile: dict):
        self.now = now
        snapshot = self.adapter.update(
            tactile.get("frames", {}),
            tactile.get("joints", {}),
            errors=tactile.get("errors", {}),
            allow_unverified_geometry=True,
        )
        estimate = self.estimator.estimate(snapshot, feedback, now)
        source_issues = tactile.get("source_issues", {})
        bad = [f"{finger}: {reason}" for finger, reason in source_issues.items() if reason]
        provenance = dict(estimate.provenance, synchronization="receipt_aligned", acquisition_synchronized=False)
        # Keep each source status even when the compatibility pass stops at an invalid finger.
        provenance["adapter_snapshot"] = asdict(snapshot)
        provenance["source_issues"] = source_issues
        for finger, reason in source_issues.items():
            if reason and finger in provenance["finger_statuses"]:
                provenance["finger_statuses"][finger].update(interpretation="UNAVAILABLE", detail=reason)
        observation = estimate.observation
        if observation is None or bad:
            reasons = tuple(bad) + estimate.reasons
            observation = ContactObservation(now, (), "receipt_aligned_tactile", False, reasons)
        return observation, provenance

    @staticmethod
    def joint_packet(feedback: JointFeedback, hand_received: float, hand_ros_stamp: float) -> dict:
        return dict(
            names=list(HAND_NAMES),
            positions=list(feedback.positions_rad[6:]),
            received=hand_received,
            receipt_ros=hand_ros_stamp,
            subscriber_ros=hand_ros_stamp,
        )
