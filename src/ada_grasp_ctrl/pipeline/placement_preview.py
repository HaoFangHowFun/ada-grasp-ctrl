"""Cached source inference and revision-bound placement selection, with no execution export."""

from copy import deepcopy
from collections import Counter
from pathlib import Path

from ada_grasp_ctrl.grasp_selection.multi_arm import select_candidates
from ada_grasp_ctrl.grasp_selection.types import jsonable
from ada_grasp_ctrl.grasp_selection.adapters.dexlearn import read_json, file_hash
from ada_grasp_ctrl.placement.feasibility import PlacementGate
from ada_grasp_ctrl.placement.session import write_json
from .dual_inference import infer_live_sides
from .dual_scene import live_scene
from .dual_selection import select_live_hierarchical
from .simulation_plan import adjust_squeeze_candidates

LABEL = "Endpoint IK preview — transport path / placement stability not validated"


class PlacementPreview:
    """One settled observation's raw predictions; callers serialize worker access."""

    def __init__(self, config: dict, selector: str, root: Path, state: dict, processes: object) -> None:
        self.config, self.selector, self.root, self.state, self.processes = (
            deepcopy(config),
            selector,
            root,
            state,
            processes,
        )
        self.rows = self.native = None
        self.failure = None
        self.robot, self.static = live_scene(
            Path(config["robot_snapshot"]), state, read_json(Path(config["table_plane"]))
        )
        self.source_hash = file_hash(root / "object/object.npz")

    def prepare(self, target: dict, destination: Path, canceled: object) -> dict | None:
        """Infer once on the source and reevaluate endpoint IK for each confirmed target."""
        if file_hash(self.root / "object/object.npz") != self.source_hash:
            raise ValueError("Original observation changed")
        if self.failure:
            raise RuntimeError(self.failure)
        if self.rows is None:
            try:
                rows = infer_live_sides(self.root / "object", self.config, self.processes, self.root)
                self.rows = adjust_squeeze_candidates(
                    rows, self.config.get("simulation", {}).get("squeeze_extrapolate_ratio", 2.0)
                )
                write_json(self.root / "simulation_candidates.json", dict(schema_version=1, candidates=self.rows))
                if self.config["grasp_model"] == "native_multimode":
                    self.native = read_json(self.root / "candidates.json")
            except Exception as error:
                self.failure = f"Source inference failed; Reset Scene to retry: {type(error).__name__}: {error}"
                raise
        if canceled():
            raise RuntimeError("Placement evaluation canceled")
        gate = PlacementGate(self.robot, self.static, target, canceled)
        if self.selector == "hierarchical":
            selection, batch, audit = select_live_hierarchical(
                self.robot,
                self.static,
                self.rows,
                self.config["enabled_arms"],
                self.state["full_positions_rad"],
                self.config["hierarchical_selector"],
                native_diagnostics=self.native,
                single_only=True,
                endpoint_gate=gate,
            )
            write_json(destination / "canonical_candidates.json", batch)
            write_json(destination / "hierarchical_selection.json", audit)
        else:
            selection = select_candidates(
                self.robot,
                self.static,
                self.rows,
                self.config["enabled_arms"],
                self.state["full_positions_rad"],
                endpoint_gate=gate,
            )
        # Preserve original grasp reasons, distinguishing skipped and place-not-evaluated.
        selection = jsonable(selection)
        counts = dict(total=len(selection["evaluations"]), evaluated=0, grasp_feasible=0, place_feasible=0)
        by_id = {row["candidate_id"]: row for row in self.rows}
        for diagnostic in selection["evaluations"]:
            row = by_id[diagnostic["candidate_id"]]
            diagnostic.update(
                arms=row["required_arms"], mode=row.get("mode", row["provenance"].get("source_model_mode"))
            )
            attempts = diagnostic.get("attempts", [])
            evaluated = bool(attempts) or any(
                r in ("invalid_candidate", "hand_joint_limits") for r in diagnostic["reasons"]
            )
            grasp_pass = any(a.get("grasp_feasible") for a in attempts)
            place_pass = any(a.get("placement", {}).get("accepted") for a in attempts)
            diagnostic.update(
                grasp_status="passed" if grasp_pass else "rejected" if evaluated else "not_evaluated",
                place_status="passed" if place_pass else "rejected" if grasp_pass else "not_evaluated",
            )
            counts["evaluated"] += int(evaluated)
            counts["grasp_feasible"] += int(grasp_pass)
            counts["place_feasible"] += int(place_pass)
        selection.update(
            preview_only=True,
            target_revision=target["target_revision"],
            target_sha256=target["target_sha256"],
            source_sha256=self.source_hash,
            counts=counts,
            label=LABEL,
            rejection_counts=dict(
                Counter(
                    reason
                    for item in selection["evaluations"]
                    for attempt in item.get("attempts", [])
                    for reason in attempt.get("reasons", [])
                )
            ),
        )
        write_json(destination / "placement_selection_report.json", selection)
        chosen = selection["selected"]
        if chosen is None:
            return None
        place = chosen["placement"]
        result = dict(
            schema_version=1,
            kind="simulation_placement_endpoint_preview",
            preview_only=True,
            execution_ready=False,
            hardware_ready=False,
            label=LABEL,
            frame_id="world",
            joint_unit="rad",
            target_revision=target["target_revision"],
            target_sha256=target["target_sha256"],
            source_sha256=self.source_hash,
            source_snapshot_sha256=target["source_sha256"],
            candidate_id=chosen["candidate_id"],
            arm=chosen["selected_side"],
            mode=chosen["candidate"].get(
                "mode", chosen["candidate"]["provenance"].get("source_model_mode", "right_full")
            ),
            effort=chosen["effort"],
            counts=counts,
            full_joint_names=list(self.robot.joint_names),
            grasp_q56=place["source_held_q56"],
            place_q56=place["q56"],
            placement=place,
            grasp_feasible=True,
            target=target,
            simulation_pose_adjustment=chosen["candidate"]["provenance"].get("simulation_pose_adjustment"),
            references={
                str(path): file_hash(path)
                for path in (
                    self.root / "candidates.json",
                    self.root / "simulation_candidates.json",
                    self.root / "robot_state.json",
                    self.root / "scene.mjb",
                    self.root / "settled_state.npz",
                    destination / "placement_target.json",
                )
            },
        )
        write_json(destination / "placement_endpoint_preview.json", result)
        return result
