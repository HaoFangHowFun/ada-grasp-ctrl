"""Small world-frame force display contract and Viser overlay; no physics or ROS."""

from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np

from .force_hud import force_overlay

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ORANGE = (255, 140, 20)
BLUE = (40, 100, 255)
GREEN = (30, 210, 60)


@dataclass(frozen=True)
class FingerForce:
    """Visualization-only equivalent contact, never a relocated wrench or MPC input."""

    side: str
    finger: str
    status: str
    position: tuple[float, float, float] | None = None
    force: tuple[float, float, float] | None = None  # Object-directed, world frame, N.
    normal: tuple[float, float, float] | None = None
    normal_load: float | None = None  # Sum of original compressive loads, not |force| or normal @ force.


@dataclass(frozen=True)
class ForceSnapshot:
    """Copied contacts at one simulation time, separate from execution logs."""

    timestamp: float
    fingers: tuple[FingerForce, ...]
    errors: tuple[str, ...] = ()
    timestamp_clock: str = "simulation"


def normal_load_totals(snapshot: ForceSnapshot) -> tuple[float | None, float | None]:
    """Require all five fingers per side; never present a partial sum as a complete load."""
    totals = []
    for side in SIDES:
        rows = [row for row in snapshot.fingers if row.side == side]
        complete = len(rows) == len(FINGERS) and {row.finger for row in rows} == set(FINGERS)
        healthy = all(
            row.status in ("VALID", "NORMAL_DEGENERATE", "NO_CONTACT")
            and row.normal_load is not None
            and isfinite(row.normal_load)
            and row.normal_load >= 0
            and (row.status != "NO_CONTACT" or row.normal_load == 0)
            for row in rows
        )
        totals.append(fsum(row.normal_load for row in rows) if complete and healthy else None)
    return tuple(totals)


def update_arrow(handle: object, point: np.ndarray, vector: np.ndarray, scale: float) -> None:
    """Reuse the single-arm arrow endpoint convention and existing handle."""
    handle.points = np.array([[point, point + scale * vector]])
    handle.visible = True


class ForceOverlay:
    """Ten persistent fingertip groups accepting only world-frame display data."""

    def __init__(self, server: object) -> None:
        # A failed construction leaves the entire subtree hidden.
        self.root = server.scene.add_frame("/contact_forces", show_axes=False, visible=False)
        self.hud = server.gui.add_html("", visible=False)
        self.groups = {}
        with server.gui.add_folder("Contact Forces", expand_by_default=False):
            self.enabled = server.gui.add_checkbox("Enable contact forces", initial_value=True)
            self.sides = {s: server.gui.add_checkbox(f"Show {s} forces", initial_value=True) for s in SIDES}
            self.reaction = server.gui.add_checkbox("Reaction (orange)", initial_value=True)
            self.object_force = server.gui.add_checkbox("Object force (blue)", initial_value=True)
            self.normal = server.gui.add_checkbox("Normal (green)", initial_value=True)
            self.scale = server.gui.add_slider("Force scale (m/N)", min=0.001, max=0.1, step=0.001, initial_value=0.01)
            server.gui.add_markdown("Red: contact. Green: hand-to-object normal, visual length 0.025 m.")
            self.readout = server.gui.add_markdown("Hidden")
        for side in SIDES:
            for finger in FINGERS:
                path = f"/contact_forces/{side}/{finger}"
                marker = server.scene.add_icosphere(path + "/contact", radius=0.002, color=(255, 30, 30), visible=False)
                arrows = tuple(
                    server.scene.add_arrows(
                        path + "/" + name,
                        points=np.zeros((1, 2, 3)),
                        colors=color,
                        shaft_radius=0.0007 if name == "normal" else 0.001,
                        head_radius=0.002 if name == "normal" else 0.003,
                        head_length=0.005 if name == "normal" else 0.008,
                        visible=False,
                    )
                    for name, color in (("reaction", ORANGE), ("object", BLUE), ("normal", GREEN))
                )
                self.groups[side, finger] = (marker, *arrows)

    def clear(self) -> None:
        """Hide the subtree before touching individual handles, including on failure."""
        self.root.visible = False
        self.hud.visible = False
        self.hud.content = ""
        for group in self.groups.values():
            for handle in group:
                handle.visible = False
        self.readout.content = "Hidden"

    def update(self, snapshot: ForceSnapshot | None) -> None:
        """Replace all ten groups, clearing disappeared or unavailable contacts."""
        self.clear()
        if snapshot is None or not self.enabled.value:
            return
        lines = [
            f"{'Simulation' if snapshot.timestamp_clock == 'simulation' else snapshot.timestamp_clock} "
            f"{snapshot.timestamp:.3f} s · resultant N",
            "",
            "| Finger | Left | Right |",
            "|---|---:|---:|",
        ]
        values = {}
        for row in snapshot.fingers:
            key = row.side, row.finger
            values[key] = row.status
            if row.status not in ("VALID", "NORMAL_DEGENERATE"):
                continue
            point, force = np.asarray(row.position), np.asarray(row.force)
            if point.shape != (3,) or force.shape != (3,) or not np.isfinite([point, force]).all():
                raise ValueError(f"Invalid display data: {row.side}/{row.finger}")
            with np.errstate(over="raise", invalid="raise"):
                magnitude = float(np.linalg.norm(force))
            if not np.isfinite(magnitude):
                raise ValueError(f"Nonfinite display magnitude: {row.side}/{row.finger}")
            values[key] = f"{magnitude:.3f}" + (" (no normal)" if row.normal is None else "")
            if not self.sides[row.side].value:
                continue
            marker, reaction, object_force, normal = self.groups[key]
            marker.position, marker.visible = point, True
            # A valid cancellation is a measured zero; only the force arrows disappear.
            if magnitude > 0:
                if self.reaction.value:
                    update_arrow(reaction, point, -force, self.scale.value)
                if self.object_force.value:
                    update_arrow(object_force, point, force, self.scale.value)
            if row.normal is not None and self.normal.value:
                vector = np.asarray(row.normal)
                if vector.shape != (3,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1):
                    raise ValueError(f"Invalid display normal: {row.side}/{row.finger}")
                update_arrow(normal, point, vector, 0.025)
        for finger in FINGERS:
            lines.append(f"| {finger} | " + " | ".join(values.get((s, finger), "UNAVAILABLE") for s in SIDES) + " |")
        lines.extend(snapshot.errors)
        self.readout.content = "\n".join(lines)
        sides = normal_load_totals(snapshot)
        total = fsum(sides) if all(value is not None for value in sides) else None
        clock = "Simulation" if snapshot.timestamp_clock == "simulation" else snapshot.timestamp_clock
        status = f"{clock} {snapshot.timestamp:.3f} s"
        if total is None:
            status += " · UNAVAILABLE: incomplete fingertip data"
        self.hud.content = force_overlay(total, None, status, sides=sides)
        self.hud.visible = True
        self.root.visible = True
