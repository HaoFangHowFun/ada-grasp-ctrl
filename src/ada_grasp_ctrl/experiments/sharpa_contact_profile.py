"""Explicit historical and DexGraspBench contact profiles for private dual-arm scenes."""

from __future__ import annotations

import mujoco
import numpy as np

PROFILE = "sharpa_native_v3"
BENCH_PROFILE = "dexgraspbench"
BENCH_FRICTION = (0.8, 0.02, 0.0001)


def apply_contact_profile(spec: mujoco.MjSpec, name: str, friction: np.ndarray) -> dict | None:
    """Restore the successful CRX/Sharpa scene's contact settings without changing gravity.

    Args:
        spec: Fully assembled scene, including robot, object, and table.
        name: ``source``, historical ``sharpa_native_v3``, or robot-only ``dexgraspbench``.
        friction: Declared object friction; must match the selected profile exactly.

    Returns:
        Resolved profile metadata, or None for the unchanged source configuration.

    Raises:
        ValueError: On unknown profiles or conflicting declared object friction.
    """
    if name == "source":
        return None
    if name == BENCH_PROFILE:
        if not np.array_equal(friction, BENCH_FRICTION):
            raise ValueError("dexgraspbench requires declared object friction [0.8, 0.02, 0.0001]")
        # Bench hardens only the imported robot; object/table keep source stiffness.
        robot_geoms = [g for b in spec.bodies if b.name.startswith(("left_", "right_")) for g in b.geoms]
        if not robot_geoms:
            raise ValueError("dexgraspbench requires named left_/right_ robot bodies")
        for geom in spec.geoms:
            geom.friction = BENCH_FRICTION
            geom.condim = 4
        for geom in robot_geoms:
            geom.solref = [0.005, 1]
            geom.solimp[:3] = [0.5, 0.99, 0.0001]
        spec.option.timestep = 0.004
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        spec.option.noslip_iterations = 2
        return dict(
            name=BENCH_PROFILE,
            friction=list(BENCH_FRICTION),
            condim=4,
            robot_solref=[0.005, 1],
            robot_solimp_first_three=[0.5, 0.99, 0.0001],
            robot_geom_count=len(robot_geoms),
            object_table_stiffness="Preserve source solref/solimp",
            timestep_s=0.004,
            integrator="implicitfast",
            cone="elliptic",
            impratio=10,
            noslip_iterations=2,
            reference="HUGS-DexGraspBench/src/util/hand_util.py: MjHO._add_hand, _set_friction",
            gravity_policy="Preserve caller gravity/force policy",
        )
    if name != PROFILE:
        raise ValueError(f"Unknown scene.contact_profile: {name!r}")
    if not np.array_equal(friction, [0.8, 0.02, 0.001]):
        raise ValueError("sharpa_native_v3 requires declared object friction [0.8, 0.02, 0.001]")
    for geom in spec.geoms:
        geom.friction = [0.8, 0.02, 0.001]
        geom.condim = 4
        geom.solref = [0.005, 1]
        geom.solimp = [0.5, 0.99, 0.0001, 0.5, 2]
    spec.option.timestep = 0.004
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.noslip_iterations = 2
    return dict(
        name=PROFILE,
        friction=[0.8, 0.02, 0.001],
        condim=4,
        solref=[0.005, 1],
        solimp=[0.5, 0.99, 0.0001, 0.5, 2],
        timestep_s=0.004,
        integrator="implicitfast",
        cone="elliptic",
        impratio=10,
        noslip_iterations=2,
        reference="docs/crx-sharpa-right-smoke.md; successful g0000/ours scene 20261110T010645248343Z-21280",
        gravity_policy="Preserve loaded scene gravity; no object force override",
    )


def diagnostic_step_interval(timestep: float) -> int:
    """Keep the existing 20 ms checks/recording when using either 2 or 4 ms physics."""
    count = round(0.02 / timestep)
    if count < 1 or not np.isclose(count * timestep, 0.02, atol=1e-12, rtol=0):
        raise ValueError("Physics timestep must divide the 20 ms diagnostic interval")
    return count
