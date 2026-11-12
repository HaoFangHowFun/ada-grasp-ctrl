"""Historical contact numerics, effective contact friction, and unchanged motor dynamics."""

import os
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import yaml

from ada_grasp_ctrl.experiments.sharpa_contact_profile import PROFILE, apply_contact_profile, diagnostic_step_interval
from ada_grasp_ctrl.experiments.simulated_object_cloud import ObjectScene
from ada_grasp_ctrl.paths import project_root


class ProfileTests(unittest.TestCase):
    def spec(self) -> mujoco.MjSpec:
        return mujoco.MjSpec.from_string("""<mujoco><worldbody>
        <geom name="table" type="plane" size="1 1 .1"/>
        <body name="object" pos="0 0 .049"><freejoint/><geom name="object" type="sphere" size=".05" mass=".1"/></body>
        </worldbody></mujoco>""")

    def test_effective_contact_restores_torsion_and_friction_with_gravity_enabled(self) -> None:
        spec = self.spec()
        gravity = spec.option.gravity.copy()
        meta = apply_contact_profile(spec, PROFILE, np.array([0.8, 0.02, 0.001]))
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.assertGreater(d.ncon, 0)
        self.assertEqual(d.contact[0].dim, 4)
        np.testing.assert_array_equal(d.contact[0].friction, [0.8, 0.8, 0.02, 0.001, 0.001])
        np.testing.assert_array_equal(m.opt.gravity, gravity)
        self.assertFalse(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))
        self.assertEqual(meta["name"], PROFILE)
        self.assertEqual(m.opt.cone, mujoco.mjtCone.mjCONE_ELLIPTIC)
        self.assertEqual(m.opt.impratio, 10)
        self.assertEqual(m.opt.noslip_iterations, 2)

    def test_source_profile_is_exact_noop_and_invalid_profile_is_rejected(self) -> None:
        spec = self.spec()
        before = spec.to_xml()
        self.assertIsNone(apply_contact_profile(spec, "source", np.array([0.8, 0.02, 0.001])))
        self.assertEqual(spec.to_xml(), before)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            apply_contact_profile(spec, "typo", np.array([0.8, 0.02, 0.001]))
        with self.assertRaisesRegex(ValueError, "requires declared"):
            apply_contact_profile(spec, PROFILE, np.array([0.6, 0.02, 0.001]))
        self.assertEqual(spec.to_xml(), before)

    def test_twenty_ms_checks_remain_twenty_ms_for_both_timesteps(self) -> None:
        self.assertEqual(diagnostic_step_interval(0.002), 10)
        self.assertEqual(diagnostic_step_interval(0.004), 5)
        with self.assertRaises(ValueError):
            diagnostic_step_interval(0.003)


@unittest.skipUnless(os.environ.get("ADA_DUAL_ARM_SNAPSHOT"), "Requires explicit Phase 1 snapshot")
class CompiledProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = yaml.safe_load((project_root() / "config/simulation_scene.yaml").read_text())
        self.config["scene"]["robot_snapshot"] = os.environ["ADA_DUAL_ARM_SNAPSHOT"]

    def scene(self, profile: str) -> ObjectScene:
        self.config["scene"]["contact_profile"] = profile
        out = self.root / profile
        out.mkdir()
        return ObjectScene(self.config, out)

    def test_profile_does_not_change_motors_geometry_mass_or_initial_state(self) -> None:
        source = self.scene("source")
        old = self.scene(PROFILE)
        a, b = source.sim.model, old.sim.model
        for field in [
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "actuator_gear",
            "actuator_ctrlrange",
            "dof_damping",
            "dof_armature",
            "dof_frictionloss",
            "body_mass",
            "body_inertia",
            "body_pos",
            "body_quat",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_contype",
            "geom_conaffinity",
        ]:
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)
        for field in ["qpos", "qvel", "ctrl"]:
            np.testing.assert_array_equal(getattr(source.sim.data, field), getattr(old.sim.data, field))
        self.assertNotIn("contact_profile", source.metadata)
        self.assertEqual(old.metadata["contact_profile"]["name"], PROFILE)
        self.assertEqual(b.opt.timestep, 0.004)

    @unittest.skipUnless(os.environ.get("ADA_SHARPA_REFERENCE_SCENE"), "Requires exact successful historical scene")
    def test_compiled_contact_settings_match_the_actual_successful_scene(self) -> None:
        scene = self.scene(PROFILE)
        current = scene.sim.model
        reference = mujoco.MjModel.from_xml_path(os.environ["ADA_SHARPA_REFERENCE_SCENE"])
        for key in ["timestep", "integrator", "cone", "impratio", "noslip_iterations", "iterations"]:
            self.assertEqual(getattr(current.opt, key), getattr(reference.opt, key), key)
        for key in ["geom_friction", "geom_condim", "geom_solref", "geom_solimp"]:
            a = np.unique(getattr(reference, key), axis=0)
            b = np.unique(getattr(current, key), axis=0)
            np.testing.assert_array_equal(a, b, err_msg=key)
        for i in range(reference.nu):
            name = reference.actuator(i).name
            if not name.startswith("right_") or name.startswith("right_J"):
                continue
            for side in ["left", "right"]:
                j = current.actuator(side + name[len("right") :]).id
                for field in ["actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_forcelimited"]:
                    np.testing.assert_array_equal(
                        getattr(reference, field)[i], getattr(current, field)[j], err_msg=name
                    )


if __name__ == "__main__":
    unittest.main()
