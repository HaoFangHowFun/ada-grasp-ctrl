"""Interactive settling accepts bounded jitter and rejects persistent motion."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ada_grasp_ctrl.experiments.settling import PoseWindow, SettlingPolicy


class PoseWindowTests(unittest.TestCase):
    def observe(self, translation, angle, *, duration=0.6, support=lambda t: True):
        window = PoseWindow(SettlingPolicy())
        for now in np.arange(0, duration + 0.001, 0.002):
            quat = Rotation.from_euler("z", angle(now), degrees=True).as_quat()[[3, 0, 1, 2]]
            window.add(float(now), np.r_[translation(now), 0, 0, quat], support(now))
        return window.measure()

    def test_contact_jitter_does_not_require_near_zero_instantaneous_velocity(self):
        # 0.5 mm at 40 Hz has peak speed 0.126 m/s, but bounded visible movement.
        report = self.observe(lambda t: 0.0005 * np.sin(t * 80 * np.pi), lambda t: 0.2 * np.sin(t * 80 * np.pi))
        self.assertTrue(report["stable"])
        self.assertLess(report["position_span_m"], 0.0011)
        self.assertLess(report["orientation_span_deg"], 0.41)

    def test_rolling_sliding_and_falling_are_rejected(self):
        for translation, angle in (
            (lambda t: 0.02 * t, lambda t: 0),
            (lambda t: 0, lambda t: 20 * t),
            (lambda t: -4.9 * t * t, lambda t: 0),
        ):
            self.assertFalse(self.observe(translation, angle)["stable"])

    def test_full_window_catches_excursions_even_with_identical_endpoints(self):
        report = self.observe(lambda t: 0.02 * np.sin(2 * np.pi * t / 0.5), lambda t: 0, duration=0.5)
        self.assertFalse(report["stable"])
        self.assertGreater(report["position_span_m"], 0.039)

    def test_short_window_and_missing_support_cannot_pass(self):
        self.assertFalse(self.observe(lambda t: 0, lambda t: 0, duration=0.2)["stable"])
        self.assertFalse(self.observe(lambda t: 0, lambda t: 0, support=lambda t: t > 0.2)["stable"])

    def test_quaternion_sign_does_not_change_stability(self):
        window = PoseWindow(SettlingPolicy())
        for i in range(251):
            window.add(i * 0.002, np.array([0, 0, 0, (-1) ** i, 0, 0, 0]), True)
        self.assertTrue(window.measure()["stable"])

    def test_malformed_policy_pose_and_clock_fail_clearly(self):
        for config in (
            {"timeout_s": 0},
            {"window_s": 11},
            {"position_span_m": float("nan")},
            {"orientation_span_deg": True},
            {"unused": 1},
        ):
            with self.assertRaises(ValueError):
                SettlingPolicy.from_config(config)
        window = PoseWindow(SettlingPolicy())
        with self.assertRaises(ValueError):
            window.add(0, np.zeros(7), True)
        pose = np.array([0, 0, 0, 1, 0, 0, 0])
        window.add(0, pose, True)
        with self.assertRaises(ValueError):
            window.add(0, pose, True)
        with self.assertRaises(ValueError):
            window.add(float("nan"), pose, True)


if __name__ == "__main__":
    unittest.main()
