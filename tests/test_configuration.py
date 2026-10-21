"""Configuration files and explicit environment arguments stay independent."""
import unittest

import numpy as np

import heligym


class ConfigurationTests(unittest.TestCase):
    def test_named_yaml_files_and_weather_overrides(self):
        env = heligym.Heli(
            heli_name="aw109.yaml",
            terrain_name="default_terrain.yaml",
            render_mode=None,
            wind_direction=90.0,
            wind_speed=12.0,
            turbulence_level=3,
            gravity=30.0,
        )
        try:
            np.testing.assert_allclose(env.wind_dyn.wind_mean_ned, [0.0, 12.0, 0.0],
                                       atol=1e-6)
            self.assertEqual(env.wind_dyn.turbulence_level, 3)
            self.assertEqual(env.heli_dyn.gravity, 30.0)
            self.assertAlmostEqual(env.heli_dyn.HELI["M"],
                                   env.heli_dyn.HELI["WT"] / 30.0)
            self.assertEqual(env.heli_dyn.east_west_extent, 6561.6798)
            self.assertEqual(env.heli_dyn.north_south_extent, 6561.6798)
        finally:
            env.close()

    def test_no_wind_overrides_configured_mean_wind(self):
        env = heligym.Heli(render_mode=None, wind_enabled=False,
                           wind_direction=90.0, wind_speed=12.0)
        try:
            np.testing.assert_array_equal(env.wind_dyn.wind_mean_ned, np.zeros(3))
        finally:
            env.close()

    def test_wind_toggle_restores_settings_without_changing_aircraft_state(self):
        env = heligym.Heli(render_mode=None, wind_enabled=False,
                           wind_direction=90., wind_speed=12.)
        try:
            env.reset(seed=42)
            state = env.heli_dyn.state.val.copy()
            env.toggle_wind()
            self.assertTrue(env.wind_enabled)
            np.testing.assert_allclose(env.heli_dyn.WIND_NED, [0., 12., 0.], atol=1e-6)
            np.testing.assert_array_equal(env.heli_dyn.state.val, state)
            env.toggle_wind()
            self.assertFalse(env.wind_enabled)
            np.testing.assert_array_equal(env.heli_dyn.WIND_NED, np.zeros(3))
            env.step(env.heli_dyn.action.copy())
            np.testing.assert_array_equal(env.heli_dyn.WIND_NED, np.zeros(3))
            env.toggle_wind()
            env.reset(seed=42)
            np.testing.assert_allclose(env.heli_dyn.WIND_NED, [0., 12., 0.], atol=1e-6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
