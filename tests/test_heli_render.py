"""Tests for the simulation-independent helicopter rendering component."""

from unittest.mock import Mock
import unittest

import numpy as np

from heligym.envs.heli_render import FT2MTR, HeliRender


class HeliRenderTests(unittest.TestCase):
    def setUp(self):
        self.native = Mock()
        self.native.create_model.side_effect = ["helicopter", "terrain", "sky"]
        self.native.create_guiText.side_effect = [1, 2, 3, 4, 5]
        self.native.is_close.return_value = False
        self.native.is_visible.return_value = False
        self.native.get_fps.return_value = 50.0
        self.renderer = HeliRender(native_renderer=self.native)
        self.native.reset_mock()
        self.state = {
            "betas": np.array([0.1, 0.2]),
            "psi_mr": np.array([0.3]),
            "psi_tr": np.array([0.4]),
            "xyz": np.array([10.0, 20.0, -30.0]),
            "euler": np.array([0.5, 0.6, 0.7]),
        }
        self.observation = np.arange(17, dtype=np.float32)
        self.action = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    def test_render_draws_snapshot_without_environment_reference(self):
        self.renderer.render(self.state, self.observation, self.action)
        self.native.rotate_MR.assert_called_once_with("helicopter", 0.2, 0.1, 0.3)
        self.native.rotate_TR.assert_called_once_with("helicopter", 0, 0.4, 0)
        self.native.translate_model.assert_any_call(
            "helicopter", *(self.state["xyz"] * FT2MTR)
        )
        self.native.rotate_model.assert_called_once_with("helicopter", 0.5, 0.6, 0.7)
        self.native.update_helicopter_camera.assert_called_once_with(
            10.0 * FT2MTR, 20.0 * FT2MTR, -30.0 * FT2MTR, 0.7
        )
        self.native.show_window.assert_called_once()
        self.native.render.assert_called_once()

    def test_simulation_metrics_are_separate_from_observations(self):
        simulation_values = self.renderer.simulation_values
        self.renderer.render(self.state, self.observation, self.action, step=42, reward=-2.5)
        np.testing.assert_array_equal(self.renderer.info_values, self.observation)
        np.testing.assert_array_equal(simulation_values, [50, 42, -2.5])
        self.assertFalse(any('FPS' in label for label in self.renderer.info_text))
        self.native.set_guiText.assert_any_call(
            self.renderer.simulation_id, self.renderer.simulation_text, simulation_values)
        # The C++ GUI retains this buffer, so updates must preserve its address.
        self.renderer.render(self.state, self.observation, self.action, step=43, reward=1.25)
        self.assertIs(self.renderer.simulation_values, simulation_values)
        np.testing.assert_array_equal(simulation_values, [50, 43, 1.25])

    def test_wind_panel_shows_settings_and_current_vector(self):
        wind_values = self.renderer.wind_values
        self.renderer.render(self.state, self.observation, self.action,
                             wind_enabled=True, wind_speed=12, wind_direction=90,
                             turbulence_level=3, wind_ned=np.array([3, 4, -12]))
        np.testing.assert_array_equal(wind_values, [1, 12, 90, 3, 13, 3, 4, -12])
        self.native.set_guiText.assert_any_call(
            self.renderer.wind_id, self.renderer.wind_text, wind_values)
        # Disabling wind clears even previously displayed enabled wind values.
        self.renderer.render(self.state, self.observation, self.action,
                             wind_enabled=False, wind_speed=12, wind_direction=90,
                             turbulence_level=3, wind_ned=np.array([3, 4, -12]))
        self.assertIs(self.renderer.wind_values, wind_values)
        np.testing.assert_array_equal(wind_values, np.zeros(8))

    def test_paused_render_only_services_events(self):
        self.renderer.render(self.state, self.observation, self.action, paused=True)
        self.native.wait_events.assert_called_once()
        self.native.render.assert_not_called()

    def test_close_releases_native_renderer(self):
        self.renderer.close()
        self.native.close.assert_called_once()
        self.native.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
