"""Tests for the simulation-independent helicopter rendering component."""

from unittest.mock import Mock
import unittest

import numpy as np

from heligym.envs.heli_render import FT2MTR, HeliRender


class HeliRenderTests(unittest.TestCase):
    def setUp(self):
        self.native = Mock()
        self.native.create_model.side_effect = ["helicopter", "terrain", "sky"]
        self.native.create_guiText.side_effect = [1, 2, 3]
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
