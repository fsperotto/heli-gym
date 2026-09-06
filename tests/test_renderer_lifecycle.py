"""Exercise shutdown without loading native libraries or opening a window."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
import numpy  # Load before temporarily patching sys.modules.


class RendererLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.native = Mock()
        self.native.is_close.return_value = False
        package = ModuleType('_renderer_test')
        package.__path__ = []
        package.pyapi = self.native
        path = Path(__file__).resolve().parents[1] / 'heligym/envs/renderer/api.py'
        spec = importlib.util.spec_from_file_location('_renderer_test.api', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'_renderer_test': package}):
            spec.loader.exec_module(module)
        self.renderer = module.Renderer()
        self.native.reset_mock()

    def test_repeated_close_render_and_terminate_are_safe(self):
        self.renderer.close()
        self.renderer.close()
        self.renderer.render()
        self.renderer.terminate()
        self.renderer.terminate()
        self.renderer.render()
        self.assertTrue(self.renderer.is_close())
        self.native.close.assert_called_once()
        self.native.render.assert_called_once()
        self.native.terminate.assert_called_once()
        self.native.is_close.assert_not_called()
        self.assertEqual([call[0] for call in self.native.mock_calls],
                         ['close', 'render', 'terminate'])

    def test_close_event_during_frame_cleans_up_once(self):
        self.native.is_close.side_effect = [False, True]
        self.renderer.render()
        self.renderer.render()
        self.renderer.close()
        self.assertEqual(self.native.render.call_count, 2)
        self.native.close.assert_called_once()
        self.assertTrue(self.renderer.is_close())

    def test_already_requested_close_cleans_up_once(self):
        self.native.is_close.return_value = True
        self.renderer.render()
        self.renderer.render()
        self.native.render.assert_called_once()
        self.native.close.assert_called_once()

    def test_normal_frame_keeps_window_open(self):
        self.renderer.render()
        self.native.render.assert_called_once()
        self.native.close.assert_not_called()
        self.assertFalse(self.renderer.is_close())

    def test_wait_events_does_not_render(self):
        self.renderer.wait_events()
        self.native.wait_events.assert_called_once()
        self.native.render.assert_not_called()

    def test_close_during_wait_cleans_up_once(self):
        self.native.is_close.side_effect = [False, True]
        self.renderer.wait_events()
        self.renderer.wait_events()
        self.native.wait_events.assert_called_once()
        self.native.close.assert_called_once()
        self.native.render.assert_called_once()  # Native cleanup branch only.

    def test_wait_after_termination_is_ignored(self):
        self.renderer.terminate()
        self.native.reset_mock()
        self.renderer.wait_events()
        self.assertEqual(self.native.mock_calls, [])

    def test_control_bars_preserve_signed_values_and_clamp_actions(self):
        for actions, expected in [([-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 1.0]),
                                  ([-2.0, -0.5, 0.25, 0.5], [-1.0, -0.5, 0.25, 0.5])]:
            self.renderer.set_control_bars(actions)
            window, values = self.native.set_control_bars.call_args.args
            self.assertEqual(window, self.renderer.window)
            numpy.testing.assert_array_equal(values, expected)

    def test_control_bars_ignore_updates_after_close(self):
        self.renderer.close()
        self.renderer.set_control_bars([0.0] * 4)
        self.native.set_control_bars.assert_not_called()

    def test_control_bars_reject_incorrect_action_count(self):
        with self.assertRaises(ValueError):
            self.renderer.set_control_bars([0.0] * 3)
        self.native.set_control_bars.assert_not_called()


if __name__ == '__main__':
    unittest.main()
