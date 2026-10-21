"""Manual controls and the simulation loop, without a graphics window."""
import importlib.util
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gymnasium
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class KeyboardControlsTests(unittest.TestCase):
    def setUp(self):
        package = ModuleType('_keyboard_test')
        package.__path__ = []
        dynamics = ModuleType('_keyboard_test.dynamics')
        dynamics.HelicopterDynamics = Mock()
        dynamics.WindDynamics = Mock()
        renderer = ModuleType('_keyboard_test.renderer.api')
        renderer.Renderer = Mock()
        spec = importlib.util.spec_from_file_location(
            '_keyboard_test.helicopter', ROOT / 'heligym/envs/helicopter.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'_keyboard_test': package,
                                     '_keyboard_test.dynamics': dynamics,
                                     '_keyboard_test.renderer.api': renderer}):
            spec.loader.exec_module(module)
        self.env = module.Heli.__new__(module.Heli)
        self.env.pilot_mode = 'manual'
        self.env.paused = False
        self.env.step_count = 12
        self.env.current_reward = -0.5
        self.env.wind_enabled = True
        self.env.wind_dyn = SimpleNamespace(wind_speed=12, wind_dir=np.pi / 2,
                                            turbulence_level=3)
        self.env.heli_dyn = SimpleNamespace(
            action=np.zeros(4), observation=np.zeros(17), state={}, n_act=4,
            WIND_NED=np.array([1, 12, 0], dtype=np.float32)
        )
        self.env.task_target = {}
        self.env.previous_action = np.zeros(4, dtype=np.float32)
        self.env.renderer = Mock()

    def test_all_controls_change_the_correct_action(self):
        for key, index, delta in [('z', 0, .2), ('s', 0, -.2),
                                  ('up', 1, .2), ('down', 1, -.2),
                                  ('right', 2, .2), ('left', 2, -.2),
                                  ('d', 3, .2), ('q', 3, -.2)]:
            with self.subTest(key=key):
                self.env.renderer.get_key_presses.return_value = [key]
                original = np.zeros(4)
                result = self.env.process_keyboard(original, update_action=True)
                expected = np.zeros(4)
                expected[index] = delta
                np.testing.assert_allclose(result, expected)
                np.testing.assert_array_equal(original, np.zeros(4))
                np.testing.assert_allclose(self.env.heli_dyn.action, expected)

    def test_zero_controls_preserves_pause_and_event_order(self):
        for paused in (False, True):
            with self.subTest(paused=paused):
                self.env.paused = paused
                self.env.renderer.get_key_presses.return_value = ['x']
                delta = self.env.process_keyboard([.5, -.8, 1, -.2])
                np.testing.assert_allclose(delta, [-.5, .8, -1, .2])
                np.testing.assert_array_equal(self.env.heli_dyn.action, np.zeros(4))
                self.assertEqual(self.env.paused, paused)
        self.env.renderer.get_key_presses.return_value = ['z', 'x', 'right']
        actions = self.env.process_keyboard(np.ones(4))
        np.testing.assert_allclose(actions, [-1, -1, .2, -1])

    def test_clipping_and_event_order(self):
        self.env.renderer.get_key_presses.return_value = ['z', 'up', 'right', 'd'] * 20
        np.testing.assert_allclose(self.env.process_keyboard(np.zeros(4)), [.2, .2, .2, .2])
        self.env.renderer.get_key_presses.return_value = ['s', 'down', 'left', 'q'] * 20
        np.testing.assert_allclose(self.env.process_keyboard(np.zeros(4)), [-.2, -.2, -.2, -.2])
        self.env.renderer.get_key_presses.return_value = ['up', 'down']
        result = self.env.process_keyboard(np.ones(4))
        self.assertAlmostEqual(result[1], -.2)

    def test_pause_and_resume_and_control_changes_while_paused(self):
        self.env.renderer.get_key_presses.return_value = ['p', 'q']
        actions = self.env.process_keyboard(np.zeros(4))
        self.assertTrue(self.env.paused)
        self.assertAlmostEqual(actions[3], -.2)
        self.env.renderer.get_key_presses.return_value = ['p']
        np.testing.assert_array_equal(self.env.process_keyboard(actions), np.zeros(4))
        self.assertFalse(self.env.paused)

    def test_paused_render_only_services_window_events(self):
        self.env.paused = True
        self.env.render()
        self.env.renderer.render.assert_called_once_with(
            state=self.env.heli_dyn.state,
            observation=self.env.heli_dyn.observation,
            action=self.env.heli_dyn.action,
            paused=True,
            step=12,
            reward=-0.5,
            wind_enabled=True,
            wind_speed=12,
            wind_direction=90,
            turbulence_level=3,
            wind_ned=self.env.heli_dyn.WIND_NED,
            target=unittest.mock.ANY,
            target_position=None,
            pilot_mode='manual',
        )

    def test_demo_keeps_servicing_window_but_skips_physics_while_paused(self):
        trim_calls = []
        class TrimDynamics:
            n_act = 4

            def __init__(self):
                self.action = np.zeros(4)
                self.state = {'position': np.zeros(3)}

            def trim(self, conditions):
                trim_calls.append(conditions)
                self.state['position'][:] = 100
                self.action = np.array([.1, -.2, .3, -.4])

        self.env.heli_dyn = TrimDynamics()
        self.env.trim_cond = {'gr_alt': 100}
        self.env.reset = Mock()
        self.env.step = Mock(return_value=(None, 0, False, False, {}))
        self.env.close = Mock()
        self.env.render = Mock()
        self.env.renderer.is_close.side_effect = lambda: self.env.render.call_count >= 5
        self.env.renderer.get_key_presses.side_effect = [[], ['p'], ['z'], ['p'], []]
        heligym = ModuleType('heligym')
        heligym.HeliHover = Mock(return_value=self.env)
        with patch.dict(sys.modules, {'heligym': heligym}):
            runpy.run_path(str(ROOT / 'test.py'), run_name='__main__')
        self.assertEqual(self.env.render.call_count, 5)
        self.assertEqual(self.env.step.call_count, 3)
        self.env.close.assert_called_once()
        self.assertEqual(trim_calls, [])
        for call in self.env.step.call_args_list:
            np.testing.assert_array_equal(call.args[0], np.zeros(4))
        np.testing.assert_array_equal(self.env.heli_dyn.state['position'], np.zeros(3))

    def test_trim_key_applies_new_controls_before_subsequent_manual_corrections(self):
        trimmed = np.array([.1, -.2, .3, -.4], dtype=np.float32)
        self.env.trim_at_current_position = Mock(side_effect=lambda: setattr(
            self.env.heli_dyn, 'action', trimmed.copy()))
        self.env.renderer.get_key_presses.return_value = ['z', 't', 'up']
        delta = self.env.process_keyboard(update_action=True)
        self.env.trim_at_current_position.assert_called_once()
        np.testing.assert_allclose(delta, [0, .2, 0, 0])
        np.testing.assert_allclose(self.env.heli_dyn.action, [.1, 0., .3, -.4], atol=1e-7)

    def test_restart_resets_dynamics_timers_and_restores_trim_controls(self):
        self.env.paused = True
        self.env.time_counter = 10
        self.env.successed_time = 5
        self.env.wind_dyn = Mock()
        self.env.wind_dyn.wind_mean_ned = np.zeros(3)
        self.env.trim_cond = {'gr_alt': 100}
        self.env._bGuiText = True
        self.env._get_info = Mock(return_value={})
        self.env.heli_dyn.observation = np.zeros(17)
        self.env.heli_dyn.set_wind = Mock()
        trimmed = np.array([.1, -.2, .3, -.4], dtype=np.float32)
        self.env.heli_dyn.reset = Mock(side_effect=lambda _: setattr(
            self.env.heli_dyn, 'action', trimmed.copy()))
        self.env.renderer.get_key_presses.return_value = ['z', 'r', 'up']
        actions = self.env.process_keyboard(np.ones(4))
        self.assertFalse(self.env.paused)
        self.assertEqual(self.env.time_counter, 0)
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.current_reward, 0.0)
        self.assertEqual(self.env.successed_time, 0)
        self.env.wind_dyn.reset.assert_called_once()
        self.env.heli_dyn.reset.assert_called_once_with(self.env.trim_cond)
        np.testing.assert_allclose(actions, [.2, .2, 0, 0], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
