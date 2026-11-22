"""Keyboard-only rendering and handoff between human and agent pilots."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

import main
from heligym.envs.helicopter import Heli


class PilotEnv(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
    process_keyboard = Heli.process_keyboard

    def __init__(self, keys):
        self.paused = False
        self.heli_dyn = SimpleNamespace(action=np.full(4, .1), n_act=4)
        self.renderer = Mock()
        self.renderer.is_close.return_value = False
        self.renderer.get_key_presses.side_effect = keys
        self.render = Mock()
        self.actions = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.heli_dyn.action = np.full(4, .1)
        return np.array([0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.heli_dyn.action = action.copy()
        self.steps += 1
        return np.array([self.steps], dtype=np.float32), 1., False, self.steps == 3, {}

    def close(self):
        self.closed = True


class ManualRenderTests(unittest.TestCase):
    def test_manual_cli_needs_no_policy_or_training_imports(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'stable_baselines3'):
                raise AssertionError(f'Manual CLI must not import {name}')
            return original(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(builtins, '__import__', side_effect=guarded), \
                patch.object(main, 'prepare_renderer'), \
                patch.object(main, 'load_model') as load, \
                patch.object(main, 'run_episode', return_value=dict(steps=0, reward=0)) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            main.main(['render', '--manual', '--task', 'goto', '--target-dir', directory])
        load.assert_not_called()
        self.assertIsNone(run.call_args.args[0])
        self.assertEqual(run.call_args.args[1].commands, ['render'])
        self.assertEqual(run.call_args.args[1].task, 'goto')

    def test_manual_control_ignores_m_and_needs_no_normalization_file(self):
        args = main.parse_args(['render', '--manual'])
        env = PilotEnv([['m', 'z'], ['m', 'up'], ['x']])
        ignored_model = Mock()
        with patch.object(main, 'make_env', return_value=env) as factory, \
                patch.object(main.time, 'sleep'):
            result = main.run_episode(ignored_model, args, None, 42, render=True)
        self.assertFalse(factory.call_args.kwargs['monitor'])
        ignored_model.predict.assert_not_called()
        np.testing.assert_allclose(env.actions[0], [.3, .1, .1, .1])
        np.testing.assert_allclose(env.actions[1], [.3, .3, .1, .1])
        np.testing.assert_allclose(env.actions[2], [0, 0, 0, 0], atol=1e-7)
        self.assertEqual(env.pilot_mode, 'manual')
        self.assertEqual(result['steps'], 3)
        self.assertTrue(env.closed)

    def test_agent_manual_agent_handoff_ignores_keys_in_agent_mode(self):
        args = main.parse_args(['render', '--no-online-normalization'])
        env = PilotEnv([['z'], ['m', 'z'], ['m', 'z', 'r']])
        model = Mock()
        model.predict.return_value = (np.full((1, 4), .2), None)
        with patch.object(main, 'make_env', return_value=env), patch.object(main.time, 'sleep'):
            main.run_episode(model, args, None, 42, render=True)
        np.testing.assert_allclose(env.actions[0], [.2, .2, .2, .2])
        np.testing.assert_allclose(env.actions[1], [.4, .2, .2, .2])
        np.testing.assert_allclose(env.actions[2], [.2, .2, .2, .2])
        self.assertEqual(model.predict.call_count, 2)
        # The resumed policy receives the observation after the manual step.
        np.testing.assert_array_equal(model.predict.call_args.args[0], [[2]])
        self.assertEqual(env.pilot_mode, 'agent')
        self.assertTrue(env.closed)

    def test_manual_restart_resets_physics_and_preserves_keyboard_control(self):
        args = main.parse_args(['render', '--manual'])
        env = PilotEnv([['z'], ['r', 'up'], [], []])
        with patch.object(main, 'make_env', return_value=env), patch.object(main.time, 'sleep'):
            result = main.run_episode(None, args, None, 42, render=True)
        self.assertEqual(len(env.actions), 4)
        np.testing.assert_allclose(env.actions[0], [.3, .1, .1, .1])
        np.testing.assert_allclose(env.actions[1], [.1, .3, .1, .1])
        self.assertEqual(result['steps'], 3)
        self.assertEqual(result['reward'], 3.)
        self.assertEqual(env.pilot_mode, 'manual')

    def test_wind_key_works_in_both_pilot_modes(self):
        for manual in (False, True):
            args = main.parse_args(['render', '--no-online-normalization'] +
                                   (['--manual'] if manual else []))
            env = PilotEnv([['w'], [], ['w']])
            env.toggle_wind = Mock()
            model = Mock()
            model.predict.return_value = (np.full((1, 4), .2), None)
            with patch.object(main, 'make_env', return_value=env), patch.object(main.time, 'sleep'):
                main.run_episode(model, args, None, 42, render=True)
            self.assertEqual(env.toggle_wind.call_count, 2)

    def test_trim_key_refreshes_observation_in_both_modes_without_physics_step(self):
        for manual in (False, True):
            args = main.parse_args(['render', '--no-online-normalization'] +
                                   (['--manual'] if manual else []))
            env = PilotEnv([['t'], [], [], []])
            def trim():
                env.heli_dyn.action[:] = .3
                return np.array([42.], dtype=np.float32)
            env.trim_at_current_position = Mock(side_effect=trim)
            model = Mock()
            model.predict.return_value = (np.full((1, 4), .2), None)
            with patch.object(main, 'make_env', return_value=env), patch.object(main.time, 'sleep'):
                result = main.run_episode(model, args, None, 42, render=True)
            env.trim_at_current_position.assert_called_once()
            self.assertEqual(result['steps'], 3)
            self.assertEqual(env.render.call_count, 5)
            if manual:
                np.testing.assert_allclose(env.actions[0], [.3, .3, .3, .3])
            else:
                np.testing.assert_array_equal(model.predict.call_args_list[0].args[0], [[42.]])

    def test_replay_pauses_without_consuming_controls_and_hands_off(self):
        args = main.parse_args(['render', '--manual'])
        env = PilotEnv([['p'], ['p', 'z'], ['x'], ['up']])
        playback = np.array([[.2, .3, .4, .5], [.5, .4, .3, .2]])
        modes = []
        env.render.side_effect = lambda: modes.append(env.pilot_mode)
        with patch.object(main, 'make_env', return_value=env), patch.object(main.time, 'sleep'):
            result = main.run_episode(None, args, None, 42, render=True, replay_actions=playback)
        self.assertEqual(result['steps'], 3)
        np.testing.assert_array_equal(env.actions[:2], playback)
        np.testing.assert_allclose(env.actions[2], [.5, .6, .3, .2])
        self.assertEqual(modes, ['replay', 'replay', 'replay', 'manual', 'manual'])
        self.assertTrue(env.closed)

    def test_replay_uses_episode_seed_and_allows_environment_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main.write_json(root / 'config.json', {
                'commands': ['record'], 'target_dir': 'old/output', 'episode_seed': 305,
                'seed': 42, 'task': 'smooth-goto', 'episode_seconds': 120, 'no_wind': True,
            })
            (root / 'actions.csv').write_text('step,collective,longitudinal_cyclic,lateral_cyclic,pedal\n'
                                              '1,0.1,0.2,0.3,0.4\n')
            args = main.parse_args(['replay', '--source-dir', directory])
            self.assertEqual((args.seed, args.task, args.episode_seconds), (305, 'smooth-goto', 120))
            self.assertNotEqual(args.target_dir, Path('old/output'))
            with patch.object(main, 'prepare_renderer') as prepare, \
                    patch.object(main, 'load_model') as load, \
                    patch.object(main, 'run_episode', return_value=dict(
                        steps=1, reward=0., termination_reasons=['truncated'])) as run:
                main.main(['replay', '--source-dir', directory, '--seed', '19', '--task', 'hover', '--wind'])
            prepare.assert_called_once()
            load.assert_not_called()
            self.assertEqual(run.call_args.args[3], 19)
            self.assertEqual(run.call_args.args[1].task, 'hover')
            self.assertFalse(run.call_args.args[1].no_wind)
            np.testing.assert_array_equal(run.call_args.kwargs['replay_actions'],
                                          np.array([[.1, .2, .3, .4]], dtype=np.float32))

    def test_manual_pause_keeps_physics_still_and_holds_controls(self):
        args = main.parse_args(['render', '--manual'])
        env = PilotEnv([['p', 'z'], ['p'], [], []])
        with patch.object(main, 'make_env', return_value=env), patch.object(main.time, 'sleep'):
            main.run_episode(None, args, None, 42, render=True)
        self.assertEqual(len(env.actions), 3)
        for action in env.actions:
            np.testing.assert_allclose(action, [.3, .1, .1, .1])
        self.assertEqual(env.render.call_count, 5)


if __name__ == '__main__':
    unittest.main()
