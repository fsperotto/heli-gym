"""Command selection, saved runs, and deterministic episode execution."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import gymnasium as gym
try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    DummyVecEnv = VecNormalize = None

import main


class CommandTests(unittest.TestCase):
    def test_default_and_selected_pipeline(self):
        self.assertEqual(main.parse_args([]).commands, list(main.COMMANDS))
        self.assertEqual(main.parse_args(['plot', 'eval', 'plot']).commands,
                         ['eval', 'plot'])
        self.assertEqual(main.parse_args(['--task', 'heli', 'train', '--seed', '7', 'eval']).commands,
                         ['train', 'eval'])
        self.assertEqual(main.parse_args(['render']).commands, ['render'])
        self.assertEqual(main.parse_args(['train']).n_checkpoints, 0)
        self.assertEqual(main.parse_args(['train', '--n-checkpoints', '4']).n_checkpoints, 4)

    def test_invalid_commands_and_removed_flags(self):
        for arguments in (['show'], ['--no-learn'], ['--no-render'], ['--wind', '--no-wind'],
                          ['--no-stats-plot'], ['eval', '--eval-episodes', '0'],
                          ['train', '--wind-speed', '-1'], ['train', '--wind-speed', 'nan'],
                          ['train', '--wind-direction', 'inf'], ['train', '--turbulence-level', '8'],
                          ['train', '--n-checkpoints', '-1'], ['train', '--checkpoint-interval', '10']):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main.parse_args(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_saved_settings_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            main.write_json(path / 'arguments.json', dict(algorithm='sac', task='heli',
                            no_online_normalization=True, no_wind=True, wind_speed=12,
                            wind_direction=90, turbulence_level=3, episode_seconds=2, commands=['train']))
            args = main.parse_args(['eval', '--target-dir', directory])
            self.assertEqual(args.algorithm, 'sac')
            self.assertEqual(args.task, 'heli')
            self.assertTrue(args.no_online_normalization)
            self.assertTrue(args.no_wind)
            self.assertEqual((args.wind_speed, args.wind_direction, args.turbulence_level), (12, 90, 3))
            self.assertEqual(args.commands, ['eval'])
            args = main.parse_args(['render', '--target-dir', directory,
                                    '--algorithm', 'ppo', '--episode-seconds', '3', '--wind',
                                    '--wind-speed', '20', '--wind-direction', '180', '--turbulence-level', '4'])
            self.assertEqual(args.algorithm, 'ppo')
            self.assertEqual(args.episode_seconds, 3)
            self.assertEqual((args.wind_speed, args.wind_direction, args.turbulence_level), (20, 180, 4))
            self.assertFalse(args.no_wind)
            args = main.parse_args(['train', '--target-dir', directory])
            self.assertEqual(args.algorithm, 'ppo')
            self.assertEqual(args.task, 'hover')
            self.assertEqual((args.wind_speed, args.wind_direction, args.turbulence_level), (None, None, None))
            self.assertFalse(args.no_wind)

    def test_plot_dispatch_does_not_load_training_dependencies(self):
        import builtins
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'stable_baselines3', 'heligym'):
                raise AssertionError(f'Unexpected import: {name}')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=checked_import), \
                patch.object(main, 'plot_statistics') as plot, \
                contextlib.redirect_stdout(io.StringIO()):
            main.main(['plot', '--target-dir', '/tmp/test-plot'])
        plot.assert_called_once_with(Path('/tmp/test-plot'))

    @unittest.skipIf(DummyVecEnv is None, 'requires training extras')
    def test_train_eval_reuses_new_model_without_plotting(self):
        model = Mock()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(main, 'train', return_value=model) as train, \
                patch.object(main, 'evaluate') as evaluate, \
                patch.object(main, 'load_model') as load, \
                patch.object(main, 'plot_statistics') as plot, \
                patch.object(main, 'run_episode') as episode, \
                contextlib.redirect_stdout(io.StringIO()):
            main.main(['eval', 'train', '--target-dir', directory])
        train.assert_called_once()
        self.assertIs(evaluate.call_args.args[0], model)
        load.assert_not_called()
        plot.assert_not_called()
        episode.assert_not_called()

    @unittest.skipIf(DummyVecEnv is None, 'requires training extras')
    def test_default_pipeline_finishes_with_render(self):
        events = []
        model = Mock()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(main, 'prepare_renderer') as prepare, \
                patch.object(main, 'train', side_effect=lambda *a: events.append('train') or model), \
                patch.object(main, 'evaluate', side_effect=lambda *a: events.append('eval')), \
                patch.object(main, 'plot_statistics', side_effect=lambda *a: events.append('plot')), \
                patch.object(main, 'run_episode', side_effect=lambda *a, **kw:
                             events.append('render') or dict(steps=2, reward=10)) as episode, \
                patch.object(main, 'load_model') as load, \
                contextlib.redirect_stdout(io.StringIO()):
            main.main(['--target-dir', directory])
        self.assertEqual(events, ['train', 'eval', 'plot', 'render'])
        prepare.assert_called_once()
        self.assertTrue(episode.call_args.kwargs['render'])
        self.assertIs(episode.call_args.args[0], model)
        load.assert_not_called()

    @unittest.skipIf(DummyVecEnv is None, 'requires training extras')
    def test_independent_eval_loads_existing_model(self):
        model = Mock()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(main, 'load_model', return_value=model) as load, \
                patch.object(main, 'train') as train, \
                patch.object(main, 'evaluate') as evaluate, \
                contextlib.redirect_stdout(io.StringIO()):
            main.main(['eval', '--target-dir', directory])
        train.assert_not_called()
        load.assert_called_once()
        self.assertIs(evaluate.call_args.args[0], model)

    def test_missing_model_and_statistics_explain_next_step(self):
        with tempfile.TemporaryDirectory() as directory:
            args = main.parse_args(['eval', '--target-dir', directory])
            with self.assertRaisesRegex(FileNotFoundError, 'run train first'):
                main.load_model(Mock(), args)
            with self.assertRaisesRegex(FileNotFoundError, 'run train or eval first'):
                main.plot_statistics(Path(directory))


class EpisodeEnv(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self):
        self.closed = False
        self.paused = False
        self.resets = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.steps = 0
        return np.array([2], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.array([2 + self.steps], dtype=np.float32), 5.0, False, self.steps == 2, {}

    def close(self):
        self.closed = True


@unittest.skipIf(DummyVecEnv is None, 'requires training extras')
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor

        torch.set_num_threads(1)
        cls.algorithm = PPO
        cls.monitor = Monitor

    def train_run(self, directory, timesteps=20, checkpoints=4, n_envs=1, normalize=True):
        arguments = ['train', '--target-dir', directory, '--total-timesteps', str(timesteps),
                     '--n-checkpoints', str(checkpoints), '--n-envs', str(n_envs),
                     '--vec-env', 'dummy', '--n-steps', '4', '--batch-size', '4',
                     '--net-arch', '8', '--n-epochs', '1', '--device', 'cpu', '--verbose', '0']
        if not normalize:
            arguments.append('--no-online-normalization')
        args = main.parse_args(arguments)
        history_lengths = []
        write_json = main.write_json

        def record_statistics(path, value):
            if path.name == 'learning_curves.json':
                history_lengths.append(len(value))
            write_json(path, value)

        with patch.object(main, 'make_env', side_effect=lambda *a, **kw: self.monitor(EpisodeEnv())), \
                patch.object(main, 'write_json', side_effect=record_statistics), \
                contextlib.redirect_stdout(io.StringIO()):
            model = main.train(self.algorithm, args)
        return model, history_lengths

    def test_checkpoints_preserve_matching_normalization_and_update_one_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, history_lengths = self.train_run(directory)
            checkpoints = sorted(root.glob('checkpoints/*/step_*'), key=lambda p: int(p.name[5:]))
            self.assertEqual([p.name for p in checkpoints], ['step_4', 'step_8', 'step_12', 'step_16'])
            self.assertEqual(history_lengths, [2, 4, 6, 8, 10])
            for checkpoint, steps in zip(checkpoints, (4, 8, 12, 16)):
                model = main.load_model(self.algorithm, main.parse_args(['eval', '--target-dir', str(checkpoint)]))
                self.assertEqual(model.num_timesteps, steps)
                normalization = VecNormalize.load(str(checkpoint / 'vecnormalize.pkl'), DummyVecEnv([EpisodeEnv]))
                try:
                    self.assertAlmostEqual(normalization.obs_rms.count, steps + 1 + 0.0001)
                finally:
                    normalization.close()
            self.assertEqual(len(json.loads((root / 'learning_curves.json').read_text())), 10)
            self.assertEqual(len(list(root.rglob('learning_curves.json'))), 1)
            self.assertFalse(list(root.rglob('*.tmp')))
            self.assertTrue((root / 'model.zip').is_file())
            self.assertTrue((root / 'vecnormalize.pkl').is_file())
            self.assertEqual(json.loads((root / 'arguments.json').read_text())['n_checkpoints'], 4)
            original_files = {p: p.read_bytes() for p in root.glob('checkpoints/*/step_*/*')}
            self.train_run(directory, timesteps=8, checkpoints=1)
            self.assertEqual(len(list(root.glob('checkpoints/*'))), 2)
            self.assertEqual(len(list(root.glob('checkpoints/*/step_*'))), 5)
            for path, content in original_files.items():
                self.assertEqual(path.read_bytes(), content)

    def test_parallel_environment_thresholds_use_total_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _ = self.train_run(directory, timesteps=23, checkpoints=4, n_envs=3, normalize=False)
            steps = sorted(int(p.name[5:]) for p in root.glob('checkpoints/*/step_*'))
            self.assertEqual(steps, [6, 9, 15, 18])
            self.assertEqual(model.num_timesteps, 24)  # PPO completes the rollout.
            self.assertFalse(list(root.rglob('vecnormalize.pkl')))
            for checkpoint in root.glob('checkpoints/*/step_*'):
                saved = json.loads((checkpoint / 'arguments.json').read_text())
                self.assertTrue(saved['no_online_normalization'])

    def test_zero_disables_intermediate_saves_but_preserves_final_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, history_lengths = self.train_run(directory, timesteps=8, checkpoints=0)
            self.assertFalse((root / 'checkpoints').exists())
            self.assertEqual(history_lengths, [4])
            self.assertTrue((root / 'model.zip').is_file())
            self.assertTrue((root / 'vecnormalize.pkl').is_file())

    def test_short_run_caps_checkpoints_and_avoids_a_duplicate_final_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _ = self.train_run(directory, timesteps=5, checkpoints=100, n_envs=2, normalize=False)
            self.assertEqual(sorted(int(p.name[5:]) for p in root.glob('checkpoints/*/step_*')), [2, 4])
            self.assertEqual(model.num_timesteps, 8)
        with tempfile.TemporaryDirectory() as directory:
            self.train_run(directory, timesteps=1, checkpoints=1, normalize=False)
            self.assertFalse((Path(directory) / 'checkpoints').exists())


@unittest.skipIf(DummyVecEnv is None, 'requires training extras')
class EpisodeTests(unittest.TestCase):
    def test_weather_defaults_overrides_and_no_wind_reach_simulator(self):
        from heligym.envs.helicopter import DEFAULT_WEATHER

        cases = [({}, DEFAULT_WEATHER['wind_speed'], DEFAULT_WEATHER['wind_direction'],
                  DEFAULT_WEATHER['turbulence_level'], True),
                 (dict(wind_speed=12, wind_direction=90, turbulence_level=3), 12, 90, 3, True),
                 (dict(wind_speed=0), 0, DEFAULT_WEATHER['wind_direction'],
                  DEFAULT_WEATHER['turbulence_level'], True),
                 (dict(no_wind=True, wind_speed=12, wind_direction=90, turbulence_level=3),
                  0, 90, 3, False)]
        for kwargs, speed, direction, turbulence, enabled in cases:
            with self.subTest(kwargs=kwargs):
                env = main.make_env('heli', 'aw109', 1, **kwargs)
                try:
                    helicopter = env.unwrapped
                    self.assertEqual(helicopter.wind_enabled, enabled)
                    self.assertEqual(helicopter.wind_dyn.wind_speed, speed)
                    self.assertAlmostEqual(helicopter.wind_dyn.wind_dir, np.deg2rad(direction))
                    self.assertEqual(helicopter.wind_dyn.turbulence_level, turbulence)
                    np.testing.assert_allclose(helicopter.wind_dyn.wind_mean_ned,
                                               speed * np.array([np.cos(np.deg2rad(direction)),
                                                                 np.sin(np.deg2rad(direction)), 0]),
                                               atol=1e-6)
                finally:
                    env.close()

    def test_headless_evaluation_uses_frozen_observation_stats_and_raw_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vecnormalize.pkl'
            normalized = VecNormalize(DummyVecEnv([EpisodeEnv]))
            normalized.reset()
            normalized.step(np.zeros((1, 1), dtype=np.float32))
            normalized.save(str(path))
            expected = normalized.normalize_obs(np.array([[2]], dtype=np.float32))
            count = normalized.obs_rms.count
            normalized.close()
            env = EpisodeEnv()
            args = main.parse_args(['eval', '--target-dir', directory, '--eval-episodes', '2'])
            model = Mock()
            model.predict.return_value = (np.zeros((1, 1), dtype=np.float32), None)
            with patch.object(main, 'make_env', return_value=env), \
                    patch.object(main.time, 'sleep') as sleep:
                result = main.run_episode(model, args, path, seed=123)
            self.assertEqual(result, dict(seed=123, steps=2, reward=10.0,
                                          terminated=False, truncated=True))
            np.testing.assert_allclose(model.predict.call_args_list[0].args[0], expected)
            for call in model.predict.call_args_list:
                self.assertTrue(call.kwargs['deterministic'])
            sleep.assert_not_called()
            self.assertTrue(env.closed)
            self.assertEqual(env.resets, 1)  # No automatic reset on termination.
            restored = VecNormalize.load(str(path), DummyVecEnv([EpisodeEnv]))
            self.assertEqual(restored.obs_rms.count, count)
            restored.close()

    def test_evaluation_persists_episode_results_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            args = main.parse_args(['eval', '--target-dir', directory, '--eval-episodes', '2'])
            results = [dict(seed=46, steps=2, reward=4, terminated=True, truncated=False),
                       dict(seed=47, steps=4, reward=8, terminated=False, truncated=True)]
            with patch.object(main, 'run_episode', side_effect=results) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                main.evaluate(Mock(), args, Path(directory) / 'vecnormalize.pkl')
            statistics = json.loads((Path(directory) / 'evaluation.json').read_text())
            self.assertEqual(statistics['episodes'], results)
            self.assertEqual(statistics['mean_reward'], 6)
            self.assertEqual(statistics['std_reward'], 2)
            self.assertEqual(statistics['mean_steps'], 3)
            self.assertEqual([call.args[3] for call in run.call_args_list],
                             [args.seed + args.n_envs, args.seed + args.n_envs + 1])

    def test_render_displays_frames_and_paces_the_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            args = main.parse_args(['render', '--no-online-normalization', '--target-dir', directory,
                                    '--wind-speed', '12', '--wind-direction', '90', '--turbulence-level', '3'])
            env = EpisodeEnv()
            env.renderer = Mock()
            env.renderer.is_close.return_value = False
            env.renderer.get_key_presses.return_value = []
            env.render = Mock()
            model = Mock()
            model.predict.return_value = (np.zeros((1, 1), dtype=np.float32), None)
            with patch.object(main, 'make_env', return_value=env) as factory, \
                    patch.object(main.time, 'sleep') as sleep:
                result = main.run_episode(model, args, None, seed=123, render=True)
            self.assertEqual(factory.call_args.kwargs['render_mode'], 'human')
            self.assertEqual(factory.call_args.kwargs['wind_speed'], 12)
            self.assertEqual(factory.call_args.kwargs['wind_direction'], 90)
            self.assertEqual(factory.call_args.kwargs['turbulence_level'], 3)
            self.assertEqual(env.render.call_count, 3)
            self.assertEqual(result['reward'], 10)
            self.assertTrue(result['truncated'])
            sleep.assert_called_once()
            self.assertTrue(env.closed)

    def test_render_pause_preserves_physics_and_resumes_policy_playback(self):
        with tempfile.TemporaryDirectory() as directory:
            args = main.parse_args(['render', '--no-online-normalization', '--target-dir', directory])
            env = EpisodeEnv()
            env.renderer = Mock()
            env.renderer.is_close.return_value = False
            env.renderer.get_key_presses.side_effect = [[], ['p'], [], ['p']]
            snapshots = []
            env.render = Mock(side_effect=lambda: snapshots.append(
                (env.paused, env.steps, env.resets)))
            model = Mock()
            model.predict.return_value = (np.zeros((1, 1), dtype=np.float32), None)
            with patch.object(main, 'make_env', return_value=env), \
                    patch.object(main.time, 'sleep') as sleep:
                result = main.run_episode(model, args, None, seed=123, render=True)
            self.assertEqual(snapshots, [(False, 0, 1), (False, 1, 1),
                                         (True, 1, 1), (True, 1, 1), (False, 2, 1)])
            self.assertEqual(model.predict.call_count, 2)
            self.assertEqual(result['steps'], 2)
            self.assertEqual(result['reward'], 10)
            sleep.assert_called_once()
            self.assertTrue(env.closed)

    def test_render_can_close_while_paused(self):
        with tempfile.TemporaryDirectory() as directory:
            args = main.parse_args(['render', '--no-online-normalization', '--target-dir', directory])
            env = EpisodeEnv()
            env.renderer = Mock()
            env.renderer.get_key_presses.return_value = ['p']
            env.render = Mock()
            env.renderer.is_close.side_effect = lambda: env.render.call_count >= 2
            model = Mock()
            with patch.object(main, 'make_env', return_value=env):
                result = main.run_episode(model, args, None, seed=123, render=True)
            self.assertEqual(result['steps'], 0)
            self.assertEqual(result['reward'], 0)
            self.assertTrue(env.paused)
            self.assertTrue(env.closed)
            model.predict.assert_not_called()

    def test_environment_is_closed_on_prediction_error(self):
        with tempfile.TemporaryDirectory() as directory:
            args = main.parse_args(['eval', '--no-online-normalization', '--target-dir', directory])
            env = EpisodeEnv()
            with patch.object(main, 'make_env', return_value=env):
                with self.assertRaisesRegex(RuntimeError, 'prediction failed'):
                    main.run_episode(Mock(predict=Mock(side_effect=RuntimeError('prediction failed'))),
                                     args, None, seed=123)
            self.assertTrue(env.closed)


if __name__ == '__main__':
    unittest.main()
