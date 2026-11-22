"""Episode seeds reproduce physics and task state independently of other workers."""

import json
import unittest

import numpy as np
try:
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    DummyVecEnv = None

import main


class EpisodeReproducibilityTests(unittest.TestCase):
    @unittest.skipIf(DummyVecEnv is None, "requires training extras")
    def test_training_worker_and_evaluation_environment_match(self):
        train = DummyVecEnv([
            lambda: main.make_env('smooth-goto', 'aw109', 40., continual=True, random_wind=True),
            lambda: main.make_env('smooth-goto', 'aw109', 40., continual=True, random_wind=True),
        ])
        self.addCleanup(train.close)
        evaluation = main.make_env('smooth-goto', 'aw109', 40., continual=True,
                                   random_wind=True, monitor=False)
        self.addCleanup(evaluation.close)
        train.seed(42)
        training_obs = train.reset()
        observation, info = evaluation.reset(seed=43)
        np.testing.assert_array_equal(training_obs[1], observation)
        self.assertEqual(info['episode_seed'], 43)
        worker = train.envs[1].unwrapped
        self.assertEqual(worker.wind_dyn.wind_speed, evaluation.wind_dyn.wind_speed)
        self.assertEqual(worker.wind_dyn.wind_dir, evaluation.wind_dyn.wind_dir)

        controls = np.stack([env.unwrapped.previous_action for env in train.envs])
        for _ in range(100):
            training_obs, training_rewards, done, training_info = train.step(controls)
            observation, reward, terminated, truncated, info = evaluation.step(controls[1])
            self.assertFalse(terminated or truncated)
            self.assertFalse(done[1])
            self.assertEqual(training_info[1]['episode_seed'], info['episode_seed'])
            np.testing.assert_array_equal(training_obs[1], observation)
            np.testing.assert_array_equal(worker.heli_dyn.state.val, evaluation.heli_dyn.state.val)
            np.testing.assert_array_equal(worker.heli_dyn.WIND_NED, evaluation.heli_dyn.WIND_NED)
            self.assertEqual(training_rewards[1], np.float32(reward))

    def test_automatic_reset_seed_can_be_replayed(self):
        env = main.make_env('goto', 'aw109', 40., random_wind=True, monitor=False)
        self.addCleanup(env.close)
        env.reset(seed=42)
        first, info = env.reset()
        control = env.previous_action.copy()
        next_observation, reward, *_ = env.step(control)
        replay, _ = env.reset(seed=info['episode_seed'])
        np.testing.assert_array_equal(first, replay)
        observation, replay_reward, *_ = env.step(control)
        np.testing.assert_array_equal(next_observation, observation)
        self.assertEqual(reward, replay_reward)


    def test_json_snapshot_preserves_flight_and_waypoint_continuation(self):
        from heligym.envs.episode_state import capture_episode_state, restore_episode_state

        for task in ('hover', 'smooth-goto'):
            with self.subTest(task=task):
                options = dict(random_wind=True, monitor=False, continual=True)
                original = main.make_env(task, 'aw109', 40., **options)
                restored = main.make_env(task, 'aw109', 40., **options)
                try:
                    original.reset(seed=157)
                    for _ in range(5):
                        original.step(original.previous_action)
                    snapshot = json.loads(json.dumps(capture_episode_state(original), allow_nan=False))
                    restored.reset(seed=1)
                    observation = restore_episode_state(restored, snapshot)
                    np.testing.assert_array_equal(observation, original._agent_observation(original.heli_dyn.observation))
                    if task == 'smooth-goto':
                        original._draw_target()
                        restored._draw_target()
                        self.assertEqual(restored.get_target(), original.get_target())
                    for _ in range(10):
                        action = original.previous_action.copy()
                        expected, reward, terminated, truncated, _ = original.step(action)
                        actual, restored_reward, restored_terminal, restored_truncated, _ = restored.step(action)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual((restored_reward, restored_terminal, restored_truncated),
                                         (reward, terminated, truncated))
                finally:
                    original.close()
                    restored.close()
