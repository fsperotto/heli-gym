"""Task observations and seeded, continuing waypoint navigation."""
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

import heligym
import main


class TaskObservationTests(unittest.TestCase):
    def test_default_targets_follow_current_measurements(self):
        env = heligym.Heli(render_mode=None, wind_enabled=False)
        try:
            observation, _ = env.reset(seed=42)
            self.assertEqual(observation.shape, (25,))
            self.assertTrue(env.observation_space.contains(observation))
            np.testing.assert_array_equal(observation[:17], env.heli_dyn.observation.astype(np.float32))
            np.testing.assert_allclose(observation[17:20], observation[13:16])
            self.assertAlmostEqual(observation[20], np.linalg.norm(observation[4:7]))
            raw = env.heli_dyn.observation.copy()
            raw[13:16] = [12., 34., 56.]
            raw[4:7] = [3., 4., 0.]
            np.testing.assert_array_equal(env._agent_observation(raw)[17:21], [12, 34, 56, 5])
            env.set_target(dict(north_loc=100, sea_alt=500, vel=20))
            np.testing.assert_array_equal(env._agent_observation(raw)[17:21], [100, 34, 500, 20])
        finally:
            env.close()

    def test_hover_exposes_its_fixed_position_target(self):
        env = heligym.HeliHover(render_mode=None, wind_enabled=False)
        try:
            observation, _ = env.reset(seed=42)
            np.testing.assert_array_equal(observation[17:20], [0, 0, 2000])
        finally:
            env.close()


class GotoTests(unittest.TestCase):
    def setUp(self):
        self.env = heligym.HeliGoto(render_mode=None, wind_enabled=False)

    def tearDown(self):
        self.env.close()

    def test_seeded_targets_have_clearance_and_no_speed_constraint(self):
        env = self.env
        first, _ = env.reset(seed=42)
        initial = env.heli_dyn.observation[13:16].copy()
        target = first[17:20].copy()
        np.testing.assert_array_equal(env.reset(seed=42)[0], first)
        self.assertGreater(np.linalg.norm(target - initial), env.target_radius)
        self.assertTrue(np.all(np.abs(target[:2] - initial[:2]) <= 100.))
        self.assertGreaterEqual(target[2] + 1e-3,
                                env.heli_dyn.ground_height_at(*target[:2]) + 50.)
        self.assertNotIn('vel', env.get_target())
        self.assertAlmostEqual(first[20], np.linalg.norm(first[4:7]))
        other, _ = env.reset(seed=43)
        self.assertFalse(np.array_equal(other[17:20], target))

    def test_arrival_rewards_old_target_then_returns_new_target_without_termination(self):
        env = self.env
        env.reset(seed=42)
        position = env.heli_dyn.state['xyz']
        env.set_target(dict(north_loc=float(position[0]), east_loc=float(position[1]),
                            sea_alt=float(-position[2])))
        old_target = env._render_target_position().copy()
        expected_reward, _ = env._calculate_reward()
        # Freeze physics at the waypoint to test transition ordering precisely.
        with patch.object(env.heli_dyn, 'step', return_value=env.heli_dyn.observation.copy()):
            observation, reward, terminated, truncated, info = env.step(env.heli_dyn.action.copy())
        self.assertAlmostEqual(reward, expected_reward)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['targets_reached'], 1)
        self.assertFalse(np.array_equal(observation[17:20], old_target))
        np.testing.assert_array_equal(observation[17:20], info['target_position'])
        self.assertTrue(env.observation_space.contains(observation))
        env.successed_time = env.success_duration + 1
        self.assertFalse(env._is_successed())
        _, info = env.reset(seed=42)
        self.assertEqual(info['targets_reached'], 0)

    def test_cli_factory_and_gym_registration(self):
        self.assertEqual(main.parse_args(['train', '--task', 'goto']).task, 'goto')
        wrapped = main.make_env('goto', 'aw109', 1., no_wind=True)
        try:
            self.assertIsInstance(wrapped.unwrapped, heligym.HeliGoto)
            observation, _ = wrapped.reset(seed=42)
            self.assertEqual(observation.shape, (25,))
        finally:
            wrapped.close()
        self.assertEqual(gym.spec('HeliGoto-v0').entry_point, 'heligym.envs:HeliGoto')


if __name__ == '__main__':
    unittest.main()
