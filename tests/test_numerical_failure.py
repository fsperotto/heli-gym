"""Numerical failures must end episodes without poisoning RL observations."""
import unittest
from unittest.mock import patch

import numpy as np
import heligym


class NumericalFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = heligym.HeliHover(render_mode=None)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.observation, _ = self.env.reset(seed=42)
        self.raw_observation = self.observation[:self.env.heli_dyn.n_obs]
        self.action = self.env.heli_dyn.action.copy()

    def assert_finite_failure(self, transition):
        observation, reward, terminated, truncated, info = transition
        np.testing.assert_array_equal(observation[:self.env.heli_dyn.n_obs],
                                      self.raw_observation)
        np.testing.assert_allclose(observation[-self.env.heli_dyn.n_act:],
                                   np.clip(self.action, -1, 1))
        self.assertTrue(np.isfinite(reward))
        self.assertLess(reward, 0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info['failed'])
        self.assertTrue(info['numerical_failure'])
        self.assertEqual(self.env.step_count, 1)
        self.assertEqual(self.env.current_reward, reward)
        self.assertTrue(np.isfinite(self.env.heli_dyn.state.val).all())
        # Reset must recover even after a partially completed integration step.
        obs, _ = self.env.reset()
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(np.isfinite(self.env.step(self.env.heli_dyn.action)[0]).all())

    def test_nonfinite_state_is_caught_before_terrain_lookup(self):
        def diverge(action):
            self.env.heli_dyn.state['xyz'][:] = np.nan
            return self.raw_observation.copy()
        with patch.object(self.env.heli_dyn, 'step', side_effect=diverge):
            transition = self.env.step(self.action)
        self.assert_finite_failure(transition)

    def test_overflow_during_integration_restores_terminal_state(self):
        def overflow(action):
            self.env.heli_dyn.state['xyz'][:] = 123
            raise FloatingPointError('overflow encountered in cast')
        with patch.object(self.env.heli_dyn, 'step', side_effect=overflow):
            transition = self.env.step(self.action)
        self.assert_finite_failure(transition)

    def test_nonfinite_reward_does_not_reach_agent(self):
        with patch.object(self.env, '_calculate_reward', return_value=(np.nan, False)):
            transition = self.env.step(self.action)
        self.assert_finite_failure(transition)

    def test_reset_templates_do_not_alias_live_state(self):
        dynamics = self.env.heli_dyn
        self.assertFalse(np.shares_memory(dynamics.state.val, dynamics.init_state.val))
        self.assertFalse(np.shares_memory(dynamics.state_dots.val, dynamics.init_state_dots.val))
        baseline_state = dynamics.init_state.val.copy()
        baseline_dots = dynamics.init_state_dots.val.copy()
        dynamics.state.val[:] = 123
        dynamics.state_dots.val[:] = 456
        self.env.reset()
        np.testing.assert_array_equal(dynamics.init_state.val, baseline_state)
        np.testing.assert_array_equal(dynamics.init_state_dots.val, baseline_dots)
        np.testing.assert_allclose(dynamics.observation, self.raw_observation)

    def test_reset_discards_previous_episode_wind(self):
        # A failed turbulent episode must not change the next trim problem.
        self.env.heli_dyn.set_wind(np.full(3, 1e20))
        observation, _ = self.env.reset()
        np.testing.assert_allclose(observation, self.observation)
        np.testing.assert_allclose(self.env.heli_dyn.action, self.action)

    def test_observation_contains_previous_direct_action(self):
        observation, _ = self.env.reset()
        self.assertEqual(observation.shape, (self.env.heli_dyn.n_obs + self.env.n_task_obs + self.env.heli_dyn.n_act,))
        np.testing.assert_allclose(observation[-self.env.heli_dyn.n_act:],
                                   self.env.heli_dyn.action)
        action = np.array([0.2, -0.3, 0.4, -0.5], dtype=np.float32)
        observation, _, _, _, _ = self.env.step(action)
        np.testing.assert_array_equal(observation[-self.env.heli_dyn.n_act:], action)

    def test_simulation_metrics_follow_steps_and_reset(self):
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.current_reward, 0.0)
        for step in (1, 2):
            _, reward, _, _, _ = self.env.step(self.action)
            self.assertEqual(self.env.step_count, step)
            self.assertEqual(self.env.current_reward, reward)
        self.env.reset()
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.current_reward, 0.0)

    def test_invalid_actions_are_rejected(self):
        with self.assertRaises(ValueError):
            self.env.step(np.full(4, np.nan))
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(self.env.current_reward, 0.0)


if __name__ == '__main__':
    unittest.main()
