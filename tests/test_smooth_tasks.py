"""Smooth tasks retain base rewards and charge changes in applied controls."""
import unittest

import gymnasium as gym
import numpy as np

import heligym
import main


class SmoothTaskTests(unittest.TestCase):
    def test_rewards_follow_applied_actions_and_reset_trim(self):
        for smooth, base in ((heligym.HeliSmoothHover, heligym.HeliHover),
                             (heligym.HeliSmoothStabilize, heligym.HeliStabilize)):
            with self.subTest(task=smooth.__name__):
                env = smooth(render_mode=None, wind_enabled=False,
                             smoothness_weight=2.0)
                try:
                    for seed in (42, 43):
                        obs, _ = env.reset(seed=seed)
                        previous = env.heli_dyn.action.copy()
                        np.testing.assert_allclose(obs[-4:], previous)
                        # First step has no change; subsequent steps change all
                        # channels, repeat controls, then exercise clipping.
                        changed = np.clip(previous + [0.1, -0.1, 0.1, -0.1], -1, 1)
                        for action in (previous.copy(), changed, changed, np.full(4, 2.0)):
                            obs, reward, terminated, truncated, info = env.step(action)
                            self.assertFalse(info.get('numerical_failure', False))
                            applied = np.clip(action, -1, 1)
                            base_reward, _ = base._calculate_reward(env)
                            expected_cost = 2.0 * np.sum((applied - previous) ** 2)
                            self.assertAlmostEqual(reward, base_reward - expected_cost, places=5)
                            np.testing.assert_allclose(obs[-4:], applied)
                            previous = applied.copy()
                            if terminated or truncated:
                                break
                finally:
                    env.close()

    def test_tasks_are_available_through_cli_factory_and_gym(self):
        for name, cls in (('smooth-hover', heligym.HeliSmoothHover),
                          ('smooth-stabilize', heligym.HeliSmoothStabilize)):
            with self.subTest(task=name):
                self.assertEqual(main.parse_args(['train', '--task', name]).task, name)
                env = main.make_env(name, 'aw109', 1.0, no_wind=True)
                try:
                    self.assertIsInstance(env.unwrapped, cls)
                finally:
                    env.close()
                self.assertEqual(gym.spec(f'{cls.__name__}-v0').entry_point,
                                 f'heligym.envs:{cls.__name__}')

    def test_invalid_weights_are_rejected(self):
        for weight in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                heligym.HeliSmoothHover(smoothness_weight=weight)


if __name__ == '__main__':
    unittest.main()
