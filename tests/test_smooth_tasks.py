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
                            expected_cost = 2.0 * np.mean(
                                (applied - previous) ** 2 + applied ** 2)
                            self.assertAlmostEqual(reward, base_reward - expected_cost, places=5)
                            np.testing.assert_allclose(obs[-4:], applied)
                            previous = applied.copy()
                            if terminated or truncated:
                                break
                finally:
                    env.close()

    def test_stabilize_only_penalizes_mean_squared_velocities(self):
        env = heligym.HeliStabilize(render_mode=None, wind_enabled=False)
        try:
            env.reset(seed=42)
            env.heli_dyn.state['uvw'][:] = np.array([1., 2., 3.]) * env.normalizers['v']
            env.heli_dyn.state['pqr'][:] = np.array([4., 5., 6.]) / env.normalizers['t']
            reward, success = env._calculate_reward()
            self.assertAlmostEqual(reward, -91. / 6., places=5)
            self.assertFalse(success)
            # Stationary flight succeeds even with tilt, at another position,
            # and at ground level; none of those add a reward penalty.
            env.heli_dyn.state['uvw'][:] = 0
            env.heli_dyn.state['pqr'][:] = 0
            env.heli_dyn.state['euler'][:] = [.3, -.2, 1.]
            env.heli_dyn.state['xyz'][:] = [100., 200., 0.]
            self.assertEqual(env._calculate_reward(), (0., True))
        finally:
            env.close()

    def test_default_and_zero_smoothness_weights(self):
        for weight in (0., .25):
            env = heligym.HeliSmoothHover(render_mode=None, wind_enabled=False,
                                        **({'smoothness_weight': weight} if weight == 0 else {}))
            try:
                env.reset(seed=42)
                self.assertEqual(env.smoothness_weight, weight)
                self.assertFalse(hasattr(env, 'trim_action'))
                env.heli_dyn.action[:] = .1
                env.previous_action[:] = env.heli_dyn.action
                base, success = heligym.HeliHover._calculate_reward(env)
                reward, smooth_success = env._calculate_reward()
                # Holding a nonzero control still incurs a magnitude cost.
                self.assertAlmostEqual(reward, base - weight * .01, places=6)
                self.assertEqual(success, smooth_success)
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
