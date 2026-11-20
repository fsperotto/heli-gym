"""Task observations and seeded, continuing waypoint navigation."""
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np

import heligym
import main


class TaskObservationTests(unittest.TestCase):
    def test_target_differences_use_current_minus_target(self):
        env = heligym.Heli(render_mode=None, wind_enabled=False)
        try:
            observation, _ = env.reset(seed=42)
            self.assertEqual(observation.shape, (30,))
            self.assertTrue(env.observation_space.contains(observation))
            np.testing.assert_array_equal(observation[:17], env.heli_dyn.observation.astype(np.float32))
            np.testing.assert_array_equal(observation[17:25], np.zeros(8))
            raw = env.heli_dyn.observation.copy()
            env.heli_dyn.state["xyz"][:] = [0., 5., -56.]
            env.heli_dyn.state["uvw"][:] = [3., 4., 0.]
            env.heli_dyn.state["euler"][2] = np.deg2rad(-179.)
            env.set_target((None, None, None, None, None))
            np.testing.assert_array_equal(env._agent_observation(raw)[17:25], np.zeros(8))
            env.set_target((3., 3., 50., 20., np.deg2rad(179.)))
            np.testing.assert_allclose(env._agent_observation(raw)[17:25],
                                       [-3., 2., 6., -17., np.deg2rad(2.), 0., 0., 0.], atol=1e-6)
            env.set_target(dict(north_loc=3, sea_alt=50))
            np.testing.assert_array_equal(env._agent_observation(raw)[17:25], [-3, 0, 6, 0, 0, 0, 0, 0])
        finally:
            env.close()

    def test_hover_exposes_differences_from_its_fixed_position_target(self):
        env = heligym.HeliHover(render_mode=None, wind_enabled=False)
        try:
            observation, _ = env.reset(seed=42)
            np.testing.assert_allclose(observation[17:20], np.zeros(3), atol=1e-4)
        finally:
            env.close()


class GotoTests(unittest.TestCase):
    def setUp(self):
        self.env = heligym.HeliGoto(render_mode=None, wind_enabled=False, continual=True)

    def tearDown(self):
        self.env.close()

    def test_default_arrival_terminates_with_reached_waypoint(self):
        env = heligym.HeliGoto(render_mode=None, wind_enabled=False)
        self.addCleanup(env.close)
        env.reset(seed=42)
        north, east, down = env.heli_dyn.state["xyz"]
        env.set_target(dict(north_loc=north, east_loc=east, sea_alt=-down))
        target = env._render_target_position()
        with patch.object(env.heli_dyn, 'step', return_value=env.heli_dyn.observation.copy()):
            observation, reward, terminated, truncated, info = env.step(env.heli_dyn.action.copy())
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["successed"])
        self.assertEqual(info["targets_reached"], 1)
        self.assertEqual(reward, env.arrival_reward)
        np.testing.assert_array_equal(info["target_position"], target)
        np.testing.assert_array_equal(observation[17:20], np.zeros(3))

    def test_seeded_targets_have_clearance_and_no_speed_constraint(self):
        env = self.env
        first, _ = env.reset(seed=42)
        initial = env.heli_dyn.observation[13:16].copy()
        target = env.get_target()
        target = np.array([target["north_loc"], target["east_loc"], target["sea_alt"]])
        np.testing.assert_allclose(first[17:20], initial - target)
        np.testing.assert_array_equal(env.reset(seed=42)[0], first)
        self.assertGreater(np.linalg.norm(target - initial), env.target_radius)
        self.assertTrue(np.all(np.abs(target[:2] - initial[:2]) <= 100.))
        self.assertGreaterEqual(target[2] + 1e-3,
                                env.heli_dyn.ground_height_at(*target[:2]) + 50.)
        self.assertIsNone(env.get_target().get('forward_speed'))
        np.testing.assert_array_equal(first[20:22], [0., 0.])
        other, _ = env.reset(seed=43)
        self.assertFalse(np.array_equal(other[17:20], first[17:20]))

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
        self.assertFalse(np.array_equal(info['target_position'], old_target))
        np.testing.assert_allclose(observation[17:20],
                                   env.heli_dyn.observation[13:16] - info['target_position'],
                                   atol=2e-4)
        self.assertTrue(env.observation_space.contains(observation))
        env.successed_time = env.success_duration + 1
        self.assertFalse(env._is_successed())
        _, info = env.reset(seed=42)
        self.assertEqual(info['targets_reached'], 0)

    def test_target_speed_stays_zero_while_moving_and_after_target_changes(self):
        env = self.env
        env.reset(seed=42)
        raw = env.heli_dyn.observation.copy()
        env.heli_dyn.state["uvw"][:] = [3., 4., 0.]
        self.assertEqual(env._agent_observation(raw)[20], 0.)
        env._draw_target()
        self.assertIsNone(env.get_target().get('forward_speed'))
        self.assertEqual(env._agent_observation(raw)[20], 0.)
        # Even a position-only target keeps goto's unused speed field zero.
        env.set_target(dict(north_loc=0., east_loc=0., sea_alt=2000.))
        self.assertEqual(env._agent_observation(raw)[20], 0.)
        env.heli_dyn.observation = raw
        env.renderer = Mock()
        env.render()
        self.assertEqual(env.renderer.render.call_args.kwargs['target'][3], 0.)

    def test_progress_uses_change_in_euclidean_distance_only(self):
        env = self.env
        env.reset(seed=42)
        start = env.heli_dyn.state['xyz'].copy()
        target = start + [100., 0., 0.]
        env.set_target(dict(north_loc=target[0], east_loc=target[1], sea_alt=-target[2]))
        for offset, expected in (([3., 0., 0.], .3),
                                 ([100., 100., 0.], 0.),
                                 ([-2., 0., 0.], -.2)):
            with self.subTest(offset=offset):
                env.heli_dyn.state['xyz'][:] = start + offset
                reward, reached = env._calculate_reward()
                self.assertAlmostEqual(reward, expected)
                self.assertFalse(reached)
        # No independent penalty on linear/angular velocities or attitude.
        env.heli_dyn.state['xyz'][:] = start
        env.heli_dyn.state['uvw'][:] = [20, 30, 40]
        env.heli_dyn.state['pqr'][:] = [1, 2, 3]
        env.heli_dyn.state['euler'][:] = [.2, -.3, .4]
        self.assertEqual(env._calculate_reward(), (0., False))

    def test_arrival_bonus_is_dominant_and_new_target_has_no_spurious_progress(self):
        env = self.env
        env.reset(seed=42)
        start = env.heli_dyn.state['xyz'].copy()
        target = start + [100., 0., 0.]
        env.set_target(dict(north_loc=target[0], east_loc=target[1], sea_alt=-target[2]))
        env.heli_dyn.state['xyz'][:] = target - [5., 0., 0.]
        self.assertEqual(env._calculate_reward(), (109.5, True))
        env._update_task()
        self.assertEqual(env.targets_reached, 1)
        self.assertEqual(env._calculate_reward(), (0., False))
        env.heli_dyn.state['xyz'][0] += 1.
        env._update_task()
        self.assertEqual(env._calculate_reward(), (0., False))

    def test_reward_weights_are_configurable_and_validated(self):
        for weight in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                heligym.HeliGoto(progress_weight=weight)
        for bonus in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                heligym.HeliGoto(arrival_reward=bonus)
        env = self.env
        env.reset(seed=42)
        position = env.heli_dyn.state['xyz']
        env.set_target(dict(north_loc=float(position[0]), east_loc=float(position[1]),
                            sea_alt=float(-position[2])))
        env.progress_weight = 0.
        env.arrival_reward = 50.
        self.assertEqual(env._calculate_reward(), (50., True))

    def test_cli_factory_and_gym_registration(self):
        self.assertEqual(main.parse_args(['train', '--task', 'goto']).task, 'goto')
        args = main.parse_args(['train', '--task', 'goto', '--continual'])
        wrapped = main.make_env('goto', 'aw109', 1., no_wind=True, continual=args.continual)
        try:
            self.assertIsInstance(wrapped.unwrapped, heligym.HeliGoto)
            self.assertTrue(wrapped.unwrapped.continual)
            observation, _ = wrapped.reset(seed=42)
            self.assertEqual(observation.shape, (30,))
        finally:
            wrapped.close()
        self.assertEqual(gym.spec('HeliGoto-v0').entry_point, 'heligym.envs:HeliGoto')


if __name__ == '__main__':
    unittest.main()
