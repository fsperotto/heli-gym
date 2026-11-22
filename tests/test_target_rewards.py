import unittest

import gymnasium as gym
import numpy as np

import heligym


class TargetRewardTests(unittest.TestCase):
    def test_all_modes_and_impacts(self):
        wrapped = gym.make("ApproximateTargetValues-v0", render_mode=None, wind_enabled=False)
        self.addCleanup(wrapped.close)
        env = wrapped.unwrapped
        wrapped.reset(seed=5)
        v = env.normalizers["v"]
        env.set_target({"forward_speed": 0., "vertical_velocity": None})
        env.heli_dyn.state["uvw"][:] = [3*v, 20., 40.]
        env._reset_task()
        env.heli_dyn.state["uvw"][0] = 2*v
        env.heli_dyn.state_dots["uvw"][0] = -env.normalizers["a"]
        for impact, expected in (("linear", (-2, 1, 1)), ("quadratic", (-4, 5, 4))):
            env.diff_impact = impact
            for mode, reward in zip(("difference", "progression", "attitude"), expected):
                with self.subTest(mode=mode, impact=impact):
                    env.mode = mode
                    self.assertAlmostEqual(env._calculate_reward()[0], reward, places=5)
        env.set_target({})
        self.assertEqual(env._calculate_reward(), (0., False))
        env.set_target(dict(lateral_velocity=1., vertical_velocity=2., yaw_rotation_velocity=.3))
        env.heli_dyn.state["pqr"][2] = -.2
        np.testing.assert_allclose(env._task_observation()[5:], [38., 19., -.5])

    def test_success_checks_physical_tolerances_per_target(self):
        env = heligym.ApproximateTargetValues(render_mode=None, wind_enabled=False,
                                              yaw_turn_seconds=30)
        self.addCleanup(env.close)
        env.reset(seed=42)
        env.set_target(dict(north_loc=0., east_loc=0., sea_alt=100.,
                            forward_speed=0., heading=0., yaw_rotation_velocity=0.))
        state = env.heli_dyn.state
        state["xyz"][:] = [12., 12., -109.]
        state["uvw"][:] = [4., 0., 0.]
        state["euler"][2] = np.deg2rad(4.)
        state["pqr"][2] = np.deg2rad(11.)
        self.assertTrue(env._calculate_reward()[1])
        for key, index, value in (("xyz", 0, 17.), ("xyz", 2, -111.),
                                  ("uvw", 0, 6.), ("euler", 2, np.deg2rad(6.)),
                                  ("pqr", 2, np.deg2rad(13.))):
            original = state[key][index]
            state[key][index] = value
            self.assertFalse(env._calculate_reward()[1])
            state[key][index] = original

    def test_fixed_forward_flight_tracks_signed_u_and_altitude_only(self):
        wrapped = gym.make("HeliForwardFlight-v0", render_mode=None, wind_enabled=False,
                           target_altitude=1500., target_forward_speed=30.)
        self.addCleanup(wrapped.close)
        env = wrapped.unwrapped
        wrapped.reset(seed=42)
        env.heli_dyn.state["xyz"][2] = -1500.
        env.heli_dyn.state["uvw"][:] = [-30., 50., 10.]
        np.testing.assert_array_equal(env._task_observation(), [0, 0, 0, -60, 0, 0, 0, 0])
        observation, reward, _, _, info = wrapped.step(env.heli_dyn.action.copy())
        self.assertTrue(wrapped.observation_space.contains(observation))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(info.get("numerical_failure", False))
        self.assertEqual(env.get_target(), dict(sea_alt=1500., forward_speed=30.))
