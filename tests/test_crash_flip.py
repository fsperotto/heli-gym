"""Ground crashes and airborne inversion terminate with a fixed failure reward."""
import unittest
from unittest.mock import patch

import numpy as np

import heligym


class CrashFlipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = heligym.HeliGoto(render_mode=None, wind_enabled=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=42)
        self.env.heli_dyn.state['euler'][:] = 0
        self.env.heli_dyn.state_dots['xyz'][:] = 0

    def test_ground_tilt_is_symmetric_in_roll_and_pitch(self):
        for axis in (0, 1):
            for angle in (-70., 70.):
                with self.subTest(axis=axis, angle=angle):
                    self.env.heli_dyn.state['euler'][:] = 0
                    self.env.heli_dyn.state['euler'][axis] = np.deg2rad(angle)
                    with patch.object(self.env.heli_dyn, '_does_hit_ground', return_value=True):
                        self.assertTrue(self.env._is_crashed())
                        self.assertTrue(self.env._is_failed())
                    with patch.object(self.env.heli_dyn, '_does_hit_ground', return_value=False):
                        self.assertFalse(self.env._is_crashed())
                        self.assertFalse(self.env._is_failed())

    def test_gentle_ground_contact_is_not_a_crash_but_fast_descent_is(self):
        with patch.object(self.env.heli_dyn, '_does_hit_ground', return_value=True):
            self.assertFalse(self.env._is_crashed())
            self.env.heli_dyn.state_dots['xyz'][2] = self.env.heli_dyn.MR['V_TIP'] * .06
            self.assertTrue(self.env._is_crashed())

    def test_flip_depends_on_physical_attitude_not_heading_or_euler_sign(self):
        for roll, pitch, expected in ((0, 0, False), (89, 0, False), (0, -89, False),
                                      (100, 0, True), (-100, 0, True),
                                      (0, 100, True), (0, -100, True),
                                      (180, 0, True), (-180, 0, True),
                                      (180, 180, False)):
            for yaw in (0, 120):
                with self.subTest(roll=roll, pitch=pitch, yaw=yaw):
                    self.env.heli_dyn.state['euler'][:] = np.deg2rad([roll, pitch, yaw])
                    self.assertEqual(self.env._is_flipped(), expected)
                    self.assertEqual(self.env._is_failed(), expected)

    def test_crash_or_flip_overrides_task_reward_and_does_not_advance_target(self):
        for crash in (False, True):
            with self.subTest(crash=crash):
                env = self.env
                self.setUp()
                if crash:
                    env.heli_dyn.state['euler'][0] = np.deg2rad(-70)
                else:
                    env.heli_dyn.state['euler'][0] = np.pi
                # Simultaneous arrival must not award a bonus or draw a target.
                position = env.heli_dyn.state['xyz']
                env.set_target(dict(north_loc=float(position[0]), east_loc=float(position[1]),
                                    sea_alt=float(-position[2])))
                target = env.get_target()
                env.successed_time = env.success_duration + 1
                with patch.object(env.heli_dyn, '_does_hit_ground', return_value=crash), \
                        patch.object(env.heli_dyn, 'step', return_value=env.heli_dyn.observation.copy()), \
                        patch.object(env, '_calculate_reward', return_value=(999., True)) as task_reward:
                    _, reward, terminated, truncated, info = env.step(env.heli_dyn.action.copy())
                self.assertEqual(reward, -100.)
                self.assertEqual(env.current_reward, -100.)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertTrue(info['failed'])
                self.assertFalse(info['successed'])
                self.assertEqual(info['crash'], crash)
                self.assertEqual(info['flip'], not crash)
                self.assertEqual(env.targets_reached, 0)
                self.assertEqual(env.get_target(), target)
                task_reward.assert_not_called()

    def test_bounds_failure_and_time_limit_are_preserved(self):
        self.env.heli_dyn.state['xyz'][0] = self.env.heli_dyn.north_south_extent
        self.assertTrue(self.env._is_failed())
        self.env.heli_dyn.state['xyz'][0] = 0
        self.env.time_counter = self.env.max_time + 1
        self.assertTrue(self.env._is_time_up())
        self.assertFalse(self.env._is_failed())


if __name__ == '__main__':
    unittest.main()
