"""Stationary trim is an in-place intervention, with rollback on solver failure."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import heligym


class RetrimTests(unittest.TestCase):
    def test_stop_at_current_position_in_still_air_and_wind(self):
        for wind in (False, True):
            with self.subTest(wind=wind):
                env = heligym.HeliGoto(render_mode=None, wind_enabled=wind)
                try:
                    env.reset(seed=42)
                    dyn = env.heli_dyn
                    dyn.state['xyz'][:] += [25., -30., -40.]
                    dyn.state['euler'][2] = .7
                    dyn.state['psi_mr'][:] = .3
                    dyn.state['psi_tr'][:] = .4
                    dyn.state['uvw'][:] = [3., -2., 1.]
                    dyn.state['pqr'][:] = [.1, -.2, .3]
                    dyn.state_dots['uvw'][:] = [1., 2., 3.]
                    dyn.state_dots['pqr'][:] = [.3, .2, .1]
                    position = dyn.state['xyz'].copy()
                    heading = dyn.state['euler'][2]
                    target = env.get_target()
                    applied_wind = dyn.WIND_NED.copy()
                    env.time_counter = 3.
                    env.step_count = 150
                    env.paused = True
                    env.pilot_mode = 'agent'
                    observation = env.trim_at_current_position()
                    np.testing.assert_array_equal(dyn.state['xyz'], position)
                    self.assertEqual(dyn.state['euler'][2], heading)
                    self.assertAlmostEqual(dyn.state['psi_mr'][0], .3)
                    self.assertAlmostEqual(dyn.state['psi_tr'][0], .4)
                    for key in ('uvw', 'pqr'):
                        np.testing.assert_array_equal(dyn.state[key], np.zeros(3))
                        np.testing.assert_array_equal(dyn.state_dots[key], np.zeros(3))
                    # Check actual forces/torques too, beyond the zeroed cache.
                    derivatives = dyn.dynamics(dyn.state)
                    np.testing.assert_allclose(derivatives['uvw'], 0., atol=1e-3)
                    np.testing.assert_allclose(derivatives['pqr'], 0., atol=1e-4)
                    np.testing.assert_array_equal(observation[4:7], np.zeros(3))
                    np.testing.assert_array_equal(observation[10:13], np.zeros(3))
                    np.testing.assert_allclose(observation[-4:], dyn.action)
                    np.testing.assert_allclose(env.previous_action, dyn.action)
                    np.testing.assert_array_equal(dyn.WIND_NED, applied_wind)
                    self.assertEqual(env.get_target(), target)
                    self.assertEqual(env.time_counter, 3.)
                    self.assertEqual(env.step_count, 150)
                    self.assertTrue(env.paused)
                    self.assertEqual(env.pilot_mode, 'agent')
                finally:
                    env.close()

    def test_failed_trim_restores_state_controls_observations_and_action_history(self):
        env = heligym.Heli(render_mode=None, wind_enabled=False)
        try:
            env.reset(seed=42)
            dyn = env.heli_dyn
            fields = ('state', 'state_dots', 'action', 'observation', 'previous_observation')
            before = {key: copy.deepcopy(getattr(dyn, key)) for key in fields}
            previous_action = env.previous_action.copy()
            def failure(*args, **kwargs):
                dyn.state.val[:] = 999
                dyn.state_dots.val[:] = -999
                dyn.action[:] = 999
                dyn.observation[:] = 999
                raise np.linalg.LinAlgError('singular trim')
            with patch.object(dyn, 'trim', side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, 'singular trim'):
                    env.trim_at_current_position()
            for key in fields:
                actual, expected = getattr(dyn, key), before[key]
                if key in ('state', 'state_dots'):
                    actual, expected = actual.val, expected.val
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(env.previous_action, previous_action)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
