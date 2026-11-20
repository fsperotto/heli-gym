import math
import unittest
from unittest.mock import patch

import heligym
from heligym.envs.helicopter import DT


class RecentYawTests(unittest.TestCase):
    def test_one_turn_in_ten_seconds_and_decay(self):
        env = heligym.Heli(render_mode=None, wind_enabled=False)
        self.addCleanup(env.close)
        index = env.observation_names.index("recent_yaw_turns")
        for direction in (-1, 1):
            observation, _ = env.reset(seed=42)
            self.assertEqual(observation[index], 0.)
            with patch.object(env.heli_dyn, "step", return_value=env.heli_dyn.observation):
                env.heli_dyn.state["pqr"][2] = direction * math.tau / 10
                for _ in range(round(10 / DT)):
                    observation, _, terminated, truncated, _ = env.step(env.heli_dyn.action)
                self.assertFalse(terminated or truncated)
                self.assertAlmostEqual(observation[index], direction, places=6)
                self.assertTrue(env.observation_space.contains(observation))
                observation, *_ = env.step(env.heli_dyn.action)
                continued_spin = abs(observation[index])
                self.assertGreater(continued_spin, 1.)
                env.heli_dyn.state["pqr"][2] = 0.
                for _ in range(round(10 / DT)):
                    observation, *_ = env.step(env.heli_dyn.action)
                self.assertAlmostEqual(abs(observation[index]), continued_spin / math.e, places=6)
