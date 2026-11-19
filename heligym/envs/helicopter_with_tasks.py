import numpy as np
from .helicopter import Heli


class ApproximateTargetValues(Heli):
    """Track any subset of Heli.task_target_keys in their physical units."""

    def __init__(self, *args, target=None, mode="progression", diff_impact="linear", **kwargs):
        if mode not in ("difference", "progression", "attitude"):
            raise ValueError("Unknown reward mode: " + str(mode))
        if diff_impact not in ("linear", "quadratic"):
            raise ValueError("Unknown difference impact: " + str(diff_impact))
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.diff_impact = diff_impact
        if target is not None:
            self.set_target(target)
        n = self.normalizers
        self._scales = np.array([n["x"], n["x"], n["x"], n["v"], 1.,
                                 n["v"], n["v"], 1. / n["t"]])

    def _reset_task(self):
        self._previous_values = self._target_values(self.heli_dyn.state)

    def step(self, actions):
        self._previous_values = self._target_values(self.heli_dyn.state)
        return super().step(actions)

    def _error(self, values):
        return self._target_differences(values) / self._scales

    def _cost(self, error):
        return np.abs(error) if self.diff_impact == "linear" else error ** 2

    def _calculate_reward(self):
        error = self._error(self._target_values(self.heli_dyn.state))
        if self.mode == "attitude":
            rate = self._target_values(self.heli_dyn.state_dots) * self.normalizers["t"] / self._scales
            gradient = np.sign(error) if self.diff_impact == "linear" else 2 * error
            reward = -np.dot(gradient, rate)
        else:
            reward = -self._cost(error).sum()
            if self.mode == "progression":
                reward += self._cost(self._error(self._previous_values)).sum()
        active = any(self.task_target.get(key) is not None for key in self.task_target_keys)
        return float(reward), bool(active and np.all(np.abs(error) < 1.))


_INITIAL = object()


class HeliHover(ApproximateTargetValues):
    """Hold the reset position, optionally overriding its sea altitude."""

    def __init__(self, heli_name="aw109", render_mode="human", wind_enabled=True,
                 target_altitude=_INITIAL, **kwargs):
        super().__init__(heli_name=heli_name, render_mode=render_mode, wind_enabled=wind_enabled, **kwargs)
        self._hover_altitude = target_altitude
        self._use_reset_position = kwargs.get("target") is None

    def set_target(self, target=()):
        super().set_target(target)
        self._use_reset_position = False

    def _reset_task(self):
        super()._reset_task()
        if self._use_reset_position:
            north, east, altitude = self._previous_values[:3]
            super().set_target(dict(north_loc=north, east_loc=east,
                                    sea_alt=altitude if self._hover_altitude is _INITIAL else self._hover_altitude))


class HeliGoto(ApproximateTargetValues):
    """Reach a random waypoint, or keep pursuing new waypoints in continual mode."""

    def __init__(self, *args, target_horizontal_range=100.0,
                 target_altitude_range=50.0, target_radius=10.0,
                 target_ground_clearance=50.0, progress_weight=0.1,
                 arrival_reward=100.0, continual=False, **kwargs):
        values = (target_horizontal_range, target_altitude_range,
                  target_radius, target_ground_clearance)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError('Goto target ranges, radius, and clearance must be positive and finite')
        if target_horizontal_range <= target_radius:
            raise ValueError('Goto horizontal range must exceed target radius')
        if not np.isfinite(progress_weight) or progress_weight < 0:
            raise ValueError('Goto progress weight must be finite and nonnegative')
        if not np.isfinite(arrival_reward) or arrival_reward <= 0:
            raise ValueError('Goto arrival reward must be positive and finite')
        self.progress_weight = float(progress_weight)
        self.arrival_reward = float(arrival_reward)
        self.target_horizontal_range = float(target_horizontal_range)
        self.target_altitude_range = float(target_altitude_range)
        self.target_radius = float(target_radius)
        self.target_ground_clearance = float(target_ground_clearance)
        self.continual = continual
        self.targets_reached = 0
        self._waypoints = kwargs.get("target") is None

        super().__init__(*args, **kwargs)


    def _draw_target(self):
        position = self.heli_dyn.state['xyz']
        current = np.array([position[0], position[1], -position[2]])
        bounds = np.array([self.heli_dyn.north_south_extent,
                           self.heli_dyn.east_west_extent]) / 2
        low = np.maximum(current[:2] - self.target_horizontal_range, -bounds + 1.)
        high = np.minimum(current[:2] + self.target_horizontal_range, bounds - 1.)
        for _ in range(100):
            north, east = self.np_random.uniform(low, high)
            ground = self.heli_dyn.ground_height_at(north, east)
            altitude = max(current[2] + self.np_random.uniform(
                -self.target_altitude_range, self.target_altitude_range),
                ground + self.target_ground_clearance)
            if np.linalg.norm(np.array([north, east, altitude]) - current) > self.target_radius:
                self.set_target(dict(north_loc=north, east_loc=east, sea_alt=altitude))
                return
        raise RuntimeError('Could not sample a goto target outside the arrival radius')

    def _reset_task(self):
        super()._reset_task()
        self.targets_reached = 0
        if not self._waypoints:
            return
        self._draw_target()
        self._previous_position = self.heli_dyn.state['xyz'].copy()

    def _distance_to_target(self, position):
        target = np.array([self.task_target['north_loc'], self.task_target['east_loc'],
                           -self.task_target['sea_alt']], dtype=np.float64)
        return float(np.linalg.norm(np.asarray(position, dtype=np.float64) - target))

    def _calculate_reward(self):
        if not self._waypoints:
            return super()._calculate_reward()
        # Both distances use the same active target. Target switching happens
        # only after this reward has been calculated.
        distance = self._distance_to_target(self.heli_dyn.state['xyz'])
        progress = self._distance_to_target(self._previous_position) - distance
        reached = distance <= self.target_radius
        reward = self.progress_weight * progress + self.arrival_reward * reached
        return float(reward), bool(reached)

    def _update_task(self):
        if not self._waypoints:
            return super()._update_task()
        if self._distance_to_target(self.heli_dyn.state['xyz']) <= self.target_radius:
            self.targets_reached += 1
            if self.continual:
                self._draw_target()
        self._previous_position = self.heli_dyn.state['xyz'].copy()

    def _is_successed(self):
        if self._waypoints:
            return not self.continual and self.targets_reached > 0
        return super()._is_successed()

    def _get_info(self):
        info = super()._get_info()
        info['targets_reached'] = self.targets_reached
        info['target_position'] = self._render_target_position()
        return info


class HeliStabilize(ApproximateTargetValues):
    """Bring the three body velocity components to zero."""

    def __init__(self, heli_name="aw109", render_mode="human", wind_enabled=True, **kwargs):
        kwargs.setdefault("target", dict(forward_speed=0., lateral_velocity=0., vertical_velocity=0.))
        super().__init__(heli_name=heli_name, render_mode=render_mode, wind_enabled=wind_enabled, **kwargs)

class _SmoothActions:
    """Penalize mean squared control changes and magnitudes relative to zero.

    Dynamics hold the current clipped action when rewards are calculated;
    previous_action still holds the preceding step (or reset trim controls).
    """

    def __init__(self, *args, smoothness_weight=0.25, **kwargs):
        if not np.isfinite(smoothness_weight) or smoothness_weight < 0:
            raise ValueError('smoothness_weight must be finite and nonnegative')
        self.smoothness_weight = float(smoothness_weight)
        super().__init__(*args, **kwargs)

    def _calculate_reward(self):
        reward, success_step = super()._calculate_reward()
        delta = self.heli_dyn.action - self.previous_action
        magnitude = self.heli_dyn.action
        penalty = self.smoothness_weight * np.mean(delta ** 2 + magnitude ** 2)
        return float(reward - penalty), success_step


class HeliSmoothStabilize(_SmoothActions, HeliStabilize):
    """Stabilize while penalizing successive control changes."""


class HeliSmoothHover(_SmoothActions, HeliHover):
    """Hold the hover target while penalizing successive control changes."""


class HeliForwardFlight(HeliGoto):
    """Track positive-up altitude and signed body-forward speed."""

    def __init__(self, heli_name="aw109", render_mode="human", wind_enabled=True,
                 target_altitude=2000, target_forward_speed=100., **kwargs):
        kwargs.setdefault("target", dict(sea_alt=target_altitude, forward_speed=target_forward_speed))
        super().__init__(heli_name=heli_name, render_mode=render_mode, wind_enabled=wind_enabled, **kwargs)
