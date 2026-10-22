
import numpy as np
from .helicopter import Heli

class HeliHover(Heli):
    def __init__(self, heli_name:str = "aw109", render_mode="human",
                 wind_enabled=True, target_altitude=2000, **heli_kwargs):
        Heli.__init__(self, heli_name=heli_name, render_mode=render_mode,
                      wind_enabled=wind_enabled, **heli_kwargs)
        max_time = Heli.default_max_time
        task_target = {
            "sea_alt": target_altitude,
            "north_loc": 0,
            "east_loc": 0
        }
        trim_cond = {
            "yaw": 0.0,
            "yaw_rate": 0.0,
            "ned_vel": [0.0, 0.0, 0.0],
            "gr_alt": 100.0,
            "xy": [0.0, 0.0],
            "psi_mr": 0.0,
            "psi_tr": 0.0
        }
        self.set_max_time(max_time)
        self.set_target(task_target)
        self.set_trim_cond(trim_cond)

    def _calculate_reward(self):
        xyz_norm = self.heli_dyn.state["xyz"] / self.normalizers["x"]
        xyzdot_norm = self.heli_dyn.state_dots["xyz"] / self.normalizers["v"]
        pqr_norm = self.heli_dyn.state["pqr"] * self.normalizers["t"] 
        pqrdot_norm = self.heli_dyn.state_dots["pqr"] * self.normalizers["t"]**2

        xyz_target_norm = np.array([self.task_target["north_loc"], self.task_target["east_loc"], -self.task_target["sea_alt"]], dtype=np.float32) / self.normalizers["x"]

        pqr_final_reward = - (pqr_norm * pqr_norm).sum()
        pqr_terminal_reward = - (np.sign(pqr_norm) * pqrdot_norm).sum()
        pqr_reward = max(
            pqr_final_reward,
            pqr_terminal_reward)

        xyz_final_reward = - ((xyz_norm - xyz_target_norm) * (xyz_norm - xyz_target_norm)).sum()
        xyz_terminal_reward = - (np.sign(xyz_norm - xyz_target_norm) * xyzdot_norm).sum()

        xyz_reward = max(
            xyz_final_reward,
            xyz_terminal_reward)

        reward = (pqr_reward + xyz_reward) / 2.0

        success_step = pqr_final_reward > -1.0 and xyz_final_reward > -1.0  

        return reward, success_step


class HeliGoto(HeliHover):
    """Pursue a sequence of random, observed position targets."""

    def __init__(self, *args, target_horizontal_range=100.0,
                 target_altitude_range=50.0, target_radius=10.0,
                 target_ground_clearance=50.0, progress_weight=0.1,
                 arrival_reward=100.0, **kwargs):
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
        self.targets_reached = 0
        super().__init__(*args, **kwargs)
        # No speed objective; expose a fixed zero target speed.
        self.set_target(dict(vel=0.0))

    def _task_observation(self, dynamics_observation):
        observation = super()._task_observation(dynamics_observation)
        observation[3] = 0.0  # Speed is not an objective for this task.
        return observation

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
                self.set_target(dict(north_loc=north, east_loc=east, sea_alt=altitude, vel=0.0))
                return
        raise RuntimeError('Could not sample a goto target outside the arrival radius')

    def _reset_task(self):
        self.targets_reached = 0
        self._draw_target()
        self._previous_position = self.heli_dyn.state['xyz'].copy()

    def _distance_to_target(self, position):
        target = np.array([self.task_target['north_loc'], self.task_target['east_loc'],
                           -self.task_target['sea_alt']], dtype=np.float64)
        return float(np.linalg.norm(np.asarray(position, dtype=np.float64) - target))

    def _calculate_reward(self):
        # Both distances use the same active target. Target switching happens
        # only after this reward has been calculated.
        distance = self._distance_to_target(self.heli_dyn.state['xyz'])
        progress = self._distance_to_target(self._previous_position) - distance
        reached = distance <= self.target_radius
        reward = self.progress_weight * progress + self.arrival_reward * reached
        return float(reward), bool(reached)

    def _update_task(self):
        if self._distance_to_target(self.heli_dyn.state['xyz']) <= self.target_radius:
            self.targets_reached += 1
            self._draw_target()
        self._previous_position = self.heli_dyn.state['xyz'].copy()

    def _is_successed(self):
        # Reaching a waypoint starts the next leg instead of ending the episode.
        return False

    def _get_info(self):
        info = super()._get_info()
        info['targets_reached'] = self.targets_reached
        info['target_position'] = self._render_target_position()
        return info


class HeliStabilize(Heli):
    """Reward stopping linear and angular motion at any position or attitude."""

    def __init__(self, heli_name: str = "aw109", render_mode="human",
                 wind_enabled=True, **heli_kwargs):
        super().__init__(heli_name=heli_name, render_mode=render_mode,
                         wind_enabled=wind_enabled, **heli_kwargs)
        self.set_max_time(self.default_max_time)

    def _calculate_reward(self):
        linear_velocity = self.heli_dyn.state["uvw"] / self.normalizers["v"]
        angular_velocity = self.heli_dyn.state["pqr"] * self.normalizers["t"]
        velocities = np.concatenate((linear_velocity, angular_velocity))
        reward = -np.mean(velocities ** 2)

        speed = np.linalg.norm(self.heli_dyn.state["uvw"])
        angular_speed = np.linalg.norm(self.heli_dyn.state["pqr"])
        success_step = speed < 0.5 and angular_speed < 0.01
        return float(reward), bool(success_step)

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


class HeliForwardFlight(Heli):

    def __init__(self, heli_name:str = "aw109", render_mode="human",
                 wind_enabled=True, target_altitude=2000, **heli_kwargs):
        Heli.__init__(self, heli_name=heli_name, render_mode=render_mode,
                      wind_enabled=wind_enabled, **heli_kwargs)
        max_time = Heli.default_max_time
        task_target = {
            "sea_alt": target_altitude,
            "heading": 0,
            "vel": 100
        }
        trim_cond = {
            "yaw": 0.0,
            "yaw_rate": 0.0,
            "ned_vel": [0.0, 0.0, 0.0],
            "gr_alt": 100.0,
            "xy": [0.0, 0.0],
            "psi_mr": 0.0,
            "psi_tr": 0.0
        }

        self.set_max_time(max_time)
        self.set_target(task_target)
        self.set_trim_cond(trim_cond)

    def _calculate_reward(self):
        vel = np.sqrt((self.heli_dyn.state["uvw"] * self.heli_dyn.state["uvw"]).sum())
        vel_norm =  vel / self.normalizers["v"]
        veldot_norm = (self.heli_dyn.state["uvw"] * self.heli_dyn.state_dots["uvw"]).sum() / vel / self.normalizers["a"]
        dwn_norm = self.heli_dyn.state["xyz"][2] / self.normalizers["x"]
        dwndot_norm = self.heli_dyn.state_dots["xyz"][2] / self.normalizers["v"]
        pqr_norm = self.heli_dyn.state["pqr"] * self.normalizers["t"] 
        pqrdot_norm = self.heli_dyn.state_dots["pqr"] * self.normalizers["t"]**2

        vel_target_norm = np.array(self.task_target["vel"], dtype=np.float32) / self.normalizers["v"]
        dwn_target_norm = np.array(-self.task_target["sea_alt"], dtype=np.float32) / self.normalizers["x"]

        pqr_final_reward = - (pqr_norm * pqr_norm).sum()
        pqr_terminal_reward = - (np.sign(pqr_norm) * pqrdot_norm).sum()
        pqr_reward = max(
            pqr_final_reward,
            pqr_terminal_reward)

        vel_final_reward = - ((vel_norm - vel_target_norm) * (vel_norm - vel_target_norm)).sum()
        vel_terminal_reward = - (np.sign(vel_norm - vel_target_norm) * veldot_norm).sum()

        vel_reward = max(
            vel_final_reward,
            vel_terminal_reward)

        dwn_final_reward = - ((dwn_norm - dwn_target_norm) * (dwn_norm - dwn_target_norm)).sum()
        dwn_terminal_reward = - (np.sign(dwn_norm - dwn_target_norm) * dwndot_norm).sum()

        dwn_reward = max(
            dwn_final_reward,
            dwn_terminal_reward)


        reward = ( pqr_reward + vel_reward + dwn_reward ) / 3.0

        success_step = pqr_final_reward > -1.0 and vel_final_reward > -1.0 and dwn_final_reward > -1.0 

        return reward, success_step
