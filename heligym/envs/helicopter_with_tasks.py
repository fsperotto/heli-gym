
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


class HeliStabilize(Heli):
    """Reward level, motionless flight and strongly discourage crashes."""

    CRASH_REWARD = -10_000.0
    MOTION_WEIGHT = 1.0
    INCLINATION_WEIGHT = 100.0

    def __init__(self, heli_name: str = "aw109", render_mode="human",
                 wind_enabled=True, **heli_kwargs):
        super().__init__(heli_name=heli_name, render_mode=render_mode,
                         wind_enabled=wind_enabled, **heli_kwargs)
        self.set_max_time(self.default_max_time)

    def _calculate_reward(self):
        altitude = -self.heli_dyn.state["xyz"][2]
        if self.heli_dyn._does_hit_ground(altitude):
            return self.CRASH_REWARD, False

        linear_velocity = self.heli_dyn.state["uvw"] / self.normalizers["v"]
        angular_velocity = self.heli_dyn.state["pqr"] * self.normalizers["t"]
        roll_pitch = self.heli_dyn.state["euler"][:2]

        motion_penalty = self.MOTION_WEIGHT * (
            np.dot(linear_velocity, linear_velocity)
            + np.dot(angular_velocity, angular_velocity)
        )
        inclination_penalty = self.INCLINATION_WEIGHT * np.dot(roll_pitch, roll_pitch)
        reward = -(motion_penalty + inclination_penalty)

        speed = np.linalg.norm(self.heli_dyn.state["uvw"])
        angular_speed = np.linalg.norm(self.heli_dyn.state["pqr"])
        inclination = np.linalg.norm(roll_pitch)
        success_step = speed < 0.5 and angular_speed < 0.01 and inclination < np.deg2rad(1.0)
        return float(reward), bool(success_step)

class _SmoothActions:
    """Add a quadratic cost on changes in the four normalized controls.

    Dynamics hold the current clipped action when rewards are calculated;
    previous_action still holds the preceding step (or reset trim controls).
    """

    def __init__(self, *args, smoothness_weight=1.0, **kwargs):
        if not np.isfinite(smoothness_weight) or smoothness_weight < 0:
            raise ValueError('smoothness_weight must be finite and nonnegative')
        self.smoothness_weight = float(smoothness_weight)
        super().__init__(*args, **kwargs)

    def _calculate_reward(self):
        reward, success_step = super()._calculate_reward()
        delta = self.heli_dyn.action - self.previous_action
        penalty = self.smoothness_weight * np.dot(delta, delta)
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
