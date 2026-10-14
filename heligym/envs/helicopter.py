"""
Helicopter control.
"""
import yaml
import sys, math
import numpy as np
import os
import copy

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding, EzPickle

from .dynamics import HelicopterDynamics
from .dynamics import WindDynamics

FPS         = 50.0
DT          = 1/FPS
FTS2KNOT    = 0.5924838 # ft/s to knots conversion
EPS         = 1e-10 # small value for divison by zero
R2D         = 180/math.pi # Rad to deg
D2R         = 1/R2D
TAU         = 2*math.pi
FLOAT_TYPE  = np.float32

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
with open(os.path.join(CONFIG_DIR, "weather", "default_weather.yaml")) as weather_file:
    DEFAULT_WEATHER = yaml.safe_load(weather_file)


def _config_path(category, name):
    """Resolve either a bundled configuration name or an explicit YAML path."""
    path = os.fspath(name)
    if os.path.dirname(path):
        return path
    filename = path if path.endswith((".yaml", ".yml")) else path + ".yaml"
    return os.path.join(CONFIG_DIR, category, filename)

class Heli(gym.Env, EzPickle):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    # Default maximum time for an episode
    default_max_time = 40.0

    # Default trim condition (steady-state / equilibrium), ground trim.
    default_trim_cond = {
        "yaw": 0.0,
        "yaw_rate": 0.0,
        "ned_vel": [0.0, 0.0, 0.0], # north, east, down velocity
        "gr_alt": 100.0,   # altitude
        "xy": [0.0, 0.0],  # position
        "psi_mr": 0.0,     # main rotor
        "psi_tr": 0.0      # tail rotor
    }

    # default task target
    default_task_target = {}

    def __init__(self,
                 heli_name:str = "aw109",
                 terrain_name: str = "default_terrain",
                 render_mode="human",
                 wind_enabled=True,
                 wind_direction=DEFAULT_WEATHER["wind_direction"],
                 wind_speed=DEFAULT_WEATHER["wind_speed"],
                 turbulence_level=DEFAULT_WEATHER["turbulence_level"],
                 gas_constant=DEFAULT_WEATHER["gas_constant"],
                 sea_level_temperature=DEFAULT_WEATHER["sea_level_temperature"],
                 temperature_lapse_rate=DEFAULT_WEATHER["temperature_lapse_rate"],
                 troposphere_height=DEFAULT_WEATHER["troposphere_height"],
                 sea_level_air_density=DEFAULT_WEATHER["sea_level_air_density"],
                 gravity=DEFAULT_WEATHER["gravity"]):

        EzPickle.__init__(self,
                          heli_name=heli_name,
                          terrain_name=terrain_name,
                          render_mode=render_mode,
                          wind_enabled=wind_enabled,
                          wind_direction=wind_direction,
                          wind_speed=wind_speed,
                          turbulence_level=turbulence_level,
                          gas_constant=gas_constant,
                          sea_level_temperature=sea_level_temperature,
                          temperature_lapse_rate=temperature_lapse_rate,
                          troposphere_height=troposphere_height,
                          sea_level_air_density=sea_level_air_density,
                          gravity=gravity)

        self.wind_enabled = wind_enabled

        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or human")
        self.render_mode = render_mode

        # Dynamics need terrain images even when no native renderer is loaded.
        os.environ["HELIGYM_RESOURCE_DIR"] = os.path.join(os.path.dirname(__file__), "renderer", "resources")

        with open(_config_path("helis", heli_name)) as config_file:
            helicopter = yaml.safe_load(config_file)["HELI"]
        with open(_config_path("terrains", terrain_name)) as config_file:
            terrain = yaml.safe_load(config_file)

        effective_wind_speed = wind_speed if self.wind_enabled else 0.0
        self.heli_dyn = HelicopterDynamics(
            helicopter=helicopter,
            dt=DT,
            resource_dir=os.environ["HELIGYM_RESOURCE_DIR"],
            hmap_path=terrain["hmap"],
            nmap_path=terrain["nmap"],
            east_west_extent=terrain["ew_max"],
            north_south_extent=terrain["ns_max"],
            maximum_ground_altitude=terrain["max_alt"],
            gas_constant=gas_constant,
            sea_level_temperature=sea_level_temperature,
            temperature_lapse_rate=temperature_lapse_rate,
            troposphere_height=troposphere_height,
            sea_level_air_density=sea_level_air_density,
            gravity=gravity,
        )
        self.wind_dyn = WindDynamics(
            dt=DT,
            wind_direction=wind_direction,
            wind_speed=effective_wind_speed,
            turbulence_level=turbulence_level,
        )
        self.heli_dyn.set_wind(self.wind_dyn.wind_mean_ned) # set mean wind as wind so that helicopter trims accordingly
        self.action_space = spaces.Box(-1, +1, (self.heli_dyn.n_act,), dtype=np.float32)
        self.previous_action = np.zeros(self.heli_dyn.n_act, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate((np.full(self.heli_dyn.n_obs, -np.inf, dtype=np.float32),
                                self.action_space.low)),
            high=np.concatenate((np.full(self.heli_dyn.n_obs, np.inf, dtype=np.float32),
                                 self.action_space.high)),
            dtype=np.float32,
        )
        self.successed_time = 0 # time counter for successing task through time.
        self.set_max_time()
        self.set_target()
        self.set_trim_cond()
        self.set_reward_weights()
        self.normalizers = {
                                "t": np.sqrt(2*self.heli_dyn.MR["R"]/self.heli_dyn.gravity),
                                "x": 2*self.heli_dyn.MR["R"],
                                "v": np.sqrt(2*self.heli_dyn.MR["R"]*self.heli_dyn.gravity),
                                "a": self.heli_dyn.gravity
                            }
        
        self.renderer = None
        if render_mode == "human":
            from .heli_render import HeliRender
            model_name = os.path.splitext(os.path.basename(os.fspath(heli_name)))[0]
            self.renderer = HeliRender(heli_name=model_name, fps=FPS)
        self.paused = False
       
    # Setter functions for RL tasks
    def set_max_time(self, max_time=None):
        self.max_time = self.default_max_time if max_time is None else max_time  # [sec] Given time for episode
        self.success_duration = self.max_time/4 # [sec] Required successfull time for maneuver
        self.task_duration = self.max_time/4 # [sec] Allowed time for successing task

    def set_target(self, target={}):
        self.task_target = copy.deepcopy(self.default_task_target)
        self.task_target.update(target)

    def get_target(self):
        return copy.deepcopy(self.task_target)

    def set_trim_cond(self, trim_cond={}):
        self.trim_cond = copy.deepcopy(self.default_trim_cond)
        self.trim_cond.update(trim_cond)

    def get_trim_cond(self):
        return copy.deepcopy(self.trim_cond)

    def set_reward_weights(self, base_reward_weight=None, terminal_reward_weight=None):
        zero_ = np.zeros((self.heli_dyn.n_obs,self.heli_dyn.n_obs))
        self.base_reward_weight = zero_ if base_reward_weight is None else base_reward_weight
        self.terminal_reward_weight = zero_ if terminal_reward_weight is None else terminal_reward_weight

    def process_keyboard(self, actions=None, update_action=False):
        """Update manual controls; callers skip step() while paused, but keep rendering."""
        if actions is None:
            #actions = np.array(self.heli_dyn.action, dtype=np.float32, copy=True)
            actions = self.heli_dyn.action
        #actions = np.array(actions, dtype=np.float32, copy=True)
        delta = np.zeros(self.heli_dyn.n_act, dtype=np.float32)
        changes = {
            'z': (0, 0.2), 's': (0, -0.2),
            'up': (1, 0.2), 'down': (1, -0.2),
            'right': (2, 0.2), 'left': (2, -0.2),
            'd': (3, 0.2), 'q': (3, -0.2),
        }
        for key in self.renderer.get_key_presses():
            if key == 'p':
                self.paused = not self.paused
            elif key == 'r':
                self.reset()
                #actions = np.array(self.heli_dyn.action, dtype=np.float32, copy=True)
            elif key == 'x':
                delta = -np.array(actions, dtype=np.float32, copy=True)
                #actions.fill(0.0)
            elif key in changes:
                index, v = changes[key]
                delta[index] = v
                #actions[index] = np.clip(actions[index] + v, -1.0, 1.0)
        # Update the action displays even when physics is paused.
        if update_action:
            self.heli_dyn.action = np.clip(actions + delta, -1.0, 1.0)
        return delta

    def render(self):
        if self.renderer is None:
            return
        self.renderer.render(
            state=self.heli_dyn.state,
            observation=self.heli_dyn.observation,
            action=self.heli_dyn.action,
            paused=self.paused,
        )
        
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
    
    def exit(self):
        sys.exit()

    def step(self, actions):
        """Advance physics, treating numerical divergence as a failed episode.

        Exploratory controls can push this model outside its numerical envelope.
        SB3 must receive finite terminal observations, including for replay and
        normalization. Keep the last valid state for the terminal frame; the
        caller must reset after any terminated/truncated transition.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.action_space.shape or not np.isfinite(actions).all():
            raise ValueError("Actions must have the action-space shape and be finite")
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        self.time_counter += DT
        previous_obs = np.copy(self.heli_dyn.observation)
        previous_state = self.heli_dyn.state.val.copy()
        previous_dots = self.heli_dyn.state_dots.val.copy()
        try:
            # Raise at the first bad numerical operation, before NaNs reach
            # terrain indexing. Underflow is harmless and remains permitted.
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                wind_action = np.concatenate([previous_obs[4:7], previous_obs[16:]])
                if self.wind_enabled:
                    self.heli_dyn.set_wind(self.wind_dyn.step(wind_action))
                observation = self.heli_dyn.step(actions)
                if not (np.isfinite(observation).all()
                        and np.isfinite(self.heli_dyn.state.val).all()
                        and np.isfinite(self.heli_dyn.state_dots.val).all()):
                    raise FloatingPointError("Non-finite helicopter dynamics")
                reward, successed_step = self._calculate_reward()
                if not np.isfinite(reward):
                    raise FloatingPointError("Non-finite task reward")
                info = self._get_info()
        except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
            self.heli_dyn.state.val[:] = previous_state
            self.heli_dyn.state_dots.val[:] = previous_dots
            self.heli_dyn.observation = previous_obs.copy()
            info = {'failed': True, 'successed': False,
                    'time_up': self._is_time_up(), 'numerical_failure': True,
                    'numerical_failure_reason': str(error)}
            self.previous_action = actions.copy()
            return self._agent_observation(previous_obs), -100.0, True, False, info
        done = info['failed'] or info['successed']
        truncated = info['time_up']
        self.successed_time += DT if successed_step else 0
        self.previous_action = actions.copy()
        return self._agent_observation(observation), float(reward), bool(done), bool(truncated), info

    def _agent_observation(self, dynamics_observation):
        """Append the last direct policy action to the dynamics observation."""
        return np.concatenate((dynamics_observation, self.previous_action)).astype(
            np.float32, copy=False
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Wind dynamics currently draw from NumPy's process-local generator.
        if seed is not None:
            np.random.seed(seed)
        self.paused = False
        self.time_counter = 0
        self.successed_time = 0
        self.wind_dyn.reset()
        # Trim in the same mean wind as on construction, never in the previous
        # episode's instantaneous turbulence (which may have diverged).
        self.heli_dyn.set_wind(self.wind_dyn.wind_mean_ned.copy())
        self.heli_dyn.reset(self.trim_cond)
        # The reset state was produced with the trim controls, so expose those
        # controls as the action immediately preceding the first policy step.
        self.previous_action[:] = self.heli_dyn.action
        return self._agent_observation(self.heli_dyn.observation), self._get_info()

    def _get_info(self):
        return {
            'failed': self._is_failed(), 
            'successed': self._is_successed(), 
            'time_up': self._is_time_up()
            }

    def _is_failed(self):
        cond1 = self.heli_dyn._does_hit_ground(-self.heli_dyn.state['xyz'][2])
        cond2 = self.heli_dyn.state_dots['xyz'][2] > self.heli_dyn.MR['V_TIP']*0.05
        cond3 = self.heli_dyn.state['euler'][0] > 60*D2R
        cond4 = self.heli_dyn.state['euler'][1] > 60*D2R
        cond5 = np.abs(self.heli_dyn.state['xyz'][0]) > self.heli_dyn.north_south_extent / 2 or np.abs(self.heli_dyn.state['xyz'][1]) > self.heli_dyn.east_west_extent / 2 \
             or -self.heli_dyn.state['xyz'][2] > self.heli_dyn.ground_touching_altitude() + 10000
        cond = (cond1 and (cond2 or cond3 or cond4)) or cond5
        return cond

    def _is_successed(self):
        return self.successed_time >= self.success_duration  

    def _is_time_up(self):
        return self.time_counter > self.max_time

    def _calculate_reward(self):
        return 0.0, False

if __name__=='__main__':
    env = Heli()
    #print(env)
    #obs = env.reset()
    #action = np.array([0.7007, 0.5391, 0.5351, 0.6429])
    #obs, reward, done, info = env.step(action)
