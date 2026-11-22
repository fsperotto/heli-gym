"""JSON snapshots of mutable episode state. The environment supplies static configuration."""

import numpy as np


STATE_FORMAT_VERSION = 1

_TASK_FIELDS = ("_previous_values", "_previous_position", "targets_reached", "_use_reset_position")
_EPISODE_FIELDS = ("episode_seed", "step_count", "time_counter", "successed_time",
                   "current_reward", "recent_yaw_turns", "wind_enabled", "_configured_wind_speed")
_DYNAMICS_ARRAYS = ("action", "observation", "previous_observation")


def _array(value):
    if value is None:
        return None
    # Keep the file valid JSON even if a terminal wind update overflowed.
    values = value.tolist() if np.isfinite(value).all() else value.astype(str).tolist()
    return {"dtype": value.dtype.name, "values": values}


def _restore_array(value):
    if value is None:
        return None
    if value["dtype"] not in ("float32", "float64"):
        raise ValueError(f"Unsupported snapshot array dtype {value['dtype']!r}")
    return np.asarray(value["values"], dtype=value["dtype"])


def _capture_dynamics(dynamics):
    return {
        "state": _array(dynamics.state.val),
        "state_dots": _array(dynamics.state_dots.val),
        **{name: _array(getattr(dynamics, name)) for name in _DYNAMICS_ARRAYS},
    }


def _restore_dynamics(dynamics, snapshot):
    for name in ("state", "state_dots"):
        current = getattr(dynamics, name)
        values = _restore_array(snapshot[name])
        if values.shape != current.val.shape:
            raise ValueError(f"Snapshot {name} does not match the current simulator")
        current.val = values
    for name in _DYNAMICS_ARRAYS:
        setattr(dynamics, name, _restore_array(snapshot[name]))


def capture_episode_state(env):
    """Capture the reset state or the state immediately after an applied control."""
    wind = env.wind_dyn
    task = {}
    for name in _TASK_FIELDS:
        if hasattr(env, name):
            value = getattr(env, name)
            task[name] = _array(value) if isinstance(value, np.ndarray) else value
    return {
        "environment_class": type(env).__name__,
        "episode": {name: getattr(env, name) for name in _EPISODE_FIELDS},
        "helicopter": _capture_dynamics(env.heli_dyn),
        "wind": {
            **_capture_dynamics(wind),
            "speed": wind.wind_speed,
            "direction": wind.wind_dir,
            "mean_ned": _array(wind.wind_mean_ned),
            "current_ned": _array(env.heli_dyn.WIND_NED),
            "noise": _array(getattr(wind, "eta", None)),
            "rng": wind._rng.bit_generator.state,
        },
        "rng": env.np_random.bit_generator.state,
        "rng_seed": env.np_random_seed,
        "waypoint_rng": env._waypoint_rng.bit_generator.state,
        "target": {key: _array(value) if isinstance(value, np.floating) else value
                   for key, value in env.task_target.items()},
        "previous_action": _array(env.previous_action),
        "task": task,
    }


def restore_episode_state(env, snapshot):
    """Restore an initialized environment and return its raw observation."""
    if snapshot["environment_class"] != type(env).__name__:
        raise ValueError("Snapshot task does not match the current environment")
    for name in _EPISODE_FIELDS:
        setattr(env, name, snapshot["episode"][name])
    _restore_dynamics(env.heli_dyn, snapshot["helicopter"])
    wind = env.wind_dyn
    wind_snapshot = snapshot["wind"]
    _restore_dynamics(wind, wind_snapshot)
    wind.wind_speed = wind_snapshot["speed"]
    wind.wind_dir = wind_snapshot["direction"]
    wind.wind_mean_ned = _restore_array(wind_snapshot["mean_ned"])
    wind.eta = _restore_array(wind_snapshot["noise"])
    wind._rng.bit_generator.state = wind_snapshot["rng"]
    env.heli_dyn.set_wind(_restore_array(wind_snapshot["current_ned"]))
    env.np_random.bit_generator.state = snapshot["rng"]
    env._np_random_seed = snapshot["rng_seed"]
    env._waypoint_rng.bit_generator.state = snapshot["waypoint_rng"]
    env.task_target = {key: _restore_array(value)[()] if isinstance(value, dict) else value
                       for key, value in snapshot["target"].items()}
    env.previous_action = _restore_array(snapshot["previous_action"])
    for name in _TASK_FIELDS:
        if name in snapshot["task"]:
            value = snapshot["task"][name]
            setattr(env, name, _restore_array(value) if isinstance(value, dict) else value)
    if env.need_normalized_state:
        env._set_normalized_state()
    return env._agent_observation(env.heli_dyn.observation)
