import gymnasium as gym
import heligym
import numpy as np
import time
import copy

env = heligym.HeliHover(wind_enabled=False, render_mode="human")
#env.set_max_time(1000)

init_state = env.reset()
init_action = env.heli_dyn.action
print(f"init_state: {init_state}")
print(f"init_action: {init_action}")

#action = np.zeros(4)

print(f"initial action (trim): {init_action}")

try:
    steps = 0
    while steps < 1000000 and not env.renderer.is_close():
        delta = env.process_keyboard()
        if env.paused:
            env.render()
            continue
        current_action = env.heli_dyn.action
        new_action = np.clip(current_action + delta, -1.0, 1.0)
        _, r, done, truncated, info = env.step(new_action)
        steps += 1
        env.render()
finally:
    env.close()
