from gymnasium.envs.registration import register
from heligym.envs import Heli, ApproximateTargetValues, HeliHover, HeliForwardFlight, HeliStabilize, HeliRender
from heligym.envs import HeliSmoothHover, HeliSmoothStabilize, HeliGoto
from heligym.envs import HeliSmooth, HeliSmoothApproximateTargetValues, HeliSmoothForwardFlight, HeliSmoothGoto

register(
    id='Heli-v0',
    entry_point='heligym.envs:Heli',
    max_episode_steps = 5000,
    reward_threshold  = 0.95,
    nondeterministic  = False
)

register(
    id='HeliHover-v0',
    entry_point='heligym.envs:HeliHover',
    max_episode_steps = 5000,
    reward_threshold  = 0.95,
    nondeterministic  = False
)

register(
    id='HeliStabilize-v0',
    entry_point='heligym.envs:HeliStabilize',
    max_episode_steps=5000,
    reward_threshold=0.0,
    nondeterministic=False
)

for task in ('ApproximateTargetValues', 'HeliForwardFlight', 'HeliSmoothStabilize', 'HeliSmoothHover', 'HeliGoto',
             'HeliSmooth', 'HeliSmoothApproximateTargetValues', 'HeliSmoothForwardFlight', 'HeliSmoothGoto'):
    register(
        id=f'{task}-v0',
        entry_point=f'heligym.envs:{task}',
        max_episode_steps=5000,
        reward_threshold=0.0,
        nondeterministic=False
    )
