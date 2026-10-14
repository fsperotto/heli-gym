"""Train a helicopter controller, save it, and watch one deterministic episode.

Examples:
    python learn.py --task hover
    python learn.py --task forward-flight --algorithm sac --net-arch 256 256
    python learn.py --total-timesteps 128 --n-envs 1 --n-steps 128 --no-render
    python learn.py --no-learn

Install training dependencies with: pip install -e '.[train]'
Defaults are starting points, not a guarantee of convergence on either task.

Tasks:
    hover
        Hold position (0, 0) at 2,000 ft sea altitude while minimizing
        body angular rates.
    forward, forward-flight
        Reach and maintain 100 ft/s at 2,000 ft sea altitude
        while minimizing body angular rates.
    stabilize
        Keep the helicopter motionless and level. Motion is penalized, tilt is
        penalized heavily, and crashing receives a severe penalty.
    heli
        Use the base helicopter environment. It has no task goal or shaped
        reward; reward remains zero until failure or the episode time limit.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import time


def positive_int(value):
    """Reject invalid counts at the command line, before creating environments."""
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args():
    """Expose common settings and algorithm-specific training controls."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task',
                        choices=['hover', 'forward-flight', 'forward', 'stabilize', 'heli'],
                        default='hover', help='training task; see task descriptions below')
    parser.add_argument('--algorithm', choices=['ppo', 'a2c', 'sac', 'td3', 'ddpg'], default='ppo')
    parser.add_argument('--heli-name', default='aw109')
    parser.add_argument('--total-timesteps', type=positive_int, default=200_000)
    parser.add_argument('--n-envs', type=positive_int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--vec-env', choices=['subproc', 'dummy'], default='subproc',
                        help='subproc runs physics in parallel; dummy runs sequentially')
    parser.add_argument('--net-arch', nargs='+', type=positive_int, default=[256, 256],
                        help='hidden layer widths for actor and critic')
    parser.add_argument('--activation', choices=['tanh', 'relu', 'elu'], default=None)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu', help='cpu, cuda, or auto')
    parser.add_argument('--torch-threads', type=positive_int, default=1,
                        help='avoid CPU thread oversubscription with small MLPs')
    parser.add_argument('--episode-seconds', type=float, default=40.0)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/heli'))
    parser.add_argument('--tensorboard-log', type=str, default=None)
    parser.add_argument('--no-normalize', action='store_true')
    parser.add_argument('--no-wind', action='store_true',
                        help='disable steady wind and turbulence during training and evaluation')
    parser.add_argument('--no-learn', action='store_true',
                        help='load the saved policy and only watch an inference episode')
    parser.add_argument('--no-render', action='store_true', help='skip the final GUI episode')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1)
    # On-policy rollout lengths are per environment, not total across workers.
    parser.add_argument('--n-steps', type=positive_int, default=None)
    parser.add_argument('--batch-size', type=positive_int, default=None)
    parser.add_argument('--n-epochs', type=positive_int, default=10)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--ent-coef', type=float, default=None,
                        help='PPO/A2C: 0 by default; SAC: automatic tuning unless specified')
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    # Replay-based methods collect transitions before sampling training batches.
    parser.add_argument('--buffer-size', type=positive_int, default=300_000)
    parser.add_argument('--learning-starts', type=int, default=10_000)
    parser.add_argument('--train-freq', type=positive_int, default=1)
    parser.add_argument('--gradient-steps', type=int, default=-1,
                        help='-1 matches updates to transitions collected across workers')
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--action-noise', type=float, default=0.1, help='TD3/DDPG exploration std')
    parser.add_argument('--algo-kwargs', type=json.loads, default={},
                        help='JSON object overriding any algorithm constructor setting')
    args = parser.parse_args()
    if not isinstance(args.algo_kwargs, dict):
        parser.error('--algo-kwargs must be a JSON object')
    if args.learning_rate <= 0 or args.episode_seconds <= 0 or not 0 <= args.gamma <= 1:
        parser.error('learning rate and episode seconds must be positive; gamma must be in [0, 1]')
    if args.learning_starts < 0 or args.gradient_steps < -1 or args.action_noise < 0:
        parser.error('learning starts/noise must be nonnegative; gradient steps must be >= -1')
    if args.no_learn and args.no_render:
        parser.error('--no-learn cannot be combined with --no-render')
    # These are supplied by the runner, rather than algorithm hyperparameters.
    reserved = {'policy', 'env', 'seed', 'device', 'verbose', 'tensorboard_log'}
    if reserved.intersection(args.algo_kwargs):
        parser.error('--algo-kwargs cannot override ' + ', '.join(sorted(reserved)))
    return args


def make_env(task, heli_name, episode_seconds, render_mode=None, no_wind=False):
    """Construct each worker locally so no native objects cross process boundaries."""
    import heligym
    from stable_baselines3.common.monitor import Monitor
    classes = {'hover': heligym.HeliHover, 'forward': heligym.HeliForwardFlight,
               'forward-flight': heligym.HeliForwardFlight,
               'stabilize': heligym.HeliStabilize, 'heli': heligym.Heli}
    env = classes[task](heli_name=heli_name, render_mode=render_mode,
                        wind_enabled=not no_wind)
    env.set_max_time(episode_seconds)
    return Monitor(env)


def agent_settings(args):
    """Only pass arguments supported by the selected algorithm."""
    import torch
    from stable_baselines3.common.noise import NormalActionNoise
    import numpy as np
    on_policy = args.algorithm in ('ppo', 'a2c')
    activation = args.activation or ('tanh' if on_policy else 'relu')
    settings = dict(learning_rate=args.learning_rate, gamma=args.gamma,
                    policy_kwargs=dict(net_arch=args.net_arch,
                                       activation_fn={'tanh': torch.nn.Tanh,
                                                      'relu': torch.nn.ReLU,
                                                      'elu': torch.nn.ELU}[activation]))
    if on_policy:
        settings.update(n_steps=args.n_steps or (1024 if args.algorithm == 'ppo' else 32),
                        gae_lambda=args.gae_lambda, ent_coef=args.ent_coef or 0.0,
                        vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm)
        if args.algorithm == 'ppo':
            settings.update(batch_size=args.batch_size or 256, n_epochs=args.n_epochs,
                            clip_range=args.clip_range)
    else:
        settings.update(buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                        batch_size=args.batch_size or 256, tau=args.tau,
                        train_freq=args.train_freq, gradient_steps=args.gradient_steps)
        if args.algorithm == 'sac':
            settings['ent_coef'] = 'auto' if args.ent_coef is None else args.ent_coef
        else:
            # All current helicopters have four normalized continuous controls.
            settings['action_noise'] = NormalActionNoise(np.zeros(4), args.action_noise * np.ones(4))
    settings.update(args.algo_kwargs)
    if args.algorithm == 'ppo':
        rollout = settings['n_steps'] * args.n_envs
        if rollout <= 1 or settings['batch_size'] <= 1 or rollout % settings['batch_size']:
            raise ValueError('PPO needs rollout > 1 and batch size > 1 dividing n_steps * n_envs')
    return settings


def watch_episode(model, args, normalization_path):
    """Freeze training statistics and show exactly one episode at simulator speed."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from heligym.envs.helicopter import DT
    factory = partial(make_env, args.task, args.heli_name, args.episode_seconds, 'human',
                      no_wind=args.no_wind)
    env = DummyVecEnv([factory])
    try:
        if not args.no_normalize:
            env = VecNormalize.load(str(normalization_path), env)
            env.training = False
            env.norm_reward = False  # Report the task's actual reward.
        env.seed(args.seed + args.n_envs)
        obs = env.reset()
        helicopter = env.envs[0].unwrapped
        total_reward = 0.0
        steps = 0
        helicopter.render()
        while not helicopter.renderer.is_close():
            start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            # Step the raw env to avoid VecEnv's automatic reset after termination:
            # this preserves the final frame instead of displaying a new episode.
            raw_obs, reward, terminated, truncated, info = env.envs[0].step(action[0])
            total_reward += float(reward)
            steps += 1
            helicopter.render()
            if terminated or truncated:
                print(f'Evaluation: steps={steps}, return={total_reward:.3f}, info={info}')
                break
            obs = raw_obs[None, :]
            if not args.no_normalize:
                obs = env.normalize_obs(obs)
            time.sleep(max(0.0, DT - (time.perf_counter() - start)))
    finally:
        env.close()


def main():
    """Train without rendering, persist model/statistics, then open the viewer."""
    
    args = parse_args()
    
    # The renderer package may restart Python once to configure native library
    # paths. Do that before training, so opening the viewer cannot restart it.
    if not args.no_render:
        import heligym.envs.renderer
    
    # Delay heavyweight imports so --help works without training dependencies.
    import torch
    from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    torch.set_num_threads(args.torch_threads)

    class LearningCurveCallback(BaseCallback):
        """Collect original episode returns and lengths from all workers."""

        def __init__(self):
            super().__init__()
            self.episodes = []

        def _on_step(self):
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode = info['episode']
                    self.episodes.append((self.num_timesteps, episode['r'], episode['l']))
            return True

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    print('Arguments:\n' + json.dumps(config, indent=2), flush=True)

    algorithms = {'ppo': PPO, 'a2c': A2C, 'sac': SAC, 'td3': TD3, 'ddpg': DDPG}
    algorithm = algorithms[args.algorithm]
    model_path = args.output_dir / 'model.zip'
    normalization_path = args.output_dir / 'vecnormalize.pkl'

    if args.no_learn:
        if not model_path.is_file():
            raise FileNotFoundError(f'No saved policy found at {model_path.resolve()}')
        if not args.no_normalize and not normalization_path.is_file():
            raise FileNotFoundError(
                f'No normalization statistics found at {normalization_path.resolve()}')
        model = algorithm.load(str(model_path), device=args.device)
        print(f'Loaded policy from {model_path.resolve()}')
    else:
        learning_curve = LearningCurveCallback()
        settings = agent_settings(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'arguments.json').write_text(json.dumps(config, indent=2) + '\n')

        factory = partial(make_env, args.task, args.heli_name, args.episode_seconds,
                          no_wind=args.no_wind)

        # Spawn is portable and avoids inheriting unsafe Torch/OpenGL process state.
        if args.n_envs > 1 and args.vec_env == 'subproc':
            env = SubprocVecEnv([factory] * args.n_envs, start_method='spawn')
        else:
            env = DummyVecEnv([factory] * args.n_envs)

        try:
            if not args.no_normalize:
                env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=settings['gamma'])
            env.seed(args.seed)
            model = algorithm('MlpPolicy', env, seed=args.seed, device=args.device,
                              verbose=args.verbose, tensorboard_log=args.tensorboard_log,
                              **settings)
            model.learn(total_timesteps=args.total_timesteps, callback=learning_curve)
            model.save(str(args.output_dir / 'model'))
            if not args.no_normalize:
                env.save(str(normalization_path))
            print(f'Training artifacts saved to {args.output_dir.resolve()}')
        finally:
            # Release workers before creating the single native GUI renderer.
            env.close()

        plot_learning_curves(learning_curve.episodes, args.output_dir)
    
    if not args.no_render:
        watch_episode(model, args, normalization_path)


def plot_learning_curves(episodes, output_dir):
    """Save and display learning curves for completed training episodes."""
    if not episodes:
        print('No completed episodes; learning curves are unavailable.')
        return

    import matplotlib.pyplot as plt

    timesteps, rewards, lengths = zip(*episodes)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for ax, values, label in zip(axes, (rewards, lengths),
                                 ('Episode return', 'Episode length (steps)')):
        ax.plot(timesteps, values, alpha=0.4, label='Episode')
        window = min(20, len(values))
        averages = [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window)
                    for i in range(len(values))]
        ax.plot(timesteps, averages, label=f'Moving average ({window} episodes)')
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel('Training timesteps')
    fig.suptitle('Learning curves')
    fig.tight_layout()
    fig.savefig(output_dir / 'learning_curves.png')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
