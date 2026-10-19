"""Train, evaluate, plot statistics, or render a helicopter controller.

Commands run in pipeline order: train -> eval -> plot -> render.
Omit commands to run the entire pipeline. Select one or more to run only those
stages. eval runs deterministic episodes without a GUI; render shows one episode.
Standalone eval/render load the model and saved run settings from --target-dir;
explicit command-line settings override saved settings.

Examples:
    python main.py --task hover
    python main.py --task forward-flight --algorithm sac --net-arch 256 256
    python main.py train eval --target-dir runs/hover --task hover
    python main.py train --total-timesteps 128 --n-envs 1 --n-steps 128 --batch-size 64
    python main.py eval plot --target-dir runs/hover
    python main.py render --target-dir runs/hover

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
    smooth-stabilize, smooth-hover
        Use the corresponding base task with an additional sum-of-squares
        penalty on changes in the four normalized actions between steps.
    heli
        Use the base helicopter environment. It has no task goal or shaped
        reward; reward remains zero until failure or the episode time limit.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
from functools import partial
import json
import math
import os
import sys
from pathlib import Path
import time


COMMANDS = ('train', 'eval', 'plot', 'render')


def log_status(message, end='\n'):
    """Show progress immediately, including when stdout is redirected to a file."""
    print(f'[heligym] {message}', end=end, flush=True)


@contextmanager
def progress(message, *, multiline=False):
    """Report timing inline unless the operation can print its own output."""
    started = time.perf_counter()
    log_status(message + '...', end='\n' if multiline else '')
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        if multiline:
            log_status(f'{message} failed after {elapsed:.2f}s')
        else:
            print(f' [failed] {elapsed:.2f}s', flush=True)
        raise
    else:
        elapsed = time.perf_counter() - started
        if multiline:
            log_status(f'{message} completed in {elapsed:.2f}s')
        else:
            print(f' [ok] {elapsed:.2f}s', flush=True)


def prepare_renderer():
    """Run the existing library-path bootstrap before importing the simulator.

    Importing heligym.envs.renderer normally imports the parent packages first.
    Those imports would be discarded if the bootstrap needs to restart Python.
    Running the lightweight bootstrap directly avoids that duplicate work.
    """
    if sys.platform != 'linux':
        return  # Other platforms configure their paths during normal import.
    import importlib.util

    bootstrap = Path(__file__).resolve().parent / 'heligym' / 'envs' / 'renderer' / '__init__.py'
    spec = importlib.util.spec_from_file_location('_heligym_renderer_bootstrap', bootstrap)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def positive_int(value):
    """Reject invalid counts at the command line, before creating environments."""
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value):
    """Accept zero to disable optional counts."""
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def parse_args(argv=None):
    """Expose common settings and algorithm-specific training controls."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('commands', nargs='*', metavar='COMMAND',
                        help='train, eval, plot, render; selected stages run in that order (default: all four)')
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--heli-name', 
                        default='aw109',
                        help='helicopter configuration name in heligym/envs/config/helis, or path to a YAML configuration (default: aw109)')
    environment.add_argument('--task',
                        choices=['hover', 'smooth-hover', 'forward-flight', 'forward',
                                 'stabilize', 'smooth-stabilize', 'heli'],
                        default='hover', 
                        help='environment task; forward is an alias for forward-flight (default: hover); see task descriptions above')
    environment.add_argument('--episode-seconds', 
                        type=float, 
                        default=40.0,
                        help='maximum simulated episode duration for training and evaluation, in seconds (default: 40)')
    wind = environment.add_mutually_exclusive_group()
    wind.add_argument('--wind', dest='no_wind', action='store_false',
                      help='enable steady wind and turbulence, overriding a saved --no-wind setting')
    wind.add_argument('--no-wind', dest='no_wind', action='store_true',
                      help='disable steady wind and turbulence during training, evaluation and rendering')
    parser.set_defaults(no_wind=False)
    environment.add_argument('--wind-speed', type=float, default=None,
                        help='steady wind speed in ft/s; uses the environment default if unspecified; ignored when wind is disabled')
    environment.add_argument('--wind-direction', type=float, default=None,
                        help='wind vector direction in degrees clockwise from north (0 north, 90 east); uses the environment default if unspecified; ignored when wind is disabled')
    environment.add_argument('--turbulence-level', type=int, choices=range(1, 8), default=None,
                        help='turbulence intensity from 1 (lightest) to 7 (strongest); uses the environment default if unspecified; ignored when wind is disabled')
    environment.add_argument('--no-online-normalization', 
                        action='store_true',
                        help='disable SB3 online observation and reward normalization during training (based on running means and variances) and skip loading observation statistics for evaluation; disabling it is not recommended for PPO/A2C, but may be useful for SAC/TD3/DDPG')

    execution = parser.add_argument_group('Training and execution')
    execution.add_argument('--algorithm', 
                        choices=['ppo', 'a2c', 'sac', 'td3', 'ddpg'], 
                        default='ppo',
                        help='algorithm used to train or load the policy (default: ppo)')
    execution.add_argument('--total-timesteps', type=positive_int, 
                        default=1_000_000, 
                        help='requested training transitions across all environments; rollout collection may exceed this count (default: 1000000)')
    execution.add_argument('--n-envs', type=positive_int, 
                        default=min(4, os.cpu_count() or 1),
                        help='number of training environments; also sets the evaluation seed offset (default: min(4, CPU count))')
    execution.add_argument('--vec-env', choices=['subproc', 'dummy'], default='subproc',
                        help='training environment backend: subproc uses parallel processes when n-envs > 1; dummy steps environments sequentially (default: subproc)')
    execution.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help='training seed; evaluation uses seed + n-envs (default: 42)')
    execution.add_argument('--device', 
                        default='auto',
                        choices=['cpu', 'cuda', 'auto'],  
                        help='PyTorch device for training or loading the policy (default: auto)')
    execution.add_argument('--torch-threads', 
                        type=positive_int, 
                        default=1,
                        help='PyTorch CPU thread count for this process (default: 1)')
    execution.add_argument('--verbose', 
                        type=int, 
                        choices=[0, 1, 2], 
                        default=1,
                        help='algorithm training verbosity: 0 silent, 1 info, 2 debug; runner messages still print (default: 1)')
    execution.add_argument('--eval-episodes', type=positive_int, default=10,
                        help='number of deterministic episodes for headless eval (default: 10); render always runs one episode')

    output = parser.add_argument_group('Artifacts and logging')
    output.add_argument('--target-dir', 
                        type=Path, 
                        default=Path(f'runs/local-{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'),
                        help='run directory for models, settings and statistics; eval/plot/render read saved artifacts here (default: runs/local-<timestamp>)')
    output.add_argument('--tensorboard-log', 
                        type=str, 
                        default=None,
                        help='directory for training TensorBoard logs (default: disabled)')
    output.add_argument('--pretrained-dir', 
                        type=Path, 
                        default=None,
                        help='currently unused; training always starts a new model; eval/render load from --target-dir')
    output.add_argument('--n-checkpoints', type=nonnegative_int, default=0,
                        help='number of intermediate model checkpoints before the final save (default: 0); evenly spaced across total timesteps, with up to n-envs - 1 transitions of rounding; capped by available vector steps')
    policy = parser.add_argument_group('Policy and shared optimization settings', description="Applicable constructor settings may be overridden by --algo-kwargs.")
    policy.add_argument('--net-arch', 
                        nargs='+', 
                        type=positive_int, default=[256, 256],
                        help='hidden layer widths for the policy and value/Q networks (default: 256 256); overridden by policy_kwargs in --algo-kwargs')
    policy.add_argument('--activation', 
                        choices=['tanh', 'relu', 'elu'], 
                        default=None,
                        help='activation function for actor and critic; defaults to tanh for on-policy methods and relu for off-policy methods')
    policy.add_argument('--learning-rate', 
                        type=float, 
                        default=3e-4,
                        help='constant optimizer learning rate (default: 0.0003); may be overridden by --algo-kwargs')
    policy.add_argument('--gamma', 
                        type=float, 
                        default=0.99,
                        help='discount factor, also used for training reward normalization (default: 0.99); may be overridden by --algo-kwargs')
    policy.add_argument('--batch-size', 
                        type=positive_int, 
                        default=None,
                        help='PPO minibatch size or SAC/TD3/DDPG replay batch size (default: 256); ignored by A2C; PPO requires it to divide rollout steps times n-envs')
    policy.add_argument('--ent-coef', 
                        type=float, 
                        default=None,
                        help='PPO/A2C/SAC only: entropy coefficient (default: 0 for PPO/A2C, automatic tuning for SAC)')

    on_policy = parser.add_argument_group('On-policy settings (PPO/A2C)', description="Applicable constructor settings may be overridden by --algo-kwargs.")
    on_policy.add_argument('--n-rollout-steps', '--n-steps', dest='n_steps', 
                        type=positive_int, 
                        default=None,
                        help='PPO/A2C only: rollout steps per environment (default: 1024 for PPO, 32 for A2C)')
    on_policy.add_argument('--n-epochs', 
                        type=positive_int, 
                        default=10,
                        help='PPO only: optimization passes over each rollout (default: 10)')
    on_policy.add_argument('--gae-lambda', 
                        type=float, 
                        default=0.95,
                        help='PPO/A2C only: generalized advantage estimation lambda (default: 0.95)')
    on_policy.add_argument('--clip-range', 
                        type=float, 
                        default=0.2,
                        help='PPO only: policy probability ratio clipping range (default: 0.2)')
    on_policy.add_argument('--vf-coef', 
                        type=float, 
                        default=0.5,
                        help='PPO/A2C only: value loss coefficient (default: 0.5)')
    on_policy.add_argument('--max-grad-norm', 
                        type=float, 
                        default=0.5,
                        help='PPO/A2C only: maximum gradient norm (default: 0.5)')

    off_policy = parser.add_argument_group('Replay and exploration settings (SAC/TD3/DDPG)', description="Applicable constructor settings may be overridden by --algo-kwargs.")
    off_policy.add_argument('--buffer-size', 
                        type=positive_int, 
                        default=300_000,
                        help='SAC/TD3/DDPG only: replay buffer capacity in transitions (default: 300000)')
    off_policy.add_argument('--learning-starts', 
                        type=int, 
                        default=10_000,
                        help='SAC/TD3/DDPG only: transitions collected before gradient updates start (default: 10000)')
    off_policy.add_argument('--train-freq', 
                        type=positive_int, 
                        default=1,
                        help='SAC/TD3/DDPG only: vector environment steps collected between training phases; each step collects n-envs transitions (default: 1)')
    off_policy.add_argument('--gradient-steps', 
                        type=int, 
                        default=-1,
                        help='SAC/TD3/DDPG only: gradient updates per training phase; -1 matches transitions just collected, 0 disables updates (default: -1)')
    off_policy.add_argument('--tau', 
                        type=float, 
                        default=0.005,
                        help='SAC/TD3/DDPG only: target network soft update coefficient (default: 0.005)')
    off_policy.add_argument('--action-noise', 
                        type=float, 
                        default=0.1, 
                        help='TD3/DDPG only: Gaussian exploration noise standard deviation for each of four normalized controls (default: 0.1)')

    advanced = parser.add_argument_group('Advanced overrides')
    advanced.add_argument('--algo-kwargs', 
                        type=json.loads, 
                        default={},
                        help='JSON object overriding algorithm constructor settings, including policy_kwargs; cannot override policy, env, seed, device, verbose or tensorboard_log')

    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_intermixed_args(argv)
    unknown_commands = set(args.commands) - set(COMMANDS)
    if unknown_commands:
        parser.error('unknown command(s): ' + ', '.join(sorted(unknown_commands))
                     + '; choose from ' + ', '.join(COMMANDS))
    commands = [command for command in COMMANDS
                if not args.commands or command in args.commands]
    # Reuse the training configuration for independent evaluation/rendering.
    # Parsing again preserves explicit CLI overrides, including --algorithm.
    saved_settings = args.target_dir / 'arguments.json'
    if 'train' not in commands and any(c in commands for c in ('eval', 'render')) and saved_settings.is_file():
        config = json.loads(saved_settings.read_text())
        run_fields = ('algorithm', 'task', 'heli_name', 'episode_seconds',
                      'no_wind', 'wind_speed', 'wind_direction', 'turbulence_level',
                      'no_online_normalization', 'seed', 'n_envs')
        parser.set_defaults(**{key: config[key] for key in run_fields if key in config})
        args = parser.parse_intermixed_args(argv)
    args.commands = commands
    if not isinstance(args.algo_kwargs, dict):
        parser.error('--algo-kwargs must be a JSON object')
    if args.learning_rate <= 0 or args.episode_seconds <= 0 or not 0 <= args.gamma <= 1:
        parser.error('learning rate and episode seconds must be positive; gamma must be in [0, 1]')
    if args.learning_starts < 0 or args.gradient_steps < -1 or args.action_noise < 0:
        parser.error('learning starts/noise must be nonnegative; gradient steps must be >= -1')
    if not args.no_wind:
        if args.wind_speed is not None and (not math.isfinite(args.wind_speed) or args.wind_speed < 0):
            parser.error('--wind-speed must be finite and nonnegative')
        if args.wind_direction is not None and not math.isfinite(args.wind_direction):
            parser.error('--wind-direction must be finite')
    # These are supplied by the runner, rather than algorithm hyperparameters.
    reserved = {'policy', 'env', 'seed', 'device', 'verbose', 'tensorboard_log'}
    if reserved.intersection(args.algo_kwargs):
        parser.error('--algo-kwargs cannot override ' + ', '.join(sorted(reserved)))
    return args


def make_env(task, heli_name, episode_seconds, render_mode=None, no_wind=False,
             wind_speed=None, wind_direction=None, turbulence_level=None):
    """Construct each worker locally so no native objects cross process boundaries."""
    import heligym
    from stable_baselines3.common.monitor import Monitor
    classes = {'hover': heligym.HeliHover, 'forward': heligym.HeliForwardFlight,
               'forward-flight': heligym.HeliForwardFlight,
               'stabilize': heligym.HeliStabilize,
               'smooth-stabilize': heligym.HeliSmoothStabilize,
               'smooth-hover': heligym.HeliSmoothHover, 'heli': heligym.Heli}
    weather = {}
    if not no_wind:
        weather = {key: value for key, value in dict(
            wind_speed=wind_speed, wind_direction=wind_direction,
            turbulence_level=turbulence_level).items() if value is not None}
    env = classes[task](heli_name=heli_name, render_mode=render_mode,
                        wind_enabled=not no_wind, **weather)
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


def run_episode(model, args, normalization_path, seed, render=False):
    """Run one episode with frozen normalization and original task rewards."""
    label = 'Creating GUI environment and loading graphics' if render else f'Creating evaluation environment (seed {seed})'
    with progress(label, multiline=render):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from heligym.envs.helicopter import DT
        factory = partial(make_env, args.task, args.heli_name, args.episode_seconds,
                          render_mode='human' if render else None, no_wind=args.no_wind,
                          wind_speed=args.wind_speed, wind_direction=args.wind_direction,
                          turbulence_level=args.turbulence_level)
        env = DummyVecEnv([factory])
    try:
        reset_label = ('Resetting the helicopter' if args.no_online_normalization else
                       'Loading normalization statistics and resetting the helicopter')
        with progress(reset_label):
            if not args.no_online_normalization:
                env = VecNormalize.load(str(normalization_path), env)
                env.training = False
                env.norm_reward = False
            env.seed(seed)
            obs = env.reset()
            helicopter = env.envs[0].unwrapped
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        if render:
            log_status('GUI ready: P pauses/resumes playback; Esc closes the window.')
            helicopter.render()
        while not render or not helicopter.renderer.is_close():
            if render:
                # Policy playback owns the controls; only consume pause/resume
                # here, without applying manual controls or resetting physics.
                for key in helicopter.renderer.get_key_presses():
                    if key == 'p':
                        helicopter.paused = not helicopter.paused
                if helicopter.paused:
                    helicopter.render()  # Wait for events without advancing physics.
                    continue
            start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            # Bypass VecEnv auto-reset to preserve the terminal frame and reward.
            raw_obs, reward, terminated, truncated, _ = env.envs[0].step(action[0])
            total_reward += float(reward)
            steps += 1
            if render:
                helicopter.render()
            if terminated or truncated:
                break
            obs = raw_obs[None, :]
            if not args.no_online_normalization:
                obs = env.normalize_obs(obs)
            if render:
                time.sleep(max(0.0, DT - (time.perf_counter() - start)))
        return dict(seed=seed, steps=steps, reward=total_reward,
                    terminated=bool(terminated), truncated=bool(truncated))
    finally:
        env.close()


def evaluate(model, args, normalization_path):
    """Save independent deterministic evaluation episodes and summary statistics."""
    import numpy as np

    log_status(f'Evaluating {args.eval_episodes} deterministic episodes without rendering')
    episodes = []
    for index in range(args.eval_episodes):
        result = run_episode(model, args, normalization_path,
                             args.seed + args.n_envs + index)
        episodes.append(result)
        print(f"Evaluation {index + 1}/{args.eval_episodes}: "
              f"steps={result['steps']}, return={result['reward']:.3f}")
    rewards = [episode['reward'] for episode in episodes]
    statistics = dict(episodes=episodes, mean_reward=float(np.mean(rewards)),
                      std_reward=float(np.std(rewards)),
                      mean_steps=float(np.mean([e['steps'] for e in episodes])))
    write_json(args.target_dir / 'evaluation.json', statistics)
    print(f"Evaluation: mean return={statistics['mean_reward']:.3f}, "
          f"std={statistics['std_reward']:.3f}")
    return statistics


def write_json(path, value):
    """Write run artifacts in a format later commands can read."""
    temporary_path = path.with_name(path.name + '.tmp')
    temporary_path.write_text(json.dumps(value, indent=2) + '\n')
    temporary_path.replace(path)


def train(algorithm, args):
    """Train and persist the model, normalization, configuration and episode data."""
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    class TrainingCallback(BaseCallback):
        """Collect episode history and checkpoint by total transitions."""
        def __init__(self):
            super().__init__()
            self.episodes = []
            # Only vector steps strictly before the requested end can produce
            # distinct intermediate checkpoints. The final model is saved below.
            self.checkpoint_count = min(args.n_checkpoints,
                                        (args.total_timesteps - 1) // args.n_envs)
            self.saved_checkpoints = 0
            self.checkpoint_root = None
            if self.checkpoint_count < args.n_checkpoints:
                print(f'Only {self.checkpoint_count} intermediate checkpoints fit '
                      f'before {args.total_timesteps} transitions with {args.n_envs} environments.')

        def _save_checkpoint(self):
            with progress(f'Saving checkpoint at {self.num_timesteps} transitions', multiline=True):
                if self.checkpoint_root is None:
                    self.checkpoint_root = args.target_dir / 'checkpoints' / datetime.now().strftime(
                        '%Y_%m_%d_%H_%M_%S_%f')
                checkpoint_dir = self.checkpoint_root / f'step_{self.num_timesteps}'
                checkpoint_dir.mkdir(parents=True, exist_ok=False)
                self.model.save(str(checkpoint_dir / 'model'))
                normalization = self.model.get_vec_normalize_env()
                if normalization is not None:
                    normalization.save(str(checkpoint_dir / 'vecnormalize.pkl'))
                # Each checkpoint can be evaluated/rendered using --target-dir.
                write_json(checkpoint_dir / 'arguments.json', config)
                # Episode history is cumulative; preserve only the latest copy.
                write_json(args.target_dir / 'learning_curves.json', self.episodes)
                print(f'Checkpoint saved at {self.num_timesteps} transitions: {checkpoint_dir}')

        def _on_step(self):
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode = info['episode']
                    self.episodes.append((self.num_timesteps, float(episode['r']),
                                          int(episode['l'])))
            if self.saved_checkpoints < self.checkpoint_count and self.num_timesteps < args.total_timesteps:
                # Integer fractions spread rounding over the run, avoiding drift
                # from repeatedly adding a rounded interval. Count all workers.
                threshold = ((self.saved_checkpoints + 1) * args.total_timesteps
                             // (self.checkpoint_count + 1))
                if self.num_timesteps >= threshold:
                    self._save_checkpoint()
                    self.saved_checkpoints += 1
            return True

    settings = agent_settings(args)
    args.target_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    write_json(args.target_dir / 'arguments.json', config)
    factory = partial(make_env, args.task, args.heli_name, args.episode_seconds,
                      no_wind=args.no_wind, wind_speed=args.wind_speed,
                      wind_direction=args.wind_direction, turbulence_level=args.turbulence_level)
    backend = 'subprocess' if args.n_envs > 1 and args.vec_env == 'subproc' else 'sequential'
    with progress(f'Creating {args.n_envs} training environments ({backend})'):
        if args.n_envs > 1 and args.vec_env == 'subproc':
            env = SubprocVecEnv([factory] * args.n_envs, start_method='spawn')
        else:
            env = DummyVecEnv([factory] * args.n_envs)
    training_callback = TrainingCallback()
    try:
        if not args.no_online_normalization:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=settings['gamma'])
        env.seed(args.seed)
        with progress(f'Initializing {args.algorithm.upper()} policy (device: {args.device})', multiline=True):
            model = algorithm('MlpPolicy', env, seed=args.seed, device=args.device,
                              verbose=args.verbose, tensorboard_log=args.tensorboard_log,
                              **settings)
        with progress(f'Training for {args.total_timesteps} transitions', multiline=True):
            model.learn(total_timesteps=args.total_timesteps, callback=training_callback)
        with progress(f'Saving final model and statistics to {args.target_dir}', multiline=True):
            model.save(str(args.target_dir / 'model'))
            if not args.no_online_normalization:
                env.save(str(args.target_dir / 'vecnormalize.pkl'))
            write_json(args.target_dir / 'learning_curves.json', training_callback.episodes)
            # Discard derived artifacts belonging to an older policy only after
            # the replacement model and its statistics have been saved.
            for name in ('evaluation.json', 'evaluation.png', 'learning_curves.png'):
                (args.target_dir / name).unlink(missing_ok=True)
            if args.no_online_normalization:
                (args.target_dir / 'vecnormalize.pkl').unlink(missing_ok=True)
            print(f'Training artifacts saved to {args.target_dir.resolve()}')
        return model
    finally:
        env.close()


def load_model(algorithm, args):
    """Load a policy for commands that run without training first."""
    model_path = args.target_dir / 'model.zip'
    if not model_path.is_file():
        raise FileNotFoundError(f'No saved policy found at {model_path.resolve()}; '
                                'run train first or select an existing --target-dir')
    normalization_path = args.target_dir / 'vecnormalize.pkl'
    if not args.no_online_normalization and not normalization_path.is_file():
        raise FileNotFoundError(f'No normalization statistics found at {normalization_path.resolve()}')
    with progress(f'Loading policy from {model_path.resolve()}'):
        model = algorithm.load(str(model_path), device=args.device)
    return model


def main(argv=None):
    """Execute selected commands in pipeline order, sharing a model when possible."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if '-h' in argv or '--help' in argv:
        parse_args(argv)  # Keep --help fast and free of startup messages.
        return
    with progress('Reading arguments and saved run settings'):
        args = parse_args(argv)
    log_status('Selected commands: ' + ' -> '.join(args.commands))
    # Configure native paths before importing any heavy packages. The bootstrap
    # may restart Python once; headless commands do not need it.
    if 'render' in args.commands:
        with progress('Preparing renderer library paths', multiline=True):
            prepare_renderer()

    if any(command in args.commands for command in ('train', 'eval', 'render')):
        with progress('Importing PyTorch'):
            import torch
        with progress('Importing Stable Baselines3'):
            from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
        torch.set_num_threads(args.torch_threads)
        algorithm = {'ppo': PPO, 'a2c': A2C, 'sac': SAC,
                     'td3': TD3, 'ddpg': DDPG}[args.algorithm]

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    print('Arguments:\n' + json.dumps(config, indent=2), flush=True)
    model = None
    normalization_path = args.target_dir / 'vecnormalize.pkl'
    for command in args.commands:
        with progress(f'Running {command} command', multiline=True):
            if command == 'train':
                model = train(algorithm, args)
            elif command in ('eval', 'render'):
                if model is None:
                    model = load_model(algorithm, args)
                if command == 'eval':
                    evaluate(model, args, normalization_path)
                else:
                    result = run_episode(model, args, normalization_path,
                                         args.seed + args.n_envs, render=True)
                    print(f"Rendered episode: steps={result['steps']}, return={result['reward']:.3f}")
            elif command == 'plot':
                plot_statistics(args.target_dir)


def plot_statistics(target_dir):
    """Plot persisted data without loading a policy or creating a simulator."""
    log_status(f'Reading saved plotting statistics from {target_dir}')
    training_path = target_dir / 'learning_curves.json'
    evaluation_path = target_dir / 'evaluation.json'
    if not training_path.is_file() and not evaluation_path.is_file():
        raise FileNotFoundError(f'No saved statistics found in {target_dir.resolve()}; '
                                'run train or eval first')
    if training_path.is_file():
        plot_learning_curves(json.loads(training_path.read_text()), target_dir)
    if evaluation_path.is_file():
        with progress('Loading Matplotlib for evaluation plots'):
            import matplotlib.pyplot as plt

        statistics = json.loads(evaluation_path.read_text())
        episodes = statistics['episodes']
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        for ax, key, label in zip(axes, ('reward', 'steps'),
                                  ('Episode return', 'Episode length (steps)')):
            ax.bar(range(1, len(episodes) + 1), [episode[key] for episode in episodes])
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=0.3)
        axes[-1].set_xlabel('Evaluation episode')
        fig.suptitle('Evaluation statistics')
        fig.tight_layout()
        fig.savefig(target_dir / 'evaluation.png')
        log_status('Displaying evaluation plots; close the plot window to continue.')
        plt.show()
        plt.close(fig)


def plot_learning_curves(episodes, target_dir):
    """Save and display learning curves for completed training episodes."""
    if not episodes:
        print('No completed episodes; learning curves are unavailable.')
        return

    with progress('Loading Matplotlib for learning curves'):
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
    fig.savefig(target_dir / 'learning_curves.png')
    log_status('Displaying learning curves; close the plot window to continue.')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
