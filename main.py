"""Train, evaluate, record, plot statistics, or render a helicopter controller.

Commands run in pipeline order: tune -> train -> eval -> record -> plot -> render. Replay and reload run separately.
With no command, display help. Select one or more commands to run those stages. eval runs deterministic episodes without a GUI; render shows one episode.
Standalone eval/render load the model and saved run settings from --target-dir.
Record can read a model from --source-dir and save episodes to --target-dir;
explicit command-line settings override saved settings.

Directory arguments by command:
    Command        Directory behavior
    -------------  ------------------------------------------------------------
    train          Writes the new run to --target-dir. Use --source-dir
                   (alias --pretrained-dir) to initialize from saved weights
                   and settings in a different directory.
    eval           Loads the policy from --target-dir and writes evaluation.json
                   there.
    record         Loads the policy from --source-dir and writes to record/
                   under --target-dir (records/ with --num-records). If only
                   one directory is given, it serves both roles. Use --full
                   to save restorable snapshots.
    plot           Reads saved statistics and writes PNGs in --target-dir.
    render         Loads the policy from --target-dir. Use --manual to fly
                   without loading a policy.
    tune           Saves fresh-policy trials under --target-dir/optuna/.
                   In tune train, --source-dir applies to training.
    replay         Reads config.json and actions.csv from --source-dir and
                   plays the recorded controls.
    reload         Reads config.json and states.json from --source-dir and
                   restores the simulator at --timestep.
    replay record  Reads recorded controls from --source-dir and saves to
                   --target-dir/record/. The episode folders must differ.

In a pipeline with train, later commands use the newly trained policy.
When recording with an existing policy, eval and render share that policy.

Examples:
    python main.py tune --task hover --target-dir runs/hover
    python main.py train eval --target-dir runs/hover --use-tuned
    python main.py train eval plot render --task hover
    python main.py train --task forward-flight --algorithm sac --net-arch 256 256
    python main.py train eval --target-dir runs/hover --task hover
    python main.py train --total-timesteps 128 --n-envs 1 --n-steps 128 --batch-size 64
    python main.py eval plot --target-dir runs/hover
    python main.py record --target-dir runs/hover
    python main.py record --source-dir pretrained/forward2 --target-dir runs/forward2-recordings
    python main.py replay --source-dir runs/hover/record
    python main.py reload --source-dir runs/hover/record --timestep 157
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
        Stop linear and angular motion at any position or attitude.
    smooth-stabilize, smooth-hover, smooth-forward-flight, smooth-goto, smooth-heli
        Add penalties for control effort, sustained yaw, and excessive roll and pitch.
    goto
        Pursue random waypoints using north/east/altitude differences in the observation.
        Reaching a target draws the next one without ending the episode.
    heli
        Use the base helicopter environment. It has no task goal or shaped
        reward; reward remains zero until failure or the episode time limit.
"""
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import csv
from datetime import datetime
from functools import partial
import json
import math
import os
import shlex
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from zoneinfo import ZoneInfo


COMMANDS = ('tune', 'train', 'eval', 'record', 'plot', 'render', 'replay', 'reload')

TOLERANCE_OPTIONS = {
    'horizontal_tolerance': 'horizontal position tolerance in ft (default: 20)',
    'altitude_tolerance': 'altitude tolerance in ft (default: 10)',
    'velocity_tolerance': 'tolerance per active body velocity component in ft/s (default: 5)',
    'heading_tolerance': 'heading tolerance in degrees (default: 5)',
    'yaw_turn_seconds': 'seconds per revolution at the yaw-rate tolerance (default: aircraft scale)',
}


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


def load_file_args(parser, path, *, saved_run=False, recording=False):
    """Read JSON defaults keyed by argparse destination names."""
    try:
        values = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        parser.error(f'{path}: {error}')
    if not isinstance(values, dict):
        parser.error(f'{path} must contain a JSON object')

    options = {action.dest: action for action in parser._actions if action.option_strings}
    if saved_run:
        # A saved run contains invocation metadata and may point to other runs.
        excluded = {'target_dir', 'pretrained_dir', 'file_args', 'use_tuned', 'tune_dir',
                    'manual', 'summary_only', 'source_dir', 'timestep', 'help'}
        if recording:
            excluded.add('num_records')
            if 'episode_seed' not in values:
                parser.error(f'{path}: recording is missing episode_seed')
            values['seed'] = values['episode_seed']
        if values.get('flip_angle') is not None and 'flip_angle_reference' not in values:
            values['flip_angle'] = 90 - values['flip_angle']
        values = {key: value for key, value in values.items()
                  if key in options and key not in excluded}
    defaults = {}
    for key, value in values.items():
        if key not in options or key in ('file_args', 'tune_dir', 'use_tuned', 'help'):
            parser.error(f'--file-args: unknown parameter {key!r}')
        action = options[key]
        if value is None and action.default is None:
            defaults[key] = None
            continue
        try:
            if isinstance(action.default, bool):
                if not isinstance(value, bool):
                    raise ValueError('expected a JSON boolean')
            elif action.type is json.loads:
                if not isinstance(value, dict):
                    raise ValueError('expected a JSON object')
            elif action.nargs == '+':
                if not isinstance(value, list) or not value:
                    raise ValueError('expected a nonempty JSON array')
                value = [action.type(str(item)) for item in value]
            else:
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    raise ValueError('expected a string or number')
                value = (action.type or str)(str(value))
            if action.choices is not None and value not in action.choices:
                raise ValueError(f'expected one of {list(action.choices)}')
        except (ValueError, argparse.ArgumentTypeError) as error:
            parser.error(f'{path}: invalid {key!r}: {error}')
        defaults[key] = value
    return defaults


def parse_args(argv=None, *, help_if_no_command=False, after_tune=False):
    """Expose common settings and algorithm-specific training controls."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('commands', nargs='*', metavar='COMMAND',
                        help='tune, train, eval, record, plot, render, or standalone replay/reload; replay record saves the playback')
    parameter_source = parser.add_mutually_exclusive_group()
    parameter_source.add_argument('--file-args', type=Path,
                        help='JSON defaults such as final_params.json; explicit CLI options override them')
    parameter_source.add_argument('--tune-dir', type=Path,
                                  help='load final_params.json from this study directory')
    parameter_source.add_argument('--use-tuned', action='store_true',
                                  help='load final_params.json from <target-dir>/optuna')
    parser.add_argument('--source-dir', type=Path,
                        help='saved policy for train or record, or episode directory for replay or reload')
    parser.add_argument('--timestep', type=nonnegative_int, default=0,
                        help='saved step to reload (default: 0, before the first action)')
    parser.add_argument('--manual', action='store_true',
                        help='render with keyboard controls only, without loading a policy')
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--heli-name', 
                        default='aw109',
                        help='helicopter configuration name in heligym/envs/config/helis, or path to a YAML configuration (default: aw109)')
    environment.add_argument('--task',
                        choices=['hover', 'smooth-hover', 'forward-flight', 'forward',
                                 'stabilize', 'smooth-stabilize', 'goto', 'heli',
                                 'smooth-goto', 'smooth-forward-flight', 'smooth-forward', 'smooth-heli'],
                        default='hover', 
                        help='environment task; forward is an alias for forward-flight (default: hover); see task descriptions above')
    environment.add_argument('--flip-angle', type=float, default=None,
                             help='maximum tilt from upright in degrees [0, 180]; defaults to the helicopter roll and pitch limits')
    environment.add_argument('--continual', action=argparse.BooleanOptionalAction, default=False,
                             help='draw a new waypoint on arrival in Goto (default: end the episode)')
    environment.add_argument('--smoothness-weight', type=float, default=0.25,
                             help='nonnegative penalty weight for smooth tasks (default: 0.25)')
    environment.add_argument('--episode-seconds', 
                        type=float, 
                        default=40.0,
                        help='maximum simulated episode duration for training and evaluation, in seconds (default: 40)')
    for name, description in TOLERANCE_OPTIONS.items():
        environment.add_argument('--' + name.replace('_', '-'), type=float, help=description)
    wind = environment.add_mutually_exclusive_group()
    wind.add_argument('--wind', dest='no_wind', action='store_false',
                      help='enable steady wind and turbulence, overriding a saved --no-wind setting')
    wind.add_argument('--no-wind', dest='no_wind', action='store_true',
                      help='disable steady wind and turbulence during training, evaluation and rendering')
    parser.set_defaults(no_wind=False)
    environment.add_argument('--random-wind', action=argparse.BooleanOptionalAction, default=False,
                             help='sample a wind speed up to --wind-speed (default 20 ft/s) and a direction for each episode seed')
    environment.add_argument('--wind-speed', type=float, default=None,
                        help='steady wind speed in ft/s; uses the environment default if unspecified; retained for W reactivation when wind is disabled')
    environment.add_argument('--wind-direction', type=float, default=None,
                        help='wind vector direction in degrees clockwise from north (0 north, 90 east); uses the environment default if unspecified; retained for W reactivation when wind is disabled')
    environment.add_argument('--turbulence-level', '--turbulence_level', type=int, choices=range(1, 8), default=None,
                        help='turbulence intensity from 1 (lightest) to 7 (strongest); uses the environment default if unspecified; retained for W reactivation when wind is disabled')
    normalization = environment.add_mutually_exclusive_group()
    normalization.add_argument('--no-online-normalization', action='store_true',
                               help='disable online observation and reward normalization')
    normalization.add_argument('--online-normalization', dest='no_online_normalization',
                               action='store_false', help='enable online normalization')
    parser.set_defaults(no_online_normalization=False)

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
                        help='training seed; standalone eval/render/record use an explicit seed directly (default training seed: 42)')
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
    execution.add_argument('--full-state', '--full', action='store_true',
                        help='record a restorable simulator snapshot at every step')
    execution.add_argument('--num-records', type=positive_int,
                        help='record N episodes under <target-dir>/records with consecutive seeds')
    execution.add_argument('--record-plot', action=argparse.BooleanOptionalAction, default=True,
                        help='save a time-series figure with each recording (default: enabled)')
    execution.add_argument('--eval-episodes', type=positive_int, default=10,
                        help='number of deterministic episodes for headless eval (default: 10); render always runs one episode')

    output = parser.add_argument_group('Artifacts and logging')
    output.add_argument('--target-dir', 
                        type=Path, 
                        help='run or output directory; record uses --source-dir if omitted (usual default: runs/local-<timestamp>)')
    tensorboard = output.add_mutually_exclusive_group()
    tensorboard.add_argument('--tensorboard', dest='tensorboard', action='store_true',
                             help='enable training TensorBoard logs in <target-dir>/tensorboard_log')
    tensorboard.add_argument('--no-tensorboard', dest='tensorboard', action='store_false',
                             help='disable TensorBoard logs (default)')
    parser.set_defaults(tensorboard=False)
    output.add_argument('--pretrained-dir', 
                        type=Path, 
                        default=None,
                        help='backward-compatible alias for --source-dir with train')
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

    tuning = parser.add_argument_group('Optuna tuning')
    tuning.add_argument('--n-trials', type=nonnegative_int, default=20,
                        help='additional search trials (default: 20)')
    tuning.add_argument('--trial-timesteps', type=positive_int, default=50_000,
                        help='training transitions per search trial (default: 50000)')
    tuning.add_argument('--refinement-timesteps', type=positive_int, default=300000,
                        help='training transitions per refinement run (default: 300000)')
    tuning.add_argument('--finalists', type=positive_int, default=3)
    tuning.add_argument('--refinement-seeds', type=positive_int, default=3,
                        help='fresh training seeds per finalist and baseline')
    tuning.add_argument('--summary-only', action='store_true',
                        help='print the tuning summary and exit')

    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_intermixed_args(argv)
    unknown_commands = set(args.commands) - set(COMMANDS)
    if unknown_commands:
        parser.error('unknown command(s): ' + ', '.join(sorted(unknown_commands))
                     + '; choose from ' + ', '.join(COMMANDS))
    if help_if_no_command and not args.commands:
        parser.print_help()
        return args
    commands = [command for command in COMMANDS if command in args.commands]
    recording_replay = commands == ['record', 'replay']
    recording_policy = 'record' in commands and 'train' not in commands and not recording_replay
    if 'tune' in commands:
        parser.set_defaults(task='smooth-hover', device='cpu', n_envs=4)
        args = parser.parse_intermixed_args(argv)
        default_target = Path('runs') / args.task
    else:
        default_target = Path(f'runs/local-{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
    target_dir = args.target_dir or (args.source_dir if recording_policy else None) or default_target
    parameter_path = args.file_args
    if 'tune' not in commands or after_tune:
        if args.use_tuned or args.tune_dir is not None:
            parameter_path = (args.tune_dir or target_dir / 'optuna') / 'final_params.json'
    file_defaults = load_file_args(parser, parameter_path) if parameter_path else {}
    if {'replay', 'reload'}.intersection(commands):
        command = 'reload' if 'reload' in commands else 'replay'
        if commands != [command] and commands != ['record', 'replay']:
            parser.error(f'{command} must be run separately, or use replay record to save playback')
        recording_dir = args.source_dir or file_defaults.get('source_dir')
        if recording_dir is None:
            parser.error(f'{command} requires --source-dir')
        recorded_defaults = load_file_args(parser, recording_dir / 'config.json',
                                           saved_run=True, recording=True)
        filename = 'states.json' if command == 'reload' else 'actions.csv'
        if not (recording_dir / filename).is_file():
            hint = '; record again with --full-state' if command == 'reload' else ''
            parser.error(f'--source-dir: no {filename} in {recording_dir}{hint}')
        file_defaults = {**recorded_defaults, **file_defaults}
    pretrained_defaults = {}
    training_source = None
    if 'train' in commands:
        # Either spelling on the command line overrides defaults from a file.
        source = args.source_dir
        pretrained = args.pretrained_dir
        if source is None and pretrained is None:
            source = file_defaults.get('source_dir')
            pretrained = file_defaults.get('pretrained_dir')
        if source is not None and pretrained is not None and source.resolve() != pretrained.resolve():
            parser.error('--source-dir and --pretrained-dir must refer to the same training source')
        training_source = source or pretrained
    # Tuning uses fresh policies. A combined command loads the source for training
    # after the tuning phase, together with any exported Optuna settings.
    if training_source is not None and ('tune' not in commands or after_tune):
        pretrained_defaults = load_file_args(parser, training_source / 'arguments.json', saved_run=True)
        if not (training_source / 'model.zip').is_file():
            parser.error(f'--source-dir: no model.zip in {training_source}')
        # Direct options beat matching entries in the source's algo_kwargs.
        inherited_kwargs = pretrained_defaults.get('algo_kwargs', {}).copy()
        options = parser._option_string_actions
        explicit = {options[name].dest for token in argv
                    if (name := token.partition('=')[0]) in options}
        for key in explicit | file_defaults.keys():
            inherited_kwargs.pop(key, None)
        if 'policy_kwargs' in inherited_kwargs and 'net_arch' in explicit | file_defaults.keys():
            policy = inherited_kwargs['policy_kwargs'].copy()
            policy['net_arch'] = args.net_arch if 'net_arch' in explicit else file_defaults['net_arch']
            inherited_kwargs['policy_kwargs'] = policy
        if 'algo_kwargs' in pretrained_defaults:
            pretrained_defaults['algo_kwargs'] = inherited_kwargs
    if pretrained_defaults or file_defaults:
        parser.set_defaults(**{**pretrained_defaults, **file_defaults})
        args = parser.parse_intermixed_args(argv)
    if args.target_dir is None:
        args.target_dir = (args.source_dir if recording_policy else None) or default_target
        parser.set_defaults(target_dir=args.target_dir)
    if 'reload' in commands:
        environment_fields = ('seed', 'task', 'heli_name', 'episode_seconds', 'flip_angle',
                              'smoothness_weight', 'continual', 'no_wind', 'random_wind',
                              'wind_speed', 'wind_direction', 'turbulence_level', *TOLERANCE_OPTIONS)
        changed = [key for key in environment_fields
                   if key in recorded_defaults and getattr(args, key) != recorded_defaults[key]]
        if changed:
            parser.error('reload requires the recorded environment settings. '
                         'Use replay to change: ' + ', '.join(changed))
    if training_source is not None:
        if args.target_dir.resolve() == training_source.resolve():
            parser.error('--target-dir must differ from --source-dir when training')
        source_algorithm = pretrained_defaults.get('algorithm')
        if source_algorithm and args.algorithm != source_algorithm:
            parser.error(f'--source-dir contains a {source_algorithm} policy. '
                         f'Training with {args.algorithm} requires a matching source policy')
        if (not args.no_online_normalization and pretrained_defaults.get('no_online_normalization') is False
                and not (training_source / 'vecnormalize.pkl').is_file()):
            parser.error(f'--source-dir: no vecnormalize.pkl in {training_source} for the normalized policy')
    # Reuse the training configuration for independent evaluation, recording, and rendering.
    # Parsing again preserves explicit CLI overrides, including --algorithm.
    policy_dir = (args.source_dir or args.target_dir) if recording_policy else args.target_dir
    saved_settings = policy_dir / 'arguments.json'
    if recording_replay:
        if (args.target_dir / 'record').resolve() == args.source_dir.resolve():
            parser.error('replay record requires a destination different from --source-dir')
        if args.num_records is not None:
            parser.error('replay record takes one recording directory at a time')
    replaying = set() if recording_replay else {'eval', 'record', 'render'}.intersection(commands)
    if replaying and 'train' not in commands and ('tune' not in commands or after_tune) and saved_settings.is_file():
        config = json.loads(saved_settings.read_text())
        if config.get('flip_angle') is not None and 'flip_angle_reference' not in config:
            config['flip_angle'] = 90 - config['flip_angle']
        run_fields = ('algorithm', 'task', 'heli_name', 'episode_seconds', 'flip_angle', 'smoothness_weight', 'continual',
                      'no_wind', 'random_wind', 'wind_speed', 'wind_direction', 'turbulence_level',
                      'no_online_normalization', 'seed', 'n_envs', 'device', 'torch_threads', 'eval_episodes', *TOLERANCE_OPTIONS)
        parser.set_defaults(**{key: config[key] for key in run_fields if key in config and key not in file_defaults})
        args = parser.parse_intermixed_args(argv)
    args.commands = commands
    if 'train' in commands:
        args.source_dir = args.pretrained_dir = training_source
    elif recording_policy:
        args.source_dir = policy_dir
    args.policy_dir = policy_dir
    if 'tune' in commands and not after_tune:
        if args.algorithm != 'ppo':
            parser.error('tune requires PPO')
        if args.random_wind and args.no_wind:
            parser.error('--random-wind requires wind to be enabled for tuning')
        if not 0 <= args.seed < 2**31:
            parser.error('tune requires --seed in [0, 2**31)')
    if not isinstance(args.algo_kwargs, dict):
        parser.error('--algo-kwargs must be a JSON object')
    if args.learning_rate <= 0 or args.episode_seconds <= 0 or not 0 <= args.gamma <= 1:
        parser.error('learning rate and episode seconds must be positive; gamma must be in [0, 1]')
    if args.learning_starts < 0 or args.gradient_steps < -1 or args.action_noise < 0:
        parser.error('learning starts/noise must be nonnegative; gradient steps must be >= -1')
    if args.wind_speed is not None and (not math.isfinite(args.wind_speed) or args.wind_speed < 0):
        parser.error('--wind-speed must be finite and nonnegative')
    if args.wind_direction is not None and not math.isfinite(args.wind_direction):
        parser.error('--wind-direction must be finite')
    if args.flip_angle is not None and (not math.isfinite(args.flip_angle) or not 0 <= args.flip_angle <= 180):
        parser.error('--flip-angle must be finite and between 0 and 180 degrees')
    if not math.isfinite(args.smoothness_weight) or args.smoothness_weight < 0:
        parser.error('--smoothness-weight must be finite and nonnegative')
    for name in TOLERANCE_OPTIONS:
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error('--' + name.replace('_', '-') + ' must be positive and finite')
    # These are supplied by the runner, rather than algorithm hyperparameters.
    reserved = {'policy', 'env', 'seed', 'device', 'verbose', 'tensorboard_log'}
    if reserved.intersection(args.algo_kwargs):
        parser.error('--algo-kwargs cannot override ' + ', '.join(sorted(reserved)))
    explicit_seed = any(arg == '--seed' or arg.startswith('--seed=') for arg in argv)
    replay_seed = args.seed if explicit_seed or 'seed' in file_defaults else args.seed + args.n_envs
    if {'eval', 'render'}.intersection(commands):
        args.evaluation_seed = args.seed + args.n_envs if 'train' in commands else replay_seed
    if 'replay' in commands and args.seed < 0:
        parser.error('replay seed must be nonnegative')
    if 'record' in commands:
        args.record_seed = args.seed if recording_replay else replay_seed
        if args.record_seed < 0 or args.record_seed + (args.num_records or 1) > 2**32:
            parser.error('recording seeds must be in [0, 2**32)')
    args.command_line = shlex.join([sys.executable, "main.py", *argv])
    return args


def make_env(task, heli_name, episode_seconds, render_mode=None, no_wind=False,
             wind_speed=None, wind_direction=None, turbulence_level=None, monitor=True, flip_angle=None,
             smoothness_weight=0.25, continual=False, tolerances=None, random_wind=False):
    """Construct each worker locally so no native objects cross process boundaries."""
    import heligym
    classes = {'hover': heligym.HeliHover, 'forward': heligym.HeliForwardFlight,
               'forward-flight': heligym.HeliForwardFlight,
               'stabilize': heligym.HeliStabilize,
               'smooth-stabilize': heligym.HeliSmoothStabilize,
               'smooth-hover': heligym.HeliSmoothHover, 'goto': heligym.HeliGoto,
               'heli': heligym.Heli, 'smooth-heli': heligym.HeliSmooth,
               'smooth-goto': heligym.HeliSmoothGoto,
               'smooth-forward': heligym.HeliSmoothForwardFlight,
               'smooth-forward-flight': heligym.HeliSmoothForwardFlight}
    weather = {key: value for key, value in dict(
        wind_speed=wind_speed, wind_direction=wind_direction,
        turbulence_level=turbulence_level).items() if value is not None}
    task_kwargs = dict(tolerances or {}) if task not in ('heli', 'smooth-heli') else {}
    if task.startswith('smooth-'):
        task_kwargs['smoothness_weight'] = smoothness_weight
    if task in ('goto', 'smooth-goto'):
        task_kwargs['continual'] = continual
    env = classes[task](heli_name=heli_name, render_mode=render_mode,
                        wind_enabled=not no_wind, random_wind=random_wind,
                        flip_angle=flip_angle, **weather, **task_kwargs)
    env.set_max_time(episode_seconds)
    if monitor:
        from stable_baselines3.common.monitor import Monitor
        return Monitor(env)
    return env


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


def episode_end_reasons(info, terminated, truncated, render=False):
    """Describe all reported terminal conditions, including simultaneous ones."""
    reasons = []
    if info.get('numerical_failure'):
        detail = info.get('numerical_failure_reason')
        reasons.append('error (numerical failure)' + (f': {detail}' if detail else ''))
    for key, label in (('crash', 'crash'), ('flip', 'flip'),
                       ('outside_limits', 'outside simulation limits')):
        if info.get(key):
            reasons.append(label)
    if info.get('failed') and not reasons:
        reasons.append('failure (unspecified)')
    if info.get('successed') and not info.get('failed'):
        reasons.append('success')
    if truncated:
        reasons.append('max time (truncated)' if info.get('time_up') else 'truncated')
    if not reasons:
        reasons.append('terminated (unspecified)' if terminated else
                       'window closed' if render else 'stopped')
    return reasons


def run_episode(model, args, normalization_path, seed, render=False, on_step=None,
                replay_actions=None, on_state=None, initial_state=None):
    """Run one episode with frozen normalization and original task rewards."""
    import numpy as np
    if render and args.manual:
        model = None
    if model is None and not render and replay_actions is None:
        raise ValueError('Evaluation requires an agent or recorded controls')
    normalize = model is not None and not args.no_online_normalization
    label = 'Creating GUI environment and loading graphics' if render else f'Creating evaluation environment (seed {seed})'
    with progress(label, multiline=render):
        from heligym.envs.helicopter import DT
        factory = partial(make_env, args.task, args.heli_name, args.episode_seconds,
                          render_mode='human' if render else None, no_wind=args.no_wind,
                          wind_speed=args.wind_speed, wind_direction=args.wind_direction,
                          random_wind=args.random_wind, turbulence_level=args.turbulence_level, flip_angle=args.flip_angle,
                          smoothness_weight=args.smoothness_weight, continual=args.continual,
                          tolerances={key: getattr(args, key) for key in TOLERANCE_OPTIONS
                                      if getattr(args, key) is not None})
        if model is None:
            env = factory(monitor=False)
            raw_env = env
        else:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            env = DummyVecEnv([factory])
            raw_env = env.envs[0]
    try:
        reset_label = ('Loading normalization statistics and resetting the helicopter' if normalize
                       else 'Resetting the helicopter')
        with progress(reset_label):
            if normalize:
                env = VecNormalize.load(str(normalization_path), env)
                env.training = False
                env.norm_reward = False
            if model is None:
                obs, _ = raw_env.reset(seed=seed)
                obs = obs[None, :]
            else:
                env.seed(seed)
                obs = env.reset()
            helicopter = raw_env.unwrapped
            if initial_state is not None:
                from heligym.envs.episode_state import restore_episode_state
                obs = restore_episode_state(helicopter, initial_state["state"])[None, :]
                helicopter.paused = True
            if on_step is not None:
                initial_obs = env.get_original_obs() if normalize else obs
                on_step(0 if initial_state is None else helicopter.step_count, None, initial_obs[0])
            replaying = replay_actions is not None and len(replay_actions) > 0
            manual = model is None and not replaying
            helicopter.pilot_mode = 'replay' if replaying else 'manual' if manual else 'agent'
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        final_info = {}
        if initial_state is not None:
            total_reward = initial_state["total_reward"]
            steps = helicopter.step_count
            terminated = initial_state["terminated"]
            truncated = initial_state["truncated"]
        if on_state is not None:
            on_state(helicopter, total_reward, terminated, truncated, final_info)
        if render:
            log_status(f'GUI ready: pilot={helicopter.pilot_mode}; M switches pilot when an agent is available; P pauses/resumes; Esc closes.')
            helicopter.render()
        while not render or not helicopter.renderer.is_close():
            if not render and replay_actions is not None and steps == len(replay_actions):
                break
            control_keys = []
            retrimmed = False
            if render:
                for key in helicopter.renderer.get_key_presses():
                    if key == 'm' and model is not None:
                        manual = not manual
                        helicopter.pilot_mode = 'manual' if manual else 'agent'
                        log_status(f'Pilot: {helicopter.pilot_mode}')
                    elif key == 't' and not replaying:
                        try:
                            raw_obs = helicopter.trim_at_current_position()
                            obs = raw_obs[None, :]
                            if normalize:
                                obs = env.normalize_obs(obs)
                            control_keys.clear()
                            retrimmed = True
                        except RuntimeError as error:
                            log_status(str(error))
                    elif key == 'w' and not replaying:
                        helicopter.toggle_wind()
                    elif key == 'p':
                        helicopter.paused = not helicopter.paused
                    elif manual and key == 'r':
                        raw_obs, _ = raw_env.reset()
                        obs = raw_obs[None, :]
                        if normalize:
                            obs = env.normalize_obs(obs)
                        total_reward = 0.0
                        steps = 0
                        final_info = {}
                        initial_state = None
                        terminated = truncated = False
                        control_keys.clear()
                    elif manual:
                        control_keys.append(key)
                if manual:
                    # Keep held commands through pause and agent/manual transitions.
                    delta = helicopter.process_keyboard(keys=control_keys)
                    helicopter.heli_dyn.action = np.clip(
                        helicopter.heli_dyn.action + delta,
                        helicopter.action_space.low, helicopter.action_space.high)
                if helicopter.paused or retrimmed or terminated or truncated:
                    helicopter.render()
                    continue
            start = time.perf_counter()
            if replaying:
                action = replay_actions[steps]
            elif manual:
                action = helicopter.heli_dyn.action.copy()
            else:
                predicted, _ = model.predict(obs, deterministic=True)
                action = predicted[0]
            # Bypass VecEnv auto-reset to preserve the terminal frame and reward.
            raw_obs, reward, terminated, truncated, final_info = raw_env.step(action)
            total_reward += float(reward)
            steps += 1
            if render and replaying and steps == len(replay_actions) and not (terminated or truncated):
                replaying = False
                manual = True
                helicopter.pilot_mode = 'manual'
                log_status('Recording exhausted. Manual control enabled.')
            if on_step is not None:
                on_step(steps, helicopter.previous_action, raw_obs)
            if on_state is not None:
                on_state(helicopter, total_reward, terminated, truncated, final_info)
            if render:
                helicopter.render()
            if terminated or truncated:
                break
            obs = raw_obs[None, :]
            if normalize:
                obs = env.normalize_obs(obs)
            if render:
                time.sleep(max(0.0, DT - (time.perf_counter() - start)))
        reasons = episode_end_reasons(final_info, terminated, truncated, render)
        if initial_state is not None and (terminated or truncated) and not final_info:
            reasons = initial_state["termination_reasons"]
        if not render and replay_actions is not None and not (terminated or truncated):
            reasons = ['recorded controls exhausted']
        return dict(seed=seed, steps=steps, reward=total_reward,
                    terminated=bool(terminated), truncated=bool(truncated),
                    termination_reasons=reasons)
    finally:
        env.close()


def read_replay_actions(directory):
    """Read and validate the step order and applied controls in a recording."""
    import numpy as np
    from heligym.envs.helicopter import Heli

    path = directory / 'actions.csv'
    with path.open(newline='') as stream:
        rows = csv.reader(stream, strict=True)
        if next(rows, None) != ['step', *Heli.action_names]:
            raise ValueError(f'{path}: unexpected action columns')
        actions = []
        for step, row in enumerate(rows, 1):
            try:
                if len(row) != len(Heli.action_names) + 1 or int(row[0]) != step:
                    raise ValueError('expected consecutive steps starting at 1 and four controls')
                values = [float(value) for value in row[1:]]
                if not all(math.isfinite(value) and -1 <= value <= 1 for value in values):
                    raise ValueError('controls must be finite and lie within [-1, 1]')
            except ValueError as error:
                raise ValueError(f'{path}, row {step + 1}: {error}') from error
            actions.append(values)
    return np.asarray(actions, dtype=np.float32).reshape(-1, len(Heli.action_names))


def replay_episode(args, *, save=False):
    """Play recorded commands and compare the resulting raw observations when available."""
    import numpy as np
    from heligym.envs.helicopter import Heli

    try:
        actions = read_replay_actions(args.source_dir)
    except (OSError, ValueError, csv.Error) as error:
        raise SystemExit(f'Cannot read replay controls: {error}') from error
    if save:
        return record_episode(None, args, None, replay_actions=actions)

    checked = 0
    first_difference = None
    observations_path = args.source_dir / 'observations.csv'
    observations_file = observations_path.open(newline='') if observations_path.is_file() else nullcontext()
    with observations_file as stream:
        rows = csv.reader(stream) if stream is not None else None
        if rows is not None and next(rows, None) != ['step', *Heli.observation_names]:
            log_status('Recorded observation columns differ from the current layout. Comparison unavailable.')
            rows = None

        def compare(step, _action, observation):
            nonlocal rows, checked, first_difference
            if rows is None or checked > len(actions):
                return
            row = next(rows, None)
            try:
                if row is None or len(row) != len(observation) + 1 or int(row[0]) != step:
                    raise ValueError
                expected = np.asarray(row[1:], dtype=np.float32)
            except ValueError:
                log_status(f'Recorded observation at step {step} is missing or invalid. Comparison stopped.')
                rows = None
                return
            if first_difference is None and not np.array_equal(expected, observation):
                first_difference = step
            checked += 1

        result = run_episode(None, args, None, args.seed, render=True, on_step=compare,
                             replay_actions=actions)

    if checked:
        if first_difference is not None:
            log_status(f'First observation difference at step {first_difference}.')
        elif checked == len(actions) + 1:
            log_status(f'All {checked} recorded observations matched exactly.')
        else:
            log_status(f'First {checked} recorded observations matched exactly.')
    reasons = ', '.join(result['termination_reasons'])
    print(f"Replay episode: steps={result['steps']}, return={result['reward']:.3f}, termination={reasons}")
    return result


def load_recorded_state(directory, timestep):
    """Read one versioned snapshot by step number."""
    from heligym.envs.episode_state import STATE_FORMAT_VERSION

    path = directory / 'states.json'
    data = json.loads(path.read_text())
    if data['version'] != STATE_FORMAT_VERSION:
        raise ValueError(f'{path}: unsupported snapshot format version {data["version"]}')
    states = data['states']
    if not 0 <= timestep < len(states) or states[timestep]['step'] != timestep:
        raise ValueError(f'{path}: no saved timestep {timestep}; available steps are 0–{len(states) - 1}')
    return states[timestep]


def reload_episode(args):
    try:
        snapshot = load_recorded_state(args.source_dir, args.timestep)
        if snapshot['terminated'] or snapshot['truncated']:
            log_status(f'Loading terminal step {args.timestep}. Press R to start a new episode.')
        else:
            log_status(f'Loading step {args.timestep} paused. Press P to resume in manual mode.')
        result = run_episode(None, args, None, args.seed, render=True, initial_state=snapshot)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f'Cannot reload recording: {error}') from error
    reasons = ', '.join(result['termination_reasons'])
    print(f"Reloaded episode: steps={result['steps']}, return={result['reward']:.3f}, termination={reasons}")
    return result


def evaluate(model, args, normalization_path):
    """Save independent deterministic evaluation episodes and summary statistics."""
    import numpy as np

    log_status(f'Evaluating {args.eval_episodes} deterministic episodes without rendering')
    episodes = []
    for index in range(args.eval_episodes):
        result = run_episode(None if args.manual else model, args, normalization_path,
                             args.evaluation_seed + index)
        episodes.append(result)
        print(f"Evaluation {index + 1}/{args.eval_episodes}: "
              f"steps={result['steps']}, return={result['reward']:.3f}")
    rewards = [episode['reward'] for episode in episodes]
    statistics = dict(episodes=episodes, mean_reward=float(np.mean(rewards)),
                      std_reward=float(np.std(rewards)),
                      mean_steps=float(np.mean([e['steps'] for e in episodes])),
                      settings={key: getattr(args, key) for key in (
                          'task', 'algorithm', 'episode_seconds', 'no_wind', 'random_wind',
                          'wind_speed', 'wind_direction', 'turbulence_level')})
    write_json(args.target_dir / 'evaluation.json', statistics)
    print(f"Evaluation: mean return={statistics['mean_reward']:.3f}, "
          f"std={statistics['std_reward']:.3f}")
    return statistics


def plot_title(heading, settings):
    """Add the episode settings available for a saved plot."""
    details = []
    if settings.get('task'):
        details.append(settings['task'])
    if settings.get('heli_name'):
        details.append(settings['heli_name'])
    if settings.get('control_mode') == 'replay':
        details.append('Recorded controls')
    elif settings.get('algorithm'):
        details.append(settings['algorithm'].upper())
    if settings.get('episode_seconds') is not None:
        details.append(f"episode limit {settings['episode_seconds']:g} s")

    weather = []
    if settings.get('no_wind'):
        weather.append('Wind off')
    elif 'no_wind' in settings or any(settings.get(key) is not None for key in (
            'random_wind', 'wind_speed', 'wind_direction', 'turbulence_level')):
        speed = settings.get('wind_speed')
        if settings.get('random_wind'):
            weather.append(f"Random wind up to {20 if speed is None else speed:g} ft/s")
        else:
            weather.append('Wind on' if speed is None else f"Wind {speed:g} ft/s")
        if settings.get('wind_direction') is not None:
            weather.append(f"{settings['wind_direction']:g}° from north")
        if settings.get('turbulence_level') is not None:
            weather.append(f"turbulence {settings['turbulence_level']}")
    return '\n'.join([heading, *filter(None, (' · '.join(details), ' · '.join(weather)))])


def plot_recording(directory, timestep, action_names, observation_names, settings):
    """Plot the saved control and observation series on a shared time axis."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    columns = len(action_names)
    count = columns + len(observation_names)
    rows = math.ceil(count / columns)
    figure = Figure(figsize=(4 * columns, 2 * rows), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, sharex=True).ravel()
    offset = 0
    for filename, names in (('actions.csv', action_names), ('observations.csv', observation_names)):
        samples = np.loadtxt(directory / filename, delimiter=',', skiprows=1, ndmin=2)
        times = samples[:, 0] * timestep
        for index, name in enumerate(names):
            axis = axes[offset + index]
            axis.plot(times, samples[:, index + 1], linewidth=1)
            axis.set_title(name.replace('_', ' '), fontsize=9)
            axis.grid(alpha=0.25)
        offset += len(names)
    for axis in axes[count:]:
        axis.set_visible(False)
    for axis in axes[max(0, count - columns):count]:
        axis.tick_params(labelbottom=True)
    figure.suptitle(plot_title('Recorded episode', settings))
    figure.supxlabel('Time (s)')
    figure.savefig(directory / 'figure.png', dpi=140)


def record_episode(model, args, normalization_path, seed=None, directory=None, replay_actions=None):
    """Save one seeded evaluation episode with raw observations and applied controls."""
    from heligym.envs.helicopter import DT, Heli

    seed = args.record_seed if seed is None else seed
    directory = args.target_dir / 'record' if directory is None else directory
    args.target_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.target_dir, prefix='.record-') as temporary:
        staging = Path(temporary)
        with ExitStack() as files:
            actions_file = files.enter_context((staging / 'actions.csv').open('w', newline=''))
            observations_file = files.enter_context((staging / 'observations.csv').open('w', newline=''))
            actions = csv.writer(actions_file)
            observations = csv.writer(observations_file)
            actions.writerow(('step', *Heli.action_names))
            observations.writerow(('step', *Heli.observation_names))

            def write_step(step, action, observation):
                if action is not None:
                    actions.writerow((step, *action))
                observations.writerow((step, *observation))

            on_state = None
            if args.full_state:
                from heligym.envs.episode_state import STATE_FORMAT_VERSION, capture_episode_state

                states = files.enter_context((staging / 'states.json').open('w'))
                states.write(f'{{"version": {STATE_FORMAT_VERSION}, "states": [\n')

                def on_state(env, total_reward, terminated, truncated, info):
                    if env.step_count:
                        states.write(',\n')
                    snapshot = {
                        'step': env.step_count, 'total_reward': total_reward,
                        'terminated': bool(terminated), 'truncated': bool(truncated),
                        'termination_reasons': episode_end_reasons(
                            info, terminated, truncated) if terminated or truncated else [],
                        'state': capture_episode_state(env),
                    }
                    states.write(json.dumps(snapshot, allow_nan=False, separators=(',', ':')))

            summary = run_episode(model, args, normalization_path, seed,
                                  on_step=write_step, on_state=on_state, replay_actions=replay_actions)
            if args.full_state:
                states.write('\n]}\n')

        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config.update(episode_seed=seed, timestep_seconds=DT, deterministic=True,
                      control_mode='agent' if replay_actions is None else 'replay',
                      model_path=(str((args.policy_dir / 'model.zip').resolve())
                                  if replay_actions is None else None),
                      normalization_path=(None if replay_actions is not None or args.no_online_normalization else
                                          str(normalization_path.resolve())),
                      flip_angle_reference='upright',
                      saved_at=datetime.now(ZoneInfo('Europe/Paris')).isoformat(timespec='seconds'))
        summary.update(success='success' in summary['termination_reasons'],
                       duration_seconds=summary['steps'] * DT)
        write_json(staging / 'config.json', config)
        write_json(staging / 'summary.json', summary)
        filenames = ['actions.csv', 'observations.csv', 'config.json']
        if args.full_state:
            filenames.append('states.json')
        if args.record_plot:
            plot_recording(staging, DT, Heli.action_names, Heli.observation_names, config)
            filenames.append('figure.png')
        directory.mkdir(parents=True, exist_ok=True)
        if not args.record_plot:
            (directory / 'figure.png').unlink(missing_ok=True)
        if not args.full_state:
            (directory / 'states.json').unlink(missing_ok=True)
        for name in (*filenames, 'summary.json'):
            (staging / name).replace(directory / name)
    log_status(f"Recorded {summary['steps']} steps to {directory.resolve()}")
    return summary


def write_json(path, value):
    """Write run artifacts in a format later commands can read."""
    temporary_path = path.with_name(path.name + '.tmp')
    temporary_path.write_text(json.dumps(value, indent=2) + '\n')
    temporary_path.replace(path)


def train(algorithm, args):
    """Train and persist the model, normalization, configuration and episode data."""
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.running_mean_std import RunningMeanStd
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
                write_json(checkpoint_dir / 'arguments.json', {**config, 'saved_at': saved_at()})
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

    def saved_at():
        return datetime.now(ZoneInfo('Europe/Paris')).isoformat(timespec='seconds')

    settings = agent_settings(args)
    args.target_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config['flip_angle_reference'] = 'upright'
    config['saved_at'] = saved_at()
    factory = partial(make_env, args.task, args.heli_name, args.episode_seconds,
                      no_wind=args.no_wind, wind_speed=args.wind_speed,
                      wind_direction=args.wind_direction, random_wind=args.random_wind, turbulence_level=args.turbulence_level, flip_angle=args.flip_angle,
                      smoothness_weight=args.smoothness_weight, continual=args.continual,
                      tolerances={key: getattr(args, key) for key in TOLERANCE_OPTIONS
                                  if getattr(args, key) is not None})
    backend = 'subprocess' if args.n_envs > 1 and args.vec_env == 'subproc' else 'sequential'
    with progress(f'Creating {args.n_envs} training environments ({backend})'):
        if args.n_envs > 1 and args.vec_env == 'subproc':
            env = SubprocVecEnv([factory] * args.n_envs, start_method='spawn')
        else:
            env = DummyVecEnv([factory] * args.n_envs)
    training_callback = TrainingCallback()
    try:
        source = None
        if args.pretrained_dir is not None:
            with progress(f'Loading pretrained policy from {args.pretrained_dir}'):
                source = algorithm.load(str(args.pretrained_dir / 'model.zip'), device='cpu')
            if (source.observation_space != env.observation_space
                    or source.action_space != env.action_space):
                raise ValueError('Pretrained observation or action space differs from the current environment')
        if not args.no_online_normalization:
            source_stats = args.pretrained_dir / 'vecnormalize.pkl' if source is not None else None
            if source_stats is not None and source_stats.is_file():
                env = VecNormalize.load(str(source_stats), env)
                env.training = True
                env.norm_reward = True
                env.gamma = settings['gamma']
                env.ret_rms = RunningMeanStd(shape=())
            else:
                env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=settings['gamma'])
        env.seed(args.seed)
        with progress(f'Initializing {args.algorithm.upper()} policy (device: {args.device})', multiline=True):
            model = algorithm('MlpPolicy', env, seed=args.seed, device=args.device,
                              verbose=args.verbose,
                              tensorboard_log=str(args.target_dir / 'tensorboard_log') if args.tensorboard else None,
                              **settings)
            if source is not None:
                try:
                    model.policy.load_state_dict(source.policy.state_dict())
                except RuntimeError as error:
                    raise ValueError('Pretrained weights do not match the requested policy architecture') from error
                del source
        write_json(args.target_dir / 'arguments.json', config)
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
            config['saved_at'] = saved_at()
            write_json(args.target_dir / 'arguments.json', config)
            print(f'Training artifacts saved to {args.target_dir.resolve()}')
        return model
    finally:
        env.close()


def load_model(algorithm, args):
    """Load a policy for commands that run without training first."""
    model_path = args.policy_dir / 'model.zip'
    if not model_path.is_file():
        raise FileNotFoundError(f'No saved policy found at {model_path.resolve()}; '
                                'run train first or select a saved policy directory')
    normalization_path = args.policy_dir / 'vecnormalize.pkl'
    if not args.no_online_normalization and not normalization_path.is_file():
        raise FileNotFoundError(f'No normalization statistics found at {normalization_path.resolve()}')
    with progress(f'Loading policy from {model_path.resolve()}'):
        model = algorithm.load(str(model_path), device=args.device)
    return model


def print_run_summary(args):
    """Show the settings most useful for the selected commands."""
    location = (f'  Source       {args.source_dir}' if {'replay', 'reload'}.intersection(args.commands)
                else f'  Output       {args.target_dir}')
    lines = ['Run settings', location]
    if args.policy_dir != args.target_dir:
        lines.append(f'  Policy       {args.policy_dir}')
    if any(command != 'plot' for command in args.commands):
        wind = 'off' if args.no_wind else 'randomized' if args.random_wind else 'on'
        lines.extend([
            f'  Environment  {args.task} · {args.heli_name} · {args.episode_seconds:g} s per episode',
            f'  Wind         {wind}',
            f'  Seed         {args.seed}',
        ])
    if 'train' in args.commands:
        if args.pretrained_dir is not None:
            lines.append(f'  Pretrained   {args.pretrained_dir}')
        lines.extend([
            f'  Training     {args.algorithm.upper()} · {args.total_timesteps:,} transitions'
            f' · {args.n_envs} envs · device {args.device}',
            f'  Discount     {args.algo_kwargs.get("gamma", args.gamma)}',
        ])
    if 'eval' in args.commands:
        lines.append(f'  Evaluation   {args.eval_episodes} episodes')
    if 'record' in args.commands:
        if 'replay' in args.commands:
            lines.append(f'  Output       {args.target_dir / "record"}')
        plot = 'with plots' if args.record_plot else 'without plots'
        lines.append(f'  Recording    {args.num_records or 1} episode(s) · seed {args.record_seed} · {plot}')
    print('\n'.join(lines), flush=True)


def main(argv=None):
    """Execute selected commands in pipeline order, sharing a model when possible."""
    original_command = sys.orig_argv if argv is None else None
    argv = list(sys.argv[1:] if argv is None else argv)
    if '-h' in argv or '--help' in argv:
        parse_args(argv)  # Keep --help fast and free of startup messages.
        return
    args = parse_args(argv, help_if_no_command=True)
    if not args.commands:
        return
    if original_command:
        args.command_line = shlex.join(original_command)
    selection = 'replay record' if args.commands == ['record', 'replay'] else ' -> '.join(args.commands)
    log_status('Selected commands: ' + selection)
    if 'tune' in args.commands:
        from ppo_tuning import tune
        tune(args)
        if args.summary_only or args.commands == ['tune']:
            return
        if args.use_tuned or args.tune_dir is not None or args.source_dir is not None:
            args = parse_args(argv, after_tune=True)
            if original_command:
                args.command_line = shlex.join(original_command)
        args.commands.remove('tune')
    if args.commands == ['record', 'replay']:
        print_run_summary(args)
        replay_episode(args, save=True)
        return
    # Configure native paths before importing any heavy packages. The bootstrap
    # may restart Python once; headless commands do not need it.
    if {'render', 'replay', 'reload'}.intersection(args.commands):
        with progress('Preparing renderer library paths', multiline=True):
            prepare_renderer()

    if any(command in args.commands for command in ('train', 'eval', 'record')) or ('render' in args.commands and not args.manual):
        with progress('Importing PyTorch'):
            import torch
        with progress('Importing Stable Baselines3'):
            from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
        torch.set_num_threads(args.torch_threads)
        algorithm = {'ppo': PPO, 'a2c': A2C, 'sac': SAC,
                     'td3': TD3, 'ddpg': DDPG}[args.algorithm]

    print_run_summary(args)
    model = None
    normalization_path = args.policy_dir / 'vecnormalize.pkl'
    for command in args.commands:
        with progress(f'Running {command} command', multiline=True):
            if command == 'train':
                model = train(algorithm, args)
            elif command in ('eval', 'record', 'render'):
                if model is None and (command in ('eval', 'record') or not args.manual):
                    model = load_model(algorithm, args)
                if command == 'eval':
                    evaluate(model, args, normalization_path)
                elif command == 'record':
                    if args.num_records is None:
                        record_episode(model, args, normalization_path)
                    else:
                        for index in range(args.num_records):
                            directory = args.target_dir / 'records' / f'{index + 1:06d}'
                            record_episode(model, args, normalization_path,
                                           seed=args.record_seed + index, directory=directory)
                else:
                    result = run_episode(None if args.manual else model, args, normalization_path,
                                         args.evaluation_seed, render=True)
                    reasons = "; ".join(result.get("termination_reasons", ["unspecified"]))
                    print(f"Rendered episode: steps={result['steps']}, return={result['reward']:.3f}, termination={reasons}")
            elif command == 'replay':
                replay_episode(args)
            elif command == 'reload':
                reload_episode(args)
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
    arguments_path = target_dir / 'arguments.json'
    settings = json.loads(arguments_path.read_text()) if arguments_path.is_file() else {}
    if training_path.is_file():
        plot_learning_curves(json.loads(training_path.read_text()), target_dir, settings)
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
        fig.suptitle(plot_title('Evaluation statistics', statistics.get('settings', settings)))
        fig.tight_layout()
        fig.savefig(target_dir / 'evaluation.png')
        log_status('Displaying evaluation plots; close the plot window to continue.')
        plt.show()
        plt.close(fig)


def plot_learning_curves(episodes, target_dir, settings=None):
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
    fig.suptitle(plot_title('Learning curves', settings or {}))
    fig.tight_layout()
    fig.savefig(target_dir / 'learning_curves.png')
    log_status('Displaying learning curves; close the plot window to continue.')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
