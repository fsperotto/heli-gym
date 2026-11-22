# heli-gym 🚁
HeliGym is a Gymnasium environment for six-degree-of-freedom (6-DOF) helicopter
simulation and reinforcement learning.
It uses a medium-complexity flight dynamics model, configurable aircraft and weather, and an optional OpenGL GUI.

The original implementation has been developed by Uğurcan Özalp (https://github.com/ugurcanozalp/heli-gym) based on the book "Minium-Complexity Helicopter Simulation Math Model" (Heffley and Mnich, 1988) (https://robertheffley.com/docs/Sim_modeling/Heffley-Mnich--Minimum-Complexity%20Helicopter%20Simulation%20Math%20Model--NASA%20CR%20177476.pdf).

For a fresh Linux clone, install the native build dependencies first (Ubuntu/Debian):

```bash
sudo apt install build-essential cmake libglfw3-dev libassimp-dev libgl1-mesa-dev
bash ./setup_install.sh
source ./init.sh
python test.py
```

`test.py` opens an OpenGL window, so it also requires a graphical session or
working display forwarding.

## Training and saved runs

Install the optional training dependencies with `pip install -e '.[train]'`.
`main.py` accepts the pipeline commands `tune`, `train`, `eval`, `record`, `plot`, and `render`, plus standalone `replay` and `reload`.
With no command it prints help. Selected commands always run in this order:
`tune → train → eval → record → plot → render`.

```bash
python main.py train eval record plot render --target-dir runs/hover
python main.py train eval --target-dir runs/hover      # headless training and evaluation
python main.py eval plot --target-dir runs/hover       # evaluate a saved policy and plot statistics
python main.py plot --target-dir runs/hover            # plot saved statistics only
python main.py render --target-dir runs/hover          # one episode in the GUI
```

### Tune PPO with Optuna

Install Optuna in the same Python environment as HeliGym. Then tune one task and
use the refinement winner to train a fresh policy.

```bash
python -m pip install optuna
python main.py tune --task stabilize --no-wind --target-dir runs/stabilize
python main.py train eval --use-tuned --target-dir runs/stabilize \
  --seed 123 --total-timesteps 2000000
```

The tuner searches learning rate, discount factor (`gamma`), GAE lambda,
clip range, and rollout length. It scores configurations by mean deterministic
evaluation return.

| Phase | Defaults |
| --- | --- |
| Search | 20 additional trials at 50,000 training transitions each |
| Refinement | Top 3 search candidates plus the default PPO baseline, each trained from scratch on 3 fresh seeds at 300,000 transitions per run |
| Evaluation | 10 episodes per trained policy, with common evaluation seeds within each phase and held-out seeds for refinement |

A fresh default study schedules 32 training runs and about 4.6 million
transitions before final training. PPO can round each budget up to fill a
rollout. For a smaller study, adjust `--n-trials`, `--trial-timesteps`,
`--finalists`, `--refinement-seeds`, `--refinement-timesteps`, and
`--eval-episodes`.

The study lives under `<target-dir>/optuna/`. Inspect `summary.json` for rankings
and scores. Reuse the same settings and target directory to add trials.
Pass `--n-trials 0` to continue refinement without adding search trials.
Completed refinement runs are reused. For a study started with the previous
983,040-transition refinement default, pass `--refinement-timesteps 983040`
when resuming.

```bash
python main.py tune --summary-only --target-dir runs/stabilize
```

Refinement writes the winner's PPO and environment settings to
`optuna/final_params.json`. Use `--use-tuned` to load it from the target
directory, `--tune-dir runs/stabilize/optuna` to load it from another study
directory, or `--file-args PATH` to load a specific JSON file. Explicit CLI
options take precedence over values from the file. The export leaves the seed,
training budget, and output directory to your new training run. You can also
run `tune train --use-tuned` in one invocation.

### Record data of an episode run

```bash
python main.py record --target-dir runs/hover
```

This runs one deterministic, headless episode using the saved policy and the
same seed as the first evaluation episode (`seed + n_envs`) by default. Pass
`--seed` to use that seed directly for recording. It writes the
following files to `<target-dir>/record/`. A later recording replaces those files.

- `config.json` contains the effective configuration, episode seed, and artifact paths.
- `summary.json` contains the step count, total reward, duration, success flag,
  and termination reasons.
- `actions.csv` contains the four applied controls for each step, starting at 1.
- `observations.csv` contains raw environment observations. Step 0 is the reset
  observation, and each later row is the observation after the action on that
  step, including the terminal observation.
- `figure.png` shows controls and observations over time, one subplot per signal.
  The plot is on by default. Use `--no-record-plot` to save just the JSON and CSV files.

To record multiple episodes, pass `--num-records`. This writes the same
files per episode under `<target-dir>/records/000001/`, `000002/`, and so on.
Seeds increment by one per episode. A new batch replaces the numbered folders
it writes.

```bash
python main.py record --target-dir runs/hover --num-records 3 --seed 100
```

### Replay a recording

```bash
python main.py replay --source-dir runs/hover/record
python main.py replay --source-dir runs/hover/records/000001 --wind --wind-speed 12
```

Replay opens the viewer and applies the controls in `actions.csv` in order. It
uses the recorded episode seed and environment settings from `config.json`.
Explicit CLI settings take precedence, including `--seed`, `--task`, and wind
options. Replay does not load a policy. When the observation layout matches,
it reports the first difference from `observations.csv` or confirms the match.

Use **P** to pause or resume. The viewer shows pilot mode **2** during replay.
When the controls run out and the episode is still active, the viewer switches
to manual control and holds the last applied command. If the episode ends
earlier, replay stops at that terminal step.

To compare flight models, record a policy episode and then save a replay under a
different aircraft configuration.

```bash
python main.py record --target-dir runs/my_run --seed 43
python main.py replay record --source-dir runs/my_run/record \
  --heli-name aw109_trekker --target-dir runs/my_run_trekker
```

The second command runs headlessly, with no policy, and writes fresh observations,
actions, and a summary to `runs/my_run_trekker/record/`. It stops when the new
episode ends or the recorded controls run out. Both recordings can be compared
through their CSV files and figures.

### Restore a saved timestep

Record with `--full-state` to save a `states.json` in every episode folder.
It contains snapshots of the flight and wind dynamics, random-generator
states, task targets and progress, previous controls, and episode timers.

```bash
python main.py record --target-dir runs/hover --full-state
python main.py reload --source-dir runs/hover/record --timestep 157
```

A timestep is the saved step number. Step 0 is the reset state, and step 157
is the state after applying control 157. Reload uses the environment settings
in the recording and opens the viewer paused at that state. Press **P** to
resume with manual controls. At an already terminal snapshot, press **R** to
start a new episode. Reload uses no policy and leaves the recorded files
untouched. Use `replay` when you want to change the environment settings and
apply the entire recorded control sequence.

## Episode termination

Across all tasks, a crash or flip terminates the episode with a fixed **−100**
reward, replacing that step's task reward. `info['crash']` and `info['flip']`
identify these failures, and `info['failed']` is true. Failure takes precedence
over success and goto arrival bonuses; no new waypoint is drawn on failure.
A crash requires ground contact combined with excessive downward speed or
absolute roll or pitch above 60°. The airborne attitude check defaults to the
helicopter's configured `MAX_ROLL` and `MAX_PITCH`, applied to the absolute
roll and pitch independently. AW109 limits are 60° roll and 25° pitch.

Use `--flip-angle 120` in the CLI or `flip_angle=120.0` in the environment
constructor to override both limits with a single tilt threshold. Tilt is
measured from earth up to the helicopter's up axis. Upright is 0°, horizontal
is 90°, and inverted is 180°. Failure occurs when tilt exceeds the threshold.
Saved runs retain their setting for evaluation, recording, and rendering. Old saved angles
measured from the horizon are converted with `90 - old_angle`. Gentle upright
ground contact alone is not a crash. Existing map/altitude limits and numerical
failure handling remain active; the episode time limit remains a truncation.

## Flight tasks

### Goto waypoints

Use `--task goto` to reach a random position target. Add `--continual` to
pursue successive waypoints in the same episode.
Its sampled waypoints leave speed and heading unspecified, so both differences
are zero.
Its reward is `progress_weight * (previous_distance - current_distance)` plus
`arrival_reward` when entering the target radius. Distances are straight-line 3D
distances in feet to the same active target. Defaults are `progress_weight=0.1`
and `arrival_reward=100`: closing 100 ft yields 10 progress points, while reaching
the target yields 100 bonus points. Moving at constant distance yields zero
progress reward; moving away yields a negative reward. There are no separate
velocity, angular-motion, attitude, or action penalties. Arrival uses the same horizontal and altitude tolerances as the other tasks.
The reward is calculated for the reached target. By default arrival ends the episode
successfully. With `--continual`, the observation instead reports differences
from a newly sampled waypoint and the episode continues. Targets are sampled within
±100 ft north/east and ±50 ft altitude of the current position; altitude is raised
where necessary to maintain 50 ft clearance above terrain at the target.
Targets remain within map boundaries and outside the arrival radius. The usual
failure rules and episode time limit still apply. `info` includes `targets_reached`
and the active `target_position` (north, east, altitude).

`HeliGoto` accepts `target_horizontal_range`, `target_altitude_range`,
and `target_ground_clearance` constructor settings in feet,
plus `progress_weight` (nonnegative) and `arrival_reward` (positive). An explicit
`target_radius` in Python selects the legacy spherical arrival check.
Sampling uses the environment random generator, so `reset(seed=...)` reproduces
its target sequence. Rendering shows task target values and a translucent cyan 3D marker with dark edges for
explicit position targets, including hover and goto.

```bash
python main.py train --task goto --target-dir runs/goto
```

### Smooth controls

The CLI offers `smooth-heli`, `smooth-hover`, `smooth-stabilize`,
`smooth-forward-flight` (alias `smooth-forward`), and `smooth-goto`.
`HeliSmoothApproximateTargetValues-v0` exposes the configurable tracking task
with the same penalties through Gymnasium.

Each smooth task subtracts `smoothness_weight` times the sum of:

- Mean squared control changes and magnitudes over the four clipped controls.
- Squared excess of `abs(recent_yaw_turns)` above 1.
- Squared relative excess of absolute roll and pitch above the aircraft's
  `SAFE_ROLL` and `SAFE_PITCH`, checked separately.

Yaw and attitude costs are zero up to their thresholds. The usual flip limits
still end the episode. Smooth Stabilize uses yaw history for its reward and
keeps the instantaneous yaw-rate check for success. Set the shared weight
with `--smoothness-weight` (default 0.25).

```bash
python main.py train --task smooth-hover --smoothness-weight 0.1 --target-dir runs/smooth-hover
```

### Target tracking rewards

Use `ApproximateTargetValues-v0` to track any subset of targets.

```python
import gymnasium as gym
import heligym

env = gym.make(
    "ApproximateTargetValues-v0",
    target={"sea_alt": 2000, "forward_speed": 100},
    mode="difference",
    diff_impact="linear",
    render_mode=None,
)
``` Velocities are signed ground-relative body components
in ft/s, altitude is positive up, and yaw rotation velocity is body rate r in rad/s.
Each active error is current minus target divided by its characteristic scale.
Position uses rotor diameter D, speed uses sqrt(Dg), angular rate uses 1/T
where T = sqrt(D/g), and heading uses radians. Unspecified targets contribute zero.

For error e, the cost is abs(e) with `diff_impact="linear"` (default) or e² with
`"quadratic"`. Rewards sum over target components. Mode `"difference"` gives
negative cost, `"progression"` (default) gives previous minus current cost for
the same target, and `"attitude"` gives the negative instantaneous cost derivative
per normalized time, taking the linear gradient as zero at zero error.

Stabilize targets zero body velocity (`u`, `v`, `w`) and zero body yaw rate
(`r`). It charges the normalized yaw-rate error on every step, including steady
spins in progression or attitude mode. Smooth Stabilize uses the thresholded yaw-history penalty described above.

Hover captures the initial position on each reset and accepts an explicit `target_altitude`. Forward flight uses fixed-target
Goto with altitude and forward speed. Goto without a constructor target retains
its random waypoint and arrival bonus. Pass `continual=True` for successive waypoints.

### Success tolerances

| CLI option | Default | Error checked |
| --- | --- | --- |
| `--horizontal-tolerance` | 20 ft | Horizontal distance across the active north and east targets |
| `--altitude-tolerance` | 10 ft | Absolute altitude error |
| `--velocity-tolerance` | 5 ft/s | Absolute error per active body velocity component |
| `--heading-tolerance` | 5° | Shortest angular distance to the heading target |
| `--yaw-turn-seconds` | Aircraft-derived yaw-rate scale | Absolute yaw-rate error. Specify seconds per revolution to set the limit to `360° / seconds`. |

Only active targets count toward success. The existing reward scales apply
independently of these thresholds. Goto uses both position tolerances for its
arrival bonus and still calculates progress using 3D distance.

## Training settings and artifacts

The default training algorithm is PPO. In a regular training run, `--n-envs`
defaults to the smaller of 4 and the logical CPU count. You can specify a higher
value. The default `subproc` backend uses a separate worker per environment when
running multiple environments. For PPO, `--n-steps` (default 1024) counts steps
per worker, and `--batch-size` (default 256) must divide the total rollout size
`n_envs × n_steps`. Set the discount with `--gamma` (default 0.99).

### Fine-tune a saved policy

Use `--pretrained-dir` with a run or checkpoint directory containing
`model.zip` and `arguments.json`. The source supplies the starting policy
weights, observation normalization statistics (if enabled), and defaults for
the new run. Explicit CLI options take precedence over settings from
`--file-args` or `--use-tuned`, which in turn take precedence over the source
run's settings.

```bash
python main.py train eval --pretrained-dir pretrained/goto_no_wind \
  --task smooth-goto --continual --smoothness-weight 0.03 \
  --target-dir runs/smooth-goto-finetune --total-timesteps 500000 --device cpu
```

The source and destination directories must differ. Use a source with the same
algorithm, policy architecture, and observation/action spaces. Training starts
new optimizer, reward-normalization, and timestep state.

Use `--tensorboard` to enable TensorBoard training logs under
`<target-dir>/tensorboard_log`. No separate log directory argument is needed.
`--no-tensorboard` disables logging, which is the default. For example:

```bash
python main.py train --task goto --target-dir runs/goto --tensorboard
tensorboard --logdir runs/goto/tensorboard_log
```

These switches also pass through the Bash launchers. They replace the previous
`--tensorboard-log PATH` option. Disabling logging leaves existing log files intact.

Use `--n-checkpoints N` to save up to N intermediate checkpoints during training
(default: 0, disabled). The count excludes the final model. For example,
`--total-timesteps 1000000 --n-checkpoints 4` saves checkpoints near 200k, 400k,
600k, and 800k transitions, then saves the final model in the target directory.
Triggers count transitions across all environments and fire on the first vector
step at or beyond each threshold; a vector step reaching the requested training
end is reserved for the final save. Very short runs cap the count to the number
of available vector steps before that end.

Checkpoints are stored under `checkpoints/<run timestamp>/step_<actual timesteps>/`,
with `model.zip`, matching `vecnormalize.pkl` when normalization is enabled, and
`arguments.json`. Previous checkpoints survive subsequent saves and fresh training
runs in the same target directory. Evaluate, record, or render any checkpoint by passing its
folder as `--target-dir`. At each checkpoint, the target directory's
`learning_curves.json` is atomically replaced with the complete episode history
so far; only this latest cumulative history is retained. Final saving updates it
again. Checkpoint timing is independent of SB3's printed training statistics.
`--n-checkpoints` replaces the previously unused `--checkpoint-interval` option.

`train` saves the policy, normalization statistics (when enabled), run arguments,
and episode data in `learning_curves.json`. The final `arguments.json` and each
checkpoint's copy include `command_line` and a timezone-aware ISO 8601
`saved_at` timestamp. `eval` runs ten deterministic episodes
without rendering and writes episode returns, lengths, and summary statistics to
`evaluation.json`; use `--eval-episodes` to change the count. `plot` reads these
JSON files, saves `learning_curves.png` and/or `evaluation.png`, and displays the
figures. On a server, set `MPLBACKEND=Agg` to save figures without opening windows.
## Render and fly manually

`render` starts one agent-piloted episode at simulator speed. Press **M** to switch
between agent and manual keyboard control. In agent mode, movement/reset keys are
ignored; in manual mode, the keyboard controls the helicopter. Returning to the
agent uses the latest observation. Press **P** to pause/resume in either mode;
the window stays responsive while physics is paused. The GUI indicates the pilot
mode. The episode ends at the configured time limit or a terminal condition.

Use `--manual` to launch a selected task without any policy, saved normalization,
or training dependencies. Use it with the `render` command. With a saved run directory, it reuses the environment settings in
`arguments.json`. Explicit CLI options override them. **M** has no effect because no agent is loaded.

```bash
python main.py render --manual --task goto --continual --episode-seconds 1000
python main.py render --target-dir runs/goto   # agent initially; M toggles manual
```

Manual controls are **Z/S** (collective), **Up/Down** (longitudinal cyclic),
**Right/Left** (lateral cyclic), and **D/Q** (pedals), each changing the command
by 0.2 per key press. **X** zeros the controls; **R** restarts the episode.
**Esc** closes the window. **V** cycles through fixed, chase, and cockpit views.
The cockpit camera follows the helicopter's full attitude and uses the supplied
AW109 cockpit artwork. Press **H** to hide the information panels for an
unobstructed cockpit view.
**W** toggles wind in either mode, preserving the configured mean speed/direction
and turbulence settings for reactivation; it does not reset or re-trim the aircraft.
**T** stops and re-trims the helicopter at its current north/east position and sea
altitude, preserving heading and rotor phases. Roll and pitch can adjust to find
stationary equilibrium under the current instantaneous wind. Linear/angular
velocities and cached accelerations are zeroed, and trim controls become the last
applied controls in the refreshed observation. Rotor rotation continues normally.
No episode clock, target, pause state, or pilot mode is reset. The next physics
step recomputes accelerations; turbulence or an active agent can move the aircraft
again. If stationary trim cannot converge or requires controls beyond their
limits, the prior state is restored and an error is printed. This key works in
both pilot modes and while paused.
**H** hides/shows all info panels and control bars (shown initially). Both toggles
also work while paused. The Observations panel shows the 17 dynamics values
and recent signed yaw turns. Target differences shows all eight target channels,
with heading displayed in degrees. Previous applied controls shows the four
action-history observations. The control bars show the current command.
Agent mode hides manual flight/reset instructions while retaining shared controls.
The rendered episode summary includes termination reasons, for example:
`Rendered episode: steps=659, return=-105.178, termination=flip`.
It reports crash, flip, success, max time (truncated), simulation limits,
numerical errors with details, or a user-closed window. Simultaneous conditions
are listed together. Evaluation episode JSON records also include
`termination_reasons`.

The GUI’s **Simulation** panel shows FPS, the current episode step, and the latest
unnormalized step reward. Step and reward reset to zero at the start of each episode.
The **Wind** panel shows whether wind is enabled, mean speed and direction,
turbulence level, and the current wind speed and north/east/down components
(including turbulence) in ft/s. Direction is clockwise from north. Disabled wind
shows zero values. These display values are not added to the agent observations.

## Wind and saved settings

Independent `eval`, `record`, and `render` commands reuse the run settings in
`arguments.json`, including the task, aircraft, and weather. Explicit CLI options
take precedence.
For example, `python main.py render --target-dir runs/hover --wind` enables wind
even if the run was trained with `--no-wind`. These two flags are mutually exclusive.
Set `--wind-speed` (ft/s), `--wind-direction` (degrees clockwise from north),
and `--turbulence-level` (1–7) to override weather settings, for example:

```bash
python main.py render --target-dir runs/hover --wind --wind-speed 12 --wind-direction 90 --turbulence-level 3
```

Use `--random-wind` to sample a reproducible speed between zero and the
`--wind-speed` upper bound (20 ft/s by default) and a direction. Training, evaluation, and recording all sample from the episode seed.
An explicit `--wind-direction` fixes the direction. Turbulence intensity stays
at the configured level. Use `--no-random-wind` for fixed weather or `--no-wind`
for calm conditions. In the GUI, **W** toggles wind using the configured weather.

### Reproduce an episode

Use the same environment settings, episode seed, and applied controls to
reproduce the raw trajectory. The seed fixes the initial waypoint sequence,
randomized mean wind, and turbulence. Training starts workers at
`seed + worker_index`; automatically reset episodes expose their generated
seed in `info["episode_seed"]`. A standalone `eval`, `render`, or `record`
uses an explicit `--seed` directly. Without one, it starts at the saved
training seed plus the worker count. A combined `train eval` keeps that
offset for evaluation.

```bash
python main.py record --target-dir runs/hover --seed 43
```

To reproduce actions from a policy, use the same checkpoint, normalization
statistics, and deterministic inference. Training may sample exploratory
actions and update both its policy and normalization during the episode.

For older runs without saved settings, supply matching options yourself. Plotting
requires the new saved episode data; an existing PNG alone cannot recreate it.
The positional commands replace `--no-learn`, `--no-render`, and `--no-stats-plot`.
Use `python main.py --help` for the remaining options.

Startup and command execution print progress immediately, with elapsed times for
PyTorch/SB3 imports, model loading, environment creation, training and saving.
These messages also appear promptly when stdout is redirected to a log file.
GUI library-path setup runs before simulator/ML imports to avoid repeating those
imports if Python needs to restart. `--help` and plot-only commands continue to
avoid loading the training dependencies.

On WSL, keep the checkout **and its virtual environment** on the native Linux
filesystem for best file-access performance. A path under `/home` that links to
`/mnt/c` or `/mnt/d` still uses the Windows-backed filesystem; `pwd -P` and
`df -T . .venv` show the actual location and filesystem. See
[Microsoft's WSL filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/interop).

-------------------------------------
## __Environment Details__

* Pure Gym environment
* Realistic Dynamic Model based on Minimum Complexity Helicopter Model ([Heffley and Mnich](https://robertheffley.com/docs/Sim_modeling/Heffley-Mnich--Minimum-Complexity%20Helicopter%20Simulation%20Math%20Model--NASA%20CR%20177476.pdf))
In addition, inflow dynamics are added and model is adjusted so that it covers multiple flight conditions. 
* Rendering is done by OpenGL.

Aircraft, terrain, and weather defaults are stored independently under
`heligym/envs/config/helis`, `heligym/envs/config/terrains`, and
`heligym/envs/config/weather`.
`Heli` accepts either bundled configuration names or YAML paths for the
aircraft and terrain. Weather and atmospheric values are constructor arguments:

```python
env = heligym.Heli(
    heli_name="aw109",
    terrain_name="default_terrain",
    wind_direction=45.0,       # degrees
    wind_speed=20.0,           # ft/s
    turbulence_level=1,
    gravity=32.2,              # ft/s²
    render_mode=None,
)
```

The other physical arguments are `gas_constant`, `sea_level_temperature`,
`temperature_lapse_rate`, `troposphere_height`, and
`sea_level_air_density`. Their defaults are defined in
`default_weather.yaml`.

![Caption](resources/ex.gif)

-------------------------------------
### __Action Space__
| Num | Act                | Unit  | Min   | Max    |
|-----|--------------------|-------|-------|--------|
| 0   | Collective         |       | -1    | 1      |
| 1   | Lon. Cyclic        |       | -1    | 1      |
| 2   | Lat. Cyclic        |       | -1    | 1      |
| 3   | Pedal              |       | -1    | 1      |

-------------------------------------
### Observation space

Each environment observation contains 30 values, in this order:

- 17 measurements produced by the helicopter dynamics.
- 8 target differences for local north position, east position, sea altitude
  (feet, positive up), body-forward speed, heading, body-down velocity, body-right velocity, and body yaw rate.
- 1 signed recent yaw rotation value.
- 4 previous applied control commands.

Target differences are current measurement minus target. The heading difference
is wrapped to `[-π, π)`. Each unspecified or `None` target contributes zero.
Pass a mapping to `set_target` with keys `north_loc`, `east_loc`, `sea_alt`,
`forward_speed`, `heading`, `vertical_velocity`, `lateral_velocity`, and
`yaw_rotation_velocity`, or pass an eight-item tuple in that order. Hover and
forward-flight use their respective configured targets. Policies and
normalization statistics from earlier observation layouts need retraining.

The `recent_yaw_turns` observation starts at zero and updates once per
physics step from the signed body yaw rate `r`. It uses an exponential
decay with a 10-second time constant. Starting from reset, a steady
360° turn over 10 seconds gives `+1` or `-1` depending on direction.
Continuing to turn can take it past that magnitude, and once turning stops
it decays toward zero. Opposite-direction turns cancel part of the
accumulated value. Smooth tasks use this observation to penalize sustained rotation beyond its threshold.

| Num | Obs                | Unit  | Min   | Max    |
|-----|--------------------|-------|-------|--------|
| 0   | power              | hp    | 0     | ∞      |
| 1   | lon. air speed     | ft/s  | -∞    | ∞      |  
| 2   | lat. air speed     | ft/s  | -∞    | ∞      |
| 3   | down air speed     | ft/s  | -∞    | ∞      |
| 4   | north velocity     | ft/s  | -∞    | ∞      |  
| 5   | east velocity      | ft/s  | -∞    | ∞      |
| 6   | descend rate       | ft/s  | -∞    | ∞      |  
| 7   | roll angle         | rad   | -π    | π      |
| 8   | pitch angle        | rad   | -π    | π      |
| 9   | yaw angle          | rad   | -π    | π      |
| 10  | roll rate (body)   | rad/s | -∞    | ∞      |  
| 11  | pitch rate (body)  | rad/s | -∞    | ∞      |  
| 12  | yaw rate (body)    | rad/s | -∞    | ∞      |   
| 13  | x loc (earth)      | ft    | -∞    | ∞      |  
| 14  | y loc (earth)      | ft    | -∞    | ∞      |  
| 15  | sea altitude       | ft    | 0     | ∞      |
| 16  | ground altitude    | ft    | 0     | ∞      |
| 17  | north difference   | ft    | -∞    | ∞      |
| 18  | east difference    | ft    | -∞    | ∞      |
| 19  | altitude difference| ft    | -∞    | ∞      |
| 20  | forward speed difference   | ft/s  | -∞    | ∞      |
| 21  | heading difference | rad   | -π    | π (exclusive) |
| 22  | vertical velocity difference | ft/s | -∞ | ∞ |
| 23  | lateral velocity difference | ft/s | -∞ | ∞ |
| 24  | yaw rotation velocity difference | rad/s | -∞ | ∞ |
| 25  | recent yaw rotation | turns (scaled) | -∞ | ∞ |
| 26  | previous collective|       | -1    | 1      |
| 27  | previous lon. cyclic|      | -1    | 1      |
| 28  | previous lat. cyclic|      | -1    | 1      |
| 29  | previous pedal     |       | -1    | 1      |

At reset, the previous controls are the trim controls used to create the
initial steady-flight state. After each step, they are the direct controls
most recently supplied by the agent.
-------------------------------------
### Dynamics state and normalization

The earth frame uses north, east, and down axes. The body frame uses forward,
right, and down axes. State vectors are stored in `heli_dyn.state` and their
derivatives in `heli_dyn.state_dots`.

| State or derivative | Meaning | Units |
| --- | --- | --- |
| `xyz` | Earth-frame position | ft |
| `state_dots["xyz"]` | Ground-relative velocity in earth axes | ft/s |
| `uvw` | The same ground-relative velocity in body axes | ft/s |
| `state_dots["uvw"]` | Rate of change of the body-frame velocity components, including the effect of rotating axes | ft/s² |
| `euler` | Roll, pitch, and yaw angles | rad |
| `pqr` | Angular velocity about the body axes | rad/s |
| `state_dots["pqr"]` | Angular acceleration about the body axes | rad/s² |

When normalized state is enabled, `xyz_norm` divides position by the main
rotor diameter `D`. `xyzdot_norm` and `uvw_norm` divide velocity by
`sqrt(Dg)`, and `uvwdot_norm` divides by gravitational acceleration `g`.
`pqr_norm` and `pqrdot_norm` multiply by `T` and `T²`, respectively,
where `T = sqrt(D/g)`. The agent observation fields follow the layout above.

## Registered environments

| Gymnasium ID | CLI task | Objective |
| --- | --- | --- |
| `Heli-v0` | `heli` | Base helicopter dynamics |
| `ApproximateTargetValues-v0` | | Track configured position, speed, velocity, heading, or yaw-rate targets |
| `HeliHover-v0` | `hover` | Hold the reset position |
| `HeliForwardFlight-v0` | `forward-flight` or `forward` | Track forward speed and sea altitude |
| `HeliStabilize-v0` | `stabilize` | Bring body-frame linear velocity and yaw rate to zero |
| `HeliSmoothHover-v0` | `smooth-hover` | Hover with a control smoothness penalty |
| `HeliSmoothStabilize-v0` | `smooth-stabilize` | Stabilize with a control smoothness penalty |
| `HeliSmooth-v0` | `smooth-heli` | Base dynamics with smooth-flight penalties |
| `HeliSmoothForwardFlight-v0` | `smooth-forward-flight` or `smooth-forward` | Forward flight with smooth-flight penalties |
| `HeliSmoothGoto-v0` | `smooth-goto` | Goto with smooth-flight penalties |
| `HeliSmoothApproximateTargetValues-v0` | | Configurable target tracking with smooth-flight penalties |
| `HeliGoto-v0` | `goto` | Reach a random waypoint or follow successive waypoints with `--continual` |

The simulator advances at a fixed 50 Hz (0.02 seconds per step). For native
graphics setup and troubleshooting, see the [renderer guide](heligym/envs/renderer/README.md).

## __Project Structure__

The repository is organised in layers: the Gymnasium-facing environments coordinate a reusable Python dynamics model, while rendering is isolated behind a Python API backed by a native C++/OpenGL library. Configuration and visual assets live next to the subsystem that consumes them, and tests mirror the behaviours they verify.

```text
heli-gym/
├── heligym/
│   ├── __init__.py                    # Public imports and Gymnasium environment registration
│   └── envs/
│       ├── __init__.py                # Environment package exports
│       ├── helicopter.py              # Base Heli environment and simulation/render loop
│       ├── helicopter_with_tasks.py   # Task-specific environments and reward functions
│       ├── config/                    # Aircraft, terrain, and weather defaults
│       ├── dynamics/
│       │   ├── __init__.py            # Dynamics package exports
│       │   ├── dynamics.py            # Generic state containers and integration framework
│       │   ├── helicopter_dynamics.py # Six-degree-of-freedom helicopter model
│       │   ├── wind_dynamics.py       # Mean-wind and turbulence model
│       │   ├── kinematic.py           # Rotation and coordinate-transform calculations
│       │   ├── lookup.py              # One- and two-dimensional lookup-table interpolation
│       │   └── utils.py               # Shared mathematical helpers
│       └── renderer/
│           ├── __init__.py            # Native-library and resource path setup
│           ├── api.py                 # High-level Python Renderer interface
│           ├── pyapi.py               # ctypes bindings to the native renderer
│           ├── src/
│           │   ├── camera.cpp/.h      # Camera position, orientation, and view transforms
│           │   ├── gWindow.cpp/.h     # OpenGL window, input, scene, and GUI lifecycle
│           │   ├── mesh.cpp/.h        # Drawable mesh data and texture handling
│           │   ├── model.cpp/.h       # Assimp-based 3D model loading and drawing
│           │   ├── py_api.cpp/.h      # C ABI exported for the Python bindings
│           │   ├── shader.h           # GLSL shader compilation and uniform helpers
│           │   └── glad.c             # Bundled OpenGL function loader implementation
│           ├── resources/
│           │   ├── models/            # Helicopter, sky, and terrain geometry/textures
│           │   └── shaders/           # GLSL vertex and fragment shader programs
│           ├── include/               # Bundled third-party C/C++ headers
│           ├── libs/                  # Native runtime dependencies
│           ├── bin/                   # Built Heligym shared libraries for Linux and Windows
│           ├── cmake/                 # CMake modules locating Assimp, GLFW, GLM, and STB
│           ├── CMakeLists.txt         # Native renderer build definition
│           └── README.md              # Renderer build and troubleshooting guide
├── tests/
│   ├── test_keyboard_controls.py      # Keyboard and paused-loop unit tests
│   ├── test_renderer_lifecycle.py     # Renderer shutdown and lifecycle unit tests
│   └── test_native_keyboard.cpp       # Native keyboard-event integration test
├── resources/ex.gif                   # Animation displayed in this README
├── main.py                            # Tuning, training, evaluation, and recording CLI
├── ppo_tuning.py                      # Optuna search and refinement
├── test.py                            # Interactive simulation/demo entry point
├── pyproject.toml                     # Package metadata, dependencies, and included data
├── setup_install.sh                   # Guided Conda/uv/virtual-environment installer
├── init.sh                            # Convenience script for activating the local environment
├── LICENSE                            # MIT licence
└── README.md                          # Project overview and user documentation
```

This separation keeps reinforcement-learning tasks independent from the equations of motion: new tasks normally belong in `helicopter_with_tasks.py`, new aircraft are described under `heligym/envs/config/helis/`, and changes to flight physics stay in `dynamics/`. Rendering remains a self-contained subsystem, and `api.py` shields environment code from the lower-level `ctypes` and C++ implementation. Generated caches and virtual environments are intentionally omitted from the tree above, while assets and bundled third-party dependencies are grouped rather than listing every generated or upstream file.

-------------------------------------
## __Setup__
Install the package directly with pip:

```bash
pip install -e .
```

Alternatively, run or source the guided installer. When a Conda environment is
active (including `base`), it first asks whether to install into that environment.
If declined, it asks whether to create a Conda environment at the project's
`.venv` path, with Python 3.12 and pip. Conda environments are identified by their
prefix paths; see the [Conda environment guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#specifying-a-location-for-an-environment).
The editable project and training dependencies are installed using the selected
environment's `python -m pip`, so another environment's pip is never used.

With no active Conda environment, or after declining both Conda options, the
existing uv/pip/standard-venv workflow runs. An existing `.venv` is never silently
overwritten. After a successful installation, the installer rewrites `init.sh`
with activation commands for the selected environment, including environments
outside the project. Run `source ./init.sh` to activate it. Sourcing the installer
itself does not change the caller's active environment or working directory.

```bash
source setup_install.sh
```

-------------------------------------
## __Usage__
For a quick start after using the guided installer, activate its virtual environment and run the interactive example:

```bash
source init.sh
python test.py
```

To use heli-gym in your own code, create an environment with either:

```python
from heligym import HeliHover
env = HeliHover()
```

or:

```python
import gymnasium as gym
import heligym
env = gym.make("HeliHover-v0")
```

Use `reset(seed=...)` and `step(action)` to interact with the environment through the Gymnasium API.

## __TODO List__

To do list, ordered according to importance order.
- [x] Fix rotor rotation rendering problem
- [x] Make terrain height map input generic
- [x] Place all helicopter related resources into single folder
- [x] Add turbulence model (probably Dryden Turbulence Model)
- [ ] Convert all dynamics related scripts into C++ code with Python api.

To do list for Reinforcement Learning tasks.
- Read http://avstop.com/ac/basichelicopterhandbook/ch11.html
- [x] Add **hover** task
- [x] Add **forward flight** task
- [ ] Add **turning flight** task
- [ ] Add **ground** task
- [ ] Add **slalom maneuver** task
