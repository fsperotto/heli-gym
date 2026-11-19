# heli-gym 🚁
OpenAI GYM environment for 6-DOF Helicopter simulation.

This is GYM environment package for reinforcement learning for helicopter flight tasks using minimum complexity helicopter model.

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
`main.py` accepts the positional commands `tune`, `train`, `eval`, `plot`, and `render`.
With no command it prints help. Selected commands run in pipeline order.

```bash
python main.py train eval plot render --target-dir runs/hover  # entire pipeline
python main.py train eval --target-dir runs/hover      # headless training and evaluation
python main.py eval plot --target-dir runs/hover       # evaluate a saved policy and plot statistics
python main.py plot --target-dir runs/hover            # plot saved statistics only
python main.py render --target-dir runs/hover          # one episode in the GUI
```

Each environment observation contains 29 values, in this order:

- 17 measurements produced by the helicopter dynamics.
- 8 target differences for local north position, east position, sea altitude
  (feet, positive up), body-forward speed, heading, body-down velocity, body-right velocity, and body yaw rate.
- 4 previous applied control commands.

Target differences are current measurement minus target. The heading difference
is wrapped to `[-π, π)`. Each unspecified or `None` target contributes zero.
Pass a mapping to `set_target` with keys `north_loc`, `east_loc`, `sea_alt`,
`forward_speed`, `heading`, `vertical_velocity`, `lateral_velocity`, and
`yaw_rotation_velocity`, or pass an eight-item tuple in that order. Hover and
forward-flight use their respective configured targets. Policies and
normalization statistics from earlier observation layouts need retraining.

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
Saved runs retain their setting for evaluation and rendering. Old saved angles
measured from the horizon are converted with `90 - old_angle`. Gentle upright
ground contact alone is not a crash. Existing map/altitude limits and numerical
failure handling remain active; the episode time limit remains a truncation.

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
velocity, angular-motion, attitude, or action penalties. On arrival within a 10 ft 3D radius, the
reward is calculated for the reached target. By default arrival ends the episode
successfully. With `--continual`, the observation instead reports differences
from a newly sampled waypoint and the episode continues. Targets are sampled within
±100 ft north/east and ±50 ft altitude of the current position; altitude is raised
where necessary to maintain 50 ft clearance above terrain at the target.
Targets remain within map boundaries and outside the arrival radius. The usual
failure rules and episode time limit still apply. `info` includes `targets_reached`
and the active `target_position` (north, east, altitude).

`HeliGoto` accepts `target_horizontal_range`, `target_altitude_range`,
`target_radius`, and `target_ground_clearance` constructor settings in feet,
plus `progress_weight` (nonnegative) and `arrival_reward` (positive).
Sampling uses the environment random generator, so `reset(seed=...)` reproduces
its target sequence. Rendering shows task target values and a translucent cyan 3D marker with dark edges for
explicit position targets, including hover and goto.

```bash
python main.py train --task goto --target-dir runs/goto
```

Use `--task smooth-stabilize` or `--task smooth-hover` to add a control
smoothness cost to the corresponding task. Each step subtracts
`lambda * mean((action - previous_action)**2 + action**2)`
over the four normalized, clipped controls. This penalizes both abrupt changes
and large command magnitudes relative to zero, without a stored trim reference.
Holding a nonzero command still incurs a magnitude cost. The change term compares
against the preceding applied action (the reset controls on the first step).
Each reset starts a new action history. Previous controls are included in the
observation.
Set lambda with `--smoothness-weight` in the CLI or `smoothness_weight` in
the environment constructor. The default is 0.25, and zero recovers the base task reward.

```bash
python main.py train --task smooth-hover --smoothness-weight 0.1 --target-dir runs/smooth-hover
```

Use `gym.make("ApproximateTargetValues-v0", target={"sea_alt": 2000, "forward_speed": 100})`
to track selected targets. Velocities are signed ground-relative body components
in ft/s, altitude is positive up, and yaw rotation velocity is body rate r in rad/s.
Each active error is current minus target divided by its characteristic scale.
Position uses rotor diameter D, speed uses sqrt(Dg), angular rate uses 1/T
where T = sqrt(D/g), and heading uses radians. Unspecified targets contribute zero.

For error e, the cost is abs(e) with `diff_impact="linear"` (default) or e² with
`"quadratic"`. Rewards sum over target components. Mode `"difference"` gives
negative cost, `"progression"` (default) gives previous minus current cost for
the same target, and `"attitude"` gives the negative instantaneous cost derivative
per normalized time, taking the linear gradient as zero at zero error.

Stabilize sets u, v and w to zero. Hover captures the initial position on each
reset and accepts an explicit target_altitude. Forward flight uses fixed-target
Goto with altitude and forward speed. Goto without a constructor target retains
its random waypoint and arrival bonus. Pass `continual=True` for successive waypoints.

```bash
python main.py train --task smooth-stabilize --target-dir runs/smooth-stabilize
python main.py train --task smooth-hover --target-dir runs/smooth-hover
```

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
runs in the same target directory. Evaluate/render any checkpoint by passing its
folder as `--target-dir`. At each checkpoint, the target directory's
`learning_curves.json` is atomically replaced with the complete episode history
so far; only this latest cumulative history is retained. Final saving updates it
again. Checkpoint timing is independent of SB3's printed training statistics.
`--n-checkpoints` replaces the previously unused `--checkpoint-interval` option.

`train` saves the policy, normalization statistics (when enabled), run arguments,
and episode data in `learning_curves.json`. `eval` runs ten deterministic episodes
without rendering and writes episode returns, lengths, and summary statistics to
`evaluation.json`; use `--eval-episodes` to change the count. `plot` reads these
JSON files, saves `learning_curves.png` and/or `evaluation.png`, and displays the
figures. On a server, set `MPLBACKEND=Agg` to save figures without opening windows.
`render` starts one agent-piloted episode at simulator speed. Press **M** to switch
between agent and manual keyboard control. In agent mode, movement/reset keys are
ignored; in manual mode, the keyboard controls the helicopter. Returning to the
agent uses the latest observation. Press **P** to pause/resume in either mode;
the window stays responsive while physics is paused. The GUI indicates the pilot
mode. The episode ends at the configured time limit or a terminal condition.

Use `--manual` to launch a selected task without any policy, saved normalization,
or training dependencies. Use it with the `render` command. It ignores saved run settings and uses the supplied environment
options and their defaults. **M** has no effect because no agent is loaded.

```bash
python main.py render --manual --task goto --continual --episode-seconds 1000
python main.py render --target-dir runs/goto   # agent initially; M toggles manual
```

Manual controls are **Z/S** (collective), **Up/Down** (longitudinal cyclic),
**Right/Left** (lateral cyclic), and **D/Q** (pedals), each changing the command
by 0.2 per key press. **X** zeros the controls; **R** restarts the episode.
**Esc** closes the window and **V** switches the camera view in either pilot mode.
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
also work while paused. The left column stacks Observations, Wind, then Target differences. The heading difference is displayed in degrees.
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

Independent `eval` and `render` commands reuse the algorithm and environment
settings in the run's `arguments.json`; explicit CLI options override them.
For example, `python main.py render --target-dir runs/hover --wind` enables wind
even if the run was trained with `--no-wind`. These two flags are mutually exclusive.
Set `--wind-speed` (ft/s), `--wind-direction` (degrees clockwise from north),
and `--turbulence-level` (1–7) to override weather settings, for example:

```bash
python main.py render --target-dir runs/hover --wind --wind-speed 12 --wind-direction 90 --turbulence-level 3
```

These settings are saved with the run and restored for independent evaluation
and rendering. Unspecified values use the environment defaults when no saved
value exists. When wind is disabled, weather settings are retained for reactivation with **W**;
they have no effect on the aircraft until wind is enabled.
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
### __Observation Space__
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
| 25  | previous collective|       | -1    | 1      |
| 26  | previous lon. cyclic|      | -1    | 1      |
| 27  | previous lat. cyclic|      | -1    | 1      |
| 28  | previous pedal     |       | -1    | 1      |

At reset, the previous controls are the trim controls used to create the
initial steady-flight state. After each step, they are the direct controls
most recently supplied by the agent.
-------------------------------------
## __Tasks__
For now only one task is available. However, rewarding is not tested well yet.

| Environment           | Details              |
|--------------------   |----------------------|
| HeliHover-v0          | Hover                |
| HeliForwardFlight-v0  | Forward Flight       |
| HeliObliqueFlight-v0  | Oblique Flight       |

-------------------------------------
## __Tested OS__
Environment tested on these OSs. If you have any problem, probably shared libraries for rendering 
make it, please look at [renderer page.](heligym/envs/renderer/README.md)

Environment should be run at least 100 FPS to simulate helicopter precisely. 
Receiving max FPS with NVIDIA 1070-TI with Intel i7-8700K given in the table. Limiting factor of FPS is CPU because of calculation of dynamics of helicopter in Python. If system cannot reach 100 FPS, please make sure that GPU driver be installed and make sure that CPU can handle with the Python.

| Tested OS      |  Max FPS |
|----------------|----------|
| Windows 10     |    350   |
| Ubuntu 18.04   |    470   |
| Ubuntu 16.04   |    500   |

For the user who has lower than the version of Ubuntu 16.04 (like Ubuntu 14.04 etc.) or other Linux distros, 
please re-compile dependent libraries. Re-compile instruction can be found at [renderer page.](heligym/envs/renderer/README.md)

-------------------------------------
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
│       ├── helis/
│       │   └── aw109.yaml             # AW109 aircraft and environment parameters
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
├── test.py                            # Interactive simulation/demo entry point
├── pyproject.toml                     # Package metadata, dependencies, and included data
├── uv.lock                            # Reproducible Python dependency lockfile
├── setup_install.sh                   # Guided Conda/uv/virtual-environment installer
├── init.sh                            # Convenience script for activating the local environment
├── LICENSE                            # MIT licence
└── README.md                          # Project overview and user documentation
```

This separation keeps reinforcement-learning tasks independent from the equations of motion: new tasks normally belong in `helicopter_with_tasks.py`, new aircraft are described under `helis/`, and changes to flight physics stay in `dynamics/`. Rendering remains a self-contained subsystem, and `api.py` shields environment code from the lower-level `ctypes` and C++ implementation. Generated caches and virtual environments are intentionally omitted from the tree above, while assets and bundled third-party dependencies are grouped rather than listing every generated or upstream file.

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

The rest is usual as of any GYM environment !

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
- [x] Add **oblique flight** task
- [ ] Add **turning flight** task
- [ ] Add **ground** task
- [ ] Add **slalom maneuver** task

### Tune PPO with Optuna

Install Optuna in the environment you use for HeliGym, then run a study for one task.

```bash
python -m pip install optuna
python main.py tune --task smooth-hover --target-dir runs/smooth-hover \
  --wind --random-wind --wind-speed 20 --turbulence-level 3
```

Wind is enabled by default. Use `--no-wind` for calm conditions. With
`--random-wind`, the tuner samples a direction and a speed between zero and
`--wind-speed` (default upper bound 20 ft/s). Training uses one sample per
training seed, and evaluation uses a fresh sample per episode. All candidates
in a phase share the same scenarios. The selected turbulence level is fixed.

The tuner ranks search trials by mean evaluation return, then retrains the top
three settings and the default PPO settings from scratch on three fresh seeds
with a larger budget. Refinement uses the same held-out evaluation seeds for
each policy. See `--help` to adjust the trial count, budgets, and seed counts.
PPO can round a requested budget up to its next full rollout.

The study lives in `<target-dir>/optuna`. An existing study prints its saved
`summary.json` before resuming. Use the
same settings and `--target-dir` to add trials, or pass `--n-trials 0` to
continue refinement without adding trials. Completed refinement runs are
reused. Use `python main.py tune --summary-only --target-dir <dir>` to read the saved results.

After refinement, `<target-dir>/optuna/final_params.json` contains the winning
PPO settings and the study's environment options. Use `--use-tuned` to start
fresh training in the same run directory.

```bash
python main.py tune --task stabilize --no-wind --target-dir runs/stabilize
python main.py train eval --target-dir runs/stabilize --use-tuned \
  --seed 123 --total-timesteps 1966080
```

To use a different study directory, pass `--tune-dir runs/stabilize/optuna`.
To load a specific JSON file, use `--file-args`. Explicit CLI options take
precedence over the exported settings. Training writes the effective
parameters to `<target-dir>/arguments.json`.
