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
`main.py` accepts the positional commands `train`, `eval`, `plot`, and `render`.
With no commands it runs all four in that order; otherwise it runs only the
selected stages, in the same pipeline order.

```bash
python main.py --target-dir runs/hover                 # entire pipeline
python main.py train eval --target-dir runs/hover      # headless training and evaluation
python main.py eval plot --target-dir runs/hover       # evaluate a saved policy and plot statistics
python main.py plot --target-dir runs/hover            # plot saved statistics only
python main.py render --target-dir runs/hover          # one episode in the GUI
```

Use `--task smooth-stabilize` or `--task smooth-hover` to add a control
smoothness cost to the corresponding task. Each step subtracts
`sum((action - previous_action)**2)` over the four normalized, clipped controls.
Large jumps cost more than small corrections; unchanged controls cost zero.
The first step compares against the reset trim controls, and every reset starts
a new action history. The previous controls are included in the observation.
When constructing these environments directly, `smoothness_weight` scales this
cost (default 1.0; zero recovers the base task reward).

`stabilize` penalizes linear/angular velocity and roll/pitch tilt, with a large
crash cost. `hover` targets north/east position and sea altitude while suppressing
angular rates; it does not explicitly track a heading. The smooth variants keep
these base objectives and success conditions. They penalize immediate control
changes, rather than imposing a hard control-rate limit.

```bash
python main.py train --task smooth-stabilize --target-dir runs/smooth-stabilize
python main.py train --task smooth-hover --target-dir runs/smooth-hover
```

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
`render` shows one deterministic episode at simulator speed. Press **P** to pause
or resume playback; the window stays responsive while physics is paused.
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
value exists. When wind is disabled, all three weather overrides are ignored.
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
| 17  | previous collective|       | -1    | 1      |
| 18  | previous lon. cyclic|      | -1    | 1      |
| 19  | previous lat. cyclic|      | -1    | 1      |
| 20  | previous pedal     |       | -1    | 1      |

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
├── setup_install.sh                   # Guided uv/virtual-environment installer
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

Alternatively, source the guided installation script. It uses `uv` to create a local `.venv` and install the package:

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
