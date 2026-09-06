# heli-gym 🚁
OpenAI GYM environment for 6-DOF Helicopter simulation.

This is GYM environment package for reinforcement learning for helicopter flight tasks using minimum complexity helicopter model.

-------------------------------------
## __Environment Details__

* Pure Gym environment
* Realistic Dynamic Model based on Minimum Complexity Helicopter Model ([Heffley and Mnich](https://robertheffley.com/docs/Sim_modeling/Heffley-Mnich--Minimum-Complexity%20Helicopter%20Simulation%20Math%20Model--NASA%20CR%20177476.pdf))
In addition, inflow dynamics are added and model is adjusted so that it covers multiple flight conditions. 
* Rendering is done by OpenGL.

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
