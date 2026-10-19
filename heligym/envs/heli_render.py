"""Optional human renderer for helicopter environments."""

import numpy as np


FT2MTR = 0.3048


class HeliRender:
    """Own native rendering resources and draw helicopter state snapshots."""

    def __init__(self, heli_name="aw109", fps=50.0, native_renderer=None):
        if native_renderer is None:
            # Keep native libraries out of headless workers and --help paths.
            from .renderer.api import Renderer
            native_renderer = Renderer(w=1024, h=768, title="heligym")
        self.native = native_renderer
        self.native.set_fps(fps)

        self.helicopter = self.native.create_model(
            f"/resources/models/{heli_name}/{heli_name}.obj",
            f"/resources/shaders/{heli_name}_vertex.vs",
            f"/resources/shaders/{heli_name}_frag.fs",
        )
        self.native.add_permanent_object_to_window(self.helicopter)
        self.terrain = self.native.create_model(
            "/resources/models/terrain/terrain.obj",
            "/resources/shaders/terrain_vertex.vs",
            "/resources/shaders/terrain_frag.fs",
        )
        self.native.add_permanent_object_to_window(self.terrain)
        self.sky = self.native.create_model("/resources/models/sky/sky.obj")
        self.native.add_permanent_object_to_window(self.sky)

        self.info_text = [
            "POWER      : %5.2f hp",
            "LON_VEL    : %5.2f ft/s",
            "LAT_VEL    : %5.2f ft/s",
            "DWN_VEL    : %5.2f ft/s",
            "N_VEL      : %5.2f ft/s",
            "E_VEL      : %5.2f ft/s",
            "DES_RATE   : %5.2f ft/s",
            "ROLL       : %5.2f rad",
            "PITCH      : %5.2f rad",
            "YAW        : %5.2f rad",
            "ROLL_RATE  : %5.2f rad/sec",
            "PITCH_RATE : %5.2f rad/sec",
            "YAW_RATE   : %5.2f rad/sec",
            "N_POS      : %5.2f ft",
            "E_POS      : %5.2f ft",
            "ALT        : %5.2f ft",
            "GR_ALT     : %5.2f ft",
        ]
        self.simulation_text = [
            "FPS        : %3.0f",
            "STEP       : %9.0f",
            "REWARD     : %9.3f",
        ]
        self.wind_text = [
            "ENABLED    : %.0f (0 off, 1 on)",
            "MEAN_SPEED : %6.2f ft/s",
            "MEAN_DIR   : %6.1f deg",
            "TURB_LEVEL : %3.0f",
            "SPEED      : %6.2f ft/s",
            "NORTH      : %6.2f ft/s",
            "EAST       : %6.2f ft/s",
            "DOWN       : %6.2f ft/s",
        ]
        self.action_text = [
            "COLLECTIVE : %6.3f",
            "LON_CYCLIC : %6.3f",
            "LAT_CYCLIC : %6.3f",
            "PEDAL      : %6.3f",
        ]
        self.command_text = [
            "P          : Pause / resume",
            "R          : Reset and restart",
            "X          : Set all controls to zero",
            "Esc        : Stop and close window",
            "Up / Down  : Lon. cyclic +0.2 / -0.2",
            "Right/Left : Lat. cyclic +0.2 / -0.2",
            "Z / S      : Collective  +0.2 / -0.2",
            "D / Q      : Pedals      +0.2 / -0.2",
            "V          : Switch fixed / chase view",
            "Controls are clipped to [-1, +1]",
            "One change per key press",
        ]
        self.info_id = self.native.create_guiText("Observations", 30.0, 30.0, 250.0, 0.0)
        self.action_id = self.native.create_guiText("Actions", 300.0, 30.0, 250.0, 0.0)
        self.simulation_id = self.native.create_guiText("Simulation", 300.0, 160.0, 250.0, 0.0)
        self.wind_id = self.native.create_guiText("Wind", 300.0, 270.0, 250.0, 0.0)
        self.command_id = self.native.create_guiText(
            "Command Instructions", 570.0, 30.0, 360.0, 0.0
        )

        # Own these arrays because the native GUI retains pointers to them.
        self.info_values = np.zeros(len(self.info_text), dtype=np.float32)
        self.action_values = np.zeros(4, dtype=np.float32)
        self.simulation_values = np.zeros(len(self.simulation_text), dtype=np.float32)
        self.command_values = np.zeros(len(self.command_text), dtype=np.float32)
        self.wind_values = np.zeros(len(self.wind_text), dtype=np.float32)
        self.native.add_guiText(self.info_id, self.info_text, self.info_values)
        self.native.add_guiText(self.action_id, self.action_text, self.action_values)
        self.native.add_guiText(self.simulation_id, self.simulation_text, self.simulation_values)
        self.native.add_guiText(self.command_id, self.command_text, self.command_values)
        self.native.add_guiText(self.wind_id, self.wind_text, self.wind_values)

    def render(self, state, observation, action, paused=False, step=0, reward=0.0,
               wind_enabled=False, wind_speed=0.0, wind_direction=0.0,
               turbulence_level=0, wind_ned=None):
        """Draw one snapshot without retaining or modifying the environment."""
        if self.is_close():
            self.close()
            return
        if paused:
            self.wait_events()
            return

        self.info_values[:] = observation
        self.simulation_values[:] = (self.native.get_fps(), step, reward)
        self.action_values[:] = action
        self.wind_values.fill(0.0)
        if wind_enabled:
            vector = np.zeros(3, dtype=np.float32) if wind_ned is None else wind_ned
            self.wind_values[:] = (1, wind_speed, wind_direction, turbulence_level,
                                   np.linalg.norm(vector), *vector)
        self.native.set_guiText(self.info_id, self.info_text, self.info_values)
        self.native.set_guiText(self.action_id, self.action_text, self.action_values)
        self.native.set_guiText(self.simulation_id, self.simulation_text, self.simulation_values)
        self.native.set_guiText(self.wind_id, self.wind_text, self.wind_values)
        self.native.set_control_bars(action)

        self.native.rotate_MR(
            self.helicopter, state["betas"][1], state["betas"][0], state["psi_mr"][0]
        )
        self.native.rotate_TR(self.helicopter, 0, state["psi_tr"][0], 0)
        self.native.translate_model(self.helicopter, *(state["xyz"] * FT2MTR))
        self.native.rotate_model(self.helicopter, *state["euler"])
        self.native.translate_model(
            self.sky,
            state["xyz"][0] * FT2MTR,
            state["xyz"][1] * FT2MTR,
            state["xyz"][2] * FT2MTR + 500,
        )
        self.native.update_helicopter_camera(
            state["xyz"][0] * FT2MTR,
            state["xyz"][1] * FT2MTR,
            state["xyz"][2] * FT2MTR,
            state["euler"][2],
        )
        if not self.native.is_visible():
            self.native.show_window()
        self.native.render()

    def get_key_presses(self):
        return self.native.get_key_presses()

    def is_close(self):
        return self.native.is_close()

    def wait_events(self):
        self.native.wait_events()

    def close(self):
        self.native.close()
        self.native.terminate()

    def terminate(self):
        self.close()
