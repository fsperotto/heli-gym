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
            "FPS        : %3.0f ",
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
        self.command_id = self.native.create_guiText(
            "Command Instructions", 570.0, 30.0, 360.0, 0.0
        )

        # Own these arrays because the native GUI retains pointers to them.
        self.info_values = np.zeros(len(self.info_text), dtype=np.float32)
        self.action_values = np.zeros(4, dtype=np.float32)
        self.command_values = np.zeros(len(self.command_text), dtype=np.float32)
        self.native.add_guiText(self.info_id, self.info_text, self.info_values)
        self.native.add_guiText(self.action_id, self.action_text, self.action_values)
        self.native.add_guiText(self.command_id, self.command_text, self.command_values)

    def render(self, state, observation, action, paused=False):
        """Draw one snapshot without retaining or modifying the environment."""
        if self.is_close():
            self.close()
            return
        if paused:
            self.wait_events()
            return

        self.info_values[0] = self.native.get_fps()
        self.info_values[1:] = observation
        self.action_values[:] = action
        self.native.set_guiText(self.info_id, self.info_text, self.info_values)
        self.native.set_guiText(self.action_id, self.action_text, self.action_values)
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
