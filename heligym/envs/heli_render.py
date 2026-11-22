"""Optional human renderer for helicopter environments."""

import numpy as np


FT2MTR = 0.3048
ANGULAR_OBSERVATIONS = slice(7, 13)


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
        self.native.set_pilot_view(self.helicopter)
        self.terrain = self.native.create_model(
            "/resources/models/terrain/terrain.obj",
            "/resources/shaders/terrain_vertex.vs",
            "/resources/shaders/terrain_frag.fs",
        )
        self.native.add_permanent_object_to_window(self.terrain)
        self.sky = self.native.create_model("/resources/models/sky/sky.obj")
        self.native.add_permanent_object_to_window(self.sky)

        self.target_marker = None
        self.target_text = [
            "NORTH      : %8.2f ft",
            "EAST       : %8.2f ft",
            "ALTITUDE   : %8.2f ft",
            "FORWARD    : %8.2f ft/s",
            "HEADING    : %8.2f deg",
            "VERTICAL   : %8.2f ft/s",
            "LATERAL    : %8.2f ft/s",
            "YAW RATE   : %8.2f rad/s",
        ]
        self.info_text = [
            "POWER      : %5.2f hp",
            "FWD AIR    : %5.2f ft/s",
            "RIGHT AIR  : %5.2f ft/s",
            "DOWN AIR   : %5.2f ft/s",
            "N_VEL      : %5.2f ft/s",
            "E_VEL      : %5.2f ft/s",
            "DES_RATE   : %5.2f ft/s",
            "ROLL       : %5.2f deg",
            "PITCH      : %5.2f deg",
            "YAW        : %5.2f deg",
            "ROLL_RATE  : %5.2f deg/sec",
            "PITCH_RATE : %5.2f deg/sec",
            "YAW_RATE   : %5.2f deg/sec",
            "N_POS      : %5.2f ft",
            "E_POS      : %5.2f ft",
            "ALT        : %5.2f ft",
            "GR_ALT     : %5.2f ft",
            "RECENT YAW : %+6.2f turns",
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
            "PILOT MODE : %.0f (0 manual, 1 agent, 2 replay)",
            "M          : Switch agent / manual (if available)",
            "P          : Pause / resume",
            "T          : Stop and trim at current position",
            "W          : Enable / disable wind",
            "H          : Hide / show info panels",
            "Esc        : Stop and close window",
            "V          : Cycle fixed / chase / pilot view",
        ]
        self.common_command_rows = len(self.command_text)
        self.command_text += [
            "R          : Reset and restart",
            "X          : Set all controls to zero",
            "Up / Down  : Lon. cyclic +0.2 / -0.2",
            "Right/Left : Lat. cyclic +0.2 / -0.2",
            "Z / S      : Collective  +0.2 / -0.2",
            "D / Q      : Pedals      +0.2 / -0.2",
            "Controls are clipped to [-1, +1]",
            "One change per key press",
        ]
        self.info_id = self.native.create_guiText("Observations", 30.0, 30.0, 250.0, 0.0)
        self.action_id = self.native.create_guiText("Previous applied controls", 300.0, 30.0, 250.0, 0.0)
        self.simulation_id = self.native.create_guiText("Simulation", 300.0, 160.0, 250.0, 0.0)
        self.wind_id = self.native.create_guiText("Wind", 30.0, 380.0, 250.0, 0.0)
        self.command_id = self.native.create_guiText(
            "Command Instructions", 570.0, 30.0, 360.0, 0.0
        )

        self.target_id = self.native.create_guiText("Target differences", 30.0, 570.0, 250.0, 0.0)

        # Reuse these arrays when sending snapshots to the native GUI.
        self.target_values = np.zeros(len(self.target_text), dtype=np.float32)
        self.native.add_guiText(self.target_id, self.target_text, self.target_values)
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
               turbulence_level=0, wind_ned=None, target=None, target_position=None,
               pilot_mode='manual', recent_yaw_turns=0.0, previous_action=None):
        """Draw one snapshot without retaining or modifying the environment."""
        if self.is_close():
            self.close()
            return
        if paused:
            self.wait_events()
            if self.is_close():
                self.close()
                return

        if target is not None:
            self.target_values[:] = target
            self.target_values[4] = np.rad2deg(self.target_values[4])
        else:
            self.target_values.fill(0.)
        self.native.set_guiText(self.target_id, self.target_text, self.target_values)
        if target_position is not None:
            if self.target_marker is None:
                self.target_marker = self.native.create_model(
                    "/resources/models/waypoint/waypoint.obj",
                    "/resources/shaders/vertex_base.vs",
                    "/resources/shaders/target_frag.fs",
                )
            north, east, altitude = target_position
            self.native.translate_model(self.target_marker,
                                        north * FT2MTR, east * FT2MTR, -altitude * FT2MTR)
            self.native.add_instantanous_object_to_window(self.target_marker)

        self.native.set_guiText_row_count(
            self.command_id, -1 if pilot_mode == 'manual' else self.common_command_rows)
        self.command_values[0] = {'manual': 0., 'agent': 1., 'replay': 2.}[pilot_mode]
        self.native.set_guiText(self.command_id, self.command_text, self.command_values)
        self.info_values[:-1] = observation
        self.info_values[-1] = recent_yaw_turns
        angular_values = self.info_values[ANGULAR_OBSERVATIONS]
        np.rad2deg(angular_values, out=angular_values)
        self.simulation_values[:] = (self.native.get_fps(), step, reward)
        self.action_values[:] = action if previous_action is None else previous_action
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
            state["euler"][2], state["euler"][0], state["euler"][1],
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
