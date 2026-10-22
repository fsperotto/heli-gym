import os
import sys
import ctypes
import re
import numpy as np
from pathlib import Path

"""
    Load shadered library w.r.t OS.
"""

renderer_dir = Path(__file__).resolve().parent

if sys.platform == 'win32':

    # Keep the handle alive so the directory remains registered.
    _dll_directory = os.add_dll_directory(str(renderer_dir / "libs"))

    lib = ctypes.CDLL(str(renderer_dir / "bin" / "Heligym.dll"))    
    #lib = ctypes.cdll.LoadLibrary(os.environ['HELIGYM_PYTHON_DIR'] + '/Heligym.dll')

elif sys.platform == 'linux':
    lib = ctypes.CDLL(str(renderer_dir / "bin" / "libHeligym.so"))    
    #lib = ctypes.cdll.LoadLibrary(os.environ['HELIGYM_PYTHON_DIR'] + '/libHeligym.so')

###################################################################################
def _to_encode(str_to_encode, encode_type = 'utf-8'):
    """
        Encoding the string into `encode_type` for char pointer of ctypes.
    """
    value = os.fspath(str_to_encode) if isinstance(str_to_encode, os.PathLike) else str_to_encode
    encoded = value if isinstance(value, bytes) else value.encode(encode_type)
    if b'\0' in encoded:
        raise ValueError('Native strings must not contain NUL bytes')
    return encoded

###################################################################################
lib.create_window.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_char_p]
lib.create_window.restype = ctypes.c_void_p

def create_window(w, h, title):
    """
        Create window with parameters which are
        >>> w     : Width of window
        >>> h     : Height of window
        >>> title : Title of window.
    """
    if any(not isinstance(value, (int, np.integer)) or not 0 < value <= 2**31 - 1 for value in (w, h)):
        raise ValueError('Window dimensions must be positive 32-bit integers')
    window = lib.create_window(w, h, ctypes.c_char_p(_to_encode(title)))
    if not window:
        raise RuntimeError('Native renderer could not create a window; check the graphics/display setup')
    return window

###################################################################################
lib.render.argtypes = [ctypes.c_void_p]
lib.render.restype = None

def render(window):
    """
        Render window.
    """
    lib.render(window)


def set_control_bars(window, values):
    """Copy four signed control values to the native renderer."""
    
    try:
        setter = lib.set_control_bars
    except AttributeError as exc:
        #print("Rebuild the native renderer to enable control bars.")
        return
    
    setter.argtypes = [ctypes.c_void_p] + [ctypes.c_float] * 4
    setter.restype = None
    setter(window, *values)


def wait_events():
    """Wait briefly for window events without drawing or swapping buffers."""
    lib.wait_events.argtypes = []
    lib.wait_events.restype = None
    lib.wait_events()


def get_key_presses(window):
    """Drain ordered key presses collected during rendering."""
    try:
        pop = lib.pop_key_press
    except:
        return []
    pop.argtypes = [ctypes.c_void_p]
    pop.restype = ctypes.c_int
    names = {84: 't', 87: 'w', 77: 'm', 80: 'p', 82: 'r', 88: 'x', 90: 'z', 83: 's', 81: 'q', 68: 'd',
             265: 'up', 264: 'down', 263: 'left', 262: 'right'}
    keys = []
    while True:
        key = pop(window)
        if key == -1:
            return keys
        if key in names:
            keys.append(names[key])

###################################################################################
lib.close.argtypes = [ctypes.c_void_p]
lib.close.restype = None

def close(window):
    """
        Close window.
    """
    lib.close(window)

###################################################################################
lib.is_close.argtypes = [ctypes.c_void_p]
lib.is_close.restype = ctypes.c_bool

def is_close(window):
    """
        Check whether window is closed or not.
    """
    return lib.is_close(window)  

###################################################################################
lib.terminate.argtypes = []
lib.terminate.restype = None

def terminate():
    """
        Terminate the OpenGL.
    """
    lib.terminate()

###################################################################################
lib.create_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.create_model.restype = ctypes.c_void_p

def create_model(object_path, vertex_shader_path, fragment_shader_path):
    """
        Create model with its shaders.
    """
    model = lib.create_model(ctypes.c_char_p(_to_encode(object_path)),
                            ctypes.c_char_p(_to_encode(vertex_shader_path)),
                            ctypes.c_char_p(_to_encode(fragment_shader_path)))
    if not model:
        raise RuntimeError(f'Native renderer could not load model: {object_path}')
    return model

###################################################################################
lib.add_permanent_to_window.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.add_permanent_to_window.restype = None

def add_permanent_to_window(window, model):
    """
        Add model as permanent drawables object to window.
    """
    lib.add_permanent_to_window(window, model)

###################################################################################
lib.add_instantaneous_to_window.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.add_instantaneous_to_window.restype = None

def add_instantaneous_to_window(window, model):
    """ 
        Add model as instantaneous drawables object to window.
    """
    lib.add_instantaneous_to_window(window, model)
    
###################################################################################
lib.translate_model.argtypes = [ctypes.c_void_p, ctypes.c_float, 
                                ctypes.c_float, ctypes.c_float]
lib.translate_model.restype = None

def translate_model(model, x, y, z):
    """
        Translate model to locations. Locations will be in OpenGL coordinates.
    """
    lib.translate_model(model, x, y, z)

###################################################################################
lib.rotate_model.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, 
                                ctypes.c_float, ctypes.c_float]
lib.rotate_model.restype = None

def rotate_model(model, angle, x, y, z):
    """
        Rotate model to angle. From will be an rotation around arbitrary angle.
    """
    lib.rotate_model(model, angle, x, y, z)

###################################################################################
lib.scale_model.argtypes = [ctypes.c_void_p, ctypes.c_float, 
                                ctypes.c_float, ctypes.c_float]
lib.scale_model.restype = None

def scale_model(model, x, y, z):
    """
        Scale model to the ratios for each axis.
    """
    lib.scale_model(model, x, y, z)

###################################################################################
lib.get_fps.argtypes = [ctypes.c_void_p]
lib.get_fps.restype = ctypes.c_float

def get_fps(window):
    """
        Get FPS from the window.
    """
    return np.round(lib.get_fps(window), 2)

###################################################################################
lib.set_fps.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.set_fps.restype = None

def set_fps(window, fps):
    """
        Set FPS of the window.
    """
    if not np.isfinite(fps) or fps <= 0 or 1e9 / fps >= 2**63:
        raise ValueError('FPS must be finite, positive, and fit the native timer')
    lib.set_fps(window, fps)

###################################################################################
lib.get_camera.argtypes = [ctypes.c_void_p]
lib.get_camera.restype = ctypes.c_void_p

def get_camera(window):
    """
        Get Camera pointer of the window.
    """
    camera = lib.get_camera(window)
    if not camera:
        raise RuntimeError('Native renderer returned a null camera')
    return camera

###################################################################################
lib.set_camera_pos.argtypes = [ctypes.c_void_p, ctypes.c_float,
                                ctypes.c_float, ctypes.c_float]
lib.set_camera_pos.restype = None

def set_camera_pos(camera, x, y, z):
    """
        Set Camera position of each axis.
    """
    lib.set_camera_pos(camera, x, y, z)

###################################################################################
lib.get_camera_pos.argtypes = [ctypes.c_void_p]
lib.get_camera_pos.restype = ctypes.POINTER(ctypes.c_float)

def get_camera_pos(camera):
    """
        Get camera position as float array.
    """
    if not camera:
        raise ValueError('Camera handle must not be null')
    position = lib.get_camera_pos(camera)
    if not position:
        raise RuntimeError('Native renderer returned a null camera position')
    return np.ctypeslib.as_array(position, shape=(3,)).copy()


def update_helicopter_camera(window, north, east, down, bearing):
    """Update the camera from a helicopter pose; V selects the tracking mode."""
    try:
        update = lib.update_helicopter_camera
    except AttributeError as exc:
        #print("Rebuild the native renderer to enable chase view.")
        return
    
    update.argtypes = [ctypes.c_void_p] + [ctypes.c_float] * 4
    update.restype = None
    update(window, north, east, down, bearing)

###################################################################################
lib.is_visible.argtypes = [ctypes.c_void_p]
lib.is_visible.restype = ctypes.c_bool

def is_visible(window):
    """
        Check whether the window is visible or not.
    """
    return lib.is_visible(window)

###################################################################################
lib.hide_window.argtypes = [ctypes.c_void_p]
lib.hide_window.restype = None

def hide_window(window):
    """
        Hide the window.
    """
    lib.hide_window(window)

###################################################################################
lib.show_window.argtypes = [ctypes.c_void_p]
lib.show_window.restype = None

def show_window(window):
    """
        Show the window.
    """
    lib.show_window(window)

###################################################################################
lib.create_guiTextVector.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
lib.create_guiTextVector.restype = ctypes.c_int

def create_guiText(window, title, pos, size):
    """
        Create guiText vector for guiTextSection.        
    """
    assert type(pos) == tuple and len(pos) == 2, 'Make sure that position of guiText should be tuple and has 2 element.'
    assert type(size) == tuple and len(size) == 2, 'Make sure that size of guiText should be tuple and has 2 element.'
    return lib.create_guiTextVector(window, ctypes.c_char_p(_to_encode(title)), pos[0], pos[1], size[0], size[1])

###################################################################################
lib.add_guiText.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')]
lib.add_guiText.restype = None

_GUI_FLOAT_FORMAT = re.compile(r'%%|%[-+ #0]*[0-9]*(?:\.[0-9]+)?[eEfFgGaA]')


def _gui_labels(labels):
    encoded = []
    for label in labels:
        text = label.decode('utf-8') if isinstance(label, bytes) else label
        if not isinstance(text, str):
            raise ValueError('GUI labels must be strings')
        formats = _GUI_FLOAT_FORMAT.findall(text)
        if '%' in _GUI_FLOAT_FORMAT.sub('', text) or sum(f != '%%' for f in formats) > 1:
            raise ValueError('GUI labels support at most one float placeholder per row')
        encoded.append(_to_encode(text))
    return encoded


def _gui_values(values, expected):
    array = np.atleast_1d(np.asarray(values, dtype=np.float32))
    if array.ndim != 1 or array.size != expected:
        raise ValueError(f'GUI requires exactly {expected} values in a one-dimensional array')
    return np.ascontiguousarray(array)


def add_guiText(window, guiText, strn, val):
    """Copy labels and matching values without modifying caller-owned objects."""
    labels = _gui_labels(strn)
    values = _gui_values(val, len(labels))
    strings = (ctypes.c_char_p * len(labels))(*labels)
    add = lib.add_guiText_checked
    add.argtypes = lib.add_guiText.argtypes
    add.restype = ctypes.c_bool
    if not add(window, guiText, len(labels), strings, values):
        raise ValueError('Invalid GUI section or value buffer')


lib.set_guiText.argtypes = [ctypes.c_void_p, ctypes.c_int,
                           np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')]
lib.set_guiText.restype = None


def set_guiText(window, guiText, labels, val):
    """Update a complete GUI section; native code checks its stored row count."""
    values = _gui_values(val, len(labels))
    update = lib.set_guiText_checked
    update.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                      np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')]
    update.restype = ctypes.c_bool
    if not update(window, guiText, len(values), values):
        raise ValueError('Invalid GUI section or mismatched stored row count')

###################################################################################
lib.rotate_MR.argtypes = [ctypes.c_void_p, ctypes.c_float, 
                                ctypes.c_float, ctypes.c_float]
lib.rotate_MR.restype = None

def rotate_MR(model, x, y, z):
    """
        Rotate the model Main Rotor with each angle which are in radians.
    """
    lib.rotate_MR(model, x, y, z)

###################################################################################
lib.rotate_TR.argtypes = [ctypes.c_void_p, ctypes.c_float, 
                                ctypes.c_float, ctypes.c_float]
lib.rotate_TR.restype = None

def rotate_TR(model, x, y, z):
    """
        Rotate the model Tail Rotor with each angle which are in radians.
    """
    lib.rotate_TR(model, x, y, z)


def set_guiText_row_count(window, section, count):
    """Limit visible rows without reallocating native GUI buffers."""
    lib.set_guiText_row_count.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.set_guiText_row_count.restype = ctypes.c_bool
    if not lib.set_guiText_row_count(window, section, count):
        raise ValueError('Invalid GUI section')
