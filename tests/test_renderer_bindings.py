"""ctypes ABI and buffer validation without loading OpenGL or opening a window."""
import ctypes
import importlib.util
from pathlib import Path
import re
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class RendererBindingTests(unittest.TestCase):
    def setUp(self):
        self.lib = Mock()
        self.lib.create_window.return_value = 0x12345678ABCDEF
        self.lib.create_model.return_value = 0x12345678ABCDEF
        self.lib.add_guiText_checked.return_value = True
        self.lib.set_guiText_checked.return_value = True
        spec = importlib.util.spec_from_file_location('_binding_test', ROOT / 'heligym/envs/renderer/pyapi.py')
        self.binding = importlib.util.module_from_spec(spec)
        with patch('ctypes.CDLL', return_value=self.lib):
            spec.loader.exec_module(self.binding)

    def test_exported_signatures_match_cpp_header(self):
        b = self.binding
        # Initialize signatures declared lazily by backwards-compatible helpers.
        b.set_control_bars(1, [0.] * 4)
        b.wait_events()
        self.lib.pop_key_press.return_value = -1
        b.get_key_presses(1)
        b.update_helicopter_camera(1, 0, 0, 0, 0)
        b.add_guiText(1, 0, ['value %.2f'], [0.])
        b.set_guiText(1, 0, ['value %.2f'], [0.])
        b.set_guiText_row_count(1, 0, -1)
        header = (ROOT / 'heligym/envs/renderer/src/py_api.h').read_text()
        types = {'void': None, 'Window*': ctypes.c_void_p, 'Camera*': ctypes.c_void_p,
                 'Model*': ctypes.c_void_p, 'float*': ctypes.POINTER(ctypes.c_float),
                 'bool': ctypes.c_bool, 'float': ctypes.c_float, 'int': ctypes.c_int,
                 'unsigned int': ctypes.c_uint, 'char*': ctypes.c_char_p,
                 'char**': ctypes.POINTER(ctypes.c_char_p)}
        for result_type, name, arguments in re.findall(
                r'extern "C" RENDERER_API\s+([\w]+\s*\*?)\s+(\w+)\((.*?)\);', header, re.S):
            with self.subTest(function=name):
                native = getattr(self.lib, name)
                self.assertIs(native.restype, types[result_type.replace(' ', '')])
                args = [] if not arguments.strip() else arguments.split(',')
                self.assertEqual(len(native.argtypes), len(args))
                for actual, argument in zip(native.argtypes, args):
                    declaration = re.sub(r'\bconst\s+', '', argument.strip())
                    typename = re.sub(r'\s*\b\w+$', '', declaration).strip()
                    typename = re.sub(r'\s*\*\s*', '*', typename)
                    if typename == 'float*' and getattr(actual, '_dtype_', None) == np.dtype('float32'):
                        continue
                    self.assertIs(actual, types[typename])

    def test_gui_labels_are_not_mutated_and_values_are_contiguous(self):
        labels = ['SPEED %.2f', 'Progress 100%%']
        before = labels.copy()
        values = np.arange(4, dtype=np.float64)[::2]
        self.binding.add_guiText(1, 0, labels, values)
        self.binding.add_guiText(1, 0, labels, values)
        self.assertEqual(labels, before)
        args = self.lib.add_guiText_checked.call_args.args
        self.assertEqual(list(args[3]), [b'SPEED %.2f', b'Progress 100%%'])
        self.assertEqual(args[4].dtype, np.float32)
        self.assertTrue(args[4].flags.c_contiguous)

    def test_gui_rejects_short_or_multidimensional_buffers(self):
        for values in ([], [1.], [[1., 2.]]):
            with self.assertRaises(ValueError):
                self.binding.add_guiText(1, 0, ['a', 'b'], values)
            with self.assertRaises(ValueError):
                self.binding.set_guiText(1, 0, ['a', 'b'], values)
        self.lib.add_guiText_checked.assert_not_called()
        self.lib.set_guiText_checked.assert_not_called()

    def test_gui_rejects_unsafe_printf_formats_and_native_errors(self):
        for label in ('%s', '%d', '%n', '%f %f', '%*f', 'bad\0label'):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.binding.add_guiText(1, 0, [label], [0.])
        self.lib.set_guiText_checked.return_value = False
        with self.assertRaises(ValueError):
            self.binding.set_guiText(1, 0, ['a'], [0.])

    def test_camera_getter_returns_owned_snapshot(self):
        values = (ctypes.c_float * 3)(1., 2., 3.)
        self.lib.get_camera_pos.return_value = ctypes.cast(values, ctypes.POINTER(ctypes.c_float))
        snapshot = self.binding.get_camera_pos(1)
        values[0] = 999.
        np.testing.assert_array_equal(snapshot, [1, 2, 3])
        self.lib.get_camera_pos.return_value = None
        with self.assertRaises(RuntimeError):
            self.binding.get_camera_pos(1)

    def test_null_handles_and_invalid_dimensions_or_fps_fail_before_use(self):
        self.lib.create_window.return_value = None
        with self.assertRaises(RuntimeError):
            self.binding.create_window(800, 600, 'test')
        self.lib.create_model.return_value = None
        with self.assertRaises(RuntimeError):
            self.binding.create_model('a.obj', 'v.vs', 'f.fs')
        for size in (0, -1, 2**32, 1.5):
            with self.assertRaises(ValueError):
                self.binding.create_window(size, 600, 'test')
        for fps in (0, -1, np.nan, np.inf, 1e-30):
            with self.assertRaises(ValueError):
                self.binding.set_fps(1, fps)
        self.lib.set_fps.assert_not_called()
        self.lib.get_camera.return_value = None
        with self.assertRaises(RuntimeError):
            self.binding.get_camera(1)
        with self.assertRaises(ValueError):
            self.binding.get_camera_pos(None)
        self.lib.set_guiText_row_count.return_value = False
        with self.assertRaises(ValueError):
            self.binding.set_guiText_row_count(1, -1, 7)


if __name__ == '__main__':
    unittest.main()
