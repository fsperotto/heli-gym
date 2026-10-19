"""Startup progress must precede slow imports and survive renderer setup."""
import builtins
import contextlib
import io
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import main


class StartupTests(unittest.TestCase):
    def test_progress_flushes_before_work_and_reports_elapsed_time(self):
        events = []
        stream = Mock()
        stream.write.side_effect = lambda value: events.append(('write', value))
        stream.flush.side_effect = lambda: events.append(('flush', None))
        with patch('sys.stdout', stream), patch.object(main.time, 'perf_counter', side_effect=[10, 12.5]):
            with main.progress('Loading example'):
                events.append(('work', None))
        self.assertLess(events.index(('flush', None)), events.index(('work', None)))
        output = ''.join(value for kind, value in events if kind == 'write')
        self.assertEqual(output, '[heligym] Loading example... [ok] 2.50s\n')

    def test_multiline_progress_keeps_intervening_output_on_its_own_line(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                patch.object(main.time, 'perf_counter', side_effect=[10, 12.5]):
            with main.progress('Training example', multiline=True):
                print('Training output')
        self.assertEqual(output.getvalue(),
                         '[heligym] Training example...\n'
                         'Training output\n'
                         '[heligym] Training example completed in 2.50s\n')

    def test_progress_reports_failure_and_preserves_exception(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, 'load failed'):
                with main.progress('Loading example'):
                    raise RuntimeError('load failed')
        self.assertIn('Loading example... [failed]', output.getvalue())
        self.assertNotIn('completed', output.getvalue())

    def test_help_has_no_startup_messages_or_training_imports(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(main, 'prepare_renderer') as prepare:
            with self.assertRaises(SystemExit) as error:
                main.main(['--help'])
        self.assertEqual(error.exception.code, 0)
        self.assertIn('--target-dir', output.getvalue())
        self.assertNotIn('[heligym]', output.getvalue())
        prepare.assert_not_called()

    def test_pytorch_progress_is_printed_before_the_import(self):
        torch = ModuleType('torch')
        torch.set_num_threads = Mock()
        sb3 = ModuleType('stable_baselines3')
        for name in ('PPO', 'A2C', 'SAC', 'TD3', 'DDPG'):
            setattr(sb3, name, Mock())
        output = io.StringIO()
        original_import = builtins.__import__
        imports = []

        def checked_import(name, *args, **kwargs):
            if name == 'torch':
                self.assertIn('Importing PyTorch...', output.getvalue())
                imports.append(name)
            if name == 'stable_baselines3':
                self.assertIn('Importing Stable Baselines3...', output.getvalue())
                imports.append(name)
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {'torch': torch, 'stable_baselines3': sb3}), \
                patch('builtins.__import__', side_effect=checked_import), \
                patch.object(main, 'train') as train, \
                contextlib.redirect_stdout(output):
            main.main(['train'])
        self.assertEqual(imports, ['torch', 'stable_baselines3'])
        train.assert_called_once()

    def test_non_linux_keeps_normal_renderer_initialization(self):
        with patch('sys.platform', 'win32'), \
                patch('importlib.util.spec_from_file_location') as load:
            main.prepare_renderer()
        load.assert_not_called()

    def test_renderer_bootstrap_avoids_simulator_imports_and_restarts_only_once(self):
        original_import = builtins.__import__

        def no_heavy_imports(name, *args, **kwargs):
            if name.split('.')[0] in ('heligym', 'torch', 'stable_baselines3', 'numpy', 'gymnasium'):
                raise AssertionError(f'Unexpected import during bootstrap: {name}')
            return original_import(name, *args, **kwargs)

        renderer_dir = Path(main.__file__).resolve().parent / 'heligym' / 'envs' / 'renderer'
        with patch('sys.platform', 'linux'), \
                patch.dict(os.environ, {'LD_LIBRARY_PATH': ''}), \
                patch('os.execv') as restart, \
                patch('builtins.__import__', side_effect=no_heavy_imports), \
                contextlib.redirect_stdout(io.StringIO()):
            main.prepare_renderer()
            self.assertEqual(os.environ['HELIGYM_LIBS_DIR'], str(renderer_dir / 'libs'))
            self.assertIn(str(renderer_dir / 'bin'), os.environ['LD_LIBRARY_PATH'].split(os.pathsep))
            main.prepare_renderer()
        restart.assert_called_once_with(sys.executable, [sys.executable] + sys.argv)


if __name__ == '__main__':
    unittest.main()
