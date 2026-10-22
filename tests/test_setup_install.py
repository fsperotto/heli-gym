"""Guided installer routes commands using mocks, without installing packages."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='heli installer ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'project with spaces'
        self.project.mkdir()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name in ('setup_install.sh', 'init.sh'):
            shutil.copy(ROOT / name, self.project / name)
        self.log = self.root / 'commands.jsonl'
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('CONDA_', 'VIRTUAL_ENV', 'PYTHONHOME'))}
        self.env.update(PATH=f'{self.bin}:/usr/bin:/bin', TEST_LOG=str(self.log))
        self.python_stub = f'''#!{sys.executable}
import json, os, pathlib, sys
with open(os.environ['TEST_LOG'], 'a') as f: f.write(json.dumps(['python', sys.argv[0], *sys.argv[1:]])+'\\n')
prefix=pathlib.Path(sys.argv[0]).parent.parent
if sys.argv[1:2] == ['-c']: print(sys.argv[0])
if sys.argv[1:4] == ['-m','pip','--version'] and (prefix/'missing-pip').exists(): sys.exit(1)
if sys.argv[1:3] == ['-m','venv']:
    target=pathlib.Path(sys.argv[3]); (target/'bin').mkdir(parents=True)
    (target/'pyvenv.cfg').touch()
    (target/'bin/python').write_text(pathlib.Path(sys.argv[0]).read_text()); (target/'bin/python').chmod(0o755)
'''
        self.env['STUB_PYTHON'] = self.python_stub
        self.script('python', self.python_stub)
        self.script('uv', f'''#!{sys.executable}
import json, os, pathlib, sys
with open(os.environ['TEST_LOG'],'a') as f: f.write(json.dumps(['uv',*sys.argv[1:]])+'\\n')
if sys.argv[1:2] == ['venv']:
    target=pathlib.Path(sys.argv[2]); (target/'bin').mkdir(parents=True)
    (target/'pyvenv.cfg').touch()
    (target/'bin/python').write_text(os.environ['STUB_PYTHON']); (target/'bin/python').chmod(0o755)
''')
        self.script('conda', f'''#!{sys.executable}
import json, os, pathlib, sys
args=sys.argv[1:]
with open(os.environ['TEST_LOG'],'a') as f: f.write(json.dumps(['conda',*args])+'\\n')
if args[:2] == ['shell.bash','hook']:
    print('conda() {{ if [[ "$1" == activate ]]; then export CONDA_PREFIX="$2"; export PATH="$2/bin:$PATH"; else "$CONDA_EXE" "$@"; fi; }}')
else:
    prefix=pathlib.Path(args[args.index('--prefix')+1])
    if args[0]=='create' and os.environ.get('FAIL_CONDA_CREATE'): sys.exit(3)
    (prefix/'bin').mkdir(parents=True,exist_ok=True); (prefix/'conda-meta').mkdir(exist_ok=True)
    (prefix/'bin/python').write_text(os.environ['STUB_PYTHON']); (prefix/'bin/python').chmod(0o755)
    (prefix/'missing-pip').unlink(missing_ok=True)
''')
        (self.project / 'compile_renderer.sh').write_text('printf \'["build"]\\n\' >> "$TEST_LOG"\n')

    def script(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def active_conda(self, name='base'):
        prefix = self.root / name
        (prefix / 'conda-meta').mkdir(parents=True)
        (prefix / 'bin').mkdir()
        python = prefix / 'bin/python'
        python.write_text(self.python_stub)
        python.chmod(0o755)
        self.env.update(CONDA_PREFIX=str(prefix), CONDA_DEFAULT_ENV=name, CONDA_SHLVL='1',
                        CONDA_EXE=str(self.bin / 'conda'))
        return prefix

    def install(self, replies, success=True):
        result = subprocess.run(['bash', str(self.project / 'setup_install.sh')],
                                input=replies, env=self.env, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return self.commands()

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_reuse_active_conda_base_or_named_environment(self):
        for name in ('base', 'research'):
            with self.subTest(name=name):
                prefix = self.active_conda(name)
                calls = self.install('y\n')
                self.assertIn(['python', str(prefix / 'bin/python'), '-m', 'pip', 'install', '-e', '.[train]'], calls)
                self.assertFalse(any(c[0] in ('uv', 'conda') for c in calls))
                self.assertIn('conda activate', (self.project / 'init.sh').read_text())
                self.log.unlink()

    def test_decline_active_and_create_local_conda(self):
        active = self.active_conda()
        calls = self.install('n\ny\n')
        local = self.project / '.venv'
        self.assertIn(['conda','create','--yes','--prefix',str(local),'python=3.12','pip'], calls)
        self.assertIn(['python',str(local/'bin/python'),'-m','pip','install','-e','.[train]'], calls)
        self.assertFalse(any(c[:3] == ['python',str(active/'bin/python'),'-m'] for c in calls))
        self.assertIn('conda activate', (self.project / 'init.sh').read_text())

    def test_both_conda_answers_no_fall_back_to_uv(self):
        self.active_conda()
        # Reject both Conda routes, accept creation of a normal local venv.
        calls = self.install('n\nn\ny\n')
        self.assertTrue(any(c[:2] == ['uv','venv'] for c in calls))
        self.assertTrue(any(c[:3] == ['uv','pip','install'] for c in calls))
        self.assertFalse(any(c[0] == 'conda' for c in calls))
        self.assertFalse((self.project / '.heligym-conda-prefix').exists())

    def test_no_active_conda_preserves_uv_route(self):
        calls = self.install('n\ny\n')
        self.assertTrue(any(c[:2] == ['uv','venv'] for c in calls))
        self.assertFalse(any(c[0] == 'conda' for c in calls))

    def test_pip_and_venv_fallback_without_uv(self):
        (self.bin / 'uv').unlink()
        # Reject uv installation and current Python, accept venv creation.
        calls = self.install('n\nn\ny\n')
        self.assertTrue(any(c[0] == 'python' and c[2:4] == ['-m','venv'] for c in calls))
        self.assertTrue(any(c[0] == 'python' and c[2:5] == ['-m','pip','install'] for c in calls))

    def test_missing_pip_is_installed_in_selected_conda_environment(self):
        prefix = self.active_conda()
        (prefix / 'missing-pip').touch()
        calls = self.install('y\n')
        self.assertIn(['conda','install','--yes','--prefix',str(prefix),'pip'], calls)

    def test_existing_non_conda_venv_is_not_overwritten(self):
        self.active_conda()
        local = self.project / '.venv'
        local.mkdir()
        (local / 'keep').write_text('original')
        calls = self.install('n\ny\n', success=False)
        self.assertEqual((local / 'keep').read_text(), 'original')
        self.assertFalse(any(c[0] in ('uv','conda') for c in calls))

    def test_failed_conda_creation_does_not_install_or_build(self):
        self.active_conda()
        self.env['FAIL_CONDA_CREATE'] = '1'
        calls = self.install('n\ny\n', success=False)
        self.assertFalse(any(c[0] in ('python','build') for c in calls))

    def test_init_activates_selected_conda_without_inheriting_arguments(self):
        prefix = self.active_conda()
        self.install('y\n')
        command = 'init_path=$1; set -- --device cuda; source "$init_path"; [[ "$CONDA_PREFIX" == "$EXPECTED_PREFIX" && "$#" == 2 ]]'
        env = dict(self.env, EXPECTED_PREFIX=str(prefix))
        result = subprocess.run(['bash','-c',command,'test',str(self.project/'init.sh')], env=env, capture_output=True,text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn(['conda','shell.bash','hook'], self.commands())

    def test_conda_exe_fallback_when_conda_is_not_on_path(self):
        self.active_conda()
        executable = self.bin / 'conda-executable'
        (self.bin / 'conda').rename(executable)
        self.env['CONDA_EXE'] = str(executable)
        calls = self.install('n\ny\n')
        self.assertTrue(any(c[:2] == ['conda','create'] for c in calls))

    def test_reuse_existing_local_conda_environment(self):
        self.active_conda()
        local = self.project / '.venv'
        (local / 'conda-meta').mkdir(parents=True)
        (local / 'bin').mkdir()
        (local / 'bin/python').write_text(self.python_stub)
        (local / 'bin/python').chmod(0o755)
        calls = self.install('n\ny\ny\n')
        self.assertFalse(any(c[:2] == ['conda','create'] for c in calls))
        self.assertIn(['python',str(local/'bin/python'),'-m','pip','install','-e','.[train]'], calls)

    def test_missing_python_is_installed_into_the_selected_conda_prefix(self):
        prefix = self.active_conda()
        (prefix / 'bin/python').unlink()
        calls = self.install('y\n')
        self.assertIn(['conda','install','--yes','--prefix',str(prefix),'python=3.12','pip'], calls)

    def test_sourcing_installer_preserves_parent_environment_and_directory(self):
        prefix = self.active_conda()
        command = ('before_path=$PATH; before_dir=$PWD; before_prefix=$CONDA_PREFIX; '
                   'source "$1"; [[ "$PATH" == "$before_path" && "$PWD" == "$before_dir" '
                   '&& "$CONDA_PREFIX" == "$before_prefix" ]]')
        result = subprocess.run(['bash','-c',command,'test',str(self.project/'setup_install.sh')],
                                input='n\ny\n',env=self.env,capture_output=True,text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(self.env['CONDA_PREFIX'], str(prefix))

    def test_init_keeps_standard_venv_activation_and_caller_arguments(self):
        local = self.project / '.venv'
        (local / 'bin').mkdir(parents=True)
        (local / 'bin/activate').write_text('export TEST_VENV_ACTIVE=yes\n')
        command = ('init_path=$1; set -- --device cuda; source "$init_path"; '
                   '[[ "$TEST_VENV_ACTIVE" == yes && "$#" == 2 ]]')
        result = subprocess.run(['bash','-c',command,'test',str(self.project/'init.sh')],
                                env=self.env,capture_output=True,text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_existing_standard_venv_reuse_preserves_uv_workflow(self):
        local = self.project / '.venv'
        (local / 'bin').mkdir(parents=True)
        (local / 'pyvenv.cfg').touch()
        (local / 'bin/activate').write_text('export TEST_VENV_ACTIVE=yes\n')
        (local / 'bin/python').write_text(self.python_stub)
        (local / 'bin/python').chmod(0o755)
        (self.project / '.heligym-conda-prefix').write_text('stale selection\n')
        calls = self.install('y\n')
        install = next(c for c in calls if c[:3] == ['uv','pip','install'])
        self.assertEqual(install[install.index('--python')+1], str(local/'bin/python'))
        self.assertFalse((self.project / '.heligym-conda-prefix').exists())
        self.assertIn('source ', (self.project / 'init.sh').read_text())
        result = subprocess.run(['bash', '-c', 'source "$1"; [[ "$TEST_VENV_ACTIVE" == yes ]]',
                                 'test', str(self.project / 'init.sh')], env=self.env)
        self.assertEqual(result.returncode, 0)

    def test_no_conda_active_python_reuse_is_still_available(self):
        calls = self.install('y\n')
        install = next(c for c in calls if c[:3] == ['uv','pip','install'])
        self.assertIn('--system', install)
        self.assertFalse((self.project / '.venv').exists())

    def test_init_requires_sourcing_before_and_after_installation(self):
        for generated in (False, True):
            with self.subTest(generated=generated):
                if generated:
                    self.active_conda()
                    self.install('y\n')
                result = subprocess.run(['bash', str(self.project / 'init.sh')],
                                        env=self.env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn('source ./init.sh', result.stderr)

    def test_closed_input_does_not_install(self):
        self.active_conda()
        calls = self.install('')
        self.assertFalse(any(c[0] in ('uv','conda','build') for c in calls))


if __name__ == '__main__':
    unittest.main()
