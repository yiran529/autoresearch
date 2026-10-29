"""Real namespaces/socket/CLI; only GPU allocation is replaced with a CPU fixture."""
# ruff: noqa: S101

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from autoresearch import diagnostic_client, diagnostic_sandbox, diagnostics
from autoresearch.artifact_paths import diagnostic_view
from test_diagnostics import task as task

from coral.grader.trusted_writes import TrustedWriteServer, TrustedWriteService


def test_analyzer_can_read_failure_correct_script_and_run_again(task, tmp_path, monkeypatch):
    if not diagnostic_sandbox.isolation_available():
        pytest.skip('real bubblewrap namespaces are unavailable')
    code = task.workspace / 'code' / ('a' * 40)
    package = code / 'c4'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('VALUE = "fixed candidate"\n')
    tools = task.workspace / 'tools'
    tools.mkdir()
    task.records.mkdir()
    shutil.copyfile(diagnostic_client.__file__, tools / 'autoresearch')
    from coral.grader import trusted_protocol
    shutil.copyfile(trusted_protocol.__file__, tools / 'trusted_protocol.py')
    secret = tmp_path / 'host-secret'
    secret.write_text('not an experiment input')
    checkpoint = task.coral_dir / 'public/eval_logs/attempt/run/checkpoints/model'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'model.pt').write_text('matching checkpoint')
    checkpoint_alias = diagnostic_view(task.workspace, checkpoint_paths=[checkpoint]).path(checkpoint)
    from autoresearch.runtime_resources import resolve_resources
    data, target = tmp_path / 'data', tmp_path / 'target'
    data.mkdir()
    target.mkdir()
    (target / 'shard').write_text('original data stream')
    (data / 'shard').symlink_to(target / 'shard')
    resources = resolve_resources([{'id': 'c4', 'path': str(data),
        'target_roots': [str(target)], 'view': '/resources/c4'}])
    state = task.read()
    state['checkpoint_paths'] = [str(checkpoint)]
    state['runtime'] = {'readonly_paths': [str(code), str(checkpoint)], 'path': '/usr/bin:/bin',
                        'resources': resources}
    diagnostics._atomic_json(task.path, state)

    class CpuReservation:
        indices = (0,)
        uuids = ('GPU-fixture',)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def check_exclusive(self, _):
            pass

    monkeypatch.setattr(diagnostic_sandbox, 'reserve_idle_gpus', lambda *_, **__: CpuReservation())
    monkeypatch.setattr(diagnostic_sandbox, 'device_minors_for_uuids', lambda _: [])
    controller = r'''
import errno, json, subprocess
from pathlib import Path
root = Path('/workspace')
sha = __SHA__
assert not Path(__HOST_SECRET__).exists()
assert (root/'code'/sha/'c4/__init__.py').read_text() == 'VALUE = "fixed candidate"\n'
assert (Path(__CHECKPOINT__)/'model.pt').read_text() == 'matching checkpoint'
script = root/'experiments/probe.py'
prefix = ('import sys\nfrom pathlib import Path\nsys.path.insert(0, '
          + repr('/workspace/code/'+sha) + ')\nfrom c4 import VALUE\nprint(VALUE, flush=True)\n')
prefix += 'assert (Path('+repr(__CHECKPOINT__)+')/"model.pt").read_text() == "matching checkpoint"\n'
prefix += 'assert Path("/resources/c4/shard").read_text() == "original data stream"\n'
prefix += 'assert Path('+repr(__DATA__)+').read_text() == "original data stream"\n'
script.write_text(prefix + 'print("x"*12000)\nprint(VALUE, flush=True)\n'
                  + 'print("y"*12000, file=sys.stderr)\nraise ValueError("probe needs correction")\n')
def run(request):
    result = subprocess.run(['/usr/bin/python3', '/workspace/tools/autoresearch',
        'diagnostic', 'run', '--gpus', '1', '--request-id', request, '--',
        '/usr/bin/python3', '/workspace/experiments/probe.py'],
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)
failed = run('failed-probe')
assert failed['status'] == 'failed', failed
assert failed['record_ref'] == 'diagnostics/N001-research-v1/records/failed-probe/record.json'
assert 'fixed candidate' in failed['stdout_tail'], failed
assert 'ValueError: probe needs correction' in failed['stderr_tail'], failed
assert 'probe needs correction' in failed['reason'], failed
assert len(failed['stdout_tail'].encode()) <= 4096
assert len(failed['stderr_tail'].encode()) <= 4096
for field in ('record_path', 'stdout_path', 'stderr_path'):
    assert failed[field].startswith('/workspace/records/failed-probe/'), failed
    assert Path(failed[field]).is_file(), failed
assert 'probe needs correction' in Path(failed['stderr_path']).read_text()
assert run('failed-probe') == failed
try:
    Path(failed['record_path']).write_text('modified')
except OSError as exc:
    assert exc.errno in (errno.EROFS, errno.EACCES)
else:
    raise AssertionError('diagnostic record is writable')
assert not list(Path('/dev').glob('nvidia*'))
script.write_text(prefix + 'Path("/workspace/results/value.json").write_text(VALUE)\n')
fixed = run('fixed-probe')
assert fixed['status'] == 'completed', fixed
assert (Path(fixed['results_path'])/'value.json').read_text() == 'fixed candidate'
assert (root/'results/value.json').read_text() == 'fixed candidate'
(root/'results/roundtrip.json').write_text(json.dumps({'failed': failed, 'fixed': fixed}))
'''.replace('__HOST_SECRET__', repr(str(secret))).replace('__SHA__', repr('a' * 40)).replace(
        '__CHECKPOINT__', repr(str(checkpoint_alias))).replace('__DATA__', repr(str(data / 'shard')))
    (task.workspace / 'experiments/controller.py').write_text(controller)
    bwrap = shutil.which('bwrap')
    inner = [bwrap, '--die-with-parent', '--unshare-all', '--share-net', '--ro-bind', '/', '/',
             '--bind', '/workspace/experiments', '/workspace/experiments',
             '--bind', '/workspace/results', '/workspace/results',
             '--tmpfs', '/tmp', '--tmpfs', '/dev/shm',
             '--ro-bind', str(task.coral_dir / 'public/trusted.sock'),
             str(task.coral_dir / 'public/trusted.sock'), '--',
             '/usr/bin/python3', '/workspace/experiments/controller.py']
    service = TrustedWriteService(task.coral_dir, SimpleNamespace(),
        diagnostic_handler=lambda payload, context: diagnostics.handle_diagnostic(payload, context))
    server = TrustedWriteServer(task.coral_dir, service)
    server.start()
    try:
        command = diagnostic_sandbox.build_sandbox(task.workspace,
            [task.coral_dir / 'public/trusted.sock', Path(bwrap).resolve().parent, checkpoint], [], inner,
            network=True, checkpoint_paths=[checkpoint], coral_dir=str(task.coral_dir))
        environment = {'PATH': '/usr/bin:/bin', 'CORAL_DIAGNOSTIC_RUN': str(task.coral_dir)}
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=50)
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.stop()
    report = json.loads((task.workspace / 'results/roundtrip.json').read_text())
    assert report['fixed']['exit_code'] == 0
    assert task.read()['executions'] == 2
    assert len(list(task.records.glob('*/record.json'))) == 2
    assert 'raise ValueError' in (task.records / 'failed-probe/scripts/probe.py').read_text()
    assert 'raise ValueError' not in (task.records / 'fixed-probe/scripts/probe.py').read_text()
