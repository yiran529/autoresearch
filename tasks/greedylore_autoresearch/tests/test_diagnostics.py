"""Diagnostic commands have their own identity, budget and durable evidence."""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from autoresearch import diagnostics


@pytest.fixture
def task(tmp_path, monkeypatch):
    from autoresearch import diagnostic_sandbox
    driver = tmp_path / 'nvidia-driver' / '0000:01:00.0'
    driver.mkdir(parents=True)
    (driver / 'information').write_text('GPU UUID: GPU-test\nDevice Minor: 3\n')
    monkeypatch.setattr(diagnostic_sandbox, '_NVIDIA_DRIVER_ROOT', driver.parent, raising=False)
    item = diagnostics.create_task(tmp_path / '.coral', 'N001', 'research', 1, 'a' * 40)
    item.bind(os.getpid())
    return item


def request(request_id='request-1', **changes):
    return {'request_id': request_id, 'argv': ['python', 'experiments/probe.py'],
            'gpus': 4, 'timeout': 300, **changes}


def execute(task, payload, cancelled):
    cancelled()
    (task.workspace / 'results' / 'value.json').write_text('{"gap": 0.25}')
    return {'status': 'completed', 'started': True, 'exit_code': 0}


def test_three_runs_share_persistent_budget(task):
    for number in range(3):
        assert task.run(request(f'run-{number}'), os.getpid(), execute)['status'] == 'completed'
    reopened = diagnostics.DiagnosticTask(task.coral_dir, task.task_id)
    with pytest.raises(ValueError, match='budget'):
        reopened.run(request('fourth'), os.getpid(), execute)
    record = json.loads((task.records / 'run-0' / 'record.json').read_text())
    assert record['research_sha'] == 'a' * 40
    assert record['argv'] == ['python', 'experiments/probe.py']
    assert record['gpus'] == 4
    assert record['exit_code'] == 0
    assert (task.workspace / 'results/value.json').is_file()


def test_duplicate_requests_execute_once(task):
    def once(*args):
        marker = task.workspace / 'results' / 'only-once'
        with marker.open('x'):
            pass
        return execute(*args)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: task.run(request(), os.getpid(), once), range(2)))
    assert results[0] == results[1]


@pytest.mark.parametrize('changes', [
    {'gpus': 0}, {'gpus': 5}, {'gpus': True}, {'timeout': 601},
    {'timeout': 0}, {'timeout': float('nan')}, {'argv': []}, {'argv': 'python a.py'},
    {'argv': ['python', '\x00']}, {'workspace': '/tmp/other'}, {'request_id': '../escape'},
])
def test_rejects_invalid_requests(task, changes):
    with pytest.raises(ValueError):
        task.run(request(**changes), os.getpid(), execute)


def test_closed_or_wrong_pid_cannot_execute(task):
    with pytest.raises(PermissionError):
        task.run(request(), 1, execute)
    task.close()
    with pytest.raises(PermissionError):
        task.run(request(), os.getpid(), execute)


def test_queue_failure_records_without_consuming_execution(task):
    def unavailable(*_args):
        return {'status': 'unavailable', 'started': False, 'reason': 'no GPUs'}
    for number in range(4):
        assert task.run(request(f'wait-{number}'), os.getpid(), unavailable)['status'] == 'unavailable'
    assert task.run(request('run'), os.getpid(), execute)['status'] == 'completed'


def test_exception_is_recorded_and_not_reexecuted(task):
    def fail(*_args):
        raise RuntimeError('probe failure')
    result = task.run(request(), os.getpid(), fail)
    assert result['status'] == 'failed'
    assert 'probe failure' in result['reason']
    assert task.run(request(), os.getpid(), execute) == result


def test_task_reopen_keeps_budget_and_rejects_conflicting_sha(task):
    task.run(request(), os.getpid(), execute)
    task.close()
    resumed = diagnostics.create_task(task.coral_dir, 'N001', 'research', 1, 'a' * 40)
    resumed.bind(os.getpid())
    assert resumed.run(request(), os.getpid(), execute)['diagnostic_id'] == f'{task.task_id}/request-1'
    with pytest.raises(ValueError, match='identity'):
        diagnostics.create_task(task.coral_dir, 'N001', 'research', 1, 'b' * 40)


def test_missing_sandbox_returns_unavailable_before_allocation(task, monkeypatch):
    from autoresearch import diagnostic_sandbox
    monkeypatch.setattr(diagnostic_sandbox.shutil, 'which', lambda _: None)
    def unexpected(*_args, **_kwargs):
        pytest.fail('GPU allocation must not occur without isolation')
    monkeypatch.setattr(diagnostic_sandbox, 'reserve_idle_gpus', unexpected)
    result = task.run(request(), os.getpid(), diagnostic_sandbox.execute_diagnostic)
    assert result['status'] == 'unavailable'
    assert result['started'] is False
    assert 'bubblewrap' in result['reason']


@pytest.mark.parametrize('allow_unix', [False, True])
def test_diagnostic_analyzer_uses_configured_model_and_effort(task, monkeypatch, allow_unix):
    task.close()
    from pathlib import Path

    from autoresearch import diagnostic_session
    config = {'analyzer_model': 'gpt-6.1-sol', 'analyzer_reasoning_effort': 'medium',
              'analyzer_allow_unix_sockets': allow_unix}
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:9999')
    monkeypatch.setenv('no_proxy', 'localhost,127.0.0.1')
    monkeypatch.setenv('UNRELATED_SECRET', 'must not reach the model')
    monkeypatch.setattr(diagnostic_session, 'settings', lambda _: config)
    monkeypatch.setattr(diagnostic_session.shutil, 'which', lambda name:
                        '/opt/coral-tools/bwrap' if name == 'bwrap' else '/usr/bin/true')
    monkeypatch.setattr(diagnostic_session, 'build_sandbox', lambda *a, **k: ['true', '--', 'true'])
    schema = task.workspace / 'input.schema.json'
    schema.write_text('{}')
    code_root = task.workspace / 'code/fixed-sha'
    code_root.mkdir(parents=True)

    class Process:
        pid = os.getpid()
        returncode = 0

        def __init__(self, command, **kwargs):
            self.command = command
            # Codex's own tool sandbox needs the isolation binary inside the outer namespace.
            assert '/opt/coral-tools' in kwargs['env']['PATH'].split(':')
            assert kwargs['env']['HTTPS_PROXY'] == 'http://127.0.0.1:9999'
            assert kwargs['env']['no_proxy'] == 'localhost,127.0.0.1'
            assert 'UNRELATED_SECRET' not in kwargs['env']

        def communicate(self, prompt):
            # The diagnostic child needs the same selected code/evidence as Codex.
            assert str(code_root) in task.read()['runtime']['readonly_paths']
            command = self.command
            assert command[command.index('--cd') + 1] == str(task.workspace / 'results')
            assert command[command.index('--add-dir') + 1] == str(task.workspace / 'experiments')
            assert f'Workspace root: {task.workspace}' in prompt
            assert 'resolved relative to the workspace root' in prompt
            assert command[command.index('--model') + 1] == 'gpt-6.1-sol'
            assert 'model_reasoning_effort="medium"' in command
            assert ('features.network_proxy.dangerously_allow_all_unix_sockets=true' in command) == allow_unix
            if allow_unix:
                assert 'sandbox_workspace_write.network_access=true' in command
                assert 'features.network_proxy.enabled=true' in command
                assert 'features.network_proxy.domains={}' in command
            assert 'danger-full-access' not in command
            Path(command[command.index('--output-last-message') + 1]).write_text('{}')
            return '', ''

        def poll(self):
            return 0

    monkeypatch.setattr(diagnostic_session.subprocess, 'Popen', Process)
    output = diagnostic_session.run_diagnostic_codex('analyze', [code_root], schema, task.workspace, task)
    assert output.read_text() == '{}'
    assert task.read()['process'] is None


def test_sandbox_exposes_only_selected_devices_and_writable_subdirectories(task):
    from autoresearch.diagnostic_sandbox import build_sandbox
    command = build_sandbox(task.workspace, [], [2, 5], ['python', 'experiments/p.py'])
    assert '--unshare-all' in command
    assert '--share-net' not in command
    binds = [command[index + 1:index + 3] for index, arg in enumerate(command)
             if arg == '--dev-bind']
    assert ['/dev/nvidia2', '/dev/nvidia2'] in binds
    assert ['/dev/nvidia5', '/dev/nvidia5'] in binds
    assert ['/dev/nvidia0', '/dev/nvidia0'] not in binds
    writable = [command[index + 1] for index, arg in enumerate(command) if arg == '--bind']
    assert writable == [str(task.workspace / 'experiments'), str(task.workspace / 'results')]


def test_only_host_selected_checkpoint_can_cross_task_boundary(task):
    from autoresearch.diagnostic_sandbox import build_sandbox

    selected = task.coral_dir / 'public/eval_logs/sha/run/checkpoints/llama60m_adamw'
    other = selected.parent / 'llama130m_muon'
    selected.mkdir(parents=True)
    other.mkdir()
    command = build_sandbox(task.workspace, [selected], [], ['true'],
                            checkpoint_paths=[selected])
    mounts = [command[i + 1:i + 3] for i, arg in enumerate(command) if arg == '--ro-bind']
    assert [str(selected), str(selected)] in mounts
    assert str(other) not in command
    for path in (other, selected.parents[2], task.coral_dir / 'private'):
        path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError):
            build_sandbox(task.workspace, [path], [], ['true'], checkpoint_paths=[selected])


def test_checkpoint_symlink_mount_is_rejected(task, tmp_path):
    from autoresearch.diagnostic_sandbox import build_sandbox

    root = task.coral_dir / 'public/eval_logs/sha/run/checkpoints'
    root.mkdir(parents=True)
    selected = root / 'llama60m_adamw'
    selected.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        build_sandbox(task.workspace, [selected], [], ['true'], checkpoint_paths=[selected])


def test_real_checkpoint_mount_is_readonly_and_other_runs_hidden(task):
    import subprocess

    from autoresearch.diagnostic_sandbox import build_sandbox, isolation_available

    if not isolation_available():
        pytest.skip('bubblewrap namespaces unavailable')
    root = task.coral_dir / 'public/eval_logs/sha/run/checkpoints/llama60m_adamw'
    root.mkdir(parents=True)
    tensor = root / 'model.pt'
    tensor.write_text('fixed state')
    sibling = task.coral_dir / 'public/eval_logs/other-run/model.pt'
    sibling.parent.mkdir(parents=True)
    sibling.write_text('must remain hidden')
    program = (
        'from pathlib import Path\nimport errno\n'
        f'p=Path({str(tensor)!r})\n'
        'assert p.read_text() == "fixed state"\n'
        'try:\n p.write_text("changed")\n'
        'except OSError as exc:\n assert exc.errno in (errno.EROFS, errno.EACCES)\n'
        'else:\n raise AssertionError("checkpoint writable")\n'
        f'assert not Path({str(sibling)!r}).exists()\n'
        f'Path({str(task.workspace / "results/passed")!r}).write_text("readonly")\n'
    )
    command = build_sandbox(task.workspace, [root], [], ['/usr/bin/python3', '-c', program],
                            checkpoint_paths=[root])
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert tensor.read_text() == 'fixed state'
    assert (task.workspace / 'results/passed').read_text() == 'readonly'


def test_diagnostic_maps_reserved_uuids_to_device_minors(task, tmp_path):
    from autoresearch import diagnostic_sandbox
    driver = tmp_path / 'driver'
    for bus, uuid, minor in [('bus-a', 'GPU-first', 3), ('bus-b', 'GPU-second', 0)]:
        entry = driver / bus
        entry.mkdir(parents=True)
        (entry / 'information').write_text(f'GPU UUID: {uuid}\nDevice Minor: {minor}\n')
    assert diagnostic_sandbox.device_minors_for_uuids(
        ['GPU-second', 'GPU-first'], driver_root=driver) == [0, 3]
    with pytest.raises(ValueError, match='GPU-missing'):
        diagnostic_sandbox.device_minors_for_uuids(['GPU-missing'], driver_root=driver)


@pytest.mark.parametrize('minor', ['-1', 'not-a-number', ''])
def test_diagnostic_rejects_invalid_device_minor(task, tmp_path, minor):
    from autoresearch import diagnostic_sandbox
    driver = tmp_path / 'driver' / 'bus'
    driver.mkdir(parents=True)
    (driver / 'information').write_text(f'GPU UUID: GPU-test\nDevice Minor: {minor}\n')
    with pytest.raises(ValueError, match='device mapping'):
        diagnostic_sandbox.device_minors_for_uuids(['GPU-test'], driver_root=driver.parent)


@pytest.mark.parametrize('uuids,minors,requested', [
    (['GPU-test', 'GPU-test'], [0, 3], ['GPU-test']),
    (['GPU-test', 'GPU-other'], [3, 3], ['GPU-test', 'GPU-other']),
])
def test_diagnostic_rejects_ambiguous_device_mapping(task, tmp_path, uuids, minors, requested):
    from autoresearch import diagnostic_sandbox
    driver = tmp_path / 'driver'
    for number, (gpu_uuid, minor) in enumerate(zip(uuids, minors, strict=True)):
        entry = driver / str(number)
        entry.mkdir(parents=True)
        (entry / 'information').write_text(f'GPU UUID: {gpu_uuid}\nDevice Minor: {minor}\n')
    with pytest.raises(ValueError, match='device mapping'):
        diagnostic_sandbox.device_minors_for_uuids(requested, driver_root=driver)


def test_missing_device_mapping_releases_reservation_without_charging(task, monkeypatch):
    from autoresearch import diagnostic_sandbox as sandbox
    released = task.workspace / 'results/released'
    class Reservation:
        indices = (0,)
        uuids = ('GPU-unmapped',)
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            released.touch()
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/unused/bwrap')
    monkeypatch.setattr(sandbox, 'reserve_idle_gpus', lambda *_args, **_kwargs: Reservation())
    result = task.run(request(), os.getpid(), sandbox.execute_diagnostic)
    assert result['status'] == 'unavailable'
    assert result['started'] is False
    assert 'GPU-unmapped' in result['reason']
    assert task.read()['executions'] == 0
    assert released.exists()


@pytest.mark.parametrize('outcome', ['success', 'nonzero', 'missing_output', 'launch_error', 'sandbox_error'])
def test_analyzer_cleans_home_but_preserves_evidence(task, tmp_path, monkeypatch, outcome):
    import subprocess
    import sys
    from pathlib import Path

    from autoresearch import diagnostic_session

    task.close()
    auth_dir = tmp_path / 'host-codex'
    auth_dir.mkdir()
    auth = auth_dir / 'auth.json'
    auth.write_text('host credentials must survive')
    monkeypatch.setenv('CODEX_HOME', str(auth_dir))
    monkeypatch.setattr(diagnostic_session, 'settings', lambda _: {})
    monkeypatch.setattr(diagnostic_session.shutil, 'which', lambda _: '/usr/bin/true')

    def sandbox(*args, **kwargs):
        if outcome == 'sandbox_error':
            raise RuntimeError('sandbox setup failed')
        return ['true', '--', 'true']

    monkeypatch.setattr(diagnostic_session, 'build_sandbox', sandbox)
    schema = task.workspace / 'input.schema.json'
    schema.write_text('{}')
    task.records.mkdir(exist_ok=True)
    evidence = task.records / 'evidence.txt'
    evidence.write_text('diagnostic evidence')
    real_popen = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        home = Path(kwargs['env']['CODEX_HOME'])
        cache = home / 'plugins/cache'
        cache.mkdir(parents=True)
        (cache / 'asset').write_bytes(b'x' * 1024)
        (cache / 'host-link').symlink_to(auth_dir, target_is_directory=True)
        if outcome == 'launch_error':
            raise OSError('launch failed')
        script = '''
import pathlib, sys
sys.stdin.read()
print('session output')
print('session error', file=sys.stderr)
if sys.argv[1] == 'success':
    pathlib.Path(sys.argv[2]).write_text('{}')
sys.exit(1 if sys.argv[1] == 'nonzero' else 0)
'''
        output = command[command.index('--output-last-message') + 1]
        process = real_popen([sys.executable, '-c', script, outcome, output], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(diagnostic_session.subprocess, 'Popen', launch)
    if outcome == 'success':
        output = diagnostic_session.run_diagnostic_codex('analyze', [], schema, task.workspace, task)
        assert output.read_text() == '{}'
    else:
        message = {'nonzero': 'session error', 'missing_output': 'output document',
                   'launch_error': 'launch failed', 'sandbox_error': 'sandbox setup failed'}[outcome]
        with pytest.raises((RuntimeError, OSError), match=message):
            diagnostic_session.run_diagnostic_codex('analyze', [], schema, task.workspace, task)
    assert not list(task.private.glob('codex-home*'))
    assert auth.read_text() == 'host credentials must survive'
    assert evidence.read_text() == 'diagnostic evidence'
    assert task.read()['process'] is None
    assert all(process.poll() is not None for process in processes)
    if outcome in {'success', 'nonzero', 'missing_output'}:
        assert (task.private / 'session.stdout.log').read_text() == 'session output\n'
        assert (task.private / 'session.stderr.log').read_text() == 'session error\n'


def test_sandbox_resolves_host_binary_before_sanitizing_child_path(task, monkeypatch):
    from autoresearch import diagnostic_sandbox
    monkeypatch.setattr(diagnostic_sandbox.shutil, 'which', lambda _: '/opt/tools/bwrap')
    command = diagnostic_sandbox.build_sandbox(task.workspace, [], [], ['true'])
    assert command[0] == '/opt/tools/bwrap'


def test_sandbox_refuses_symlink_workspace(task, tmp_path):
    from autoresearch.diagnostic_sandbox import build_sandbox
    outside = tmp_path / 'outside'
    outside.mkdir()
    (task.workspace / 'results').rmdir()
    (task.workspace / 'results').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        build_sandbox(task.workspace, [], [0], ['true'])


def test_private_runtime_mounts_are_rejected(task):
    from autoresearch.diagnostic_sandbox import build_sandbox
    with pytest.raises(ValueError, match='private'):
        build_sandbox(task.workspace, [task.coral_dir / 'private'], [0], ['true'])


def test_runtime_mount_cannot_expose_run_ancestor_or_another_task(task):
    from autoresearch.diagnostic_sandbox import build_sandbox
    for path in (task.coral_dir.parent, task.coral_dir / 'public/diagnostics/another-task'):
        path.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match='private|broad|another'):
            build_sandbox(task.workspace, [path], [0], ['true'])


def test_ordinary_analysis_does_not_require_private_grader_package():
    import subprocess
    import sys
    from pathlib import Path

    script = '''
import sys, importlib.abc
class RejectPrivateGrader(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('greedylore_autoresearch_grader'):
            raise RuntimeError('private grader is not installed in producer environment')
sys.meta_path.insert(0, RejectPrivateGrader())
from autoresearch.diagnostic_session import settings
print('ordinary analysis dependencies available')
'''
    env = {**os.environ, 'PYTHONPATH': str(Path(diagnostics.__file__).parents[1])}
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'ordinary analysis dependencies available' in result.stdout


def test_real_cpu_sandbox_blocks_host_and_evidence_writes(task, tmp_path):
    import shutil
    import subprocess

    from autoresearch.diagnostic_sandbox import build_sandbox
    if not shutil.which('bwrap'):
        pytest.skip('bubblewrap is not installed; actual namespace isolation is unverified')
    sentinel = tmp_path / 'host-secret'
    sentinel.write_text('secret')
    evidence = task.workspace / 'evidence'
    evidence.mkdir()
    (evidence / 'fixed').write_text('immutable')
    script = task.workspace / 'experiments' / 'probe.sh'
    script.write_text(f'! cat {sentinel}\n! echo modified > evidence/fixed\necho ok > results/result\n')
    completed = subprocess.run(build_sandbox(task.workspace, [], [], ['sh', 'experiments/probe.sh']),
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (evidence / 'fixed').read_text() == 'immutable'
    assert (task.workspace / 'results/result').read_text() == 'ok\n'


def test_trusted_dispatch_uses_analysis_identity_not_producer_identity(task):
    from types import SimpleNamespace

    from coral.grader.trusted_writes import TrustedWriteError, TrustedWriteService
    def handler(payload, context):
        return diagnostics.handle_diagnostic(payload, context, executor=execute)
    service = TrustedWriteService(task.coral_dir, SimpleNamespace(), diagnostic_handler=handler)
    result = service.handle('run_diagnostic', request(), os.getpid())
    assert result['status'] == 'completed'
    assert (task.workspace / 'results/value.json').read_text() == '{"gap": 0.25}'
    with pytest.raises((TrustedWriteError, PermissionError)):
        service.handle('run_diagnostic', request('stranger'), 1)


def test_cli_diagnostic_preserves_argv_and_returns_record(task, monkeypatch, capsys):
    from autoresearch import cli
    monkeypatch.setenv('CORAL_DIAGNOSTIC_RUN', str(task.coral_dir))
    def channel(coral_dir, operation, payload, **kwargs):
        assert operation == 'run_diagnostic'
        assert payload['argv'] == ['python', 'experiments/probe.py', '--output', 'results/my result.json']
        assert kwargs['timeout'] >= 4800
        assert payload['timeout'] == 600
        return task.run(payload, os.getpid(), execute)
    monkeypatch.setattr(cli, 'request_trusted', channel)
    cli.main(['diagnostic', 'run', '--gpus', '4', '--',
              'python', 'experiments/probe.py', '--output', 'results/my result.json'])
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'


def test_another_session_cannot_revoke_live_analysis_identity(task):
    with pytest.raises(PermissionError):
        with task.session():
            pytest.fail('must not enter an already active analysis')
    assert task.run(request(), os.getpid(), execute)['status'] == 'completed'


def test_task_evidence_cannot_be_replaced_by_retry(tmp_path):
    task = diagnostics.create_task(tmp_path / '.coral', 'N001', 'engineering', 1,
                                   'a' * 40, evidence_hash='old')
    with pytest.raises(ValueError, match='identity'):
        diagnostics.create_task(task.coral_dir, 'N001', 'engineering', 1,
                                'a' * 40, evidence_hash='new')


@pytest.mark.parametrize('sharing', [None, False, True])
def test_timeout_cleans_process_and_releases_reservation(task, monkeypatch, sharing):
    import sys

    from autoresearch import diagnostic_sandbox as sandbox
    if sharing is not None:
        (task.coral_dir / 'config.yaml').write_text(json.dumps({
            'grader': {'args': {'pretraining_allow_other_users': sharing}},
        }))
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:9999')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only-never-forward')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-host-unassigned')
    released = task.workspace / 'results/released'
    checked_exclusive = task.workspace / 'results/checked-exclusive'
    class Reservation:
        indices = (0,)
        uuids = ('GPU-test',)
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            released.write_text('released')
        def check_exclusive(self, pid):
            if sharing is True:
                raise RuntimeError('timing-only exclusivity check rejects shared diagnostics')
            checked_exclusive.touch()
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/unused/bwrap')
    def reserve(*_args, **kwargs):
        assert kwargs.get('allow_other_users', False) is (sharing is True)
        assert kwargs['wait_timeout'] >= 3600
        assert kwargs['source_role'] == 'analyzer'
        assert kwargs['execution_group'] == str(task.coral_dir)
        assert kwargs['execution_limit'] == 1
        return Reservation()
    monkeypatch.setattr(sandbox, 'reserve_idle_gpus', reserve)
    # Only namespace/GPU dependencies are replaced; run a real process and timeout.
    def cpu_sandbox(_ws, _roots, devices, argv, **_kwargs):
        # NVML index 0 is deliberately not /dev/nvidia0 in this fixture.
        assert list(devices) in ([], [3])
        return argv
    monkeypatch.setattr(sandbox, 'build_sandbox', cpu_sandbox)
    script = task.workspace / 'experiments' / 'slow.py'
    script.write_text('import os,time\nprint("started", flush=True)\n'
                      'print("cuda_visibility", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)\n'
                      'print("model_environment_leaked", any(k in os.environ for k in '
                      '("HTTPS_PROXY", "OPENAI_API_KEY")), flush=True)\ntime.sleep(30)\n')
    result = task.run(request(argv=[sys.executable, str(script)], timeout=0.2),
                      os.getpid(), sandbox.execute_diagnostic)
    assert result['status'] == 'timed_out'
    assert released.read_text() == 'released'
    assert checked_exclusive.exists() is (sharing is not True)
    assert 'started' in (task.records / 'request-1/stdout.log').read_text()
    assert 'model_environment_leaked False' in (task.records / 'request-1/stdout.log').read_text()
    assert 'cuda_visibility GPU-test' in (task.records / 'request-1/stdout.log').read_text()
    assert (task.records / 'request-1/scripts/slow.py').read_text() == script.read_text()


def test_real_socket_and_standalone_client_record_one_execution(task):
    import subprocess
    import sys
    from types import SimpleNamespace

    from autoresearch import diagnostic_client

    from coral.grader.trusted_writes import TrustedWriteServer, TrustedWriteService
    def handler(payload, context):
        return diagnostics.handle_diagnostic(payload, context, executor=execute)
    service = TrustedWriteService(task.coral_dir, SimpleNamespace(), diagnostic_handler=handler)
    server = TrustedWriteServer(task.coral_dir, service)
    server.start()
    try:
        command = [sys.executable, diagnostic_client.__file__, 'diagnostic', 'run',
                   '--gpus', '4', '--timeout', '300', '--request-id', 'same-request', '--',
                   'python', 'experiments/probe.py']
        env = {**os.environ, 'CORAL_DIAGNOSTIC_RUN': str(task.coral_dir)}
        first = subprocess.run(command, capture_output=True, text=True, env=env, timeout=5)
        assert first.returncode == 0, first.stderr
        second = subprocess.run(command, capture_output=True, text=True, env=env, timeout=5)
        assert second.returncode == 0, second.stderr
        assert json.loads(first.stdout) == json.loads(second.stdout)
        assert task.read()['executions'] == 1
        record = json.loads((task.records / 'same-request/record.json').read_text())
        assert record['gpus'] == 4
        assert record['argv'] == ['python', 'experiments/probe.py']
    finally:
        server.stop()


def test_archive_never_follows_a_replaced_parent_directory(task, tmp_path, monkeypatch):
    from autoresearch.diagnostic_sandbox import snapshot_files
    source = task.workspace / 'experiments'
    nested = source / 'nested'
    nested.mkdir()
    (nested / 'probe.txt').write_text('public script')
    secret = tmp_path / 'secret'
    secret.mkdir()
    (secret / 'probe.txt').write_text('private credential')
    original = os.open
    swapped = False
    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if str(path).endswith('probe.txt') and not swapped:
            swapped = True
            nested.rename(source / 'moved')
            nested.symlink_to(secret, target_is_directory=True)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, 'open', racing_open)
    target = tmp_path / 'archive'
    snapshot_files(source, target)
    assert (target / 'nested/probe.txt').read_text() == 'public script'


def test_archive_limits_actual_bytes_not_only_initial_stat(task, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from autoresearch.diagnostic_sandbox import snapshot_files
    (task.workspace / 'experiments/probe').write_text('0123456789')
    original = os.fstat
    def small_stat(fd):
        value = original(fd)
        return SimpleNamespace(st_mode=value.st_mode, st_size=1)
    monkeypatch.setattr(os, 'fstat', small_stat)
    with pytest.raises(ValueError, match='budget'):
        snapshot_files(task.workspace / 'experiments', tmp_path / 'archive', limit=5)


def test_execution_reads_frozen_scripts_even_if_analyzer_changes_original(task, monkeypatch):
    import sys

    from autoresearch import diagnostic_sandbox as sandbox
    script = task.workspace / 'experiments/probe.py'
    source = 'from pathlib import Path\nprint("frozen source")\nPath("results/value").write_text("measured")\n'
    script.write_text(source)
    class Reservation:
        indices = (0,)
        uuids = ('GPU-test',)
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def check_exclusive(self, pid):
            pass
    original_build = sandbox.build_sandbox
    def cpu_command(workspace, roots, devices, argv, *, write_sources=None, **kwargs):
        original_build(workspace, roots, devices, argv, write_sources=write_sources, **kwargs)
        if write_sources:
            script.write_text('print("changed live source")\n')
            return [sys.executable, '-c',
                    f'import os, runpy; os.chdir({str(write_sources["experiments"].parent)!r}); '
                    'runpy.run_path("experiments/probe.py")']
        return argv
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/unused/bwrap')
    monkeypatch.setattr(sandbox, 'reserve_idle_gpus', lambda *_args, **_kwargs: Reservation())
    monkeypatch.setattr(sandbox, 'build_sandbox', cpu_command)
    result = task.run(request(argv=[sys.executable, 'experiments/probe.py']),
                      os.getpid(), sandbox.execute_diagnostic)
    assert result['status'] == 'completed'
    assert (task.records / 'request-1/stdout.log').read_text() == 'frozen source\n'
    assert (task.records / 'request-1/scripts/probe.py').read_text() == source
    assert (task.records / 'request-1/results/value').read_text() == 'measured'
    assert (task.workspace / 'results/value').read_text() == 'measured'


def test_snapshot_preserves_executable_scripts(task, tmp_path):
    import subprocess

    from autoresearch.diagnostic_sandbox import snapshot_files
    script = task.workspace / 'experiments' / 'probe.sh'
    script.write_text('#!/bin/sh\nprintf executable')
    script.chmod(0o700)
    snapshot_files(script.parent, tmp_path / 'archive')
    result = subprocess.run([str(tmp_path / 'archive/probe.sh')], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout == 'executable'


def test_unusable_namespace_backend_is_unavailable(monkeypatch):
    from autoresearch import diagnostic_sandbox
    monkeypatch.setattr(diagnostic_sandbox.shutil, 'which', lambda _: '/usr/bin/false')
    assert diagnostic_sandbox.isolation_available() is False


def test_configured_handler_runs_in_validator_process_without_gpu(task, monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from coral.grader.trusted_writes import TrustedWriteService
    source = Path(diagnostics.__file__).parents[1]
    grader = source.parents[1] / 'grader' / 'src'
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join((str(source), str(grader))))
    monkeypatch.setenv('PATH', '/nonexistent')  # missing isolation backend, never allocate GPUs
    monkeypatch.setattr('coral.grader.trusted_worker.grader_python_path', lambda _: Path(sys.executable))
    config = SimpleNamespace(grader=SimpleNamespace(args={
        'trusted_diagnostic_handler': 'autoresearch.diagnostics:handle_diagnostic'}))
    service = TrustedWriteService(task.coral_dir, config)
    result = service.handle('run_diagnostic', request(), os.getpid())
    assert result['status'] == 'unavailable'
    assert result['started'] is False
    assert task.read()['executions'] == 0
    assert (task.records / 'request-1/record.json').is_file()


def test_ten_minute_diagnostic_is_accepted(task):
    assert task.run(request(timeout=600), os.getpid(), execute)["status"] == "completed"
