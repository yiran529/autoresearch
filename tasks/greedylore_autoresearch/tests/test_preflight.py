"""CPU-only immutable preflight and real namespace regressions."""
# ruff: noqa: S101
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from autoresearch.diagnostic_sandbox import build_sandbox
from test_submission import _git, _repo, _submission, _write_node_spec


def workspace(root):
    for name in ('experiments', 'results', 'records'):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def test_real_localhost_and_loopback(tmp_path):
    code = ('import socket; assert socket.gethostbyname("localhost")=="127.0.0.1"; '
            's=socket.socket(); s.bind(("localhost",0)); s.listen(); '
            'c=socket.create_connection(s.getsockname()); a,_=s.accept(); '
            'c.sendall(b"ok"); assert a.recv(2)==b"ok"')
    result = subprocess.run(build_sandbox(workspace(tmp_path / 'workspace'), [], [],
                            ['/usr/bin/python3', '-c', code]), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_explicit_hosts_mount_is_preserved(tmp_path):
    command = build_sandbox(workspace(tmp_path), ['/etc/hosts'], [], ['true'], network=True)
    bindings = [command[i + 1:i + 3] for i, value in enumerate(command) if value == '--ro-bind']
    assert [pair for pair in bindings if pair[1] == '/etc/hosts'] == [['/etc/hosts', '/etc/hosts']]


@pytest.fixture
def candidate(tmp_path):
    repo, base = _repo(tmp_path)
    for name in ('c4/run_llama_pretraining.py', 'c4/table_v_timing.py'):
        path = repo / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('raise RuntimeError("candidate must never be imported")\n')
    (repo / 'submission.yaml').write_text(yaml.safe_dump(_submission(base)))
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-m', 'candidate')
    coral = tmp_path / '.coral'
    _write_node_spec(coral, base)
    (coral / 'private').mkdir()
    raw = {'schema_version': 1, 'recovery': {'max_infra_retries': 2}, 'data': {}}
    declarations = []
    for key in ('c4_path', 'tokenizer_path'):
        data = tmp_path / key
        data.mkdir()
        (data / 'sample').write_text('readable')
        raw['data'][key] = str(data)
        declarations.append({'id': key, 'path': str(data)})
    (coral / 'private/eval_contract.yaml').write_text(yaml.safe_dump(raw))
    (coral / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
        'staged_research': True, 'diagnostic_resources': declarations}}}))
    state = {'version': 2, 'revision': 0, 'next_node_seq': 24, 'next_engineer_seq': 1,
             'next_ready_seq': 1, 'engineering_workers': {}, 'nodes': {'N023': {'research': {'agent_id': 'research-N023', 'status': 'active',
                                          'protocol': 'staged-v2'}, 'engineering_claims': []}}}
    state['nodes']['N000'] = {'research': {}}
    (coral / 'private/autoresearch_state.json').write_text(json.dumps(state))
    context = {'coral_dir': str(coral), 'worktree': str(repo), 'agent_id': 'research-N023',
               'island_id': None, 'is_manager': False}
    return repo, _git(repo, 'rev-parse', 'HEAD'), context


def check(candidate):
    from autoresearch.state import handle_workflow
    _, sha, context = candidate
    return handle_workflow('preflight_assignment', {'sha': sha}, context)


def test_preflight_real_namespace_is_readonly_and_credential_free(candidate, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-cross')
    coral = Path(candidate[2]['coral_dir'])
    before = {str(p): p.read_bytes() for p in coral.rglob('*') if p.is_file()}
    result = check(candidate)
    assert result['status'] == 'passed', result
    assert result['candidate_sha'] == candidate[1]
    assert len(result['contract_hash']) == 64
    assert result['entry_sources'][0]['source'].startswith('git:')
    assert 'static' in result['scope'] and 'semantic' in result['scope']
    assert [p['phase'] for p in result['checks']] == ['metadata', 'ownership', 'protocol_assets', 'resources', 'entry', 'namespace']
    assert before == {str(p): p.read_bytes() for p in coral.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure,phase', [('metadata','metadata'), ('entry','entry'),
    ('data','resources'), ('protocol','protocol_assets'), ('owner','ownership')])
def test_preflight_precise_failures(candidate, failure, phase, monkeypatch):
    repo, sha, context = candidate
    coral = Path(context['coral_dir'])
    if failure in ('metadata', 'entry'):
        if failure == 'metadata':
            (repo / 'submission.yaml').write_text('mode: wrong\n')
        else:
            (repo / 'c4/table_v_timing.py').unlink()
        _git(repo, 'add', '-A')
        _git(repo, 'commit', '-m', failure)
        sha = _git(repo, 'rev-parse', 'HEAD')
        # Preserve provenance while changing the immediate candidate parent.
        spec_path = coral / 'public/notes/nodes/N023/spec.md'
        spec = yaml.safe_load(spec_path.read_text())
        spec['parent_commit'] = _git(repo, 'rev-parse', 'HEAD^')
        spec_path.write_text(yaml.safe_dump(spec))
    elif failure == 'data':
        (coral.parent / 'c4_path/sample').unlink()
        (coral.parent / 'c4_path').rmdir()
    elif failure == 'protocol':
        # Selected legacy assets are absent here even in compatibility snapshots.
        monkeypatch.setattr('autoresearch.preflight._TASK_ROOT', coral.parent / 'missing-task-assets')
        state_path = coral / 'private/autoresearch_state.json'
        state = json.loads(state_path.read_text())
        state['nodes']['N023']['research']['protocol'] = 'legacy'
        state_path.write_text(json.dumps(state))
    else:
        context['agent_id'] = 'another-agent'
    result = check((repo, sha, context))
    assert result['status'] == 'failed', result
    assert result['phase'] == phase, result
    assert result['candidate_sha'] == sha
    if failure == 'protocol':
        assert 'legacy' in result['reason']


def test_cli_preflight_has_no_host_fields(monkeypatch, capsys):
    from autoresearch import cli
    calls = []
    monkeypatch.setattr(cli, '_coral_dir', lambda: Path('/unused'))
    monkeypatch.setattr(cli, 'request_trusted', lambda *a: calls.append(a) or {'status':'passed'})
    cli.main(['preflight', '--sha', 'a'*40])
    assert calls == [(Path('/unused'), 'preflight_assignment', {'sha':'a'*40})]
    assert json.loads(capsys.readouterr().out)['status'] == 'passed'


def test_service_preflight_authentication_and_field_schema(tmp_path):
    from coral.grader.trusted_writes import AgentBinding, TrustedWriteError, TrustedWriteService
    binding = AgentBinding('engineer-1', os.getpid(), 1, str(tmp_path / 'owned'), None)
    calls = []
    service = TrustedWriteService(tmp_path / '.coral', SimpleNamespace(grader=SimpleNamespace(args={})),
        identity_resolver=lambda *_: binding,
        workflow_handler=lambda *a: calls.append(a) or {'status':'passed'})
    assert service.handle('preflight_assignment', {'sha':'a'*40}, os.getpid())['status']=='passed'
    assert calls[0][2]['worktree'] == binding.worktree
    assert calls[0][2]['agent_id'] == 'engineer-1'
    with pytest.raises(TrustedWriteError, match='fields'):
        service.handle('preflight_assignment', {'sha':'a'*40, 'worktree':'/private'}, os.getpid())


def test_ipv4_launcher_parser_and_autoport():
    from autoresearch.ipv4_launch import launch_command, parser
    args = parser().parse_args(['--nproc-per-node','2','--','probe.py','--foo'])
    command = launch_command(args)
    assert '127.0.0.1' in command and '--standalone' not in command
    assert int(command[command.index('--master-port') + 1]) > 0
    assert command[-2:] == ['probe.py', '--foo']
    fixed = launch_command(parser().parse_args(['--port','23456','probe.py']))
    assert fixed[fixed.index('--master-port') + 1] == '23456'
    for port in ('-1','65536'):
        with pytest.raises(SystemExit):
            parser().parse_args(['--port', port, 'probe.py'])


@pytest.mark.parametrize('worker_fails', [False, True, 'signal'])
def test_real_two_process_ipv4_rendezvous(tmp_path, worker_fails):
    python = Path(os.environ.get('CORAL_CPU_TORCH_PYTHON', '/home/wyr/greedy_lore/.venv/bin/python'))
    assert python.is_file(), 'approved existing Torch environment is required for this integration check'
    root = workspace(tmp_path / 'workspace')
    script = root / 'experiments/ranks.py'
    script.write_text('''import os
import torch
import torch.distributed as dist
assert os.environ['MASTER_ADDR'] == '127.0.0.1'
assert int(os.environ['MASTER_PORT']) > 0
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
dist.init_process_group('gloo')
value = torch.tensor([dist.get_rank() + 1])
dist.all_reduce(value)
assert value.item() == 3 and dist.get_world_size() == 2
print('rank passed', dist.get_rank(), flush=True)
dist.destroy_process_group()
''')
    if worker_fails is True:
        with script.open('a') as stream:
            stream.write('raise RuntimeError("expected worker failure")\n')
    elif worker_fails == 'signal':
        with script.open('a') as stream:
            stream.write('import time; time.sleep(60)\n')
    mounts = [str(python.parent.parent), str(python.resolve().parent.parent.parent)]
    command = build_sandbox(root, mounts, [], [str(python), '/run/coral-ipv4-launch.py',
                            '--nproc-per-node', '2', '--port', '0', str(script)])
    environment = {'PATH': '/usr/bin:/bin', 'HOME': '/tmp', 'TMPDIR': '/tmp', 'LANG':'C.UTF-8',
                   'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'GLOO_SOCKET_IFNAME':'lo',
                   'PYTHONDONTWRITEBYTECODE':'1'}
    if worker_fails == 'signal':
        import select
        import time

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
        output = b''
        try:
            deadline = time.monotonic() + 30
            while output.count(b'rank passed') < 2 and time.monotonic() < deadline:
                readable, _, _ = select.select([process.stdout], [], [], 1)
                if readable:
                    chunk = os.read(process.stdout.fileno(), 4096)
                    if not chunk:
                        break
                    output += chunk
            assert output.count(b'rank passed') == 2, output
            process.terminate()
            stdout, stderr = process.communicate(timeout=10)
            result = subprocess.CompletedProcess(command, process.returncode,
                                                  (output + stdout).decode(), stderr.decode())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
        assert result.returncode != 0
    else:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment)
    if worker_fails is True:
        assert result.returncode != 0 and 'expected worker failure' in result.stderr
    elif worker_fails is False:
        assert result.returncode == 0, result.stdout + result.stderr
    assert 'host-managed --port' in result.stderr
    assert result.stdout.count('rank passed') == 2


def test_preflight_reads_committed_entry_and_submission(candidate):
    repo, _, _ = candidate
    (repo / 'submission.yaml').write_text('forged: dirty\n')
    (repo / 'c4/table_v_timing.py').write_text('this is not valid python!!!')
    result = check(candidate)
    assert result['status'] == 'passed', result


def test_preflight_rejects_undeclared_contract_data(candidate):
    coral = Path(candidate[2]['coral_dir'])
    config = yaml.safe_load((coral / 'config.yaml').read_text())
    config['grader']['args']['diagnostic_resources'] = []
    (coral / 'config.yaml').write_text(yaml.safe_dump(config))
    result = check(candidate)
    assert result['phase'] == 'resources' and 'not declared' in result['reason']


def test_preflight_does_not_apply_budget_authorizer(candidate, monkeypatch):
    from autoresearch import state
    monkeypatch.setattr(state, 'authorize_attempt', lambda *_: pytest.fail('budget authorizer called'))
    monkeypatch.setattr(state, 'read_attempts', lambda *_a, **_kw: pytest.fail('formal budget queried'))
    monkeypatch.setattr('autoresearch.diagnostic_sandbox.reserve_idle_gpus',
                        lambda *_a, **_kw: pytest.fail('GPU reservation called'))
    result = check(candidate)
    assert result['status'] == 'passed', result


def test_preflight_cli_failure_is_nonzero(monkeypatch, capsys):
    from autoresearch import cli
    monkeypatch.setattr(cli, '_coral_dir', lambda: Path('/unused'))
    monkeypatch.setattr(cli, 'request_trusted', lambda *_: {'status':'failed','phase':'metadata'})
    with pytest.raises(SystemExit) as failure:
        cli.main(['preflight', '--sha', 'a'*40])
    assert failure.value.code == 1
    assert json.loads(capsys.readouterr().out)['phase'] == 'metadata'


def test_real_authenticated_socket_preflight(candidate):
    from autoresearch.client import request_trusted
    from autoresearch.state import handle_workflow

    from coral.grader.trusted_writes import (
        AgentBinding,
        TrustedWriteServer,
        TrustedWriteService,
        process_start_time,
        write_trusted_agent_registry,
    )
    repo, sha, context = candidate
    coral = Path(context['coral_dir'])
    write_trusted_agent_registry(coral, [AgentBinding(context['agent_id'], os.getpid(),
        process_start_time(os.getpid()), str(repo), None)])
    service = TrustedWriteService(coral, SimpleNamespace(), workflow_handler=handle_workflow)
    server = TrustedWriteServer(coral, service)
    server.start()
    try:
        result = request_trusted(coral, 'preflight_assignment', {'sha': sha})
        assert result['status'] == 'passed', result
        assert result['candidate_sha'] == sha
    finally:
        server.stop()


def test_engineer_preflight_uses_engineer_ownership_and_frozen_anchor(candidate):
    from coral.hub.attempts import write_attempt
    from coral.types import Attempt

    repo, research_sha, context = candidate
    coral = Path(context['coral_dir'])
    declaration = yaml.safe_load((repo / 'submission.yaml').read_text())
    declaration.update(mode='engineering', base_commit=research_sha, quality_anchor=research_sha)
    (repo / 'submission.yaml').write_text(yaml.safe_dump(declaration))
    _git(repo, 'add', 'submission.yaml')
    _git(repo, 'commit', '-m', 'engineering')
    sha = _git(repo, 'rev-parse', 'HEAD')
    state_path = coral / 'private/autoresearch_state.json'
    state = json.loads(state_path.read_text())
    state['nodes']['N023']['research'].update(status='completed', selected_sha=research_sha)
    state['nodes']['N023']['engineering_claims'] = [{'agent_id':'engineer-1','status':'active'}]
    state_path.write_text(json.dumps(state))
    write_attempt(coral, Attempt(commit_hash=research_sha, status='improved', title='frozen research',
        score=1.0, parent_hash=None, timestamp='2026-10-29T00:00:00+00:00',
        metadata={'mode':'research','node_id':'N023'}, agent_id='research-N023'))
    before = {str(p):p.read_bytes() for p in coral.rglob('*') if p.is_file()}
    result = check((repo, sha, {**context, 'agent_id':'engineer-1'}))
    assert result['status'] == 'passed', result
    assert before == {str(p):p.read_bytes() for p in coral.rglob('*') if p.is_file()}


def test_preflight_checks_resource_file_reads_not_only_directory_listing(candidate):
    data = Path(candidate[2]['coral_dir']).parent / 'c4_path/sample'
    data.chmod(0)
    try:
        result = check(candidate)
        assert result['status'] == 'failed' and result['phase'] == 'namespace', result
        assert 'PermissionError' in result['reason']
    finally:
        data.chmod(0o600)
