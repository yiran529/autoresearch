"""Engineer diagnostics use live claim identity and durable, separate budgets."""

# ruff: noqa: S101, F811
import json
import os
from pathlib import Path

import pytest
from autoresearch import diagnostics
from test_diagnostics import request
from test_workflow_state import _engineer_with_attempts, _git, _state, _store
from test_workflow_state import engineering_ready as engineering_ready
from test_workflow_state import root_run as root_run

from coral.grader.trusted_writes import (
    AgentBinding,
    process_start_time,
    write_trusted_agent_registry,
)


@pytest.fixture
def engineer(engineering_ready):
    coral, repo, sha = engineering_ready
    assignment = _store(coral, repo).reconcile(0, 1)[0]
    binding = AgentBinding(assignment.agent_id, os.getpid(), process_start_time(os.getpid()),
                           str(repo), None)
    write_trusted_agent_registry(coral, [binding])
    scripts = repo / 'experiments'
    scripts.mkdir()
    (scripts / 'probe.py').write_text('print("probe")\n')
    return coral, repo, sha, assignment.agent_id, scripts


def test_live_engineer_runs_with_frozen_candidate_and_separate_budget(engineer):
    coral, repo, sha, agent, scripts = engineer
    payload = request(scripts_dir=str(scripts))

    def execute(task, payload, cancelled):
        cancelled()
        state = task.read()
        assert state['source_role'] == 'engineer'
        assert (task.workspace / 'code' / sha / 'model.txt').read_text() == 'candidate\n'
        assert (task.workspace / 'experiments/probe.py').read_text() == 'print("probe")\n'
        return {'status': 'completed', 'started': True}

    result = diagnostics.handle_diagnostic(payload, {'coral_dir': str(coral),
        'peer_pid': os.getpid()}, executor=execute)
    record = json.loads(Path(result['record_path']).read_text())
    assert record['candidate_sha'] == sha
    assert record['owner_agent_id'] == agent
    assert record['analysis_version'] is None
    assert list((coral / 'public/attempts').glob('*.json'))
    task = diagnostics.DiagnosticTask(coral, result['diagnostic_id'].split('/')[0])
    assert task.read()['executions'] == 1
    assert diagnostics.handle_diagnostic(payload, {'coral_dir': str(coral),
        'peer_pid': os.getpid()}, executor=lambda *_: pytest.fail('duplicate')) == result


@pytest.mark.parametrize('change', ['completed', 'finalizing', 'foreign', 'dirty', 'escape'])
def test_engineer_diagnostic_rejects_invalid_owner_stage_or_inputs(engineer, change, tmp_path):
    coral, repo, _, agent, scripts = engineer
    state = _state(coral)
    if change == 'completed':
        state['engineering_workers'][agent]['status'] = 'completed'
    elif change == 'finalizing':
        state['engineering_workers'][agent]['finish_requested'] = True
    elif change == 'foreign':
        state['engineering_workers'][agent]['node_id'] = 'N099'
    elif change == 'dirty':
        (repo / 'model.txt').write_text('dirty')
    else:
        scripts = tmp_path
    (coral / 'private/autoresearch_state.json').write_text(json.dumps(state))
    with pytest.raises((ValueError, PermissionError)):
        diagnostics.handle_diagnostic(request(scripts_dir=str(scripts)),
            {'coral_dir': str(coral), 'peer_pid': os.getpid()}, executor=lambda *_: pytest.fail('ran'))


@pytest.mark.parametrize('result_status,limit', [('completed', 5), ('failed', 15)])
def test_engineer_diagnostic_requires_own_scripts_and_does_not_reset_on_resume(engineer, result_status, limit):
    coral, _, _, _, scripts = engineer
    context = {'coral_dir': str(coral), 'peer_pid': os.getpid()}
    with pytest.raises((ValueError, PermissionError), match='scripts'):
        diagnostics.handle_diagnostic(request(), context)
    for n in range(limit):
        diagnostics.handle_diagnostic(request(str(n), scripts_dir=str(scripts)), context,
            executor=lambda *_: {'status': result_status, 'started': True})
    with pytest.raises(ValueError, match='budget'):
        diagnostics.handle_diagnostic(request('beyond-limit', scripts_dir=str(scripts)), context,
            executor=lambda *_: pytest.fail('ran'))


def test_engineer_cli_uses_current_run_without_analyzer_environment(engineer, monkeypatch, capsys):
    from autoresearch import cli
    coral, repo, _, _, scripts = engineer
    monkeypatch.chdir(repo)
    monkeypatch.delenv('CORAL_DIAGNOSTIC_RUN', raising=False)
    monkeypatch.setattr(cli, '_coral_dir', lambda: coral)
    def trusted(run, operation, payload, **_):
        assert run == coral and operation == 'run_diagnostic'
        assert payload['scripts_dir'] == str(scripts)
        return {'status': 'completed'}
    monkeypatch.setattr(cli, 'request_trusted', trusted)
    cli.main(['diagnostic', 'run', '--scripts-dir', str(scripts), '--',
              'python', 'experiments/probe.py'])
    assert 'completed' in capsys.readouterr().out


def test_diagnostic_records_are_sealed_and_cannot_change_after_publication(engineer, engineering_ready):
    from autoresearch.analysis import AnalysisContext, _artifact_evidence, resolve_evidence
    from autoresearch.lifecycle import run_worker

    coral, repo, _, agent, scripts = engineer
    result = diagnostics.handle_diagnostic(request(scripts_dir=str(scripts)),
        {'coral_dir': str(coral), 'peer_pid': os.getpid()},
        executor=lambda *_: {'status': 'completed', 'started': True})
    store, node, body = _engineer_with_attempts(engineering_ready, count=1)
    store.submit_engineering_analysis(agent, body)
    assert run_worker(str(coral))['errors'] == {}
    document = json.loads((node / 'engineering_analysis_v1.json').read_text())
    records = document['evidence_manifest']['evidence_records']
    diagnostic_records = [r for r in records if r['path'].startswith('diagnostics/')]
    assert diagnostic_records and all(r['status'] == 'present' and r['sha256'] for r in diagnostic_records)
    with pytest.raises(PermissionError):
        diagnostics.handle_diagnostic(request('after-finish', scripts_dir=str(scripts)),
            {'coral_dir': str(coral), 'peer_pid': os.getpid()})
    from coral.hub.attempts import read_attempts
    context = AnalysisContext(coral, coral / 'public/notes', None, tuple(read_attempts(coral)),
                              coral / 'public/eval_logs', repo)
    current = resolve_evidence('N001', 'engineering', context=context)
    _artifact_evidence(document, current, context)
    Path(result['record_path']).write_text('changed evidence')
    with pytest.raises(ValueError, match='changed'):
        _artifact_evidence(document, current, context)


def test_finish_rejects_running_diagnostic(engineer, engineering_ready):
    coral, _, _, agent, scripts = engineer
    def execute(task, *_):
        store, _, body = _engineer_with_attempts(engineering_ready, count=1)
        with pytest.raises(ValueError, match='running Engineering diagnostic'):
            store.submit_engineering_analysis(agent, body)
        return {'status': 'completed', 'started': True}
    diagnostics.handle_diagnostic(request(scripts_dir=str(scripts)),
        {'coral_dir': str(coral), 'peer_pid': os.getpid()}, executor=execute)


def test_candidate_snapshot_preserves_executable_helpers(engineer):
    coral, repo, _, _, scripts = engineer
    helper = repo / 'helper.sh'
    helper.write_text('#!/bin/sh\necho helper\n')
    helper.chmod(0o755)
    _git(repo, 'add', 'helper.sh')
    _git(repo, 'commit', '-m', 'executable helper')
    sha = _git(repo, 'rev-parse', 'HEAD')
    def execute(task, *_):
        assert (task.workspace / 'code' / sha / 'helper.sh').stat().st_mode & 0o100
        return {'status': 'completed', 'started': True}
    result = diagnostics.handle_diagnostic(request(scripts_dir=str(scripts)),
        {'coral_dir': str(coral), 'peer_pid': os.getpid()}, executor=execute)
    assert result['status'] == 'completed'


def test_engineer_executor_archives_outputs_and_reads_frozen_code(engineer, monkeypatch):
    """Real CPU subprocess; GPU reservation and namespace execution substituted."""
    import sys

    from autoresearch import diagnostic_sandbox as sandbox

    coral, repo, _, _, scripts = engineer
    source = ('import os\nfrom pathlib import Path\n'
              'p=Path(os.environ["CORAL_DIAGNOSTIC_CODE"])/"model.txt"\n'
              'assert p.read_text()=="candidate\\n"\n'
              'Path("results/value.json").write_text("verified frozen code")\n'
              'print("diagnostic completed")\n')
    (scripts / 'probe.py').write_text(source)
    class Reservation:
        indices = (0,)
        uuids = ('GPU-test',)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def check_exclusive(self, _):
            pass
    def reserve(*_, **kwargs):
        assert kwargs['source_role'] == 'engineer'
        (repo / 'model.txt').write_text('changed live code')
        return Reservation()
    original_build = sandbox.build_sandbox
    def cpu_command(workspace, roots, devices, argv, *, write_sources=None, **kwargs):
        original_build(workspace, roots, devices, argv, write_sources=write_sources, **kwargs)
        if not write_sources:
            return argv
        return [sys.executable, '-c',
                f'import os,runpy; os.chdir({str(write_sources["results"].parent)!r}); '
                'runpy.run_path("experiments/probe.py")']
    monkeypatch.setattr(sandbox.shutil, 'which', lambda _: '/unused/bwrap')
    monkeypatch.setattr(sandbox, 'reserve_idle_gpus', reserve)
    monkeypatch.setattr(sandbox, 'device_minors_for_uuids', lambda _: [0])
    monkeypatch.setattr(sandbox, 'build_sandbox', cpu_command)
    result = diagnostics.handle_diagnostic(request(scripts_dir=str(scripts)),
        {'coral_dir': str(coral), 'peer_pid': os.getpid()})
    assert result['status'] == 'completed'
    record = Path(result['record_path']).parent
    assert (record / 'stdout.log').read_text() == 'diagnostic completed\n'
    assert (record / 'scripts/probe.py').read_text() == source
    assert (record / 'results/value.json').read_text() == 'verified frozen code'


def test_registered_researcher_cannot_request_gpu_diagnostic(engineer):
    coral, repo, _, _, scripts = engineer
    binding = AgentBinding('research-N001', os.getpid(), process_start_time(os.getpid()), str(repo), None)
    write_trusted_agent_registry(coral, [binding])
    with pytest.raises(PermissionError, match='active Engineer'):
        diagnostics.handle_diagnostic(request(scripts_dir=str(scripts)),
            {'coral_dir': str(coral), 'peer_pid': os.getpid()}, executor=lambda *_: pytest.fail('ran'))


def test_engineer_scripts_cannot_follow_symlink(engineer, tmp_path):
    coral, _, _, _, scripts = engineer
    outside = tmp_path / 'outside'
    outside.mkdir()
    link = scripts / 'link'
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        diagnostics.handle_diagnostic(request(scripts_dir=str(link)),
            {'coral_dir': str(coral), 'peer_pid': os.getpid()}, executor=lambda *_: pytest.fail('ran'))
    result = diagnostics.handle_diagnostic(request('fixed', scripts_dir=str(scripts)),
        {'coral_dir': str(coral), 'peer_pid': os.getpid()},
        executor=lambda *_: {'status': 'completed', 'started': True})
    assert result['status'] == 'completed'
