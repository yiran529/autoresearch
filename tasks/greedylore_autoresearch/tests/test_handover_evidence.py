"""Real publication/recovery contracts for Analyzer-visible evidence references."""

# ruff: noqa: S101

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace

import pytest
from autoresearch.analysis import _artifact_evidence, analyze, resolve_evidence
from autoresearch.validation import load_document, validate_analysis
from test_analysis_recovery import _attempt, _context, _fake_code_snapshot, _publisher
from test_workflow_state import root_run as root_run


def payload(refs):
    return {'summary': 'Measured mechanism.', 'findings': ['Bounded observation.'],
            'limitations': ['Not a full quality trial.'], 'handover': {
                'mechanism_verdict': 'partially_solved', 'practical_verdict': 'inconclusive',
                'evidence_refs': refs, 'new_information': ['One controlled observation.'],
                'follow_up_questions': []}}


@pytest.fixture
def context(tmp_path):
    ctx = _context(tmp_path)
    repo = ctx.checkout_root
    repo.mkdir()
    for args in [('init', '-q'), ('config', 'user.name', 'Test'),
                 ('config', 'user.email', 'test@example.invalid')]:
        subprocess.run(['git', '-C', str(repo), *args], check=True)
    (repo / 'model.py').write_text('value = 1\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'fixed model'], check=True)
    sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    attempt = _attempt(sha, 'research', '2026-01-01T00:00:00+00:00')
    ctx = replace(ctx, attempts=(attempt,))
    log = ctx.log_root / sha / 'raw.json'
    log.parent.mkdir(parents=True)
    log.write_text('{"loss":3.8}')
    return ctx


def runner_with(refs):
    def runner(_prompt, _roots, _schema, scratch):
        out = scratch / 'output.json'
        out.write_text(json.dumps(payload(refs)))
        return out
    return runner


def test_staged_file_reference_publishes_as_canonical_and_recovers(context):
    sha = context.attempts[0].commit_hash
    ref = f'eval_logs/{sha}/run-{sha}/quality/all_results.json'
    nested = context.log_root / sha / f'run-{sha}/quality/all_results.json'
    nested.parent.mkdir(parents=True)
    nested.write_text('{"loss":3.8}')
    result = analyze('N001', 'research', context=context, runner=runner_with(['evidence/' + ref]),
                     publisher=_publisher(context.notes_root))
    doc = load_document(result.analysis_path)
    assert doc['handover']['evidence_refs'] == [ref]
    recovered = analyze('N001', 'research', context=context,
                        runner=lambda *_: pytest.fail('published report must not call model again'),
                        publisher=_publisher(context.notes_root))
    assert load_document(recovered.analysis_path) == doc


@pytest.mark.parametrize('prefix', ['', 'evidence/'])
def test_frozen_directory_reference_keeps_a_valid_canonical_alias(context, prefix):
    sha = context.attempts[0].commit_hash
    ref = f'eval_logs/{sha}/run-{sha}'
    result = analyze('N001', 'research', context=context, runner=runner_with([prefix + ref]),
                     publisher=_publisher(context.notes_root))
    assert load_document(result.analysis_path)['handover']['evidence_refs'] == [ref]


def test_fixed_code_is_sealed_from_git_and_ignores_worktree_changes(context):
    sha = context.attempts[0].commit_hash
    ref = f'code/{sha}/model.py'
    result = analyze('N001', 'research', context=context, runner=runner_with([ref]),
                     publisher=_publisher(context.notes_root))
    doc = load_document(result.analysis_path)
    assert any(r['path'] == ref and r['status'] == 'present'
               for r in doc['evidence_manifest']['evidence_records'])
    (context.checkout_root / 'model.py').write_text('uncommitted = True\n')
    evidence = resolve_evidence('N001', 'research', context=context)
    validate_analysis(doc, _artifact_evidence(doc, evidence, context))


@pytest.mark.parametrize('scenario', ['completed', 'timed_out', 'large_uncited_log',
                                     'interrupted_placeholder', 'interrupted_missing_record'])
def test_own_diagnostic_is_sealed_without_changing_frozen_research(context, scenario):
    frozen = resolve_evidence('N001', 'research', context=context)
    captured = {}

    def diagnostic_runner(_prompt, _roots, _schema, scratch, task):
        task.bind(os.getpid())
        captured['task'] = task

        def execute(item, request, cancelled):
            cancelled()
            folder = item.records / request['request_id'] / 'results'
            folder.mkdir()
            (folder / 'probe.json').write_text('{"gap":0.2}')
            if scenario == 'large_uncited_log':
                (folder.parent / 'stdout.log').write_bytes(b'x' * (1024 * 1024 + 1))
            if scenario == 'timed_out':
                return {'status': 'timed_out', 'started': True, 'reason': 'time budget'}
            return {'status': 'completed', 'started': True, 'exit_code': 0}

        task.run({'request_id': 'probe', 'argv': ['python', 'experiments/probe.py'],
                  'gpus': 1, 'timeout': 30}, os.getpid(), execute)
        if scenario.startswith('interrupted_'):
            from coral.grader.trusted_writes import _atomic_json

            state = task.read()
            request = {**state['requests']['probe']['request'], 'request_id': 'interrupted'}
            result = {'status': 'interrupted', 'started': True,
                      'diagnostic_id': task.task_id + '/interrupted', 'reason': 'worker interrupted'}
            state['requests']['interrupted'] = {'request': request, 'result': result}
            state['executions'] += 1
            _atomic_json(task.path, state)
            record = json.loads((task.records / 'probe/record.json').read_text())
            record.update(request)
            record.update(result)
            record.pop('finished_at')
            record.pop('record_path')
            if scenario == 'interrupted_placeholder':
                _atomic_json(task.records / 'interrupted/record.json', record)
        out = scratch / 'results/output.json'
        out.write_text(json.dumps(payload(['records/probe/results/probe.json'])))
        return out

    result = analyze('N001', 'research', context=context, diagnostic_runner=diagnostic_runner,
                     publisher=_publisher(context.notes_root))
    doc = load_document(result.analysis_path)
    ref = 'diagnostics/N001-research-v1/records/probe/results/probe.json'
    assert doc['handover']['evidence_refs'] == [ref]
    assert any(r['path'] == ref for r in doc['evidence_manifest']['evidence_records'])
    if scenario == 'large_uncited_log':
        assert not any(r['path'].endswith('/stdout.log') for r in doc['evidence_manifest']['evidence_records'])
    if scenario.startswith('interrupted_'):
        assert not any('/interrupted/' in r['path'] for r in doc['evidence_manifest']['evidence_records'])
    current = resolve_evidence('N001', 'research', context=context)
    assert current.manifest_hash == frozen.manifest_hash
    validate_analysis(doc, _artifact_evidence(doc, current, context))
    (captured['task'].records / 'probe/results/probe.json').write_text('{"gap":99}')
    with pytest.raises(ValueError, match='evidence changed'):
        _artifact_evidence(doc, current, context)


@pytest.mark.parametrize('ref', [
    'invented/results.json', '../private/key', '/etc/passwd',
    'code/' + 'f' * 40 + '/model.py', 'records/unknown/results/probe.json',
    'diagnostics/N002-research-v1/records/probe/results/probe.json',
    'diagnostics/N001-research-v2/records/probe/results/probe.json',
])
def test_rejects_unknown_foreign_or_unsafe_sources(context, ref):
    with pytest.raises(ValueError):
        analyze('N001', 'research', context=context, runner=runner_with([ref]),
                publisher=_publisher(context.notes_root))
    assert not (context.notes_root / 'nodes/N001/research_analysis_v1.md').exists()


def test_unknown_reference_error_identifies_the_exact_reference(context):
    with pytest.raises(ValueError, match='invented/results.json'):
        analyze('N001', 'research', context=context,
                runner=runner_with(['invented/results.json']), publisher=_publisher(context.notes_root))


def test_validation_retries_receive_feedback_and_stop_after_three_failures(root_run):
    from autoresearch.lifecycle import run_worker
    from autoresearch.state import WorkflowStore

    (root_run / 'public/notes/nodes/N000/research_analysis_v1.md').unlink()
    prompts = []

    def runner(prompt, _roots, _schema, scratch):
        prompts.append(prompt)
        out = scratch / 'output.json'
        out.write_text(json.dumps(payload(['invented/results.json'])))
        return out

    for index in range(5):
        run_worker(str(root_run), runner=runner, code_snapshotter=_fake_code_snapshot,
                   now=lambda: 100000 + index * 10000)
    state = json.loads(WorkflowStore(root_run).state_path.read_text())
    job = state['nodes']['N000']['analysis']
    assert len(prompts) == 3
    assert 'invented/results.json' in prompts[1]
    assert job['status'] == 'blocked'
    assert job['validation_failures'] == 3
