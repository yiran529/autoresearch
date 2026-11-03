"""CLI waits on durable host jobs, with only IPC and expensive execution replaced."""
# ruff: noqa: S101

import json
import threading

import pytest
from test_staged_research import host


@pytest.fixture
def client(tmp_path, monkeypatch):
    from autoresearch import cli

    h, record, backend = host(tmp_path)
    threads = []
    calls = []

    def request(root, operation, payload):
        assert root == h.store.coral_dir and operation == 'research_action'
        action = payload['action']
        calls.append(action)
        if action == 'wait':
            return h.wait(payload['request_id'])
        if action == 'status':
            return h.status()
        job = h.enqueue(action, payload['data'], payload['request_id'], {})
        thread = threading.Thread(target=h.drain)
        threads.append(thread)
        thread.start()
        return job

    monkeypatch.setattr(cli, 'request_trusted', request)
    monkeypatch.setattr(cli, '_coral_dir', lambda: h.store.coral_dir)
    monkeypatch.setattr(cli, '_RESEARCH_POLL_INTERVAL', .01, raising=False)
    yield cli, h, record, backend, calls
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()


def train_args(tmp_path, **extra):
    path = tmp_path / 'request.json'
    path.write_text(json.dumps({'iteration': 1, 'sha': 'b' * 40, **extra}))
    return ['research', 'train', '--node', 'N001', '--file', str(path), '--request-id', 'train1']


@pytest.mark.parametrize('pause', [None, 1200])
def test_train_waits_for_published_completion_or_pause(client, tmp_path, capsys, pause):
    cli, h, record, backend, _ = client
    release = threading.Event()
    run = backend.run

    def delayed(host, job):
        assert release.wait(timeout=5)
        return run(host, job)

    backend.run = delayed
    timer = threading.Timer(.05, release.set)
    timer.start()
    try:
        cli.main(train_args(tmp_path, **({'pause_after_steps': pause} if pause else {})))
        result = json.loads(capsys.readouterr().out)
        assert result['status'] == 'completed'
        assert result['result']['status'] == ('paused' if pause else 'completed')
        assert record.read()['iterations'][0]['observations'][-1]['status'] == result['result']['status']
        assert len(backend.calls) == 1
        assert h.status()['jobs']['train1']['status'] == 'completed'
    finally:
        release.set()
        timer.join()


@pytest.mark.parametrize('infrastructure', [False, True])
def test_train_failure_returns_evidence_and_nonzero(client, tmp_path, capsys, infrastructure):
    from greedylore_autoresearch_grader.runtime import InfraFailure

    cli, _, record, backend, _ = client

    def failed(host, job):
        raise InfraFailure('lost reservation') if infrastructure else ValueError('bad algorithm')

    backend.run = failed
    with pytest.raises(SystemExit) as error:
        cli.main(train_args(tmp_path))
    assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result['result']['status'] == ('infra_failed' if infrastructure else 'failed')
    assert 'reason' in result['result']
    assert record.read()['iterations'][0]['observations'][-1]['status'] == result['result']['status']


def test_wait_retrieves_original_job_without_reexecution(client, tmp_path, capsys):
    cli, h, _, backend, calls = client
    h.enqueue('train', {'iteration': 1, 'sha': 'b' * 40, 'pause_after_steps': 1200}, 'train1', {})
    h.drain()
    cli.main(['research', 'wait', '--node', 'N001', '--request-id', 'train1'])
    result = json.loads(capsys.readouterr().out)
    assert result['result']['status'] == 'paused'
    assert calls == ['wait']
    assert len(backend.calls) == 1


def test_wait_returns_publication_error_without_hanging(client, capsys, monkeypatch):
    cli, h, record, _, _ = client
    h.enqueue('train', {'iteration': 1, 'sha': 'b' * 40}, 'train1', {})
    monkeypatch.setattr(record, 'observe', lambda *_: (_ for _ in ()).throw(OSError('disk unavailable')))
    # Host and fixture share the durable record, but use distinct Python wrappers.
    monkeypatch.setattr(h.record, 'observe', record.observe)
    h.drain()
    with pytest.raises(SystemExit) as error:
        cli.main(['research', 'wait', '--node', 'N001', '--request-id', 'train1'])
    assert error.value.code == 1
    assert 'disk unavailable' in json.loads(capsys.readouterr().out)['error']


def test_wait_requires_request_id(client):
    cli, *_ = client
    with pytest.raises(SystemExit) as error:
        cli.main(['research', 'wait', '--node', 'N001'])
    assert error.value.code == 2


def test_no_wait_is_explicit_submission_only(client, tmp_path, capsys):
    cli, _, _, backend, _ = client
    release = threading.Event()
    run = backend.run

    def delayed(host, job):
        assert release.wait(timeout=5)
        return run(host, job)

    backend.run = delayed
    try:
        cli.main([*train_args(tmp_path), '--no-wait'])
        assert json.loads(capsys.readouterr().out)['status'] == 'queued'
        assert not backend.calls
    finally:
        release.set()


def test_cancelled_request_is_returned_without_starting_training(client, capsys):
    cli, h, _, backend, _ = client
    h.enqueue('train', {'iteration': 1, 'sha': 'b' * 40}, 'train1', {})
    path = h.store.coral_dir / 'private/autoresearch_state.json'
    state = json.loads(path.read_text())
    state['nodes']['N001']['research']['status'] = 'cancelled'
    path.write_text(json.dumps(state))
    h.drain()
    with pytest.raises(SystemExit) as error:
        cli.main(['research', 'wait', '--node', 'N001', '--request-id', 'train1'])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)['result']['status'] == 'cancelled'
    assert not any(call[0] == 'train' for call in backend.calls)


def test_transport_error_preserves_request_id_and_reconnect_instruction(client, tmp_path, monkeypatch, capsys):
    from autoresearch.client import TrustedWriteError

    cli, _, _, _, _ = client
    monkeypatch.setattr(cli, 'request_trusted', lambda *_: (_ for _ in ()).throw(TrustedWriteError('channel lost')))
    with pytest.raises(SystemExit, match='research wait --node N001 --request-id train1'):
        cli.main(train_args(tmp_path))
    assert 'research request: train1' in capsys.readouterr().err


def test_unknown_job_and_other_owner_cannot_wait(tmp_path):
    from autoresearch.staged_research import StagedResearch

    h, _, _ = host(tmp_path)
    with pytest.raises(ValueError, match='unknown.*request'):
        h.wait('missing')
    other = StagedResearch(h.store, 'research-N002', 'N001', backend=h.backend)
    with pytest.raises(PermissionError):
        other.wait('missing')


def test_status_summary_flag_sends_projection_request(monkeypatch, capsys):
    from autoresearch import cli
    calls = []
    monkeypatch.setattr(cli, '_coral_dir', lambda: '/tmp/test-run')
    monkeypatch.setattr(cli, 'request_trusted', lambda *args: calls.append(args) or {'jobs': {}})
    cli.main(['research', 'status', '--node', 'N001', '--summary'])
    assert calls[0][2]['data'] == {'summary': True}
    assert 'jobs' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(['research', 'recover', '--node', 'N001', '--summary'])


@pytest.mark.parametrize('action', ['design', 'decision'])
def test_record_write_receipt_and_full_output_preserve_idempotent_host_call(tmp_path, monkeypatch, capsys, action):
    from autoresearch import cli
    from test_research_record import design, initialize, record

    r = record(tmp_path)
    initialize(r)
    r.begin(1, 'b' * 40, 'run1')
    r.observe(1, {'status': 'completed', 'trace': list(range(5000))}, 'observe1')
    payload = design('second') if action == 'design' else {'iteration': 1, 'action': 'revise', 'reason': 'new evidence'}
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(payload))
    calls = []

    def request(root, operation, data):
        calls.append(data)
        assert operation == 'research_action'
        if action == 'design':
            return r.append_design(data['data'], data['request_id'])
        return r.decide(**data['data'], request_id=data['request_id'])

    monkeypatch.setattr(cli, '_coral_dir', lambda: r.store.coral_dir)
    monkeypatch.setattr(cli, 'request_trusted', request)
    argv = ['research', action, '--node', 'N001', '--file', str(path), '--request-id', 'write1']
    cli.main(argv)
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['recorded'] is True
    assert receipt['operation'] == action and receipt['request_id'] == 'write1'
    assert receipt['record_path'] == 'notes/nodes/N001/research.json'
    assert 'iterations' not in receipt and len(json.dumps(receipt)) < 1000
    if action == 'decision':
        assert receipt['iteration'] == 1 and receipt['decision'] == 'revise'
    else:
        assert 'iteration' not in receipt  # the current tail is not the replayed request's iteration
    # Append another candidate before replay to expose misleading tail-based receipts.
    if action == 'design':
        r.begin(2, 'b' * 40, 'next-run')
        r.observe(2, {'status': 'failed'}, 'next-observe')
        r.append_design(design('third'), 'third')
    cli.main(argv)
    assert json.loads(capsys.readouterr().out) == receipt
    cli.main([*argv, '--full-output'])
    assert json.loads(capsys.readouterr().out) == r.read()
    assert all(call['data'] == payload for call in calls)


def test_full_output_rejected_for_execution_actions_before_any_request(monkeypatch):
    from autoresearch import cli
    monkeypatch.setattr(cli, 'request_trusted', lambda *a: pytest.fail('must not submit'))
    with pytest.raises(SystemExit) as error:
        cli.main(['research', 'train', '--node', 'N001', '--full-output'])
    assert error.value.code == 2
