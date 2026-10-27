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
