"""Standalone model retries are bounded and do not retry permanent errors."""

from types import SimpleNamespace

import pytest

from coral.agent import model_retry


@pytest.mark.parametrize(('errors', 'starts'), [
    (['workspace routing discovery failed', 'request timed out', None], 3),
    (['workspace routing discovery failed'] * 4, 3),
    (['401 Unauthorized: request timed out'], 1),
    (['invalid configuration'], 1),
    (['ERROR: unexpected status 503 Service Unavailable', None], 2),
    (['ERROR: HTTP 500 Internal Server Error', None], 2),
    (['ERROR: HTTP 502 Bad Gateway', None], 2),
    (['ERROR: HTTP 504 Gateway Timeout', None], 2),
    (['ERROR: server_error: upstream failed', None], 2),
    (['ERROR: Selected model is at capacity. Please try a different model.', None], 2),
    (['ERROR: HTTP 429 Too Many Requests', None], 2),
    (['ERROR: HTTP 503 Service Unavailable'] * 4, 3),
    (['ERROR: HTTP 401 Unauthorized; upstream returned HTTP 503'], 1),
    (['ERROR: HTTP 403 Forbidden; request timed out'], 1),
    (['ERROR: HTTP 429 insufficient_quota'], 1),
])
def test_shared_connection_retry_limits_and_permanent_errors(monkeypatch, errors, starts):
    times, waits = [], []
    clock = [0.0]
    monkeypatch.setattr(model_retry.time, 'monotonic', lambda: clock[0])

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(model_retry.time, 'sleep', sleep)

    def invoke(remaining, attempt):
        times.append(remaining)
        error = errors[attempt - 1]
        result = SimpleNamespace(returncode=1 if error else 0, stderr=error or '')
        model_retry.check_connection_failure(result)
        return result

    if errors[starts - 1]:
        if starts == 3:
            with pytest.raises(RuntimeError):
                model_retry.retry_model_connection(invoke, timeout=120)
        else:
            assert model_retry.retry_model_connection(invoke, timeout=120).returncode == 1
    else:
        assert model_retry.retry_model_connection(invoke, timeout=120).returncode == 0
    assert len(times) == starts
    assert waits == [15, 30][:starts - 1]
    assert times == [120, 105, 75][:starts]


def test_connection_retry_never_resets_overall_timeout(monkeypatch):
    monkeypatch.setattr(model_retry.time, 'monotonic', lambda: 0)

    def invoke(remaining, attempt):
        model_retry.check_connection_failure(SimpleNamespace(
            returncode=1, stderr='workspace routing discovery failed'))

    with pytest.raises(RuntimeError):
        model_retry.retry_model_connection(invoke, timeout=10)


@pytest.mark.parametrize('error', [
    'workspace routing discovery failed', 'unexpected status 503 Service Unavailable',
])
def test_standalone_worker_retries_real_process_and_keeps_input(tmp_path, monkeypatch, error):
    import sys

    from coral.grader.trusted_worker import run_workflow_model

    marker = tmp_path / 'starts'
    monkeypatch.setattr(model_retry.time, 'sleep', lambda _: None)
    code = (
        'import sys; from pathlib import Path; '
        f'p=Path({str(marker)!r}); '
        'n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); '
        'data=sys.stdin.read(); '
        f'print({("ERROR: " + error)!r} if n == 0 else "", file=sys.stderr); '
        'print(data); sys.exit(1 if n == 0 else 0)'
    )
    result = run_workflow_model([sys.executable, '-c', code], input='same fixed task', timeout=120)
    assert result.returncode == 0
    assert result.stdout.strip() == 'same fixed task'
    assert marker.read_text() == '2'


def test_stop_during_retry_backoff_prevents_another_model_launch(tmp_path, monkeypatch):
    import sys

    from coral.grader.trusted_worker import run_workflow_model

    private = tmp_path / 'private'
    private.mkdir()
    monkeypatch.setenv('CORAL_WORKFLOW_CORAL_DIR', str(tmp_path))
    monkeypatch.setattr(model_retry.time, 'sleep', lambda _: (private / 'workflow_stopping').touch())
    with pytest.raises(PermissionError, match='stopping'):
        run_workflow_model([sys.executable, '-c',
                            'import sys; sys.stderr.write("workspace routing discovery failed"); sys.exit(1)'],
                           input='', timeout=120)
    assert not (private / 'workflow_model.json').exists()
