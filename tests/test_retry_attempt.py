from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from coral.grader.trusted_writes import (
    TrustedWriteError,
    begin_grader_run,
    finalize_grader_run,
    retry_attempt,
)
from coral.hub.attempts import read_attempt, write_attempt
from coral.types import Attempt


def _attempt(status: str = "crashed", failure_class: str = "infra") -> Attempt:
    return Attempt(
        commit_hash="a" * 40,
        agent_id="agent-1",
        title="attempt",
        score=None,
        status=status,
        parent_hash="b" * 40,
        timestamp=datetime.now(UTC).isoformat(),
        metadata={
            "failure_class": failure_class,
            "contract_hash": "contract-v1",
            "infra_retry_count": 0,
            "grader_runs": [
                {
                    "run_id": "r1",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "finished_at": "2026-01-01T00:01:00+00:00",
                    "status": status,
                    "failure_class": failure_class,
                    "contract_hash": "contract-v1",
                    "log_dir": "eval_logs/" + "a" * 40 + "/r1",
                }
            ],
        },
    )


def test_retry_accepts_only_terminal_infra_attempt(tmp_path):
    original = _attempt()
    write_attempt(tmp_path, original)

    retried = retry_attempt(
        tmp_path,
        original.commit_hash,
        None,
        expected_agent_id="agent-1",
        expected_contract_hash="contract-v1",
        max_infra_retries=2,
    )

    assert retried.status == "pending"
    assert retried.commit_hash == original.commit_hash
    assert retried.metadata["grader_runs"] == original.metadata["grader_runs"]
    assert retried.metadata["infra_retry_count"] == 1


@pytest.mark.parametrize(
    ("status", "failure_class"),
    [("pending", "infra"), ("improved", "infra"), ("crashed", "candidate")],
)
def test_retry_rejects_non_retryable_attempt(tmp_path, status, failure_class):
    attempt = _attempt(status, failure_class)
    write_attempt(tmp_path, attempt)

    with pytest.raises(TrustedWriteError):
        retry_attempt(
            tmp_path,
            attempt.commit_hash,
            None,
            expected_agent_id="agent-1",
            expected_contract_hash="contract-v1",
            max_infra_retries=2,
        )


def test_retry_rejects_wrong_agent_contract_and_exhausted_budget(tmp_path):
    attempt = _attempt()
    attempt.metadata["infra_retry_count"] = 2
    write_attempt(tmp_path, attempt)

    for kwargs, message in [
        ({"expected_agent_id": "agent-2", "expected_contract_hash": "contract-v1"}, "originating Agent"),
        ({"expected_agent_id": "agent-1", "expected_contract_hash": "contract-v2"}, "contract changed"),
        ({"expected_agent_id": "agent-1", "expected_contract_hash": "contract-v1"}, "retry budget"),
    ]:
        with pytest.raises(TrustedWriteError, match=message):
            retry_attempt(
                tmp_path,
                attempt.commit_hash,
                None,
                max_infra_retries=2,
                **kwargs,
            )


def test_concurrent_retry_allows_exactly_one_winner(tmp_path):
    attempt = _attempt()
    write_attempt(tmp_path, attempt)

    def retry() -> bool:
        try:
            retry_attempt(
                tmp_path,
                attempt.commit_hash,
                None,
                expected_agent_id="agent-1",
                expected_contract_hash="contract-v1",
                max_infra_retries=2,
            )
            return True
        except TrustedWriteError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _unused: retry(), range(2)))

    assert sorted(results) == [False, True]


def test_dispatch_finalize_history_is_append_only(tmp_path):
    pending = _attempt(status="pending", failure_class="infra")
    pending.metadata["grader_runs"] = []
    pending.metadata.pop("failure_class")
    write_attempt(tmp_path, pending)

    dispatched, run_id = begin_grader_run(tmp_path, pending.commit_hash, None)
    assert dispatched.metadata["grader_runs"][0]["status"] == "running"
    assert dispatched.metadata["grader_invocation_count"] == 1

    finalized = finalize_grader_run(
        tmp_path,
        pending.commit_hash,
        None,
        run_id,
        status="crashed",
        failure_class="infra",
        score=None,
        feedback="worker failed",
        metadata_updates={
            "worker": "grader",
            "resource_usage": {"peak_memory_bytes": 123},
            "gpu_uuids": ["GPU-test"],
        },
    )
    assert finalized.metadata["grader_runs"][0]["run_id"] == run_id
    assert finalized.metadata["grader_runs"][0]["status"] == "crashed"
    assert finalized.metadata["worker"] == "grader"
    assert finalized.metadata["grader_runs"][0]["resource_usage"] == {
        "peak_memory_bytes": 123
    }
    assert finalized.metadata["grader_runs"][0]["gpu_uuids"] == ["GPU-test"]
    assert read_attempt(tmp_path, pending.commit_hash).metadata["grader_runs"] == finalized.metadata["grader_runs"]


def test_retry_cannot_race_running_finalize(tmp_path):
    pending = _attempt(status="pending")
    pending.metadata["grader_runs"] = []
    pending.metadata.pop("failure_class")
    write_attempt(tmp_path, pending)
    _dispatched, run_id = begin_grader_run(tmp_path, pending.commit_hash, None)

    with pytest.raises(TrustedWriteError, match="terminal"):
        retry_attempt(
            tmp_path,
            pending.commit_hash,
            None,
            expected_agent_id="agent-1",
            expected_contract_hash="contract-v1",
            max_infra_retries=2,
        )

    finalized = finalize_grader_run(
        tmp_path,
        pending.commit_hash,
        None,
        run_id,
        status="crashed",
        failure_class="infra",
        score=None,
        feedback="infra",
    )
    assert len(finalized.metadata["grader_runs"]) == 1


def test_finalize_and_retry_race_never_loses_run_history(tmp_path):
    import threading

    pending = _attempt(status="pending")
    pending.metadata["grader_runs"] = []
    pending.metadata.pop("failure_class")
    write_attempt(tmp_path, pending)
    _dispatched, run_id = begin_grader_run(tmp_path, pending.commit_hash, None)
    barrier = threading.Barrier(2)

    def finalize():
        barrier.wait()
        return finalize_grader_run(
            tmp_path,
            pending.commit_hash,
            None,
            run_id,
            status="crashed",
            failure_class="infra",
            score=None,
            feedback="infra",
        )

    def retry():
        barrier.wait()
        try:
            return retry_attempt(
                tmp_path,
                pending.commit_hash,
                None,
                expected_agent_id="agent-1",
                expected_contract_hash="contract-v1",
                max_infra_retries=2,
            )
        except TrustedWriteError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(finalize), pool.submit(retry)]
        [future.result() for future in results]

    current = read_attempt(tmp_path, pending.commit_hash)
    assert current is not None
    assert len(current.metadata["grader_runs"]) == 1
    assert current.metadata["grader_runs"][0]["run_id"] == run_id
    assert current.metadata["grader_runs"][0]["status"] == "crashed"
    assert current.status in {"crashed", "pending"}


def test_dispatch_claim_rejects_second_live_dispatch(tmp_path):
    pending = _attempt(status="pending")
    pending.metadata["grader_runs"] = []
    pending.metadata.pop("failure_class")
    write_attempt(tmp_path, pending)

    _first, run_id = begin_grader_run(tmp_path, pending.commit_hash, None)
    with pytest.raises(TrustedWriteError, match="already running"):
        begin_grader_run(tmp_path, pending.commit_hash, None)

    current = read_attempt(tmp_path, pending.commit_hash)
    assert [run["run_id"] for run in current.metadata["grader_runs"]] == [run_id]
    assert current.metadata["grader_invocation_count"] == 1


def test_stale_dispatch_is_finalized_before_redispatch(tmp_path):
    pending = _attempt(status="pending")
    pending.metadata["grader_runs"] = []
    pending.metadata.pop("failure_class")
    write_attempt(tmp_path, pending)
    dispatched, _run_id = begin_grader_run(tmp_path, pending.commit_hash, None)
    dispatched.metadata["grader_runs"][0]["dispatcher_start_time"] += 1
    write_attempt(tmp_path, dispatched)

    with pytest.raises(TrustedWriteError, match="stale dispatch"):
        begin_grader_run(tmp_path, pending.commit_hash, None)

    current = read_attempt(tmp_path, pending.commit_hash)
    assert current.status == "crashed"
    assert current.metadata["failure_class"] == "infra"
    assert current.metadata["grader_runs"][0]["status"] == "crashed"
