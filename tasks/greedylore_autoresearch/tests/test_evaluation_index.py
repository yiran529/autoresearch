from __future__ import annotations

import json
import re
from urllib.parse import unquote

import pytest
from autoresearch.evaluation import refresh_node_evaluation

from coral.types import Attempt


def _attempt(sha: str, mode: str, timestamp: str) -> Attempt:
    return Attempt(
        commit_hash=sha,
        agent_id="agent",
        title=mode,
        score=1.25,
        status="improved",
        parent_hash=None,
        timestamp=timestamp,
        metadata={
            "mode": mode,
            "node_id": "N001",
            "contract_hash": "contract-v1",
            "quality_anchor": "r" * 40 if mode == "engineering" else None,
            "grader_runs": [
                {
                    "run_id": f"run-{sha[0]}",
                    "status": "improved",
                    "failure_class": None,
                    "log_dir": f"eval_logs/{sha}/run-{sha[0]}",
                }
            ],
            "scores": {"efficiency_score": {"value": 1.25}},
            "evidence_paths": [f"eval_logs/{sha}/raw"],
            "comparable": True,
        },
    )


@pytest.mark.parametrize("engineering_count", [0, 2])
def test_evaluation_rebuild_is_deterministic(tmp_path, engineering_count):
    research = _attempt("r" * 40, "research", "2026-01-01T00:00:00+00:00")
    engineering = [
        _attempt(str(index) * 40, "engineering", f"2026-01-0{index + 2}T00:00:00+00:00")
        for index in range(engineering_count)
    ]
    for attempt in [research, *engineering]:
        raw = tmp_path / attempt.commit_hash / f"run-{attempt.commit_hash[0]}"
        raw.mkdir(parents=True)
        (raw / "result.json").write_text(json.dumps({"contract_hash": "contract-v1"}))

    first = refresh_node_evaluation("N001", [research, *engineering], tmp_path)
    second = refresh_node_evaluation("N001", [*reversed(engineering), research], tmp_path)

    assert first == second
    assert first.count("| engineering |") == engineering_count
    assert "run-r" in first


def test_multiple_research_shas_show_pending_selection(tmp_path):
    attempts = [
        _attempt("a" * 40, "research", "2026-01-01T00:00:00+00:00"),
        _attempt("b" * 40, "research", "2026-01-02T00:00:00+00:00"),
    ]
    report = refresh_node_evaluation("N001", attempts, tmp_path)
    assert "Selected Research SHA: pending Research finish." in report
    assert report.count("| research |") == 2


def test_evaluation_reads_submission_from_fixed_commit(tmp_path):
    import subprocess

    import yaml

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ("init",),
        ("config", "user.name", "Test"),
        ("config", "user.email", "test@example.com"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "submission.yaml").write_text(yaml.safe_dump({"mode": "research", "node_id": "N001"}))
    subprocess.run(["git", "add", "submission.yaml"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "research"], cwd=repo, check=True, capture_output=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    attempt = _attempt(sha, "research", "2026-01-01T00:00:00+00:00")
    attempt.metadata["source_worktree"] = str(repo)
    (repo / "submission.yaml").write_text("mode: forged\n")

    rebuilt = refresh_node_evaluation("N001", [attempt], tmp_path / "logs")

    assert "research:N001" in rebuilt


def test_evaluation_preserves_retry_status_and_failure_class(tmp_path):
    attempt = _attempt("r" * 40, "research", "2026-01-01T00:00:00+00:00")
    attempt.metadata["grader_runs"] = [
        {"run_id": "first", "status": "crashed", "failure_class": "infra"},
        {"run_id": "second", "status": "improved", "failure_class": None},
    ]

    rebuilt = refresh_node_evaluation("N001", [attempt], tmp_path)

    assert "first:crashed/infra" in rebuilt
    assert "second:improved" in rebuilt


@pytest.mark.parametrize("canonical", [False, True])
def test_evaluation_survives_retired_source_worktree(tmp_path, canonical):
    attempt = _attempt("a" * 40, "research", "2026-01-01T00:00:00+00:00")
    attempt.metadata["source_worktree"] = str(tmp_path / "retired-agent")
    if canonical:
        attempt.metadata["submission_hash"] = "b" * 64
    rebuilt = refresh_node_evaluation("N001", [attempt], tmp_path / "logs")
    assert "| research |" in rebuilt
    if canonical:
        assert "research:N001" in rebuilt


def test_evaluation_links_resolve_without_enumerating_every_result(tmp_path):
    public = tmp_path / 'public'
    node = public / 'notes/nodes/N001'
    node.mkdir(parents=True)
    attempt = _attempt('a' * 40, 'engineering', '2026-01-01T00:00:00+00:00')
    run = public / 'eval_logs' / attempt.commit_hash / 'run-a'
    run.mkdir(parents=True)
    for number in range(40):
        (run / f'result-{number}.json').write_text('{}')
    records = public / 'attempts'
    records.mkdir()
    (records / f'{attempt.commit_hash}.json').write_text('{}')
    attempt.metadata['evidence_paths'] = [str(p.relative_to(public)) for p in run.iterdir()]

    report = refresh_node_evaluation('N001', [attempt], public / 'eval_logs')
    links = re.findall(r'\]\(([^)]+)\)', report)
    targets = [(node / unquote(link)).resolve() for link in links]
    assert (records / f'{attempt.commit_hash}.json').resolve() in targets
    assert run.resolve() in targets
    assert all(p.exists() for p in targets)
    assert len(links) <= 4
    assert 'result-39.json' not in report


def test_evaluation_displays_cross_topology_warning(tmp_path):
    attempt = _attempt('a' * 40, 'research', '2026-01-01T00:00:00+00:00')
    attempt.metadata.update(comparability='cross_topology', hardware_comparison={
        'warnings': ['topology_fingerprint differs; comparison allowed.'],
    })
    report = refresh_node_evaluation('N001', [attempt], tmp_path)
    assert 'cross_topology' in report
    assert 'topology_fingerprint differs' in report
