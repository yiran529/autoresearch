from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from autoresearch.analysis import AnalysisContext, analyze, resolve_evidence

from coral.types import Attempt


def _fake_code_snapshot(evidence, _context, scratch):
    root = Path(scratch) / "code"
    commits = [evidence.research_sha, *(a.commit_sha for a in evidence.engineering_attempts)]
    for commit in commits:
        target = root / commit
        target.mkdir(parents=True, exist_ok=True)
        (target / "SOURCE_SNAPSHOT").write_text(commit)
    return root


def _analyze(*args, **kwargs):
    kwargs.setdefault("code_snapshotter", _fake_code_snapshot)
    return analyze(*args, **kwargs)


def _attempt(sha: str, mode: str, timestamp: str, node_id: str = "N001") -> Attempt:
    return Attempt(
        commit_hash=sha,
        agent_id="agent",
        title=mode,
        score=1.0,
        status="improved",
        parent_hash=None,
        timestamp=timestamp,
        metadata={
            "mode": mode,
            "node_id": node_id,
            "contract_hash": "contract-v1",
            "comparability": "comparable",
            "grader_runs": [
                {
                    "run_id": f"run-{sha}",
                    "status": "improved",
                    "failure_class": None,
                    "log_dir": f"eval_logs/{sha}/run-{sha}",
                }
            ],
            "evidence_paths": [f"eval_logs/{sha}/raw.json"],
        },
    )


def _context(tmp_path: Path, engineering_count: int = 0) -> AnalysisContext:
    notes = tmp_path / "notes"
    node = notes / "nodes" / "N001"
    node.mkdir(parents=True)
    (node / "spec.md").write_text(
        yaml.safe_dump(
            {
                "node_id": "N001",
                "base_commit": "b" * 40,
                "parent_commit": "p" * 40,
                "analysis_refs": {
                    "parent_research": None,
                    "parent_engineering": None,
                },
                "review_refs": {
                    "parent_research": None,
                    "parent_engineering": None,
                },
            }
        )
    )
    (node / "evaluation.md").write_text("# Evaluation N001\n")
    attempts = [_attempt("r" * 40, "research", "2026-01-01T00:00:00+00:00")]
    attempts.extend(
        _attempt(
            str(index) * 40,
            "engineering",
            f"2026-01-{index + 2:02d}T00:00:00+00:00",
        )
        for index in range(engineering_count)
    )
    return AnalysisContext(
        coral_dir=tmp_path / ".coral",
        notes_root=notes,
        island_id=None,
        attempts=tuple(attempts),
        log_root=tmp_path / "eval_logs",
        checkout_root=tmp_path / "repo",
    )


def _analysis_payload(_evidence):
    return {
        "summary": "Evidence-bounded summary.",
        "findings": ["Measured evidence only."],
        "limitations": ["No unmeasured claims."],
    }


def _review_payload(_evidence, version, decision="pass"):
    return {
        "analysis_version": version,
        "workflow_checks": ["fresh analyzer and reviewer sessions"],
        "evidence_checks": ["fixed evidence manifest"],
        "issues": [],
        "actions": [],
        "decision": decision,
    }


class FakeCodex:
    def __init__(self, context, outputs=None):
        self.context = context
        self.outputs = list(outputs or [])
        self.analysis_calls = 0
        self.review_calls = 0
        self.scratch = []
        self.prompts = []
        self.read_roots = []

    def __call__(self, prompt, read_roots, output_schema, scratch):
        self.prompts.append(prompt)
        self.read_roots.append(tuple(Path(root) for root in read_roots))
        evidence = resolve_evidence("N001", self.mode, context=self.context)
        is_review = Path(output_schema).name.startswith("review")
        if is_review:
            self.review_calls += 1
            default = _review_payload(evidence, self.version)
        else:
            self.analysis_calls += 1
            default = _analysis_payload(evidence)
        self.scratch.append(Path(scratch))
        value = self.outputs.pop(0) if self.outputs else default
        if isinstance(value, Exception):
            raise value
        if value == "analysis":
            value = _analysis_payload(evidence)
        elif value == "review":
            value = _review_payload(evidence, self.version)
        elif value == "review_fail":
            value = _review_payload(evidence, self.version, decision="fail")
        output = Path(scratch) / "output.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value))
        return output


def _publisher(notes_root):
    def publish(kind, content, context):
        node = notes_root / "nodes" / context["node_id"]
        node.mkdir(parents=True, exist_ok=True)
        if kind in {"analysis", "review"}:
            target = node / f"{context['mode']}_{kind}_v{context['version']}.md"
            if target.exists():
                raise RuntimeError("immutable artifact already exists")
        elif kind == "evaluation":
            target = node / "evaluation.md"
        else:
            raise AssertionError(kind)
        target.write_text(content)
        return {"path": str(target.relative_to(notes_root))}

    return publish


def _run(tmp_path, mode="research", engineering_count=0, outputs=None, revise=False):
    context = _context(tmp_path, engineering_count)
    runner = FakeCodex(context, outputs)
    runner.mode = mode
    runner.version = 2 if revise else 1
    result = _analyze(
        "N001",
        mode,
        revise=revise,
        context=context,
        runner=runner,
        publisher=_publisher(context.notes_root),
    )
    return result, runner, context


def test_review_failure_only_retries_missing_review(tmp_path):
    context = _context(tmp_path)
    runner = FakeCodex(context, ["analysis", RuntimeError("review crashed"), "review"])
    runner.mode = "research"
    runner.version = 1
    publisher = _publisher(context.notes_root)

    with pytest.raises(RuntimeError, match="review crashed"):
        _analyze(
            "N001",
            "research",
            context=context,
            runner=runner,
            publisher=publisher,
        )
    result = _analyze(
        "N001",
        "research",
        context=context,
        runner=runner,
        publisher=publisher,
    )

    assert runner.analysis_calls == 1
    assert runner.review_calls == 2
    assert result.analysis_version == 1


def test_engineering_analysis_allows_no_engineering_attempt(tmp_path):
    result, _, context = _run(tmp_path, mode="engineering")

    assert result.evidence.engineering_attempts == ()
    evaluation = context.notes_root / "nodes" / "N001" / "evaluation.md"
    assert "r" * 40 in evaluation.read_text()


def test_engineering_analysis_freezes_all_attempts(tmp_path):
    result, _, _ = _run(tmp_path, mode="engineering", engineering_count=3)

    assert [item.commit_sha for item in result.evidence.engineering_attempts] == [
        "0" * 40,
        "1" * 40,
        "2" * 40,
    ]
    assert all(item.eval_refs for item in result.evidence.engineering_attempts)


def test_research_analysis_excludes_engineering_attempts(tmp_path):
    result, _, _ = _run(tmp_path, mode="research", engineering_count=2)

    assert result.evidence.engineering_attempts == ()


def test_research_snapshot_and_roots_exclude_engineering_evidence(tmp_path):
    result, runner, context = _run(tmp_path, mode="research", engineering_count=1)

    assert "0" * 40 not in result.evidence.evaluation_snapshot
    assert context.log_root not in runner.read_roots[0]
    assert context.notes_root / "nodes/N001" not in runner.read_roots[0]
    assert context.checkout_root not in runner.read_roots[0]
    assert str(context.notes_root) not in runner.prompts[0]
    code_root = next(root for root in runner.read_roots[0] if root.name == "code")
    assert (code_root / result.evidence.research_sha / "SOURCE_SNAPSHOT").is_file()
    snapshot_root = next(root for root in runner.read_roots[0] if root.name == "evidence")
    assert "0" * 40 not in (snapshot_root / "evaluation.md").read_text()


def test_revise_keeps_v1_and_publishes_v2(tmp_path):
    first, _, context = _run(tmp_path)
    runner = FakeCodex(context)
    runner.mode = "research"
    runner.version = 2
    second = _analyze(
        "N001",
        "research",
        revise=True,
        context=context,
        runner=runner,
        publisher=_publisher(context.notes_root),
    )

    assert first.analysis_path.exists() and first.review_path.exists()
    assert second.analysis_version == second.review_version == 2
    assert first.analysis_path.exists() and first.review_path.exists()


def test_completed_fail_review_is_idempotent_not_passed(tmp_path):
    context = _context(tmp_path)
    runner = FakeCodex(context)
    runner.mode = "research"
    runner.version = 1
    runner.outputs = ["analysis", "review_fail"]
    publisher = _publisher(context.notes_root)

    first = _analyze("N001", "research", context=context, runner=runner, publisher=publisher)
    second = _analyze("N001", "research", context=context, runner=runner, publisher=publisher)

    assert first.decision == second.decision == "fail"
    assert runner.analysis_calls == runner.review_calls == 1


def test_published_engineering_analysis_keeps_original_attempt_snapshot(tmp_path):
    from dataclasses import replace

    first, runner, context = _run(tmp_path, mode="engineering", engineering_count=1)
    later = _attempt(
        "9" * 40,
        "engineering",
        "2026-02-01T00:00:00+00:00",
    )
    updated = replace(context, attempts=(*context.attempts, later))

    second = _analyze(
        "N001",
        "engineering",
        context=updated,
        runner=runner,
        publisher=_publisher(context.notes_root),
    )

    assert [item.commit_sha for item in second.evidence.engineering_attempts] == ["0" * 40]
    assert second.analysis_version == 1
    assert runner.analysis_calls == runner.review_calls == 1


def test_missing_review_uses_published_evaluation_snapshot(tmp_path):
    from dataclasses import replace

    context = _context(tmp_path, engineering_count=1)
    runner = FakeCodex(context, ["analysis", RuntimeError("review crashed")])
    runner.mode = "engineering"
    runner.version = 1
    publisher = _publisher(context.notes_root)

    with pytest.raises(RuntimeError, match="review crashed"):
        _analyze("N001", "engineering", context=context, runner=runner, publisher=publisher)

    later = _attempt("9" * 40, "engineering", "2026-02-01T00:00:00+00:00")
    updated = replace(context, attempts=(*context.attempts, later))
    _analyze("N001", "engineering", context=updated, runner=runner, publisher=publisher)

    assert "9" * 40 in (context.notes_root / "nodes/N001/evaluation.md").read_text()
    assert "9" * 40 not in runner.prompts[-1]
    snapshot = runner.scratch[-1] / "evidence/evaluation.md"
    assert snapshot.is_file()
    assert "9" * 40 not in snapshot.read_text()
    assert context.notes_root / "nodes/N001" not in runner.read_roots[-1]


def test_recovery_rejects_tampered_manifest(tmp_path):
    first, _, context = _run(tmp_path)
    document = json.loads(first.analysis_path.read_text())
    document["evidence_manifest"]["contract_hash"] = "tampered"
    first.analysis_path.write_text(json.dumps(document))
    runner = FakeCodex(context)
    runner.mode = "research"
    runner.version = 1

    with pytest.raises(ValueError, match="manifest hash"):
        _analyze(
            "N001",
            "research",
            context=context,
            runner=runner,
            publisher=_publisher(context.notes_root),
        )


def test_raw_evidence_content_changes_manifest(tmp_path):
    context = _context(tmp_path)
    raw = context.log_root / ("r" * 40) / "raw.json"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"value": 1}')

    first = resolve_evidence("N001", "research", context=context)
    raw.write_text('{"value": 2}')
    second = resolve_evidence("N001", "research", context=context)

    assert first.manifest_hash != second.manifest_hash
    assert first.evidence_records[0].sha256 != second.evidence_records[0].sha256


def test_recovery_rejects_changed_raw_evidence(tmp_path):
    context = _context(tmp_path)
    raw = context.log_root / ("r" * 40) / "raw.json"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"value": 1}')
    runner = FakeCodex(context, ["analysis", RuntimeError("review crashed")])
    runner.mode = "research"
    runner.version = 1
    publisher = _publisher(context.notes_root)

    with pytest.raises(RuntimeError, match="review crashed"):
        _analyze("N001", "research", context=context, runner=runner, publisher=publisher)
    raw.write_text('{"value": 2}')

    with pytest.raises(ValueError, match="evidence changed"):
        _analyze("N001", "research", context=context, runner=runner, publisher=publisher)


def test_late_raw_file_is_not_added_to_recovered_snapshot(tmp_path):
    context = _context(tmp_path)
    runner = FakeCodex(context, ["analysis", RuntimeError("review crashed")])
    runner.mode = "research"
    runner.version = 1
    publisher = _publisher(context.notes_root)

    with pytest.raises(RuntimeError, match="review crashed"):
        _analyze("N001", "research", context=context, runner=runner, publisher=publisher)
    raw = context.log_root / ("r" * 40) / "raw.json"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"late": true}')
    _analyze("N001", "research", context=context, runner=runner, publisher=publisher)

    snapshot_root = next(root for root in runner.read_roots[-1] if root.name == "evidence")
    assert not (snapshot_root / f"eval_logs/{'r' * 40}/raw.json").exists()
    assert context.log_root not in runner.read_roots[-1]


def test_missing_raw_evidence_is_recorded_not_rejected(tmp_path):
    context = _context(tmp_path)

    evidence = resolve_evidence("N001", "research", context=context)

    assert any(record.status == "missing" for record in evidence.evidence_records)


def test_analyzer_and_reviewer_use_distinct_scratch_directories(tmp_path):
    _, runner, _ = _run(tmp_path)

    assert len(runner.scratch) == 2
    assert runner.scratch[0] != runner.scratch[1]


def test_research_analysis_accepts_terminal_failed_attempt(tmp_path):
    context = _context(tmp_path)
    failed = context.attempts[0]
    failed.status = "crashed"
    failed.metadata["failure_class"] = "candidate"
    runner = FakeCodex(context)
    runner.mode = "research"
    runner.version = 1

    result = _analyze(
        "N001",
        "research",
        context=context,
        runner=runner,
        publisher=_publisher(context.notes_root),
    )

    assert result.evidence.research_sha == "r" * 40


def test_review_requires_actual_workflow_and_evidence_checks(tmp_path):
    context = _context(tmp_path)
    runner = FakeCodex(context)
    runner.mode = "research"
    runner.version = 1
    runner.outputs = ["analysis", "review"]
    original_call = runner.__call__

    def invalid_review(prompt, read_roots, output_schema, scratch):
        output = original_call(prompt, read_roots, output_schema, scratch)
        if Path(output_schema).name.startswith("review"):
            payload = json.loads(output.read_text())
            payload["workflow_checks"] = []
            payload["evidence_checks"] = []
            output.write_text(json.dumps(payload))
        return output

    with pytest.raises(ValueError, match="workflow_checks"):
        _analyze(
            "N001",
            "research",
            context=context,
            runner=invalid_review,
            publisher=_publisher(context.notes_root),
        )

    assert not (context.notes_root / "nodes" / "N001" / "research_review_v1.md").exists()


def test_evidence_rejects_spec_for_another_node(tmp_path):
    context = _context(tmp_path)
    spec = context.notes_root / "nodes" / "N001" / "spec.md"
    payload = yaml.safe_load(spec.read_text())
    payload["node_id"] = "N002"
    spec.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="spec"):
        resolve_evidence("N001", "research", context=context)
