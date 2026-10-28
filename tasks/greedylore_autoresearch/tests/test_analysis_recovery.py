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


def test_published_analysis_preserves_readable_unicode(tmp_path):
    context = _context(tmp_path)
    runner = FakeCodex(context, [{
        "summary": "中文机制分析", "findings": ["引用实测证据"], "limitations": ["单次实验"],
    }])
    runner.mode = "research"
    result = _analyze("N001", "research", context=context, runner=runner,
                      publisher=_publisher(context.notes_root))
    content = result.analysis_path.read_text()
    assert "中文机制分析" in content
    assert json.loads(content)["summary"] == "中文机制分析"


def test_missing_markdown_is_rebuilt_from_json_without_another_model_call(tmp_path):
    from coral.grader.analysis_artifacts import load_analysis, publish_analysis

    context = _context(tmp_path)
    original = _publisher(context.notes_root)

    def publisher(kind, content, metadata):
        if kind == 'analysis':
            path = context.notes_root / 'nodes/N001/research_analysis_v1.md'
            publish_analysis(path, json.loads(content))
            return {'path': 'nodes/N001/research_analysis_v1.md'}
        return original(kind, content, metadata)

    first = _analyze('N001', 'research', context=context, publisher=publisher,
                     runner=lambda *_, **__: _write_analysis_output(context))
    doc = load_analysis(first.analysis_path)
    assert first.analysis_path.read_text().startswith('# N001 Research Analysis')
    first.analysis_path.unlink()
    recovered = _analyze('N001', 'research', context=context, publisher=publisher,
                         runner=lambda *_, **__: pytest.fail('must reuse authoritative JSON'))
    assert recovered.analysis_version == 1
    assert load_analysis(recovered.analysis_path) == doc


def _write_analysis_output(context):
    output = context.notes_root / 'output.json'
    output.write_text(json.dumps(_analysis_payload(None)))
    return output


def test_parent_snapshot_retains_json_needed_by_markdown_link(tmp_path):
    from autoresearch.analysis import _prepare_evidence_snapshot

    from coral.grader.analysis_artifacts import load_analysis, publish_analysis

    context = _context(tmp_path)
    parent = context.notes_root / 'nodes/N000/research_analysis_v1.md'
    publish_analysis(parent, {**_analysis_payload(None), 'node_id': 'N000',
                              'mode': 'research', 'research_sha': 'a' * 40})
    spec = context.notes_root / 'nodes/N001/spec.md'
    values = yaml.safe_load(spec.read_text())
    values['analysis_refs']['parent_research'] = 'nodes/N000/research_analysis_v1.md'
    spec.write_text(yaml.safe_dump(values))
    evidence = resolve_evidence('N001', 'research', context=context)
    snapshot = _prepare_evidence_snapshot(evidence, context, tmp_path / 'snapshot')
    copy = snapshot / 'notes/nodes/N000/research_analysis_v1.md'
    assert copy.read_bytes() == parent.read_bytes()
    assert load_analysis(copy) == load_analysis(parent)


def test_fallback_analyzer_uses_the_same_configured_timeout(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from autoresearch import diagnostic_session

    context = _context(tmp_path)
    monkeypatch.setattr(diagnostic_session, 'settings', lambda _: {
        'analyzer_diagnostics': False, 'analyzer_timeout_seconds': 17,
    })

    def process(command, **kwargs):
        assert kwargs['timeout'] == 17
        output = Path(command[command.index('--output-last-message') + 1])
        output.write_text(json.dumps(_analysis_payload(None)))
        return SimpleNamespace(returncode=0, stderr='')

    monkeypatch.setattr('coral.grader.trusted_worker.run_workflow_model', process)
    result = _analyze('N001', 'research', context=context, publisher=_publisher(context.notes_root))
    assert result.analysis_path.is_file()


def test_diagnostic_runner_keeps_results_and_revokes_identity(tmp_path):
    import os

    from autoresearch.diagnostics import handle_diagnostic
    context = _context(tmp_path)
    # Use an actual SHA-shaped identity; the other legacy fixtures use synthetic 'r'.
    context.attempts[0].commit_hash = 'a' * 40
    observed = {}

    def diagnostic_runner(prompt, roots, schema, scratch, task):
        task.bind(os.getpid())
        observed['task'] = task
        assert scratch == task.workspace
        assert 'diagnostic run' in prompt
        assert '10 successful' in prompt and '20 total' in prompt
        script = scratch / 'experiments/probe.py'
        script.write_text('print("evidence")')
        def execute(item, payload, cancelled):
            cancelled()
            (item.workspace / 'results/probe.json').write_text('{"gap":0.2}')
            return {'status': 'completed', 'started': True, 'exit_code': 0}
        result = handle_diagnostic(
            {'request_id': 'mechanism', 'argv': ['python', 'experiments/probe.py'],
             'gpus': 4, 'timeout': 300},
            {'coral_dir': str(context.coral_dir), 'peer_pid': os.getpid()}, executor=execute)
        output = scratch / 'results/output.json'
        output.write_text(json.dumps({'summary': 'Measured gap.',
            'findings': [f"Diagnostic {result['diagnostic_id']}: gap 0.2."],
            'limitations': ['A small diagnostic, not a quality measurement.']}))
        return output

    result = _analyze('N001', 'research', context=context,
                      publisher=_publisher(context.notes_root), diagnostic_runner=diagnostic_runner)
    assert 'N001-research-v1/mechanism' in result.analysis_path.read_text()
    task = observed['task']
    assert (task.records / 'mechanism/record.json').is_file()
    assert task.read()['process'] is None


class FakeCodex:
    def __init__(self, context, outputs=None):
        self.context = context
        self.outputs = list(outputs or [])
        self.analysis_calls = 0
        self.scratch = []
        self.prompts = []
        self.read_roots = []

    def __call__(self, prompt, read_roots, output_schema, scratch):
        self.prompts.append(prompt)
        self.read_roots.append(tuple(Path(root) for root in read_roots))
        evidence = resolve_evidence("N001", self.mode, context=self.context)
        self.analysis_calls += 1
        default = _analysis_payload(evidence)
        self.scratch.append(Path(scratch))
        value = self.outputs.pop(0) if self.outputs else default
        if isinstance(value, Exception):
            raise value
        if value == "analysis":
            value = _analysis_payload(evidence)
        output = Path(scratch) / "output.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value))
        return output


def _publisher(notes_root):
    def publish(kind, content, context):
        node = notes_root / "nodes" / context["node_id"]
        node.mkdir(parents=True, exist_ok=True)
        if kind == "analysis":
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


def test_analyze_runs_one_session_and_publishes_no_review(tmp_path):
    result, runner, _ = _run(tmp_path)

    assert runner.analysis_calls == 1
    assert result.analysis_path.name == "research_analysis_v1.md"
    assert not list(result.analysis_path.parent.glob("*_review_v*.md"))


@pytest.mark.parametrize("revise", [False, True])
def test_independent_engineering_analysis_is_rejected_before_side_effects(tmp_path, revise):
    context = _context(tmp_path, engineering_count=1)
    evaluation = context.notes_root / "nodes/N001/evaluation.md"
    original = evaluation.read_bytes()

    with pytest.raises(ValueError, match="engineering finish --analysis-file"):
        _analyze("N001", "engineering", revise=revise, context=context,
                 runner=lambda *_: pytest.fail("independent Engineering Analyzer started"),
                 publisher=_publisher(context.notes_root))

    assert evaluation.read_bytes() == original
    assert not list(context.notes_root.glob("nodes/N001/engineering_analysis_v*.md"))


def test_analyze_cli_rejects_engineering_mode(capsys):
    from autoresearch.cli import main

    with pytest.raises(SystemExit) as error:
        main(["analyze", "--node", "N001", "--mode", "engineering"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_engineering_evidence_allows_no_engineering_attempt(tmp_path):
    context = _context(tmp_path)
    evidence = resolve_evidence("N001", "engineering", context=context)

    assert evidence.engineering_attempts == ()
    assert evidence.research_sha == "r" * 40


def test_engineering_evidence_freezes_all_attempts(tmp_path):
    context = _context(tmp_path, engineering_count=3)
    evidence = resolve_evidence("N001", "engineering", context=context)

    assert [item.commit_sha for item in evidence.engineering_attempts] == [
        "0" * 40,
        "1" * 40,
        "2" * 40,
    ]
    assert all(item.eval_refs for item in evidence.engineering_attempts)


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

    assert first.analysis_path.exists()
    assert second.analysis_version == 2
    assert second.analysis_path.name == "research_analysis_v2.md"
    assert first.analysis_path.exists()


def test_default_returns_latest_valid_analysis(tmp_path):
    _, _, context = _run(tmp_path)
    reviser = FakeCodex(context)
    reviser.mode = "research"
    revised = _analyze(
        "N001",
        "research",
        revise=True,
        context=context,
        runner=reviser,
        publisher=_publisher(context.notes_root),
    )
    runner = FakeCodex(context)
    runner.mode = "research"
    result = _analyze(
        "N001",
        "research",
        context=context,
        runner=runner,
        publisher=_publisher(context.notes_root),
    )

    assert revised.analysis_version == result.analysis_version == 2
    assert result.analysis_path == revised.analysis_path
    assert runner.analysis_calls == 0


def test_completed_analysis_is_idempotent(tmp_path):
    context = _context(tmp_path)
    runner = FakeCodex(context)
    runner.mode = "research"
    runner.version = 1
    publisher = _publisher(context.notes_root)

    first = _analyze("N001", "research", context=context, runner=runner, publisher=publisher)
    second = _analyze("N001", "research", context=context, runner=runner, publisher=publisher)

    assert first.analysis_version == second.analysis_version == 1
    assert runner.analysis_calls == 1


def test_published_engineering_analysis_keeps_original_attempt_snapshot(tmp_path):
    from dataclasses import replace

    from autoresearch.analysis import _artifact_evidence
    from autoresearch.validation import load_document, seal_document, validate_analysis

    context = _context(tmp_path, engineering_count=1)
    evidence = resolve_evidence("N001", "engineering", context=context)
    path = context.notes_root / "nodes/N001/engineering_analysis_v1.md"
    path.write_text(json.dumps(seal_document(_analysis_payload(evidence), evidence)))
    original = path.read_bytes()
    later = _attempt(
        "9" * 40,
        "engineering",
        "2026-02-01T00:00:00+00:00",
    )
    updated = replace(context, attempts=(*context.attempts, later))

    document = load_document(path)
    current = resolve_evidence("N001", "engineering", context=updated)
    recovered = _artifact_evidence(document, current, updated)
    validate_analysis(document, recovered)

    assert [item.commit_sha for item in recovered.engineering_attempts] == ["0" * 40]
    assert path.read_bytes() == original


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
    runner = FakeCodex(context)
    runner.mode = "research"
    runner.version = 1
    publisher = _publisher(context.notes_root)

    _analyze("N001", "research", context=context, runner=runner, publisher=publisher)
    raw.write_text('{"value": 2}')

    with pytest.raises(ValueError, match="evidence changed"):
        _analyze("N001", "research", context=context, runner=runner, publisher=publisher)


def test_late_raw_file_is_not_added_to_recovered_snapshot(tmp_path):
    context = _context(tmp_path)
    runner = FakeCodex(context)
    runner.mode = "research"
    runner.version = 1
    publisher = _publisher(context.notes_root)

    _analyze("N001", "research", context=context, runner=runner, publisher=publisher)
    raw = context.log_root / ("r" * 40) / "raw.json"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"late": true}')
    _analyze("N001", "research", context=context, runner=runner, publisher=publisher)

    snapshot_root = next(root for root in runner.read_roots[0] if root.name == "evidence")
    assert not (snapshot_root / f"eval_logs/{'r' * 40}/raw.json").exists()
    assert runner.analysis_calls == 1


def test_missing_raw_evidence_is_recorded_not_rejected(tmp_path):
    context = _context(tmp_path)

    evidence = resolve_evidence("N001", "research", context=context)

    assert any(record.status == "missing" for record in evidence.evidence_records)


def test_analyzer_uses_one_scratch_directory(tmp_path):
    _, runner, _ = _run(tmp_path)

    assert len(runner.scratch) == 1


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


def test_evidence_rejects_spec_for_another_node(tmp_path):
    context = _context(tmp_path)
    spec = context.notes_root / "nodes" / "N001" / "spec.md"
    payload = yaml.safe_load(spec.read_text())
    payload["node_id"] = "N002"
    spec.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="spec"):
        resolve_evidence("N001", "research", context=context)


def test_fixed_snapshot_includes_only_referenced_parent_analysis(tmp_path):
    from autoresearch.analysis import _prepare_evidence_snapshot

    context = _context(tmp_path)
    spec_path = context.notes_root / 'nodes/N001/spec.md'
    spec = yaml.safe_load(spec_path.read_text())
    spec['analysis_refs']['parent_research'] = 'nodes/N000/research_analysis_v1.md'
    spec_path.write_text(yaml.safe_dump(spec))
    parent = context.notes_root / 'nodes/N000/research_analysis_v1.md'
    parent.parent.mkdir()
    parent.write_text('{"summary":"Parent evidence"}')
    (parent.parent / 'unrelated-secret').write_text('not staged')
    evidence = resolve_evidence('N001', 'research', context=context)
    snapshot = _prepare_evidence_snapshot(evidence, context, tmp_path / 'scratch')
    assert (snapshot / 'notes/nodes/N000/research_analysis_v1.md').read_bytes() == parent.read_bytes()
    assert not (snapshot / 'notes/nodes/N000/unrelated-secret').exists()


def test_controlled_analyzer_final_input_and_workspace_are_delivered(tmp_path, monkeypatch):
    import os

    from autoresearch import diagnostic_session

    context = _context(tmp_path)
    context.attempts[0].commit_hash = 'a' * 40
    captured = {}
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'empty-codex-home'))
    monkeypatch.setattr(diagnostic_session.shutil, 'which', lambda name:
                        '/usr/bin/true')
    monkeypatch.setattr(diagnostic_session, 'settings', lambda _: {})
    def record_sandbox(workspace, mounts, gpu_indices, argv, **kwargs):
        captured['sandbox'] = {'workspace': str(workspace), 'mounts': [str(p) for p in mounts],
                               'gpu_indices': gpu_indices, 'argv': argv, **kwargs}
        return ['true', '--', 'true']

    monkeypatch.setattr(diagnostic_session, 'build_sandbox', record_sandbox)

    class InputRecorder:
        pid = os.getpid()
        returncode = 0

        def __init__(self, command, **_kwargs):
            self.command = command
            captured['argv'] = command

        def communicate(self, prompt, timeout=None):
            captured['stdin'] = prompt
            output = Path(self.command[self.command.index('--output-last-message') + 1])
            output.write_text(json.dumps({
                'summary': 'Input-delivery test only.',
                'findings': ['No scientific measurement.'],
                'limitations': ['No actual namespace or GPU execution.'],
            }))
            return '', ''

        def poll(self):
            return 0

    monkeypatch.setattr('coral.grader.trusted_worker.register_workflow_model', lambda *_, **__: None)
    monkeypatch.setattr('coral.grader.trusted_worker.stop_workflow_model', lambda *_, **__: None)
    monkeypatch.setattr(diagnostic_session.subprocess, 'Popen', InputRecorder)
    result = _analyze('N001', 'research', context=context, publisher=_publisher(context.notes_root),
                      diagnostic_runner=diagnostic_session.run_diagnostic_codex)
    prompt = captured['stdin']
    role = (Path(__file__).resolve().parents[1] / 'prompts/research_analysis.md').read_text()
    assert prompt.startswith(role)
    assert '10 successful' in prompt and '20 total' in prompt
    assert '600 seconds' in prompt
    assert 'at most 3600 seconds per queue wait' in prompt
    assert '--timeout 600 --' in prompt
    assert 'resolved relative to the workspace root' in prompt
    sandbox = captured['sandbox']
    workspace = Path(sandbox['workspace'])
    expected_roots = [workspace / 'code', workspace / 'evidence',
                      Path(__file__).resolve().parents[1] / 'references']
    assert prompt.split('Fixed read roots:\n', 1)[1].splitlines() == [str(p).replace(str(workspace), '/workspace') for p in expected_roots]
    assert set(map(str, expected_roots)) <= set(sandbox['mounts'])
    assert sandbox['gpu_indices'] == [] and sandbox['network'] is True
    assert (workspace / 'code' / ('a' * 40) / 'SOURCE_SNAPSHOT').read_text() == 'a' * 40
    assert (workspace / 'evidence/spec.md').read_bytes() == (context.notes_root / 'nodes/N001/spec.md').read_bytes()
    assert (workspace / 'evidence/evaluation.md').read_bytes() == (context.notes_root / 'nodes/N001/evaluation.md').read_bytes()
    command = captured['argv']
    assert '--ephemeral' in command and 'resume' not in command
    assert command[command.index('--sandbox') + 1] == 'workspace-write'
    assert command[command.index('--cd') + 1].endswith('/workspace/results')
    assert command[command.index('--add-dir') + 1].endswith('/workspace/experiments')
    schema = json.loads(Path(command[command.index('--output-schema') + 1]).read_text())
    assert set(schema['required']) == {'summary', 'findings', 'limitations', 'handover'}
    assert result.analysis_path.is_file()
    (tmp_path / 'controlled-analyzer-input.json').write_text(json.dumps(captured, indent=2))
