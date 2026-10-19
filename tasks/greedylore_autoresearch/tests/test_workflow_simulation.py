"""Exercise the real workflow with scripted processes and synthetic scientific results."""

# ruff: noqa: S101

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario", ["normal", "faults"])
def test_complete_workflow_without_models_or_training(tmp_path, scenario):
    import simulation_support

    root = tmp_path / scenario
    root.mkdir()
    env = os.environ.copy()
    task = Path(__file__).resolve().parents[1]
    repo = task.parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).parent),
            str(task / "autoresearch/src"),
            str(task / "grader/src"),
            str(repo),
            env.get("PYTHONPATH", ""),
        ]
    )
    log = root / "driver.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(simulation_support.__file__)), "driver", str(root), scenario],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            try:
                code = process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                pytest.fail(f"simulation timed out; inspect {log}\n{log.read_text()[-6000:]}")
        finally:
            simulation_support.cleanup_processes(root)
    assert code == 0, f"simulation failed; inspect {log}\n{log.read_text()[-6000:]}"
    report = json.loads((root / "report.json").read_text())
    assert report["simulated"] is True
    assert report["completed_research"] == ["N001", "N002", "N003"]
    assert report["quality_failed"] == ["N002"]
    assert report["engineering_attempts"] == 2
    assert report["engineering_completed"] == 1
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "analyzer" and e.get("mode") == "research" for e in timeline)
    assert not any(e.get("event") == "analyzer" and e.get("mode") == "engineering" for e in timeline)
    assert report["parallel_research_verified"] is True
    assert report["retired_worktrees"] is True
    assert report["memory_idempotent"] is True
    assert report["authority_rejections"] == ["other_node", "reconcile", "spec", "tree"]
    assert report["blocked_command_invocations"] == []
    assert report["remaining_simulation_processes"] == []
    if scenario == "faults":
        assert report["same_session_resumed"] is True
        assert report["analyzer_failure_recovered"] is True
        assert report["claim_conflict_verified"] is True
        assert report["infra_retry_same_sha_verified"] is True
    _check_delivered_prompts(root, scenario)
    print(f"\nSimulation report: {root / 'report.json'}")


def test_cleanup_tracks_spawned_worker_before_its_first_event(tmp_path):
    import simulation_support

    from coral.grader.trusted_writes import process_start_time

    # The parent has launched a child, but the child has not imported its task yet.
    # A recorded, matching PID must remain visible even without worker_start.
    (tmp_path / 'timeline.jsonl').write_text(json.dumps({
        'event': 'worker_spawn', 'child_pid': os.getpid(),
        'child_start': process_start_time(os.getpid()),
    }) + '\n')
    assert simulation_support.live_processes(tmp_path) == [os.getpid()]


def _check_delivered_prompts(root, scenario):
    task = Path(__file__).resolve().parents[1]
    capture_paths = sorted((root / 'prompt_audit').glob('*.json'))
    captures = [json.loads(p.read_text()) for p in capture_paths]
    assert captures, 'No actual CLI-boundary prompt capture was produced'
    assert {p['role'] for p in captures} == {'researcher', 'engineer', 'research_analyzer', 'memory'}
    state = json.loads((root / 'run/.coral/private/autoresearch_state.json').read_text())
    producers = [p for p in captures if p['role'] in {'researcher', 'engineer'}]
    for capture in producers:
        aid = capture['agent_id']
        expected_role = 'researcher' if aid.startswith('research-') else 'engineer'
        assert capture['role_text'] == (task / f'roles/{expected_role}.md').read_text()
        assert f'.codex/roles/{aid}.md' in capture['instructions']
        assert '.codex/notes/toolkit/' in capture['instructions']
        assert capture['startup_prompt']
        assert capture['argv'][0] == 'exec'
        assert capture['resources']['workflow_rules.md'] == (task / 'roles/workflow_rules.md').read_text()
        for ref in ('common', 'initial_diagnostics', 'low_rank', 'quantization', 'sparsification', 'temporal', 'hybrid', 'engineer_runtime', 'semantic_criteria', 'system_ref'):
            assert capture['resources'][f'references/{ref}.md'] == (task / f'references/{ref}.md').read_text()
        if expected_role == 'researcher':
            node = aid.removeprefix('research-')
            mode = state['nodes'][node]['research'].get('research_mode', 'REFINE')
            assert f'`{mode}`' in capture['instructions']
            assert f'`{node}`' in capture['instructions']
            assert 'one formal Research evaluation' in capture['instructions']
        else:
            assert '--analysis-file /absolute/scratch/final.json' in capture['instructions']
            assert 'author the final Engineering Analysis yourself' in capture['role_text']
    assert {state['nodes'][p['agent_id'].removeprefix('research-')]['research']['research_mode']
            for p in producers if p['role'] == 'researcher'} == {'REFINE', 'EXPLORE'}
    analyzers = [p for p in captures if p['role'] == 'research_analyzer']
    assert {p['node_id'] for p in analyzers} >= {'N000', 'N001', 'N002', 'N003'}
    for capture in analyzers:
        assert capture['stdin'].startswith((task / 'prompts/research_analysis.md').read_text())
        assert 'each at most 600 seconds' in capture['stdin']
        assert 'no new algorithms' in capture['stdin']
        assert '--ephemeral' in capture['argv'] and 'resume' not in capture['argv']
        assert capture['argv'][capture['argv'].index('--sandbox') + 1] == 'read-only'
        assert set(capture['schema']['required']) == {'summary', 'findings', 'limitations'}
        assert all(r['exists'] and '/private/' not in r['path'] for r in capture['read_roots'])
        assert set(capture['code_shas']) == {capture['research_sha']}
        archive_bytes = subprocess.check_output(
            ['git', '-C', str(root / 'run/repo'), 'archive', capture['research_sha']])
        with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
            expected_code = {f"{capture['research_sha']}/{entry.name}":
                             hashlib.sha256(archive.extractfile(entry).read()).hexdigest()
                             for entry in archive if entry.isfile()}
        assert capture['code_hashes'] == expected_code
        manifest = capture['manifest']
        expected_evidence = {'spec.md': manifest['spec_sha256'],
                             'evaluation.md': manifest['evaluation_version']}
        expected_evidence.update({entry['path']: entry['sha256']
                                  for entry in manifest['evidence_records']
                                  if entry['status'] == 'present'})
        assert capture['evidence_hashes'] == expected_evidence
        assert str(task / 'references') in {r['path'] for r in capture['read_roots']}
        if capture['node_id'] != 'N000':
            assert 'notes/nodes/N000/research_analysis_v1.md' in capture['evidence_files']
    for capture in (p for p in captures if p['role'] == 'memory'):
        assert capture['stdin'].startswith((task / 'prompts/global_memory_manager.md').read_text())
        assert set(capture['schema']['required']) == {'research_summary', 'engineering_summary'}
        assert 'new mechanistic evidence' in capture['stdin']
        assert '--ephemeral' in capture['argv'] and 'resume' not in capture['argv']
        assert all('/private/' not in r['path'] for r in capture['read_roots'])
        assert capture['analysis_refs']
        assert set(capture['analysis_hashes']) == set(capture['analysis_refs'])
        for ref, digest in capture['analysis_hashes'].items():
            source = root / 'run/.coral/public/notes' / ref
            assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
        trigger = json.JSONDecoder().raw_decode(
            capture['stdin'][re.search(r'\{\s*"engineering_summary"\s*:', capture['stdin']).start():])[0]
        assert trigger['trigger_research_refs']
        assert set(trigger['trigger_research_refs']) <= set(capture['analysis_refs'])
    if scenario == 'faults':
        resumed = [p for p in producers if p['agent_id'] == 'research-N002']
        assert len(resumed) >= 2
        assert resumed[0]['instructions'] == resumed[1]['instructions']
        assert resumed[0]['role_text'] == resumed[1]['role_text']
        assert resumed[1]['argv'][1:3] == ['resume', resumed[0]['session_id']]

    index = '\n'.join(f"| {p['role']} | {p.get('agent_id', p.get('node_id', 'global'))} | {p.get('resumed', 'n/a')} | [{path.name}]({path.name}) |"
                      for path, p in zip(capture_paths, captures))
    (root / 'prompt_audit/CHECKS.md').write_text(
        '# Prompt delivery audit: PASS\n\n'
        'Captured at the substituted Codex executable boundary using the actual CodexRuntime '
        'and one-shot launcher. Checked initial/resumed Researcher and Engineer instructions, '
        'pinned REFINE/EXPLORE modes, all task references, Research Analyzer role/schema/frozen '
        'evidence, and Memory role/schema/staged sources. No independent Engineering Analyzer.\n\n'
        'This verifies application input delivery, not live Codex instruction loading, model '
        'compliance, OS isolation or scientific effectiveness. Full inputs are in adjacent JSON files.\n\n'
        '| Role | Agent / Node | Resumed | Captured input |\n| --- | --- | --- | --- |\n' + index + '\n')
