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
from urllib.parse import unquote

import pytest


@pytest.mark.parametrize("scenario", ["normal", "faults"])
def test_complete_workflow_without_models_or_training(tmp_path, scenario):
    import simulation_support

    output = os.environ.get('CORAL_ARTIFACT_REVIEW_DIR')
    root = (Path(output) if output else tmp_path) / scenario
    root.mkdir(parents=True)
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
    assert report["engineering_attempts"] == 9
    assert report["engineering_completed"] == 3
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "analyzer" and e.get("mode") == "research" for e in timeline)
    assert not any(e.get("event") == "analyzer" and e.get("mode") == "engineering" for e in timeline)
    assert report["parallel_research_verified"] is True
    assert report["retired_worktrees"] is True
    assert report["memory_idempotent"] is True
    assert report['wisdom_published'] is True
    from autoresearch.memory import read_processed_analyses
    expected = {f'nodes/{n}/research_analysis_v1.md' for n in ('N000', 'N001', 'N002', 'N003')}
    expected |= {f'nodes/{n}/engineering_analysis_v1.md' for n in ('N000', 'N001', 'N003')}
    assert read_processed_analyses(root / 'run/.coral/public/notes/wisdom.md') == expected
    assert report["authority_rejections"] == ["other_node", "reconcile", "spec", "tree"]
    assert report["blocked_command_invocations"] == []
    assert report["remaining_simulation_processes"] == []
    if scenario == "faults":
        assert report["same_session_resumed"] is True
        assert report["analyzer_failure_recovered"] is True
        assert report["host_binding_verified"] is True
        assert report["infra_retry_same_sha_verified"] is True
    _check_delivered_prompts(root, scenario)
    _check_artifact_handoff(root, report, timeline)
    _check_problem_handover(root, report)
    print(f"\nSimulation report: {root / 'report.json'}")


def _check_problem_handover(root, report):
    import yaml

    public = root / "run/.coral/public"
    notes = public / "notes"
    for node in ("N000", "N001", "N002", "N003"):
        value = json.loads((notes / f"nodes/{node}/handover.json").read_text())
        assert value["legacy"] is False
        assert value["sources"]["research_analysis"]["document_sha256"]
        assert len(value["follow_up_questions"]) <= 2
        assert value["mechanism_verdict"] in {"solved", "not_solved", "partially_solved", "inconclusive"}
        if node != "N000":
            spec = yaml.safe_load((notes / f"nodes/{node}/spec.md").read_text())
            intent = spec["research_intent"]
            snapshot = spec["parent_handover"]["snapshot"]
            assert snapshot["node_id"] == "N000"
            assert intent["engineering_review"]
            assert spec["analysis_refs"]["parent_engineering"].startswith("nodes/N000/engineering_analysis_")
            if spec["research_mode"] == "EXPLORE":
                assert intent["problem_ref"] is None
            else:
                assert intent["problem_ref"] == "N000:Q1"
            assert value["question"] == intent["question"]
            assert value["problem_ref"] == intent["problem_ref"]
    attempts = [json.loads(p.read_text()) for p in (public / "attempts").glob("*.json")]
    for attempt in attempts:
        metadata = attempt["metadata"]
        assert metadata["utility"]["advisory_only"] is True
        assert metadata["evaluation_policy"]["max_relative_eval_loss"] == 1.05
        assert metadata["timing_protocol"]["measured_iterations"] <= 800
        if metadata["node_id"] != "N000":
            assert metadata["timing_declaration"]["period_steps"] == 200
    passing_low_u = [a for a in attempts if a["metadata"]["node_id"] == "N001"
                     and a["metadata"]["mode"] == "research"]
    assert len(passing_low_u) == 1
    assert passing_low_u[0]["metadata"]["utility"]["value"] < 1
    assert passing_low_u[0]["metadata"]["scores"]["quality_gate"]["value"] == 1
    assert any(a["metadata"]["node_id"] == "N001" and a["metadata"]["mode"] == "engineering"
               for a in attempts)
    timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
    reads = [e for e in timeline if e["event"] == "handover_read"]
    assert {e["node"] for e in reads} == {"N001", "N002", "N003"}
    assert all(type(e["engineering_summary_read"]) is bool for e in reads)
    checks = ["structured handover publication and frozen parent snapshots",
              "linked REFINE and independent EXPLORE through actual CLI/trusted service",
              "parent Engineering report pre-read and available Summary pre-read",
              "timing declaration transported from committed submission to grader",
              "real advisory scoring; low-U candidate still receives Engineering",
              "complete Analysis document hashes and separate mechanism/practical verdicts"]
    (root / "handover_audit.json").write_text(json.dumps({
        "simulated": True, "checks_passed": checks,
        "live_model_compliance_verified": False, "scientific_performance_verified": False,
    }, indent=2) + "\n")
    with (root / "coverage.md").open("a") as stream:
        stream.write("\n## New workflow policy coverage\n\n" + "\n".join("- " + c for c in checks) + "\n")


def _check_artifact_handoff(root, report, timeline):
    import yaml

    public = root / 'run/.coral/public'
    tree = (public / 'notes/tree.md').read_text()
    state = json.loads((root / 'run/.coral/private/autoresearch_state.json').read_text())
    assert '| Node | Parent | Research mode |' in tree
    for node_id, record in state['nodes'].items():
        row = next(line for line in tree.splitlines() if line.startswith(f'| [{node_id}]'))
        index = public / f'notes/nodes/{node_id}/README.md'
        if node_id == 'N000':
            assert '| Root | — |' in row
            continue
        spec = index.parent / 'spec.md'
        if not spec.exists():
            assert '| Not selected |' in row
            assert 'Parent: Not selected' in index.read_text()
            continue
        parent = yaml.safe_load(spec.read_text())['parent_node']
        assert f'| [{parent}](nodes/{parent}/README.md) | {record["research"]["research_mode"]} |' in row
        assert f'Parent: [{parent}](../{parent}/README.md)' in index.read_text()
    archived = [e for e in timeline if e.get('event') == 'supporting_evidence_archived']
    assert len(archived) == 6  # Three Researchers and three Engineers.
    assert all(e['scratch_removed'] for e in archived)
    for entry in archived:
        bundle = public / entry['reference']
        manifest = json.loads((bundle / 'manifest.json').read_text())
        assert manifest['execution_verified'] is False
        for file in manifest['files']:
            assert hashlib.sha256((bundle / file['name']).read_bytes()).hexdigest() == file['sha256']
    for node in ('N000', 'N001', 'N002', 'N003'):
        index = public / f'notes/nodes/{node}/README.md'
        links = re.findall(r'\]\(([^)]+)\)', index.read_text())
        assert links
        assert all((index.parent / unquote(link)).resolve().exists() for link in links)
        assert all((index.parent / unquote(link)).resolve().is_relative_to(public.resolve()) for link in links)
        assert f'(nodes/{node}/README.md)' in (public / 'notes/tree.md').read_text()
        for report_path in index.parent.glob('*.md'):
            for link in re.findall(r'\]\(([^)]+)\)', report_path.read_text()):
                assert (report_path.parent / unquote(link)).resolve().exists(), (report_path, link)
    for entry in report['attempts']:
        subprocess.run(['git', '-C', str(root / 'run/repo'), 'cat-file', '-e',
                        entry['commit'] + '^{commit}'], check=True)
    for analysis in public.glob('notes/nodes/*/*_analysis_v*.json'):
        document = json.loads(analysis.read_text())
        if document['node_id'] == 'N000' and document['mode'] == 'research':
            continue
        assert any('/attachments/' in record['path']
                   for record in document['evidence_manifest']['evidence_records'])


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


@pytest.mark.parametrize("worker", ["run_finalization", "run_memory"])
def test_worker_shutdown_records_end_during_active_work(tmp_path, worker, monkeypatch):
    import simulation_support

    monkeypatch.setenv("CORAL_SIM_ROOT", str(tmp_path))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    script = """
import signal, sys
import autoresearch.lifecycle
import autoresearch.memory
import simulation_support

def block(*args, **kwargs):
    signal.pause()

autoresearch.lifecycle.run_worker = block
autoresearch.memory.consolidate_memory = block
getattr(simulation_support, sys.argv[1])(sys.argv[2])
"""
    process = subprocess.Popen([sys.executable, "-c", script, worker, str(tmp_path)], env=env)
    try:
        simulation_support.wait_for(
            lambda: any(e["event"] == "worker_start"
                        for e in simulation_support.read_events(tmp_path)),
            "worker entered active task", timeout=5,
        )
        process.terminate()
        process.wait(timeout=5)
        records = simulation_support.read_events(tmp_path)
        assert [e["event"] for e in records] == ["worker_start", "worker_end"]
        assert not simulation_support.live_processes(tmp_path)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


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
        role_file = 'researcher_legacy' if expected_role == 'researcher' else expected_role
        assert capture['role_text'] == (task / f'roles/{role_file}.md').read_text()
        assert f'.codex/roles/{aid}.md' in capture['instructions']
        assert '.codex/notes/toolkit/' in capture['instructions']
        assert capture['startup_prompt']
        assert 'AGENTS.md' in capture['startup_prompt']
        assert f'.codex/roles/{aid}.md' in capture['startup_prompt']
        assert 'inspect the existing Spec, Evaluation/Attempts and Worktree first' in capture['startup_prompt']
        assert capture['argv'][0] == 'exec'
        assert capture['resources']['workflow_rules.md'] == (task / 'roles/workflow_rules_legacy.md').read_text()
        assert capture['resources']['README.md'] == (task / 'README.md').read_text()
        for template in ('research_intent.json', 'submission.yaml'):
            assert capture['resources'][f'templates/{template}'] == (task / 'templates' / template).read_text()
        for ref in ('common', 'initial_diagnostics', 'low_rank', 'quantization', 'sparsification', 'temporal', 'hybrid', 'engineer_runtime', 'semantic_criteria', 'algorithm_construction'):
            assert capture['resources'][f'references/{ref}.md'] == (task / f'references/{ref}.md').read_text()
        if expected_role == 'researcher':
            node = aid.removeprefix('research-')
            mode = state['nodes'][node]['research'].get('research_mode', 'REFINE')
            assert f'`{mode}`' in capture['instructions']
            assert f'`{node}`' in capture['instructions']
            assert '3 successful scored formal Research runs' in capture['instructions']
            assert 'autoresearch node complete --mode research' in capture['instructions']
            assert '10 total formal Research runs' in capture['startup_prompt']
            assert 'first terminal result' in capture['startup_prompt']
        else:
            node = state['engineering_workers'][aid]['node_id']
            assert f'`{node}`' in capture['startup_prompt']
            assert 'at most 4 successful formal Engineering runs and 10 total formal Engineering runs' in capture['startup_prompt']
            assert 'do not wait for Research Analysis' in capture['startup_prompt']
            assert 'continue while evidence supports a clear, substantial end-to-end improvement opportunity' in capture['startup_prompt']
            assert 'one improved candidate is not sufficient to stop' in capture['role_text']
            assert '--analysis-file /absolute/scratch/final.json' in capture['instructions']
            assert 'author the final engineering analysis yourself' in capture['role_text'].lower()
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
        assert set(capture['schema']['required']) == {'summary', 'findings', 'limitations', 'handover'}
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
        assert set(capture['schema']['required']) == {'research_summary', 'engineering_summary', 'wisdom'}
        assert '--ephemeral' in capture['argv'] and 'resume' not in capture['argv']
        assert all('/private/' not in r['path'] for r in capture['read_roots'])
        assert capture['analysis_refs']
        assert set(capture['analysis_hashes']) == set(capture['analysis_refs'])
        for ref, digest in capture['analysis_hashes'].items():
            source = root / 'run/.coral/public/notes' / ref
            assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
        memory_prompt = (task / 'prompts/global_memory_manager.md').read_text()
        trigger = json.JSONDecoder().raw_decode(capture['stdin'][len(memory_prompt):].lstrip())[0]
        assert trigger['trigger_refs']
        assert set(trigger['trigger_refs']) <= set(capture['analysis_refs'])
        assert set(trigger['required_refs']) == set(trigger['trigger_refs'])
        assert set(trigger['required_refs']) <= set(trigger['covered_refs'])
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
