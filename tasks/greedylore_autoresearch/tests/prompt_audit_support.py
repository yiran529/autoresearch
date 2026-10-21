"""Offline Codex executable boundary: capture delivered inputs, then script outputs.

This module is test-only. It never calls a model or launches a GPU workload.
It captures no environment variables or credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml


def _capture(record: dict) -> None:
    from simulation_support import root

    directory = root() / 'prompt_audit'
    directory.mkdir(exist_ok=True)
    identity = record.get('agent_id') or record.get('node_id') or 'global'
    path = directory / f"{record['role']}-{identity}-{os.getpid()}-{time.monotonic_ns()}.json"
    record['input_sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


def _hash_files(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def _fixed_roots(prompt: str) -> list[Path]:
    marker = 'Read only the following fixed evidence roots:\n'
    tail = prompt.split(marker, 1)[1]
    roots = []
    for line in tail.splitlines():
        if not line.startswith('- '):
            break
        roots.append(Path(line[2:]))
    return roots


def main() -> None:
    from simulation_support import agent, event, gate

    argv = sys.argv[1:]
    if not argv or argv[0] != 'exec':
        raise RuntimeError('audit recorder accepts only codex exec')
    if '--output-schema' not in argv:
        aid = Path('.coral_agent_id').read_text().strip()
        coral_dir = Path(Path('.coral_dir').read_text().strip())
        resumed = argv[1] == 'resume'
        sid = argv[2] if resumed else f'simulation-{aid}'
        prompt = argv[3] if resumed else argv[1]
        toolkit = Path('.codex/notes/toolkit')
        resources = {'workflow_rules.md': (toolkit / 'workflow_rules.md').read_text()}
        resources.update({f'references/{p.name}': p.read_text() for p in (toolkit / 'references').glob('*.md')})
        resources['README.md'] = (toolkit / 'README.md').read_text()
        resources.update({f'templates/{p.name}': p.read_text() for p in (toolkit / 'templates').iterdir() if p.is_file()})
        _capture({'role': 'researcher' if aid.startswith('research-') else 'engineer',
                  'agent_id': aid, 'session_id': sid, 'resumed': resumed, 'argv': argv,
                  'cwd': str(Path.cwd()), 'startup_prompt': prompt,
                  'instructions': Path('AGENTS.md').read_text(),
                  'role_text': Path(f'.codex/roles/{aid}.md').read_text(), 'resources': resources})
        print(json.dumps({'type': 'thread.started', 'thread_id': sid}), flush=True)
        agent(coral_dir, aid)
        return

    prompt = sys.stdin.read()
    schema = json.loads(Path(argv[argv.index('--output-schema') + 1]).read_text())
    output = Path(argv[argv.index('--output-last-message') + 1])
    roots = _fixed_roots(prompt)
    record = {'argv': argv, 'stdin': prompt, 'schema': schema,
              'read_roots': [{'path': str(p), 'exists': p.is_dir()} for p in roots]}
    if set(schema['required']) == {'research_summary', 'engineering_summary'}:
        from autoresearch.memory import list_analysis_refs

        refs = list(list_analysis_refs(roots[0]))
        record.update(role='memory', analysis_refs=refs,
                      analysis_hashes={ref: hashlib.sha256((roots[0] / ref).read_bytes()).hexdigest()
                                       for ref in refs})
        _capture(record)
        event('memory_model', refs=refs)
        summary = ('---\n' + yaml.safe_dump({'processed_analyses': refs})
                   + '---\n# SIMULATED Summary\n\nWorkflow exercise only; no scientific conclusions.\n')
        output.write_text(json.dumps({'research_summary': summary, 'engineering_summary': summary}))
        return

    evidence_root = next(p for p in roots if (p / 'spec.md').is_file())
    code_root = next(p for p in roots if p.name == 'code')
    spec = yaml.safe_load((evidence_root / 'spec.md').read_text())
    match = re.search(r'\{\s*"fixed_evidence"\s*:', prompt)
    envelope, _ = json.JSONDecoder().raw_decode(prompt[match.start():])
    manifest = envelope['fixed_evidence']['manifest']
    if manifest['mode'] != 'research':
        raise RuntimeError('unexpected independent Engineering Analyzer')
    node = spec['node_id']
    record.update(role='research_analyzer', node_id=node, research_sha=manifest['research_sha'],
                  manifest=manifest, code_hashes=_hash_files(code_root),
                  evidence_hashes=_hash_files(evidence_root),
                  code_shas=sorted(p.name for p in code_root.iterdir()),
                  evidence_files=sorted(str(p.relative_to(evidence_root)) for p in evidence_root.rglob('*') if p.is_file()))
    _capture(record)
    event('analyzer', node=node, mode='research', code_snapshots=[str(p) for p in code_root.iterdir()])
    if (os.environ['CORAL_SIM_SCENARIO'] == 'faults' and node == 'N002'
            and not gate('analyzer-failed').exists()):
        gate('analyzer-failed').touch()
        event('analyzer_failure', node=node)
        raise RuntimeError('injected first Analyzer failure')
    attachments = sorted(evidence_root.glob('diagnostics/*/attachments/*/check.log'))
    supporting = []
    for attachment in attachments:
        if 'SIMULATED self-check attachment' not in attachment.read_text():
            raise RuntimeError('supporting attachment was not frozen into Analyzer evidence')
        supporting.append(f'Agent-provided fixture: evidence/{attachment.relative_to(evidence_root)}; not a scientific measurement.')
    output.write_text(json.dumps({
        'summary': f'SIMULATED research result for {node}; no scientific measurement.',
        'findings': ['Only workflow behavior was exercised.', *supporting],
        'limitations': ['No model reasoning, training, correctness, or efficiency validation.'],
        'handover': {
            'mechanism_verdict': 'inconclusive', 'practical_verdict': 'inconclusive',
            'evidence_refs': ['spec.md', 'evaluation.md'],
            'new_information': [],
            'follow_up_questions': [{
                'id': 'Q1', 'question': 'SIMULATED open mechanism question; no scientific finding.',
                'importance': 'Exercises a persistent parent-to-child problem link.',
                'discriminator': 'SIMULATED matched mechanism comparison; not executed scientifically.',
            }],
        },
    }))
