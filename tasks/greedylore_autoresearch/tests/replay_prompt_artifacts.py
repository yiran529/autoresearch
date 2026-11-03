# ruff: noqa: S101
"""Offline prompt replay from published Analyses; never invokes a model or GPU.

Copies hash-verified text evidence into --output; the source Run is read-only.
Published manifests can include post-analysis citations, so this is a realistic
payload replay, not a reconstruction of the exact historical opening message.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(TASK / 'autoresearch/src'), str(TASK / 'grader/src')]

from autoresearch import analysis, evaluation_context  # noqa: E402
from autoresearch.artifact_paths import diagnostic_view  # noqa: E402


def baseline_module(name, revision):
    path = Path(analysis.__file__).with_name(name + '.py')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                     f'{revision}:{path.relative_to(ROOT)}'], text=True)
    module = types.ModuleType('autoresearch._before_' + name)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def replay(run, output, revision):
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', revision], text=True).strip()
    before = baseline_module('analysis', revision)
    before_context = baseline_module('evaluation_context', revision)
    public = run / '.coral/public'
    rows = []
    for artifact in sorted(public.glob('notes/nodes/*/research_analysis_v*.json')):
        document = json.loads(artifact.read_text())
        manifest = document['evidence_manifest']
        case = output / (artifact.parent.name + '-' + artifact.stem)
        case.mkdir(parents=True, exist_ok=False)
        source = case / 'source/.coral/public'
        for record in manifest['evidence_records']:
            if record['status'] != 'present':
                continue
            ref = record['path']
            relative = Path(ref)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe artifact path')
            if ref.startswith('code/'):
                _, sha, name = ref.split('/', 2)
                data = subprocess.check_output(['git', '-C', str(run / 'repo'), 'show', f'{sha}:{name}'])
                target = source / 'eval_logs' / relative
            else:
                data = (public / relative).read_bytes()
                target = source / relative
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError(f'changed evidence: {ref}')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        spec = artifact.parent / 'spec.md'
        evidence = analysis.EvidenceBundle(
            node_id=manifest['node_id'], mode=manifest['mode'],
            research_sha=manifest['research_sha'], contract_hash=manifest['contract_hash'],
            evaluation_version=manifest['evaluation_version'], evaluation_snapshot=document['evaluation_snapshot'],
            spec_sha256=manifest['spec_sha256'], spec_path=str(spec), evaluation_path=str(artifact.parent / 'evaluation.md'),
            engineering_attempts=tuple(analysis.EvidenceAttempt(**a) for a in manifest['engineering_attempts']),
            research_attempts=tuple(analysis.EvidenceAttempt(**a) for a in manifest.get('research_attempts', [])),
            evidence_paths=tuple(manifest['evidence_paths']),
            evidence_records=tuple(analysis.EvidenceRecord(**r) for r in manifest['evidence_records']),
            manifest_hash=document['evidence_manifest_hash'])
        context = analysis.AnalysisContext(source.parent, source / 'notes', None, (), source / 'eval_logs', run / 'repo')
        workspace = case / 'workspace'
        snapshot = analysis._prepare_evidence_snapshot(evidence, context, workspace)
        old = before._prompt(evidence) + before_context.render_context(evidence, snapshot)
        new = analysis._prompt(evidence, snapshot) + evaluation_context.render_context(evidence, snapshot)
        view = diagnostic_view(workspace)
        projected = view.rewrite_text(new)
        assert '/workspace/evidence/evidence_manifest.json' in projected
        spilled = json.loads((snapshot / 'evidence_manifest.json').read_text())
        assert spilled == {'manifest': manifest, 'manifest_hash': document['evidence_manifest_hash']}
        assert len(new) < len(old)
        for name, value in [('before.txt', old), ('after.txt', new), ('sandbox-after.txt', projected)]:
            (case / name).write_text(value)
        # No tokenizer download: characters are exact; no token estimate is reported.
        rows.append({'node': evidence.node_id, 'artifact': str(artifact),
                     'evidence_records': len(evidence.evidence_records),
                     'before_chars': len(old), 'after_chars': len(new),
                     'reduction_percent': round(100 * (1 - len(new) / len(old)), 1),
                     'index_chars': len((snapshot / 'evidence_index.md').read_text()),
                     'after_plus_index_chars': len(new) + len((snapshot / 'evidence_index.md').read_text()),
                     'evaluation_configs': len(list(snapshot.rglob('evaluation_context.json'))),
                     'manifest_roundtrip': True, 'source_hashes_verified': True})
    if not rows:
        raise ValueError('no published Research Analysis artifacts')
    (output / 'report.json').write_text(json.dumps({'baseline': revision, 'cases': rows,
        'scope': 'Published-artifact payload replay; excludes checkpoint guidance, runner suffix and model behavior.'}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    args = parser.parse_args()
    replay(args.run.resolve(), args.output.resolve(), args.baseline)
