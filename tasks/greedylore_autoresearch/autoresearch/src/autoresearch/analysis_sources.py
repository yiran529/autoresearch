"""Seal Analyzer citations against fixed Git blobs and its own diagnostic ledger."""

from __future__ import annotations

import copy
import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path, PurePosixPath

from .validation import ArtifactValidationError

_FILE_LIMIT = 1024 * 1024
_TOTAL_LIMIT = 8 * _FILE_LIMIT
_TERMINAL = {'completed', 'failed', 'timed_out', 'interrupted'}


def _relative(reference):
    path = PurePosixPath(reference)
    if (not reference or path.is_absolute() or '..' in path.parts
            or path.as_posix() != reference or '\\' in reference):
        raise ArtifactValidationError(f'unsafe Analysis evidence reference: {reference}')
    return path


def _regular_bytes(path, root):
    if (not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file() or path.stat().st_size > _FILE_LIMIT):
        raise ArtifactValidationError(f'Analysis evidence is not a bounded regular file: {path}')
    return path.read_bytes()


def read_source(reference, evidence, context, *, version=None):
    """Read only a pinned regular Git blob or an authenticated terminal diagnostic."""
    relative = _relative(reference)
    if reference.startswith('code/'):
        parts = reference.split('/', 2)
        allowed = {evidence.research_sha, *(a.commit_sha for a in evidence.research_attempts),
                   *(a.commit_sha for a in evidence.engineering_attempts)}
        if len(parts) != 3 or parts[1] not in allowed or not re.fullmatch('[a-f0-9]{40}', parts[1]):
            raise ArtifactValidationError(f'code reference is outside fixed commits: {reference}')
        def git(*args):
            return subprocess.check_output(['git', '-C', str(context.checkout_root), *args],
                                           stderr=subprocess.PIPE, timeout=30)
        try:
            entries = git('ls-tree', '-z', parts[1], '--', parts[2]).split(b'\0')
            entries = [entry for entry in entries if entry]
            if len(entries) != 1:
                raise ValueError('missing blob')
            metadata, name = entries[0].split(b'\t', 1)
            mode, kind, oid = metadata.split()
            if (mode not in {b'100644', b'100755'} or kind != b'blob'
                    or name.decode() != parts[2] or int(git('cat-file', '-s', oid.decode())) > _FILE_LIMIT):
                raise ValueError('not a bounded regular blob')
            return git('cat-file', 'blob', oid.decode())
        except (subprocess.SubprocessError, ValueError) as exc:
            raise ArtifactValidationError(f'fixed code evidence unavailable: {reference}') from exc
    parts = relative.parts
    if len(parts) < 5 or parts[0] != 'diagnostics' or parts[2] != 'records':
        raise ArtifactValidationError(f'unknown Analysis evidence reference: {reference}')
    match = re.fullmatch(rf'{evidence.node_id}-{evidence.mode}-v([1-9][0-9]*)', parts[1])
    if not match or (version is not None and int(match[1]) != version):
        raise ArtifactValidationError(f'diagnostic belongs to another Analysis: {reference}')
    from .diagnostics import DiagnosticTask

    task = DiagnosticTask(context.coral_dir, parts[1])
    try:
        ledger = json.loads(_regular_bytes(task.path, task.private))
        identity = {'node_id': evidence.node_id, 'mode': evidence.mode, 'version': int(match[1]),
                    'research_sha': evidence.research_sha, 'evidence_hash': evidence.manifest_hash}
        if any(ledger.get(key) != value for key, value in identity.items()):
            raise ValueError('ledger identity mismatch')
        request = ledger['requests'][parts[3]]
        record_path = (task.root / 'records') / parts[3] / 'record.json'
        record = json.loads(_regular_bytes(record_path, (task.root / 'records')))
        public_identity = {('analysis_version' if key == 'version' else key): value
                           for key, value in identity.items()}
        if (any(record.get(key) != value for key, value in public_identity.items())
                or request['result'].get('status') not in _TERMINAL
                or not isinstance(record.get('finished_at'), (float, int))
                or any(record.get(key) != value for key, value in
                       {**request['request'], **request['result']}.items())):
            raise ValueError('diagnostic record is not authenticated and terminal')
        suffix = parts[4:]
        if suffix[0] not in {'record.json', 'stdout.log', 'stderr.log', 'results', 'scripts'}:
            raise ValueError('not a diagnostic text/log/script result')
        if Path(suffix[-1]).suffix not in {'.json', '.log', '.txt', '.md', '.csv', '.tsv', '.py'}:
            raise ValueError('binary diagnostic payload is not citation evidence')
        return _regular_bytes((task.root / 'records') / parts[3] / Path(*suffix), (task.root / 'records'))
    except (OSError, KeyError, ValueError) as exc:
        raise ArtifactValidationError(f'diagnostic evidence unavailable: {reference}: {exc}') from exc


def seal_sources(authored, evidence, context, version, *, diagnostic_evidence=None):
    """Canonicalize new citations and freeze hashes without changing the input job."""
    from .analysis import EvidenceRecord, _hash_bytes, _manifest_hash

    document = copy.deepcopy(authored)
    records = {r.path: r for r in evidence.evidence_records}
    references = set(evidence.evidence_paths)
    allowed = {'spec.md', 'evaluation.md', *references,
               *(ref.removeprefix('evidence/') for ref in references),
               *(r.path for r in records.values() if r.status == 'present')}
    task_id = f'{evidence.node_id}-{evidence.mode}-v{version}'
    total = 0

    def add(reference, *, required=True):
        nonlocal total
        if reference in records and records[reference].status == 'present':
            return
        # Public context is pinned separately by immutable task version. Keep
        # diagnostic authentication on the original formal input manifest so
        # recovery and older artifact readers use the same ledger identity.
        data = read_source(reference, diagnostic_evidence or evidence, context, version=version)
        if total + len(data) > _TOTAL_LIMIT or len(records) >= 1024:
            if required:
                raise ArtifactValidationError(f'Analysis citation evidence budget exceeded: {reference}')
            return
        total += len(data)
        records[reference] = EvidenceRecord(reference, 'present', _hash_bytes(data))
        references.add(reference)
        allowed.add(reference)

    # Explicit citations have priority over optional narrative evidence retention.
    if 'handover' in document:
        from .handover import validate_body

        try:
            validate_body(document['handover'])
        except ValueError as exc:
            raise ArtifactValidationError(str(exc)) from exc
        canonical = []
        for original in document['handover']['evidence_refs']:
            _relative(original)
            reference = original.removeprefix('evidence/')
            if reference.startswith('records/'):
                reference = f'diagnostics/{task_id}/{reference}'
            if reference not in allowed:
                # Nested log directories are readable in the snapshot but only
                # their files carry frozen hashes. Expand from that manifest,
                # never from a fresh filesystem walk after the model returns.
                children = sorted(r.path for r in evidence.evidence_records
                                  if r.status == 'present'
                                  and r.path.startswith(reference + '/')) if reference.startswith('eval_logs/') else []
                if children:
                    canonical.extend(children)
                    continue
                if reference.startswith(('code/', 'diagnostics/')):
                    add(reference)
                else:
                    raise ArtifactValidationError(f'handover evidence_ref is not in the frozen evidence: {original}')
            canonical.append(reference)
        document['handover']['evidence_refs'] = list(dict.fromkeys(canonical))
    from .diagnostics import DiagnosticTask

    task = DiagnosticTask(context.coral_dir, task_id)
    if task.path.is_file():
        ledger = task.read()
        for request_id, request in ledger.get('requests', {}).items():
            status = request.get('result', {}).get('status')
            if status not in _TERMINAL:
                continue
            root = (task.root / 'records') / request_id
            if status == 'interrupted':
                # The pre-execution interruption placeholder is deliberately
                # charged, but is not a completed evidence record after a crash.
                record_path = root / 'record.json'
                if not record_path.exists():
                    continue
                record = json.loads(_regular_bytes(record_path, (task.root / 'records')))
                if 'finished_at' not in record:
                    continue
            paths = [root / 'record.json', root / 'stdout.log', root / 'stderr.log']
            for name in ('results', 'scripts'):
                paths.extend(sorted((root / name).rglob('*')))
            for path in paths:
                if (path.is_file() and path.stat().st_size <= _FILE_LIMIT
                        and path.suffix in {'.json', '.log', '.txt', '.md', '.csv', '.tsv', '.py'}):
                    add(path.relative_to(context.coral_dir / 'public').as_posix(), required=False)
    expanded = replace(evidence, evidence_records=tuple(sorted(records.values(), key=lambda r: r.path)),
                       evidence_paths=tuple(sorted(references)))
    expanded = replace(expanded, manifest_hash=_manifest_hash(expanded.manifest()))
    return document, expanded
