"""Bounded, immutable public knowledge supplied to a new Analysis version.

These are context copies, not authenticated diagnostic executions. Keeping them
under the existing public diagnostics namespace also lets older artifact readers
verify their hashes without granting access to mutable live notes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import replace
from pathlib import Path

from .analysis import EvidenceRecord, _manifest_hash

MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_FILES = 256
MAX_ENTRIES = 4096
MAX_METADATA_BYTES = 1024 * 1024
_GLOBALS = ('tree.md', 'research_summary.md', 'engineering_summary.md', 'wisdom.md')
_NODE_FILES = ('README.md', 'spec.md', 'evaluation.md', 'research.json')
_REPORT = re.compile(r'(research|engineering)_analysis_v[1-9][0-9]*\.(md|json)')
_NODE = re.compile(r'N[0-9]{3}')


def _open_directory(path: Path) -> int:
    """Open every ancestor without following symlinks, including the root."""
    absolute = path.absolute()
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            if part in {'.', '..'}:
                raise ValueError('unsafe context directory')
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read(path: Path, limit: int) -> bytes:
    parent = _open_directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        with os.fdopen(descriptor, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('context source is not a regular file')
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise OverflowError('context source exceeds byte limit')
        data.decode('utf-8')
        if b'\x00' in data:
            raise ValueError('context source is not text')
        return data
    finally:
        os.close(parent)


def _names(path: Path) -> list[str]:
    descriptor = _open_directory(path)
    try:
        # Bound even enumeration of unexpected public content.
        with os.scandir(descriptor) as entries:
            result = []
            for entry in entries:
                result.append(entry.name)
                if len(result) > MAX_ENTRIES:
                    raise OverflowError('context directory exceeds entry limit')
        return sorted(result)
    finally:
        os.close(descriptor)


def _allowed(source: str, node_id: str) -> bool:
    parts = source.split('/')
    return (source in _GLOBALS or (
        len(parts) == 3 and parts[0] == 'nodes' and _NODE.fullmatch(parts[1])
        and parts[1] != node_id
        and (parts[2] in _NODE_FILES or _REPORT.fullmatch(parts[2]))
    ))


def _collect(notes: Path, node_id: str, target: Path) -> list[dict]:
    inventory: list[dict] = []
    copied = total = 0

    def capture(source: str) -> None:
        nonlocal copied, total
        entry = {'source': source, 'status': 'missing'}
        try:
            if copied >= MAX_FILES:
                entry['status'] = 'file_limit'
            else:
                data = _read(notes / source, MAX_FILE_BYTES)
                if total + len(data) > MAX_TOTAL_BYTES:
                    entry['status'] = 'total_byte_limit'
                else:
                    destination = target / 'notes' / source
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    entry.update(status='present', bytes=len(data),
                                 sha256=hashlib.sha256(data).hexdigest())
                    copied += 1
                    total += len(data)
        except FileNotFoundError:
            pass
        except OverflowError:
            entry['status'] = 'file_byte_limit'
        except (OSError, ValueError, UnicodeError):
            entry['status'] = 'unsafe_or_unreadable'
        inventory.append(entry)

    for source in _GLOBALS:
        capture(source)
    try:
        nodes = _names(notes / 'nodes')
    except (OSError, ValueError, OverflowError):
        inventory.append({'source': 'nodes', 'status': 'unavailable_or_entry_limit'})
        return inventory
    for node in nodes:
        if not _NODE.fullmatch(node) or node == node_id:
            continue
        if len(inventory) >= MAX_ENTRIES - 1:
            inventory.append({'source': 'nodes', 'status': 'inventory_limit'})
            break
        try:
            names = _names(notes / 'nodes' / node)
        except (OSError, ValueError, OverflowError):
            inventory.append({'source': f'nodes/{node}', 'status': 'unsafe_or_entry_limit'})
            continue
        for name in sorted(set(_NODE_FILES) | {n for n in names if _REPORT.fullmatch(n)}):
            if len(inventory) >= MAX_ENTRIES - 1:
                inventory.append({'source': f'nodes/{node}', 'status': 'inventory_limit'})
                return inventory
            capture(f'nodes/{node}/{name}')
    return inventory


def _load(root: Path, identity: dict) -> tuple[EvidenceRecord, ...]:
    metadata = _read(root / 'context.json', MAX_METADATA_BYTES)
    document = json.loads(metadata)
    if (not isinstance(document, dict) or document.get('identity') != identity
            or document.get('kind') != 'public_context_snapshot'
            or not isinstance(document.get('inventory'), list)
            or len(document['inventory']) > MAX_ENTRIES):
        raise ValueError('invalid frozen Analysis context identity or inventory')
    prefix = 'diagnostics/' + root.name
    records = [EvidenceRecord(prefix + '/context.json', 'present',
                              hashlib.sha256(metadata).hexdigest())]
    total = 0
    sources = set()
    for entry in document['inventory']:
        if not isinstance(entry, dict) or not isinstance(entry.get('source'), str):
            raise ValueError('invalid frozen Analysis context entry')
        source = entry['source']
        if source in sources:
            raise ValueError('duplicate frozen Analysis context source')
        sources.add(source)
        if entry.get('status') != 'present':
            continue
        if not _allowed(source, identity['node_id']):
            raise ValueError('unsafe frozen Analysis context source')
        data = _read(root / 'notes' / source, MAX_FILE_BYTES)
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.get('sha256') or len(data) != entry.get('bytes'):
            raise ValueError('frozen Analysis context evidence changed')
        total += len(data)
        if total > MAX_TOTAL_BYTES or len(records) > MAX_FILES:
            raise ValueError('frozen Analysis context exceeds limits')
        records.append(EvidenceRecord(prefix + '/notes/' + source, 'present', digest))
    return tuple(records)


def with_public_context(evidence, context, version: int):
    """Freeze once, reuse on retry, and preserve unfinished pre-feature tasks."""
    if type(version) is not int or version < 1 or not _NODE.fullmatch(evidence.node_id):
        raise ValueError('invalid Analysis context identity')
    task_id = f'{evidence.node_id}-{evidence.mode}-v{version}'
    identity = {'node_id': evidence.node_id, 'mode': evidence.mode, 'version': version,
                'base_evidence_hash': evidence.manifest_hash}
    root = context.coral_dir / 'public/diagnostics' / f'analysis-context-{task_id}'
    lock = context.coral_dir / 'private/analysis_context' / f'{task_id}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if not root.exists() and not root.is_symlink():
            legacy_task = context.coral_dir / 'private/diagnostics' / task_id / 'task.json'
            if legacy_task.exists():
                state = json.loads(legacy_task.read_text())
                if state.get('evidence_hash') != evidence.manifest_hash:
                    raise ValueError('Analysis task context is missing or has changed')
                return evidence
            root.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix='.analysis-context-', dir=root.parent))
            try:
                inventory = _collect(context.notes_root, evidence.node_id, temporary)
                document = {'kind': 'public_context_snapshot', 'identity': identity,
                            'inventory': inventory,
                            'limits': {'files': MAX_FILES, 'file_bytes': MAX_FILE_BYTES,
                                       'total_bytes': MAX_TOTAL_BYTES},
                            'scope': 'Public text at collection time; links are not followed. '
                                     'Summaries are derived knowledge; unfinished Nodes are provisional. '
                                     'This is not a diagnostic execution record.'}
                (temporary / 'context.json').write_text(json.dumps(document, sort_keys=True) + '\n')
                _load(temporary, identity)
                temporary.rename(root)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        records = _load(root, identity)
    merged = tuple(sorted((*evidence.evidence_records, *records), key=lambda item: item.path))
    paths = tuple(sorted({*evidence.evidence_paths, *(f'evidence/{r.path}' for r in records)}))
    extended = replace(evidence, evidence_records=merged, evidence_paths=paths)
    return replace(extended, manifest_hash=_manifest_hash(extended.manifest()))
