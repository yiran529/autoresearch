"""Bounded, immutable agent-provided supporting files (not verified executions)."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from coral.grader.trusted_writes import _atomic_json, trusted_writes_lock

MAX_ATTACHMENT_BYTES = 32 * 1024
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}')


def archive(store, agent_id: str, payload: dict) -> dict:
    from .state import WorkflowError, _owned_active, _reject_finalizing

    if set(payload) != {'node_id', 'request_id', 'commit_sha', 'files'}:
        raise WorkflowError('invalid evidence archive fields')
    node, request_id, sha = (payload[k] for k in ('node_id', 'request_id', 'commit_sha'))
    if (not isinstance(node, str) or not re.fullmatch(r'N[0-9]{3}', node)
            or not isinstance(request_id, str) or not _NAME.fullmatch(request_id)
            or not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha)):
        raise WorkflowError('invalid evidence archive identity')
    files = payload['files']
    if not isinstance(files, list) or not 1 <= len(files) <= 32:
        raise WorkflowError('archive requires 1–32 files')
    contents = {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {'name', 'content'}:
            raise WorkflowError('invalid attachment fields')
        name, encoded = item['name'], item['content']
        if (not isinstance(name, str) or not _NAME.fullmatch(name)
                or name == 'manifest.json' or name in contents
                or not isinstance(encoded, str) or len(encoded) > 45000):
            raise WorkflowError('invalid or duplicate attachment name/content')
        try:
            contents[name] = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise WorkflowError('attachment content must be base64') from exc
    if sum(map(len, contents.values())) > MAX_ATTACHMENT_BYTES:
        raise WorkflowError('archive exceeds 32 KiB; use managed diagnostics for larger outputs')
    with trusted_writes_lock(store.coral_dir):
        state = store._load_unlocked()
        mode = next((mode for mode in ('research', 'engineering')
                     if _owned_active(state, agent_id, node, mode)), None)
        if mode is None:
            raise WorkflowError('evidence requires this Agent\'s active Node assignment')
        _reject_finalizing(state, agent_id, node, mode)
        if store.repo_path is not None and subprocess.run(
            ['git', '-C', str(store.repo_path), 'cat-file', '-e', sha + '^{commit}'],
            capture_output=True, check=False,
        ).returncode:
            raise WorkflowError('attachment commit is unavailable in the run repository')
        reference = f'diagnostics/{node}-{agent_id}/attachments/{request_id}'
        target = store.island_root / reference
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise WorkflowError('attachment destination traverses a symlink')
        manifest = {'node_id': node, 'mode': mode, 'agent_id': agent_id,
                    'commit_sha': sha, 'request_id': request_id, 'execution_verified': False,
                    'commit_sha_source': 'Uploader-declared context SHA; verified to exist, not verified as executed code.',
                    'provenance': 'Agent-provided files; content archived by host, execution not verified.',
                    'files': [{'name': name, 'size_bytes': len(content),
                               'sha256': hashlib.sha256(content).hexdigest()}
                              for name, content in sorted(contents.items())]}
        records = state['nodes'][node].setdefault('attachments', [])
        if target.exists():
            if json.loads((target / 'manifest.json').read_text()) != manifest or any(
                not (target / name).is_file() or (target / name).is_symlink()
                or (target / name).read_bytes() != content for name, content in contents.items()
            ):
                raise WorkflowError('immutable attachment conflicts with previous archive')
        else:
            if sum(r['agent_id'] == agent_id for r in records) >= 64:
                raise WorkflowError('attachment limit reached: 64 bundles per assignment')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=store.coral_dir / 'private', prefix='attachment-') as td:
                staged = Path(td) / 'bundle'
                staged.mkdir()
                for name, content in contents.items():
                    with (staged / name).open('wb') as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    (staged / name).chmod(0o644)
                staged.chmod(0o755)
                _atomic_json(staged / 'manifest.json', manifest, mode=0o644)
                os.replace(staged, target)
                descriptor = os.open(target.parent, os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        if not any(record['path'] == reference for record in records):
            records.append({'path': reference, 'mode': mode, 'agent_id': agent_id,
                            'commit_sha': sha})
            store._save_unlocked(state)
        return {'path': reference, 'files': [f'{reference}/{name}' for name in sorted(contents)]}


def evidence_references(coral_dir: Path, node_id: str, mode: str) -> list[str]:
    path = coral_dir / 'private/autoresearch_state.json'
    if not path.exists():
        return []
    state = json.loads(path.read_text())
    return [record['path'] for record in state['nodes'].get(node_id, {}).get('attachments', [])
            if record['mode'] == 'research' or mode == 'engineering']
