"""Host-owned external tensors with small, persistent public evidence.

The private ledger is authoritative for external checkpoint mounts. A candidate
manifest cannot nominate an arbitrary host path. No public symlinks are used.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from contextlib import nullcontext
from pathlib import Path

import yaml

from coral.grader.trusted_writes import _atomic_json, trusted_writes_lock


def safe_path(path: Path) -> Path:
    if '..' in path.parts:
        raise ValueError('artifact path contains parent traversal')
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('artifact path traverses a symlink')
    return path


def settings(coral_dir: Path) -> dict:
    path = Path(coral_dir) / 'config.yaml'
    config = yaml.safe_load(path.read_text()) if path.exists() else {}
    return (config or {}).get('grader', {}).get('args', {})


class RuntimeArtifacts:
    def __init__(self, coral_dir):
        self.coral_dir = Path(coral_dir).resolve()
        self.options = settings(self.coral_dir)
        self.root = Path(self.options.get('dynamic_artifact_root', '/dev/shm/wyr_tmp')).expanduser()
        if not self.root.is_absolute() or self.root == Path('/') or '..' in self.root.parts:
            raise ValueError('dynamic_artifact_root must be a dedicated absolute directory')
        safe_path(self.root)
        self.run_id = hashlib.sha256(str(self.coral_dir).encode()).hexdigest()[:24]
        self.namespace = self.root / 'coral' / self.run_id
        self.ledger_path = self.coral_dir / 'private/dynamic_artifacts.json'
        self.min_free = self.options.get('dynamic_artifact_min_free_bytes', 1024**3)
        if type(self.min_free) is not int or self.min_free < 0:
            raise ValueError('dynamic_artifact_min_free_bytes must be nonnegative')

    def _read(self):
        if self.ledger_path.exists():
            value = json.loads(safe_path(self.ledger_path).read_text())
            if value.get('root') != str(self.root) or value.get('run_id') != self.run_id:
                raise ValueError('dynamic artifact root conflicts with persisted ownership')
            return value
        return {'version': 1, 'root': str(self.root), 'run_id': self.run_id,
                'nodes': {}, 'checkpoints': {}}

    def node_root(self, node_id):
        if not isinstance(node_id, str) or not re.fullmatch(r'N[0-9]{3}', node_id):
            raise ValueError('invalid dynamic artifact node')
        target = safe_path(self.namespace / 'nodes' / node_id)
        with trusted_writes_lock(self.coral_dir):
            ledger = self._read()
            if ledger['nodes'].get(node_id, {}).get('cleaned'):
                raise ValueError('dynamic artifacts for this node were cleaned')
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
            row = ledger['nodes'].get(node_id, {'path': str(target), 'cleaned': False})
            ledger['nodes'][node_id] = row
            _atomic_json(self.ledger_path, ledger)
        return target

    def temporary_directory(self, node_id, execution_id):
        """Keep Unix socket names short, while retaining private node ownership."""
        self.node_root(node_id)
        identity = f'{self.run_id}/{node_id}/{execution_id}'
        name = hashlib.sha256(identity.encode()).hexdigest()[:24]
        target = safe_path(self.root / 'tmp' / name)
        with trusted_writes_lock(self.coral_dir):
            ledger = self._read()
            row = ledger['nodes'][node_id]
            if row.get('cleaned') or row.get('cleanup_requested'):
                raise ValueError('temporary artifacts for this node are being cleaned')
            paths = row.setdefault('temporary_paths', [])
            if str(target) not in paths:
                paths.append(str(target))
                _atomic_json(self.ledger_path, ledger)
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
        return target

    def path(self, node_id, relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('dynamic artifact path escapes node')
        return safe_path(self.node_root(node_id) / relative)

    def require_space(self, node_id, required_bytes=0):
        if type(required_bytes) is not int or required_bytes < 0:
            raise ValueError('required artifact space must be nonnegative')
        root = self.node_root(node_id)
        if shutil.disk_usage(root).free < self.min_free + required_bytes:
            raise OSError('insufficient space in dynamic artifact root; no fallback')
        return root

    def publish_checkpoint(self, node_id, checkpoint, public_manifest):
        checkpoint, public_manifest = safe_path(Path(checkpoint)), safe_path(Path(public_manifest))
        if not checkpoint.is_relative_to(self.node_root(node_id)):
            raise ValueError('checkpoint is outside owned node artifacts')
        public_root = self.coral_dir / 'public'
        if not public_manifest.is_relative_to(public_root) or public_manifest.name != 'manifest.json':
            raise ValueError('public checkpoint manifest must be under public evidence')
        content = (checkpoint / 'manifest.json').read_bytes()
        manifest = json.loads(content)
        if manifest.get('status') != 'complete':
            raise ValueError('cannot publish an incomplete checkpoint')
        digest = hashlib.sha256(content).hexdigest()
        relative = public_manifest.relative_to(public_root).as_posix()
        with trusted_writes_lock(self.coral_dir):
            ledger = self._read()
            record = {'node_id': node_id, 'path': str(checkpoint), 'manifest_sha256': digest}
            old = ledger['checkpoints'].get(relative)
            if old is not None and old != record:
                raise ValueError('immutable checkpoint manifest conflicts')
            if public_manifest.exists() and public_manifest.read_bytes() != content:
                raise ValueError('public checkpoint manifest conflicts')
            public_manifest.parent.mkdir(parents=True, exist_ok=True)
            from .atomic import atomic_write_text

            atomic_write_text(public_manifest, content.decode(), mode=0o644)
            ledger['checkpoints'][relative] = record
            _atomic_json(self.ledger_path, ledger)

    def resolve_checkpoint(self, public_manifest):
        public_manifest = safe_path(Path(public_manifest))
        relative = public_manifest.relative_to(self.coral_dir / 'public').as_posix()
        ledger = self._read()
        row = ledger['checkpoints'].get(relative)
        if row is None:
            return None
        if ledger['nodes'][row['node_id']].get('cleaned'):
            return None
        checkpoint = safe_path(Path(row['path']))
        expected_root = self.namespace / 'nodes' / row['node_id']
        if not checkpoint.is_relative_to(expected_root):
            raise ValueError('checkpoint ownership mismatch')
        content = (checkpoint / 'manifest.json').read_bytes()
        if (hashlib.sha256(content).hexdigest() != row['manifest_sha256']
                or content != public_manifest.read_bytes()):
            raise ValueError('external checkpoint manifest integrity mismatch')
        return checkpoint

    def cleanup_node(self, node_id, *, terminal, _already_locked=False):
        if not terminal:
            return False
        with (nullcontext() if _already_locked else trusted_writes_lock(self.coral_dir)):
            ledger = self._read()
            row = ledger['nodes'].get(node_id)
            if row is None or row.get('cleaned'):
                return True
            root = safe_path(self.namespace / 'nodes' / node_id)
            if row['path'] != str(root):
                raise ValueError('cleanup ownership mismatch')
            # Mark intent before unlinking, so recovery can finish the same cleanup.
            row['cleanup_requested'] = True
            _atomic_json(self.ledger_path, ledger)
            if root.exists():
                shutil.rmtree(root)
            for value in row.get('temporary_paths', []):
                temporary = safe_path(Path(value))
                if temporary.parent != self.root / 'tmp' or not re.fullmatch('[a-f0-9]{24}', temporary.name):
                    raise ValueError('temporary cleanup ownership mismatch')
                if temporary.exists():
                    shutil.rmtree(temporary)
            row['cleaned'] = True
            _atomic_json(self.ledger_path, ledger)
        return True

    def preserve_small(self, node_id, source, destination, *, max_file_bytes=2**20):
        """Copy bounded textual evidence; record omitted large files explicitly."""
        source, destination = safe_path(Path(source)), safe_path(Path(destination))
        owned = safe_path(self.namespace / 'nodes' / node_id)
        row = self._read()['nodes'].get(node_id)
        if not row or row.get('cleaned') or row.get('path') != str(owned):
            raise ValueError('evidence node ownership is absent or cleaned')
        if not source.is_relative_to(owned):
            raise ValueError('evidence source is outside node artifacts')
        if not destination.is_relative_to(self.coral_dir / 'public'):
            raise ValueError('persistent evidence destination is not public')
        records, copied_bytes, count = [], 0, 0
        for path in source.rglob('*'):
            count += 1
            if count > 4096:
                records.append({'path': '.', 'preserved': False, 'reason': 'entry budget exceeded'})
                break
            safe_path(path)
            if not path.is_file() or path.suffix not in {'.json', '.jsonl', '.log', '.txt', '.py', '.md', '.csv', '.tsv',
                    '.sh', '.yaml', '.yml', '.toml', '.cu', '.cpp', '.h', '.c'}:
                continue
            size = path.stat().st_size
            copied = size <= max_file_bytes and copied_bytes + size <= 16 * 2**20
            relative = path.relative_to(source)
            if copied:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                safe_path(target)
                shutil.copyfile(path, target)
                copied_bytes += size
            records.append({'path': relative.as_posix(), 'size_bytes': size, 'preserved': copied,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if copied else None})
        return records
