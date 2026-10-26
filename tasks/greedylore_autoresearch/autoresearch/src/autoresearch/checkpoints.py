"""Select only checkpoints recorded under the current Attempt's evidence roots."""

import hashlib
import json
from pathlib import Path

from .checkpoint_retention import checkpoint_is_pruned
from .validation import ArtifactValidationError


def is_checkpoint_payload(path: Path) -> bool:
    # Keep the small completion manifest, not tensor files or interrupted temp files.
    return path.parent.parent.name == 'checkpoints' and path.name != 'manifest.json'


def discover_checkpoints(evidence, context):
    roots, descriptions = [], []
    for record in evidence.evidence_records:
        relative = Path(record.path.removeprefix('eval_logs/'))
        if relative.name != 'manifest.json' or relative.parent.parent.name != 'checkpoints':
            continue
        path = context.log_root / relative
        root = path.parent
        if (relative.is_absolute() or '..' in relative.parts or root.is_symlink()
                or not root.resolve().is_relative_to(context.log_root.resolve())
                or any(parent.is_symlink() for parent in path.parents)):
            raise ArtifactValidationError('checkpoint path escapes fixed evaluation logs')
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or value.get('version') != 1:
                raise ValueError('unsupported manifest')
            if checkpoint_is_pruned(context.coral_dir, path):
                descriptions.append(f'{root.name}: tensor payload pruned by checkpoint retention; '
                                    'original manifest and logs retained.')
                continue
            if value.get('status') != 'complete':
                descriptions.append(f'{root.name}: unavailable; ' + json.dumps(value))
                continue
            from .runtime_artifacts import RuntimeArtifacts
            artifacts = RuntimeArtifacts(context.coral_dir)
            managed = artifacts.resolve_checkpoint(path) if path.is_relative_to(context.coral_dir / 'public') else None
            if managed is not None:
                root = managed
            elif (path.is_relative_to(context.coral_dir / 'public') and
                  path.relative_to(context.coral_dir / 'public').as_posix() in artifacts._read()['checkpoints']):
                descriptions.append(f'{root.name}: managed tensor payload unavailable or cleaned; manifest retained.')
                continue
            world = value['world_size']
            if not isinstance(world, int) or isinstance(world, bool) or not 1 <= world <= 4:
                raise ValueError('invalid world size')
            if value['contract_hash'] != evidence.contract_hash:
                raise ValueError('contract mismatch')
            expected = {'model.pt', *(f'rank-{rank}.pt' for rank in range(world))}
            if set(value['files']) != expected:
                raise ValueError('rank files missing')
            for name, info in value['files'].items():
                source = root / name
                if source.is_symlink() or source.stat().st_size != info['size_bytes']:
                    raise ValueError('size or link mismatch')
                with source.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != info['sha256']:
                        raise ValueError('hash mismatch')
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ArtifactValidationError(f'checkpoint integrity check failed: {path}: {exc}') from exc
        if root not in roots:
            roots.append(root)
            descriptions.append(
                f'{root.name}: read-only path {root}; final update step {value["update_step"]}; '
                f'world size {world}. Includes model, per-rank optimizer/scheduler, compressor '
                'and RNG state, and a saved training batch. Use diagnostic_checkpoint.load_checkpoint '
                'from the matching fixed-SHA code snapshot; construct matching objects before loading. '
                'For DDP continuation register via register_checkpoint_hook before loading '
                '(the training registration helper does this when diagnostic_checkpoint is enabled). '
                'Inspect manifest.json for config and limitations. Restore on the matching world '
                'size and DDP bucket layout. Staged checkpoints record deterministic data replay; '
                'legacy checkpoints do not record the original dataloader position. '
                'CPU inspection of a CUDA checkpoint requires restore_rng=False.')
    text = '\n\n## Available final checkpoints\n' + ('\n'.join(descriptions) if descriptions else
        'Unavailable for this Attempt. No historical checkpoint is reconstructed automatically.')
    return roots, text
