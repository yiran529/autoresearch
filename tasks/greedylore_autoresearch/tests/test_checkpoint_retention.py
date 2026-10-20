"""CPU-only retention tests; payloads are opaque bytes, never model objects."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from autoresearch.checkpoint_retention import prune_research_checkpoints
from autoresearch.checkpoints import discover_checkpoints
from autoresearch.validation import ArtifactValidationError

from coral.types import Attempt


def attempt(sha, score=1, *, quality=1, status='improved', timestamp='2026-01-01', runs=()):
    return Attempt(sha * 40, 'researcher', 'candidate', score, status, None, timestamp,
                   metadata={'node_id': 'N001', 'mode': 'research', 'comparable': True,
                             'scores': {'quality_gate': {'value': quality}}, 'grader_runs': list(runs)})


def run(name, status='improved'):
    return {'run_id': name, 'status': status}


def checkpoint(coral, candidate, run_id=None):
    root = coral / 'public/eval_logs' / candidate.commit_hash
    if run_id:
        root /= run_id
    root /= 'checkpoints/test-config'
    root.mkdir(parents=True)
    files = {}
    for name in ('model.pt', 'rank-0.pt'):
        payload = b'checkpoint tensor'
        (root / name).write_bytes(payload)
        files[name] = {'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}
    (root / 'manifest.json').write_text(json.dumps({
        'version': 1, 'status': 'complete', 'world_size': 1,
        'contract_hash': 'contract', 'update_step': 4, 'files': files,
    }))
    (root.parent.parent / 'stdout.log').write_text('training log')
    return root


def discover(coral, roots):
    log_root = coral / 'public/eval_logs'
    evidence = SimpleNamespace(contract_hash='contract', evidence_records=[
        SimpleNamespace(path='eval_logs/' + str((root / 'manifest.json').relative_to(log_root)))
        for root in roots])
    return discover_checkpoints(evidence, SimpleNamespace(coral_dir=coral, log_root=log_root))


def test_keeps_selected_payload_and_all_original_evidence(tmp_path):
    best, worse = attempt('a', 2), attempt('b', 1)
    kept, pruned = checkpoint(tmp_path, best), checkpoint(tmp_path, worse)
    manifest = (pruned / 'manifest.json').read_bytes()
    (pruned / 'config.json').write_text('config evidence')
    (pruned / 'training.log').write_text('log evidence')
    assert prune_research_checkpoints(tmp_path, 'N001', [best, worse], best.commit_hash) == best.commit_hash
    assert (kept / 'model.pt').exists()
    assert not (pruned / 'model.pt').exists()
    assert (pruned / 'manifest.json').read_bytes() == manifest
    assert (pruned / 'config.json').read_text() == 'config evidence'
    assert (pruned / 'training.log').read_text() == 'log evidence'
    assert (pruned.parent.parent / 'stdout.log').read_text() == 'training log'
    mounted, text = discover(tmp_path, [kept, pruned])
    assert mounted == [kept]
    assert 'payload pruned' in text


def test_pending_candidates_and_running_retry_roots_are_untouched(tmp_path):
    best = attempt('a')
    pending = attempt('b', None, status='pending', runs=[run('active', 'running')])
    kept = checkpoint(tmp_path, best)
    active = checkpoint(tmp_path, pending, 'active')
    prune_research_checkpoints(tmp_path, 'N001', [best, pending], best.commit_hash)
    assert (kept / 'model.pt').exists()
    assert (active / 'model.pt').exists()


def test_selection_switch_prunes_previous_best(tmp_path):
    old, new = attempt('a', 1), attempt('b', 2)
    old_root = checkpoint(tmp_path, old)
    new_root = checkpoint(tmp_path, new)
    prune_research_checkpoints(tmp_path, 'N001', [old], old.commit_hash)
    prune_research_checkpoints(tmp_path, 'N001', [old, new], new.commit_hash)
    assert not (old_root / 'model.pt').exists()
    assert (new_root / 'model.pt').exists()


def test_retry_retains_only_latest_terminal_run(tmp_path):
    candidate = attempt('a', runs=[run('old', 'crashed'), run('new')])
    old, new = checkpoint(tmp_path, candidate, 'old'), checkpoint(tmp_path, candidate, 'new')
    prune_research_checkpoints(tmp_path, 'N001', [candidate], candidate.commit_hash)
    assert not (old / 'model.pt').exists()
    assert (new / 'model.pt').exists()
    assert discover(tmp_path, [old, new])[0] == [new]


def test_crash_after_ledger_write_is_recoverable(tmp_path, monkeypatch):
    best, worse = attempt('a', 2), attempt('b', 1)
    checkpoint(tmp_path, best)
    pruned = checkpoint(tmp_path, worse)
    from pathlib import Path
    real_unlink = Path.unlink
    def interrupted(path, *args, **kwargs):
        if path.name == 'rank-0.pt':
            raise OSError('interrupted cleanup')
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        prune_research_checkpoints(tmp_path, 'N001', [best, worse], best.commit_hash)
    assert discover(tmp_path, [pruned])[0] == []
    monkeypatch.setattr(Path, 'unlink', real_unlink)
    prune_research_checkpoints(tmp_path, 'N001', [best, worse], best.commit_hash)
    prune_research_checkpoints(tmp_path, 'N001', [best, worse], best.commit_hash)
    assert list(pruned.iterdir()) == [pruned / 'manifest.json']


def test_unmarked_missing_payload_is_integrity_failure(tmp_path):
    candidate = attempt('a')
    root = checkpoint(tmp_path, candidate)
    (root / 'model.pt').unlink()
    with pytest.raises(ArtifactValidationError, match='integrity'):
        discover(tmp_path, [root])
    with pytest.raises(ArtifactValidationError, match='integrity'):
        best = attempt('b', 2)
        prune_research_checkpoints(tmp_path, 'N001', [candidate, best], best.commit_hash)


def test_pruned_manifest_mutation_is_integrity_failure(tmp_path):
    best, worse = attempt('a', 2), attempt('b', 1)
    root = checkpoint(tmp_path, worse)
    prune_research_checkpoints(tmp_path, 'N001', [best, worse], best.commit_hash)
    value = json.loads((root / 'manifest.json').read_text())
    value['update_step'] = 100
    (root / 'manifest.json').write_text(json.dumps(value))
    with pytest.raises(ArtifactValidationError, match='manifest integrity'):
        discover(tmp_path, [root])


@pytest.mark.parametrize('attack', ['commit', 'run', 'symlink', 'payload'])
def test_rejects_unsafe_paths_without_deleting_outside_files(tmp_path, attack):
    best, bad = attempt('a', 2), attempt('b', 1)
    outside = tmp_path / 'outside'
    outside.mkdir()
    sentinel = outside / 'model.pt'
    sentinel.write_bytes(b'untouched')
    if attack == 'commit':
        bad.commit_hash = '../outside'
    elif attack == 'run':
        bad.metadata['grader_runs'] = [run('../outside')]
    else:
        root = checkpoint(tmp_path, bad)
        if attack == 'symlink':
            (root / 'model.pt').unlink()
            (root / 'model.pt').symlink_to(sentinel)
        else:
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['files']['../model.pt'] = manifest['files'].pop('model.pt')
            (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ArtifactValidationError):
        prune_research_checkpoints(tmp_path, 'N001', [best, bad], best.commit_hash)
    assert sentinel.read_bytes() == b'untouched'


def test_root_is_never_pruned(tmp_path):
    root_attempt = attempt('a')
    root_attempt.metadata['node_id'] = 'N000'
    root = checkpoint(tmp_path, root_attempt)
    prune_research_checkpoints(tmp_path, 'N000', [root_attempt], root_attempt.commit_hash)
    assert (root / 'model.pt').exists()


def test_no_quality_pass_keeps_latest_terminal_failure(tmp_path):
    old = attempt('a', 3, quality=0, timestamp='2026-01-01')
    latest = attempt('b', None, quality=0, status='crashed', timestamp='2026-01-02')
    pending = attempt('c', None, status='pending', timestamp='2026-01-03')
    old_root, latest_root = checkpoint(tmp_path, old), checkpoint(tmp_path, latest)
    assert prune_research_checkpoints(tmp_path, 'N001', [old, latest, pending]) == latest.commit_hash
    assert not (old_root / 'model.pt').exists()
    assert (latest_root / 'model.pt').exists()


def test_default_selection_prefers_quality_passing_candidate(tmp_path):
    good, fail = attempt('a', 1), attempt('b', 3, quality=0)
    good_root, fail_root = checkpoint(tmp_path, good), checkpoint(tmp_path, fail)
    assert prune_research_checkpoints(tmp_path, 'N001', [good, fail]) == good.commit_hash
    assert (good_root / 'model.pt').exists()
    assert not (fail_root / 'model.pt').exists()


def test_pending_only_is_noop(tmp_path):
    pending = attempt('a', None, status='pending')
    root = checkpoint(tmp_path, pending)
    assert prune_research_checkpoints(tmp_path, 'N001', [pending]) is None
    assert (root / 'model.pt').exists()


def test_terminal_partial_without_manifest_is_cleaned_but_pending_is_untouched(tmp_path):
    best, failed = attempt('a', 2), attempt('b', None, status='crashed')
    pending = attempt('c', None, status='pending')
    partial = checkpoint(tmp_path, failed)
    (partial / 'manifest.json').unlink()
    (partial / 'rank-0.pt.123.tmp').write_bytes(b'interrupted writer')
    (partial / 'config.json').write_text('keep config')
    active = checkpoint(tmp_path, pending)
    (active / 'manifest.json').unlink()
    prune_research_checkpoints(tmp_path, 'N001', [best, failed, pending], best.commit_hash)
    assert list(partial.iterdir()) == [partial / 'config.json']
    assert (active / 'model.pt').exists()
    ledger = json.loads((tmp_path / 'private/checkpoint_retention.json').read_text())
    record = ledger['pruned'][str((partial / 'manifest.json').relative_to(tmp_path))]
    assert record['manifest_missing'] is True
    assert record['manifest_sha256'] is None
    prune_research_checkpoints(tmp_path, 'N001', [best, failed, pending], best.commit_hash)


def test_missing_previously_recorded_manifest_remains_integrity_failure(tmp_path):
    best, failed = attempt('a', 2), attempt('b', 1)
    root = checkpoint(tmp_path, failed)
    prune_research_checkpoints(tmp_path, 'N001', [best, failed], best.commit_hash)
    (root / 'manifest.json').unlink()
    with pytest.raises(ArtifactValidationError, match='manifest disappeared'):
        prune_research_checkpoints(tmp_path, 'N001', [best, failed], best.commit_hash)


def test_partial_cleanup_crash_retries_from_durable_ledger(tmp_path, monkeypatch):
    from pathlib import Path
    best, failed = attempt('a', 2), attempt('b', None, status='crashed')
    root = checkpoint(tmp_path, failed)
    (root / 'manifest.json').unlink()
    original_unlink = Path.unlink
    def interrupted(path, *args, **kwargs):
        if path.name == 'rank-0.pt':
            raise OSError('interrupted partial cleanup')
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', interrupted)
    with pytest.raises(OSError, match='interrupted partial'):
        prune_research_checkpoints(tmp_path, 'N001', [best, failed], best.commit_hash)
    assert (tmp_path / 'private/checkpoint_retention.json').is_file()
    monkeypatch.setattr(Path, 'unlink', original_unlink)
    prune_research_checkpoints(tmp_path, 'N001', [best, failed], best.commit_hash)
    assert not list(root.iterdir())
