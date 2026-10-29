"""Host declarations must match the readable diagnostic resource view."""
import json
import subprocess
import tempfile
from pathlib import Path

import pytest


def setup_resources(tmp_path):
    data, target = tmp_path / 'data', tmp_path / 'target'
    data.mkdir()
    target.mkdir()
    (target / 'shard').write_text('original stream')
    (data / 'shard').symlink_to(target / 'shard')
    return data, target


def test_declared_link_target_must_be_explicit(tmp_path):
    from autoresearch.runtime_resources import resolve_resources
    data, _ = setup_resources(tmp_path)
    with pytest.raises(ValueError, match='authorized'):
        resolve_resources([{'id': 'c4', 'path': str(data), 'target_roots': []}])


def test_catalog_preserves_original_paths_and_stable_view(tmp_path):
    from autoresearch.runtime_resources import resolve_resources
    data, target = setup_resources(tmp_path)
    catalog = resolve_resources([{'id': 'c4', 'path': str(data), 'target_roots': [str(target)],
                                  'view': '/resources/c4'}])
    assert [str(data), str(data)] in catalog['mounts']
    assert [str(target), str(target)] in catalog['mounts']
    assert [str(data), '/resources/c4'] in catalog['mounts']
    assert catalog['resources'][0]['access'] == 'managed_diagnostic_readonly'
    assert catalog['resources'][0]['path'] == str(data)
    assert catalog['resources'][0]['view'] == '/resources/c4'


@pytest.mark.parametrize('failure', ['missing', 'loop', 'escape', 'broad', 'budget'])
def test_resource_discovery_fails_closed(tmp_path, failure):
    from autoresearch.runtime_resources import resolve_resources
    data, target = setup_resources(tmp_path)
    declaration = {'id': 'c4', 'path': str(data), 'target_roots': [str(target)]}
    if failure == 'missing':
        (target / 'shard').unlink()
    elif failure == 'loop':
        (target / 'cycle').symlink_to(target, target_is_directory=True)
    elif failure == 'escape':
        (target / 'secret').symlink_to(tmp_path / 'outside')
    elif failure == 'broad':
        declaration['target_roots'] = ['/dev/shm']
    with pytest.raises(ValueError):
        resolve_resources([declaration], max_entries=1 if failure == 'budget' else 4096)


def test_declared_shard_readable_after_real_tmpfs(tmp_path):
    from autoresearch.diagnostic_sandbox import build_sandbox, isolation_available
    from autoresearch.runtime_resources import resolve_resources
    if not isolation_available():
        pytest.skip('real bubblewrap namespaces unavailable')
    with tempfile.TemporaryDirectory(prefix='coral-resources-', dir='/dev/shm') as temporary:
        data, target = setup_resources(Path(temporary))
        catalog = resolve_resources([{'id': 'c4', 'path': str(data), 'target_roots': [str(target)],
                                     'view': '/resources/c4'}])
        workspace = tmp_path / 'workspace'
        for name in ('experiments', 'results', 'records'):
            (workspace / name).mkdir(parents=True)
        probe = f'''from pathlib import Path
assert Path({str(data / 'shard')!r}).read_text() == 'original stream'
assert Path('/resources/c4/shard').read_text() == 'original stream'
assert not Path({str(Path(temporary) / 'foreign')!r}).exists()
try:
 Path('/resources/c4/shard').write_text('changed')
except OSError:
 pass
else:
 raise AssertionError('writable resource')
print('readable readonly shard')
'''
        (Path(temporary) / 'foreign').write_text('secret')
        result = subprocess.run(build_sandbox(workspace, [], [], ['/usr/bin/python3', '-c', probe],
                                              resources=catalog), capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert 'readable readonly shard' in result.stdout


def test_catalog_publication_matches_execution_catalog_and_is_readonly(tmp_path):
    from autoresearch.runtime_resources import configured_resources, publish_catalog
    data, target = setup_resources(tmp_path)
    options = {'diagnostic_resources': [{'id': 'c4', 'path': str(data), 'target_roots': [str(target)],
                                         'view': '/resources/c4'}]}
    toolkit = tmp_path / 'public/notes/toolkit'
    public = publish_catalog(toolkit, options)
    assert json.loads(public.read_text()) == configured_resources(options)
    assert public.stat().st_mode & 0o222 == 0


def test_stable_view_rejects_relative_links_escaping_its_root(tmp_path):
    from autoresearch.runtime_resources import resolve_resources
    data, target = setup_resources(tmp_path)
    (data / 'shard').unlink()
    (data / 'shard').symlink_to('../target/shard')
    declaration = {'id': 'c4', 'path': str(data), 'target_roots': [str(target)]}
    assert resolve_resources([declaration])['resources'][0]['view'] == str(data)
    with pytest.raises(ValueError, match='relative link'):
        resolve_resources([{**declaration, 'view': '/resources/c4'}])


def test_researcher_prepares_same_host_catalog_for_executor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from autoresearch import staged_diagnostics
    from autoresearch.runtime_resources import configured_resources
    data, target = setup_resources(tmp_path)
    options = {'diagnostic_resources': [{'id': 'c4', 'path': str(data), 'target_roots': [str(target)],
                                         'view': '/resources/c4'}]}
    repo = tmp_path / 'repo'
    scripts = repo / 'scripts'
    scripts.mkdir(parents=True)
    (scripts / 'probe.py').write_text('print("probe")')
    execution = SimpleNamespace(host=SimpleNamespace(agent_id='research-N001'),
        coral=tmp_path / '.coral', options=options, _check=lambda: None,
        lease=SimpleNamespace(register_job=lambda _: None, clear_job=lambda _: None, borrow=lambda: None))
    def execute(task, payload, cancelled, *, borrowed):
        assert task.read()['runtime']['resources'] == configured_resources(options)
        return {'status': 'completed'}
    monkeypatch.setattr(staged_diagnostics, 'execute_diagnostic', execute)
    result = staged_diagnostics.execute(execution,
        {'scripts_dir': str(scripts), 'context': {'worktree': str(repo)}, 'request_id': 'same-request',
         'argv': ['true'], 'timeout': 1}, repo, tmp_path / 'work', {'observations': []})
    assert result['status'] == 'completed'


def test_resources_cannot_authorize_other_node_tensor_namespaces(tmp_path):
    from autoresearch.runtime_resources import configured_resources
    dynamic = tmp_path / 'dynamic'
    (dynamic / 'coral/run/nodes/N099').mkdir(parents=True)
    with pytest.raises(ValueError, match='artifact namespace'):
        configured_resources({'dynamic_artifact_root': str(dynamic),
            'diagnostic_resources': [{'id': 'all-tensors', 'path': str(dynamic)}]})


def test_parent_after_directory_link_cannot_hide_out_of_scope_target(tmp_path):
    from autoresearch.runtime_resources import resolve_resources
    data, target = setup_resources(tmp_path)
    outside = tmp_path / 'outside'
    (outside / 'deep').mkdir(parents=True)
    (outside / 'secret').write_text('foreign')
    (target / 'jump').symlink_to(outside / 'deep', target_is_directory=True)
    (target / 'secret').write_text('lexical decoy')
    (data / 'shard').unlink()
    (data / 'shard').symlink_to('../target/jump/../secret')
    with pytest.raises(ValueError, match='parent traversal'):
        resolve_resources([{'id': 'data', 'path': str(data),
                            'target_roots': [str(target), str(outside / 'deep')]}])
