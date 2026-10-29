"""Artifact paths must remain usable across actual namespace boundaries."""
# ruff: noqa: S101

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_legacy_diagnostic_root_stays_stable_after_sandbox_placeholder_cleanup(tmp_path):
    from autoresearch.artifact_paths import research_diagnostics_root

    coral = tmp_path / '.coral'
    original = coral / 'public/research_diagnostics'
    original.parent.mkdir(parents=True)
    original.touch()
    fallback = research_diagnostics_root(coral)
    # bwrap removes the file it created when the last legacy sandbox exits.
    original.unlink()
    assert research_diagnostics_root(coral) == fallback
    # A later sandbox may prepare the normal directory; this run keeps its root.
    original.mkdir()
    assert research_diagnostics_root(coral) == fallback


def test_broken_legacy_fallback_cannot_silently_switch_publication_root(tmp_path):
    from autoresearch.artifact_paths import research_diagnostics_root

    coral = tmp_path / '.coral'
    fallback = coral / 'public/diagnostics/research_probes'
    fallback.parent.mkdir(parents=True)
    fallback.symlink_to(tmp_path / 'missing')
    assert research_diagnostics_root(coral) == fallback


def test_legacy_root_selection_rejects_symlink_ancestor_before_creating_fallback(tmp_path):
    from autoresearch.artifact_paths import research_diagnostics_root

    coral = tmp_path / '.coral'
    original = coral / 'public/research_diagnostics'
    original.parent.mkdir(parents=True)
    original.touch()
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (coral / 'public/diagnostics').symlink_to(foreign)
    with pytest.raises(ValueError, match='symlink'):
        research_diagnostics_root(coral)
    assert not list(foreign.iterdir())


def test_projection_preserves_host_record_and_rejects_foreign_paths(tmp_path):
    from autoresearch.artifact_paths import PathView

    source, target = tmp_path / 'dynamic', tmp_path / 'public'
    view = PathView([(source, target)])
    result = {'stdout_path': str(source / 'records/r1/stdout.log'), 'status': 'failed',
              'stderr_tail': 'same diagnostic failure'}
    projected = view.project_result(result)
    assert projected['stdout_path'] == str(target / 'records/r1/stdout.log')
    assert result['stdout_path'] == str(source / 'records/r1/stdout.log')
    assert projected['stderr_tail'] == 'same diagnostic failure'
    with pytest.raises(ValueError, match='outside'):
        view.project_result({'stdout_path': str(tmp_path / 'foreign/stdout.log')})
    with pytest.raises(ValueError):
        view.path(source / '../foreign')


def test_prompt_and_path_projection_use_specific_roots_without_prefix_collisions(tmp_path):
    from autoresearch.artifact_paths import PathView

    root = tmp_path / 'artifacts'
    view = PathView([(root, Path('/workspace')),
                     (root / 'special', Path('/workspace/checkpoints/selected'))])
    assert view.path(root / 'special/model.pt') == Path('/workspace/checkpoints/selected/model.pt')
    text = f'code: {root}/code/a.py; checkpoint: {root}/special/model.pt; other: {root}-other/x'
    assert view.rewrite_text(text) == (
        f'code: /workspace/code/a.py; checkpoint: /workspace/checkpoints/selected/model.pt; other: {root}-other/x')


def test_selected_checkpoint_visible_and_readonly_after_nested_tmpfs(tmp_path):
    from autoresearch.artifact_paths import diagnostic_view
    from autoresearch.diagnostic_sandbox import build_sandbox, isolation_available

    if not isolation_available():
        pytest.skip('real bubblewrap namespaces unavailable')
    coral = tmp_path / '.coral'
    workspace = coral / 'public/diagnostics/N001-research-v1/workspace'
    for name in ('experiments', 'results', 'records'):
        (workspace / name).mkdir(parents=True)
    checkpoint = coral / 'public/eval_logs/attempt/run/checkpoints/model'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'model.pt').write_text('selected checkpoint bytes')
    foreign = checkpoint.parent / 'foreign'
    foreign.mkdir()
    (foreign / 'model.pt').write_text('unselected bytes')
    view = diagnostic_view(workspace, checkpoint_paths=[checkpoint])
    alias = view.path(checkpoint)
    probe = """
import errno, json
from pathlib import Path
checkpoint = Path(__ALIAS__)
assert (checkpoint/'model.pt').read_text() == 'selected checkpoint bytes'
assert not Path(__HOST__).exists()
assert not Path(__FOREIGN__).exists()
try:
    (checkpoint/'model.pt').write_text('tampered')
except OSError as exc:
    assert exc.errno in (errno.EROFS, errno.EACCES)
else:
    raise AssertionError('checkpoint writable')
print(json.dumps({'checkpoint_visible': True, 'readonly': True}))
""".replace('__ALIAS__', repr(str(alias))).replace('__HOST__', repr(str(checkpoint))).replace(
        '__FOREIGN__', repr(str(foreign)))
    binary = shutil.which('bwrap')
    inner = [binary, '--die-with-parent', '--unshare-all', '--ro-bind', '/', '/',
             '--tmpfs', '/tmp', '--tmpfs', '/dev/shm', '--', '/usr/bin/python3', '-c', probe]
    command = build_sandbox(workspace, [checkpoint, Path(binary).parent], [], inner,
                            checkpoint_paths=[checkpoint], coral_dir=coral)
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {'checkpoint_visible': True, 'readonly': True}


def test_public_result_readable_in_real_srt_after_dynamic_files_are_removed(tmp_path):
    from autoresearch.artifact_paths import PathView
    from autoresearch.diagnostic_sandbox import isolation_available

    from coral.config import SandboxConfig
    from coral.sandbox.srt import build_srt_settings

    node = Path('/home/wyr/.local/lib/coral-engineer-tools/node/bin/node')
    srt = Path('/home/wyr/.local/lib/coral-engineer-tools/srt/node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js')
    if not (node.is_file() and srt.is_file() and isolation_available()):
        pytest.skip('local SRT installation and namespaces unavailable')
    with tempfile.TemporaryDirectory(prefix='coral-path-test-', dir='/dev/shm') as temporary:
        dynamic = Path(temporary)
        (dynamic / 'records/r1/results').mkdir(parents=True)
        (dynamic / 'records/r1/stdout.log').write_text('probe output')
        (dynamic / 'records/r1/results/value.json').write_text('{"valid":true}')
        raw = {'stdout_path': str(dynamic / 'records/r1/stdout.log'),
               'results_path': str(dynamic / 'records/r1/results')}
        public = tmp_path / '.coral/public/diagnostics/task'
        shutil.copytree(dynamic, public)
        result = PathView([(dynamic, public)]).project_result(raw, published=True)
        settings = tmp_path / 'srt.json'
        worktree, repo = tmp_path / 'agents/researcher', tmp_path / 'repo'
        worktree.mkdir(parents=True)
        (repo / '.git').mkdir(parents=True)
        config = SandboxConfig(enabled=True, network='allowlist',
            extra_settings={'network': {'allowAllUnixSockets': True}})
        settings.write_text(json.dumps(build_srt_settings(config, proxy_port=None,
            worktree_path=worktree, coral_dir=tmp_path / '.coral', repo_dir=repo,
            shared_dir_name='.codex')))
        probe = """
import errno, json
from pathlib import Path
result = __RESULT__
assert not Path(__RAW__).exists()
assert Path(result['stdout_path']).read_text() == 'probe output'
assert json.loads((Path(result['results_path'])/'value.json').read_text()) == {'valid': True}
try:
    Path(result['stdout_path']).write_text('tampered')
except OSError as exc:
    assert exc.errno in (errno.EROFS, errno.EACCES)
else:
    raise AssertionError('public evidence writable')
print('readable and readonly')
""".replace('__RESULT__', repr(result)).replace('__RAW__', repr(raw['stdout_path']))
        environment = dict(os.environ)
        extra = Path('/home/wyr/.local/lib/coral-engineer-tools')
        environment['PATH'] = os.pathsep.join([str(extra / 'bin'), str(extra / 'unpacked/usr/bin'),
                                              environment.get('PATH', os.defpath)])
        command = [str(node), str(srt), '--settings', str(settings), '--',
                   '/usr/bin/python3', '-c', probe]
        for _ in range(2):
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30, env=environment)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            assert 'readable and readonly' in completed.stdout
            shutil.rmtree(dynamic, ignore_errors=True)


def test_checkpoint_metadata_readable_under_home_in_real_srt(tmp_path):
    import yaml
    from autoresearch.diagnostic_sandbox import isolation_available
    from autoresearch.runtime_artifacts import RuntimeArtifacts

    from coral.config import SandboxConfig
    from coral.sandbox.srt import build_srt_settings

    node = Path('/home/wyr/.local/lib/coral-engineer-tools/node-direct/bin/node')
    srt = Path('/home/wyr/.local/lib/coral-engineer-tools/srt/node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js')
    if not (node.is_file() and srt.is_file() and isolation_available()):
        pytest.skip('local SRT installation and namespaces unavailable')
    with tempfile.TemporaryDirectory(prefix='coral-checkpoint-test-', dir=Path.home()) as temporary:
        root = Path(temporary)
        coral = root / 'run/.coral'
        coral.mkdir(parents=True)
        (coral / 'config.yaml').write_text(yaml.safe_dump({'grader': {'args': {
            'dynamic_artifact_root': str(root / 'dynamic'), 'dynamic_artifact_min_free_bytes': 0}}}))
        store = RuntimeArtifacts(coral)
        checkpoint = store.path('N001', 'checkpoints/step-1')
        checkpoint.mkdir(parents=True)
        (checkpoint / 'manifest.json').write_text('{"status":"complete"}')
        (checkpoint / 'model.pt').write_text('selected tensor')
        public = coral / 'public/eval_logs/sha/run/checkpoints/model/manifest.json'
        store.publish_checkpoint('N001', checkpoint, public)
        repo, worktree = root / 'repo', root / 'agents/researcher'
        (repo / '.git').mkdir(parents=True)
        worktree.mkdir(parents=True)
        settings = tmp_path / 'checkpoint-srt.json'
        settings.write_text(json.dumps(build_srt_settings(SandboxConfig(enabled=True,
            network='allowlist', extra_settings={'network': {'allowAllUnixSockets': True}}),
            proxy_port=None, worktree_path=worktree, coral_dir=coral, repo_dir=repo, shared_dir_name='.codex')))
        descriptor = store.checkpoint_descriptor(public)
        probe = f'''import json
from pathlib import Path
assert json.loads(Path({descriptor['manifest_path']!r}).read_text())['status'] == 'complete'
assert not Path({str(checkpoint / 'model.pt')!r}).exists()
try:
 Path({descriptor['manifest_path']!r}).write_text('tampered')
except OSError:
 pass
else:
 raise AssertionError('metadata writable')
print('metadata readable; private tensors denied')
'''
        command = [str(node), str(srt), '--settings', str(settings), '--', '/usr/bin/python3', '-c', probe]
        for _ in range(2):
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
            assert 'private tensors denied' in result.stdout
            store.cleanup_node('N001', terminal=True)
        assert store.checkpoint_descriptor(public)['retention'] == 'cleaned'


def test_projected_checkpoint_diagnostic_path_reads_exact_selected_tensors(tmp_path):
    from autoresearch.diagnostic_sandbox import build_sandbox, isolation_available
    from autoresearch.runtime_artifacts import RuntimeArtifacts
    from test_runtime_artifacts import configured
    if not isolation_available():
        pytest.skip('real bubblewrap namespaces unavailable')
    coral = configured(tmp_path)
    artifacts = RuntimeArtifacts(coral)
    selected = artifacts.path('N001', 'checkpoints/step-1')
    selected.mkdir(parents=True)
    (selected / 'manifest.json').write_text('{"status":"complete"}')
    (selected / 'model.pt').write_text('selected bytes')
    foreign = artifacts.path('N002', 'model.pt')
    foreign.write_text('foreign bytes')
    public = coral / 'public/eval_logs/sha/run/checkpoints/model/manifest.json'
    artifacts.publish_checkpoint('N001', selected, public)
    descriptor = artifacts.project_response({'checkpoint_path': str(selected)})['checkpoint']
    workspace = coral / 'public/diagnostics/task/workspace'
    for name in ('experiments', 'results', 'records'):
        (workspace / name).mkdir(parents=True)
    probe = f'''from pathlib import Path
assert (Path({descriptor['diagnostic_path']!r})/'model.pt').read_text() == 'selected bytes'
assert (Path({str(selected)!r})/'model.pt').read_text() == 'selected bytes'
assert not Path({str(foreign)!r}).exists()
print('projected diagnostic path is usable')
'''
    result = subprocess.run(build_sandbox(workspace, [], [], ['/usr/bin/python3', '-c', probe],
        checkpoint_paths=[selected], coral_dir=coral), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'projected diagnostic path is usable' in result.stdout
