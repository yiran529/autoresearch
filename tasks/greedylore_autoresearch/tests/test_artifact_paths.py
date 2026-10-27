"""Artifact paths must remain usable across actual namespace boundaries."""
# ruff: noqa: S101

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


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
