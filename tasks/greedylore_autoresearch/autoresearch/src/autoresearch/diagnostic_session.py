"""Run an Analyzer with writable scratch but no directly accessible GPU devices."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import yaml

from coral.grader.trusted_writes import _atomic_json

from .diagnostic_sandbox import build_sandbox


def settings(coral_dir: Path) -> dict:
    path = coral_dir / 'config.yaml'
    if not path.exists():
        return {}
    config = yaml.safe_load(path.read_text()) or {}
    return config.get('grader', {}).get('args', {})


def run_diagnostic_codex(prompt, read_roots, output_schema, scratch, task):
    """Only explicit runtime inputs and the trusted socket cross the namespace."""
    if not shutil.which('bwrap'):
        raise RuntimeError('bubblewrap is required for diagnostic-enabled Analyzer sessions')
    binary = shutil.which('codex')
    if not binary:
        raise RuntimeError('Codex executable is unavailable')
    config = settings(task.coral_dir)
    runtime_roots = [Path(path).absolute() for path in config.get('diagnostic_runtime_roots', [])]
    # Runtime dependencies are opt-in trusted configuration, never model-supplied.
    # No private grader venv or credentials are included in experiment mounts.
    runtime_path = config.get('diagnostic_path', '/usr/local/bin:/usr/bin:/bin')
    with task.locked():
        state = task.read()
        state['runtime'] = {'readonly_paths': [str(path) for path in runtime_roots],
                            'path': runtime_path, 'gpu_lock_dir': config.get('gpu_lock_dir')}
        _atomic_json(task.path, state)
    tools = task.workspace / 'tools'
    tools.mkdir(exist_ok=True)
    client = tools / 'autoresearch'
    shutil.copyfile(Path(__file__).with_name('diagnostic_client.py'), client)
    client.chmod(0o755)
    schema = task.workspace / 'analysis.schema.json'
    shutil.copyfile(output_schema, schema)
    output = scratch / 'results/output.json'
    home = task.private / 'codex-home'
    home.mkdir(parents=True, exist_ok=True)
    (home / '.codex').mkdir(exist_ok=True)
    # Codex needs its own API credentials. These are never mounted in experiments.
    auth = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'auth.json'
    binary = Path(binary).resolve()
    socket = task.coral_dir / 'public/trusted.sock'
    mounts = [*runtime_roots, binary.parent, *read_roots, schema, tools, socket,
              task.records]
    task.records.mkdir(exist_ok=True)
    for name in ('/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf', '/etc/ssl/certs'):
        if Path(name).exists():
            mounts.append(Path(name).resolve())
    # build_sandbox forbids private mounts; home is a new, otherwise empty directory
    # explicitly added for Codex bookkeeping, not the grader's actual home.
    command = build_sandbox(task.workspace, mounts, [], ['true'], network=True)
    command = command[:-2]  # replace the placeholder command following '--'
    command.extend(['--bind', str(home), str(home)])
    if auth.is_file():
        command.extend(['--ro-bind', str(auth), str(home / '.codex/auth.json')])
    command.extend(['--', str(binary), 'exec', '--ephemeral', '--ignore-user-config',
                    '--sandbox', 'workspace-write', '--skip-git-repo-check', '--cd', str(scratch),
                    '--output-schema', str(schema), '--output-last-message', str(output), '-'])
    environment = {'PATH': f'{tools}:{runtime_path}', 'HOME': str(home),
                   'CODEX_HOME': str(home / '.codex'), 'LANG': 'C.UTF-8',
                   'CORAL_DIAGNOSTIC_RUN': str(task.coral_dir), 'CUDA_VISIBLE_DEVICES': ''}
    # API-key based installations may not have auth.json.
    for key in ('OPENAI_API_KEY', 'SSL_CERT_FILE'):
        if key in os.environ:
            environment[key] = os.environ[key]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env=environment)
    try:
        task.bind(process.pid)
        _stdout, stderr = process.communicate(prompt)
        if process.returncode:
            raise RuntimeError(stderr.strip()[-2000:] or 'diagnostic Analyzer session failed')
        if not output.is_file():
            raise RuntimeError('Analyzer did not produce its output document')
        return output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        task.close()
