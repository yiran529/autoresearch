"""Run an Analyzer with writable scratch but no directly accessible GPU devices."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from coral.grader.trusted_writes import _atomic_json

from .artifact_paths import diagnostic_view
from .atomic import atomic_write_text
from .diagnostic_sandbox import build_sandbox
from .runtime_resources import configured_resources, describe_resources


def settings(coral_dir: Path) -> dict:
    path = coral_dir / 'config.yaml'
    if not path.exists():
        return {}
    config = yaml.safe_load(path.read_text()) or {}
    return config.get('grader', {}).get('args', {})


def analyzer_timeout(coral_dir: Path) -> float:
    value = settings(coral_dir).get('analyzer_timeout_seconds', 18000)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('analyzer_timeout_seconds must be a finite positive number')
    return float(value)


def run_diagnostic_codex(prompt, read_roots, output_schema, scratch, task):
    """Retry transient connections with the same task, workspace and budget."""
    from coral.agent.model_retry import retry_model_connection

    previous = [int(path.name.split('.')[0].removeprefix('session-attempt-'))
                for path in task.private.glob('session-attempt-*.stderr.log')]
    offset = max(previous, default=0)

    def invoke(remaining, attempt):
        if (task.coral_dir / 'private/workflow_stopping').exists():
            raise PermissionError('workflow is stopping')
        return _run_diagnostic_codex_once(
            prompt, read_roots, output_schema, scratch, task,
            timeout=remaining, attempt=offset + attempt,
        )

    return retry_model_connection(invoke, timeout=analyzer_timeout(task.coral_dir))


def _run_diagnostic_codex_once(prompt, read_roots, output_schema, scratch, task, *, timeout, attempt):
    """Only explicit runtime inputs and the trusted socket cross the namespace."""
    if not shutil.which('bwrap'):
        raise RuntimeError('bubblewrap is required for diagnostic-enabled Analyzer sessions')
    binary = shutil.which('codex')
    if not binary:
        raise RuntimeError('Codex executable is unavailable')
    config = settings(task.coral_dir)
    deadline = time.monotonic() + timeout

    def remaining():
        return max(0, deadline - time.monotonic())
    resources = configured_resources(config)
    # Runtime dependencies are opt-in trusted configuration, never model-supplied.
    # No private grader venv or credentials are included in experiment mounts.
    runtime_path = config.get('diagnostic_path', '/usr/local/bin:/usr/bin:/bin')
    with task.locked(timeout=min(5, remaining())):
        state = task.read()
        checkpoint_paths = state.get('checkpoint_paths', [])
        state['runtime'] = {'readonly_paths': [str(path) for path in read_roots],
                            'resources': resources,
                            'path': runtime_path, 'gpu_lock_dir': config.get('gpu_lock_dir')}
        _atomic_json(task.path, state)
    tools = task.workspace / 'tools'
    tools.mkdir(exist_ok=True)
    (task.workspace / 'records').mkdir(exist_ok=True)
    client = tools / 'autoresearch'
    shutil.copyfile(Path(__file__).with_name('diagnostic_client.py'), client)
    from coral.grader import trusted_protocol
    shutil.copyfile(Path(trusted_protocol.__file__), tools / 'trusted_protocol.py')
    client.chmod(0o755)
    schema = task.workspace / 'analysis.schema.json'
    shutil.copyfile(output_schema, schema)
    output = scratch / 'results/output.json'
    # Codex re-execs helper aliases from its home inside a second sandbox.
    # A home on /dev/shm disappears under that sandbox's fresh tmpfs mount.
    # Keep only this small, ephemeral session metadata on the host filesystem;
    # model/experiment tensors and caches remain in the managed dynamic root.
    cache_root = task.private
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='codex-home-', dir=cache_root) as home_dir:
        home = Path(home_dir)
        (home / '.codex').mkdir(exist_ok=True)
        # Codex needs its own API credentials. These are never mounted in experiments.
        auth = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'auth.json'
        binary = Path(binary).resolve()
        isolation_binary = Path(shutil.which('bwrap')).resolve()
        socket = task.coral_dir / 'public/trusted.sock'
        mounts = [binary.parent, isolation_binary.parent, *read_roots, schema, tools, socket,
                  task.records]
        task.records.mkdir(exist_ok=True)
        for name in ('/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf', '/etc/ssl/certs'):
            if Path(name).exists():
                mounts.append(Path(name).resolve())
        # build_sandbox forbids private mounts; home is a new, otherwise empty directory
        # explicitly added for Codex bookkeeping, not the grader's actual home.
        command = build_sandbox(task.workspace, mounts, [], ['true'], network=True,
                                checkpoint_paths=checkpoint_paths, coral_dir=str(task.coral_dir), resources=resources)
        command = command[:-2]  # replace the placeholder command following '--'
        # build_sandbox supplies the same stable view to model and experiments.
        view = diagnostic_view(task.workspace, checkpoint_paths=checkpoint_paths)
        model_workspace = view.path(task.workspace)
        command.extend(['--bind', str(home), str(home)])
        if auth.is_file():
            command.extend(['--ro-bind', str(auth), str(home / '.codex/auth.json')])
        from coral.agent.builtin.codex import DEFAULT_BACKGROUND_TERMINAL_MAX_TIMEOUT_MS

        command.extend(['--', str(binary), 'exec', '--ephemeral', '--ignore-user-config',
                        '-c', f'background_terminal_max_timeout={DEFAULT_BACKGROUND_TERMINAL_MAX_TIMEOUT_MS}'])
        if config.get('analyzer_model'):
            command.extend(['--model', config['analyzer_model']])
        if config.get('analyzer_reasoning_effort'):
            command.extend(['-c', 'model_reasoning_effort=' + json.dumps(config['analyzer_reasoning_effort'])])
        if config.get('analyzer_allow_unix_sockets') is True:
            # Opt-in only for this externally isolated Analyzer. Keep outbound
            # destinations denied by the tool proxy while permitting local IPC.
            for option in (
                'sandbox_workspace_write.network_access=true',
                'features.network_proxy.enabled=true',
                'features.network_proxy.domains={}',
                'features.network_proxy.dangerously_allow_all_unix_sockets=true',
            ):
                command.extend(['-c', option])
        # Codex creates protected metadata paths inside each writable root. Keep its
        # cwd in results rather than the outer namespace's read-only workspace root.
        command.extend(['--sandbox', 'workspace-write', '--skip-git-repo-check',
                        '--cd', str(model_workspace / 'results'),
                        '--add-dir', str(model_workspace / 'experiments'),
                        '--output-schema', str(schema), '--output-last-message', str(output), '-'])
        environment = {'PATH': f"{model_workspace / 'tools'}:{isolation_binary.parent}:{runtime_path}", 'HOME': str(home),
                       'CODEX_HOME': str(home / '.codex'), 'LANG': 'C.UTF-8',
                       'CORAL_DIAGNOSTIC_RUN': str(task.coral_dir), 'CUDA_VISIBLE_DEVICES': ''}
        # API-key based installations may not have auth.json.
        # Preserve model API proxy configuration. Diagnostic experiments receive a
        # separate sanitized environment in diagnostic_sandbox, not these variables.
        for key in ('OPENAI_API_KEY', 'SSL_CERT_FILE', 'HTTP_PROXY', 'HTTPS_PROXY',
                    'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
            if key in os.environ:
                environment[key] = os.environ[key]
        from coral.grader.trusted_worker import (
            prepare_workflow_model,
            register_workflow_model,
            stop_workflow_model,
            workflow_model_environment,
        )

        environment = workflow_model_environment(environment)
        registry = prepare_workflow_model(environment)
        process = None
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=environment,
                                       start_new_session=True)
            register_workflow_model(process, token=environment['CORAL_WORKFLOW_PROCESS_TOKEN'])
            task.bind(process.pid, timeout=min(5, remaining()))
            location = (
                f'\n## Workspace paths\n\nWorkspace root: {model_workspace}\n\n'
                f'- Your initial working directory is {model_workspace / "results"}.\n'
                '- Only experiments/ and results/ beneath the workspace root are writable.\n'
                '- Use absolute workspace paths for reading fixed code/evidence and writing scripts.\n'
                '- '
                'The authoritative diagnostic command argv is resolved relative to the workspace root, '
                'not this initial working directory.\n- Do not change code/ or evidence/.\n\n'
                '- Diagnostic commands use the same /workspace paths.\n'
                '- Read returned logs/results under /workspace/records/<request-id>/; records are read-only.\n'
                '- Cite record_ref or records/<request-id>/... in the final document.\n\n'
                + describe_resources(resources) + '\n\nFixed read roots:\n' + '\n'.join(str(path) for path in read_roots)
            )
            _stdout, stderr = process.communicate(view.rewrite_text(prompt + location),
                                                  timeout=remaining())
            atomic_write_text(task.private / 'session.stdout.log', _stdout)
            atomic_write_text(task.private / 'session.stderr.log', stderr)
            atomic_write_text(task.private / f'session-attempt-{attempt}.stdout.log', _stdout)
            atomic_write_text(task.private / f'session-attempt-{attempt}.stderr.log', stderr)
            if process.returncode:
                from coral.agent.model_retry import check_connection_failure

                check_connection_failure(subprocess.CompletedProcess(
                    command, process.returncode, _stdout, stderr))
                raise RuntimeError(stderr.strip()[-2000:] or 'diagnostic Analyzer session failed')
            if not output.is_file():
                raise RuntimeError('Analyzer did not produce its output document')
            return output
        finally:
            if process is not None:
                stop_workflow_model(process, token=environment['CORAL_WORKFLOW_PROCESS_TOKEN'])
            task.close(timeout=min(5, remaining()))
            if registry is not None:
                registry.unlink(missing_ok=True)
