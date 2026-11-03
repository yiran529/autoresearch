"""Explicit, seed-independent preparation of a public diagnostic Python runtime."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

from .atomic import atomic_write_text
from .diagnostic_sandbox import build_sandbox
from .runtime_resources import _root, configured_resources

_PROBE = '''
import json, os, sys
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not any(key in os.environ for key in ('OPENAI_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'PYTHONPATH'))
assert not list(Path('/dev').glob('nvidia*'))
import torch
import torch.distributed
import transformers
import torchvision
assert torch.tensor([1., 2.]).sum().item() == 3.
assert torch.distributed.is_available()
print(json.dumps(dict(python=sys.executable, torch=torch.__version__,
    distributed=True, cuda_devices_exposed=False)))
'''


def configure_diagnostic_runtime(config, python):
    """Declare a public venv and its interpreter; never expose the grader venv."""
    python = Path(python).absolute()
    venv = _root(python.parent.parent)
    if not python.is_file():
        raise ValueError(f'diagnostic Python is missing: {python}')
    target = python
    for _ in range(40):
        if not target.is_symlink():
            break
        link = Path(os.readlink(target))
        target = link if link.is_absolute() else target.parent / link
    else:
        raise ValueError('diagnostic Python symlink chain is too long')
    interpreter = target.parent.parent
    # uv's stable interpreter directory can itself be a link to a versioned
    # installation. Mount their shared installation directory to preserve it.
    alias = next((part for part in (*reversed(interpreter.parents), interpreter)
                  if part.is_symlink()), None)
    interpreter = _root(alias.parent if alias else interpreter)
    options = config.setdefault('grader', {}).setdefault('args', {})
    roots = options.setdefault('diagnostic_runtime_roots', [])
    reads = config.setdefault('agents', {}).setdefault('sandbox', {}).setdefault('allow_read', [])
    for root in (str(venv), str(interpreter)):
        if root not in roots:
            roots.append(root)
        if root not in reads:
            reads.append(root)
    path = options.get('diagnostic_path', '/usr/local/bin:/usr/bin:/bin')
    options['diagnostic_path'] = os.pathsep.join(dict.fromkeys([str(python.parent), *path.split(os.pathsep)]))
    options['analyzer_allow_unix_sockets'] = True


def probe_diagnostic_runtime(workspace, options):
    """Load the real ML dependencies in the actual experiment namespace, on CPU."""
    workspace = Path(workspace)
    for name in ('experiments', 'results', 'records'):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    command = build_sandbox(workspace, [], [], ['python', '-I', '-B', '-c', _PROBE],
                            resources=configured_resources(options))
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=60, env={'PATH': options.get('diagnostic_path', '/usr/local/bin:/usr/bin:/bin'),
                        'HOME': '/tmp', 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8',
                        'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'})
    if result.returncode:
        raise ValueError('diagnostic Python runtime probe failed: ' + result.stderr[-2000:])
    return json.loads(result.stdout)


def prepare_diagnostics(coral_dir, python):
    """Save only verified runtime configuration; leave Root and budgets untouched."""
    path = Path(coral_dir) / 'config.yaml'
    config = yaml.safe_load(path.read_text())
    configure_diagnostic_runtime(config, python)
    with tempfile.TemporaryDirectory(prefix='coral-runtime-preflight-') as temporary:
        report = probe_diagnostic_runtime(Path(temporary) / 'workspace', config['grader']['args'])
    atomic_write_text(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return report
