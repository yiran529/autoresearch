"""Read-only committed-candidate checks in an authenticated host context.

Candidate code is inspected as Git blobs/AST, never imported here. The only
executed namespace code is the fixed host-owned CPU probe below.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from .diagnostic_sandbox import build_sandbox
from .runtime_artifacts import settings
from .runtime_resources import configured_resources
from .submission import validate_submission

_TASK_ROOT = Path(__file__).resolve().parents[3]

# No worktree imports or inherited Python/environment configuration.
_NAMESPACE_PROBE = '''
import json, os, socket, sys
from pathlib import Path
assert not any(k in os.environ for k in ('OPENAI_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'PYTHONPATH'))
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert not list(Path('/dev').glob('nvidia*'))
pending = [Path(value) for value in json.loads(sys.argv[1])]
seen = set()
while pending:
    path = pending.pop()
    info = path.stat()
    identity = (info.st_dev, info.st_ino)
    if identity in seen:
        continue
    seen.add(identity)
    if len(seen) > 4096:
        raise ValueError('namespace resource read budget exceeded')
    if path.is_dir():
        for child in path.iterdir():
            pending.append(child)
            if len(pending) > 4096:
                raise ValueError('namespace resource read budget exceeded')
    else:
        with path.open('rb') as stream:
            stream.read(1)
assert socket.gethostbyname('localhost') == '127.0.0.1'
with socket.socket(socket.AF_INET) as server:
    server.settimeout(5)
    server.bind(('127.0.0.1', 0))
    server.listen()
    with socket.create_connection(server.getsockname(), timeout=5) as client:
        peer, _ = server.accept()
        with peer:
            client.sendall(b'coral')
            assert peer.recv(5) == b'coral'
print('CPU file access, localhost and IPv4 loopback passed')
'''


def probe_namespace(workspace, resources, *, readonly_paths=(), coral_dir=None, required_paths=()):
    """Exercise the actual diagnostic mounts/network without devices or credentials."""
    paths = list(dict.fromkeys([*(row['view'] for row in resources['resources']), *required_paths]))
    command = build_sandbox(workspace, readonly_paths, [],
        ['/usr/bin/python3', '-I', '-c', _NAMESPACE_PROBE, json.dumps(paths)],
        resources=resources, coral_dir=coral_dir)
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        timeout=20, env={'PATH': '/usr/bin:/bin', 'HOME': '/tmp', 'TMPDIR': '/tmp',
                         'LANG': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
    if result.returncode:
        raise ValueError('diagnostic namespace probe failed: ' + result.stderr[-2000:])
    return result.stdout.strip()


def _git(repo, *args):
    result = subprocess.run(['git', *args], cwd=repo, capture_output=True, timeout=10)
    if result.returncode:
        raise ValueError('committed candidate file unavailable: ' + result.stderr.decode(errors='replace')[-1000:])
    return result.stdout


def _entry_sources(repo, sha, contract):
    entries = ['c4/run_llama_pretraining.py', 'c4/table_v_timing.py']
    if 'vision' in contract:
        entries.append('pytorch-cifar/train_ddp.py')
    result = []
    for name in entries:
        entry = _git(repo, 'ls-tree', sha, '--', name).decode().split()
        if len(entry) < 3 or entry[0] not in {'100644', '100755'} or entry[1] != 'blob':
            raise ValueError(f'entry must be a committed regular file: {name}')
        size = int(_git(repo, 'cat-file', '-s', entry[2]))
        if size > 2 * 1024 * 1024:
            raise ValueError(f'entry exceeds static inspection limit: {name}')
        contents = _git(repo, 'cat-file', 'blob', entry[2])
        ast.parse(contents, filename=name)
        result.append({'path': name, 'source': f'git:{sha}:{name}', 'git_blob': entry[2],
                       'sha256': hashlib.sha256(contents).hexdigest(), 'inspection': 'static_ast'})
    return result


def preflight_assignment(store, payload, context):
    """Only the trusted workflow dispatcher supplies identity and host paths."""
    from greedylore_autoresearch_grader.contract import canonical_contract_hash

    from .state import _owned_active

    sha = payload.get('sha')
    report = {'status': 'failed', 'candidate_sha': sha, 'contract_hash': None,
              'entry_sources': [], 'checks': [],
              'scope': 'Committed metadata and static AST entry inspection plus CPU diagnostic namespace checks; '
                       'no candidate imports, dependency loading, training, or semantic equivalence certification.'}
    phase = 'metadata'
    try:
        if set(payload) != {'sha'} or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{40}', sha):
            raise ValueError('preflight requires only a full lowercase candidate SHA')
        repo = Path(context['worktree'])
        contract_path = store.coral_dir / 'private/eval_contract.yaml'
        report['contract_hash'] = canonical_contract_hash(contract_path)
        parents = _git(repo, 'show', '-s', '--format=%P', sha).decode().split()
        canonical = validate_submission(sha, {**context, 'contract_path': str(contract_path),
                                               'parent_hash': parents[0] if parents else None})
        report['checks'].append({'phase': phase, 'status': 'passed'})
        phase = 'ownership'
        state = store._load_unlocked()
        if not _owned_active(state, context.get('agent_id'), canonical['node_id'], canonical['mode']):
            raise ValueError('preflight requires the active Node assignment owner')
        report['checks'].append({'phase': phase, 'status': 'passed'})
        phase = 'protocol_assets'
        options = settings(store.coral_dir)
        research = state['nodes'][canonical['node_id']]['research']
        protocol = research.get('protocol') or ('staged-v2' if options.get('staged_research') is True else 'legacy')
        report['protocol'] = protocol
        if protocol not in {'staged-v2', 'legacy'}:
            raise ValueError(f'unsupported selected protocol: {protocol}')
        root = _TASK_ROOT
        assets = (['roles/workflow_rules.md', 'roles/workflow_research.md', 'roles/researcher.md']
                  if protocol == 'staged-v2' else ['roles/workflow_rules_legacy.md', 'roles/researcher_legacy.md'])
        assets += ['roles/engineer.md', 'templates/submission.yaml', 'references/engineer_commands.md']
        for asset in assets:
            if not (root / asset).is_file():
                raise ValueError(f'selected protocol {protocol} asset missing: {asset}')
        report['checks'].append({'phase': phase, 'status': 'passed', 'assets': assets})
        phase = 'resources'
        resources = configured_resources(options)
        contract = yaml.safe_load(contract_path.read_text())
        required = [contract['data']['c4_path'], contract['data']['tokenizer_path']]
        if 'vision' in contract:
            required.append(contract['vision']['data_path'])
        for value in required:
            path = Path(value)
            if not path.is_absolute() or '..' in path.parts or not path.exists():
                raise ValueError(f'required contract resource missing or invalid: {value}')
            # Existence alone grants nothing: the actual declared namespace must cover it.
            if not any(path.is_relative_to(Path(target)) for _, target in resources['mounts']):
                raise ValueError(f'required contract resource is not declared for diagnostics: {value}')
        report['checks'].append({'phase': phase, 'status': 'passed'})
        phase = 'entry'
        report['entry_sources'] = _entry_sources(repo, sha, contract)
        report['checks'].append({'phase': phase, 'status': 'passed'})
        phase = 'namespace'
        with tempfile.TemporaryDirectory(prefix='coral-preflight-') as temporary:
            workspace = Path(temporary) / 'workspace'
            for name in ('experiments', 'results', 'records'):
                (workspace / name).mkdir(parents=True)
            detail = probe_namespace(workspace, resources, coral_dir=store.coral_dir, required_paths=required)
        report['checks'].append({'phase': phase, 'status': 'passed', 'detail': detail})
        report.update(status='passed', phase='complete')
    except (ValueError, RuntimeError, OSError, SyntaxError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        report.update(phase=phase, reason=str(exc))
        report['checks'].append({'phase': phase, 'status': 'failed', 'reason': str(exc)})
    return report
