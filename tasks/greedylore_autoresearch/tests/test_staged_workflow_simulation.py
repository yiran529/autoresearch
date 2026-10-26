"""Real CLI, trusted ownership, staged controller, publication and reclamation."""
# ruff: noqa: S101

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.mark.parametrize('scenario', ['success', 'transfer_quality_failed', 'transfer_failed', 'no_eligible'])
def test_staged_workflow_simulation(tmp_path, scenario):
    import staged_simulation_support

    output = os.environ.get('CORAL_STAGED_SIMULATION_DIR')
    destination = (Path(output) if output else tmp_path) / scenario
    root = Path(tempfile.mkdtemp(prefix='cstage-'))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(sys.path)
    log = root / 'driver.log'
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, str(Path(staged_simulation_support.__file__)),
                                 str(root), scenario], env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=120)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(root, destination, symlinks=True)
    assert result.returncode == 0, log.read_text()[-9000:]
    report = json.loads((root / 'report.json').read_text())
    if output:
        audit_root = Path(output)
        aggregate_path = audit_root / 'report.json'
        aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.exists() else {'simulated': True, 'scenarios': {}}
        aggregate['scenarios'][scenario] = report
        aggregate_path.write_text(json.dumps(aggregate, indent=2) + '\n')
        with (audit_root / 'timeline.jsonl').open('a') as stream:
            for line in (root / 'timeline.jsonl').read_text().splitlines():
                stream.write(json.dumps({**json.loads(line), 'scenario': scenario}) + '\n')
        with (audit_root / 'driver.log').open('a') as stream:
            stream.write(f'{scenario}:\n{log.read_text()}\n')
    assert report['simulated'] is True
    assert report['authenticated_cli'] and report['other_owner_rejected']
    assert report['staged_prompt_delivery']
    assert report['analysis_count'] == 1
    assert report['engineering_status'] == ('completed' if scenario == 'success' else 'skipped')
    assert report['cleanup_status'] == 'done'
    assert report['public_evidence_survives']
    assert report['same_sha_resume'] and report['repeat_request_idempotent']
    assert report['post_freeze_rejected']
    assert report['transfer_runs'] == (0 if scenario == 'no_eligible' else 1)
    assert report['budget_enforced'] == (scenario == 'no_eligible')
