"""The fixed Root uses the complete current protocol without a development loop."""
# ruff: noqa: S101
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from greedylore_autoresearch_grader.gpu import GpuDevice, gpu_topology_fingerprint
from greedylore_autoresearch_grader.grader import Grader
from greedylore_autoresearch_grader.resources import ALLOCATION_POLICY_VERSION
from test_final_references import references
from test_validation_vision import validation_payload

from coral.config import GraderConfig
from coral.types import Task


def root_grader(tmp_path, monkeypatch, *, invalid_vision=False):
    _, contract, _, reference_root = references(tmp_path)
    devices = tuple(GpuDevice(i, f'GPU-{i}', 'A100', 0, 0, (), 0) for i in range(4))
    nccl = {'NCCL_P2P_DISABLE': '1', 'NCCL_SHM_DISABLE': '0', 'NCCL_DEBUG': 'INFO'}
    topology = gpu_topology_fingerprint(tuple(d.uuid for d in devices), 'test-topology', nccl, (0,)*4)
    environment = dict(gpu_uuids=[d.uuid for d in devices], gpu_names=['A100']*4,
                       topology_fingerprint=topology, numa_nodes=[0]*4, nccl_env=nccl)
    monitoring = dict(allocation_policy_version=ALLOCATION_POLICY_VERSION,
                      interval_seconds=1, evidence_path='monitoring.json')
    (reference_root / 'monitoring.json').write_text(json.dumps(dict(
        kind='monitored_reference', contract_hash=contract.contract_hash,
        environment=environment, allocation_policy_version=ALLOCATION_POLICY_VERSION, interval_seconds=1)))
    path = reference_root / 'efficiency.json'
    reference = json.loads(path.read_text())
    reference.update(environment=environment, monitoring=monitoring)
    path.write_text(json.dumps(reference))

    @contextmanager
    def reserve(*args, **kwargs):
        yield SimpleNamespace(indices=(0, 1, 2, 3), devices=devices, topology='test-topology',
                              uuids=tuple(d.uuid for d in devices), check_exclusive=lambda *_: None)
    monkeypatch.setattr('greedylore_autoresearch_grader.gpu.reserve_idle_gpus', reserve)

    class RootGrader(Grader):
        def __init__(self):
            super().__init__(GraderConfig())
            self.commands = []

        def _run(self, spec):
            command = list(spec.command)
            self.commands.append(command)
            def value(flag):
                actual = flag if flag in command else flag.replace('_', '-')
                return command[command.index(actual)+1]
            output = Path(value('--output_dir'))
            output.mkdir(parents=True, exist_ok=True)
            payload = dict(status='completed', contract_hash=contract.contract_hash)
            if 'c4/run_llama_pretraining.py' in command:
                payload.update(world_size=4, update_step=int(value('--stop_after_steps')), final_eval_loss=3.)
            elif 'pytorch-cifar/train_ddp.py' in command:
                payload.update(validation_payload(contract.quality['resnet18_cifar100_muon']), compressor='top_subspace')
                if invalid_vision:
                    payload['validation_samples'] = 4999
                if invalid_vision in {'array', 'null', 'string'}:
                    payload = {'array': [], 'null': None, 'string': 'bad result'}[invalid_vision]
            else:
                payload.update(measured_steps=400, strict_blocking_communication=False,
                    bucket_layout_recorded=True, observed_bucket_bytes=[[100]],
                    topology_fingerprint=topology, mean_iteration_seconds=.1,
                    first_measured_update=102, last_measured_update=501,
                    timing_cycle={'period_steps': 200, 'rank_event_steps': [[301, 501]]*4})
            (output / 'all_results.json').write_text(json.dumps(payload))
            return SimpleNamespace(timed_out=False, returncode=0, stderr_tail='', infra_failure=None)

    grader = RootGrader()
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    grader.codebase_path, grader.private_dir = str(candidate), str(reference_root.parent)
    grader.tasks = [Task('root', 'Root A0', 'Fixed baseline', dict(
        mode='research', node_id='N000', contract_hash=contract.contract_hash))]
    return grader


def test_root_runs_current_language_budgets_vision_and_fresh_formal_timing(tmp_path, monkeypatch):
    grader = root_grader(tmp_path, monkeypatch)
    bundle = grader.evaluate()
    assert bundle.metadata['failure_class'] is None
    assert bundle.scores['quality_gate'].value == 1
    assert bundle.scores['resnet18_quality_pass'].value == 1
    language = [cmd for cmd in grader.commands if 'c4/run_llama_pretraining.py' in cmd]
    budgets = {Path(cmd[cmd.index('--model_config')+1]).name:
               (int(cmd[cmd.index('--stop_after_steps')+1]), int(cmd[cmd.index('--num_training_steps')+1]))
               for cmd in language}
    assert budgets == {'llama_130m.json': (4000, 20000), 'llama_60m_table_iv.json': (5000, 8393)}
    assert sum('pytorch-cifar/train_ddp.py' in cmd for cmd in grader.commands) == 1
    assert sum('c4/table_v_timing.py' in cmd for cmd in grader.commands) == 4
    assert len(grader.commands) == 7


def test_root_incomplete_vision_result_is_a_candidate_failure(tmp_path, monkeypatch):
    grader = root_grader(tmp_path, monkeypatch, invalid_vision=True)
    bundle = grader.evaluate()
    assert bundle.metadata['failure_class'] == 'candidate'
    assert bundle.aggregated is None
    assert 'vision' in bundle.feedback and 'validation_samples' in bundle.feedback


def test_root_rejects_previous_training_budget_references_before_gpu(tmp_path, monkeypatch):
    import yaml
    from greedylore_autoresearch_grader.contract import canonical_contract_hash
    grader = root_grader(tmp_path, monkeypatch)
    def unexpected_reservation(*args, **kwargs):
        raise AssertionError('invalid references must be rejected before GPU reservation')
    monkeypatch.setattr('greedylore_autoresearch_grader.gpu.reserve_idle_gpus', unexpected_reservation)
    private = Path(grader.private_dir)
    previous = yaml.safe_load((private / 'eval_contract.yaml').read_text())
    previous['quality']['workloads']['llama130m_muon']['stop_after_steps'] = 2500
    previous['quality']['workloads']['llama60m_adamw']['stop_after_steps'] = 4000
    old = tmp_path / 'previous-contract.yaml'
    old.write_text(yaml.safe_dump(previous))
    for name in ('quality', 'efficiency'):
        path = private / 'dense_references' / f'{name}.json'
        payload = json.loads(path.read_text())
        payload['contract_hash'] = canonical_contract_hash(old)
        path.write_text(json.dumps(payload))
    bundle = grader.evaluate()
    assert bundle.metadata['failure_class'] == 'infra'
    assert 'contract hash mismatch' in bundle.feedback
    assert grader.commands == []


@pytest.mark.parametrize('payload', ['array', 'null', 'string'])
def test_root_non_object_vision_json_is_a_candidate_failure(tmp_path, monkeypatch, payload):
    grader = root_grader(tmp_path, monkeypatch, invalid_vision=payload)
    bundle = grader.evaluate()
    assert bundle.metadata['failure_class'] == 'candidate'
    assert bundle.aggregated is None
    assert 'JSON object' in bundle.feedback
