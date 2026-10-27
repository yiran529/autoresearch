"""Real CPU PyTorch checks, invoked in the existing local ML environment."""

import importlib.util
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoConfig


def main():
    seed, output = map(Path, sys.argv[1:3])
    spec = importlib.util.spec_from_file_location('diagnostic_checkpoint', seed / 'diagnostic_checkpoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def step():
        optimizer.zero_grad()
        model(batch).square().mean().backward()
        optimizer.step()
        scheduler.step()

    step()
    hook = SimpleNamespace(process_group=object(), param_to_name={}, param_state=None,
                           iter=12, error_dict={0: torch.tensor([2., 3.])},
                           ms_dict={0: [torch.eye(2)]}, rng=torch.Generator().manual_seed(123),
                           generator={0: torch.Generator().manual_seed(8)})
    config = {
        'compressor': 'top_subspace', 'contract_hash': 'test-contract',
        'lr': 0.01, 'code_sha256': 'a' * 64,
        'model': AutoConfig.from_pretrained(seed / 'c4/configs/llama_130m.json').to_dict(),
        'phases': (1, 2), 'metadata': {'optional': None},
    }
    assert 0 in config['model']['id2label']  # Real HF configuration uses integer keys.
    status = module.save_final_checkpoint(output / 'final', model, optimizer, scheduler, hook,
        config=config,
        update_step=1, batch={'input_ids': batch}, min_free_bytes=0)
    assert status['status'] == 'complete', status
    manifest = json.loads((output / 'final/manifest.json').read_text())
    assert manifest['world_size'] == 1
    assert manifest['update_step'] == 1
    assert set(manifest['files']) == {'model.pt', 'rank-0.pt'}
    expected_rng = (torch.rand(2), random.random(), np.random.rand())
    step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    hook.error_dict[0].zero_()
    module.load_checkpoint(output / 'final', model, optimizer, scheduler, hook,
                           contract_hash='test-contract', expected_config=config)
    assert status['config'] == manifest['config'] == json.loads(json.dumps(config))
    assert 0 in config['model']['id2label']  # Saving must not mutate the caller's config.
    assert torch.equal(torch.rand(2), expected_rng[0])
    assert random.random() == expected_rng[1]
    assert np.random.rand() == expected_rng[2]
    assert hook.error_dict[0].tolist() == [2., 3.]
    assert hook.iter == 12
    assert torch.equal(torch.rand(2, generator=hook.rng),
                       torch.rand(2, generator=torch.Generator().manual_seed(123)))
    step()
    assert all(torch.equal(expected[k], v) for k, v in model.state_dict().items())
    changes = []
    for key, value in [('lr', 0.02), ('code_sha256', 'b' * 64)]:
        changed = deepcopy(config)
        changed[key] = value
        changes.append((changed, key))
    changed = deepcopy(config)
    changed['model']['id2label'][0] = 'changed label'
    changes.append((changed, 'model.id2label.0'))
    changed = deepcopy(config)
    changed['phases'] = (1, 3)
    changes.append((changed, 'phases[1]'))
    changed = deepcopy(config)
    del changed['metadata']['optional']
    changes.append((changed, 'metadata.optional'))
    changed = deepcopy(config)
    changed['extra'] = None
    changes.append((changed, 'extra'))
    for changed, path in changes:
        before = {key: value.clone() for key, value in model.state_dict().items()}
        try:
            module.load_checkpoint(output / 'final', model, expected_config=changed)
        except ValueError as exc:
            assert 'checkpoint code/config mismatch' in str(exc)
            assert path in str(exc), (path, str(exc))
        else:
            raise AssertionError(f'changed configuration accepted: {path}')
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    class DDP:
        def register_comm_hook(self, state, callback):
            self.state, self.callback = state, callback
    class Bucket:
        def parameters(self):
            return [model.weight]
        def index(self):
            return 0
    ddp = DDP()
    hook.param_to_name = {model.weight: 'weight'}
    module.register_checkpoint_hook(ddp, hook, lambda state, bucket: 'measured')
    assert ddp.callback(hook, Bucket()) == 'measured'
    assert hook._checkpoint_bucket_layout == {0: [('weight', (2, 3))]}
    hook._checkpoint_expected_layout = {0: [('another-parameter', (2, 3))]}
    try:
        ddp.callback(hook, Bucket())
    except ValueError as exc:
        assert 'bucket layout' in str(exc)
    else:
        raise AssertionError('wrong bucket layout reused compressed state')
    guarded = module.save_final_checkpoint(output / 'guarded', model, optimizer, scheduler, hook,
        config={}, update_step=2, min_free_bytes=0)
    assert guarded['status'] == 'complete'
    unregistered = SimpleNamespace()
    try:
        module.load_checkpoint(output / 'guarded', model, optimizer, scheduler, unregistered)
    except ValueError as exc:
        assert 'register_checkpoint_hook' in str(exc)
    else:
        raise AssertionError('bucket state restored without layout verification')
    for kwargs, text in [({'contract_hash': 'wrong'}, 'contract'),
                         ({'world_size': 2}, 'world')]:
        try:
            module.load_checkpoint(output / 'final', model, **kwargs)
        except ValueError as exc:
            assert text in str(exc)
        else:
            raise AssertionError('incompatible checkpoint accepted')
    hook.unsupported = object()
    failed = module.save_final_checkpoint(output / 'failed', model, optimizer, scheduler, hook,
        config={}, update_step=2, min_free_bytes=0)
    assert failed['status'] == 'unavailable'
    assert json.loads((output / 'failed/manifest.json').read_text())['status'] == 'unavailable'
    low_disk = module.save_final_checkpoint(output / 'low-disk', model, optimizer, scheduler, None,
        config={}, update_step=2, min_free_bytes=10**30)
    assert low_disk['status'] == 'unavailable'
    assert not (output / 'low-disk/model.pt').exists()
    nonfinite = module.save_final_checkpoint(output / 'nonfinite', model, optimizer, scheduler, None,
        config={'lr': float('nan')}, update_step=2, min_free_bytes=0)
    assert nonfinite['status'] == 'unavailable'
    assert json.loads((output / 'nonfinite/manifest.json').read_text())['status'] == 'unavailable'
    assert not (output / 'nonfinite/model.pt').exists()
    with (output / 'final/rank-0.pt').open('ab') as stream:
        stream.write(b'corrupt')
    try:
        module.load_checkpoint(output / 'final', model)
    except ValueError as exc:
        assert 'integrity' in str(exc)
    else:
        raise AssertionError('corrupt checkpoint accepted')
    print('CPU checkpoint roundtrip, optimizer continuation, RNG, failure and integrity checks passed')


if __name__ == '__main__':
    main()
