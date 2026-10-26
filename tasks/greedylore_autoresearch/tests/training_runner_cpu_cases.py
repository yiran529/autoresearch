"""Exercise the real runner; replace only CUDA/distributed/data/logging boundaries."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

import torch

from training_stream_cpu_cases import equal


def main():
    seed, output = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(seed))
    import c4.run_llama_pretraining as runner
    from diagnostic_checkpoint import load_checkpoint

    fetched = []
    evaluations = []

    class Data:
        def shuffle(self, seed):
            assert seed == 42
            return self

        def __iter__(self):
            for i in range(40):
                fetched.append(i)
                yield {'text': str(i)}

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            value = int(text) + 1
            return {'input_ids': torch.tensor([[value, value + 1, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 0]])}

    config = SimpleNamespace(to_dict=lambda: {'tiny': True}, pad_token_id=0)

    class Model(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.dropout = torch.nn.Dropout(0.25)
            self.config = config
            self.generation_config = SimpleNamespace(pad_token_id=0)

        def forward(self, input_ids, attention_mask=None, labels=None):
            assert torch.equal(labels == -100, input_ids == 0)
            prediction = self.dropout(self.linear(input_ids.float()))
            return SimpleNamespace(loss=(prediction - labels.clamp(min=0)).square().mean())

    def evaluate(*args):
        evaluations.append(True)
        # Validation can consume RNG; periodic saves must follow this consumption.
        return torch.rand(1).item() + 1, 12

    wandb = SimpleNamespace(
        init=lambda **kw: None, config=SimpleNamespace(update=lambda *a, **kw: None),
        save=lambda *a, **kw: None, log=lambda *a, **kw: None,
        run=SimpleNamespace(url='offline', dir=str(output), id='test'),
    )
    original_loader = torch.utils.data.DataLoader
    actual_workers = []
    def recording_loader(*args, **kwargs):
        loader = original_loader(*args, **kwargs)
        actual_workers.append(loader.num_workers)
        return loader

    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and str(args[0]).startswith('cuda'):
            args = ('cpu', *args[1:])
        if str(kwargs.get('device', '')).startswith('cuda'):
            kwargs['device'] = 'cpu'
        return original_to(tensor, *args, **kwargs)

    def args_for(root, *extra):
        root.mkdir(parents=True, exist_ok=True)
        return runner.parse_args([
            '--model_config', str(seed / 'configs/llama_60m.json'),
            '--batch_size', '2', '--workers', '0', '--num_training_steps', '7', '--warmup_steps', '1',
            '--optimizer', 'adamw', '--dtype', 'float32', '--single_gpu', '--eval_every', '3',
            '--checkpoint_root', str(root / 'checkpoints'), '--checkpoint_interval_steps', '3',
            '--training_metrics', str(root / 'metrics.jsonl'), '--output_dir', str(root),
            '--contract_hash', 'runner-test', *extra,
        ])

    with ExitStack() as stack:
        patches = [
        patch.dict('os.environ', {'LOCAL_RANK': '0', 'RANK': '0', 'WORLD_SIZE': '1'}),
        patch.object(runner.dist, 'init_process_group'),
        patch.object(runner.dist, 'get_world_size', return_value=1),
        patch.object(runner.dist, 'get_rank', return_value=0),
        patch.object(runner.dist, 'is_initialized', return_value=False),
        patch.object(torch.cuda, 'set_device'), patch.object(torch.cuda, 'current_device', return_value=0),
        patch.object(torch.cuda, 'is_bf16_supported', return_value=False),
        patch.object(torch.cuda, 'reset_peak_memory_stats'),
        patch.object(torch.cuda, 'max_memory_allocated', return_value=0),
        patch.object(torch.cuda, 'empty_cache'), patch.object(torch.Tensor, 'to', cpu_to),
        patch.object(runner, 'wandb', wandb),
        patch.object(torch.utils.data, 'DataLoader', side_effect=recording_loader),
        patch.object(runner, 'load_c4_split', side_effect=lambda *a, **kw: Data()),
        patch.object(runner.AutoTokenizer, 'from_pretrained', return_value=Tokenizer()),
        patch.object(runner.AutoConfig, 'from_pretrained', return_value=config),
        patch.object(runner, 'LlamaForCausalLM', Model),
        patch.object(runner, 'evaluate_model', side_effect=evaluate)
        ]
        for item in patches:
            stack.enter_context(item)
        default_args = runner.parse_args(['--model_config', 'tiny.json', '--batch_size', '2',
                                          '--optimizer', 'adamw'])
        assert default_args.checkpoint_interval_steps == 1000
        default_args = args_for(output / 'defaults')
        assert default_args.checkpoint_interval_steps == 3
        for bad in ('0', '-3'):
            try:
                args_for(output / 'bad', '--checkpoint_interval_steps', bad)
            except ValueError:
                pass
            else:
                raise AssertionError('nonpositive checkpoint interval accepted')
        runner.main(args_for(output / 'full'))
        assert len(fetched) == 14, fetched  # No terminal overfetch.
        full_evals = len(evaluations)
        fetched.clear()
        evaluations.clear()
        runner.main(args_for(output / 'part', '--pause_after_steps', '2'))
        paused = json.loads((output / 'part/all_results.json').read_text())
        assert paused['status'] == 'paused'
        assert paused['update_step'] == 2
        assert paused['world_size'] == 1 and paused['contract_hash'] == 'runner-test'
        assert len(evaluations) == 1  # Update-1 evaluation, no final scoring.
        assert len(fetched) == 4
        runner.main(args_for(output / 'part', '--resume_checkpoint', paused['checkpoint_path'],
                             '--pause_after_steps', '4'))
        paused = json.loads((output / 'part/all_results.json').read_text())
        assert paused['status'] == 'paused' and paused['update_step'] == 4
        runner.main(args_for(output / 'part', '--resume_checkpoint', paused['checkpoint_path']))
        completed = json.loads((output / 'part/all_results.json').read_text())
        assert completed['status'] == 'completed' and completed['update_step'] == 7
        assert len(evaluations) == full_evals
        assert {p.name for p in (output / 'full/checkpoints').iterdir()} == {
            'step-00000003', 'step-00000006', 'step-00000007'}
        for step in (3, 6, 7):
            left = output / f'full/checkpoints/step-{step:08d}'
            right = output / f'part/checkpoints/step-{step:08d}'
            equal(torch.load(left / 'model.pt', weights_only=True),
                  torch.load(right / 'model.pt', weights_only=True))
            a = torch.load(left / 'rank-0.pt', weights_only=True)
            b = torch.load(right / 'rank-0.pt', weights_only=True)
            for key in ('optimizer', 'scheduler', 'torch_rng', 'python_rng', 'numpy_rng', 'batch'):
                equal(a[key], b[key])
            equal(a['data_stream_state'], b['data_stream_state'])
            assert a['training_state']['global_step'] == step
            assert a['training_state']['tokens_seen'] == step * 4
            assert torch.equal(a['batch']['labels'] == -100, a['batch']['input_ids'] == 0)
        rows = [json.loads(line) for line in (output / 'part/metrics.jsonl').read_text().splitlines()]
        assert [row['update_step'] for row in rows] == list(range(1, 8))
        assert all('loss' in row and 'lr' in row and 'tokens_seen' in row for row in rows)
        # Preserve the quality contract's four-worker configuration instead of
        # changing its source sharding/order when staged checkpoints are enabled.
        runner.main(args_for(output / 'workers4', '--workers', '4', '--pause_after_steps', '2'))
        assert actual_workers[-1] == 4
        workers_saved = torch.load(output / 'workers4/checkpoints/step-00000002/rank-0.pt', weights_only=True)
        loader_identity = workers_saved['data_stream_state']['identity']
        assert loader_identity['workers'] == 4 and loader_identity['requested_workers'] == 4
        assert loader_identity['prefetch_factor'] == 2
        assert loader_identity['persistent_workers'] is False and loader_identity['ordered_delivery'] is True
        manifest = json.loads((output / 'workers4/checkpoints/step-00000002/manifest.json').read_text())
        assert manifest['config']['effective_workers'] == 4
        # The legacy final diagnostic path retains its final score and now saves
        # the actual last input/labels pair without consuming an extra batch.
        fetched.clear()
        legacy = output / 'legacy'
        legacy_args = runner.parse_args([
            '--model_config', 'tiny.json', '--batch_size', '2', '--num_training_steps', '7',
            '--warmup_steps', '1', '--optimizer', 'adamw', '--dtype', 'float32',
            '--single_gpu', '--workers', '0', '--output_dir', str(legacy),
            '--diagnostic_checkpoint', str(legacy / 'checkpoint'),
        ])
        runner.main(legacy_args)
        assert len(fetched) == 14
        result = json.loads((legacy / 'all_results.json').read_text())
        assert result['status'] == 'completed'
        saved = torch.load(legacy / 'checkpoint/rank-0.pt', weights_only=True)
        assert saved['batch']['input_ids'][0, 0].item() == 13
        assert torch.equal(saved['batch']['labels'] == -100, saved['batch']['input_ids'] == 0)
        # Fresh staged starts never accept the old dense-prefix entrypoint.
        try:
            args_for(output / 'invalid-prefix', '--continue_from', str(legacy))
        except ValueError as exc:
            assert 'prefix' in str(exc)
        else:
            raise AssertionError('staged legacy prefix reuse accepted')
    print('real runner CPU pause/restart, periodic/final checkpoints, paired last batch and metrics passed')


if __name__ == '__main__':
    main()
