"""Run the 32-cell strict-blocking GreedyLoRE Muon timing matrix serially."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'outputs/CM134-CM165-table-v-muon-strict-blocking'
START_NUMBER = 134
BF16_PAPER_ARTIFACTS = ROOT / 'outputs/CM166-CM173-table-v-muon-strict-blocking-bf16-paper'
BF16_PAPER_START_NUMBER = 166
SYSTEM_MATRIX_ARTIFACTS = ROOT / 'outputs/CM188-CM243-table-v-muon-strict-blocking-system-matrix'
SYSTEM_MATRIX_START_NUMBER = 188
BUCKET_SWEEP_ARTIFACTS = ROOT / 'outputs/CM244-CM247-table-v-muon-bucket-sweep'
BUCKET_SWEEP_START_NUMBER = 244
WARMUP_ITERATIONS = 101
MEASURED_ITERATIONS = 400
BUCKET_CAP_MB = 8192
MODELS = (
    ('60m', 'llama_60m_table_iv.json', 10000, 1000, 128),
    ('130m', 'llama_130m.json', 20000, 2000, 128),
    ('350m', 'llama_350m.json', 60000, 6000, 128),
    ('1b', 'llama_1b.json', 100000, 10000, 64),
)
GROUPS = (
    # name, world size, dtype, batch, checkpointing, channels, transport
    ('paper-batch-ws4', 4, 'float32', 'paper', False, 'default', 'shm'),
    ('b32-ws4', 4, 'float32', 32, False, 'default', 'shm'),
    ('b1-ws4', 4, 'float32', 1, False, 'default', 'shm'),
    ('b1-ws8', 8, 'float32', 1, False, 'default', 'shm'),
)
BF16_PAPER_GROUPS = (
    ('paper-batch-ws4-bf16', 4, 'bfloat16', 'paper', True, 'default', 'shm'),
)
SYSTEM_MATRIX_GROUPS = (
    ('ws4-fp32-default', 4, 'float32', 1, False, 'default', 'shm'),
    ('ws4-fp32-ch1', 4, 'float32', 1, False, 'one', 'shm'),
    ('ws8-fp32-default', 8, 'float32', 1, False, 'default', 'shm'),
    ('ws8-fp32-ch1', 8, 'float32', 1, False, 'one', 'shm'),
    ('ws8-bf16-default', 8, 'bfloat16', 1, False, 'default', 'shm'),
    ('ws8-bf16-ch1', 8, 'bfloat16', 1, False, 'one', 'shm'),
    ('ws8-bf16-socket', 8, 'bfloat16', 1, False, 'one', 'socket'),
)
SYSTEM_MATRIX_MODELS = MODELS
BUCKET_SWEEP_GROUPS = (
    ('ws8-fp32-bucket256', 8, 'float32', 1, False, 'default', 'shm'),
    ('ws8-fp32-bucket1024', 8, 'float32', 1, False, 'default', 'shm'),
)
BUCKET_SWEEP_MODELS = (
    ('350m', 'llama_350m.json', 60000, 6000, 128),
)
ARMS = (
    ('dense', 'none', 'm001'),
    ('greedylore', 'top_subspace', 'm002'),
)
CONTROLLED_NCCL_KEYS = (
    'NCCL_P2P_DISABLE',
    'NCCL_SHM_DISABLE',
    'NCCL_CUMEM_HOST_ENABLE',
    'NCCL_MAX_CTAS',
    'NCCL_MIN_CTAS',
    'NCCL_MAX_NCHANNELS',
    'NCCL_MIN_NCHANNELS',
    'NCCL_NET',
    'NCCL_SOCKET_IFNAME',
    'NCCL_SOCKET_NTHREADS',
    'NCCL_NSOCKS_PERTHREAD',
    'NCCL_DEBUG',
)


def timestamp():
    return datetime.now().astimezone().isoformat()


def cells(
    groups=None, start_number=None, artifacts=None, models=None,
    alternate_arms=False, bucket_cap_mb=BUCKET_CAP_MB,
    strict_blocking=True, record_bucket_layout=False,
):
    groups = GROUPS if groups is None else groups
    number = START_NUMBER if start_number is None else start_number
    artifacts = ARTIFACTS if artifacts is None else artifacts
    models = MODELS if models is None else models
    for group_index, group in enumerate(groups):
        group_name, world_size, dtype, group_batch, checkpointing, channels, transport = group
        for model_index, (model, config, schedule_steps, lr_warmup_steps, paper_batch) in enumerate(models):
            batch_size = paper_batch if group_batch == 'paper' else group_batch
            pair_index = group_index * len(models) + model_index
            arms = tuple(reversed(ARMS)) if alternate_arms and pair_index % 2 else ARMS
            for arm, compressor, method in arms:
                run_id = (
                    f'CM{number:03d}-{method}-{arm}-muon-llama{model}-c4-'
                    f'{group_name}-s1243'
                )
                command = [
                    str(ROOT / '.venv/bin/torchrun'),
                    '--standalone',
                    f'--nproc_per_node={world_size}',
                    '--',
                    'c4/table_v_timing.py',
                    '--model_config',
                    f'c4/configs/{config}',
                    '--dataset_path',
                    'c4/c4_en',
                    '--batch_size',
                    str(batch_size),
                    '--max_length',
                    '256',
                    '--dtype',
                    dtype,
                    '--warmup_iterations',
                    str(WARMUP_ITERATIONS),
                    '--measured_iterations',
                    str(MEASURED_ITERATIONS),
                    '--scheduler_steps',
                    str(schedule_steps),
                    '--lr_warmup_steps',
                    str(lr_warmup_steps),
                    '--workers',
                    '4',
                    '--seed',
                    '1243',
                    '--optimizer',
                    'muon',
                    '--lr',
                    '0.01',
                    '--muon_scalar_lr',
                    '0.001',
                    '--muon_mu',
                    '0.95',
                    '--muon_adjust_lr',
                    'spectral_norm',
                    '--weight_decay',
                    '0',
                    '--muon_scalar_weight_decay',
                    '0',
                    '--grad_clipping',
                    '1',
                    '--compressor',
                    compressor,
                    '--compress_rank',
                    '32',
                    '--start_compress_iter',
                    '100',
                    '--update_proj_gap',
                    '200',
                    '--use_error_feedback',
                    'ef14',
                    '--min_compression_rate',
                    '1.0',
                    '--beta_ef',
                    '0',
                    '--error_inherit',
                    '0',
                    '--ddp_bucket_cap_mb',
                    str(bucket_cap_mb),
                ]
                if strict_blocking:
                    command.append('--strict_blocking_communication')
                if record_bucket_layout:
                    command.append('--record_bucket_layout')
                command.extend(['--output_dir', str(artifacts / run_id)])
                if checkpointing:
                    command.append('--activation_checkpointing')
                yield {
                    'number': number,
                    'run_id': run_id,
                    'group': group_name,
                    'world_size': world_size,
                    'dtype': dtype,
                    'batch_size': batch_size,
                    'activation_checkpointing': checkpointing,
                    'channels': channels,
                    'transport': transport,
                    'bucket_cap_mb': bucket_cap_mb,
                    'strict_blocking_communication': strict_blocking,
                    'record_bucket_layout': record_bucket_layout,
                    'measured_iterations': MEASURED_ITERATIONS,
                    'model': model,
                    'arm': arm,
                    'compressor': compressor,
                    'command': command,
                }
                number += 1


def bf16_paper_cells():
    return cells(
        groups=BF16_PAPER_GROUPS,
        start_number=BF16_PAPER_START_NUMBER,
        artifacts=BF16_PAPER_ARTIFACTS,
    )


def system_matrix_cells():
    return cells(
        groups=SYSTEM_MATRIX_GROUPS,
        start_number=SYSTEM_MATRIX_START_NUMBER,
        artifacts=SYSTEM_MATRIX_ARTIFACTS,
        models=SYSTEM_MATRIX_MODELS,
        alternate_arms=True,
    )


def bucket_sweep_cells():
    number = BUCKET_SWEEP_START_NUMBER
    for group, bucket_cap_mb in zip(BUCKET_SWEEP_GROUPS, (256, 1024)):
        yield from cells(
            groups=(group,),
            start_number=number,
            artifacts=BUCKET_SWEEP_ARTIFACTS,
            models=BUCKET_SWEEP_MODELS,
            bucket_cap_mb=bucket_cap_mb,
            strict_blocking=False,
            record_bucket_layout=True,
        )
        number += len(ARMS) * len(BUCKET_SWEEP_MODELS)


def selected_gpus_are_idle(gpus):
    active = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
        text=True,
    )
    inventory = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=index,uuid,memory.free',
            '--format=csv,noheader,nounits',
        ],
        text=True,
    )
    occupied = {
        line.split(',')[0].strip() for line in active.splitlines() if line.strip()
    }
    selected = [
        line.split(',') for line in inventory.splitlines()
        if line.split(',')[0].strip() in gpus
    ]
    return len(selected) == len(gpus) and all(
        uuid.strip() not in occupied and int(free.strip()) >= 23000
        for _, uuid, free in selected
    )


def environment_for(cell, all_gpus):
    environment = dict(os.environ)
    for key in CONTROLLED_NCCL_KEYS:
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES=','.join(all_gpus[:cell['world_size']]),
        NCCL_P2P_DISABLE='1',
        NCCL_CUMEM_HOST_ENABLE='0',
        NCCL_SHM_DISABLE='0',
        NCCL_DEBUG='INFO',
        HF_HOME='/dev/shm/wyr_tmp/hf',
        HF_HUB_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',
        PYTHONPATH=str(ROOT),
    )
    if cell['channels'] == 'one':
        environment.update(
            NCCL_MAX_CTAS='1',
            NCCL_MIN_CTAS='1',
            NCCL_MAX_NCHANNELS='1',
            NCCL_MIN_NCHANNELS='1',
        )
    if cell['transport'] == 'socket':
        environment.update(
            NCCL_SHM_DISABLE='1',
            NCCL_NET='Socket',
            NCCL_SOCKET_IFNAME='lo',
            NCCL_SOCKET_NTHREADS='1',
            NCCL_NSOCKS_PERTHREAD='1',
        )
    return environment


def write_comparisons(rows):
    completed = {
        (row['group'], row['model'], row['arm']): row
        for row in rows if row['status'] == 'completed'
    }
    comparisons = []
    for group, *_ in GROUPS:
        for model, *_ in MODELS:
            dense = completed.get((group, model, 'dense'))
            greedylore = completed.get((group, model, 'greedylore'))
            if dense is None or greedylore is None:
                continue
            dense_seconds = dense['mean_iteration_seconds']
            method_seconds = greedylore['mean_iteration_seconds']
            dense_hook_seconds = dense.get('mean_blocking_hook_seconds')
            method_hook_seconds = greedylore.get('mean_blocking_hook_seconds')
            comparisons.append({
                'group': group,
                'model': model,
                'dense_seconds': dense_seconds,
                'greedylore_seconds': method_seconds,
                'speedup': dense_seconds / method_seconds,
                'time_reduction_percent': (
                    (dense_seconds - method_seconds) / dense_seconds * 100
                ),
                'dense_blocking_hook_seconds': dense_hook_seconds,
                'greedylore_blocking_hook_seconds': method_hook_seconds,
                'blocking_hook_speedup': (
                    dense_hook_seconds / method_hook_seconds
                    if dense_hook_seconds is not None and method_hook_seconds is not None
                    else None
                ),
            })
    (ARTIFACTS / 'comparisons.json').write_text(
        json.dumps(comparisons, indent=2) + '\n', encoding='utf-8'
    )
    report = [
        '# GreedyLoRE Muon strict-blocking timing matrix', '',
        '| Group | Model | Dense iter (s) | GreedyLoRE iter (s) | Iter speedup | Dense hook (s) | GreedyLoRE hook (s) | Hook speedup |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in comparisons:
        dense_hook = (
            f"{row['dense_blocking_hook_seconds']:.6f}"
            if row['dense_blocking_hook_seconds'] is not None else '—'
        )
        method_hook = (
            f"{row['greedylore_blocking_hook_seconds']:.6f}"
            if row['greedylore_blocking_hook_seconds'] is not None else '—'
        )
        hook_speedup = (
            f"{row['blocking_hook_speedup']:.4f}x"
            if row['blocking_hook_speedup'] is not None else '—'
        )
        report.append(
            f"| {row['group']} | {row['model']} | {row['dense_seconds']:.6f} | "
            f"{row['greedylore_seconds']:.6f} | {row['speedup']:.4f}x | "
            f"{dense_hook} | {method_hook} | {hook_speedup} |"
        )
    (ARTIFACTS / 'results.md').write_text(
        '\n'.join(report) + '\n', encoding='utf-8'
    )


def main():
    global ARTIFACTS, START_NUMBER, GROUPS, MODELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument(
        '--matrix', choices=['fp32', 'bf16-paper', 'system', 'bucket'], default='fp32'
    )
    args = parser.parse_args()
    if args.matrix == 'bf16-paper':
        ARTIFACTS = BF16_PAPER_ARTIFACTS
        START_NUMBER = BF16_PAPER_START_NUMBER
        GROUPS = BF16_PAPER_GROUPS
    elif args.matrix == 'system':
        ARTIFACTS = SYSTEM_MATRIX_ARTIFACTS
        START_NUMBER = SYSTEM_MATRIX_START_NUMBER
        GROUPS = SYSTEM_MATRIX_GROUPS
        MODELS = SYSTEM_MATRIX_MODELS
    elif args.matrix == 'bucket':
        ARTIFACTS = BUCKET_SWEEP_ARTIFACTS
        START_NUMBER = BUCKET_SWEEP_START_NUMBER
        GROUPS = BUCKET_SWEEP_GROUPS
        MODELS = BUCKET_SWEEP_MODELS
    all_gpus = args.gpus.split(',')
    if (
        len(all_gpus) != 8 or len(set(all_gpus)) != 8
        or any(not gpu.isdigit() for gpu in all_gpus)
    ):
        parser.error('--gpus must contain eight distinct GPU indices')

    matrix = (
        list(bucket_sweep_cells()) if args.matrix == 'bucket'
        else list(cells(alternate_arms=args.matrix == 'system'))
    )
    if args.dry_run:
        print(
            f'cells={len(matrix)} ids=CM{matrix[0]["number"]:03d}-'
            f'CM{matrix[-1]["number"]:03d}'
        )
        for cell in matrix:
            print(shlex.join(cell['command']))
        return

    os.chdir(ROOT)
    if not ROOT.joinpath('c4/c4_en/en').is_dir():
        raise FileNotFoundError('C4 data directory does not exist')
    if not selected_gpus_are_idle(all_gpus):
        raise RuntimeError('Selected GPUs are not idle')
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    (ARTIFACTS / 'manifest.json').write_text(
        json.dumps({
            'started_at': timestamp(),
            'gpus': all_gpus,
            'cell_count': len(matrix),
            'cells': matrix,
            'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, indent=2) + '\n',
        encoding='utf-8',
    )
    (ARTIFACTS / 'gpu_inventory.txt').write_text(
        subprocess.check_output(['nvidia-smi', '-L'], text=True), encoding='utf-8'
    )
    data_files = sorted(ROOT.joinpath('c4/c4_en/en').glob('c4-train*.json*'))
    (ARTIFACTS / 'data_inventory.json').write_text(
        json.dumps([
            {'path': str(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            for path in data_files
        ], indent=2) + '\n',
        encoding='utf-8',
    )
    rows = []

    def record_status(run_id, state, detail=''):
        with (ARTIFACTS / 'status.tsv').open('a', encoding='utf-8') as handle:
            handle.write(f'{timestamp()}\t{run_id}\t{state}\t{detail}\n')
        print(f'{run_id}: {state} {detail}', flush=True)

    record_status('queue', 'started', f'serial; {len(matrix)} cells; OOM continues')
    for cell in matrix:
        run_id = cell['run_id']
        environment = environment_for(cell, all_gpus)
        record_status(run_id, 'started', cell['group'])
        (ARTIFACTS / f'{run_id}.command.txt').write_text(
            shlex.join(cell['command']) + '\n', encoding='utf-8'
        )
        (ARTIFACTS / f'{run_id}.environment.json').write_text(
            json.dumps({
                key: environment[key]
                for key in ('CUDA_VISIBLE_DEVICES', *CONTROLLED_NCCL_KEYS)
                if key in environment
            }, indent=2) + '\n',
            encoding='utf-8',
        )
        with (ARTIFACTS / f'{run_id}.log').open('x', encoding='utf-8') as log:
            completed = subprocess.run(
                ['timeout', '--signal=TERM', '--kill-after=60s', '6h', *cell['command']],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result_path = ARTIFACTS / run_id / 'all_results.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        accepted = (
            completed.returncode == 0 and result is not None
            and result.get('status') == 'completed'
            and result.get('measured_steps') == MEASURED_ITERATIONS
            and result.get('optimizer') == 'muon'
            and result.get('compressor') == cell['compressor']
            and result.get('strict_blocking_communication')
                is cell['strict_blocking_communication']
            and (
                'mean_blocking_hook_seconds' in result
                if cell['strict_blocking_communication']
                else result.get('bucket_layout_recorded') is True
            )
        )
        row = {
            key: cell[key] for key in (
                'run_id', 'group', 'world_size', 'dtype', 'batch_size',
                'activation_checkpointing', 'channels', 'transport', 'model', 'arm',
            )
        }
        row.update(
            exit_code=completed.returncode,
            status='completed' if accepted else 'failed',
            mean_iteration_seconds=result['mean_iteration_seconds'] if accepted else None,
            mean_blocking_hook_seconds=(
                result.get('mean_blocking_hook_seconds') if accepted else None
            ),
            observed_bucket_counts=(
                result['observed_bucket_counts'] if accepted else None
            ),
            observed_bucket_bytes=(
                result['observed_bucket_bytes'] if accepted else None
            ),
            parameter_count=result['parameter_count'] if accepted else None,
            peak_allocated_gib=(
                max(result['rank_peak_allocated_bytes']) / 2**30 if accepted else None
            ),
        )
        rows.append(row)
        record_status(
            run_id, row['status'],
            f'exit_code={completed.returncode}; seconds={row["mean_iteration_seconds"]}',
        )
        (ARTIFACTS / 'summary.json').write_text(
            json.dumps(rows, indent=2) + '\n', encoding='utf-8'
        )

    with (ARTIFACTS / 'summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_comparisons(rows)
    record_status(
        'queue', 'finished',
        f'completed={sum(row["status"] == "completed" for row in rows)}/{len(rows)}',
    )


if __name__ == '__main__':
    main()
