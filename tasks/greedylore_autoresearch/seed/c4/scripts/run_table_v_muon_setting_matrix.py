"""Run the 72-cell GreedyLoRE Muon timing setting matrix serially."""

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
ARTIFACTS = ROOT / 'outputs/CM062-CM133-table-v-muon-setting-matrix'
START_NUMBER = 62
WARMUP_ITERATIONS = 101
MEASURED_ITERATIONS = 400
BUCKET_CAP_MB = 1024
MODELS = (
    ('60m', 'llama_60m_table_iv.json', 10000, 1000, 128),
    ('130m', 'llama_130m.json', 20000, 2000, 128),
    ('350m', 'llama_350m.json', 60000, 6000, 128),
    ('1b', 'llama_1b.json', 100000, 10000, 64),
)
GROUPS = (
    # name, world size, dtype, batch, checkpointing, channels, transport
    ('gl-paper-batch', 4, 'bfloat16', 'paper', True, 'default', 'shm'),
    ('ws4-bf16-b1-default', 4, 'bfloat16', 1, False, 'default', 'shm'),
    ('ws4-fp32-b1-default', 4, 'float32', 1, False, 'default', 'shm'),
    ('ws4-fp32-b1-ch1', 4, 'float32', 1, False, 'one', 'shm'),
    ('ws8-fp32-b1-default', 8, 'float32', 1, False, 'default', 'shm'),
    ('ws8-fp32-b1-ch1', 8, 'float32', 1, False, 'one', 'shm'),
    ('ws8-bf16-b1-ch1', 8, 'bfloat16', 1, False, 'one', 'shm'),
    ('ws8-fp32-b32-ch1', 8, 'float32', 32, False, 'one', 'shm'),
    ('ws8-fp32-b1-ch1-socket', 8, 'float32', 1, False, 'one', 'socket'),
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


def cells():
    number = START_NUMBER
    for group in GROUPS:
        group_name, world_size, dtype, group_batch, checkpointing, channels, transport = group
        for model, config, schedule_steps, lr_warmup_steps, paper_batch in MODELS:
            batch_size = paper_batch if group_batch == 'paper' else group_batch
            for arm, compressor, method in ARMS:
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
                    str(BUCKET_CAP_MB),
                    '--output_dir',
                    str(ARTIFACTS / run_id),
                ]
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
                    'bucket_cap_mb': BUCKET_CAP_MB,
                    'measured_iterations': MEASURED_ITERATIONS,
                    'model': model,
                    'arm': arm,
                    'compressor': compressor,
                    'command': command,
                }
                number += 1


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
            comparisons.append({
                'group': group,
                'model': model,
                'dense_seconds': dense_seconds,
                'greedylore_seconds': method_seconds,
                'speedup': dense_seconds / method_seconds,
                'time_reduction_percent': (
                    (dense_seconds - method_seconds) / dense_seconds * 100
                ),
            })
    (ARTIFACTS / 'comparisons.json').write_text(
        json.dumps(comparisons, indent=2) + '\n', encoding='utf-8'
    )
    report = [
        '# GreedyLoRE Muon timing setting matrix', '',
        '| Group | Model | Dense (s) | GreedyLoRE (s) | Speedup |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for row in comparisons:
        report.append(
            f"| {row['group']} | {row['model']} | {row['dense_seconds']:.6f} | "
            f"{row['greedylore_seconds']:.6f} | {row['speedup']:.4f}x |"
        )
    (ARTIFACTS / 'results.md').write_text(
        '\n'.join(report) + '\n', encoding='utf-8'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    all_gpus = args.gpus.split(',')
    if (
        len(all_gpus) != 8 or len(set(all_gpus)) != 8
        or any(not gpu.isdigit() for gpu in all_gpus)
    ):
        parser.error('--gpus must contain eight distinct GPU indices')

    matrix = list(cells())
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
