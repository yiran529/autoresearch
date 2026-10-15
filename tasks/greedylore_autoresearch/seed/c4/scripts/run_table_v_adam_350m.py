"""Run the 350M Dense/GreedyLore AdamW Table V timing pair serially."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'outputs/CM060-CM061-table-v-adam-350m'
ARMS = [
    ('CM060-m003-dense-adamw-llama350m-c4-table-v-bf16-s1243', 'dense', 'none'),
    ('CM061-m005-greedylore-adamw-llama350m-c4-table-v-r32-bf16-s1243',
     'greedylore', 'top_subspace'),
]


def cells():
    for run_id, arm, compressor in ARMS:
        command = [
            str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4',
            'c4/table_v_timing.py', '--model_config', 'c4/configs/llama_350m.json',
            '--dataset_path', 'c4/c4_en', '--batch_size', '128',
            '--max_length', '256', '--dtype', 'bfloat16', '--activation_checkpointing',
            '--warmup_iterations', '100', '--measured_iterations', '500',
            '--scheduler_steps', '60000', '--lr_warmup_steps', '6000',
            '--workers', '4', '--seed', '1243', '--optimizer', 'adamw',
            '--lr', '0.001', '--beta1', '0.9', '--beta2', '0.999', '--eps', '1e-8',
            '--weight_decay', '0', '--grad_clipping', '1',
            '--compressor', compressor, '--compress_rank', '32',
            '--start_compress_iter', '100', '--update_proj_gap', '200',
            '--use_error_feedback', 'ef14', '--min_compression_rate', '1.0',
            '--beta_ef', '0', '--error_inherit', '0',
            '--output_dir', str(ARTIFACTS / run_id),
        ]
        yield {'run_id': run_id, 'arm': arm, 'command': command}


def timestamp():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gpus', default='0,1,2,3')
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(not gpu.isdigit() for gpu in gpus):
        parser.error('--gpus must list four distinct indices')
    matrix = list(cells())
    if args.dry_run:
        for cell in matrix:
            print(shlex.join(cell['command']))
        return

    os.chdir(ROOT)
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=','.join(gpus), NCCL_P2P_DISABLE='1',
        NCCL_SHM_DISABLE='0', NCCL_DEBUG='INFO', HF_HOME='/dev/shm/wyr_tmp/hf',
        HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
        PYTHONPATH=str(ROOT),
    )
    (ARTIFACTS / 'manifest.json').write_text(json.dumps({
        'started_at': timestamp(), 'gpus': gpus, 'cells': matrix,
    }, indent=2) + '\n')
    rows = []
    for cell in matrix:
        run_id = cell['run_id']
        with (ARTIFACTS / 'status.tsv').open('a') as handle:
            handle.write(f'{timestamp()}\t{run_id}\tstarted\n')
        (ARTIFACTS / f'{run_id}.command.txt').write_text(
            shlex.join(cell['command']) + '\n'
        )
        with (ARTIFACTS / f'{run_id}.log').open('x') as log:
            completed = subprocess.run(
                ['timeout', '--signal=TERM', '--kill-after=60s', '4h', *cell['command']],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        result_file = ARTIFACTS / run_id / 'all_results.json'
        row = {'run_id': run_id, 'arm': cell['arm'], 'exit_code': completed.returncode}
        if completed.returncode == 0 and result_file.exists():
            result = json.loads(result_file.read_text())
            if result.get('status') != 'completed' or result.get('measured_steps') != 500:
                raise RuntimeError(f'Invalid completed result: {run_id}')
            row.update(status='completed',
                       mean_iteration_seconds=result['mean_iteration_seconds'])
        else:
            row.update(status='failed', mean_iteration_seconds=None)
        rows.append(row)
        with (ARTIFACTS / 'status.tsv').open('a') as handle:
            handle.write(
                f'{timestamp()}\t{run_id}\t{row["status"]}\t'
                f'exit_code={completed.returncode}; seconds={row["mean_iteration_seconds"]}\n'
            )
        (ARTIFACTS / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')

    with (ARTIFACTS / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['run_id', 'arm', 'exit_code', 'status',
                                'mean_iteration_seconds'],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
