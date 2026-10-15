"""Serial Table V Muon matrix; preserve failed cells and aggregate completed cells."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'outputs/CM048-CM059-table-v-muon'
MODELS = [
    ('60m', 'llama_60m_table_iv.json', 128, 10000, 1000),
    ('130m', 'llama_130m.json', 128, 20000, 2000),
    ('350m', 'llama_350m.json', 128, 60000, 6000),
    ('1b', 'llama_1b.json', 64, 100000, 10000),
]
ARMS = [('dense', 'none', 'm001'), ('powersgd', 'powersgd', 'm004'), ('greedylore', 'top_subspace', 'm002')]


def cells():
    number = 48
    for model, config, batch, schedule, warmup in MODELS:
        for arm, compressor, method in ARMS:
            run_id = f'CM{number:03d}-{method}-{arm}-muon-llama{model}-c4-table-v-bf16-s1243'
            command = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4',
                       'c4/table_v_timing.py', '--model_config', f'c4/configs/{config}',
                       '--dataset_path', 'c4/c4_en', '--batch_size', str(batch),
                       '--max_length', '256', '--dtype', 'bfloat16', '--activation_checkpointing',
                       '--warmup_iterations', '1000', '--measured_iterations', '500',
                       '--scheduler_steps', str(schedule), '--lr_warmup_steps', str(warmup),
                       '--workers', '4', '--seed', '1243', '--optimizer', 'muon',
                       '--lr', '0.01', '--muon_scalar_lr', '0.001', '--muon_mu', '0.95',
                       '--muon_adjust_lr', 'spectral_norm', '--weight_decay', '0',
                       '--muon_scalar_weight_decay', '0', '--grad_clipping', '1',
                       '--compressor', compressor, '--compress_rank', '32',
                       '--start_compress_iter', '1000', '--update_proj_gap', '200',
                       '--use_error_feedback', 'ef14', '--min_compression_rate', '1.0',
                       '--beta_ef', '0', '--error_inherit', '0',
                       '--output_dir', str(ARTIFACTS / run_id)]
            yield dict(run_id=run_id, model=model, arm=arm, command=command)
            number += 1


def timestamp():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gpus', default='0,1,2,3')
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(not g.isdigit() for g in gpus):
        parser.error('--gpus must list four distinct indices')
    matrix = list(cells())
    if args.dry_run:
        for cell in matrix:
            print(shlex.join(cell['command']))
        return
    os.chdir(ROOT)
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, NCCL_P2P_DISABLE='1',
               NCCL_SHM_DISABLE='0', NCCL_DEBUG='INFO', HF_HOME='/dev/shm/wyr_tmp/hf',
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', PYTHONPATH=str(ROOT))
    (ARTIFACTS / 'manifest.json').write_text(json.dumps(dict(started_at=timestamp(), gpus=args.gpus, cells=matrix,
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')
    inventory = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    (ARTIFACTS / 'gpu_inventory.txt').write_text(inventory)
    data_files = sorted((ROOT / 'c4/c4_en/en').glob('c4-train*.json*'))
    (ARTIFACTS / 'data_inventory.json').write_text(json.dumps([dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in data_files], indent=2) + '\n')
    rows = []
    def status(run, state, detail=''):
        with (ARTIFACTS / 'status.tsv').open('a') as handle:
            handle.write(f'{timestamp()}\t{run}\t{state}\t{detail}\n')
        print(f'{run}: {state} {detail}', flush=True)

    def wait_for_idle():
        # Process-owned resource waiting; no agent polling and no interference.
        waiting_logged = False
        while True:
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
            occupied = {line.split(',')[0].strip() for line in raw.splitlines() if line.strip()}
            selected = [line.split(',') for line in inventory.splitlines() if line.split(',')[0].strip() in gpus]
            if len(selected) == 4 and all(uuid.strip() not in occupied and int(free.strip()) >= 23000 for _, uuid, free in selected):
                return
            if not waiting_logged:
                status('queue', 'waiting_for_idle', args.gpus)
                waiting_logged = True
            time.sleep(60)

    status('queue', 'started', 'serial; 12 cells; no automatic batch/sequence reduction')
    for cell in matrix:
        wait_for_idle()
        run_id = cell['run_id']
        status(run_id, 'started')
        (ARTIFACTS / f'{run_id}.command.txt').write_text(shlex.join(cell['command']) + '\n')
        with (ARTIFACTS / f'{run_id}.log').open('x') as log:
            command = ['timeout', '--signal=TERM', '--kill-after=60s', '6h', *cell['command']]
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        result_file = ARTIFACTS / run_id / 'all_results.json'
        row = dict(run_id=run_id, model=cell['model'], arm=cell['arm'], exit_code=completed.returncode)
        if completed.returncode == 0 and result_file.exists():
            result = json.loads(result_file.read_text())
            if result.get('status') != 'completed' or result.get('measured_steps') != 500:
                raise RuntimeError(f'Invalid completed result: {run_id}')
            row.update(status='completed', mean_iteration_seconds=result['mean_iteration_seconds'],
                       parameter_count=result['parameter_count'],
                       peak_allocated_gib=max(result['rank_peak_allocated_bytes']) / 2**30)
        else:
            row.update(status='failed', mean_iteration_seconds=None, parameter_count=None, peak_allocated_gib=None)
            if result_file.exists():
                row['partial_result_present'] = True
        rows.append(row)
        status(run_id, row['status'], f'exit_code={completed.returncode}; seconds={row["mean_iteration_seconds"]}')
        (ARTIFACTS / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (ARTIFACTS / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['run_id','model','arm','exit_code','status','mean_iteration_seconds','parameter_count','peak_allocated_gib'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    report = ['# Table V Muon 本地测速', '', '连续窗口：更新步 1001–1500，共 500 步；以最慢 rank 的总墙钟时间除以 500。',
              '配置和已知偏差见 docs/table-v-muon-protocol.md。失败组不纳入比较。', '',
              '| 模型 | Dense Muon (s) | PowerSGD + Muon (s) | GreedyLore + Muon (s) | GreedyLore 加速比 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for model, *_ in MODELS:
        values = {r['arm']: r['mean_iteration_seconds'] for r in rows if r['model'] == model}
        display = [f'{values[arm]:.6f}' if values[arm] is not None else 'failed' for arm, *_ in ARMS]
        speedup = f'{values["dense"] / values["greedylore"]:.4f}x' if values['dense'] and values['greedylore'] else '—'
        report.append(f'| {model} | ' + ' | '.join(display) + f' | {speedup} |')
    (ARTIFACTS / 'results.md').write_text('\n'.join(report) + '\n')
    status('queue', 'finished', f'completed={sum(r["status"] == "completed" for r in rows)}/12')


if __name__ == '__main__':
    main()
