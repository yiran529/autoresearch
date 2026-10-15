"""Two parallel dense LR screens, parallel dense/r32, then r256 on four GPUs."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'outputs'
QUEUE = OUTPUTS / 'CM248-CM252-table-iv-350m-muon-existing-data'
DATA = ROOT / 'c4/c4_en'
REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'


def stages(lr):
    def cell(number, arm, lr, steps, gpus):
        method = 'm001-dense' if arm == 'dense' else 'm002-greedylore'
        phase = 'sweep' if steps == 10000 else 'formal'
        return dict(run_id=f'{number}-{method}-muon-llama350m-c4-{arm}-{phase}'
                    f'-lr{lr.replace(".", "p")}-bf16-s1243',
                    arm=arm, lr=lr, steps=steps, gpus=gpus)
    return [
        [cell('CM248', 'dense', '0.01', 10000, '0,1,2,3'),
         cell('CM249', 'dense', '0.005', 10000, '4,5,6,7')],
        [cell('CM250', 'dense', lr, 60000, '0,1,2,3'),
         cell('CM251', 'r32', lr, 60000, '4,5,6,7')],
        [cell('CM252', 'r256', lr, 60000, '0,1,2,3')],
    ]


def read_result(outputs, run_id, steps):
    result = json.loads((outputs / run_id / 'all_results.json').read_text())
    loss = float(result['final_eval_loss'])
    if result.get('update_step') != steps or not math.isfinite(loss):
        raise ValueError(f'{run_id}: incomplete or nonfinite result')
    return result


def choose_lr(outputs, runs):
    return min((float(read_result(outputs, run, 10000)['final_eval_loss']),
                float(lr), lr) for run, lr in runs)[2]


def require_unused_numbers(outputs):
    for number in range(248, 253):
        if list(outputs.glob(f'CM{number}-m*')):
            raise FileExistsError(f'CM{number} already exists')


def command(cell):
    arm = cell['arm']
    rank = 256 if arm == 'r256' else 32
    return [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4',
            'c4/run_llama_pretraining.py', '--model_config', 'c4/configs/llama_350m.json',
            '--dataset_path', str(DATA), '--max_length', '256', '--dtype', 'bfloat16',
            '--num_training_steps', '60000', '--stop_after_steps', str(cell['steps']),
            '--repeat_training_data',
            '--warmup_steps', '6000', '--activation_checkpointing', '--scheduler', 'cosine',
            '--min_lr_ratio', '0.1', '--eval_every', '1000', '--save_every', '120000',
            '--total_batch_size', '512', '--batch_size', '128', '--gradient_accumulation', '1',
            '--workers', '4', '--seed', '1243', '--optimizer', 'muon', '--lr', cell['lr'],
            '--weight_decay', '0', '--muon_mu', '0.95', '--muon_epsilon', '1e-8',
            '--muon_scalar_lr', '0.001', '--muon_scalar_beta1', '0.9',
            '--muon_scalar_beta2', '0.999', '--muon_scalar_eps', '1e-8',
            '--muon_scalar_weight_decay', '0', '--muon_adjust_lr', 'spectral_norm',
            '--compressor', 'none' if arm == 'dense' else 'top_subspace',
            '--compress_rank', str(rank), '--min_compression_rate', '1.0',
            '--start_compress_iter', '1000', '--update_proj_gap', '200',
            '--use_error_feedback', 'ef14', '--beta_ef', '0', '--error_inherit', '0',
            '--grad_clipping', '1.0', '--wandb_project', 'GreedyLore-TableIV',
            '--wandb_job_type', 'formal-pretraining', '--wandb_run_name', cell['run_id'],
            '--output_dir', str(OUTPUTS / cell['run_id'])]


def record(run, state, detail=''):
    line = f'{datetime.now().astimezone().isoformat()}\t{run}\t{state}\t{detail}\n'
    with (QUEUE / 'status.tsv').open('a') as handle:
        handle.write(line)
    print(line, end='', flush=True)


def environment(gpus, mode='online'):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, WANDB_MODE=mode,
                HF_HOME=str(Path.home() / '.cache/huggingface'), HF_HUB_OFFLINE='1',
                HF_DATASETS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                PYTHONPATH=str(ROOT), NCCL_P2P_DISABLE='1', NCCL_SHM_DISABLE='0')


def require_idle(gpus, max_wait=60):
    started = time.monotonic()
    requested = set(map(int, gpus.split(',')))
    while True:
        output = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        idle = set()
        for line in output.splitlines():
            index, memory, utilization = map(int, line.split(','))
            if memory <= 1500 and utilization <= 10:
                idle.add(index)
        if requested <= idle:
            return
        if time.monotonic() - started >= max_wait:
            raise RuntimeError(f'GPUs {gpus} are occupied; not starting this stage')
        time.sleep(5)


def run_stage(cells):
    processes = []
    logs = []
    # Check the entire pair and protect all outputs before launching either job.
    require_idle(','.join(cell['gpus'] for cell in cells))
    for cell in cells:
        if (OUTPUTS / cell['run_id']).exists():
            raise FileExistsError(cell['run_id'])
    try:
        for cell in cells:
            output = OUTPUTS / cell['run_id']
            output.mkdir()
            cmd = command(cell)
            (output / 'command.txt').write_text(shlex.join(cmd) + '\n')
            (output / 'protocol.json').write_text(json.dumps(cell, indent=2) + '\n')
            log = (output / 'train.log').open('x')
            logs.append(log)
            env = environment(cell['gpus'])
            env['WANDB_DIR'] = str(output)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((cell, proc))
            record(cell['run_id'], 'started', f'gpus={cell["gpus"]}; pid={proc.pid}')
        failed = []
        for cell, proc in processes:
            code = proc.wait()
            try:
                if code:
                    raise RuntimeError(f'exit_code={code}')
                result = read_result(OUTPUTS, cell['run_id'], cell['steps'])
                record(cell['run_id'], 'completed', json.dumps(result))
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                failed.append(cell['run_id'])
                record(cell['run_id'], 'failed', str(error))
        if failed:
            raise RuntimeError(f'Stage failed; later stages blocked: {failed}')
    finally:
        for cell, proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait()
                record(cell['run_id'], 'interrupted', 'controller stopped')
        for log in logs:
            log.close()


def check_existing_data():
    train = sorted((DATA / 'en').glob('c4-train.*.json.gz'))
    validation = sorted((DATA / 'en').glob('c4-validation.*.json.gz'))
    if len(train) != 50 or len(validation) != 8:
        raise ValueError('Expected existing 50 train / 8 validation C4 shards; no downloads allowed')
    (QUEUE / 'data-files.json').write_text(json.dumps(dict(
        dataset_path=str(DATA), repeat_training_data=True,
        files=[dict(name=str(path.relative_to(DATA)), bytes=path.stat().st_size)
               for path in train + validation]), indent=2) + '\n')
    record('data', 'completed', 'existing 50 train / 8 validation shards; repeat enabled; no downloads')


def smoke_tests():
    for arm, compressor, rank in [('dense', 'none', 32), ('r32', 'top_subspace', 32),
                                  ('r256', 'top_subspace', 256)]:
        output = QUEUE / f'smoke-{arm}'
        result_file = output / 'all_results.json'
        if result_file.exists():
            result = json.loads(result_file.read_text())
            if result.get('status') != 'completed' or result.get('measured_steps') != 2:
                raise ValueError(f'Invalid saved smoke result: {arm}')
            record(f'smoke-{arm}', 'already-completed')
            continue
        require_idle('0,1,2,3')
        cmd = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4',
               'c4/table_v_timing.py', '--model_config', 'c4/configs/llama_350m.json',
               '--dataset_path', str(DATA), '--batch_size', '128', '--max_length', '256',
               '--dtype', 'bfloat16', '--activation_checkpointing',
               '--warmup_iterations', '3', '--measured_iterations', '2',
               '--scheduler_steps', '60000', '--lr_warmup_steps', '6000',
               '--workers', '4', '--seed', '1243', '--optimizer', 'muon',
               '--lr', '0.01', '--muon_scalar_lr', '0.001', '--muon_mu', '0.95',
               '--muon_adjust_lr', 'spectral_norm', '--weight_decay', '0',
               '--grad_clipping', '1', '--compressor', compressor,
               '--compress_rank', str(rank), '--start_compress_iter', '2',
               '--update_proj_gap', '200', '--use_error_feedback', 'ef14',
               '--min_compression_rate', '1.0', '--beta_ef', '0', '--error_inherit', '0',
               '--output_dir', str(output)]
        record(f'smoke-{arm}', 'started')
        with (QUEUE / f'smoke-{arm}.log').open('x') as log:
            subprocess.run(['timeout', '--signal=TERM', '--kill-after=60s', '20m', *cmd],
                           cwd=ROOT, env=environment('0,1,2,3', 'disabled'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((output / 'all_results.json').read_text())
        if result.get('status') != 'completed' or result.get('measured_steps') != 2:
            raise ValueError(f'Invalid smoke result: {arm}')
        record(f'smoke-{arm}', 'completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Resume an interrupted preflight; formal run IDs must still be unused.')
    args = parser.parse_args()
    if args.dry_run:
        for stage in stages('SELECTED_LR'):
            print('parallel stage:')
            for cell in stage:
                print(f'CUDA_VISIBLE_DEVICES={cell["gpus"]} ' + shlex.join(command(cell)))
        return
    require_unused_numbers(OUTPUTS)
    if args.resume and not QUEUE.is_dir():
        raise FileNotFoundError(QUEUE)
    QUEUE.mkdir(parents=True, exist_ok=args.resume)
    tracked = ['c4/run_llama_pretraining.py', 'c4/scripts/queue_table_iv_350m_muon.py',
               'c4/configs/llama_350m.json', 'optimizer/muon.py', 'optimizer/muon_utils.py',
               'comm_hooks/subspace_hook.py', 'comm_hooks/utils.py',
               'c4/pept_utils/dataloader.py', 'c4/pept_utils/c4_data.py']
    manifest = dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                    text=True).strip(),
                    code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in tracked}, stages=stages('SELECTED_LR'))
    manifest_name = f'resume-manifest-{time.time_ns()}.json' if args.resume else 'manifest.json'
    (QUEUE / manifest_name).write_text(json.dumps(manifest, indent=2) + '\n')
    record('queue', 'started', 'existing data -> smoke -> LR screens -> dense/r32 -> r256; no downloads')
    try:
        check_existing_data()
        smoke_tests()
        sweep = stages('SELECTED_LR')[0]
        run_stage(sweep)
        lr = choose_lr(OUTPUTS, [(cell['run_id'], cell['lr']) for cell in sweep])
        record('selection', 'completed', f'lr={lr}; criterion=final loss at step 10000')
        (QUEUE / 'selection.json').write_text(json.dumps(dict(matrix_lr=lr,
              criterion='final_eval_loss_at_10000', stages=stages(lr)), indent=2) + '\n')
        for stage in stages(lr)[1:]:
            run_stage(stage)
        record('queue', 'finished', 'all five experiments completed')
    except BaseException as error:
        record('queue', 'blocked', f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
