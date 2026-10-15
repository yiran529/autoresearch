"""Serial r64 LR sweep, gated by validation loss at update step 5000."""

import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
STEP = 5000
RESULT_STEP = 20000
EVAL_LINE = re.compile(r"Eval loss at step (\d+): (nan|[+-]?inf|[0-9.eE+-]+)", re.IGNORECASE)
RUNS = (
    ("CM036-m002-greedylore-muon-llama130m-c4-r64-lr0p004-slr0p001-bf16-s1243", "0.004", "0.001"),
    ("CM037-m002-greedylore-muon-llama130m-c4-r64-lr0p1-slr0p001-bf16-s1243", "0.1", "0.001"),
    ("CM038-m002-greedylore-muon-llama130m-c4-r64-lr0p02-slr0p0002-bf16-s1243", "0.02", "0.0002"),
    ("CM039-m002-greedylore-muon-llama130m-c4-r64-lr0p02-slr0p005-bf16-s1243", "0.02", "0.005"),
)
BASELINE_ID = "CM033-m002-greedylore-muon-llama130m-c4-r64-bf16-s1243"


def read_eval_loss(log_path, step):
    path = Path(log_path)
    if not path.exists():
        return None
    loss = None
    for match in EVAL_LINE.finditer(path.read_text(errors="replace")):
        if int(match.group(1)) == step:
            loss = float(match.group(2))
    return loss


def stop_process_group(proc):
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_screened(command, log_path, baseline_loss, poll_seconds=1, env=None):
    proc = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    improved = False
    try:
        while True:
            loss = read_eval_loss(log_path, STEP)
            if loss is not None and not improved:
                if not math.isfinite(loss):
                    stop_process_group(proc)
                    return "stopped_nonfinite"
                if loss >= baseline_loss:
                    stop_process_group(proc)
                    return "stopped_no_improvement"
                improved = True
            exit_code = proc.poll()
            if exit_code is not None:
                return "completed_improved" if improved and exit_code == 0 else "failed"
            time.sleep(poll_seconds)
    except BaseException:
        if proc.poll() is None:
            stop_process_group(proc)
        raise


def idle_gpu_ids(output):
    idle = set()
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        if memory <= 1500 and utilization <= 10:
            idle.add(index)
    return frozenset(idle)


def select_stable_gpus(previous, current):
    if previous is None:
        return None
    common = sorted(previous & current)
    return tuple(common[:4]) if len(common) >= 4 else None


def wait_for_gpus():
    previous = None
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        current = idle_gpu_ids(output)
        selected = select_stable_gpus(previous, current)
        if selected is not None:
            return selected
        previous = current
        time.sleep(300)


def full_result_complete(run_id):
    result = OUTPUTS / run_id / "all_results.json"
    return result.exists() and json.loads(result.read_text()).get("update_step") == RESULT_STEP


def main():
    queue_dir = OUTPUTS / "CM036-CM039-r64-lr-sweep"
    queue_dir.mkdir(parents=True, exist_ok=True)
    status = queue_dir / "status.tsv"
    if status.exists():
        raise FileExistsError(f"queue already started: {status}")
    status.write_text("run\tstate\tdetail\n")

    def record(run_id, state, detail):
        with status.open("a") as file:
            file.write(f"{run_id}\t{state}\t{detail}\n")

    for run_id, _, _ in RUNS:
        if (OUTPUTS / run_id).exists():
            raise FileExistsError(f"run already exists: {run_id}")

    dataset = ROOT / "c4/c4_en"
    if len(list((dataset / "en").glob("c4-train.*.json.gz"))) != 50:
        raise RuntimeError("expected 50 C4 train shards")
    if len(list((dataset / "en").glob("c4-validation.*.json.gz"))) != 8:
        raise RuntimeError("expected 8 C4 validation shards")

    baseline_log = OUTPUTS / BASELINE_ID / "train.log"
    baseline_loss = read_eval_loss(baseline_log, STEP)
    if baseline_loss is None or not math.isfinite(baseline_loss) or not full_result_complete(BASELINE_ID):
        raise RuntimeError("CM033 reference is missing its step-5000 loss or complete result")
    record(BASELINE_ID, "reference", f"step5000_loss={baseline_loss}; prior C4 data")

    common_env = dict(os.environ, C4_DATASET_PATH=str(dataset),
                      HF_HOME=str(Path.home() / ".cache/huggingface"))

    for run_id, matrix_lr, scalar_lr in RUNS:
        gpus = wait_for_gpus()
        env = dict(common_env, RUN_ID=run_id, MUON_MATRIX_LR=matrix_lr,
                   MUON_SCALAR_LR=scalar_lr, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
        record(run_id, "started", f"matrix_lr={matrix_lr}; scalar_lr={scalar_lr}; gpus={env['CUDA_VISIBLE_DEVICES']}")
        log = OUTPUTS / run_id / "train.log"
        outcome = run_screened(["bash", "c4/scripts/run_table_iv_130m.bash", "r64"],
                               log, baseline_loss, env=env)
        loss = read_eval_loss(log, STEP)
        if outcome == "completed_improved" and not full_result_complete(run_id):
            outcome = "failed"
        record(run_id, outcome, f"step5000_loss={loss}; baseline_loss={baseline_loss}")

    record("queue", "finished", "all candidates processed")


if __name__ == "__main__":
    main()
