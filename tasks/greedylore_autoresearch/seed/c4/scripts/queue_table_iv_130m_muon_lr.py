"""Run two dense Muon LRs, then GreedyLoRE r32/r256 with the winner."""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
QUEUE_DIR = OUTPUTS / "CM040-CM043-muon-matrix-lr"
STEPS = 20000
SCALAR_LR = "0.001"
DENSE_RUNS = (
    ("CM040-m001-dense-muon-llama130m-c4-dense-lr0p01-bf16-s1243", "0.01"),
    ("CM041-m001-dense-muon-llama130m-c4-dense-lr0p005-bf16-s1243", "0.005"),
)


def read_final_loss(outputs, run_id):
    result = json.loads((outputs / run_id / "all_results.json").read_text())
    if result.get("update_step") != STEPS:
        raise ValueError(f"{run_id}: expected {STEPS} steps, got {result.get('update_step')}")
    loss = float(result["final_eval_loss"])
    if not math.isfinite(loss):
        raise ValueError(f"{run_id}: nonfinite final validation loss: {loss}")
    return loss


def choose_best_dense(outputs, runs=DENSE_RUNS):
    """Choose by final validation loss; favor the smaller LR on an exact tie."""
    losses = [(read_final_loss(outputs, run_id), float(lr), lr) for run_id, lr in runs]
    return min(losses)[2]


def idle_gpu_ids(output):
    idle = set()
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        if memory <= 1500 and utilization <= 10:
            idle.add(index)
    return frozenset(idle)


def wait_for_gpus():
    previous = None
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        current = idle_gpu_ids(output)
        common = sorted(previous & current) if previous is not None else []
        print(f"idle GPUs: {sorted(current)}; stable: {common}", flush=True)
        if len(common) >= 4:
            return tuple(common[:4])
        previous = current
        time.sleep(300)


def run_one(run_id, lr, arm, record, common_env, gpus=None):
    if gpus is None:
        gpus = wait_for_gpus()
    env = dict(common_env, RUN_ID=run_id, MUON_MATRIX_LR=lr,
               MUON_SCALAR_LR=SCALAR_LR, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
    record(run_id, "started", f"arm={arm}; matrix_lr={lr}; scalar_lr={SCALAR_LR}; gpus={env['CUDA_VISIBLE_DEVICES']}")
    proc = subprocess.Popen(["bash", "c4/scripts/run_table_iv_130m.bash", arm],
                            cwd=ROOT, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        exit_code = proc.wait()
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
        record(run_id, "interrupted", "controller stopped")
        raise
    if exit_code != 0:
        record(run_id, "failed", f"exit_code={exit_code}")
        raise RuntimeError(f"{run_id}: training exited {exit_code}")
    try:
        loss = read_final_loss(OUTPUTS, run_id)
    except (OSError, ValueError, KeyError) as error:
        record(run_id, "failed", f"invalid result: {error}")
        raise
    record(run_id, "completed", f"step={STEPS}; final_eval_loss={loss}")


def main():
    dataset = ROOT / "c4/c4_en"
    if len(list((dataset / "en").glob("c4-train.*.json.gz"))) != 50:
        raise RuntimeError("expected 50 C4 train shards")
    if len(list((dataset / "en").glob("c4-validation.*.json.gz"))) != 8:
        raise RuntimeError("expected 8 C4 validation shards")
    for prefix in ("CM040-m001-", "CM041-m001-", "CM042-m002-", "CM043-m002-"):
        if list(OUTPUTS.glob(prefix + "*")):
            raise FileExistsError(f"run already exists: {prefix}")
    if QUEUE_DIR.exists():
        raise FileExistsError(f"queue already exists: {QUEUE_DIR}")
    QUEUE_DIR.mkdir(parents=True)
    status = QUEUE_DIR / "status.tsv"
    status.write_text("run\tstate\tdetail\n")

    def record(run_id, state, detail):
        with status.open("a") as file:
            file.write(f"{run_id}\t{state}\t{detail}\n")
        print(f"{run_id}: {state}: {detail}", flush=True)

    common_env = dict(os.environ, C4_DATASET_PATH=str(dataset),
                      HF_HOME=str(Path.home() / ".cache/huggingface"))
    for index, (run_id, lr) in enumerate(DENSE_RUNS):
        run_one(run_id, lr, "dense", record, common_env,
                gpus=(0, 1, 2, 3) if index == 0 else None)
    winner = choose_best_dense(OUTPUTS)
    record("selection", "completed", f"matrix_lr={winner}; criterion=lower final_eval_loss at step {STEPS}")
    lr_tag = winner.replace(".", "p")
    for number, arm in (("CM042", "r32"), ("CM043", "r256")):
        run_id = f"{number}-m002-greedylore-muon-llama130m-c4-{arm}-lr{lr_tag}-bf16-s1243"
        run_one(run_id, winner, arm, record, common_env)
    record("queue", "finished", "all four runs completed")


if __name__ == "__main__":
    main()
