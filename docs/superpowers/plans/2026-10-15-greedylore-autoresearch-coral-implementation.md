# GreedyLoRE CORAL AutoResearch V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在锁定的 CORAL fork 中交付一个以 GreedyLoRE r32/EF14 为 Root、支持 Research/Engineering 分流、受控证据发布和固定 GPU 评测合同的 AutoResearch V1 task。

**Architecture:** CORAL 只增加 heartbeat 关闭、validation metadata、受控本地 RPC、同 SHA retry 和 SRT 保护等通用能力。GreedyLoRE task 自己拥有 evaluation contract、grader、Node/Analysis/Review 规则和模板；正式 quality 使用 60M+AdamW 完训与 130M+Muon 前 2,500 updates，正式 efficiency 只使用 130M+AdamW 的 BF16/FP32 两个 cell。

**Tech Stack:** Python 3.11、CORAL dataclass/OmegaConf 配置、pytest、Unix domain socket、PyYAML、PyTorch 2.4 DDP/NCCL、Hugging Face Transformers/Datasets、Git worktree、SRT/bubblewrap。

**Spec:** `docs/superpowers/specs/2026-10-15-greedylore-autoresearch-coral-design.md`

## Global Constraints

- CORAL 基线固定为 `0123dfb939b35228cf2c1fde224cd0561e727408`；GreedyLoRE 来源固定为 `d2c38bc2062e06fdfd73185c508e7cc652933de6`。
- C4 只读路径固定为 `/home/wyr/greedy_lore/c4/c4_en/en`。
- tokenizer 固定为 `/home/wyr/.cache/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`，所有正式运行设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。
- 所有正式 workload 使用 4 GPU；不硬编码物理卡号，从所有空闲 GPU 中原子预留任意 4 张，不足 4 张时报告 `failure_class=infra`。
- quality gate 对两个 workload 分别要求 `candidate_final_eval_loss <= 1.10 * dense_reference_final_eval_loss`，必须同时通过。
- efficiency 只有 `130m_adamw_bf16` 和 `130m_adamw_fp32`；每卡 batch 1、SHM、异步 DDP、101 warmup、400 measured、每 cell 两次。
- 正式 timing 不传 `--strict_blocking_communication`，只用被动 bucket layout 记录；Dense/candidate 都使用两次均值，综合分数为两个 speedup 的几何平均。
- Agent 不能直接写 Attempts、eval logs、Node history、Analysis/Review 或 private data；所有发布通过受信任 broker。
- 每个下列 Task 是一个“大 step”：其测试通过后立即创建一次 commit；commit 本身不触发 push、长训练或 calibration。
- 未实际执行的 GPU smoke、dense calibration、完整训练和 SRT 攻击测试不得标记为通过。

---

### Task 1: 建立可复核的 GreedyLoRE task seed 与固定 contract

**Files:**
- Create: `tasks/greedylore_autoresearch/scripts/copy_seed.py`
- Create: `tasks/greedylore_autoresearch/task.yaml`
- Create: `tasks/greedylore_autoresearch/taskdata/eval_contract.yaml`
- Create: `tasks/greedylore_autoresearch/taskdata/dense_references/.gitkeep`
- Create: `tasks/greedylore_autoresearch/seed/`（由脚本从固定 commit 生成）
- Create: `tasks/greedylore_autoresearch/tests/test_seed_manifest.py`
- Create: `tasks/greedylore_autoresearch/tests/test_contract_static.py`

**Interfaces:**
- Consumes: clean `/home/wyr/greedy_lore` checkout at the pinned SHA.
- Produces: `copy_seed(source: Path, destination: Path, expected_sha: str) -> dict[str, object]`；`seed/source_manifest.json`；后续 grader 唯一读取的 `taskdata/eval_contract.yaml`。

- [ ] **Step 1: 写 manifest 与 contract 的失败测试**

```python
def test_manifest_covers_exact_tracked_source_files(seed_dir, source_repo):
    manifest = json.loads((seed_dir / "source_manifest.json").read_text())
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=source_repo
    ).decode().rstrip("\0").split("\0")
    assert manifest["source_commit"] == "d2c38bc2062e06fdfd73185c508e7cc652933de6"
    assert sorted(manifest["files"]) == sorted(tracked)
    assert not (seed_dir / ".git").exists()


def test_contract_has_reviewed_paths_and_two_efficiency_cells(contract):
    assert contract["data"]["c4_path"] == "/home/wyr/greedy_lore/c4/c4_en/en"
    assert contract["data"]["tokenizer_path"].endswith(
        "models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    )
    assert list(contract["efficiency"]["cells"]) == [
        "130m_adamw_bf16", "130m_adamw_fp32"
    ]
    assert contract["efficiency"]["repetitions"] == 2
```

- [ ] **Step 2: 运行测试并确认因 task 文件缺失而失败**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_seed_manifest.py tasks/greedylore_autoresearch/tests/test_contract_static.py -v`

Expected: FAIL，错误指向缺少 `copy_seed.py`、manifest 或 contract。

- [ ] **Step 3: 实现拒绝脏源和错误 SHA 的复制脚本**

```python
def copy_seed(source: Path, destination: Path, expected_sha: str) -> dict[str, object]:
    head = git(source, "rev-parse", "HEAD").strip()
    if head != expected_sha:
        raise RuntimeError(f"source HEAD {head} != pinned {expected_sha}")
    if git(source, "status", "--porcelain").strip():
        raise RuntimeError("source repository is dirty")
    tracked = git(source, "ls-files", "-z").rstrip("\0").split("\0")
    for relative in tracked:
        src = source / relative
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    files = {
        relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
        for relative in tracked
    }
    return {"source_commit": head, "files": files}
```

生成 seed 后把源 `AGENTS.md` 移到 `seed/docs/greedylore-agent-guidance.md`；manifest 同时记录该重定位和 `locked_coral_commit`。不得复制 `.git`、`.venv`、outputs、cache、dataset 或日志。

- [ ] **Step 4: 写入无隐含默认值的 YAML contract**

Contract 明确编码：4 GPU、seed 1243、seq 256；60M AdamW 8,393 updates；130M Muon `scheduler_steps=20000/stop_after_steps=2500`；quality compression start 1000；timing compression start 100；AdamW LR 0.0025；两个 timing cells、两次顺序 `BF16,FP32` 与 `FP32,BF16`；GPU idle 阈值 `compute_processes=0`、`memory_used_mib<=512`、`utilization_percent<=5`。

- [ ] **Step 5: 生成 seed 并验证源文件 hash**

Run: `uv run python tasks/greedylore_autoresearch/scripts/copy_seed.py --source /home/wyr/greedy_lore --destination tasks/greedylore_autoresearch/seed --expected-sha d2c38bc2062e06fdfd73185c508e7cc652933de6 --coral-sha 0123dfb939b35228cf2c1fde224cd0561e727408`

Expected: 输出 `copied=119`，manifest 中每个 source SHA-256 重新计算一致。

- [ ] **Step 6: 运行 task 静态测试**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_seed_manifest.py tasks/greedylore_autoresearch/tests/test_contract_static.py -v`

Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add tasks/greedylore_autoresearch
git commit -m "feat: scaffold GreedyLoRE autoresearch task"
```

---

### Task 2: 增加 heartbeat disable 与 validation metadata

**Files:**
- Modify: `coral/config.py`
- Modify: `coral/agent/manager.py`
- Modify: `coral/hub/heartbeat.py`
- Modify: `coral/cli/heartbeat.py`
- Modify: `coral/task/validation.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_heartbeat.py`
- Modify: `tests/test_validation.py`

**Interfaces:**
- Produces: `AgentConfig.heartbeat_enabled: bool = True`；`write_agent_heartbeat(..., heartbeat_enabled: bool = True)`；`write_global_heartbeat(..., heartbeat_enabled: bool = True)`；validation Task metadata `{"coral_validation": True}`。

- [ ] **Step 1: 写 disabled round-trip、start/resume 和 validation metadata 失败测试**

```python
def test_heartbeat_disabled_preserves_empty_actions():
    config = CoralConfig.from_dict({
        "task": {"name": "t", "description": "d"},
        "agents": {"heartbeat_enabled": False, "heartbeat": []},
    })
    assert config.agents.heartbeat_enabled is False
    assert config.agents.heartbeat == []
    assert config.to_dict()["agents"]["heartbeat_enabled"] is False


def test_validation_marks_baseline_task(monkeypatch, task_dir):
    captured = {}
    async def grade(_path, tasks):
        captured.update(tasks[0].metadata)
        return ScoreBundle(scores={}, aggregated=1.0)
    monkeypatch.setattr("coral.task.validation.load_grader", lambda *_a, **_k: SimpleNamespace(grade=grade))
    result = run_validation(task_dir)
    assert result.successful
    assert captured == {"coral_validation": True}
```

- [ ] **Step 2: 运行目标测试并确认失败**

Run: `uv run --frozen --extra dev pytest tests/test_config.py tests/test_heartbeat.py tests/test_validation.py -q`

Expected: 新断言 FAIL，原测试保持通过。

- [ ] **Step 3: 实现配置预处理与 manager fast path**

`_preprocess()` 在 `heartbeat_enabled=false` 时丢弃 legacy cadence、保持显式空列表且不补默认 actions。`AgentManager.start_all()` 和 `_setup_agent()` 不 seed heartbeat 文件；`_get_heartbeat_runner()` 返回 `HeartbeatRunner([])`；monitor 不产生 heartbeat interrupt。

- [ ] **Step 4: 禁止 disabled run 的 CLI mutation**

`coral heartbeat show` 输出 `Heartbeat is disabled by agents.heartbeat_enabled=false.`；`set/remove/reset` 以 exit 1 返回同一原因。Hub writer 收到 `heartbeat_enabled=False` 时原样写空 actions，不恢复 protected reflect/consolidate。

- [ ] **Step 5: 给 validation baseline 加只读标记**

```python
task = Task(
    id=config.task.name,
    name=config.task.name,
    description=config.task.description,
    metadata={"coral_validation": True},
)
```

- [ ] **Step 6: 运行回归测试**

Run: `uv run --frozen --extra dev pytest tests/test_config.py tests/test_heartbeat.py tests/test_validation.py tests/test_manager_reliability.py -q`

Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add coral/config.py coral/agent/manager.py coral/hub/heartbeat.py coral/cli/heartbeat.py coral/task/validation.py tests
git commit -m "feat: allow tasks to disable heartbeat"
```

---

### Task 3: 建立受信任 control RPC、formal submit 与同 SHA infra retry

**Files:**
- Create: `coral/control/protocol.py`
- Create: `coral/control/client.py`
- Create: `coral/control/server.py`
- Create: `coral/control/loader.py`
- Modify: `coral/config.py`
- Modify: `coral/agent/manager.py`
- Modify: `coral/hooks/post_commit.py`
- Modify: `coral/cli/eval.py`
- Modify: `coral/cli/__init__.py`
- Modify: `coral/grader/daemon.py`
- Create: `tests/test_control_rpc.py`
- Create: `tests/test_retry_attempt.py`
- Modify: `tests/test_hooks.py`
- Modify: `tests/test_grader_daemon.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/pyproject.toml`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/provider.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/submission.py`

**Interfaces:**
- Produces: `ControlConfig(enabled=False, entrypoint="")`；newline-delimited JSON UDS request `{"version":1,"operation":str,"payload":dict}`；`ControlProvider.handle(operation, payload, context) -> dict`；`request_control(coral_dir, operation, payload) -> dict`；`retry_attempt(coral_dir, commit_hash, island_id) -> Attempt`。

- [ ] **Step 1: 写 transport、unknown operation、path injection 与 retry 失败测试**

```python
def test_retry_accepts_only_terminal_infra_attempt(coral_dir):
    attempt = make_attempt(status="crashed", metadata={
        "failure_class": "infra",
        "grader_runs": [{"run_id": "r1", "status": "crashed"}],
    })
    write_attempt(coral_dir, attempt)
    retried = retry_attempt(coral_dir, attempt.commit_hash, None)
    assert retried.status == "pending"
    assert retried.commit_hash == attempt.commit_hash
    assert retried.metadata["grader_runs"] == attempt.metadata["grader_runs"]


@pytest.mark.parametrize("payload", [
    {"path": "../../private"}, {"command": "sh -c id"}, {"operation": "write_file"}
])
def test_provider_rejects_untrusted_fields(provider, payload):
    with pytest.raises(SubmissionError):
        provider.handle("submit_attempt", payload, context())
```

- [ ] **Step 2: 运行测试并确认模块缺失**

Run: `uv run --frozen --extra dev pytest tests/test_control_rpc.py tests/test_retry_attempt.py tests/test_hooks.py tests/test_grader_daemon.py -q`

Expected: FAIL，缺少 control 与 retry 接口。

- [ ] **Step 3: 实现长度限制、peer ownership 和固定 operation 的 UDS server**

Server socket 位于 `.coral/public/control.sock`，mode `0600`；每条 request 最大 64 KiB；只接受 `submit_attempt`、`retry_attempt`、`create_node`、`refresh_evaluation`、`analyze`、`revise_analysis`。Server 由 manager 在 grader 之后、agents 之前启动，在 stop/resume 路径成对关闭/恢复。

- [ ] **Step 4: 把 formal pending write 移到 provider**

`submit_eval()` 仍执行 git add/commit/checkpoint，但 `control.enabled=true` 时不调用 `write_attempt()`，而是发送 commit、agent、title、parent、budget class、island 与 checkpoint hash。Provider 必须用 `git show <sha>:submission.yaml` 自行读取提交内容并验证：schema、HEAD、base、quality anchor、Root diff、同 Node 唯一 Research SHA；payload 不允许覆盖这些字段。

- [ ] **Step 5: 实现 append-only grader run history**

Daemon 每次 dispatch 先追加：

```python
{
    "run_id": uuid.uuid4().hex,
    "started_at": datetime.now(UTC).isoformat(),
    "finished_at": None,
    "status": "running",
    "failure_class": None,
    "contract_hash": attempt.metadata.get("contract_hash"),
    "log_dir": f"eval_logs/{attempt.commit_hash}/{run_id}",
}
```

finalize 时只替换同 `run_id` 项；retry 保持 commit SHA 和历史，只有 `failure_class=infra` 可重新设为 pending。Retry run 不增加 formal Attempt 数，也不增加 eval counter。

- [ ] **Step 6: 注册 `coral retry HASH` 并覆盖并发**

CLI 通过 control socket 请求，不直接改 JSON；provider 在 `.coral/private/control.lock` 的同一 flock 临界区内检查 terminal state、failure class、contract hash 和 pending 状态。两个并发 retry 只有一个成功。

- [ ] **Step 7: 运行 RPC/daemon/hook 回归测试**

Run: `uv run --frozen --extra dev pytest tests/test_control_rpc.py tests/test_retry_attempt.py tests/test_hooks.py tests/test_grader_daemon.py tests/test_cli_test_eval_alias.py -q`

Expected: PASS。

- [ ] **Step 8: Commit**

```bash
git add coral/control coral/config.py coral/agent/manager.py coral/hooks/post_commit.py coral/cli coral/grader/daemon.py tests tasks/greedylore_autoresearch/autoresearch
git commit -m "feat: add trusted attempt control channel"
```

---

### Task 4: 实现评测 primitives、动态 GPU 预留和进程清理

**Files:**
- Create: `tasks/greedylore_autoresearch/grader/pyproject.toml`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/contract.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/gpu.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/process.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/results.py`
- Create: `tasks/greedylore_autoresearch/tests/test_contract.py`
- Create: `tasks/greedylore_autoresearch/tests/test_gpu_allocator.py`
- Create: `tasks/greedylore_autoresearch/tests/test_process.py`
- Create: `tasks/greedylore_autoresearch/tests/test_results.py`

**Interfaces:**
- Produces: immutable `EvalContract` dataclasses；`GpuInventory.from_nvidia_smi()`；`reserve_idle_gpus(count, lock_dir, policy) -> GpuReservation` context manager；`run_process_group(spec: ProcessSpec) -> ProcessResult`；`quality_gate()`、`timing_mean()`、`geometric_mean_speedup()`。

- [ ] **Step 1: 写边界、GPU 选择和 cleanup 失败测试**

```python
def test_quality_gate_boundary():
    assert quality_gate(1.10, 1.0)
    assert not quality_gate(1.1000001, 1.0)


def test_two_repetition_mean_and_speedup():
    assert timing_mean([0.20, 0.22]) == pytest.approx(0.21)
    assert geometric_mean_speedup({"bf16": 2.0, "fp32": 0.5}) == pytest.approx(1.0)


def test_allocator_uses_any_four_idle_cards(fake_nvidia_smi, tmp_path):
    reservation = reserve_idle_gpus(4, tmp_path, policy())
    assert reservation.indices == (0, 2, 5, 7)
    assert reservation.cuda_visible_devices == "0,2,5,7"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_contract.py tasks/greedylore_autoresearch/tests/test_gpu_allocator.py tasks/greedylore_autoresearch/tests/test_process.py tasks/greedylore_autoresearch/tests/test_results.py -v`

Expected: FAIL，评测模块尚不存在。

- [ ] **Step 3: 实现 contract 解析与稳定 hash**

只接受已知 YAML keys；所有路径 resolve 后必须存在；以 canonical JSON（sorted keys、UTF-8、紧凑 separators）计算 SHA-256。Validation mode 可把 iteration 数降为 smoke 值，但必须输出独立 `budget_class=validation`，不能伪装正式 contract hash。

- [ ] **Step 4: 实现 GPU inventory 与原子多卡锁**

先查询 GPU index/UUID/name/memory/utilization 和 compute-app UUID/PID；按 index 排序过滤 idle；尝试对 UUID lock files 逐一 `flock(LOCK_EX|LOCK_NB)`，拿齐 4 张后再次查询确认仍空闲，失败则释放全部并返回 `InfraFailure("fewer than four idle GPUs")`。环境中记录 index、UUID、name；dense 与 candidate 只要求 name/count 相同，不要求 index 相同。

- [ ] **Step 5: 实现 process-group 生命周期**

`subprocess.Popen(..., start_new_session=True)`，stdout/stderr 直接写 immutable log files。正常、timeout 和异常均执行 `SIGTERM -> wait 15s -> SIGKILL -> wait`；结果记录 command、env allowlist、return code、started/finished、timeout、cleanup 信号和 stderr tail。

- [ ] **Step 6: 实现结果聚合与无效输入拒绝**

两次 repetition 必须均为 finite positive 且 contract/bucket layout/hardware 一致；否则 cell 为 invalid，不使用单次结果替代。`relative_spread = abs(a-b) / ((a+b)/2)` 只报告，不触发第三次运行。

- [ ] **Step 7: 运行 primitive 测试**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_contract.py tasks/greedylore_autoresearch/tests/test_gpu_allocator.py tasks/greedylore_autoresearch/tests/test_process.py tasks/greedylore_autoresearch/tests/test_results.py -v`

Expected: PASS。

- [ ] **Step 8: Commit**

```bash
git add tasks/greedylore_autoresearch/grader tasks/greedylore_autoresearch/tests
git commit -m "feat: add deterministic evaluation primitives"
```

---

### Task 5: 适配 GreedyLoRE quality/timing 入口并实现 packaged grader

**Files:**
- Modify: `tasks/greedylore_autoresearch/seed/c4/run_llama_pretraining.py`
- Modify: `tasks/greedylore_autoresearch/seed/c4/table_v_timing.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/quality.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/efficiency.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/grader.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/__init__.py`
- Create: `tasks/greedylore_autoresearch/tests/test_quality_commands.py`
- Create: `tasks/greedylore_autoresearch/tests/test_efficiency_commands.py`
- Create: `tasks/greedylore_autoresearch/tests/test_grader.py`

**Interfaces:**
- Produces: `build_quality_command(profile, candidate, output, gpus) -> ProcessSpec`；`build_timing_command(cell, repetition, candidate, output, gpus) -> ProcessSpec`；`Grader(TaskGrader).evaluate() -> ScoreBundle`。

- [ ] **Step 1: 写精确命令和 grader mode 的失败测试**

```python
def test_130m_quality_preserves_full_schedule_prefix(contract):
    cmd = build_quality_command(contract.quality["llama130m_muon"], PATH, OUT, GPUS).argv
    assert subsequence(cmd, ["--num_training_steps", "20000"])
    assert subsequence(cmd, ["--stop_after_steps", "2500"])
    assert subsequence(cmd, ["--optimizer", "muon", "--lr", "0.01"])


def test_efficiency_matrix_is_only_130m_adamw(contract):
    commands = [build_timing_command(cell, 0, PATH, OUT, GPUS).argv for cell in contract.efficiency.cells.values()]
    assert len(commands) == 2
    assert all(subsequence(cmd, ["--optimizer", "adamw", "--lr", "0.0025"]) for cmd in commands)
```

- [ ] **Step 2: 运行命令测试并确认失败**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_quality_commands.py tasks/greedylore_autoresearch/tests/test_efficiency_commands.py tasks/greedylore_autoresearch/tests/test_grader.py -v`

Expected: FAIL，runner/grader 尚不存在。

- [ ] **Step 3: 给训练入口增加显式离线 tokenizer**

`run_llama_pretraining.py` 新增 `--tokenizer_path`，训练和 validation 都调用 `AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, model_max_length=args.max_length)`；结果 JSON 增加 `status=completed`、dtype、optimizer、world size、contract hash。不得改变 compressor 或训练数学逻辑。

- [ ] **Step 4: 收紧 timing 结果 schema**

保留当前 normal async hook；正式 runner 总是传 `--record_bucket_layout` 且绝不传 strict blocking。结果必须包含 measured steps 400、slowest-rank mean、全部 rank elapsed、bucket layouts、peak memory、GPU identity、NCCL env、代码 hash 和 loss finite 检查。

- [ ] **Step 5: 实现 quality runner**

60M 命令固定 AdamW、8,393 updates、batch 128/card、activation checkpointing、BF16；130M 固定 Muon matrix/scalar LR 0.01/0.001、2,500/20,000 schedule、batch 128/card、activation checkpointing、BF16。两者 world size 4、seed 1243、data/tokenizer 固定路径、`WANDB_MODE=disabled`。

- [ ] **Step 6: 实现 efficiency runner**

运行顺序固定为 round 1 BF16→FP32、round 2 FP32→BF16；每次重新 torchrun；两个 cell 都是 130M AdamW、batch 1/card、SHM env、101+400、bucket 1024 MiB、compression start 100。

- [ ] **Step 7: 实现 Research/Engineering/validation 三种 grader 路径**

`Task.metadata["coral_validation"]` 为 true 时只运行 3-step CPU/parser 或短 GPU smoke。Formal `submission.yaml.mode=research` 运行两项 quality 加两项 efficiency；engineering 只运行两项 efficiency并要求有效 `quality_anchor`。Named scores 包含两项 loss ratio/pass、两项 timing mean/spread、两项 speedup、quality gate 和 efficiency score；metadata 写 `failure_class`、contract hash 与 evidence paths。

- [ ] **Step 8: 运行 unit grader tests 和 GreedyLoRE 既有 timing tests**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_quality_commands.py tasks/greedylore_autoresearch/tests/test_efficiency_commands.py tasks/greedylore_autoresearch/tests/test_grader.py -v`

Run: `/home/wyr/greedy_lore/.venv/bin/python -m pytest tasks/greedylore_autoresearch/seed/tests/test_table_v_timing.py -v`

Expected: PASS；不启动完整训练。

- [ ] **Step 9: Commit**

```bash
git add tasks/greedylore_autoresearch/seed/c4 tasks/greedylore_autoresearch/grader tasks/greedylore_autoresearch/tests
git commit -m "feat: add GreedyLoRE quality and timing grader"
```

---

### Task 6: 实现 Node、evaluation rebuild 与 Root bootstrap

**Files:**
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/atomic.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/nodes.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/evaluation.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/bootstrap.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/cli.py`
- Create: `tasks/greedylore_autoresearch/tests/test_node_rules.py`
- Create: `tasks/greedylore_autoresearch/tests/test_evaluation_index.py`
- Create: `tasks/greedylore_autoresearch/tests/test_root_bootstrap.py`

**Interfaces:**
- Produces: `create_node(request, evidence) -> NodeRecord`；`refresh_node_evaluation(node_id, attempts, log_root) -> Path`；`bootstrap_root(repo, notes_root, source_manifest) -> RootRecord`；CLI `autoresearch node create` 与 `autoresearch refresh-evaluation`。

- [ ] **Step 1: 写唯一 Research SHA、rebuild 和 Root diff 失败测试**

```python
def test_evaluation_rebuild_preserves_retry_and_engineering_rows(fixture_run):
    path = refresh_node_evaluation("N001", fixture_run.attempts, fixture_run.logs)
    first = path.read_text()
    path.unlink()
    second = refresh_node_evaluation("N001", fixture_run.attempts, fixture_run.logs).read_text()
    assert first == second
    assert "research-a" in second and "retry-run-2" in second
    assert "engineering-b" in second and "engineering-c" in second


def test_root_a0_changes_only_submission(repo, s0):
    assert changed_paths(repo, s0, "A0") == {"submission.yaml"}
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_node_rules.py tasks/greedylore_autoresearch/tests/test_evaluation_index.py tasks/greedylore_autoresearch/tests/test_root_bootstrap.py -v`

Expected: FAIL，Node/rebuild/bootstrap 接口尚不存在。

- [ ] **Step 3: 实现原子 artifact publisher**

所有发布限定在 `.coral/public/notes/tree.md` 或 `.coral/public/notes/nodes/N[0-9]{3}/`；resolve 后再次检查 `is_relative_to(notes_root)`；写临时文件、flush/fsync、`os.replace`、fsync parent。拒绝 symlink target、绝对路径、`..` 和未知文件名。

- [ ] **Step 4: 实现 Node validation 与 create**

Root 固定 `N000`、parent null、base `S0`；普通 child ID 单调分配，必须引用 parent 的 terminal Research Attempt 和明确 Analysis/Review version，且 review decision 只能 pass/warning。Child spec 保存 selection rationale、fixed base commit、hypothesis、delta、risk 与 falsification criteria。

- [ ] **Step 5: 实现 deterministic evaluation rebuild**

只从 Attempts、`grader_runs`、raw result JSON 和对应 commit 的 `submission.yaml` 生成。排序为 Research 固定 SHA、其 retry started time、Engineering attempt timestamp；多个不同 Research SHA 输出 consistency error，不覆盖；缺值写 `—`。

- [ ] **Step 6: 实现 Root bootstrap**

验证 run repo 的 seed commit S0、source manifest 与 pinned SHA；发布 N000 spec；从固定模板只新增根 `submission.yaml`；在 broker 接受 A0 前用 `git diff --name-only S0 A0` 强制 `{submission.yaml}`。

- [ ] **Step 7: 运行 Node 测试**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_node_rules.py tasks/greedylore_autoresearch/tests/test_evaluation_index.py tasks/greedylore_autoresearch/tests/test_root_bootstrap.py -v`

Expected: PASS。

- [ ] **Step 8: Commit**

```bash
git add tasks/greedylore_autoresearch/autoresearch tasks/greedylore_autoresearch/tests
git commit -m "feat: add node and evaluation artifact workflow"
```

---

### Task 7: 实现 fresh Analysis/Reviewer 状态机

**Files:**
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/analysis.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/validation.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/schemas/analysis.schema.json`
- Create: `tasks/greedylore_autoresearch/autoresearch/schemas/review.schema.json`
- Create: `tasks/greedylore_autoresearch/tests/test_analysis_cli.py`
- Create: `tasks/greedylore_autoresearch/tests/test_analysis_recovery.py`

**Interfaces:**
- Produces: `analyze(node_id, mode, attempt_sha, revise=False) -> AnalysisResult`；`run_fresh_codex(prompt, read_roots, output_schema, scratch) -> Path`；Analysis/Review validators。

- [ ] **Step 1: 写幂等、partial recovery 与 revise 失败测试**

```python
def test_review_failure_only_retries_missing_review(fake_codex, node):
    fake_codex.outputs = [VALID_ANALYSIS, RuntimeError("review crashed"), VALID_REVIEW]
    with pytest.raises(RuntimeError):
        analyze(node.id, "research", node.research_sha)
    result = analyze(node.id, "research", node.research_sha)
    assert fake_codex.analysis_calls == 1
    assert fake_codex.review_calls == 2
    assert result.analysis_version == 1


def test_revise_keeps_v1_and_publishes_v2(node):
    first = analyze(node.id, "engineering", node.engineering_sha)
    second = analyze(node.id, "engineering", node.engineering_sha, revise=True)
    assert first.analysis_path.exists() and first.review_path.exists()
    assert second.analysis_version == 2 and second.review_version == 2
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_analysis_cli.py tasks/greedylore_autoresearch/tests/test_analysis_recovery.py -v`

Expected: FAIL，analysis state machine 尚不存在。

- [ ] **Step 3: 实现 fresh Codex runner**

每次 Analyzer 与 Reviewer 分别执行新的：

```bash
codex exec --ephemeral --ignore-user-config --sandbox read-only \
  --skip-git-repo-check --cd SCRATCH \
  --output-schema SCHEMA --output-last-message OUTPUT -
```

Prompt 通过 stdin；scratch 是新建临时目录；仅把固定 candidate checkout、eval logs 和 references 作为只读 evidence 路径写进 prompt。不得使用 `resume` 或复用 thread/session id。

- [ ] **Step 4: 实现 Analysis 与 Review schema validation**

Analysis 必须匹配 node、mode、唯一 Research SHA/Engineering SHA、evaluation version 和 evidence paths；Review 必须匹配同一 Analysis version，decision 枚举 `pass|warning|fail`，并含 workflow checks、analysis checks、issues/actions。

- [ ] **Step 5: 实现幂等与恢复状态机**

普通调用在有效 Analysis+Review 均存在时直接返回；只有 Analysis 时仅运行 Reviewer；Review decision fail 仍算 terminal；`--revise` 分配下一版本并运行两次 fresh session。任何 publish 前先 refresh evaluation，失败时保留 scratch log 而不发布部分文件。

- [ ] **Step 6: 运行 Analysis 测试**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_analysis_cli.py tasks/greedylore_autoresearch/tests/test_analysis_recovery.py -v`

Expected: PASS，fake runner 证明 Analyzer/Reviewer session 分离。

- [ ] **Step 7: Commit**

```bash
git add tasks/greedylore_autoresearch/autoresearch tasks/greedylore_autoresearch/tests
git commit -m "feat: add fresh analysis and review workflow"
```

---

### Task 8: 部署 roles、prompts、references、templates 并完成 task 配置

**Files:**
- Create: `tasks/greedylore_autoresearch/roles/researcher.md`
- Create: `tasks/greedylore_autoresearch/roles/engineer.md`
- Create: `tasks/greedylore_autoresearch/prompts/research_analysis.md`
- Create: `tasks/greedylore_autoresearch/prompts/engineering_analysis.md`
- Create: `tasks/greedylore_autoresearch/prompts/reviewer.md`
- Create: `tasks/greedylore_autoresearch/references/algorithm_ref.md`
- Create: `tasks/greedylore_autoresearch/references/system_ref.md`
- Create: `tasks/greedylore_autoresearch/references/semantic_criteria.md`
- Create: `tasks/greedylore_autoresearch/templates/node_spec.md`
- Create: `tasks/greedylore_autoresearch/templates/evaluation.md`
- Create: `tasks/greedylore_autoresearch/templates/research_analysis.md`
- Create: `tasks/greedylore_autoresearch/templates/engineering_analysis.md`
- Create: `tasks/greedylore_autoresearch/templates/review.md`
- Create: `tasks/greedylore_autoresearch/templates/submission.yaml`
- Modify: `tasks/greedylore_autoresearch/task.yaml`
- Create: `tasks/greedylore_autoresearch/tests/test_handoff_assets.py`

**Interfaces:**
- Consumes: exact Appendix A/B/C sections from `/home/wyr/CORAL_AutoResearch_V1_Handoff_Refined.md`.
- Produces: two Codex assignments with different `role_file` values；private dense references；packaged grader/control setup；`agents.heartbeat_enabled=false`；`grader.parallel.max_workers=1`；SRT enabled。

- [ ] **Step 1: 写 asset hash/heading 与 task config 失败测试**

```python
def test_task_has_two_distinct_codex_roles(task_yaml):
    assignments = task_yaml["agents"]["assignments"]
    assert [a["runtime_options"]["role_file"] for a in assignments] == [
        "roles/researcher.md", "roles/engineer.md"
    ]
    assert task_yaml["agents"]["heartbeat_enabled"] is False
    assert task_yaml["grader"]["parallel"]["max_workers"] == 1
    assert "taskdata/dense_references" in task_yaml["grader"]["private"]
```

- [ ] **Step 2: 从 handoff 精确提取 Appendix assets**

Researcher/Engineer/Analyzer/Reviewer 对应 Appendix A.1–A.5；algorithm/system/semantic 对应 B.1–B.3；artifact templates 对应 C.1–C.6。只替换固定命令、路径和本 spec 已批准的两-cell evaluation contract，不删减 evidence/reviewer 边界。

- [ ] **Step 3: 完成 task.yaml**

配置 `workspace.repo_path` 指向 task seed；grader setup 安装 `./grader` 与 `./autoresearch`；control entrypoint 指向 `autoresearch.provider:Provider`；两个 Codex assignments 各 count 1；sandbox provider srt；network allowlist 只保留运行所需域名，正式 evaluator 本身 offline。

- [ ] **Step 4: 运行 asset/config 测试与 CORAL structural validation**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_handoff_assets.py tasks/greedylore_autoresearch/tests/test_contract_static.py -v`

Run: `uv run --frozen --extra dev coral validate tasks/greedylore_autoresearch --json`

Expected: unit tests PASS；validation 使用 `coral_validation=true` 的便宜路径，不能启动正式 8,393/2,500-step 训练。

- [ ] **Step 5: Commit**

```bash
git add tasks/greedylore_autoresearch
git commit -m "feat: deploy autoresearch roles and artifact contracts"
```

---

### Task 9: 收紧 SRT 写边界并完成 S1–S15 dummy acceptance

**Files:**
- Modify: `coral/sandbox/srt.py`
- Modify: `tests/test_sandbox.py`
- Create: `tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py`
- Create: `tasks/greedylore_autoresearch/tests/test_acceptance_s1_s15.py`
- Create: `tasks/greedylore_autoresearch/tests/fixtures/attempts/`
- Create: `tasks/greedylore_autoresearch/tests/fixtures/analyses/`
- Create: `tasks/greedylore_autoresearch/tests/fixtures/reviews/`

**Interfaces:**
- Produces: `_protected_write_paths(coral_dir, island_id) -> list[str]`；agent 能 connect control socket，但不能创建/覆盖 Attempt、eval logs、tree/nodes 或 private data。

- [ ] **Step 1: 写 settings 与真实 symlink/path traversal 攻击失败测试**

```python
def test_settings_deny_protected_shared_artifacts(tmp_path):
    settings = build_srt_settings(SandboxConfig(enabled=True), proxy_port=1, **_paths(tmp_path))
    denied = set(settings["filesystem"]["denyWrite"])
    assert str((tmp_path / ".coral/public/attempts").resolve()) in denied
    assert str((tmp_path / ".coral/public/eval_logs").resolve()) in denied
    assert str((tmp_path / ".coral/public/notes/tree.md").resolve()) in denied
    assert str((tmp_path / ".coral/public/notes/nodes").resolve()) in denied
```

- [ ] **Step 2: 运行 sandbox tests 并确认新断言失败**

Run: `uv run --frozen --extra dev pytest tests/test_sandbox.py tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py -v`

Expected: FAIL，现有 settings 仍允许写整个 coral_dir。

- [ ] **Step 3: 添加 protected denyWrite 且保留必要写入**

Single-island 与 multi-island 都 deny attempts、eval_logs、notes/tree.md、notes/nodes 和 private。保留 own worktree、repo `.git`、checkpoint notes/skills、runtime home 与 `/tmp`。control socket 只允许 connect；server 端重新验证每项 request，不能依赖 socket 权限作授权。

- [ ] **Step 4: 实现 fail-closed host preflight**

保持 `ensure_sandbox_supported()` 对 `srt --version`、bubblewrap、socat、rg 的检查；task start 缺任一依赖即在创建 agent 前失败，不退化为 unsandboxed。测试 mock 所有 dependency branches。

- [ ] **Step 5: 实现 S1–S15 dummy acceptance fixtures**

覆盖：唯一 Research SHA、infra retry、Analysis 幂等/partial/revise、child version refs、evaluation rebuild 多/零 engineering、Reviewer pass/warning/fail、GPU lock max_workers=1、Root diff、reference citations、selection rationale、protected path/symlink/absolute path 攻击。Dummy tests 不运行 GPU 长训练。

- [ ] **Step 6: 运行 framework 与 task acceptance**

Run: `uv run --frozen --extra dev pytest tests/test_sandbox.py tests/test_config.py tests/test_heartbeat.py tests/test_hooks.py tests/test_grader_daemon.py tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py tasks/greedylore_autoresearch/tests/test_acceptance_s1_s15.py -q`

Expected: PASS。

- [ ] **Step 7: 运行全量非 GPU 回归**

Run: `uv run --frozen --extra dev pytest -q`

Expected: PASS；GPU/依赖缺失导致的显式 skip 必须有原因，不允许 xfail 掩盖实现错误。

- [ ] **Step 8: Commit**

```bash
git add coral/sandbox/srt.py tests tasks/greedylore_autoresearch/tests
git commit -m "feat: enforce autoresearch sandbox boundaries"
```

---

### Task 10: 真实环境 smoke、dense calibration 与 Root N000

**Files:**
- Create after real runs: `tasks/greedylore_autoresearch/taskdata/dense_references/manifest.json`
- Create after real runs: `tasks/greedylore_autoresearch/taskdata/dense_references/quality.json`
- Create after real runs: `tasks/greedylore_autoresearch/taskdata/dense_references/efficiency.json`
- Modify after real runs: `tasks/greedylore_autoresearch/README.md`

**Interfaces:**
- Consumes: installed SRT/bubblewrap/socat、至少四张空闲且同型号 GPU、fixed C4/tokenizer、通过的 Tasks 1–9。
- Produces: verified dense references；run repo S0/A0；N000 Research Attempt、Analysis、Review；完整 limitation record。

- [ ] **Step 1: 检查主机依赖并 fail closed**

Run: `npx --no-install srt --version`

Run: `command -v bwrap`

Run: `command -v socat`

Run: `command -v rg`

Run: `command -v nvidia-smi`

Expected: 全部返回 0。缺失依赖时停止，不启动 CORAL run；安装系统/npm 依赖需要单独的用户授权。

- [ ] **Step 2: 运行动态 GPU 资源探测**

Run: `uv run --project tasks/greedylore_autoresearch/grader python -m greedylore_autoresearch_grader.gpu --count 4 --json`

Expected: 返回四个 idle GPU 的 index/UUID/name；没有固定 `3,4,5,6` 假设。

- [ ] **Step 3: 运行 4 GPU correctness smoke**

使用正式 data/tokenizer、Root compressor 和动态 GPU，分别运行 60M AdamW BF16 与 130M Muon BF16 各 3 updates；验证 finite loss、完整 world size、正常 process cleanup 和 result schema。

- [ ] **Step 4: 运行两个 efficiency cell 的短 smoke**

130M AdamW BF16/FP32 各执行 `warmup=3/measured=3`，两轮反序；验证 SHM NCCL 日志、normal async hook、bucket layout、slowest-rank wall time 和不同 repetition 的新 process group。

- [ ] **Step 5: Commit smoke 修正（仅在 smoke 暴露代码问题时）**

```bash
git add coral tasks/greedylore_autoresearch
git commit -m "fix: address four-GPU smoke findings"
```

如果 smoke 未产生代码或文档变更，不创建空 commit。

- [ ] **Step 6: 串行校准 dense references**

Run: `uv run --project tasks/greedylore_autoresearch/autoresearch autoresearch calibrate-dense --task tasks/greedylore_autoresearch`

Expected: 60M AdamW 完整 8,393、130M Muon 2,500/20,000，以及 130M AdamW BF16/FP32 每 cell 两次均完成；manifest 锁定 contract hash、hardware/software fingerprint、commands 与 raw log hashes。

- [ ] **Step 7: 验证并提交 dense references**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_contract.py tasks/greedylore_autoresearch/tests/test_grader.py -v`

```bash
git add tasks/greedylore_autoresearch/taskdata/dense_references tasks/greedylore_autoresearch/README.md
git commit -m "data: freeze dense evaluation references"
```

- [ ] **Step 8: Bootstrap Root 并运行唯一 formal Research Attempt**

启动 CORAL task，记录 S0；受控创建 N000 和仅含 `submission.yaml` 的 A0；运行两项 quality 与两项 efficiency；刷新 `evaluation.md`；运行 fresh Research Analysis 与 Reviewer。

- [ ] **Step 9: 验证 Root artifacts 与安全边界**

确认：S0→A0 只有 submission；N000 只有一个不同 Research SHA；所有 grader runs/log hashes 可重建 evaluation；Review 有 pass/warning/fail 明确结论；Agent 对 protected artifacts 的直接写入被拒绝，而 `coral eval`、retry、refresh、analyze 仍能工作。

- [ ] **Step 10: 提交真实环境记录**

```bash
git add tasks/greedylore_autoresearch/README.md
git commit -m "docs: record Root N000 validation"
```

README 只记录真实执行过的命令、结果、失败和限制；run-local Attempts/Analysis/Review 留在 CORAL run artifacts，不复制为伪造的静态结果。

---

## Final Verification

- [ ] Run: `git diff --check`
- [ ] Run: `uv run --frozen --extra dev ruff check coral tests tasks/greedylore_autoresearch/grader tasks/greedylore_autoresearch/autoresearch`
- [ ] Run: `uv run --frozen --extra dev pytest -q`
- [ ] Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests -q`
- [ ] Run: `uv run --frozen --extra dev coral validate tasks/greedylore_autoresearch --json`
- [ ] 检查 `git log --oneline 0123dfb..HEAD`，确认每个完成的大 Task 都有独立 commit，且没有把长训练或 push 隐藏在 commit hook 中。
- [ ] 检查最终工作区只包含明确记录的未提交 run artifacts；源 `/home/wyr/greedy_lore` 保持 clean 且 HEAD 未变化。
