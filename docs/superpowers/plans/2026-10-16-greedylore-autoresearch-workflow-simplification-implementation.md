# GreedyLoRE AutoResearch V1 Workflow Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use subagents only when explicitly requested and useful for a bounded independent check. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在锁定的 CORAL fork 中交付一个以 GreedyLoRE r32/EF14 为 Root、支持 Research/Engineering 分流、受控证据发布和固定 GPU 评测合同的 AutoResearch V1 task。

**Architecture:** 复用 CORAL 原生 Commit → Attempt → Grader 主链，只在现有 grader daemon 内增加固定用途的 trusted submit/retry/publish 入口，并收紧 SRT 写边界；不增加通用 RPC、第二套状态数据库或工作流引擎。GreedyLoRE task 拥有固定 evaluation contract、Analyzer-only Node workflow、按需 Subagent prompts 和低频 Global Memory；Memory 只在 Researcher start/resume/restart Hook 中异步检查三 Node 门槛。

**Tech Stack:** Python 3.11、CORAL dataclass/OmegaConf 配置、pytest、Unix domain socket、PyYAML、PyTorch 2.4 DDP/NCCL、Hugging Face Transformers/Datasets、Git worktree、SRT/bubblewrap。

**Spec:** `docs/superpowers/specs/2026-10-15-greedylore-autoresearch-coral-design.md`

## Global Constraints

- CORAL 基线固定为 `0123dfb939b35228cf2c1fde224cd0561e727408`；GreedyLoRE 来源固定为 `d2c38bc2062e06fdfd73185c508e7cc652933de6`。
- C4 只读路径固定为 `/home/wyr/greedy_lore/c4/c4_en/en`。
- tokenizer 固定为 `/home/wyr/.cache/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`，所有正式运行设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。
- 所有正式 workload 使用 4 GPU；不硬编码物理卡号，从空闲 GPU 中原子预留 4 张，不足时报告 `failure_class=infra`。正式 **efficiency** 与对应 dense reference 应优先使用相同 GPU UUID 集合；不能相同时，须证实互联拓扑、NUMA/NCCL 环境等效，否则结果不可比较，需在可比 GPU 集合上重测 dense reference。
- quality gate 对两个 workload 分别要求 `candidate_final_eval_loss <= 1.10 * dense_reference_final_eval_loss`，必须同时通过。
- efficiency 只有 `130m_adamw_bf16` 和 `130m_adamw_fp32`；每卡 batch 1、SHM、异步 DDP、101 warmup、400 measured、每 cell 两次。
- 正式 timing 不传 `--strict_blocking_communication`，只用被动 bucket layout 记录；Dense/candidate 都使用两次均值，**只有环境与 GPU 拓扑可比时**才计算 speedup，综合分数为两个 speedup 的几何平均。
- Agent 不能直接写 Attempts、eval logs、Node history、Analysis、Global Summary 或 private data；`coral eval`、`autoresearch retry` 和 task-local Artifact 命令通过固定用途的可信入口发布。
- 每个下列 Task 是一个“大 step”：其测试通过后立即创建一次 commit；commit 本身不触发 push、长训练或 calibration。
- 未实际执行的 GPU smoke、dense calibration、完整训练和 SRT 攻击测试不得标记为通过。
- **防御性与复杂度成比例：** V1 只为会直接影响正式结果可信性、权限边界、GPU 资源冲突和分布式进程清理的常见故障路径增加强校验。对尚未观察到、极少见或高度平台相关的异常，不预先构建通用恢复层、复杂兼容解析或额外状态机；先清晰失败并保留日志，只有真实失败或验收测试证明必要时再增加最小处理。

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

Contract 明确编码：4 GPU、seed 1243、seq 256；60M AdamW 8,393 updates；130M Muon `scheduler_steps=20000/stop_after_steps=2500`；quality compression start 1000；timing compression start 100；AdamW LR 0.0025；两个 timing cells、两次顺序 `BF16,FP32` 与 `FP32,BF16`；GPU idle 阈值 `compute_processes=0`、`memory_used_mib<=512`、`utilization_percent<=5`；efficiency 对照使用同 UUID 集合优先、不同集合必须验证 GPU 拓扑与 NCCL/NUMA 环境等效，不可比则拒绝 speedup。`max_infra_retries` 在受保护 contract 中固定，默认 2。

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

### Task 3: 建立最小可信 submit/retry 与同 SHA infra retry

**Files:**
- Create: `coral/grader/trusted_writes.py`
- Create: `coral/grader/trusted_worker.py`
- Modify: `coral/hooks/post_commit.py`
- Modify: `coral/grader/daemon.py`
- Modify: `coral/grader/task_grader.py`
- Modify: `coral/agent/manager.py`（只把已有 Agent 进程与 Worktree 绑定原子写入受保护登记文件，不引入新的运行管理机制）
- Create: `tests/test_trusted_writes.py`
- Create: `tests/test_retry_attempt.py`
- Modify: `tests/test_hooks.py`
- Modify: `tests/test_grader_daemon.py`
- Modify: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/contract.py`
- Modify: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/process.py`
- Modify: `tasks/greedylore_autoresearch/seed/source_manifest.json`
- Modify: `tasks/greedylore_autoresearch/tests/conftest.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_process.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_seed_manifest.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/pyproject.toml`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/client.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/submission.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/artifacts.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/cli.py`
- Create: `tasks/greedylore_autoresearch/tests/test_submission.py`
- Modify: `tasks/greedylore_autoresearch/taskdata/eval_contract.yaml`
- Modify: `tasks/greedylore_autoresearch/task.yaml`

**Interfaces:**
- Consumes: CORAL 原生 `submit_eval()` 的 Git commit/parent 语义、`checkpoint()` API 和 grader daemon 生命周期；`grader.args.trusted_submission_validator="autoresearch.submission:validate_submission"` 与 `grader.args.trusted_artifact_validator="autoresearch.artifacts:validate_artifact"` 由 trusted grader venv 调用。
- Produces: `.coral/public/trusted.sock`；固定 request `{"version":1,"operation":"submit_attempt|retry_attempt|publish_artifact","payload":{...}}`；`request_trusted(coral_dir, operation, payload) -> dict`；`invoke_task_validator(entrypoint, payload) -> dict`（只使用 grader-private Python）；`validate_submission(commit_hash, context) -> dict[str, object]`；`validate_artifact(kind, content, context) -> dict[str, object]`；CLI `autoresearch retry HASH`；仅供可信宿主进程直接调用、且不暴露为 socket operation 的 `submit_root_attempt(worktree, commit_hash)`；`.coral/private/trusted_agents.json` 中的可信 Agent 根 PID/start time/worktree/island 登记；`peer_pid → /proc 父链 → registered_agent/worktree` 来源上下文；受保护的 retry 上限与实际 Grader 执行次数记录。不存在 `ControlConfig`、动态 `ControlProvider` 或任意 operation registry。

- [x] **Step 1: 写固定 operation、submission ownership 与 retry 失败测试**

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
    assert retried.metadata["infra_retry_count"] == 1


def test_submit_reads_committed_submission(trusted_server, committed_repo):
    result = request_trusted(trusted_server.coral_dir, "submit_attempt", {
        "commit_hash": committed_repo.research_sha,
        "agent_id": "researcher-0",
        "title": "research N001",
        "parent_hash": committed_repo.parent_sha,
        "budget_class": "real",
        "island_id": None,
    })
    assert result["metadata"]["node_id"] == "N001"
    assert result["metadata"]["mode"] == "research"


def test_same_uid_cannot_spoof_other_registered_agent(trusted_server, two_agent_sessions):
    # 同 UID、不同 Agent 进程；请求里伪造 agent_id 或使用其他 worktree 的 SHA 均不能授权提交。
    result = two_agent_sessions.engineer.submit(
        agent_id="researcher-0", commit_hash=two_agent_sessions.researcher.head
    )
    assert not result.ok


def test_retry_budget_and_finalize_are_serialized(trusted_server, infra_attempt):
    # 达到受保护 retry 上限后拒绝；重试和 daemon finalize 不可丢失 grader_runs。
    infra_attempt.exhaust_retry_budget()
    assert not trusted_server.retry(infra_attempt.commit_hash).ok
    assert infra_attempt.all_run_records_preserved()


def test_research_uniqueness_check_and_write_share_one_lock(
    trusted_server, two_clients_same_node
):
    results = two_clients_same_node.submit_concurrently()
    assert sum(result.ok for result in results) == 1
    assert len(read_research_attempts(trusted_server.coral_dir, "N001")) == 1
```

- [x] **Step 2: 运行测试并确认最小入口缺失**

Run: `uv run --frozen --extra dev pytest tests/test_trusted_writes.py tests/test_retry_attempt.py tests/test_hooks.py tests/test_grader_daemon.py tasks/greedylore_autoresearch/tests/test_submission.py -q`

Expected: FAIL，缺少 trusted-writes、validator 与 retry 接口。

- [x] **Step 3: 在现有 grader daemon 生命周期内实现固定用途 UDS**

`run_daemon()` 在 pending scan loop 前启动 `.coral/public/trusted.sock`，并在同一个 `stop_event` 路径关闭；不增加第二个受管进程。socket mode `0600`，单 request 最大 64 KiB，只识别 `submit_attempt`、`retry_attempt`、`publish_artifact`。拒绝任意路径、shell command、嵌套 operation 和未知字段。

**提交者绑定：** 不能信任客户端传入的 `agent_id`、`island_id`、`parent_hash` 或 UID 作为 Agent 身份证明。manager 复用每次启动/重启后的 `AgentHandle`，把根 PID、`/proc/<pid>/stat` start time、worktree realpath 和 island 原子写入 `.coral/private/trusted_agents.json`；停止时清空。服务端使用内核提供的 `SO_PEERCRED` PID，沿当前 `/proc` 父链归属到一个仍存活且 start time 匹配的登记根进程；不要要求调用 `coral eval` 的子进程 PID 等于 Agent 根 PID，也不只依赖可被子进程重建的 process group。映射失效或无法验证时 **Agent 发起的正式提交** fail-closed，不因同 UID 放行；若 SRT 或其他包装层使 `SO_PEERCRED` 指向代理进程，只有该代理也能通过同一父链规则归属时才放行。**Root Bootstrap 的受信任宿主入口是独立例外**，由 manager/可信进程固定来源 S0、目标 N000 和提交内容，不允许 Agent 伪装该入口。验证者从服务端可信登记信息推导身份和工作区；不增加通用身份服务。

- [x] **Step 4: 把 formal pending write 移到 trusted submit**

`submit_eval()` 保留 Git add/commit、parent lookup 和 wait/poll，但 Agent 进程不再调用 `checkpoint()` 或 `write_attempt()`。handler 对 Agent 请求使用已验证的 Agent→Worktree 绑定，检查提交 SHA 是该 Worktree 当前提交且 parent/provenance 一致；对 Root Bootstrap 仅接受可信宿主进程直接调用的 `submit_root_attempt()`，固定校验 S0→A0/N000 且不暴露为 socket operation，也不复用任意 Agent payload。客户端的身份与 parent 字段仅作声明，不授予权限。在 `.coral/private/trusted_writes.lock` 内执行 shared checkpoint、解析 parent checkpoint provenance，并用 `git show <sha>:submission.yaml` 读取 committed 内容，调用 grader venv 中固定的 `validate_submission`，再原子执行唯一 Research reservation 与 pending Attempt 写入。validator 校验 schema、HEAD、base、quality anchor、Root diff 和 Node 引用；`mode/node_id/base_commit/quality_anchor/contract_hash` 只能来自 committed submission 与 trusted config。

- [x] **Step 5: 实现最小 append-only grader run history 与 failure 分类边界**

```python
{
    "run_id": uuid.uuid4().hex,
    "started_at": datetime.now(UTC).isoformat(),
    "finished_at": None,
    "status": "running",
    "failure_class": None,
    "contract_hash": attempt.metadata.get("contract_hash"),
    "log_dir": f"eval_logs/{attempt.commit_hash}/{run_id}",
    "dispatcher_pid": os.getpid(),
    "dispatcher_start_time": process_start_time(os.getpid()),
    "resource_usage": None,
    "gpu_uuids": None,
}
```

finalize 只更新同 `run_id` 项；**dispatch、finalize、retry 对 Attempt metadata 的读改写使用同一锁或等效原子事务**，避免并发覆盖 `grader_runs`。只有 trusted grader 明确写入的 `failure_class=infra` 可 retry；candidate crash、算法 OOM、non-finite、deadlock、invalid submission 和 quality failure 都不可 retry。Retry 保持 SHA 和历史，不增加 formal Research Attempt 数；可保持 CORAL 原有 `eval counter` 语义，但 **每一次实际 Grader 调用必须单独计入运行次数、时间和 GPU 资源消耗**。增加从受保护 contract 读取的 `max_infra_retries`（V1 默认 2），达到上限即停止自动/手动重试；`grader_runs` 只是 Attempt metadata，不是独立数据库。

- [x] **Step 6: 实现 `autoresearch retry HASH` 与受限 Artifact 发布**

`autoresearch retry` 只请求 trusted socket；handler 在同一 flock 临界区检查 terminal state、infra 分类、contract hash、来源 Agent/Worktree、剩余 retry 预算以及 pending/running 状态，两个并发 retry 只有一个成功。`publish_artifact` 只接受 `kind/node_id/mode/version/content`，不接受路径；task validator 返回 canonical target，core 再限定到 `notes/tree.md` 或 `notes/nodes/N[0-9]{3}/`，校验后用临时文件、`fsync`、`os.replace` 发布。Task 6/7 只在写正式 Artifact 时使用它。

- [x] **Step 7: 运行 trusted submit/daemon/hook 回归测试**

Run: `uv run --frozen --extra dev pytest tests/test_trusted_writes.py tests/test_retry_attempt.py tests/test_hooks.py tests/test_grader_daemon.py tests/test_cli_test_eval_alias.py tasks/greedylore_autoresearch/tests/test_submission.py -q`

Expected: PASS；额外覆盖同 UID 伪造身份/跨 Worktree SHA 提交被拒、retry/finalize 并发不丢记录、retry 上限，以及长 Grader 运行时 socket 仍可响应、daemon stop/resume 后入口可用。

- [x] **Step 8: Commit**

```bash
git add coral/grader/trusted_writes.py coral/grader/trusted_worker.py coral/grader/daemon.py coral/grader/task_grader.py coral/hooks/post_commit.py coral/agent/manager.py tests tasks/greedylore_autoresearch/autoresearch tasks/greedylore_autoresearch/tests/test_submission.py tasks/greedylore_autoresearch/taskdata/eval_contract.yaml
git commit -m "feat: add minimal trusted attempt submission"
```

---

### Task 4: 实现评测 primitives、动态 GPU 预留和进程清理

**Files:**
- Create: `tasks/greedylore_autoresearch/grader/pyproject.toml`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/contract.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/gpu.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/process.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/results.py`
- Create: `tasks/greedylore_autoresearch/tests/conftest.py`
- Create: `tasks/greedylore_autoresearch/tests/test_contract.py`
- Create: `tasks/greedylore_autoresearch/tests/test_gpu_allocator.py`
- Create: `tasks/greedylore_autoresearch/tests/test_gpu_comparability.py`
- Create: `tasks/greedylore_autoresearch/tests/test_process.py`
- Create: `tasks/greedylore_autoresearch/tests/test_results.py`

**Interfaces:**
- Produces: immutable `EvalContract` dataclasses；`GpuInventory.from_nvidia_smi()`；`reserve_idle_gpus(count, lock_dir, policy) -> GpuReservation` context manager；`run_process_group(spec: ProcessSpec) -> ProcessResult`；`quality_gate()`、`timing_mean()`、`geometric_mean_speedup()`；`gpu_topology_fingerprint(gpu_uuids, nccl_env)` 与 `validate_efficiency_comparability(dense_env, candidate_env)`。

- [x] **Step 1: 写边界、GPU 选择和 cleanup 失败测试**

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


def test_efficiency_rejects_noncomparable_topology(dense_env, candidate_env):
    candidate_env.gpu_uuids = ("gpu-a", "gpu-c", "gpu-d", "gpu-f")
    candidate_env.topology_fingerprint = "different-links"
    assert not validate_efficiency_comparability(dense_env, candidate_env).valid
```

- [x] **Step 2: 运行测试并确认失败**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_contract.py tasks/greedylore_autoresearch/tests/test_gpu_allocator.py tasks/greedylore_autoresearch/tests/test_gpu_comparability.py tasks/greedylore_autoresearch/tests/test_process.py tasks/greedylore_autoresearch/tests/test_results.py -v`

Expected: FAIL，评测模块尚不存在。

- [x] **Step 3: 实现 contract 解析与稳定 hash**

只接受已知 YAML keys；所有路径 resolve 后必须存在；以 canonical JSON（sorted keys、UTF-8、紧凑 separators）计算 SHA-256。Validation mode 可把 iteration 数降为 smoke 值，但必须输出独立 `budget_class=validation`，不能伪装正式 contract hash。

- [x] **Step 4: 实现 GPU inventory 与原子多卡锁**

先查询 GPU index/UUID/name/memory/utilization 和 compute-app UUID/PID；按 index 排序过滤 idle；尝试对 UUID lock files 逐一 `flock(LOCK_EX|LOCK_NB)`，拿齐 4 张后再次查询确认仍空闲，失败则释放全部并返回 `InfraFailure("fewer than four idle GPUs")`。环境中记录 index、UUID、name、选定四卡的互联拓扑指纹（如 `nvidia-smi topo -m` 中的对应关系）、NUMA 信息及 NCCL 相关配置。选择任意空闲四卡仍允许，但**正式 efficiency 与对应 dense reference 优先使用完全相同的 GPU UUID 集合**；若不同，必须有可验证的相同 GPU 类型、互联拓扑与 NUMA/NCCL 环境等效性，否则禁止计算可比较的 speedup，并在可比卡组重新测量 dense。

- [x] **Step 5: 实现 process-group 生命周期**

`subprocess.Popen(..., start_new_session=True)`，stdout/stderr 直接写 immutable log files。正常、timeout 和异常均执行 `SIGTERM -> wait 15s -> SIGKILL -> wait`；结果记录 command、env allowlist、return code、started/finished、timeout、cleanup 信号和 stderr tail。

- [x] **Step 6: 实现结果聚合与无效输入拒绝**

两次 repetition 必须均为 finite positive 且 contract/bucket layout/hardware 一致；否则 cell 为 invalid，不使用单次结果替代。speedup 计算还须比较 dense 与 candidate 的 GPU UUID 集合或经验证的 topology fingerprint、NUMA 与 NCCL 配置，不可比时标记 `incomparable_hardware`，不发布有效 speedup。`relative_spread = abs(a-b) / ((a+b)/2)` 只报告，不触发第三次运行。

- [x] **Step 7: 运行 primitive 测试**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_contract.py tasks/greedylore_autoresearch/tests/test_gpu_allocator.py tasks/greedylore_autoresearch/tests/test_gpu_comparability.py tasks/greedylore_autoresearch/tests/test_process.py tasks/greedylore_autoresearch/tests/test_results.py -v`

Expected: PASS。

- [x] **Step 8: Commit**

```bash
git add tasks/greedylore_autoresearch/grader tasks/greedylore_autoresearch/tests
git commit -m "feat: add deterministic evaluation primitives"
```

---

### Task 5: 适配 GreedyLoRE quality/timing 入口并实现 packaged grader

**Files:**
- Modify: `tasks/greedylore_autoresearch/seed/c4/run_llama_pretraining.py`
- Modify: `tasks/greedylore_autoresearch/seed/c4/table_v_timing.py`
- Modify: `tasks/greedylore_autoresearch/seed/tests/test_c4_model_setup.py`
- Modify: `tasks/greedylore_autoresearch/seed/tests/test_table_v_timing.py`
- Modify: `coral/grader/daemon.py`
- Modify: `tests/test_grader_daemon.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/quality.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/efficiency.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/grader.py`
- Create: `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/__init__.py`
- Create: `tasks/greedylore_autoresearch/tests/test_quality_commands.py`
- Create: `tasks/greedylore_autoresearch/tests/test_efficiency_commands.py`
- Create: `tasks/greedylore_autoresearch/tests/test_grader.py`

**Interfaces:**
- Produces: `build_quality_command(profile, candidate, output, gpus) -> ProcessSpec`；`build_timing_command(cell, repetition, candidate, output, gpus) -> ProcessSpec`；`Grader(TaskGrader).evaluate() -> ScoreBundle`。

- [x] **Step 1: 写精确命令和 grader mode 的失败测试**

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

- [x] **Step 2: 运行命令测试并确认失败**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_quality_commands.py tasks/greedylore_autoresearch/tests/test_efficiency_commands.py tasks/greedylore_autoresearch/tests/test_grader.py -v`

Expected: FAIL，runner/grader 尚不存在。

- [x] **Step 3: 给训练入口增加显式离线 tokenizer**

`run_llama_pretraining.py` 新增 `--tokenizer_path`，训练和 validation 都调用 `AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, model_max_length=args.max_length)`；结果 JSON 增加 `status=completed`、dtype、optimizer、world size、contract hash。不得改变 compressor 或训练数学逻辑。

- [x] **Step 4: 收紧 timing 结果 schema**

保留当前 normal async hook；正式 runner 总是传 `--record_bucket_layout` 且绝不传 strict blocking。结果必须包含 measured steps 400、slowest-rank mean、全部 rank elapsed、bucket layouts、peak memory、GPU UUID 与 topo fingerprint、NUMA/NCCL env、代码 hash 和 loss finite 检查。

- [x] **Step 5: 实现 quality runner**

60M 命令固定 AdamW、8,393 updates、batch 128/card、activation checkpointing、BF16；130M 固定 Muon matrix/scalar LR 0.01/0.001、2,500/20,000 schedule、batch 128/card、activation checkpointing、BF16。两者 world size 4、seed 1243、data/tokenizer 固定路径、`WANDB_MODE=disabled`。

- [x] **Step 6: 实现 efficiency runner**

运行顺序固定为 round 1 BF16→FP32、round 2 FP32→BF16；每次重新 torchrun；两个 cell 都是 130M AdamW、batch 1/card、SHM env、101+400、bucket 1024 MiB、compression start 100。

- [x] **Step 7: 实现 Research/Engineering/validation 三种 grader 路径**

`Task.metadata["coral_validation"]` 为 true 时只运行 3-step CPU/parser 或短 GPU smoke。Formal `submission.yaml.mode=research` 运行两项 quality 加两项 efficiency；engineering 只运行两项 efficiency并要求有效 `quality_anchor`。Named scores 包含两项 loss ratio/pass、两项 timing mean/spread、两项 speedup、quality gate 和 efficiency score；metadata 写 `failure_class`、contract hash、GPU topology/UUID comparability verdict 与 evidence paths；efficiency 分数只有与 dense reference 确认可比时才有效。

- [x] **Step 8: 运行 unit grader tests 和 GreedyLoRE 既有 timing tests**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_quality_commands.py tasks/greedylore_autoresearch/tests/test_efficiency_commands.py tasks/greedylore_autoresearch/tests/test_grader.py -v`

Run from `tasks/greedylore_autoresearch/seed`: `PYTHONPATH=. /home/wyr/greedy_lore/.venv/bin/python -m unittest tests.test_c4_model_setup tests.test_table_v_timing`

Expected: PASS；不启动完整训练。

- [x] **Step 9: Commit**

```bash
git add coral/grader/daemon.py tests/test_grader_daemon.py docs/superpowers/plans/2026-10-15-greedylore-autoresearch-coral-implementation.md tasks/greedylore_autoresearch/seed tasks/greedylore_autoresearch/grader tasks/greedylore_autoresearch/tests
git commit -m "feat: add GreedyLoRE quality and timing grader"
```

---

### Task 6: 收敛 Analyzer-only Node workflow 与角色资产

**Files:**
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/analysis.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/validation.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/nodes.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/cli.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/submission.py`
- Delete: `tasks/greedylore_autoresearch/autoresearch/schemas/review.schema.json`
- Modify: `tasks/greedylore_autoresearch/templates/node_spec.md`
- Delete: `tasks/greedylore_autoresearch/templates/review.md`
- Modify: `tasks/greedylore_autoresearch/roles/workflow_rules.md`
- Modify: `tasks/greedylore_autoresearch/roles/researcher.md`
- Modify: `tasks/greedylore_autoresearch/roles/engineer.md`
- Modify: `tasks/greedylore_autoresearch/prompts/research_analysis.md`
- Modify: `tasks/greedylore_autoresearch/prompts/engineering_analysis.md`
- Delete: `tasks/greedylore_autoresearch/prompts/reviewer.md`
- Modify: `tasks/greedylore_autoresearch/tests/test_analysis_cli.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_analysis_recovery.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_node_rules.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_submission.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_root_bootstrap.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_handoff_assets.py`

**Interfaces:**
- Consumes: existing `resolve_evidence(node_id, mode) -> EvidenceBundle`,
  `run_fresh_codex(...) -> Path`, deterministic evaluation rebuild and Task 3
  `publish_artifact`.
- Produces: `AnalysisResult(evidence, analysis_version, analysis_path)`;
  `analyze(node_id, mode, revise=False) -> AnalysisResult`; Child spec with only
  fixed Analysis refs; Research Analysis publication as the sole Child prerequisite.
  There is no Review schema, session, file, decision, recovery stage, or approval state.

- [ ] **Step 1: Rewrite tests for the Analyzer-only contract**

~~~python
def test_analyze_runs_one_session_and_publishes_no_review(context):
    runner = FakeCodex(context, ["analysis"])
    result = analyze(
        "N001",
        "research",
        context=context,
        runner=runner,
        publisher=FakePublisher(context),
    )
    assert runner.analysis_calls == 1
    assert result.analysis_path.name == "research_analysis_v1.md"
    assert not list(result.analysis_path.parent.glob("*_review_v*.md"))


def test_completed_analysis_is_idempotent(context):
    runner = FakeCodex(context, ["analysis"])
    first = analyze(
        "N001", "research", context=context, runner=runner,
        publisher=FakePublisher(context),
    )
    second = analyze(
        "N001", "research", context=context, runner=runner,
        publisher=FakePublisher(context),
    )
    assert first.analysis_version == second.analysis_version == 1
    assert runner.analysis_calls == 1


def test_child_requires_only_fixed_parent_analysis(tmp_path):
    evidence = parent_evidence(tmp_path, analysis_version=2)
    spec = yaml.safe_load(create_node(child_request(), evidence).read_text())
    assert spec["analysis_refs"]["parent_research"] == (
        "nodes/N001/research_analysis_v2.md"
    )
    assert "review_refs" not in spec
~~~

Delete tests whose only behavior is missing-review recovery, decisions, or paired
Analysis/Review versions. Preserve tests for Research-only evidence, zero/multiple
Engineering Attempts, immutable evaluation snapshots, fixed-SHA code snapshots,
`--revise`, and evidence-content validation.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

~~~bash
uv run --frozen --extra dev pytest \
  tasks/greedylore_autoresearch/tests/test_analysis_cli.py \
  tasks/greedylore_autoresearch/tests/test_analysis_recovery.py \
  tasks/greedylore_autoresearch/tests/test_node_rules.py \
  tasks/greedylore_autoresearch/tests/test_submission.py \
  tasks/greedylore_autoresearch/tests/test_root_bootstrap.py \
  tasks/greedylore_autoresearch/tests/test_handoff_assets.py -q
~~~

Expected: FAIL because the implementation still returns Review fields, starts a
second session, and requires a Parent Review.

- [ ] **Step 3: Reduce Analysis to one stage**

`AnalysisResult` contains only `evidence`, `analysis_version`, and
`analysis_path`. Versions come only from `<mode>_analysis_vN.md`: the default
returns the latest valid version, no version creates v1, and `--revise` creates
`max(existing)+1`.

The implementation order is refresh evaluation → resolve evidence → validate and
return an existing Analysis, or run one fresh Analyzer → `seal_document` →
`validate_analysis` → trusted publication. Delete the reviewer prompt branch,
`validate_review()`, review schema lookup, second scratch/session and decision
output. Do not change EvidenceBundle, code snapshots, or evidence manifests.

`autoresearch analyze` prints only:

~~~json
{
  "analysis_version": 2,
  "analysis_path": "/notes/nodes/N001/research_analysis_v2.md"
}
~~~

- [ ] **Step 4: Remove approval references from Node and submission rules**

`create_node()` verifies only that the selected Parent Research Analysis path
exists and matches the Parent Node. Specs retain
`analysis_refs.parent_research` and optional `parent_engineering`, and contain no
`review_refs`. The Node CLI no longer constructs a Review path. Update submission
validation, Root bootstrap, and templates to match. Existing obsolete files in an
old run may remain readable, but no new workflow reads or creates them.

- [ ] **Step 5: Update Producer and Analyzer prompts**

Researcher reads `research_summary.md` when present, Parent spec/evaluation, and
the fixed Parent Research Analysis; it may extend a Child immediately after a
valid Research Analysis. Engineer reads `engineering_summary.md` when present;
Engineering Analysis remains optional and never blocks Researcher. Both Analyzer
prompts remove approval-stage language.

Add this rule to Researcher, Engineer, Research Analyzer, and Engineering
Analyzer:

~~~text
遇到实质性不确定性时，可以自主调用 Subagent 进行独立检查或辅助分析。
优先自行解决简单问题，不要为常规代码修改、实验结果或 Analysis 强制调用
Subagent。Subagent 必须有明确、有限的任务与输出，并继承当前 Sandbox、
Evaluation 和 Artifact 发布边界。
~~~

`workflow_rules.md` exposes only the user-facing commands and Node/Artifact
rules. It does not explain UDS, locks, grader run history, or SRT internals.
Do not modify the three Reference Toolkit files.

- [ ] **Step 6: Delete obsolete assets and verify GREEN**

Delete the review schema, prompt, and template. Run the Step 2 command.

Expected: PASS. Asset tests assert that obsolete files do not exist, no Analysis
command creates them, Child creation has no approval state, and all four target
prompts contain optional—not mandatory—Subagent guidance.

- [ ] **Step 7: Commit**

~~~bash
git add -A tasks/greedylore_autoresearch/autoresearch \
  tasks/greedylore_autoresearch/roles \
  tasks/greedylore_autoresearch/prompts \
  tasks/greedylore_autoresearch/templates \
  tasks/greedylore_autoresearch/tests
git commit -m "feat: simplify autoresearch to analyzer-only workflow"
~~~

---

### Task 7: 添加启动触发的 Global Memory 及完整受控边界

**Files:**
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/memory.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/schemas/memory.schema.json`
- Create: `tasks/greedylore_autoresearch/prompts/global_memory_manager.md`
- Create: `tasks/greedylore_autoresearch/tests/test_memory.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/artifacts.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/cli.py`
- Modify: `tasks/greedylore_autoresearch/task.yaml`
- Modify: `tasks/greedylore_autoresearch/tests/test_analysis_cli.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_handoff_assets.py`
- Create: `coral/agent/startup_hook.py`
- Modify: `coral/agent/builtin/codex.py`
- Create: `tests/test_codex_startup_hook.py`
- Modify: `coral/grader/trusted_writes.py`
- Modify: `tests/test_trusted_writes.py`
- Modify: `coral/sandbox/srt.py`
- Modify: `tests/test_sandbox.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py`

**Interfaces:**
- Consumes: Task 3 fixed `publish_artifact`; Task 6 published Analysis layout;
  Codex `runtime.start()` shared by start/resume/automatic restart.
- Produces:
  - `list_analysis_refs(notes_root: Path, mode: str | None = None) -> tuple[str, ...]`
  - `read_processed_analyses(summary: Path) -> frozenset[str]`
  - `memory_due(notes_root: Path, threshold: int = 3) -> bool`
  - `validate_summary(content: str, notes_root: Path) -> frozenset[str]`
  - `consolidate_memory(*, if_due: bool, context=None, runner=..., publisher=None) -> MemoryResult`
  - CLI `autoresearch consolidate-memory [--if-due] [--async]`
  - Codex reserved option `runtime_options.startup_hook: list[str]`
  - Artifact kinds `research_summary` and `engineering_summary`.

There is no timer, heartbeat action, Analysis publication hook, EvidenceBundle/hash,
or persistent scheduling service.

- [ ] **Step 1: Write failing gate and Summary validation tests**

~~~python
def test_gate_counts_unprocessed_research_nodes_only(notes_root):
    publish_analysis(notes_root, "N001", "research", 1)
    publish_analysis(notes_root, "N001", "research", 2)
    publish_analysis(notes_root, "N002", "research", 1)
    publish_analysis(notes_root, "N003", "engineering", 1)
    assert memory_due(notes_root) is False

    publish_analysis(notes_root, "N003", "research", 1)
    assert memory_due(notes_root) is True


def test_partial_summary_update_does_not_suppress_retry(notes_root):
    write_summary(
        notes_root / "research_summary.md",
        ["nodes/N001/research_analysis_v1.md"],
    )
    write_summary(notes_root / "engineering_summary.md", [])
    assert memory_due(notes_root, threshold=1) is True


def test_summary_validator_only_checks_format_and_existing_refs(notes_root):
    publish_analysis(notes_root, "N001", "research", 1)
    content = """---
processed_analyses:
  - nodes/N001/research_analysis_v1.md
---
# Research Summary

A bounded finding.
"""
    assert validate_summary(content, notes_root) == frozenset(
        {"nodes/N001/research_analysis_v1.md"}
    )
~~~

Add rejection cases for invalid frontmatter, empty body, non-canonical paths, and
missing referenced files. Python must not interpret Summary semantics.

- [ ] **Step 2: Write failing Manager, async, and idempotency tests**

~~~python
def test_manager_publishes_two_fixed_summaries(memory_context):
    runner = FakeMemoryRunner(valid_memory_output(memory_context.analysis_refs))
    publisher = RecordingPublisher()
    result = consolidate_memory(
        if_due=True,
        context=memory_context,
        runner=runner,
        publisher=publisher,
    )
    assert runner.calls == 1
    assert [call.kind for call in publisher.calls] == [
        "research_summary",
        "engineering_summary",
    ]
    assert result.status == "published"


def test_processed_set_skips_manager(memory_context):
    write_both_summaries(memory_context.notes_root, memory_context.analysis_refs)
    runner = FakeMemoryRunner()
    result = consolidate_memory(
        if_due=True,
        context=memory_context,
        runner=runner,
        publisher=RecordingPublisher(),
    )
    assert result.status == "not_due"
    assert runner.calls == 0


def test_analyze_never_launches_memory(monkeypatch, analysis_context):
    monkeypatch.setattr("autoresearch.memory.consolidate_memory", pytest.fail)
    analyze("N001", "research", context=analysis_context, runner=FakeAnalyzer())
~~~

Mock `subprocess.Popen` for `--async` and assert it starts a hidden worker command
then returns immediately. The worker—not the launcher—takes the non-blocking lock,
checks the gate, and runs the session. An occupied lock returns `locked` without a
Codex call. Runner or publisher failure must leave Analysis, Evaluation, and
Reference files unchanged.

- [ ] **Step 3: Write failing trusted-target, sandbox, and startup-Hook tests**

~~~python
@pytest.mark.parametrize(
    ("kind", "target"),
    [
        ("research_summary", "research_summary.md"),
        ("engineering_summary", "engineering_summary.md"),
    ],
)
def test_summary_artifacts_have_fixed_targets(kind, target):
    result = validate_artifact(
        kind,
        valid_summary(),
        {"node_id": None, "mode": None, "version": None},
    )
    assert result["relative_path"] == target


def test_only_researcher_has_memory_hook(task_yaml):
    researcher, engineer = task_yaml["agents"]["assignments"]
    assert researcher["runtime_options"]["startup_hook"] == [
        ".venv/bin/autoresearch",
        "consolidate-memory",
        "--if-due",
        "--async",
    ]
    assert "startup_hook" not in engineer["runtime_options"]


def test_hook_failure_still_execs_runtime(monkeypatch):
    calls = record_hook_and_exec(monkeypatch, hook_returncode=1)
    run_startup_hook(["hook"], ["codex", "exec"])
    assert calls == [("hook",), ("exec", ("codex", "exec"))]
~~~

SRT tests require both Summary paths in `denyWrite` while
`memory-consolidation.lock` remains writable. Trusted writer tests require fixed
Summary targets to support atomic replacement and reject every other notes-root
target.

- [ ] **Step 4: Run the new tests and verify RED**

Run:

~~~bash
uv run --frozen --extra dev pytest \
  tasks/greedylore_autoresearch/tests/test_memory.py \
  tasks/greedylore_autoresearch/tests/test_analysis_cli.py \
  tasks/greedylore_autoresearch/tests/test_handoff_assets.py \
  tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py \
  tests/test_codex_startup_hook.py \
  tests/test_trusted_writes.py \
  tests/test_sandbox.py -q
~~~

Expected: FAIL because `memory.py`, Summary kinds, startup Hook, and protected
Summary paths do not exist.

- [ ] **Step 5: Implement the minimal file-reference gate**

`list_analysis_refs()` accepts only
`nodes/Nddd/(research|engineering)_analysis_vN.md`.
`read_processed_analyses()` parses only the YAML frontmatter string list and
returns an empty set for an absent Summary. `memory_due()` intersects the two
Summary processed sets, extracts distinct Node IDs from unprocessed Research
refs, and returns true at three Nodes. Engineering refs never count. Multiple
unprocessed versions of one Node count once.

`validate_summary()` requires valid frontmatter, a nonempty Markdown body,
canonical Analysis refs, and existing referenced files. It does not read
Analysis bodies, build EvidenceBundle, calculate hashes, judge conclusions, or
enforce processed-list completeness.

- [ ] **Step 6: Implement the independent Memory worker**

The Manager runs one
`codex exec --ephemeral --ignore-user-config --sandbox read-only` session with
the current island root as its read root, so Node notes and selected
`eval_logs` are available without exposing another island. Its prompt delegates
discovery of new/revised Research and Engineering Analyses, filtering, semantic
deduplication, conflict analysis, and uncertainty to Codex. Engineering Analysis
is input to consolidation but never contributes to the three-Research-Node gate.
The Manager may inspect Evaluation logs only as needed. The output schema is
exactly:

~~~json
{
  "research_summary": "<Markdown with processed_analyses frontmatter>",
  "engineering_summary": "<Markdown with processed_analyses frontmatter>"
}
~~~

The worker takes `LOCK_EX | LOCK_NB` on
`.coral/public/memory-consolidation.lock`; this file is synchronization, not
workflow state. `--async` starts the same CLI in hidden worker mode and returns.
After validating both outputs, publish them sequentially through the trusted
publisher and report success only after both complete. On partial publication,
the next gate uses the intersection of both processed lists. Do not retry in a
loop or alter source Artifacts.

- [ ] **Step 7: Add fixed Summary publication and SRT protection**

`validate_artifact()` accepts both Summary kinds only when node/mode/version are
null and returns their two fixed root filenames. Extend CORAL's canonical
allowlist by these exact names; do not accept caller paths. Summaries are
replaceable via temp-file/fsync/replace, while versioned Analysis and spec remain
immutable.

Add both Summary paths to `_protected_write_paths()`. Keep the lock file
writable and retain existing Attempts/logs/tree/nodes/private protections. Do not
add an Artifact router, role matrix, or permission manager.

- [ ] **Step 8: Implement the minimal Codex startup Hook**

Create a wrapper that accepts a nonempty Hook argv and normal runtime argv, calls
`subprocess.run(hook, check=False)`, logs Hook `OSError`, and always finishes with
`os.execvpe(runtime[0], runtime, os.environ)`. No shell string is allowed.

Codex runtime removes `startup_hook` from runtime options before constructing
Codex `-c` arguments and wraps the normal command before `apply_sandbox()`:

~~~text
python -m coral.agent.startup_hook --hook-json '<JSON argv>' -- <codex command...>
~~~

This keeps Hook, async worker, and Codex under the same SRT/registered-Agent
process root. Because every start, resume, and automatic restart calls
`runtime.start()`, manager lifecycle code remains unchanged. Configure only the
Researcher assignment with:

~~~yaml
startup_hook:
  - .venv/bin/autoresearch
  - consolidate-memory
  - --if-due
  - --async
~~~

Engineer has no Hook. Existing trusted-client startup retry handles the short PID
registry race. If Researcher exits before worker publication, the worker may fail
and the next Researcher launch checks again.

- [ ] **Step 9: Add the Memory prompt and verify GREEN**

`global_memory_manager.md` states that Summaries are derived navigation, preserves
Node/Analysis citations, applicability and uncertainty, outputs the complete
`processed_analyses` list, and never modifies Reference, Analysis, Evaluation, or
commits. Asset tests require this prompt and verify the static Reference Toolkit
content remains unchanged.

Run the Step 4 command.

Expected: PASS.

- [ ] **Step 10: Run core acceptance and full non-GPU regression**

Run:

~~~bash
uv run --frozen --extra dev pytest \
  tests/test_trusted_writes.py \
  tests/test_sandbox.py \
  tests/test_codex_startup_hook.py \
  tasks/greedylore_autoresearch/tests/test_analysis_cli.py \
  tasks/greedylore_autoresearch/tests/test_analysis_recovery.py \
  tasks/greedylore_autoresearch/tests/test_node_rules.py \
  tasks/greedylore_autoresearch/tests/test_evaluation_index.py \
  tasks/greedylore_autoresearch/tests/test_memory.py \
  tasks/greedylore_autoresearch/tests/test_handoff_assets.py \
  tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py -q
~~~

Run: `uv run --frozen --extra dev pytest -q`

Expected: PASS without starting a real Codex Memory session, GPU training, timer,
or background service.

- [ ] **Step 11: Commit**

~~~bash
git add coral/agent/startup_hook.py coral/agent/builtin/codex.py \
  coral/grader/trusted_writes.py coral/sandbox/srt.py \
  tests/test_codex_startup_hook.py tests/test_trusted_writes.py tests/test_sandbox.py \
  tasks/greedylore_autoresearch
git commit -m "feat: add startup-triggered autoresearch memory"
~~~

---

### Task 8: 真实环境 smoke、dense calibration 与 Root N000

**Files:**
- Create after real runs: `tasks/greedylore_autoresearch/taskdata/dense_references/manifest.json`
- Create after real runs: `tasks/greedylore_autoresearch/taskdata/dense_references/quality.json`
- Create after real runs: `tasks/greedylore_autoresearch/taskdata/dense_references/efficiency.json`
- Modify after real runs: `tasks/greedylore_autoresearch/README.md`

**Interfaces:**
- Consumes: installed SRT/bubblewrap/socat, four idle same-model GPUs, fixed
  C4/tokenizer, and passing Tasks 1–7.
- Produces: verified dense references; S0/A0; N000 Research Attempt and Research
  Analysis; a real startup-Hook skip-path record; documented limitations.

- [ ] **Step 1: Check host dependencies and fail closed**

Run each command:

~~~bash
npx --no-install srt --version
command -v bwrap
command -v socat
command -v rg
command -v nvidia-smi
~~~

Expected: every command returns 0. Stop on missing dependencies; installation
requires separate user authorization.

- [ ] **Step 2: Probe dynamic GPU resources**

Run:

~~~bash
uv run --project tasks/greedylore_autoresearch/grader \
  python -m greedylore_autoresearch_grader.gpu --count 4 --json
~~~

Expected: four idle GPU indices/UUIDs/names plus topology/NUMA fingerprint, with no
fixed index assumption.

- [ ] **Step 3: Run four-GPU correctness smoke**

Use fixed data/tokenizer and dynamic GPUs for 3 updates of 60M AdamW BF16 and
130M Muon BF16. Verify finite loss, world size 4, cleanup, and result schema.

- [ ] **Step 4: Run short efficiency smoke**

Run 130M AdamW BF16/FP32 with `warmup=3/measured=3` in reversed two-run order.
Verify SHM NCCL, normal asynchronous hook, bucket layout, slowest-rank wall time,
and a fresh process group per repetition.

- [ ] **Step 5: Commit smoke fixes only if smoke exposes a defect**

~~~bash
git add coral tasks/greedylore_autoresearch
git commit -m "fix: address four-GPU smoke findings"
~~~

Do not create an empty commit.

- [ ] **Step 6: Calibrate dense references serially**

Run:

~~~bash
uv run --project tasks/greedylore_autoresearch/autoresearch \
  autoresearch calibrate-dense --task tasks/greedylore_autoresearch
~~~

Expected: complete 60M AdamW, 2,500-update 130M Muon, and two repetitions of both
130M AdamW timing cells. Freeze contract/source hashes, GPU UUIDs,
topology/NUMA/NCCL and software fingerprints, commands, and raw-log hashes.

- [ ] **Step 7: Verify and commit dense references**

Run:

~~~bash
uv run --frozen --extra dev pytest \
  tasks/greedylore_autoresearch/tests/test_contract.py \
  tasks/greedylore_autoresearch/tests/test_grader.py -v
~~~

Commit:

~~~bash
git add tasks/greedylore_autoresearch/taskdata/dense_references \
  tasks/greedylore_autoresearch/README.md
git commit -m "data: freeze dense evaluation references"
~~~

- [ ] **Step 8: Bootstrap Root and run its unique Research Attempt**

Start CORAL. Confirm Researcher startup Hook calls
`consolidate-memory --if-due --async` and skips because fewer than three Research
Nodes exist. Record S0, create N000 and A0 whose only diff is `submission.yaml`,
run both quality workloads and both efficiency cells, rebuild `evaluation.md`,
then publish one fresh Root Research Analysis. Analysis publication itself must
not trigger Memory.

- [ ] **Step 9: Verify Root Artifacts and write boundaries**

Confirm: one N000 Research SHA; reconstructible grader history; fixed Analysis
evidence; Child creation immediately after Research Analysis; direct Agent writes
to Attempts/logs/Node history/Summaries are denied; `coral eval`,
`autoresearch retry`, `refresh-evaluation`, `analyze`, and the startup-Hook skip
path work. Speedup must use comparable hardware.

- [ ] **Step 10: Commit only factual real-environment records**

~~~bash
git add tasks/greedylore_autoresearch/README.md
git commit -m "docs: record Root N000 validation"
~~~

Run-local Attempts, Analysis, and Summaries stay in CORAL Artifacts.

---

## Final Verification

- [ ] Run: `git diff --check`
- [ ] Run: `uv run --frozen --extra dev ruff check coral tests tasks/greedylore_autoresearch/grader tasks/greedylore_autoresearch/autoresearch`
- [ ] Run: `uv run --frozen --extra dev pytest -q`
- [ ] Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests -q`
- [ ] Run: `uv run --frozen --extra dev coral validate tasks/greedylore_autoresearch --json`
- [ ] Inspect `git log --oneline 0123dfb..HEAD` and confirm one commit per completed
  large Task, with no hidden push, long training, or Memory session in hooks.
- [ ] Confirm `/home/wyr/greedy_lore` remains clean and unchanged.
