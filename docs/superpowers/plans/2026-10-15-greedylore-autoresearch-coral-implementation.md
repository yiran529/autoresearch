# GreedyLoRE CORAL AutoResearch V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在锁定的 CORAL fork 中交付一个以 GreedyLoRE r32/EF14 为 Root、支持 Research/Engineering 分流、受控证据发布和固定 GPU 评测合同的 AutoResearch V1 task。

**Architecture:** 复用 CORAL 原生 Commit → Attempt → Grader 主链，只在现有 grader daemon 内增加固定用途的 trusted submit/retry/publish 入口，并收紧 SRT 写边界；不增加通用 RPC、第二套状态数据库或工作流引擎。GreedyLoRE task 自己拥有 evaluation contract、submission/artifact 校验、Node/Analysis/Review 规则和模板；正式 quality 使用 60M+AdamW 完训与 130M+Muon 前 2,500 updates，正式 efficiency 只使用 130M+AdamW 的 BF16/FP32 两个 cell。

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
- Agent 不能直接写 Attempts、eval logs、Node history、Analysis/Review 或 private data；`coral eval`、`autoresearch retry` 和 task-local Artifact 命令通过固定用途的可信入口发布。
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

### Task 6: 实现极简 Node、evaluation rebuild 与 Root bootstrap

**Files:**
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/atomic.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/nodes.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/evaluation.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/bootstrap.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/cli.py`
- Modify: `coral/grader/trusted_writes.py`
- Modify: `tests/test_trusted_writes.py`
- Create: `tasks/greedylore_autoresearch/tests/test_node_rules.py`
- Create: `tasks/greedylore_autoresearch/tests/test_evaluation_index.py`
- Create: `tasks/greedylore_autoresearch/tests/test_root_bootstrap.py`

**Interfaces:**
- Consumes: Task 3 的 `publish_artifact` 与唯一 Research Attempt 保证；CORAL Attempts、`grader_runs`、raw eval logs 和 committed `submission.yaml`。
- Produces: `create_node(request, evidence) -> Path`；`refresh_node_evaluation(node_id, attempts, log_root) -> str`；`bootstrap_root(repo, source_manifest) -> Path`；CLI `autoresearch node create` 与 `autoresearch refresh-evaluation`。不存在 Node 状态数据库、Engineering Iteration 模型或通用 Artifact Publisher。

- [x] **Step 1: 写 fixed reference、rebuild、零/multi Engineering 与 Root 幂等失败测试**

```python
def test_child_freezes_selected_parent_versions(parent_node):
    spec = parse_spec(create_node(child_request(parent_node, analysis_version=2)))
    assert spec["parent_commit"] == parent_node.research_sha
    assert spec["analysis_ref"].endswith("research_analysis_v2.md")
    assert spec["review_ref"].endswith("research_review_v2.md")
    assert spec["selection_rationale"]


@pytest.mark.parametrize("engineering_count", [0, 2])
def test_evaluation_rebuild_is_deterministic(fixture_run, engineering_count):
    fixture_run.keep_engineering(engineering_count)
    first = refresh_node_evaluation("N001", fixture_run.attempts, fixture_run.logs)
    second = refresh_node_evaluation("N001", fixture_run.attempts, fixture_run.logs)
    assert first == second
    assert first.count("| engineering |") == engineering_count


def test_root_bootstrap_is_idempotent(root_fixture):
    first = bootstrap_root(root_fixture.repo, root_fixture.manifest)
    assert bootstrap_root(root_fixture.repo, root_fixture.manifest) == first
    assert root_fixture.changed_paths("S0", "A0") == {"submission.yaml"}
```

- [x] **Step 2: 运行测试并确认接口缺失**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_node_rules.py tasks/greedylore_autoresearch/tests/test_evaluation_index.py tasks/greedylore_autoresearch/tests/test_root_bootstrap.py -v`

Expected: FAIL，Node/rebuild/bootstrap 接口尚不存在。

- [x] **Step 3: 实现 task-specific Artifact 校验和原子发布**

`artifacts.py` 只把 kind 映射到 `tree.md`、`nodes/Nxxx/spec.md`、`evaluation.md` 或版本化 Analysis/Review 文件；调用者不能提供路径。验证 Node ID、mode/version、symlink 和 resolved target，再由 Task 3 trusted writer 执行临时文件、flush/fsync、`os.replace` 和 parent fsync。不要抽象通用 Publisher 类或可配置路径规则。

- [x] **Step 4: 实现最小 Node spec 规则**

Root 固定 `N000`、parent null、base `S0`。普通 child ID 从已发布 Node 单调选择候选；trusted publication lock 保证同一 ID 只有一个发布成功，极少见的并发冲突由调用方重试；spec 保存 hypothesis、parent node、fixed parent/base commit、Node Selection Rationale、delta、risk、falsification criteria 和创建时选中的 Analysis/Review 精确版本。只有 Research Review pass/warning 才允许扩展；Research Attempt 唯一性由 Task 3 保证。

- [x] **Step 5: 实现 deterministic evaluation rebuild**

只从 Attempts、`grader_runs`、raw result JSON 和 committed `submission.yaml` 生成。输出唯一 Research SHA/retry 历史，并把每次正式 efficiency evaluation 映射成一个 CORAL Engineering Attempt；允许零 Engineering Attempt，不创建 iteration record。多个不同 Research SHA 报 consistency error 且不覆盖；缺值写 `—`，不可比 contract 不排序。

- [x] **Step 6: 实现一次性幂等 Root bootstrap**

区分 upstream source SHA 与运行仓库 S0，按 source manifest（含 adaptations/relocations）验证 seed；canonical A0/N000 已存在且匹配时幂等恢复缺失发布，冲突时失败；通过受限入口发布 spec。immutable Artifact 的相同内容重发幂等，不同内容仍拒绝。从固定模板只新增 `submission.yaml`，Task 3 接受 A0 前强制 `git diff --name-only S0 A0 == {"submission.yaml"}`。

- [x] **Step 7: 运行 Node/evaluation/bootstrap 测试**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_node_rules.py tasks/greedylore_autoresearch/tests/test_evaluation_index.py tasks/greedylore_autoresearch/tests/test_root_bootstrap.py -v`

Expected: PASS。

- [x] **Step 8: Commit**

```bash
git add tasks/greedylore_autoresearch/autoresearch tasks/greedylore_autoresearch/tests
git commit -m "feat: add artifact-driven node workflow"
```

---

### Task 7: 实现轻量 Analysis / Reviewer CLI

**Files:**
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/analysis.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/validation.py`
- Create: `tasks/greedylore_autoresearch/autoresearch/schemas/analysis.schema.json`
- Create: `tasks/greedylore_autoresearch/autoresearch/schemas/review.schema.json`
- Create: `tasks/greedylore_autoresearch/tests/test_analysis_cli.py`
- Create: `tasks/greedylore_autoresearch/tests/test_analysis_recovery.py`

**Interfaces:**
- Produces: `analyze(node_id, mode, revise=False) -> AnalysisResult`；`resolve_evidence(node_id, mode) -> EvidenceBundle`；`run_fresh_codex(prompt, read_roots, output_schema, scratch) -> Path`；CLI `autoresearch analyze --node Nxxx --mode research|engineering [--revise]`。完成状态只由已校验、已发布的 Artifact 推导。

- [ ] **Step 1: 写幂等、缺 Review 恢复、revision 与零 Engineering Attempt 测试**

```python
def test_review_failure_only_retries_missing_review(fake_codex, node):
    fake_codex.outputs = [VALID_ANALYSIS, RuntimeError("review crashed"), VALID_REVIEW]
    with pytest.raises(RuntimeError):
        analyze(node.id, "research")
    result = analyze(node.id, "research")
    assert fake_codex.analysis_calls == 1
    assert fake_codex.review_calls == 2
    assert result.analysis_version == 1


def test_engineering_analysis_allows_no_engineering_attempt(fake_codex, research_only_node):
    result = analyze(research_only_node.id, "engineering")
    assert result.evidence.engineering_attempts == []


def test_engineering_analysis_freezes_all_attempts(fake_codex, node_with_three_eng_attempts):
    result = analyze(node_with_three_eng_attempts.id, "engineering")
    assert [a.commit_sha for a in result.evidence.engineering_attempts] == ["E1", "E2", "E3"]
    assert all(a.eval_refs for a in result.evidence.engineering_attempts)


def test_revise_keeps_v1_and_publishes_v2(node):
    first = analyze(node.id, "research")
    second = analyze(node.id, "research", revise=True)
    assert first.analysis_path.exists() and first.review_path.exists()
    assert second.analysis_version == second.review_version == 2
```

- [ ] **Step 2: 运行测试并确认轻量 CLI 缺失**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_analysis_cli.py tasks/greedylore_autoresearch/tests/test_analysis_recovery.py -v`

Expected: FAIL，evidence resolver、fresh runners 和 Artifact 恢复逻辑尚不存在。

- [ ] **Step 3: 从 Node 与 evaluation Artifact 固定 evidence**

Research mode 绑定唯一 terminal Research SHA、contract hash、evaluation version 和 Node spec。Engineering mode 绑定相同 Research anchor，并从 `evaluation.md` 与原始 Attempts/logs **按提交时间固定该 Node 的全部 Engineering Attempts**，逐项保存 `commit_sha`、正式 efficiency run IDs/原始日志及可比性状态；`engineering_attempts: []` 表示没有工程提交。启动时固定完整 evidence manifest/hash，Analysis/Review 均引用该版本；后续新增工程 Attempt 不得悄悄改变已发布 Analysis 的证据集。没有 Engineering Attempt 时只分析 Research 实现、Research evaluation 和已有 profiling evidence，不得声称未测量的性能提升。analyze 不重跑正式训练。

- [ ] **Step 4: 实现两个独立 fresh Codex session**

```bash
codex exec --ephemeral --ignore-user-config --sandbox read-only \
  --skip-git-repo-check --cd SCRATCH \
  --output-schema SCHEMA --output-last-message OUTPUT -
```

Analyzer 与 Reviewer 使用不同 scratch 和全新命令；只读固定 checkout、已发布 evidence 和 references；不得 `resume`、复用 session/thread 或继承对话。

- [ ] **Step 5: 校验并通过受限入口发布**

Analysis 匹配 node、mode、Research SHA、**固定且可为空的 Engineering Attempt 列表及各 run 引用**、evaluation version、evidence manifest hash 和 evidence paths；Review 必须引用完全相同的证据快照及 Analysis version，包含 workflow/evidence checks、issues/actions 和唯一 `decision: pass|warning|fail`。scratch 输出完整校验后才调用 `publish_artifact`；版本分配与 canonical target 检查在 trusted lock 内完成。

- [ ] **Step 6: 实现 Artifact 驱动恢复规则**

无 Analysis 时运行 Analyzer；有 Analysis、无 Review 时只运行 Reviewer；两者有效时幂等返回。`fail` 是 terminal Review 但不是通过，也不自动修订；`--revise` 才创建下一版本并保留历史。Research Review pass/warning 后即可扩展 Child，不等待 Engineering Analysis。

- [ ] **Step 7: 运行 Analysis / Reviewer 测试**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_analysis_cli.py tasks/greedylore_autoresearch/tests/test_analysis_recovery.py -v`

Expected: PASS；证明 session 分离、artifact-only recovery、零 Engineering Attempt、多个 Engineering Attempt 的完整固定历史及 Analysis/Review 引用一致。

- [ ] **Step 8: Commit**

```bash
git add tasks/greedylore_autoresearch/autoresearch tasks/greedylore_autoresearch/tests
git commit -m "feat: add artifact-driven analysis review"
```

---

### Task 8: 部署 roles、prompts、references、templates 并完成 task 配置

**Files:**
- Create: `tasks/greedylore_autoresearch/roles/workflow_rules.md`
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
- Consumes: Appendix A/B/C from `/home/wyr/CORAL_AutoResearch_V1_Handoff_Refined.md` and Tasks 3/6/7 的最终 CLI。
- Produces: two Codex assignments with distinct role files；简短共享 `workflow_rules.md`；private dense references；task-local validators/CLI 安装；`agents.heartbeat_enabled=false`；`grader.parallel.max_workers=1`；SRT enabled。

- [ ] **Step 1: 写 asset、task config 与 Prompt 边界失败测试**

```python
def test_task_has_two_distinct_codex_roles(task_yaml):
    assignments = task_yaml["agents"]["assignments"]
    assert [a["runtime_options"]["role_file"] for a in assignments] == [
        "roles/researcher.md", "roles/engineer.md"
    ]
    assert task_yaml["agents"]["heartbeat_enabled"] is False
    assert task_yaml["grader"]["parallel"]["max_workers"] == 1


def test_prompts_expose_workflow_not_transport(prompt_texts):
    combined = "\n".join(prompt_texts)
    for required in ("coral eval", "autoresearch retry", "autoresearch analyze"):
        assert required in combined
    for internal in ("trusted.sock", "flock", "grader_runs", "ControlProvider"):
        assert internal not in combined
```

- [ ] **Step 2: 提取 assets 并加入简短共享规则**

Researcher/Engineer/Analyzer/Reviewer 对应 Appendix A.1–A.5；algorithm/system/semantic 对应 B.1–B.3；templates 对应 C.1–C.6。`workflow_rules.md` 只描述任务、输入 Artifact、完成条件、Node/Attempt 规则、`coral eval`、`autoresearch retry`、`autoresearch analyze`、读写/发布边界和失败记录。保留 Node Selection、Reference Toolkit、Semantic Criteria；V1 Semantic Test 仍是 Prompt + Criteria 软约束。Prompt 不解释 UDS、锁、`grader_runs` 或 SRT internals。

- [ ] **Step 3: 完成 task.yaml 与隔离安装**

`workspace.repo_path` 指向 seed；`workspace.setup` 只在 CORAL 新建的每个 worktree `.venv` 中安装 seed 依赖和 task-local autoresearch CLI，不修改 `/home/wyr/greedy_lore/.venv` 或其他既有环境。`grader.setup` 在 `.coral/private/grader_venv` 安装 grader 与 autoresearch；`grader.args` 固定 trusted submission/artifact validator entrypoints。两个 Codex assignments 各 count 1；sandbox provider srt；network allowlist 只保留 Agent 必需域名，正式 evaluator offline。没有 control provider 配置。

- [ ] **Step 4: 运行 asset/config 测试与 CORAL structural validation**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_handoff_assets.py tasks/greedylore_autoresearch/tests/test_contract_static.py -v`

Run: `uv run --frozen --extra dev coral validate tasks/greedylore_autoresearch --json`

Expected: PASS；validation 使用 `coral_validation=true` 便宜路径，不启动正式 8,393/2,500-step 训练。

- [ ] **Step 5: Commit**

```bash
git add tasks/greedylore_autoresearch
git commit -m "feat: deploy autoresearch roles and artifact contracts"
```

---

### Task 9: 收紧粗粒度 SRT 边界并完成 S1–S15 dummy acceptance

**Files:**
- Modify: `coral/sandbox/srt.py`
- Modify: `tests/test_sandbox.py`
- Create: `tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py`
- Create: `tasks/greedylore_autoresearch/tests/test_acceptance_s1_s15.py`
- Create: `tasks/greedylore_autoresearch/tests/fixtures/attempts/`
- Create: `tasks/greedylore_autoresearch/tests/fixtures/analyses/`
- Create: `tasks/greedylore_autoresearch/tests/fixtures/reviews/`

**Interfaces:**
- Consumes: Task 3 的 `trusted.sock` 和 Task 8 的 Agent commands。
- Produces: `_protected_write_paths(coral_dir, island_id) -> list[str]`；read published evidence, write own worktree/scratch, publish explicitly；Agent 可完成 Git commit、`coral eval`、`autoresearch retry/analyze`，但不能改写 formal records、private data 或其他 Agent worktree。不增加角色权限矩阵或 Permission Manager。

- [ ] **Step 1: 写 settings 与真实边界攻击失败测试**

```python
def test_settings_deny_formal_records_but_keep_own_worktree(tmp_path):
    settings = build_srt_settings(SandboxConfig(enabled=True), proxy_port=1, **_paths(tmp_path))
    denied = set(settings["filesystem"]["denyWrite"])
    assert str((tmp_path / ".coral/public/attempts").resolve()) in denied
    assert str((tmp_path / ".coral/public/eval_logs").resolve()) in denied
    assert str((tmp_path / ".coral/public/notes/tree.md").resolve()) in denied
    assert str((tmp_path / ".coral/public/notes/nodes").resolve()) in denied
    assert str((tmp_path / ".coral/.git").resolve()) in denied
    assert str(_paths(tmp_path)["worktree_path"].resolve()) in settings["filesystem"]["allowWrite"]


@pytest.mark.parametrize("attack", ["symlink", "absolute", "dotdot", "sibling_worktree"])
def test_real_srt_blocks_protected_write(launched_sandbox, attack):
    assert launched_sandbox.try_attack(attack).returncode != 0
```

- [ ] **Step 2: 运行 sandbox tests 并确认新断言失败**

Run: `uv run --frozen --extra dev pytest tests/test_sandbox.py tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py -v`

Expected: FAIL，锁定 fork 仍允许写整个 `coral_dir`。

- [ ] **Step 3: 实现最小 protected path 策略**

Single/multi-island 不再 allow-write 整个 `.coral`；显式 deny Attempts、eval logs、`notes/tree.md`、`notes/nodes`、checkpoint repo 和 private，private 同时 deny-read。只保留 own worktree、candidate repo `.git`、scratch、必要 runtime home、`/tmp` 和 trusted socket connect；同 island sibling worktree 与已发布 evidence 可读但不可写，跨 island 不可读。不要建立角色矩阵，也不要依赖 Prompt 代替 OS 隔离。服务端仍验证 request，socket permission 不是授权。

- [ ] **Step 4: 验证 Git/eval/trusted commands 兼容和 fail-closed preflight**

真实 SRT test 在 sandbox 内依次完成普通修改、Git commit、`coral eval --no-wait`、`autoresearch retry` 请求和 `autoresearch analyze` 受限发布；同时证明直接写 Attempt/eval log/node、symlink escape、绝对路径和 sibling write 失败。**共享 Git 元数据单独验收**：记录 worktree `.git` 指向的 common dir，检验普通 `git commit` 必需的 ref/index 写入，且其他 Agent 不能直接覆盖兄弟工作文件；共享 `.git` 如允许写入，不宣称 Git refs 具有独立 OS 级写保护。受信任提交端须以固定 SHA 创建受保护 checkpoint/ref，并验证该 SHA 的树对象不受 Agent 后续分支移动影响。测试尝试移动其他 Agent 分支引用后，正式 Attempt 的 pinned SHA/评测内容仍可验证；无法保障时应明确失败，不得以 worktree 隔离替代 Git provenance 保护。保持 `ensure_sandbox_supported()` 对 `srt --version`、bubblewrap、socat、rg 的检查；任一缺失时在创建 Agent 前失败，不得静默 unsandboxed。mock 每个 dependency failure branch。

- [ ] **Step 5: 实现 S1–S15 dummy acceptance fixtures**

覆盖唯一 Research SHA/并发 reservation、同 UID 身份伪造、infra-only retry 上限及 finalize 并发、Analysis 幂等/缺 Review 恢复/revise、Engineering 零/多 Attempt 完整证据、child fixed refs、evaluation rebuild 多/零 Engineering、Reviewer pass/warning/fail、GPU lock max_workers=1、GPU 拓扑可比性、Root diff、reference citations、selection rationale、共享 Git refs 与 protected path attacks。Dummy tests 不运行 GPU 长训练。

- [ ] **Step 6: 运行 framework 与 task acceptance**

Run: `uv run --frozen --extra dev pytest tests/test_sandbox.py tests/test_config.py tests/test_heartbeat.py tests/test_hooks.py tests/test_grader_daemon.py tests/test_trusted_writes.py tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py tasks/greedylore_autoresearch/tests/test_acceptance_s1_s15.py -q`

Expected: PASS；真实 SRT case 仅可因明确记录的缺失 host dependency skip，`coral start` 对同一缺失必须 fail closed。

- [ ] **Step 7: 运行全量非 GPU 回归**

Run: `uv run --frozen --extra dev pytest -q`

Expected: PASS；GPU/host dependency 缺失导致的显式 skip 必须有原因，不允许 xfail 掩盖错误。

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

Expected: 返回四个 idle GPU 的 index/UUID/name、拓扑/NUMA 指纹；没有固定 `3,4,5,6` 假设。正式 efficiency 与 dense 对照需匹配相同 UUID 集合或已核实的拓扑等效环境。

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

Expected: 60M AdamW 完整 8,393、130M Muon 2,500/20,000，以及 130M AdamW BF16/FP32 每 cell 两次均完成；manifest 锁定 contract hash、GPU UUID 集合、拓扑/NUMA/NCCL fingerprint、software fingerprint、commands 与 raw log hashes。后续 candidate 如果未能匹配该卡组且无法证明拓扑等效，先在候选卡组补测对应 dense reference，不直接使用原始 speedup。

- [ ] **Step 7: 验证并提交 dense references**

Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests/test_contract.py tasks/greedylore_autoresearch/tests/test_grader.py -v`

```bash
git add tasks/greedylore_autoresearch/taskdata/dense_references tasks/greedylore_autoresearch/README.md
git commit -m "data: freeze dense evaluation references"
```

- [ ] **Step 8: Bootstrap Root 并运行唯一 formal Research Attempt**

启动 CORAL task，记录 S0；受控创建 N000 和仅含 `submission.yaml` 的 A0；运行两项 quality 与两项 efficiency；刷新 `evaluation.md`；运行 fresh Research Analysis 与 Reviewer。

- [ ] **Step 9: 验证 Root artifacts 与安全边界**

确认：S0→A0 只有 submission；N000 只有一个 Research SHA；所有 grader runs/log hashes 可重建 evaluation；Review 有 pass/warning/fail 明确结论；Agent 对 protected artifacts 的直接写入被拒绝，而 `coral eval`、`autoresearch retry`、`autoresearch refresh-evaluation`、`autoresearch analyze` 仍能工作；同 UID 提交者身份伪造被拒，共享 Git ref 改动不影响固定 SHA 评测证据，speedup 使用可比 GPU 环境。

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
- [ ] 核对 Task 3/9：同 UID 伪造 Agent 身份被拒、trusted server 在长评测/stop-resume 下可用、Git 公共元数据的权限边界已实测。
- [ ] 核对 Task 4/5/10：正式 efficiency speedup 使用相同 GPU UUID 集合或经过验证的拓扑等效对照；不可比时标记无效并补测 dense。
- [ ] 核对 Task 7：Engineering Analysis 固定全部已有 Engineering Attempts（可以为空），Review 与 Analysis 引用同一个证据快照。
