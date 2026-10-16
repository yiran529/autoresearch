# GreedyLoRE AutoResearch V1 Dynamic Node Agents — Incremental Design

**日期：** 2026-10-16
**状态：** 已确认
**基线：** `bcbb249`–`8834bd6` 已实现的 Trusted Attempt/Retry、Evaluation、Node Artifact、Analysis/Reviewer、角色资产和 Sandbox；后续 workflow simplification 尚未实现。

## 1. 目标与边界

本增量把两个长期 Producer 改成按需创建的 Node-bound Agent，并同时完成既定的 Analyzer-only 与 Global Memory 简化：

- 一个 Tree Node 是一个持久化 Research 单元，对应唯一正式 Research Attempt；
- 每个普通 Node 绑定一个独立 Researcher Agent、Git Worktree 和 Codex Session；
- 合格 Node 可选绑定一个独立 Engineer Agent，但 Engineering 不创建新的 Tree Node；
- Scheduler 维持 `max_researchers` / `max_engineers` 并发，Agent 完成后立即补位；
- 所有共享 workflow 状态通过同一受信任入口原子修改；
- Reviewer 删除，Analysis 由一次独立 Analyzer session 生成；
- Researcher 每次启动仍检查 Global Memory 三 Node 门槛。

不改变 Evaluation Contract、GPU 分配与清理、Research Attempt 唯一性、同 SHA infra retry、Root Evaluation、Reference Toolkit 或 Grader 证据格式。默认静态 CORAL Agent 模式保持兼容。V1 动态模式只支持单 island。

## 2. 概念模型

### 2.1 Tree Node 与 Agent

Tree Node 是研究事实，不是运行进程。Node 永久保存 Parent、Hypothesis、Research Commit、Attempt、Evaluation 和 Analysis。Agent 是执行该流程的临时运行实体。

```text
Node N012
├── Researcher Agent research-N012（必有，唯一）
│   └── 一个正式 Research Attempt
└── Engineer Agent engineer-E0007（可选）
    └── 零个或多个 Engineering Attempts
```

Researcher 完成后，Node 的 Research 状态不会因后续 Engineering 改变。Engineering Commit、Attempts、Evaluation 和 Analysis 均归属原 Node，不覆盖 Research 成果。

### 2.2 生命周期

Research 状态保持最小：

```text
reserved -> active -> completed
             |
             +-> cancelled
```

Engineering 使用可追加的 claim 记录，每条 claim 有 `reserved / active / completed / cancelled`。V1 policy 将每个 Research Node 的最大 Engineering 流程数设为 1，但状态格式不把 1 固化，未来可提高 policy 而无需迁移 Node schema。

`completed` 表示流程与证据已封存，不表示结果优秀。Quality gate 是独立事实：Research quality 失败的 Node 仍可完成，但不能被 Engineer 认领。

## 3. 权威 Workflow State

`.coral/private/autoresearch_state.json` 是协调状态的唯一权威来源；它不复制 Evaluation 数值或 Analysis 正文。最小内容为：

```json
{
  "version": 1,
  "revision": 8,
  "next_node_seq": 13,
  "next_engineer_seq": 8,
  "nodes": {
    "N012": {
      "research": {
        "agent_id": "research-N012",
        "status": "active",
        "spec_ref": "nodes/N012/spec.md"
      },
      "engineering_claims": []
    }
  },
  "engineering_workers": {}
}
```

Root `N000` 由现有 host bootstrap 创建；第一次 reconcile 在验证其 Spec、terminal Research Attempt和Research Analysis后幂等登记为 completed。Root Analysis发布前不分配普通Node。Node ID 与 Agent ID 单调分配且永不复用；取消的 reservation 可以留下编号空洞。`tree.md` 是从 state、Node Specs 与 Analysis 重建的公开投影，不接受独立编辑。

所有读改写均在现有 `.coral/private/trusted_writes.lock` 的同一个 `flock` 临界区内完成：取锁后重新读取、校验当前状态、修改、递增 revision，并以 temp file + `fsync` + `os.replace` 发布。不得基于进程内缓存分配 ID。State handler自己持锁；trusted service调用该handler时不重复持锁，handler持锁时也不通过socket调用Publisher。Spec由handler用现有task-local原子写入helper直接发布；若Spec已写但state尚未转active，相同内容的initialize重试可修复。Tree投影若中断，下次reconcile重建。

## 4. 固定可信 Workflow 操作

复用 grader daemon 已有 `trusted.sock`、peer PID 认证和 task-private Python。该入口增加以下固定操作：

- `reconcile_workflow`：仅 manager peer 可调用；按可信 config 中的容量返回应运行的 assignments，并原子补足 Research reservations / 必要的未绑定 Engineer workers；
- `initialize_research`：仅 reservation 对应的 Researcher 可调用；验证 Parent Research Analysis、干净的固定 base、Node payload 后发布 `spec.md` 并转为 active；
- `claim_engineering`：仅未绑定 Engineer 可调用；在锁内重新验证目标资格并写入 claim；
- `complete_assignment`：验证调用者归属、Attempt/Evaluation/Analysis 和无 pending run，再完成 Research 或 Engineering；
- `cancel_assignment`：只取消尚未产生正式 Attempt 的 reservation/claim，保留 ID 和历史记录。

Task handler 由 `grader.args.trusted_workflow_handler` 固定配置；另配置只读的`trusted_workflow_authorizer`。Agent payload 不能指定任意路径、Agent 身份、计数器或最终状态。Manager operation 通过 `SO_PEERCRED` 匹配 grader daemon 的 parent PID；Agent operation 继续使用现有 registered process-root binding。容量只从可信task config读取，不接受Manager或Agent请求覆盖。

现有 `submit_attempt` 在同一Attempt锁内、创建Attempt之前调用只读authorizer校验 workflow ownership：Research Attempt 必须来自该 Node 的 Researcher；Engineering Attempt 必须来自当前有效 claim 的 Engineer。这样旧 Session、错误 Worktree 或其他 Agent 无法把结果归属到该 Node。

## 5. 并发与一致性

### 5.1 Node ID 分配

`reconcile_workflow` 在锁内读取 `next_node_seq`、写入 `reserved` record、递增 counter 并原子保存。两个并发调用分别获得不同编号；当前扫描 `notes/nodes` 的 `_next_node_id()` 被删除。

Manager 在 reservation 已提交但响应丢失后重试时，reconcile 先返回已有未完成 assignment，不重复分配。相同 Node 或 Agent ID 永不复用。

### 5.2 Engineering 认领

Scheduler 只判断是否至少存在未认领的合格 Node，不排序候选。Engineer 自主阅读 Tree、Analysis、Evaluation、Profiling 和 Summary 后提交目标 Node ID。`claim_engineering` 在锁内重新验证：

- Research 已 `completed`；
- Quality gate 通过；
- Research Commit 与 code snapshot 可读取；
- 没有 active claim；
- 已完成 claim 数低于当前 policy limit。

并行 Engineer 选择同一 Node 时只有第一个 claim 成功；失败者重新读取 evidence 并选择其他目标。已无目标时取消自己的未绑定 assignment 并退出。

### 5.3 防止旧结果覆盖

Agent ID 每次 workflow 唯一，trusted registry 只认证当前进程根。状态转换检查 expected current status 和 owner；immutable Node Spec / versioned Analysis 不覆盖；Attempt 仍以 commit SHA 为键。完成后的 Agent 从 desired assignments 中移除，后续写入被 ownership 检查拒绝。

### 5.4 Artifact 与状态恢复

Research/Engineering事实仍由 Git、Attempts、eval logs、Node Spec 和 Analysis组成。已有普通 Node 的运行若丢失private state，启动须报错并停止分配；不能从空计数器重新发放N001。管理员可以根据这些持久证据人工恢复状态。`evaluation.md` 和 `tree.md` 始终可重建。

## 6. Researcher Workflow

1. Root N000的Research Analysis就绪后，Manager调用reconcile，原子预留 `Nxxx` 与 `research-Nxxx`。
2. Manager 从 Researcher template 创建独立 Worktree和全新 Codex Session；启动 Hook异步检查 Memory。
3. Researcher 阅读 Tree、Summary和固定 Analyses，自主选择 Parent与假设。
4. `autoresearch node initialize` 将干净 Worktree定位到选定 Parent Research Commit，并通过可信操作发布 `spec.md`。
5. Researcher实现算法，通过 `coral eval`提交该 Node唯一正式 Research Attempt。
6. Grader发布 Quality Evaluation；`autoresearch analyze --mode research` 启动独立 Analyzer并发布 Analysis。
7. `autoresearch node complete --mode research` 验证证据并原子完成。
8. Manager停止已完成进程，保存 Git/日志证据，安全移除干净 Worktree并立即补位。未完成退出则恢复原 Node、Worktree和 Session。

Research quality 无论通过或失败都可完成；只有通过者满足 Engineering 资格。Research Analysis 发布后即可作为 Child evidence，不等待 Engineering。

## 7. Engineer Workflow

1. Reconcile 仅在存在合格且未认领 Node 时，按 `max_engineers` 创建未绑定 Engineer worker。
2. Manager 为其创建独立 Agent ID、Worktree和 Session。
3. Engineer 自主比较全局 evidence 并调用 `autoresearch engineering claim Nxxx`。
4. 可信入口原子认领并返回固定 Research Commit；CLI 将干净 Engineer Worktree定位到该 commit。
5. Engineer实施系统优化，通过 `coral eval`提交属于原 Node的 Engineering Attempts。
6. Grader执行固定 Efficiency Evaluation；Analyzer发布 Engineering Analysis。
7. `autoresearch node complete --mode engineering` 完成 claim，Manager释放资源并按需补位。
8. 崩溃恢复同一 Engineer Agent、claim、Worktree和 Session；不得创建新的 Engineering 流程。

## 8. CORAL Dynamic Agent 集成

静态 `agents.assignments` 保持原行为。动态模式新增：

```yaml
agents:
  dynamic:
    enabled: true
    max_researchers: 2
    max_engineers: 1
  assignments:
    - role: research
      runtime_options:
        role_file: ./roles/researcher.md
    - role: engineering
      runtime_options:
        role_file: ./roles/engineer.md
```

Assignments 在动态模式下是角色 templates，不是长期 Agent。`AgentManager` 继续负责 Worktree provisioning、Runtime启动、Sandbox、PID registry、Session提取、crash circuit breaker、pause/resume 和 grader lifecycle；task-local handler 只返回 desired assignments，不启动进程。

Manager 在 start/resume、workflow state改变和 Agent退出后 reconcile：

- desired 且未运行：从对应 template 构造 `AgentSpec` 并启动；
- desired 且异常退出：使用现有 `_restart_agent()` 恢复 Session；
- 不再 desired：不重启，停止 handle 并回收 Worktree；
- 容量空缺：下一次 reconcile 原子补足。

Run resume 以 authoritative desired assignments 为准，不通过遍历所有旧 Worktree 决定应恢复谁。静态模式、islands、migration 和原有 assignment tests 不改变；V1 对 `dynamic.enabled && islands.count != 1` 明确报错。

## 9. Analyzer、Subagent 与 Global Memory

Reviewer schema、Prompt、Artifact和调用阶段删除。`autoresearch analyze`只运行一次独立 Analyzer，保留幂等和 `--revise`。Research Analysis有效发布即可创建 Child。

Researcher、Engineer和Analyzer遇到实质不确定性时可自主使用有限 Subagent；常规工作不强制使用，Subagent继承父 Agent权限。

每次 Researcher Agent初启或恢复时调用：

```bash
autoresearch consolidate-memory --if-due --async
```

一 Node 一 Researcher Session使该检查自然随研究推进。Gate仍按两份 Summary共同未处理的 Research Analysis所属不同 Node计数，达到 3 才运行；Engineering Analysis不计数但纳入输入。

CLI不构建 Evidence Bundle/hash或解释知识差异，但必须记录本次 gate 使用的 Research Analysis引用，并要求两份输出的 `processed_analyses` 都覆盖这些引用。Memory Manager负责筛选、去重、冲突分析和完整 processed list。简单非阻塞 flock防并发；失败不阻塞研究，下次 Researcher启动重试。

异步 worker测试覆盖：launcher立即返回、worker持锁完成两份发布、Researcher启动包装器退出后worker仍可完成，以及worker失败后下一次检查仍 due。不新增 timer、heartbeat action或常驻调度器。

## 10. 完成条件与资源回收

Research completion要求：合法 spec、唯一 terminal Research Attempt、无可重试 infra failure、可重建 evaluation、固定 code snapshot、有效 Research Analysis和无 pending run。

Engineering completion要求：有效 claim、至少一个 terminal Engineering Attempt、可重建 evaluation、有效 Engineering Analysis和无 pending run。负收益可以完成并保留为失败经验。

Worktree仅在 workflow completed/cancelled且 `git status --porcelain`为空时非强制移除；branch、commit、Attempt、logs和Artifacts保留。清理失败只记录 diagnostics，不把已完成 workflow重新运行。暂停或异常 Agent保留 Worktree和 Session。

## 11. 验收重点

- 并发 reconcile不重复分配 Node ID，取消 ID不复用；
- 一个 Node只有一个 Researcher和一个正式 Research SHA；
- 两个 Engineer不能认领同一 Node；
- Research完成后Child可扩展，Engineering可缺席或晚到；
- Researcher/Engineer异常退出恢复原 Agent与 Session，完成退出不重启；
- manager restart只恢复 authoritative active/reserved assignments；
- 动态并发不超过 `max_researchers` / `max_engineers`，完成后自动补位；
- stale Agent不能提交 Attempt、完成状态或覆盖 Artifact；
- Analyzer-only、Memory三 Node gate、最小 completeness检查和Summary幂等成立；
- 原有静态 Agent、Attempt/Retry、Grader、Sandbox、GPU协议和Root流程回归通过。
