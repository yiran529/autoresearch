# GreedyLoRE AutoResearch V1 Dynamic Node Agents — Incremental Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前已实现的 GreedyLoRE AutoResearch V1 上，移除固定 Reviewer，将 Research workflow 改成一 Node 一 Researcher Agent，并为合格 Node 按需启动独立 Engineer Agent，同时保证共享状态并发安全和 Global Memory 低频更新。

**Architecture:** 保留现有 Commit → Attempt → Grader、可信 Artifact 发布、Sandbox、Worktree 和 Runtime。Task-local authoritative workflow store负责 Node ID、Research生命周期和Engineering claim；CORAL `AgentManager`只增加动态 assignment的启动、恢复、停止和补位能力。Research Node始终对应唯一 Research Attempt，Engineering归属原 Node而不创建新 Tree Node。

**Tech Stack:** Python 3.11、pytest、PyYAML、Codex CLI、Unix domain socket / `SO_PEERCRED`、`flock`、Git worktree、SRT/bubblewrap。

**Spec:** `docs/superpowers/specs/2026-10-16-greedylore-autoresearch-dynamic-agent-design.md`

## Global Constraints

- 当前实现基线为 `bcbb249`–`8834bd6`；本计划只描述尚未实现的增量。
- 原计划 `docs/superpowers/plans/2026-10-15-greedylore-autoresearch-coral-implementation.md` 保留，不重写已完成工作。
- 不修改固定 Evaluation Contract、quality/efficiency workload、GPU资源预留、分布式进程清理、同 SHA infra retry、dense references或Root正式评测语义。
- Tree Node只表示Research单元：每个普通 Node有且仅有一个Researcher Agent和一个正式Research Attempt；Engineer是该 Node可选的独立Agent，不创建新Node。
- Node ID与Agent ID单调分配且永不复用；允许取消reservation留下编号空洞。
- 所有共享workflow mutation使用同一个`.coral/private/trusted_writes.lock`；取锁后必须重新读取，不以缓存状态分配ID。
- `tree.md`、`evaluation.md`和Summary是派生Artifact；Git、Attempts、eval logs、Node Spec和versioned Analysis保留研究事实。
- 默认静态CORAL agents完全兼容；V1动态模式仅支持单island并在其他配置下明确拒绝。
- Subagent是Researcher、Engineer和Analyzer遇到实质不确定性时的可选能力，不是固定阶段。
- 不新增timer、heartbeat action、常驻Memory服务、candidate pool、LLM target scheduler或通用workflow engine。
- 只运行无GPU单元/集成测试；不执行训练、dense calibration、Root Evaluation或真实Memory Codex session。
- 每个Task通过后单独commit；commit本身不触发push、训练或外部发布。

---

### Task 1: 收敛 Analyzer-only workflow 与公开 Artifact 规则

**Files:**
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/analysis.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/validation.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/nodes.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/cli.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/submission.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/artifacts.py`
- Delete: `tasks/greedylore_autoresearch/autoresearch/schemas/review.schema.json`
- Modify: `tasks/greedylore_autoresearch/templates/node_spec.md`
- Delete: `tasks/greedylore_autoresearch/templates/review.md`
- Modify: `tasks/greedylore_autoresearch/roles/workflow_rules.md`
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
- Consumes: existing `resolve_evidence(node_id, mode) -> EvidenceBundle`, `run_fresh_codex(...) -> Path`, deterministic evaluation rebuild and trusted `publish_artifact`.
- Produces: `AnalysisResult(evidence, analysis_version, analysis_path)` and `analyze(node_id, mode, revise=False) -> AnalysisResult`; Child spec只固定Parent Analysis refs，不再存在Review schema/session/file/decision。

- [ ] **Step 1: Rewrite Analyzer tests**

```python
def test_analyze_runs_one_fresh_session(context):
    runner = FakeCodex(context, ["analysis"])
    result = analyze("N001", "research", context=context, runner=runner,
                     publisher=FakePublisher(context))
    assert runner.analysis_calls == 1
    assert result.analysis_path.name == "research_analysis_v1.md"
    assert not list(result.analysis_path.parent.glob("*_review_v*.md"))


def test_completed_analysis_is_idempotent(context):
    runner = FakeCodex(context, ["analysis"])
    first = analyze("N001", "research", context=context, runner=runner,
                    publisher=FakePublisher(context))
    second = analyze("N001", "research", context=context, runner=runner,
                     publisher=FakePublisher(context))
    assert first.analysis_version == second.analysis_version == 1
    assert runner.analysis_calls == 1


def test_revise_allocates_next_analysis_version(context):
    publish_valid_analysis(context, mode="research", version=2)
    result = analyze("N001", "research", revise=True, context=context,
                     runner=FakeCodex(context, ["analysis"]),
                     publisher=FakePublisher(context))
    assert result.analysis_version == 3
```

Delete tests whose only contract is Reviewer recovery, decision or paired versioning. Preserve fixed evidence snapshots, zero/multiple Engineering Attempts, immutable code snapshots and evidence-content validation.

- [ ] **Step 2: Rewrite Node and Artifact tests**

```python
def test_child_requires_fixed_parent_research_analysis(notes_root):
    evidence = parent_evidence(notes_root, analysis_version=2)
    spec = yaml.safe_load(create_node(child_request(), evidence).read_text())
    assert spec["analysis_refs"]["parent_research"] == (
        "nodes/N001/research_analysis_v2.md"
    )
    assert "review_refs" not in spec


def test_review_artifact_kind_is_removed():
    with pytest.raises(ValueError, match="unsupported artifact kind"):
        validate_artifact("review", "{}", {
            "node_id": "N001", "mode": "research", "version": 1,
        })
```

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
uv run --frozen --extra dev pytest \
  tasks/greedylore_autoresearch/tests/test_analysis_cli.py \
  tasks/greedylore_autoresearch/tests/test_analysis_recovery.py \
  tasks/greedylore_autoresearch/tests/test_node_rules.py \
  tasks/greedylore_autoresearch/tests/test_submission.py \
  tasks/greedylore_autoresearch/tests/test_root_bootstrap.py \
  tasks/greedylore_autoresearch/tests/test_handoff_assets.py -q
```

Expected: FAIL because current code still starts Reviewer and requires Review refs.

- [ ] **Step 4: Implement one-stage Analysis**

`AnalysisResult`只保留`evidence`、`analysis_version`和`analysis_path`。默认返回最新有效Analysis；没有版本时创建v1；`--revise`创建`max(existing)+1`。执行顺序固定为refresh evaluation → resolve evidence → existing-valid check或一次fresh Analyzer → `seal_document` → `validate_analysis` → trusted publication。

删除`validate_review()`、review schema lookup、第二scratch/session和decision输出。CLI JSON只打印：

```json
{"analysis_version": 2, "analysis_path": "/notes/nodes/N001/research_analysis_v2.md"}
```

- [ ] **Step 5: Remove Review dependencies from Node/submission/assets**

Node Spec只保留`analysis_refs.parent_research`和可选`parent_engineering`。Research Analysis存在且Node匹配即可作为Child evidence。更新Root、submission provenance、templates和validators；旧run中的Review文件可以读取，但新workflow不引用或创建。

在Research/Engineering Analyzer prompts中加入一致的可选Subagent规则：

```text
遇到实质性不确定性时，可以自主调用 Subagent 进行独立检查或辅助分析。
简单问题应优先自行解决，不为常规修改或结果强制调用 Subagent。每个
Subagent 必须有明确、有限的任务与输出，并继承当前 Sandbox、Evaluation
和 Artifact 发布边界。
```

不要修改`algorithm_ref.md`、`system_ref.md`或`semantic_criteria.md`。

- [ ] **Step 6: Verify GREEN and commit**

Run Step 3 command. Expected: PASS.

```bash
git add -A tasks/greedylore_autoresearch/autoresearch \
  tasks/greedylore_autoresearch/prompts \
  tasks/greedylore_autoresearch/templates \
  tasks/greedylore_autoresearch/roles \
  tasks/greedylore_autoresearch/tests
git commit -m "feat: simplify autoresearch to analyzer-only workflow"
```

---

### Task 2: 建立原子 Workflow State 与固定可信操作

**Files:**
- Create: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/state.py`
- Create: `tasks/greedylore_autoresearch/tests/test_workflow_state.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/nodes.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/bootstrap.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/cli.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/submission.py`
- Modify: `tasks/greedylore_autoresearch/autoresearch/src/autoresearch/artifacts.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_node_rules.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_root_bootstrap.py`
- Modify: `tasks/greedylore_autoresearch/tests/test_submission.py`
- Modify: `tasks/greedylore_autoresearch/task.yaml`
- Modify: `coral/grader/trusted_writes.py`
- Modify: `coral/grader/trusted_worker.py`
- Modify: `tests/test_trusted_writes.py`

**Interfaces:**
- Produces `WorkflowAssignment(agent_id, role, node_id, status)`.
- Produces `WorkflowStore(coral_dir, island_id=None)` with:
  - `reconcile(max_researchers: int, max_engineers: int) -> tuple[WorkflowAssignment, ...]`
  - `initialize_research(agent_id: str, node_id: str, request: Mapping[str, Any]) -> dict[str, Any]`
  - `claim_engineering(agent_id: str, node_id: str) -> dict[str, Any]`
  - `complete_assignment(agent_id: str, mode: str) -> dict[str, Any]`
  - `cancel_assignment(agent_id: str) -> dict[str, Any]`
- Produces `handle_workflow(operation: str, payload: dict, context: dict) -> dict` configured as `grader.args.trusted_workflow_handler`.
- Produces `authorize_attempt(canonical: dict, context: dict) -> dict` configured as `grader.args.trusted_workflow_authorizer`; this read-only callback runs while the existing trusted Attempt lock is held.
- Extends trusted operations only with `reconcile_workflow`, `initialize_research`, `claim_engineering`, `complete_assignment`, `cancel_assignment`.

- [ ] **Step 1: Write state allocation and recovery tests**

```python
def test_concurrent_reconcile_allocates_each_node_once(store):
    def call_reconcile(_):
        independent = WorkflowStore(store.coral_dir)
        return independent.reconcile(2, 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        batches = list(pool.map(call_reconcile, range(2)))
    state = read_state(store)
    assert set(state.nodes) == {"N000", "N001", "N002"}
    assert state.next_node_seq == 3
    assert all({item.node_id for item in batch} == {"N001", "N002"}
               for batch in batches)


def test_cancelled_node_id_is_not_reused(store):
    first = store.reconcile(1, 0)[0]
    store.cancel_assignment(first.agent_id)
    second = store.reconcile(1, 0)[0]
    assert (first.node_id, second.node_id) == ("N001", "N002")


def test_reconcile_after_lost_response_returns_existing_reservation(store):
    first = store.reconcile(1, 0)
    second = WorkflowStore(store.coral_dir).reconcile(1, 0)
    assert second == first


def test_root_is_registered_once_before_child_allocation(store, root_artifacts):
    first = store.reconcile(1, 0)
    second = store.reconcile(1, 0)
    assert "N000" in read_state(store).nodes
    assert read_state(store).nodes["N000"]["research"]["status"] == "completed"
    assert first == second
    assert first[0].node_id == "N001"


def test_no_child_is_reserved_before_root_analysis(store, root_without_analysis):
    assert store.reconcile(2, 1) == ()
    assert read_state(store).next_node_seq == 1


def test_missing_state_in_existing_run_refuses_id_allocation(store, existing_n001):
    store.state_path.unlink()
    with pytest.raises(WorkflowError, match="state is missing"):
        store.reconcile(2, 1)
```

Use a real `flock` and separate `WorkflowStore` instances so tests do not pass only because of an in-process mutex.

- [ ] **Step 2: Write Engineering claim race and eligibility tests**

```python
def test_only_one_engineer_claims_a_node(store, completed_quality_node):
    workers = seed_engineering_workers(store, count=2)
    outcomes = concurrently(
        lambda worker: store.claim_engineering(worker.agent_id, completed_quality_node),
        workers,
    )
    assert sum(outcome.ok for outcome in outcomes) == 1


@pytest.mark.parametrize("missing", [
    "research_complete", "quality_pass", "code_snapshot", "unclaimed",
])
def test_engineering_claim_rechecks_every_eligibility_fact(store, node, missing):
    break_eligibility(node, missing)
    with pytest.raises(WorkflowError, match="not eligible"):
        store.claim_engineering("engineer-E0001", node.node_id)
```

Also test that V1 policy `max_engineering_runs_per_node=1` rejects a second completed flow while the JSON schema stores an appendable `engineering_claims` list.

- [ ] **Step 3: Write transition, provenance and projection tests**

Cover ownership, evidence and projection with concrete assertions:

```python
def test_initialize_requires_reserved_owner(store, reservation, request):
    with pytest.raises(WorkflowError, match="owner"):
        store.initialize_research("research-N999", reservation.node_id, request)
    result = store.initialize_research(
        reservation.agent_id, reservation.node_id, request
    )
    assert result["status"] == "active"
    assert result["spec_ref"] == f"nodes/{reservation.node_id}/spec.md"


def test_research_complete_requires_evidence(store, active_research):
    with pytest.raises(WorkflowError, match="Research Attempt"):
        store.complete_assignment(active_research.agent_id, "research")
    publish_terminal_research_attempt(active_research, quality_pass=False)
    publish_research_analysis(active_research)
    result = store.complete_assignment(active_research.agent_id, "research")
    assert result["status"] == "completed"
    assert not store.is_engineering_eligible(active_research.node_id)


def test_engineering_complete_requires_attempt_and_analysis(store, claimed_engineer):
    with pytest.raises(WorkflowError, match="Engineering Attempt"):
        store.complete_assignment(claimed_engineer.agent_id, "engineering")
    publish_terminal_engineering_attempt(claimed_engineer, speedup=-0.1)
    publish_engineering_analysis(claimed_engineer)
    assert store.complete_assignment(
        claimed_engineer.agent_id, "engineering"
    )["status"] == "completed"


def test_completed_owner_cannot_publish_another_attempt(service, binding):
    complete_research_assignment(binding.agent_id)
    with pytest.raises(TrustedWriteError, match="completed"):
        service.handle_authenticated(
            "submit_attempt", new_research_payload(binding), binding
        )


def test_tree_projection_rebuilds_from_state_and_artifacts(store):
    complete_research_with_analysis(store, "N001")
    store.tree_path.unlink()
    store.rebuild_tree()
    assert "N001" in store.tree_path.read_text()
```

Also test that pending Attempts and retryable infra failures prevent completion. Engineering completion accepts a terminal negative-speedup evaluation. State stores only refs, ownership and status, not copied metrics.

- [ ] **Step 4: Write fixed trusted-operation authorization tests**

```python
def test_only_manager_peer_can_reconcile(service, manager_pid, agent_binding):
    assert service.handle_peer(
        "reconcile_workflow", capacities(), peer_pid=manager_pid
    )["assignments"]
    with pytest.raises(TrustedWriteError):
        service.handle_authenticated("reconcile_workflow", capacities(), agent_binding)


def test_agent_can_only_mutate_its_assignment(service, researcher_binding):
    with pytest.raises(TrustedWriteError, match="owner"):
        service.handle_authenticated(
            "initialize_research", init_payload(node_id="N999"), researcher_binding
        )
```

Test that unknown fields, arbitrary paths, caller-supplied agent IDs/status/counters and unconfigured workflow handlers are rejected. Existing three trusted operations remain unchanged.

- [ ] **Step 5: Run focused tests and verify RED**

Run:

```bash
uv run --frozen --extra dev pytest \
  tasks/greedylore_autoresearch/tests/test_workflow_state.py \
  tasks/greedylore_autoresearch/tests/test_node_rules.py \
  tasks/greedylore_autoresearch/tests/test_submission.py \
  tests/test_trusted_writes.py -q
```

Expected: FAIL because state store and workflow operations do not exist and `_next_node_id()` still scans directories.

- [ ] **Step 6: Implement the minimal authoritative store**

Store `.coral/private/autoresearch_state.json` with `version`、`revision`、`next_node_seq`、`next_engineer_seq`、`nodes`和`engineering_workers`。Every mutation:

1. takes `.coral/private/trusted_writes.lock` with exclusive `flock`;
2. reloads and validates the JSON after locking;
3. checks owner and current-state preconditions;
4. mutates and increments revision;
5. writes temp + flush/fsync + replace + directory fsync;
6. rebuilds `tree.md` from committed state and immutable Artifacts.

`reconcile()` first verifies and registers the existing `N000` Root spec, terminal Research Attempt and Research Analysis as a fixed completed Root record. Repeat calls are idempotent; conflicting Root evidence is an error. Until Root Analysis exists, it returns no ordinary assignments. It then atomically fills missing Research slots. It creates unbound Engineer workers only when eligible unclaimed Node count exceeds active/reserved workers, but never chooses a target. Delete `_next_node_id()` and reject caller-selected IDs. The existing `bootstrap_root()` remains responsible for creating A0 and the Root Artifact; it never allocates an ordinary Node ID.

`initialize_research()` validates Parent Research Analysis and writes the immutable `nodes/Nxxx/spec.md` directly with the existing task-local atomic text helper while holding the workflow lock, then marks active. It must not call `publish_artifact` through the socket from inside the lock. Retry with byte-identical content is idempotent; a spec-written/reserved-state interruption is repaired by repeating the same call. `tree.md` is regenerated after the state commit and may be repaired by the next reconcile if that projection write is interrupted.

- [ ] **Step 7: Extend the existing trusted socket without a new control plane**

Read the request envelope before binding resolution. For `reconcile_workflow`, require `peer_pid == os.getppid()` in the grader daemon; for Agent operations, use existing registry/descendant resolution. Dispatch only the five fixed operations through the configured task handler using grader-private Python. Context supplies canonical `coral_dir`、authenticated `agent_id`、`island_id`、`is_manager` and the capacities from trusted config; payload cannot override them. The task handler owns `trusted_writes.lock` for state mutations. The service must not hold that lock while calling the task handler, and the handler must not call the socket publisher while holding it.

Do not add a second socket, dynamic Provider, arbitrary operation names or path routing. Keep the 64 KiB bound and existing server lifecycle.

- [ ] **Step 8: Bind Attempts and CLI commands to ownership**

The existing `_submit()` keeps its pre-lock commit/schema validation, then calls the configured task-local ownership authorizer **inside** `trusted_writes.lock` immediately before Attempt creation. The authorizer only reads state; it never acquires the lock again. It checks:

- research mode: authenticated Agent is the Node's active Researcher;
- engineering mode: authenticated Agent owns the Node's active Engineering claim;
- completed/cancelled/stale assignments cannot submit.

Replace free Node creation with:

```bash
autoresearch node initialize --parent N001 --hypothesis "stable gradients" \
  --selection-rationale "N001 quality passes" --delta "gate compression" \
  --risk "slower convergence" --falsification-criteria "quality gate fails"
autoresearch engineering claim N001
autoresearch node complete --mode research
autoresearch node complete --mode engineering
```

Before initialize/claim changes Git base, CLI requires `git status --porcelain` empty. Research initialize positions the Worktree at Parent Research Commit; Engineering claim positions it at the target Research Commit. Trusted state remains reserved/claimed if a later local checkout fails, so the same Agent can repair and resume.

- [ ] **Step 9: Verify GREEN and commit**

Run Step 5 command. Expected: PASS.

```bash
git add coral/grader/trusted_writes.py coral/grader/trusted_worker.py \
  tests/test_trusted_writes.py tasks/greedylore_autoresearch
git commit -m "feat: add atomic autoresearch workflow state"
```

---

### Task 3: 复用 AgentManager 实现 Node-bound 动态生命周期

**Files:**
- Create: `coral/agent/dynamic.py`
- Modify: `coral/config.py`
- Modify: `coral/agent/assignments.py`
- Modify: `coral/agent/manager.py`
- Modify: `coral/template/coral_md.py`
- Modify: `coral/workspace/worktree.py`
- Modify: `coral/workspace/breadcrumbs.py`
- Create: `tests/test_dynamic_agents.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_assignments.py`
- Modify: `tests/test_manager_reliability.py`
- Modify: `tests/test_workspace.py`
- Modify: `tasks/greedylore_autoresearch/task.yaml`
- Modify: `tasks/greedylore_autoresearch/roles/researcher.md`
- Modify: `tasks/greedylore_autoresearch/roles/engineer.md`
- Modify: `tasks/greedylore_autoresearch/roles/workflow_rules.md`
- Modify: `tasks/greedylore_autoresearch/tests/test_handoff_assets.py`

**Interfaces:**
- Produces `DynamicAgentConfig(enabled=False, max_researchers=1, max_engineers=0)`.
- Adds optional `AgentAssignmentConfig.role: str = ""`; dynamic task requires exactly one `research` and one `engineering` template.
- Extends `AgentSpec` with `workflow_role: str | None` and `node_id: str | None`.
- Produces `DynamicAssignmentClient.reconcile() -> tuple[WorkflowAssignment, ...]` using existing `request_trusted()`; trusted server supplies capacities from config.
- Produces `retire_agent_worktree(repo_path: Path, worktree_path: Path) -> bool`; never force-deletes dirty worktrees.
- Produces `AgentManager._reconcile_dynamic_agents() -> None`, used by start, resume and monitor.

- [ ] **Step 1: Write config/template compatibility tests**

```python
def test_dynamic_config_uses_role_templates():
    cfg = CoralConfig.from_dict(dynamic_task_config(2, 1))
    assert cfg.agents.dynamic.max_researchers == 2
    assert cfg.agents.dynamic.max_engineers == 1
    assert [item.role for item in cfg.agents.assignments] == [
        "research", "engineering",
    ]


def test_static_assignments_keep_existing_semantics():
    cfg = CoralConfig.from_dict(static_task_config(count=2))
    assert len(resolve_agent_specs(cfg)) == 2


def test_dynamic_multi_island_is_rejected():
    with pytest.raises(ValueError, match="single island"):
        CoralConfig.from_dict(dynamic_task_config(2, 1, islands=2))
```

- [ ] **Step 2: Write dynamic start and capacity tests**

```python
def test_manager_starts_exact_desired_assignments(manager, state_client):
    state_client.assignments = [
        assignment("research-N001", "research", "N001"),
        assignment("research-N002", "research", "N002"),
        assignment("engineer-E0001", "engineering", None),
    ]
    manager._reconcile_dynamic_agents()
    assert {handle.agent_id for handle in manager.handles} == {
        "research-N001", "research-N002", "engineer-E0001",
    }
    assert state_client.last_capacities == (2, 1)
```

Assert every Agent gets its own Worktree, log, registry binding and fresh session; assignment context written into generated instructions includes only role/node/commands, not private state or locks.

- [ ] **Step 3: Write completion, crash and resume tests**

```python
def test_completed_agent_is_retired_and_slot_is_refilled(manager, state_client):
    start_node(manager, "research-N001")
    state_client.assignments = [assignment("research-N002", "research", "N002")]
    manager._reconcile_dynamic_agents()
    assert ids(manager.handles) == {"research-N002"}
    assert manager.restart_count("research-N001") == 0


def test_crashed_active_agent_resumes_same_session(manager, state_client):
    handle = start_node(manager, "research-N001", session_id="session-1")
    handle.mark_dead(exit_code=1)
    manager.monitor_once()
    assert ids(manager.handles) == {"research-N001"}
    assert manager.last_resume_session("research-N001") == "session-1"


def test_resume_ignores_completed_old_worktrees(manager, paths, state_client):
    old = provision_worktree(paths, "research-N001")
    state_client.assignments = [assignment("research-N002", "research", "N002")]
    manager.resume_all(paths)
    assert ids(manager.handles) == {"research-N002"}
    assert old not in {handle.worktree_path for handle in manager.handles}
```

Also cover circuit-breaker pause, manager restart with reserved-before-initialize Node, active Engineering claim recovery and completed Engineering non-restart.

- [ ] **Step 4: Write safe retirement tests**

```python
def test_clean_completed_worktree_is_removed_but_branch_survives(repo, worktree):
    branch = current_branch(worktree)
    assert retire_agent_worktree(repo, worktree) is True
    assert not worktree.exists()
    assert git_branch_exists(repo, branch)


def test_dirty_worktree_is_left_for_diagnostics(repo, worktree):
    (worktree / "uncommitted.txt").write_text("keep me")
    assert retire_agent_worktree(repo, worktree) is False
    assert worktree.exists()
```

Retirement failure must not resurrect a completed assignment or block filling the slot.

- [ ] **Step 5: Run focused tests and verify RED**

Run:

```bash
uv run --frozen --extra dev pytest \
  tests/test_config.py tests/test_assignments.py \
  tests/test_dynamic_agents.py tests/test_manager_reliability.py \
  tests/test_workspace.py \
  tasks/greedylore_autoresearch/tests/test_handoff_assets.py -q
```

Expected: FAIL because CORAL currently expands static specs only and restarts every dead handle.

- [ ] **Step 6: Add dynamic config and templates without changing static mode**

Add `agents.dynamic` and assignment `role`. In static mode, `resolve_agent_specs()` and nickname behavior remain byte-for-byte compatible. In dynamic mode, assignments become immutable runtime/model/options templates; `count` is not used for concurrency, and the explicit max fields are authoritative.

`DynamicAssignmentClient` requests `reconcile_workflow` with an empty payload over existing UDS and validates the bounded response schema before building `AgentSpec`; the trusted service supplies capacities from its loaded config. Task responses cannot supply runtime, model, role file, mounts or Sandbox options. The manager must start grader/trusted endpoint before its first reconcile. If no Root Analysis is published, the desired roster is empty.

- [ ] **Step 7: Integrate reconcile into AgentManager**

Refactor setup so grader and trusted server start before the first dynamic reconcile. `_reconcile_dynamic_agents()` diffs desired assignment IDs against current handles/specs:

- missing desired: instantiate from the trusted role template and call existing `_setup_and_start_agent()`;
- existing desired: preserve handle, Worktree and Session;
- no longer desired: stop handle, revoke registry entry, remove in-memory spec/runtime maps and attempt safe Worktree retirement;
- desired dead: let existing `_restart_agent()` and circuit breaker resume it.

Call reconcile at initial start, resume, after a terminal Attempt, after an Agent exit and when workflow state revision changes. Update PID/registry/state files after every batch, not between individual additions/removals.

In dynamic resume, authoritative assignments determine which Agent IDs return. Existing Worktrees are implementation resources, not the desired roster. Static resume keeps current directory discovery.

- [ ] **Step 8: Update role workflows and task configuration**

Configure:

```yaml
grader:
  args:
    trusted_workflow_handler: autoresearch.state:handle_workflow
    trusted_workflow_authorizer: autoresearch.state:authorize_attempt
    max_engineering_runs_per_node: 1
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

Researcher prompt: choose Parent/hypothesis → initialize → implement → one Research eval → research analyze → complete → exit. Engineer prompt: autonomously inspect global evidence → claim atomically → optimize fixed Research Commit → Engineering eval(s) → engineering analyze → complete → exit. Claim conflict means reselect; no candidates means cancel and exit.

Both prompts read the appropriate Summary when present and include the same optional Subagent rule. They never mention UDS、flock、private state schema或grader run internals。

- [ ] **Step 9: Verify GREEN and commit**

Run Step 5 command. Expected: PASS.

```bash
git add coral/config.py coral/agent coral/template/coral_md.py \
  coral/workspace/worktree.py coral/workspace/breadcrumbs.py \
  tests/test_config.py tests/test_assignments.py tests/test_dynamic_agents.py \
  tests/test_manager_reliability.py tests/test_workspace.py \
  tasks/greedylore_autoresearch
git commit -m "feat: add node-bound dynamic agent lifecycle"
```

---

### Task 4: 添加 Researcher 启动触发的 Global Memory 并完成并发验收

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
- Modify: `tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py`
- Create: `coral/agent/startup_hook.py`
- Modify: `coral/agent/builtin/codex.py`
- Create: `tests/test_codex_startup_hook.py`
- Modify: `coral/grader/trusted_writes.py`
- Modify: `tests/test_trusted_writes.py`
- Modify: `coral/sandbox/srt.py`
- Modify: `tests/test_sandbox.py`

**Interfaces:**
- Produces `list_analysis_refs(notes_root, mode=None) -> tuple[str, ...]`.
- Produces `read_processed_analyses(summary) -> frozenset[str]`.
- Produces `memory_due(notes_root, threshold=3) -> MemoryGate`, where `MemoryGate.due` and `MemoryGate.trigger_research_refs` are immutable.
- Produces `validate_summary(content, notes_root, required_refs=frozenset()) -> frozenset[str]`.
- Produces `consolidate_memory(*, if_due, context=None, runner=run_fresh_codex, publisher=None) -> MemoryResult`.
- Adds CLI `autoresearch consolidate-memory [--if-due] [--async]`.
- Adds Codex reserved `runtime_options.startup_hook: list[str]` and Artifact kinds `research_summary` / `engineering_summary`.

- [ ] **Step 1: Write gate and minimal completeness tests**

```python
def test_gate_counts_unprocessed_research_nodes_only(notes_root):
    publish_analysis(notes_root, "N001", "research", 1)
    publish_analysis(notes_root, "N001", "research", 2)
    publish_analysis(notes_root, "N002", "research", 1)
    publish_analysis(notes_root, "N003", "engineering", 1)
    assert memory_due(notes_root).due is False
    publish_analysis(notes_root, "N003", "research", 1)
    gate = memory_due(notes_root)
    assert gate.due is True
    assert node_ids(gate.trigger_research_refs) == {"N001", "N002", "N003"}


def test_both_summaries_must_cover_trigger_refs(notes_root):
    refs = publish_three_research_analyses(notes_root)
    with pytest.raises(MemoryError, match="trigger references"):
        validate_summary(valid_summary(refs[:-1]), notes_root,
                         required_refs=frozenset(refs))
```

The completeness check compares canonical filenames only. Python must not read Analysis bodies, compute hashes or judge Summary semantics.

- [ ] **Step 2: Write worker, idempotency and recovery tests**

```python
def test_manager_publishes_two_fixed_summaries(memory_context):
    runner = FakeMemoryRunner(valid_memory_output(memory_context.analysis_refs))
    publisher = RecordingPublisher()
    result = consolidate_memory(if_due=True, context=memory_context,
                                runner=runner, publisher=publisher)
    assert runner.calls == 1
    assert [call.kind for call in publisher.calls] == [
        "research_summary", "engineering_summary",
    ]
    assert result.status == "published"


def test_processed_set_skips_manager(memory_context):
    write_both_summaries(memory_context.notes_root, memory_context.analysis_refs)
    runner = FakeMemoryRunner()
    assert consolidate_memory(if_due=True, context=memory_context,
                              runner=runner,
                              publisher=RecordingPublisher()).status == "not_due"
    assert runner.calls == 0
```

Mock `subprocess.Popen` and assert `--async` starts a hidden worker with `start_new_session=True`, returns immediately, and the worker—not launcher—takes `LOCK_EX | LOCK_NB`. Occupied lock returns `locked`; runner/publisher failure leaves Analysis、Evaluation和Reference unchanged; next check remains due.

- [ ] **Step 3: Write startup Hook, fixed-target and Sandbox tests**

```python
def test_only_research_template_has_memory_hook(task_yaml):
    templates = {item["role"]: item for item in task_yaml["agents"]["assignments"]}
    assert templates["research"]["runtime_options"]["startup_hook"] == [
        ".venv/bin/autoresearch", "consolidate-memory", "--if-due", "--async",
    ]
    assert "startup_hook" not in templates["engineering"]["runtime_options"]


def test_hook_failure_still_execs_runtime(monkeypatch):
    calls = record_hook_and_exec(monkeypatch, hook_returncode=1)
    run_startup_hook(["hook"], ["codex", "exec"])
    assert calls == [("hook",), ("exec", ("codex", "exec"))]
```

Trusted writer tests require two exact replaceable targets. SRT denies direct writes to both Summary paths while the run-scoped `memory-consolidation.lock` remains writable. No arbitrary notes-root target is accepted.

- [ ] **Step 4: Run focused tests and verify RED**

Run:

```bash
uv run --frozen --extra dev pytest \
  tasks/greedylore_autoresearch/tests/test_memory.py \
  tasks/greedylore_autoresearch/tests/test_analysis_cli.py \
  tasks/greedylore_autoresearch/tests/test_handoff_assets.py \
  tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py \
  tests/test_codex_startup_hook.py tests/test_trusted_writes.py \
  tests/test_sandbox.py tests/test_dynamic_agents.py -q
```

Expected: FAIL because Memory CLI, Summary targets and startup Hook do not exist.

- [ ] **Step 5: Implement file-reference gate and independent worker**

Accept only `nodes/Nddd/(research|engineering)_analysis_vN.md`. The processed baseline is the intersection of both Summary lists so partial publication never suppresses retry. Gate deduplicates unprocessed Research refs by Node; Engineering refs never count.

At worker start, acquire `<island_root>/memory-consolidation.lock`, recompute the gate and freeze `trigger_research_refs`. Run one ephemeral read-only Codex session. The prompt delegates discovery of new/revised Analyses, evidence selection, semantic deduplication, conflict handling and uncertainty. Output is exactly:

```json
{
  "research_summary": "<Markdown with processed_analyses frontmatter>",
  "engineering_summary": "<Markdown with processed_analyses frontmatter>"
}
```

Validate both outputs before publishing either: nonempty Markdown, canonical existing refs, and both processed lists covering the frozen trigger refs. Do not require all Engineering refs or interpret prose.

- [ ] **Step 6: Add fixed Summary publication and startup Hook**

`validate_artifact()` maps Summary kinds to fixed root filenames only when node/mode/version are null. They are replaceable by temp/fsync/replace; Analysis and spec remain immutable.

Codex runtime removes `startup_hook` before building Codex `-c` options and wraps the runtime command with `python -m coral.agent.startup_hook --hook-json '<argv JSON>' -- '<codex argv>'`. The wrapper executes argv without shell, logs failure, and always `execvpe()`s the runtime. Every dynamic Researcher start/resume/restart therefore checks Memory; Engineering never does.

The detached bounded worker may fail if the owning Researcher dies before publication; it performs no retry loop. The unchanged processed refs keep the gate due, and the next start/restart retries. Tests must prove launcher exit does not terminate the worker and failure is recoverable; do not claim a daemon-like lifecycle.

- [ ] **Step 7: Add Memory prompt and verify GREEN**

`global_memory_manager.md` identifies Summaries as derived navigation, preserves Node/Analysis citations、applicability和uncertainty, emits full processed refs, and forbids modifying Reference、Analysis、Evaluation或commits。Asset tests hash or compare the three static Reference files before/after fake consolidation.

Run Step 4 command. Expected: PASS.

- [ ] **Step 8: Run race-focused and full non-GPU regression**

Run:

```bash
uv run --frozen --extra dev pytest \
  tests/test_dynamic_agents.py tests/test_manager_reliability.py \
  tests/test_trusted_writes.py tests/test_codex_startup_hook.py \
  tests/test_sandbox.py \
  tasks/greedylore_autoresearch/tests/test_workflow_state.py \
  tasks/greedylore_autoresearch/tests/test_node_rules.py \
  tasks/greedylore_autoresearch/tests/test_submission.py \
  tasks/greedylore_autoresearch/tests/test_analysis_cli.py \
  tasks/greedylore_autoresearch/tests/test_analysis_recovery.py \
  tasks/greedylore_autoresearch/tests/test_evaluation_index.py \
  tasks/greedylore_autoresearch/tests/test_memory.py \
  tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py -q
```

Then run:

```bash
uv run --frozen --extra dev pytest -q
```

Expected: PASS without real Codex sessions, GPU training, timer or background service.

- [ ] **Step 9: Commit**

```bash
git add coral/agent/startup_hook.py coral/agent/builtin/codex.py \
  coral/grader/trusted_writes.py coral/sandbox/srt.py \
  tests/test_codex_startup_hook.py tests/test_trusted_writes.py tests/test_sandbox.py \
  tests/test_dynamic_agents.py tasks/greedylore_autoresearch
git commit -m "feat: add startup-triggered autoresearch memory"
```

---

## Final Verification

- [ ] Run: `git diff --check`
- [ ] Run: `uv run --frozen --extra dev ruff check coral tests tasks/greedylore_autoresearch/autoresearch`
- [ ] Run every focused command from Tasks 1–4 exactly as listed.
- [ ] Run: `uv run --frozen --extra dev pytest -q`
- [ ] Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests -q`
- [ ] Run: `uv run --frozen --extra dev coral validate tasks/greedylore_autoresearch --json`
- [ ] Confirm two concurrent reconcile calls cannot allocate duplicate Node IDs.
- [ ] Confirm two Engineer claims for one Node yield exactly one winner.
- [ ] Confirm completed Agent is retired/refilled while crashed active Agent resumes the same Session.
- [ ] Confirm static CORAL assignments retain their original spawn/resume behavior.
- [ ] Confirm the diff contains no Evaluation Contract、grader workload、GPU protocol、dense reference或Root evaluation changes.
- [ ] Confirm no test starts a real Codex session、GPU workload、timer或background scheduling service.

## Execution Order

Tasks must run in order: Task 1 removes Review assumptions used by Node provenance; Task 2 establishes authoritative ownership; Task 3 lets AgentManager consume that ownership; Task 4 relies on per-Node Researcher starts for reliable low-frequency Memory checks. Each Task is independently reviewed and committed before the next begins.
