# GreedyLoRE AutoResearch V1 Workflow Simplification — Incremental Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use subagents only when explicitly requested and useful for a bounded independent check. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前已实现的 GreedyLoRE AutoResearch V1 上，移除固定 Reviewer 阶段，并增加由 Researcher 启动 Hook 低频触发的 Global Memory Manager。

**Architecture:** 以当前分支已有的 Commit → Attempt → Grader、可信 Artifact 发布、Node/Evaluation/Analysis 和 SRT 隔离为基线，只修改 workflow、Prompt 和启动 Hook。Analysis 变为单阶段 Analyzer；Memory Manager 由 Researcher start/resume/restart 时异步检查三 Research Node 门槛，并通过现有可信入口更新两份 Summary。

**Tech Stack:** Python 3.11、pytest、PyYAML、Codex CLI、CORAL Codex runtime、Unix `flock`、SRT/bubblewrap。

**Spec:** `docs/superpowers/specs/2026-10-15-greedylore-autoresearch-coral-design.md` 中的 Workflow Simplification 与 Global Memory 章节。

## Baseline and Scope

- 当前实现基线包括 `bcbb249` 至 `8834bd6`：Trusted Attempt/Retry、评测 primitives、GreedyLoRE grader、Node/Evaluation、Analysis/Reviewer、角色资产和 Sandbox 均已存在，不在本计划中重建。
- 不修改固定 Evaluation Contract、quality/efficiency workload、GPU 预留、分布式进程清理、同 SHA retry 或 Root bootstrap 语义。
- 不运行 GPU training、dense calibration 或正式 Root Evaluation；本计划只运行无 GPU 单元/集成测试。
- 删除 Reviewer 后，Research Analysis 有效发布即可创建 Child；Engineering Analysis 仍可缺席且不阻塞 Researcher。
- Subagent 仅是 Researcher、Engineer 和 Analyzer 遇到实质不确定性时的可选能力，不建立 Manager、固定触发或额外 Artifact。
- Global Memory 只由 Researcher start/resume/automatic restart Hook 调用 `autoresearch consolidate-memory --if-due --async`；Engineer、`autoresearch analyze` 和 Analysis 发布均不触发。
- `--if-due` 只统计两份 Summary 尚未共同处理的 Research Analysis 所属不同 Node；达到 3 个 Node 才启动独立 Memory session。Engineering Analysis 不计数，但纳入整理输入。
- Summary frontmatter 的完整 `processed_analyses` 列表是唯一幂等依据；Python 不构建 Evidence Bundle、不计算 Evidence Hash、不比较 Analysis 内容。
- Reference Toolkit 保持静态。Memory worker 不修改 Node Analysis、Evaluation、Reference 或 Git commit。
- 防御强度与现有 CORAL 一致：覆盖正常科研场景下的固定目标、格式、锁和写边界，不增加通用权限框架、调度服务或低概率攻击状态机。
- 每个 Task 测试通过后单独 commit；commit 不触发训练、Memory session 或外部发布。

---

### Task 1: 收敛 Analyzer-only Node workflow 与角色资产

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
  `run_fresh_codex(...) -> Path`, deterministic evaluation rebuild and the existing trusted
  `publish_artifact` operation.
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


def test_review_artifact_kind_is_removed():
    with pytest.raises(ValueError, match="unsupported artifact kind"):
        validate_artifact(
            "review", "{}",
            {"node_id": "N001", "mode": "research", "version": 1},
        )
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
command creates them, Child creation has no approval state, `validate_artifact()`
rejects the removed `review` kind, and all four target prompts contain
optional—not mandatory—Subagent guidance.

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

### Task 2: 添加启动触发的 Global Memory 及完整受控边界

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
- Consumes: the existing fixed `publish_artifact`; Task 1 published Analysis layout;
  Codex `runtime.start()` shared by start/resume/automatic restart.
- Produces:
  - `list_analysis_refs(notes_root: Path, mode: str | None = None) -> tuple[str, ...]`
  - `read_processed_analyses(summary: Path) -> frozenset[str]`
  - `memory_due(notes_root: Path, threshold: int = 3) -> bool`
  - `validate_summary(content: str, notes_root: Path) -> frozenset[str]`
  - `MemoryContext(coral_dir, island_root, notes_root, island_id)`
  - `MemoryResult(status, research_summary_path=None, engineering_summary_path=None)`
  - `consolidate_memory(*, if_due: bool, context: MemoryContext | None = None, runner=run_fresh_codex, publisher: Publisher | None = None) -> MemoryResult`
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
`<island_root>/memory-consolidation.lock`; `island_root()` resolves to
`.coral/public` for a single-island run and `.coral/islands/<id>` otherwise.
This file is synchronization, not workflow state. `--async` starts the same CLI in hidden worker mode and returns.
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

## Final Verification

- [ ] Run: `git diff --check`
- [ ] Run: `uv run --frozen --extra dev ruff check coral tests tasks/greedylore_autoresearch/autoresearch`
- [ ] Run the focused Task 1 and Task 2 commands exactly as listed above.
- [ ] Run: `uv run --frozen --extra dev pytest -q`
- [ ] Run: `uv run --frozen --extra dev pytest tasks/greedylore_autoresearch/tests -q`
- [ ] Run: `uv run --frozen --extra dev coral validate tasks/greedylore_autoresearch --json`
- [ ] Confirm the diff contains no Evaluation Contract, grader workload, GPU protocol, dense reference or Root bootstrap changes.
- [ ] Confirm no test starts a real Codex session, GPU workload, timer or background scheduling service.
