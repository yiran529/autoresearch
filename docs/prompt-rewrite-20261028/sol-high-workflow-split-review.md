# GreedyLoRE workflow split review

**Date:** 2026-10-28

**Review baseline:** `421c49a` (`feat(prompts): integrate shared research goal across agent roles`)

**Requested reviewer configuration:** GPT-5.6 Sol, high reasoning
**Scope:** Read-only review of the staged GreedyLoRE prompt split, its delivery paths, and focused tests. The review did not edit production files or tests.

## Findings

### Critical

None.

### Important

One important omission was found during review and fixed before signoff. The first split preserved original predictions and failures, incomparable-timing counting, and the general instruction not to repeat failures blindly, but it no longer explicitly preserved regressions and unavailable evidence for Research and no longer prohibited empty commits as manufactured retries. `roles/workflow_research.md` now says:

> Preserve failures, regressions, incomparable and unavailable evidence; do not blindly rerun or use empty commits to manufacture retries.

This restores the explicit retry/evidence behavior from the baseline without expanding the Researcher role. No important finding remains open.

### Minor and advisory

None requiring a prompt change. The split intentionally leaves host lifecycle mechanics in the task README and host implementation rather than repeating them in an agent supplement. That is acceptable because those items do not require an additional role action and the actionable consequence remains explicit in the role or supplement.

## Disposition

**Approved.** At reviewer signoff, the split reduced the 2,075-word shared workflow to 381 words of common execution boundaries plus 498 words of Research-only conditions, including the restored retry/evidence rule. The Researcher role remained effectively unchanged in size and purpose, and the Engineer role gained only the two previously missing explicit constraints. Subsequent user-directed edits are recorded separately below.

The shared research goal is unchanged. No new scientific goal, preferred method, promotion criterion, or role responsibility was introduced.

## Coverage

| Requirement family | Final reachable owner | Assessment |
|---|---|---|
| Shared write and private boundaries | `roles/workflow_rules.md` | Retained explicitly: task-command publication, Tree/Spec/Attempts/formal-log/history protection, shared conclusions, private evaluation material and credentials. Worktree/scratch authorization remains in generated instructions and roles. |
| Resource and storage boundaries | `roles/workflow_rules.md`; `references/research_commands.md`; Engineer role/reference | Retained: host paths, dynamic storage, no arbitrary/root copies, no retention/cleanup changes, and host-managed GPU entry points. Formal timing instrumentation prohibitions remain role-specific. |
| Waiting and request identity | `roles/workflow_rules.md`; role command references | Retained: supported blocking waits, no watcher/short polling/log polling, original ID and unchanged payload after a lost reply, new ID for a new execution, and failure inspection before retry. Research `status`/`wait`/`recover` and Engineering same-SHA retry details remain explicit. |
| Persistent evidence | `roles/workflow_rules.md`; both roles | Retained: archive timing, 32 KiB/64-bundle limits, unique basenames, managed diagnostics for larger evidence, public-relative paths, actual commands/inputs/scope, temporary-path exclusion, and attachment identity versus execution certification. |
| Research registration and lease | Researcher role; `roles/workflow_research.md`; `references/research_commands.md` | Retained: planning acquire/renew/release/loss behavior, latest registry/overlap review, clean Parent SHA, origin and initial design before implementation, failed-registration lease retention, host-generated Spec/handover, and no separate intent. |
| Research preferences | `roles/researcher.md` | Retained without being converted into hard workflow rules: low-rank and sparsification priority while allowing motivated alternatives; Parent selection by research value, continuity, information gain, and validation cost; handover-first assessment with observation/property/mechanism starts allowed; REFINE/EXPLORE distinction; combinations, adaptive policies, and falsifiable design fields. |
| Research budgets and counting | Researcher role; `roles/workflow_research.md`; host state | Retained: normally two trajectories, hard cap three with justification, three designs including timing screens, timing 5 completed/15 total, diagnostics 10 successful/20 total/600 seconds, infrastructure recovery default two, pending reservations, failures/stops/restarts, and no reset on pause/resume. Host state and saved contracts remain authoritative. |
| Code binding and continuation | Researcher role; `roles/workflow_research.md`; command reference | Retained: first accepted train/timing binds SHA, changed code requires appended design, decisions/design records do not execute work, exact same-iteration/SHA continuation restores all required state, prefix reuse requires matching behavior/state, revised designs train from scratch with matched initialization/data order, and Parent-checkpoint evidence remains local rather than replacement quality evidence. |
| Diagnostics and timing | Researcher role; Research supplement; command/evaluation references | Retained: diagnostics only on paused/completed trajectories, no candidate contamination/free training, fresh-data qualification, optional development timing, default result reuse, `repeat_reason`, latest matching result, incomparable counting, complete-cycle/event/hardware validity, and no profiling in formal timing. |
| Research submission and evidence judgment | Researcher role; Research supplement; evaluation reference | Retained: `mode: research`, Spec references, event telemetry, complete `period_steps` or null, fixed workload settings, pre-training checks, distinction between mechanism and practical benefit, original predictions and counterevidence, and the restored failure/regression/incomparable/unavailable-evidence rule. |
| Freeze and Final evaluation | Researcher role; `roles/workflow_research.md`; command reference | Retained: terminal development work, nullable nomination, representative SHA when nothing ran, frozen SHA/design/policy, fresh Final timing plus language transfer and vision, independent Final timing budget, no speedup-size gate, no post-freeze switch/tuning/reopen, same-frozen-evaluation recovery, per-component retry behavior, preservation of completed components, vision restart, and terminal/explicit-skip requirements. |
| Research lifecycle and Parent eligibility | Researcher role; Research supplement; task README; host state | Actionable parts remain explicit. Parent eligibility still requires published Research Analysis and completed or legitimate host-skipped Engineering. The two-unfinished-Node policy, cleanup timing, Root A0 lifecycle, and child scheduling remain in the public README and host enforcement, which is appropriate for host-owned context. |
| Engineering anchor and semantics | `roles/engineer.md`; `references/engineer_runtime.md` | Retained: claim and fixed Research anchor, selected iteration versus Root A0, missing-handover failure, fixed `base_commit`/`quality_anchor`, semantics and adaptive/RNG behavior, managed diagnostics/timing, and inherited provisional quality. |
| Engineering budgets, retry, and stopping | `roles/engineer.md` | Retained: 4 successful/10 total, valid regressions as success, failures/retries/pending counting, no reset on recovery/resume, diagnostic 10/20 counting including failures/timeouts/interruptions, same-SHA infrastructure retry versus lost reply, no empty-commit retries, stop at caps, and continue beyond one improvement while material supported opportunities remain. |
| Engineering completion | `roles/engineer.md` | Retained: at least one terminal Attempt, wait for diagnostics, archive evidence, cover every claim Attempt and SHA, exact four-field JSON, trusted finish, sealing/versioned publication, `finalizing` distinction, immutable resubmission, no independent Engineering Analyzer, and capacity waiting is not a valid skip reason. |
| Evaluation navigation | `roles/workflow_rules.md`; `references/evaluation_tasks.md`; Node README | Retained by concise routing: fixed SHA identity, current HEAD/source-worktree exclusion, Analysis Markdown/JSON distinction, Evaluation refresh command, saved contracts, exact quality gates, timing windows, matching dense/Root evidence, historical contract boundaries, and advisory U. |
| Analyzer boundary | `prompts/research_analysis.md`; `autoresearch/analysis.py` | Unchanged and self-contained. Analyzer still receives its own fixed-evidence, sandbox, diagnostics, retry/wait, no-code-mutation, role, and output-schema rules. It is not made dependent on either workflow supplement. |
| Memory boundary | `prompts/global_memory_manager.md`; `autoresearch/memory.py` | Unchanged and self-contained. Memory remains a one-shot read-only curator with its own evidence scope, attribution limits, publication contract, and no-experiment boundary. It is not made dependent on either workflow supplement. |
| Shared research goal | `prompts/shared_research_goal.md`; composition helper and Analyzer/Memory call sites | Unchanged from `421c49a`. Generated Researcher and Engineer instructions still compose the same goal once; Analyzer and Memory still receive it through their existing independent prompt composition. |

## Delivery and legacy compatibility

The new Research supplement is actually reachable, not merely present in the source tree:

1. Staged reconciliation copies `roles/workflow_research.md` to public `notes/toolkit/workflow_research.md`.
2. Generated staged Research instructions explicitly link that path.
3. `roles/researcher.md` names both the common and Research-only supplements at the points where their conditions matter.
4. Generated Engineer instructions link only the common supplement. Engineer-specific constraints remain inline in the Engineer role.
5. Prompt-audit support captures the actual staged bytes, and the staged simulation asserts both the copied content and the initial/resumed instruction link.

The production change is conditional on staged Research. The non-staged selection and copy paths are otherwise unchanged. The two deleted legacy role files visible in this worktree predate this task and are user-owned unstaged changes; they are outside the reviewed implementation. Compatibility verification used a temporary source snapshot retaining the committed legacy assets.

## Verification

- Independent focused run by this reviewer: `59 passed` for dynamic-agent delivery, handoff assets, and origin/toolkit delivery.
- Final focused verification reported after the review fix: `122 passed`.
- Expanded relevant verification in a temporary snapshot retaining committed legacy assets: `215 passed` before the prose-only final evidence-rule addition.
- Staged success simulation with fake model/CPU execution: `1 passed, 5 deselected` in about 42.6 seconds. It verifies actual toolkit copy, initial and resumed prompt delivery, and shared-goal composition; it performs no GPU training or real model call.
- `git diff --check 421c49a` passes.
- The shared-goal file/helper and Analyzer/Memory prompt/composition files have no diff from `421c49a`.

## User-directed follow-up after reviewer signoff

The user subsequently requested the following English prompt edits:

- Restore explicit 1800-second terminal waits (`yield_time_ms=1800000`) in the shared supplement, applying to both Researcher and Engineer, including enclosing execution/wait calls. Completed work may return early. The Research command reference points to that same rule.
- Merge Researcher's resource table and following prose into one table, preserving actual toolkit paths, source-checking guidance and required common/Research-only reading conditions.

Primary-agent verification after those edits: **122 relevant tests passed**, Ruff and `git diff --check` passed, and direct checks confirmed the wait duration, both roles' shared-rule entry points, the table-only resource section and preservation of the shared research goal. These follow-up edits were explicitly requested by the user and are not presented as part of the earlier independent model review.

## Remaining limitations

This is a semantic and delivery review, not an Agent-behavior experiment. It does not establish that shorter prompts improve research choices, completion rate, token use, or scientific quality. No GPU experiment, prompt A/B run, or real Analyzer/Memory model execution was performed for this review.

The common toolkit is public to both active roles, so Engineer can technically open `workflow_research.md`; the important boundary is that Engineer startup and role instructions do not direct it to consume Research-only conditions. Per-role filesystem isolation of that single file was not part of this design.

Exact enforcement still belongs to the host. Prompt coverage reduces operational ambiguity but is not a substitute for the existing validators, budgets, authorization checks, frozen records, and lifecycle state machine.
