# Gap-driven GreedyLoRE AutoResearch

## Exclusive method selection

New Research Nodes acquire a run-wide selection lock before reading registered
methods and choosing their own. `autoresearch node planning acquire --wait`
returns a token and the shared registry location (`notes/tree.md`, linking to
canonical Specs). Hold the lock through reading, reasoning and overlap review;
then register once through `autoresearch research register` with `origin`, first
`design`, and top-level `planning_token` / `overlap_review` controls. The host records the review in
the immutable Spec and releases the lock atomically with registration. Models
judge semantic overlap; the host enforces exclusive ownership, not similarity.

The default task sets `grader.args.research_planning_lease_seconds: 1800`.
Renew using `node planning renew --token <token>` before expiry, or explicitly
release with `node planning release --token <token>`. Expired tokens cannot
register; reacquire and reread before deciding again. Cancelled Nodes and dead
owner processes relinquish ownership on the next request. The lease survives a
service restart and authenticates the owning Agent process identity. It does
not hold the global write lock while a model thinks, so implementation,
training, grading and Engineer work remain concurrent. There is no timed
Researcher launch stagger. Existing Specs remain immutable. Historical run
configs without this option retain their prior initialization behavior; enable
the option to require the selection lock for their uninitialized Nodes.

## Evaluation and handover policy

- Contract v2 requires language final loss / same-contract Dense final loss <= **1.05**,
  and final-epoch ResNet18/CIFAR100 test accuracy drop <= **1.0 percentage point**
  by default (operator configurable). See `references/evaluation_tasks.md` for fixed
  task settings and candidate-owned compression policy. Historical v1 contracts,
  Attempts and verdicts retain their original behavior. New matching dense/Root
  references are required; historical seed results are not authoritative new references.
- Attempts retain optional advisory `utility`: `U = S * exp(-d)`, where `S` is
  the geometric mean BF16/FP32 speedup over a fixed, stage-matched N000 reference,
  and `d = max(0, candidate minus baseline loss over the two quality workloads)`.
  This discounts speedup by the worst PPL degradation; it does not reward quality
  improvement or estimate time to quality. It never gates exploration, Parent
  eligibility or candidate selection. Missing/incompatible evidence produces null
  U with a reason. Engineering quality remains inherited and provisional.
- New methods must commit `timing.period_steps` in `submission.yaml`; only
  `requested_steps` is optional. Legacy submissions may omit `timing`. Period
  means the effective full cycle, including interacting schedules; use null
  when there is no fixed cycle (including adaptive or uniform per-step methods). The host aligns the window,
  normally 400 and at most 800 steps, retaining two repetitions. Only the trusted
  `grader.args.timing_measured_iterations` may override the request, within bounds
  and complete-cycle alignment.
- The timing harness records `timing_cycle.period_steps` and per-rank
  `rank_event_steps` (1-based update numbers). The default hook records actual
  refresh execution into `state.timing_refresh_steps` when timing enables it.
  New mechanisms must maintain this telemetry and set `state.timing_period_steps`
  if their cycle differs from `update_proj_gap` (null for nonperiodic methods).
  Record actual intermittent maintenance executions, including all interacting
  schedules; multiple events per period or update are allowed. Fixed-period
  evidence must repeat the same event pattern across measured cycles. For null
  periods, events may be irregular or absent. All event lists must be ordered,
  within the measured window and consistent across ranks.
  This checks declared schedules against observations, not the true algorithmic
  period. Nonperiodic timing covers only the measured window; absent rare events
  do not establish their long-run cost.
- Longer-than-cap cycles, missing/disagreeing event evidence or absent matching
  reference windows retain raw timing but are not eligible performance claims.
  Dense timing caches may supply `efficiency.json.timing_windows["800"]` with
  `measured_iterations: 800` and actually measured `cells` under the same contract
  and hardware environment. Never extrapolate a cached window or relabel old
  measurements. Root U references must also match the measured window.
- Before implementation, Researcher reads available Engineering Summary and
  parent Engineering Analysis; record why either report is unavailable. Supply
  the `origin` in `templates/research_origin.json`: question, parent response,
  evidence and Engineering reuse/defer decisions; initial predictions/tests live
  in the first `design`. The host generates compatibility Spec/handover fields
  without a separately authored intent object. New Nodes
  prioritize mechanisms; rank/width/gap sweeps refine a mechanism within its budget.
- Either mode may begin from an observation, desired property or mechanism hypothesis.
  REFINE builds on the Parent method/mechanism; EXPLORE uses a substantively
  different mechanism. Reference a real parent problem when directly addressing
  it; otherwise use null and explain the motivation/connection, without inventing a gap. The host freezes the parent handover snapshot
  and pins available Engineering Analysis alongside Research Analysis.
- Analyzer returns separate mechanism/practical verdicts, persistent evidence
  references, incremental information (possibly empty) and at most two follow-up
  questions with discriminating experiments. Host-derived `handover.json` binds
  complete Analysis document hashes and is recoverable without another model.
  Old reports remain readable and are explicitly marked as legacy handovers;
  unavailable evidence is not replaced by invented conclusions.

Use this policy for new runs with the updated seed, contract and role snapshots.
Paused historical runs are not automatically migrated or restarted.

Task resources are host-published under the run's public `notes/toolkit/`. Each Agent's generated instructions locate that directory under its runtime shared directory (for Codex, `.codex/notes/toolkit/`). It contains workflow rules and route-specific references. It does not contain private evaluation inputs. Research Analysis receives a fixed code/evidence snapshot and only the parent analyses actually referenced by its Spec.

Research Analysis remains an independent session. Arrange investigation according to the supplied evidence and worthwhile research questions; report useful findings, counterevidence and unknowns without proposing algorithms. Each Analysis has at most five successful diagnostic executions and fifteen total executions including failures, each up to 600 seconds, with a separate GPU queue budget of one hour per request. Failed diagnostics may be debugged and retried with new request IDs; all retries retain records and count toward the total budget. See `GPU_COORDINATION.md` for host/client wait budgets.

## Node workflow

New task configurations enable staged Research (`grader.args.staged_research: true`). The role, generated startup instructions, public workflow rules and command guide use the same protocol. `references/research_commands.md` provides full registration, pause/diagnose/resume, revision and finalization examples; `templates/research_origin.json` is the one registration request template.

Register `{origin, design}` plus planning controls. The origin fixes the Node's question/core hypothesis, Parent, evidence/selection rationale and parent/Engineering response. Each design states method, explanation, predictions and tests. The host supplies provenance, stores version-3 `research.json`, generates an immutable initial Spec with compatibility handover fields and the first-design hash, and appends runtime observations/decisions. Researcher submits no separate intent or final report. Existing version-2 records remain readable; fresh staged registrations cannot use the old registration shape or initialization command.

Development experiments normally use two, at most three trajectories, including failures/stops/restarts, within three append-only designs including timing-screened designs. Same-SHA exact continuation stays within its original fixed budget. Researcher uses managed diagnostics at meaningful pauses, judges predictions and records reasons before continuing/revising/stopping. Optional development timing can run before training, while paused or after completion, within its own 5-completed/15-total budget. Diagnostics retain their 5-success/15-total/600-second limits.

When development ends, `research finish` requires one completed candidate with passing development validity/quality/resource evidence, or records none. It first freezes the selected SHA, design, compression policy and workload mapping. Final evaluation then runs fresh mandatory timing, language transfer and ResNet18/CIFAR100; final timing is separate from development timing and never reuses development measurements. Preserve timing validity/cycle/hardware checks without adding a speedup-magnitude gate. The host retains its four GPUs until every final component is terminal or explicitly skipped, then releases them, retires Researcher and starts independent Analysis and quality-gated Engineering. Final feedback cannot reopen development or authorize tuning. Infrastructure recovery permits two retries per final component by default, retaining completed component results; vision retries restart from scratch, not exact checkpoint continuation. Same-component cache reuse is for recovery only. Spec/record publication retries preserve the original design and do not repeat completed computation.

Candidates define per-workload compression mapping in committed `method_policy.py` (`configure_compression(args, workload)`) and the shared hook. Verify the method applies to matrix and convolution gradients; freeze this policy before final feedback. Rank 64, dense prefix 4000 and refresh 1200 are stock GreedyLoRE vision defaults, not fixed candidate constraints. Dense references keep compression disabled. Fixed training settings and quality gates are in `references/evaluation_tasks.md`. Independent Analysis reads the full frozen record including `final_evaluation` component evidence, distinguishing development-selected findings from final transfer/robustness evidence. Reuse by later Nodes does not make final evidence an independent campaign held-out.

Saved configurations retain their recorded protocol. Explicit non-staged configurations select the retained compatibility role/rules; new Researcher instructions contain only the staged workflow. No running experiment is automatically migrated.

Formal timing still requires matching GPU counts/models and NCCL settings, with unique GPU UUIDs. Topology fingerprint differences, NUMA differences, and unavailable NUMA information produce a `cross_topology` warning and allow scoring and candidate selection. The host records baseline/candidate hardware and differences in `evaluation_context.json` and Attempt metadata, includes a feedback warning, and displays the status and warning in `evaluation.md`.

The task sets `grader.args.max_active_nodes: 2`: at most two unfinished Nodes advance concurrently. Each Node keeps one slot across Research, Research Analysis and Engineering; Researcher retirement does not release it. Full completion releases one slot immediately, without waiting for the other Node. The oldest waiting Node resumes before another is created. Node admission is the only task concurrency limit: separate `agents.dynamic.max_researchers` and `max_engineers` values are ignored while `max_active_nodes` is positive. Either admitted Node can be researching or engineering, and its Research Analysis can run alongside Engineering. On resume with an older backlog, later Nodes retain their Spec, code, worktree, session and consumed budgets while their Agents, Analyses and pending formal evaluations wait. The trusted dispatch filter rechecks admission before recording a grader invocation, so deferral does not spend a formal run or finalize an interrupted Attempt. Missing or zero `max_active_nodes` retains legacy unlimited-Node scheduling; enabled runs also configure `trusted_dispatch_filter: autoresearch.state:filter_dispatch`.

Research Analyses for admitted Nodes run concurrently. A single workflow supervisor keeps dispatch idempotent, while each Node has its own session lock; a slow Analyzer does not hold up another Node's ready job or retry. State changes and publication use short shared locks. Every model session has a separate private recovery record, so stopping or recovering the run also stops all parallel Analyzer models; legacy single-model recovery records remain supported.

A Node is an eligible Parent only after Research Analysis is published and Engineering is completed, or explicitly skipped because quality failed or was unavailable in a terminal failure. For iterative Research, missing or invalid authoritative quality/score evidence cannot qualify a candidate for Engineering. Researcher retirement alone does not complete the Node. Children still start from the Parent's Research SHA, not its Engineering commit.

Active iterative Research retains tensor payloads for its current best candidate and in-flight runs. Other terminal candidates' payloads are pruned with a private cleanup ledger; original manifests, configurations, logs and Attempt records remain available. After freezing, only the selected candidate's terminal checkpoint set is retained for Analysis. If no candidate qualifies, a deterministic failure representative is retained; archived Attempts can serve only as the last-resort failure representative and cannot enable Engineering.

Once an ordinary Node has completed Research, validated Research Analysis, and completed Engineering with validated Analysis (or explicitly skipped Engineering), host reconciliation removes all of that Node's evaluation checkpoint tensors, including the selected candidate and archived/retried Attempts. Cleanup waits for pending/running graders, diagnostic commands and Analyzer sessions to drain; completed Nodes reject further diagnostics. Original manifests, configurations, metrics, logs, Attempts and reports remain unchanged, so the Node remains usable as a code/Analysis Parent, but its tensors cannot be used for later continuation or diagnostics. Root N000 retains its baseline checkpoints. The host saves completion before deletion and records each removal in `private/checkpoint_retention.json` before unlinking payloads. Per-Node `cleanup.status` in `private/autoresearch_state.json` becomes `done` only after successful cleanup; errors remain `pending` with `last_error` and are retried on the next reconciliation, including after restart. Existing completed version-2 Nodes without a cleanup record are also eligible.

Root bootstrap remains one fixed baseline Attempt A0; historical completed Nodes are not reopened. Root N000 follows the same two branches after the host pins S0, creates A0 and submits the fixed baseline Attempt. Dynamic scheduling can start before Root Analysis exists, but does not reserve ordinary Researchers until N000 is fully complete. Root Engineering uses A0 as its quality anchor and ordinary Engineer authorization; the host-only bootstrap exception remains restricted to Root Research. The standalone `bootstrap-root-analysis` command produces baseline evidence only and does not certify full Node completion or automatically start Engineering.

The fixed Root evaluation follows the recorded contract: the current default runs development-language quality for 4000 updates, language-transfer quality for 5000 updates, ResNet18/CIFAR100 quality and fresh formal timing. Original language schedules and warmups remain unchanged. Root has no iterative development or candidate selection; its algorithm is fixed at A0. Changed budgets invalidate the previous contract's dense/Root references, so establish matching references and evaluate a new run rather than reopening or relabelling historical Nodes.

The existing Analysis worker handles only Research Analysis. Short host reconciliation publishes sealed Engineering summaries without waiting for that worker. Analysis retries preserve their fixed evidence/version and diagnostic budget, apply bounded backoff, and do not block later eligible jobs. A validated private output or published immutable Analysis is reused after interruption. Workflow/model process identities are recorded for stop and restart cleanup; Analyzer sessions have a default overall limit of 18000 seconds, including diagnostic queue waits.

Standalone Analyzer and Memory model calls recover transient connection failures with at most three launches and 15/30-second backoff within the original overall timeout. Authentication/configuration failures are not retried. Analyzer reconnects keep the same task/workspace and diagnostic budget and retain per-launch logs; they do not resume an ephemeral conversation. Researcher/Engineer recovery continues to use AgentManager's existing Session resume and circuit breaker, without another launch-retry layer.

Research Analysis canonicalizes staged `evidence/...` citations to the fixed evidence inventory. Fixed-SHA regular Git files and the current Analysis task's authenticated terminal diagnostic text/log/script results are additionally hashed into the published evidence manifest; this does not change the frozen Research input or reset diagnostic budgets. Recovery verifies those hashes and identities again. A missing or foreign citation is reported by its exact path. Publication validation errors are passed to the next Analyzer session and block the Analysis after three validation failures; the host must repair and explicitly recover that job before it can advance. Historical immutable reports remain unchanged.

New Research and Engineering Analyses are published as an immutable full `.json` record and a deterministic `.md` reading view. The host renders Summary, Findings, Limitations and evidence navigation without another model call; verbose evidence manifests and frozen Evaluation snapshots remain in JSON. Publication writes JSON first and completes/repairs the view on retry before declaring success. Machine readers verify the Markdown against its sibling JSON; legacy JSON-in-MD remains supported without automatic rewriting. Explicit host conversion must preserve a backup of the original. Parent snapshots and Memory staging retain both files, while producers read Markdown first and follow evidence links as needed.

Agent timeout is a 10800-second no-output watchdog, not a total runtime limit; formal grading has no total timeout (`grader.timeout: 0`), and pending evaluations retain the existing 176400-second watchdog exemption, allowing a single evaluation to exceed one hour.

## Engineering completion

An assignment permits at most 3 successful formal Engineering runs and 10 total formal runs, counted across resumes. Success means a completed evaluation with a valid score, including a performance regression. Failures and permitted same-SHA infrastructure retries count toward the total; pending submissions and retries reserve capacity. Reaching either limit ends evaluation, including any remaining per-SHA retry allowance, but does not retire the Engineer: it must still author and submit the final summary.

Within those limits, Engineer continues until no clear, substantial semantics-preserving end-to-end improvement opportunity remains; one improved candidate is not sufficient to stop. The final Analysis briefly explains the stopping reason and any remaining headroom.

Write JSON in authorized scratch outside the tracked Worktree, containing exactly:

- `summary`: non-empty text.
- `findings`: a non-empty array of non-empty strings covering measured results, each candidate, semantic checks and supported attribution.
- `limitations`: a non-empty array of non-empty strings, with uncertainty, missing evidence and check coverage.
- `attempt_shas`: all distinct formal candidate SHAs in this claim, exactly once, including failures and regressions.

Run `autoresearch engineering finish --analysis-file /absolute/scratch/final.json`. The host authenticates ownership, checks terminal Attempts and complete SHA coverage, seals the evidence and persists the submission before returning `finalizing` with the intended Analysis ref. This is not confirmation of completed publication. The host then publishes an immutable version and completes the claim. On interruption, reuse the identical body; conflicting submissions are rejected. No independent Engineering Analyzer is launched. Quality remains inherited (provisional) from the fixed Research SHA.

Historical Analyses remain readable. Node-bound scheduling uses workflow state version 2; version 1 runs are rejected without modification and must continue with their original runtime. No automatic migration or reopening of historical claims is performed. Saved configuration and role snapshots must match the runtime before resuming a version 2 run. Legacy scheduling without a positive Node limit requires at least one Engineer slot; both modes require exactly one Engineering assignment per quality-passing Node; standalone baseline-only tools retain their original behavior.

## Validation

Active Engineer claims can use `autoresearch diagnostic run --scripts-dir /absolute/own/scripts --gpus 4 --timeout 600 -- python experiments/probe.py --output results/probe.json`. See `references/engineer_runtime.md` for staging and paths. Tracked code must be committed before diagnosis; scripts come from the authenticated Worktree or `/tmp/<agent-id>/`. Host-owned records preserve candidate code, scripts, logs and results separately from formal Attempts. Diagnostics have their own persistent 5-success/15-total budget, share existing GPU locks, and cannot run after Engineering finish. `engineer_diagnostics: false` disables them. The final Analysis seals matching diagnostic records; a running diagnostic must finish before final submission. Ordinary scratch output outside this interface is not automatically archived.

For a real Engineer-only workflow on existing passing Root evidence, run:

```bash
uv run python tasks/greedylore_autoresearch/scripts/run_root_engineer.py \
  --source-run /path/to/root/run --output /path/to/new/engineer-experiment \
  --model gpt-6.1-sol --effort medium
```

The driver snapshots current code, independently copies Root evidence, and uses normal AgentManager, Codex/SRT, prompts, formal grader and trusted Engineering publication. Researcher capacity is zero; existing Research Analysis is reused, and no Analyzer or Memory worker is launched. Normal workspace dependencies and SRT must be available. `status.json`, `run_contract.json`, delivered instructions/role, startup log entries and the run's logs/Artifacts record the result. Use the same command with `--resume` to recover without resetting Attempts or budgets. Root quality must pass; this runs real GPU timing and does not repeat Root bootstrap or quality training.

The driver uses an isolated Codex home and the shared SRT mount rules, retaining private filesystem exclusions. The formal task's Linux IPC opt-in (`allowAllUnixSockets`) permits Unix and VM sockets so trusted task commands can connect; it is broader than a per-path socket grant. Both formal and isolated runs use this setting. Shared worktree reads are enumerated at each Agent start; worktrees created later become readable after the Agent restarts. The provider prepares empty SRT protection anchors and locally excludes only newly created files from Git, preserving existing user files.

For a real Linux sandbox check without a model or GPU, install SRT and its dependencies on `PATH`, then run `CORAL_SRT_SMOKE=1 uv run pytest -q -s tests/test_sandbox_srt_integration.py`. It verifies worktree writes, Git commits, sibling read-only access, private evidence isolation and a real trusted-socket response with authorization still enforced. Saved run configurations retain their old settings; use the updated task configuration when starting a new formal run.

Scripted workflow simulations cover normal operation and injected failures, including same-SHA retry, automatic Node binding, foreign-claim rejection, publication and parallel Research. They use synthetic results, with no model, training or GPU calls. Scientific effectiveness requires a separate real low-rank pilot with matching budgets and legitimate saved training states; passing workflow tests does not establish scientific improvement.


## Memory

After each Node (including Root) completes Research, Research Analysis, and completed or explicitly skipped Engineering, the host asynchronously queues Global Memory consolidation. One worker serially drains the private `memory_state.json` queue. Failures remain pending; sealed outputs allow interrupted publication to resume without another model call. Memory publication does not gate Node completion or pause Researchers.

Memory returns three Markdown bodies: `research_summary`, `engineering_summary`, and `wisdom`. The host publishes them to `notes/` and supplies cumulative `processed_analyses` frontmatter. All three publications must succeed before acknowledging a Node. Coverage records incorporated Analysis versions, not a claim that all history was reread. Existing runs with only two summaries bootstrap wisdom once; old two-document cached responses are regenerated.

`wisdom.md` contains a small set of reusable findings, applicability limits/counterexamples, design implications and exact versioned Analysis sources. New evidence updates an existing finding; contrary evidence narrows or retracts it. No new insight means the body may remain unchanged. Research and Engineering Summaries provide progress/results navigation to wisdom and original Analyses. Researcher and Engineer read wisdom within its evidence limits and consult the source before relying on it.

Each call requires only the trigger Node's Analyses, plus historical coverage missing from wisdom during migration. Previously covered Analyses remain available for relevant lookup; only required Nodes supply Specs/Evaluations. Raw evidence is staged only when named in a required Analysis's frozen manifest and its bytes match the recorded SHA-256. Allowed sources are public evaluation logs, diagnostic text and referenced research records. Missing hashes, mismatches, symlinks, scripts and binary tensors are omitted. Limits are 1 MB per file, 256 files and 8 MB total across raw evidence; omissions mean unavailable evidence, not negative findings.


## Host-assigned research modes

The host draws a mode once when reserving a Research assignment, using trusted `grader.args.p_explore` (initially `0.20`). It persists the result in private workflow state, sends it through the authenticated roster and generated Agent instructions, and records it in new Node Specs. Resumes, same-SHA retries and duplicate reconcile/initialize requests retain the original mode even if the configured probability changes. Pre-upgrade assignments missing a mode use `REFINE`; existing immutable Specs are not rewritten.

`REFINE` deepens or challenges the Parent method/mechanism, starting from evidence, a desired property or a testable mechanism; a listed handover problem is not required. `EXPLORE` prioritizes another communication mechanism or a falsifiable cross-mechanism interaction that the current evaluation contract can fairly evaluate; it may also start from a desired property rather than an existing finding. It does not relax the contract or quality gate. Researcher cannot set the mode in `node initialize`, draw it locally, or use multiple code Parents. The host-assigned mode is part of assignment context, not an additional Research initialization field.


## Prompt delivery audit

`tests/test_workflow_simulation.py` runs normal and fault/recovery scenarios through the actual `CodexRuntime` launcher and the actual one-shot Analyzer/Memory launcher. A test-only `codex` executable records argv/stdin, then runs the scripted agents or supplies synthetic JSON. The host's random draws are fixed in the test handler to exercise both REFINE and EXPLORE. No live model or GPU workload is started.

Every run writes `prompt_audit/*.json` alongside its `report.json`. Producer captures include their startup/resume text, generated `AGENTS.md`, exact role content and task resource snapshots. Analyzer/Memory captures include their final stdin, supplied Schema, staged evidence paths and identities. The tests verify mode pinning, resume arguments and instruction-file continuity, role/reference delivery, full code-snapshot hashes against the fixed Git commit, staged evidence hashes against the manifest (including parent Analysis), Memory Analysis hashes against published sources and trigger coverage and absence of independent Engineering Analyzer calls. `prompt_audit/CHECKS.md` is written only after the delivery assertions pass.

The diagnostic-enabled Analyzer has an additional final-input test at the namespace/process boundary, checking the 5-success/15-total diagnostic limits and 600-second execution limit, separate 3600-second queue wait, fixed roots, writable scratch and JSON Schema. That test substitutes namespace/process execution, so it does not establish OS isolation or GPU execution.

These checks establish application input delivery. Live Codex instruction loading, conversation persistence, model compliance and scientific effectiveness require separate live acceptance tests.
