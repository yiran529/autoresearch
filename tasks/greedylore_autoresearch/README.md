# Gap-driven GreedyLoRE AutoResearch

Task resources are host-published under the run's public `notes/toolkit/`. Each Agent's generated instructions locate that directory under its runtime shared directory (for Codex, `.codex/notes/toolkit/`). It contains workflow rules and route-specific references. It does not contain private evaluation inputs. Research Analysis receives a fixed code/evidence snapshot and only the parent analyses actually referenced by its Spec.

Research Analysis remains an independent session. Choose applicable initial diagnostics, investigate a principal unexplained observation, and report evidence, counterevidence and unknowns without proposing algorithms. Each Analysis has at most five successful diagnostic executions and fifteen total executions including failures, each up to 600 seconds, with a separate GPU queue budget of one hour per request. Failed diagnostics may be debugged and retried with new request IDs; all retries retain records and count toward the total budget. See `GPU_COORDINATION.md` for host/client wait budgets.

## Node workflow

Each Research Node fixes one method hypothesis, Parent/base_commit and Spec. Within the Node, candidates may refine the method or meaningful parameters such as refresh period or rank, and descend from earlier candidates. Core mechanism or hypothesis changes require a child Node. Record each candidate's change and rationale in `change_summary` and `selection_rationale`.

Research allows at most 3 successful scored formal runs and 10 total formal runs across resumes and archived Attempts. Valid scored regressions count as successful regardless of quality passing; failures and permitted same-SHA infrastructure retries count toward the total, and pending runs reserve capacity. The Researcher remains active after the first terminal result and continues justified refinement within budget. Finish earlier with `autoresearch node complete --mode research` when no justified refinement remains and all runs have ended; at either cap the host freezes Research after all runs end.

The host selects the quality-passing, comparable candidate with a valid score and highest efficiency, using deterministic ties. If none qualifies, it retains a representative failure and skips Engineering. Only after Research freezes does the host retire the Researcher and start two independent branches: Research Analysis over all Attempts (including failures and regressions), and Engineering when the authoritative quality gate passes. Analysis distinguishes method gains from hyperparameter gains. The host binds each Engineer to a fixed Node and Research SHA; `autoresearch engineering claim Nxxx` confirms that assignment and prepares its Worktree, rather than selecting a target. Required Engineering waits in FIFO order when the configured Engineer capacity is occupied. Researcher and Engineer capacities remain separate.

A Node is an eligible Parent only after Research Analysis is published and Engineering is completed, or explicitly skipped because quality failed or was unavailable in a terminal failure. For iterative Research, missing or invalid authoritative quality/score evidence cannot qualify a candidate for Engineering. Researcher retirement alone does not complete the Node. Children still start from the Parent's Research SHA, not its Engineering commit.

Active iterative Research retains tensor payloads for its current best candidate and in-flight runs. Other terminal candidates' payloads are pruned with a private cleanup ledger; original manifests, configurations, logs and Attempt records remain available. After freezing, only the selected candidate's terminal checkpoint set is retained for Analysis. If no candidate qualifies, a deterministic failure representative is retained; archived Attempts can serve only as the last-resort failure representative and cannot enable Engineering.

Root bootstrap remains one fixed baseline Attempt A0; historical completed Nodes are not reopened. Root N000 follows the same two branches after the host pins S0, creates A0 and submits the fixed baseline Attempt. Dynamic scheduling can start before Root Analysis exists, but does not reserve ordinary Researchers until N000 is fully complete. Root Engineering uses A0 as its quality anchor and ordinary Engineer authorization; the host-only bootstrap exception remains restricted to Root Research. The standalone `bootstrap-root-analysis` command produces baseline evidence only and does not certify full Node completion or automatically start Engineering.

The existing Analysis worker handles only Research Analysis. Short host reconciliation publishes sealed Engineering summaries without waiting for that worker. Analysis retries preserve their fixed evidence/version and diagnostic budget, apply bounded backoff, and do not block later eligible jobs. A validated private output or published immutable Analysis is reused after interruption. Workflow/model process identities are recorded for stop and restart cleanup; Analyzer sessions have a default overall limit of 18000 seconds, including diagnostic queue waits.

Standalone Analyzer and Memory model calls recover transient connection failures with at most three launches and 15/30-second backoff within the original overall timeout. Authentication/configuration failures are not retried. Analyzer reconnects keep the same task/workspace and diagnostic budget and retain per-launch logs; they do not resume an ephemeral conversation. Researcher/Engineer recovery continues to use AgentManager's existing Session resume and circuit breaker, without another launch-retry layer.

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

Historical Analyses remain readable. Node-bound scheduling uses workflow state version 2; version 1 runs are rejected without modification and must continue with their original runtime. No automatic migration or reopening of historical claims is performed. Saved configuration and role snapshots must match the runtime before resuming a version 2 run. Complete dynamic runs require at least one Engineer slot and exactly one Engineering assignment per quality-passing Node; standalone baseline-only tools retain their original behavior.

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

The existing Research-driven Memory trigger and two Markdown Summary strings are retained. Summaries distinguish new mechanism evidence, replication, refutation/scope restrictions and open questions, with exact Analysis versions. Research Summary covers algorithm/state/update evidence; Engineering Summary covers measured performance and Engineer-authored attribution. Engineering Analysis is an input, not an independent trigger.

For bounded verification, Memory can read public diagnostic records matching a published Node/mode/version/Research SHA, plus text results. Staging excludes symlinks, scripts, code/state snapshots, binary tensors and unrelated identities, with limits of 1 MB per file, 256 files and 8 MB total. Omitted files must be treated as unavailable, not as negative evidence.


## Host-assigned research modes

The host draws a mode once when reserving a Research assignment, using trusted `grader.args.p_explore` (initially `0.20`). It persists the result in private workflow state, sends it through the authenticated roster and generated Agent instructions, and records it in new Node Specs. Resumes, same-SHA retries and duplicate reconcile/initialize requests retain the original mode even if the configured probability changes. Pre-upgrade assignments missing a mode use `REFINE`; existing immutable Specs are not rewritten.

`REFINE` starts from supported gaps or important unresolved mechanisms in published analyses. `EXPLORE` prioritizes another communication mechanism or a falsifiable cross-mechanism interaction that the current evaluation contract can fairly evaluate. It does not relax the contract or quality gate. Researcher cannot set the mode in `node initialize`, draw it locally, or use multiple code Parents. The host-assigned mode is part of assignment context, not an additional Research initialization field.


## Prompt delivery audit

`tests/test_workflow_simulation.py` runs normal and fault/recovery scenarios through the actual `CodexRuntime` launcher and the actual one-shot Analyzer/Memory launcher. A test-only `codex` executable records argv/stdin, then runs the scripted agents or supplies synthetic JSON. The host's random draws are fixed in the test handler to exercise both REFINE and EXPLORE. No live model or GPU workload is started.

Every run writes `prompt_audit/*.json` alongside its `report.json`. Producer captures include their startup/resume text, generated `AGENTS.md`, exact role content and task resource snapshots. Analyzer/Memory captures include their final stdin, supplied Schema, staged evidence paths and identities. The tests verify mode pinning, resume arguments and instruction-file continuity, role/reference delivery, full code-snapshot hashes against the fixed Git commit, staged evidence hashes against the manifest (including parent Analysis), Memory Analysis hashes against published sources and trigger coverage and absence of independent Engineering Analyzer calls. `prompt_audit/CHECKS.md` is written only after the delivery assertions pass.

The diagnostic-enabled Analyzer has an additional final-input test at the namespace/process boundary, checking the 5-success/15-total diagnostic limits and 600-second execution limit, separate 3600-second queue wait, fixed roots, writable scratch and JSON Schema. That test substitutes namespace/process execution, so it does not establish OS isolation or GPU execution.

These checks establish application input delivery. Live Codex instruction loading, conversation persistence, model compliance and scientific effectiveness require separate live acceptance tests.
