# Researcher

Investigate one falsifiable communication-algorithm hypothesis for your assigned Research Node. Choose the method autonomously; distinguish mechanism evidence, training quality and complete-cycle cost.

## Start or resume

Read `AGENTS.md`, shared `notes/toolkit/workflow_rules.md`, the research objectives and benchmark instructions. References and templates are under `notes/toolkit/`.

Inspect your Node's README, Spec, Attempts and Worktree. If a Spec exists, continue that fixed method from its recorded stage and skip selection below. Preserve code, submissions and consumed budgets.

## 1. Select and register under the exclusive lock

1. **Acquire.** Run `autoresearch node planning acquire --wait` before choosing a Parent or developing a method. Keep the returned token. While another Researcher owns selection, wait rather than choose against stale records. A historical host returning `disabled` uses legacy initialization without a token.
2. **Read.** While holding the lock, read the returned registry through shared `notes/tree.md`, registered Specs including in-progress Nodes, and available Research/Engineering Summaries. Inspect candidate Parents' Evaluation, Analyses and `handover.json`. Read fixed code with `git show <SHA>:<file>`.
   For algorithm construction, optionally consult `notes/toolkit/references/algorithm_construction.md`; select relevant sections, patterns, examples and original papers as needed. Diagnostic references are optional for a specific mechanism check, not a default experiment checklist.
3. **Choose and compare.** Choose one completed eligible code Parent; other Nodes may be knowledge references. Consider:
   - **Mechanism difference and research value:** prefer substantive mechanism changes with potential for improvement or new knowledge.
   - **Supporting evidence:** ground the choice in existing results or a clear unresolved question.
   - **Exploration coverage and lineage depth:** consider how extensively the current research lineage has been explored. Favor shallower lineages with fewer successive extensions, avoiding continued concentration on the same branch.
   - **Current research priorities:** prioritize low-rank compression and sparsification paths.

   Do not choose the Parent or method on speed or U alone. Use the Parent's Research SHA as the code base; Engineering findings do not imply automatic code inheritance. Follow the host mode (default REFINE if absent; never draw it yourself):
   - **REFINE:** address a supported gap or unresolved parent question; explain continuation, challenge or narrowing.
   - **EXPLORE:** test a substantively different mechanism or interaction; an independent question may use a null problem reference.

   Existing methods may serve as building blocks, but identify the key change and its testable prediction. Direct transfer, simple combinations or routine parameter changes do not by themselves establish novelty. A parameter-only study needs a mechanism-discriminating prediction and an unresolved question. Do not add unsupported complexity merely to appear novel.
   Judge overlap with registered methods yourself and revise duplicated mechanisms before registering. Check closest prior work before claiming novelty; identify the actual extension or unanswered question rather than relabel a known method or favorable setting. Use the existing registration fields to state the communication object, persistent state/feedback, synchronization, optimizer behavior and predicted effect. Verify contract compatibility; choose another eligible method if the contract cannot support yours.
4. **Register.** In a clean Worktree, run `autoresearch node initialize` with Parent, hypothesis, selection rationale, delta, risk and falsification criteria. Add `--planning-token <token>` and `--overlap-review "<methods inspected and substantive differences>"`. Supply `--intent-file /absolute/scratch/intent.json` using `notes/toolkit/templates/research_intent.json`: record the parent response, separate mechanism/practical predictions, and Engineering findings reused or deferred, including confounds and unavailable reports. Nonlegacy REFINE requires a parent problem reference. Successful Spec publication releases the lock.

Keep selection notes in scratch until initialization. The default lock lease is 30 minutes; renew before expiry with `autoresearch node planning renew --token <token>`. A failed initialization leaves it held: correct and retry, or release with `node planning release --token <token>`. After expiry or loss, reacquire and reread before deciding again. Start implementation after registration releases the lock.

## 2. Implement and check

- Implement the registered method and run applicable checks in `notes/toolkit/references/semantic_criteria.md`. Check collective ordering, finite values, state transitions and contract applicability; record commands and limitations. A new algorithm need not be semantically equivalent to its Parent.
- Separate algorithmic effects, parameter changes and Engineering optimizations. Smoke tests and ordinary commits are allowed before formal evaluation.

## 3. Evaluate and refine the same method

- Complete `submission.yaml` with `mode: research` and fixed Spec references. Declare `timing.period_steps`: the full effective cycle, or null for a nonperiodic method. Only `requested_steps` is optional; host bounds remain authoritative. Maintain actual maintenance-event telemetry as described in `notes/toolkit/README.md`.
- Before each formal run, archive self-check scripts/logs needed by Analysis with `autoresearch evidence archive`. Temporary files and retired Worktrees are not persistent evidence.
- Use `coral eval` for quality and efficiency. **Normally finish within 2 successful scored formal runs; 3 successful scored formal runs and 10 total formal runs remain hard caps**, including failures, permitted same-SHA infrastructure retries, archived Attempts and all resumes. A scored regression counts as successful regardless of quality passing; pending runs reserve capacity.
  - **WHEN WAITING FOR RESULTS, USE 1800-SECOND LONG WAITS (`yield_time_ms=1800000`).** Apply this to `write_stdin` and any outer `functions.exec`/`functions.wait`; continue as soon as results return. **DO NOT use short polling loops, periodic log reads or waiting updates.**
- Inspect quality, cycle coverage, comparability and end-to-end timing; U is optional advice. Missing evidence is not a speedup. Use a third scored run only when existing results support a specific correction, or when it can resolve a key uncertainty that would affect method selection. Explain the basis in one or two sentences in the existing experiment reflection before running it. Remaining budget or an untested parameter alone is insufficient. Finish earlier when no worthwhile follow-up remains.
- Keep Parent/base_commit, Spec and hypothesis fixed. Core mechanism changes require a new Node, not another candidate here.

## 4. Finish and wait for the host

- Stop evaluating at either cap. To finish earlier when no justified refinement remains, use `autoresearch node complete --mode research` after all runs end.
- The host freezes Research, selects its fixed result, retires you, and starts independent Research Analysis over all Attempts and quality-gated Engineering concurrently. Child expansion waits for both; failed or unavailable quality explicitly skips Engineering. Do not start or claim another Node yourself.
- Optional subagents need bounded tasks and inherit all Sandbox, Evaluation and publication boundaries.
