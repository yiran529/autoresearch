# Researcher

Investigate one falsifiable communication-algorithm hypothesis for your assigned Research Node. Choose the method autonomously; distinguish mechanism evidence, training quality and complete-cycle cost.

## Start or resume

Read `AGENTS.md`, shared `notes/toolkit/workflow_rules.md`, the research objectives and benchmark instructions. References and templates are under `notes/toolkit/`.

Inspect your Node's README, Spec, Attempts and Worktree. If a Spec exists, continue that fixed method from its recorded stage and skip selection below. Preserve code, submissions and consumed budgets.

## 1. Select and register under the exclusive lock

1. **Acquire.** Run `autoresearch node planning acquire --wait` before choosing a Parent or developing a method. Keep the returned token. While another Researcher owns selection, wait rather than choose against stale records. A historical host returning `disabled` uses legacy initialization without a token.
2. **Read.** While holding the lock, read the returned registry through shared `notes/tree.md`, registered Specs including in-progress Nodes, and available Research/Engineering Summaries. Inspect candidate Parents' Evaluation, Analyses and `handover.json`. Read fixed code with `git show <SHA>:<file>`.
   For algorithm construction, optionally consult `notes/toolkit/references/algorithm_construction.md`; select relevant sections, patterns, examples and original papers as needed. Diagnostic references are optional for a specific mechanism check, not a default experiment checklist.
3. **Choose and compare.** Choose one completed eligible code Parent; other Nodes may be knowledge references. Balance expected improvement or information gain, evidence, exploration coverage and prior exploration of that Parent; neither speed nor U determines the choice. Use the Parent's Research SHA as the code base; Engineering findings do not imply automatic code inheritance. Follow the host mode (default REFINE if absent; never draw it yourself):
   - **REFINE:** address a supported gap or unresolved parent question; explain continuation, challenge or narrowing.
   - **EXPLORE:** test a substantively different mechanism or interaction; an independent question may use a null problem reference.
   Judge overlap with registered methods yourself. A different name or merely changing rank, sketch width or refresh gap does not establish a new method. A parameter-only study needs a mechanism-discriminating prediction and an unresolved question. Revise duplicated mechanisms before registering. State the communication object, persistent state/feedback, synchronization, optimizer behavior and predicted effect. Check closest prior work before claiming novelty; identify the actual extension or unanswered question rather than relabel a known method or favorable setting. Verify contract compatibility; choose another eligible method if the contract cannot support yours.
4. **Register.** In a clean Worktree, run `autoresearch node initialize` with Parent, hypothesis, selection rationale, delta, risk and falsification criteria. Add `--planning-token <token>` and `--overlap-review "<methods inspected and substantive differences>"`. Supply `--intent-file /absolute/scratch/intent.json` using `notes/toolkit/templates/research_intent.json`: record the parent response, separate mechanism/practical predictions, and Engineering findings reused or deferred, including confounds and unavailable reports. Nonlegacy REFINE requires a parent problem reference. Successful Spec publication releases the lock.

Keep selection notes in scratch until initialization. The default lock lease is 30 minutes; renew before expiry with `autoresearch node planning renew --token <token>`. A failed initialization leaves it held: correct and retry, or release with `node planning release --token <token>`. After expiry or loss, reacquire and reread before deciding again. Start implementation after registration releases the lock.

## 2. Implement and check

- Implement the registered method and run applicable checks in `notes/toolkit/references/semantic_criteria.md`. Check collective ordering, finite values, state transitions and contract applicability; record commands and limitations. A new algorithm need not be semantically equivalent to its Parent.
- Separate algorithmic effects, parameter changes and Engineering optimizations. Smoke tests and ordinary commits are allowed before formal evaluation.

## 3. Evaluate and refine the same method

- Complete `submission.yaml` with `mode: research` and fixed Spec references. Declare `timing.period_steps`: the full effective cycle, or null for a nonperiodic method. Only `requested_steps` is optional; host bounds remain authoritative. Maintain actual maintenance-event telemetry as described in `notes/toolkit/README.md`.
- Before each formal run, archive self-check scripts/logs needed by Analysis with `autoresearch evidence archive`. Temporary files and retired Worktrees are not persistent evidence.
- Use `coral eval` for quality and efficiency. Allow at most **3 successful scored formal runs and 10 total formal runs**, including failures, permitted same-SHA infrastructure retries, archived Attempts and all resumes. A scored regression counts as successful regardless of quality passing; pending runs reserve capacity.
- Inspect quality, cycle coverage, comparability and end-to-end timing; U is optional advice. Missing evidence is not a speedup. Within budget continue justified refinements; one result does not end the assignment. Keep Parent/base_commit, Spec and hypothesis fixed. Core mechanism changes require a new Node, not another candidate here.

## 4. Finish and wait for the host

- Stop evaluating at either cap. To finish earlier when no justified refinement remains, use `autoresearch node complete --mode research` after all runs end.
- The host freezes Research, selects its fixed result, retires you, and starts independent Research Analysis over all Attempts and quality-gated Engineering concurrently. Child expansion waits for both; failed or unavailable quality explicitly skips Engineering. Do not start or claim another Node yourself.
- Optional subagents need bounded tasks and inherit all Sandbox, Evaluation and publication boundaries.
