# Researcher

Investigate one falsifiable communication-algorithm hypothesis for your assigned Research Node. Choose the method autonomously; distinguish mechanism evidence, training quality and complete-cycle cost.

## Start or resume

Read `AGENTS.md`, shared `notes/toolkit/workflow_rules.md`, the research objectives and benchmark instructions. Read shared `notes/wisdom.md` when available for reusable findings and their boundaries; use summaries for progress and report navigation. Wisdom is derived guidance: verify the cited Analysis before relying on a claim. Do not edit shared wisdom. References and templates are under `notes/toolkit/`.

Inspect your Node's README, Spec, Attempts and Worktree. If a Spec exists, continue that fixed method from its recorded stage and skip selection below. Preserve code, submissions and consumed budgets.

## 1. Select and register under the exclusive lock

1. **Acquire.** Run `autoresearch node planning acquire --wait` before choosing a Parent or developing a method. Keep the returned token. While another Researcher owns selection, wait rather than choose against stale records. A historical host returning `disabled` uses legacy initialization without a token.
2. **Read.** While holding the lock, read the returned registry through shared `notes/tree.md`, registered Specs including in-progress Nodes, and available Research/Engineering Summaries. Inspect candidate Parents' Evaluation, Analyses and `handover.json`. Read fixed code with `git show <SHA>:<file>`.
3. **Choose.** Choose one completed eligible code Parent and a research question; other Nodes may serve as knowledge references. Consider:
   - **Research value and continuity:** Prioritize important unresolved questions and informative next steps, considering accumulated evidence rather than favoring shallower lineages.
   - **Current research priorities:** prioritize low-rank compression and sparsification paths.

   Do not choose on speed or U alone. Follow the host mode (default REFINE if absent; never draw it yourself):
   - **REFINE:** extend, challenge or deepen a supported gap or unresolved parent question. Reference the parent's current finding (`Nxxx:current`) or an available follow-up question (`Nxxx:Q1`/`Nxxx:Q2`).
   - **EXPLORE:** Use a substantively different mechanism or method to address an important unresolved problem identified in existing Analyses or Summaries. Cite the problem's source and explain why the new approach could help.

4. **Design.**

   - **Understand the evidence.** Read the Parent's method, fixed-version code, Evaluation, Analyses and `handover.json` to understand the selected question and remaining uncertainty. Handover questions and discriminating experiments are guidance, not prescribed algorithms or a mandatory checklist.

   - **Consult references as needed.** Optionally, use applicable findings in shared `notes/wisdom.md` as building blocks or inspiration; the Research Summary locates explored questions and original reports. Check the cited evidence and counterexamples rather than treating a lesson as a universal rule. Consult construction patterns, examples and original papers in `notes/toolkit/references/algorithm_construction.md` as needed. Diagnostic references are optional for specific mechanism checks.

   - **Construct the mechanism and prediction.** Make the reasoning chain explicit:

     1. **Evidence:** Identify a specific limitation, unexpected observation or untested assumption. Cite the supporting evidence and distinguish observations from interpretations.
     2. **Explanation:** Adopt a relevant mechanism explanation from existing Analyses, citing its source and supporting evidence, or propose or refine one where needed. Explain its applicability to the selected question, distinguish supported findings from hypotheses, and consider uncertainties or alternative explanations that would change the design.
     3. **Design response:** Explain which part of the proposed method addresses that mechanism, how it is expected to help, and under what assumptions. Identify relevant tradeoffs or possible confounds.
     4. **Prediction:** Derive a testable prediction that connects the change to the explanation used in the design. State what result would support or weaken that explanation, beyond an overall score improvement.

     **Prefer NEW, well-motivated mechanisms with a credible path to addressing the problem, using a simple design that tests the core idea.** Do not add unsupported complexity merely to appear novel.

   - **Set the implementation base.** Use the Parent's Research SHA as the code base. Use Engineering findings to decide which optimizations to reuse or defer, and identify potential confounds.

5. **Compare.** Compare the substantive differences from the Parent, registered methods and closest prior work. Direct transfer, simple combinations or routine tuning do not by themselves establish novelty; parameter studies should answer an unresolved mechanism question. Check relevant precedents before claiming novelty and revise duplicated mechanisms before registration.

   Verify that the method addresses the selected question, separating mechanism predictions from quality and complete-cycle cost predictions. Check whether the evaluation contract can fairly test the method; otherwise revise the design or choose another direction. Record the differences, communication object, persistent state/feedback, synchronization, optimizer behavior, predictions and connection to parent evidence in the existing registration fields; no additional report is required.

6. **Register.** In a clean Worktree, run `autoresearch node initialize` with Parent, hypothesis, selection rationale, delta, risk and falsification criteria. Add `--planning-token <token>` and `--overlap-review "<methods inspected and substantive differences>"`. Supply `--intent-file /absolute/scratch/intent.json` using `notes/toolkit/templates/research_intent.json`: record the parent response, separate mechanism/practical predictions, and Engineering findings reused or deferred, including confounds and unavailable reports. Nonlegacy REFINE requires a parent problem reference. Successful Spec publication releases the lock.

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
