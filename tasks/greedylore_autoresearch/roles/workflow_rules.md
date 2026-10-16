# AutoResearch V1 — Shared Workflow Rules

Read project code and published Node evidence freely. Direct writes belong only in your own worktree or scratch; publish formal Attempts and Artifacts through the task commands. Private evaluation data and credentials are out of scope.

- A Node has exactly one formal Research Attempt SHA. Before it, smoke tests and ordinary commits are allowed; after it, algorithm corrections belong in a child Node.
- Run formal evaluation with `coral eval` after completing `submission.yaml`. Do not edit formal Attempts, evaluation logs, or Node history.
- Retry only a same-SHA infrastructure failure with `autoresearch retry <sha>`. Candidate/algorithm failures are evidence, not infrastructure retries.
- Publish one independent Analysis with `autoresearch analyze --node Nxxx --mode research|engineering`; use `--revise` only for an explicit new Analysis version.
- A valid published Research Analysis permits child expansion. Engineering Analysis is optional for child creation and may be produced with no Engineering Attempt.
- Record failed, regressed, incomparable, and missing results honestly. Never infer quality from CORAL's `improved` label or claim unmeasured efficiency/semantic guarantees.
