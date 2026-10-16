# AutoResearch V1 — Shared Workflow Rules

Read project code and published Node evidence freely. Direct writes belong only in your own worktree or scratch; publish formal Attempts and Artifacts through the task commands. Private evaluation data and credentials are out of scope.

- A Node has exactly one formal Research Attempt SHA. Before it, smoke tests and ordinary commits are allowed; after it, algorithm corrections belong in a child Node.
- Each Researcher Agent owns one reserved Node and one Codex Session. Choose a Parent, then run `autoresearch node initialize` with the required hypothesis and rationale fields. The command creates the fixed Node Spec; do not write Tree or Spec directly.
- An Engineer first inspects the Tree and published evidence, then runs `autoresearch engineering claim Nxxx` for one eligible Node. A failed claim requires choosing another target. Only if no eligible unclaimed target remains, run `autoresearch engineering cancel` before claiming and exit.
- Run formal evaluation with `coral eval` after completing `submission.yaml`. Do not edit formal Attempts, evaluation logs, or Node history.
- Retry only a same-SHA infrastructure failure with `autoresearch retry <sha>`. Candidate/algorithm failures are evidence, not infrastructure retries.
- Publish one independent Analysis with `autoresearch analyze --node Nxxx --mode research|engineering`; use `--revise` only for an explicit new Analysis version.
- A valid published Research Analysis permits child expansion. Engineering Analysis is optional for child creation. Finish owned work with `autoresearch node complete --mode research|engineering` and exit; an Engineering completion requires a terminal Engineering Attempt and Analysis.
- Record failed, regressed, incomparable, and missing results honestly. Never infer quality from CORAL's `improved` label or claim unmeasured efficiency/semantic guarantees.
