# Gap-driven GreedyLoRE AutoResearch

Task resources are host-published under the run's public `notes/toolkit/`. Each Agent's generated instructions locate that directory under its runtime shared directory (for Codex, `.codex/notes/toolkit/`). It contains workflow rules and route-specific references. It does not contain private evaluation inputs. Research Analysis receives a fixed code/evidence snapshot and only the parent analyses actually referenced by its Spec.

Research Analysis remains an independent session. Choose applicable initial diagnostics, investigate a principal unexplained observation, and report evidence, counterevidence and unknowns without proposing algorithms. Each Analysis has at most three diagnostic executions, each up to 600 seconds, with a separate GPU queue budget of one hour. See `GPU_COORDINATION.md` for host/client wait budgets.

## Engineering completion

A claim permits at most two distinct formal Engineering candidates. Same-SHA infrastructure retries are counted separately under the existing retry policy. The second candidate does not retire the Engineer: it must still author and submit the final summary.

Write JSON in authorized scratch outside the tracked Worktree, containing exactly:

- `summary`: non-empty text.
- `findings`: a non-empty array of non-empty strings covering measured results, each candidate, semantic checks and supported attribution.
- `limitations`: a non-empty array of non-empty strings, with uncertainty, missing evidence and check coverage.
- `attempt_shas`: all distinct formal candidate SHAs in this claim, exactly once, including failures and regressions.

Run `autoresearch engineering finish --analysis-file /absolute/scratch/final.json`. The host authenticates ownership, checks terminal Attempts and complete SHA coverage, seals the evidence and persists the submission before returning `finalizing` with the intended Analysis ref. This is not confirmation of completed publication. The host then publishes an immutable version and completes the claim. On interruption, reuse the identical body; conflicting submissions are rejected. No independent Engineering Analyzer is launched. Quality remains inherited (provisional) from the fixed Research SHA.

Historical Analyses remain readable. Existing runs use their saved config and role snapshots: update those deliberately before restarting the run. A legacy active Engineering claim with `finish_requested` but no author submission must resume its same owner and submit a final summary; it is not silently completed by an Analyzer. Already completed historical claims are unaffected.

## Validation

Scripted workflow simulations cover normal operation and injected failures, including same-SHA retry, claim races, publication and parallel Research. They use synthetic results, with no model, training or GPU calls. Scientific effectiveness requires a separate real low-rank pilot with matching budgets and legitimate saved training states; passing workflow tests does not establish scientific improvement.
