# Engineer-authored final Engineering Analysis

Write the final JSON in authorized scratch outside the tracked Worktree, then run `autoresearch engineering finish --analysis-file /absolute/scratch/final.json`.

Required fields:
- `summary`: non-empty English text stating the overall result and supported attribution.
- `findings`: non-empty string array, covering each claim Attempt and fixed SHA, all formal results including failures/regressions, semantic-check commands/results, and measured implementation headroom versus algorithmic costs or unresolved attribution. Cite exact evidence IDs and paths; profiling is separate from formal timing.
- `limitations`: non-empty string array with uncertainty, missing measurements, contradictory observations, check coverage and inherited (provisional) quality. Do not claim a theoretical lower bound.
- `attempt_shas`: every distinct formal Engineering candidate SHA in this claim, each exactly once, including failed outcomes. Same-SHA infrastructure retries remain one entry.

The host adds the immutable evidence envelope. Do not supply host provenance fields, propose a replacement algorithm, or write the shared Analysis yourself. The `finalizing` response acknowledges saved submission; publication and claim completion follow on the host, with no independent Engineering Analyzer.
