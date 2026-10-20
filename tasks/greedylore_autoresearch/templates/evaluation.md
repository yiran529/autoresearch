# Evaluation Summary — <node_id>

> AUTO-GENERATED from CORAL Attempts, grader logs and committed submission.yaml.
> Derived index only; **do not edit manually**. Raw CORAL records are authoritative.

Last refreshed: <UTC timestamp>

## Research (all formal candidate Attempts)

Selected Research SHA: `<host-selected SHA after freezing, or pending selection>`

| Research Attempt / Commit (full SHA) | Status | Quality (measured) | Efficiency (measured) | Contract / environment | Evidence / retry logs |
|---|---|---|---|---|---|
| `<research candidate SHA; one row per candidate>` | `<terminal or pending>` | `<value/quality result or —>` | `<value ms or —>` | `<contract/version/hardware>` | `<attempt + eval logs; same-SHA infra retry logs if any>` |

## Engineering (one formal efficiency Attempt per iteration)

| Attempt / Commit (full SHA) | Change | Status | Quality source | Efficiency (step time) | Semantic self-check (agent-reported) | Contract / environment | Evidence |
|---|---|---|---|---|---|---|---|
| `<engineering SHA>` | `<buffer reuse>` | `<terminal status>` | `inherited (provisional) from <research SHA>` | `<value ms or —>` | `<pass/partial/fail/not_run; test logs/limits>` | `<contract/version/hardware>` | `<CORAL attempt + grader logs>` |

## Validity / incomparability

- <Quality failure, semantic self-check gaps and profiler-only results must be reported as such.>
- <Preserve every Research candidate, including failures, regressions and archived Attempts. After freezing, identify the host-selected quality-passing, comparable candidate with the highest valid score; if none qualifies, identify the failure representative and skipped Engineering.>
- <Do not rank incomparable hardware/contract/workloads; pending, failed or missing metrics stay visible as —.>

## Source of truth

- CORAL Attempts and immutable raw grader outputs; rerunning `refresh_node_evaluation(node_id)` must regenerate this file.
- Infrastructure retry on identical Research SHA does not create a second research version; preserve logs for all retry outcomes.
- No `engineering_history.md` or separate iteration database; the host freezes the selected Research SHA in workflow state.
