# Evaluation Summary — <node_id>

> AUTO-GENERATED from CORAL Attempts, grader logs and committed submission.yaml.
> Derived index only; **do not edit manually**. Raw CORAL records are authoritative.

Last refreshed: <UTC timestamp>

## Research (one formal Attempt SHA per Node)

| Research Attempt / Commit (full SHA) | Status | Quality (measured) | Efficiency (measured) | Contract / environment | Evidence / retry logs |
|---|---|---|---|---|---|
| `<unique research SHA>` | `<terminal or pending>` | `<value/quality result or —>` | `<value ms or —>` | `<contract/version/hardware>` | `<attempt + eval logs; same-SHA infra retry logs if any>` |

## Engineering (one formal efficiency Attempt per iteration)

| Attempt / Commit (full SHA) | Change | Status | Quality source | Efficiency (step time) | Semantic self-check (agent-reported) | Contract / environment | Evidence |
|---|---|---|---|---|---|---|---|
| `<engineering SHA>` | `<buffer reuse>` | `<terminal status>` | `inherited (provisional) from <research SHA>` | `<value ms or —>` | `<pass/partial/fail/not_run; test logs/limits>` | `<contract/version/hardware>` | `<CORAL attempt + grader logs>` |

## Validity / incomparability

- <Quality failure, semantic self-check gaps and profiler-only results must be reported as such.>
- <If more than one DISTINCT research SHA exists for this Node, flag contract violation; do not overwrite.>
- <Do not rank incomparable hardware/contract/workloads; pending, failed or missing metrics stay visible as —.>

## Source of truth

- CORAL Attempts and immutable raw grader outputs; rerunning `refresh_node_evaluation(node_id)` must regenerate this file.
- Infrastructure retry on identical Research SHA does not create a second research version; preserve logs for all retry outcomes.
- No `engineering_history.md`, no separate iteration database, no runtime Best selection.
