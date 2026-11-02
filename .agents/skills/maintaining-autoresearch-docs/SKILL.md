---
name: maintaining-autoresearch-docs
description: Use when changing CORAL AutoResearch behavior, seeds, evaluation contracts, scheduling, role instructions, command interfaces, artifacts, or recovery, including shared CORAL changes that affect the task.
---

# Maintaining AutoResearch Documentation

Keep the maintained AutoResearch documents accurate in the same change as the
implementation. They explain the system and its meaningful boundaries; exact
commands and schemas belong in the task's existing references and validators.
All paths below are relative to the CORAL repository root.

## Required impact check

For every change in scope, inspect the actual diff, relevant implementation and
saved-contract compatibility before finishing. Updating only the task README or
passing tests does not complete documentation synchronization. Identify which
maintained documents describe the affected behavior, then update those sections
in the same change. Apply user-requested scope limits to the edit; report any
necessary update outside that scope instead of silently omitting it.

| Change concerns | Maintained document under `docs/` |
| --- | --- |
| Seed support, module responsibilities, role handoffs, admission or scheduling | `greedylore-autoresearch-architecture.md` |
| Root preparation, research phases, evaluation tasks, gates, budgets, Engineering or compatibility | `greedylore-autoresearch-workflow.md` |
| Run layout, provenance, records, Analysis formats, evidence, retention or cleanup | `greedylore-autoresearch-artifacts.md` |
| Authentication, command acceptance/wait/recovery, sandboxes, resources or scratch | `greedylore-runtime-reliability-design-20261029.md` |
| Newly executed validation worth retaining | `greedylore-runtime-reliability-validation-20261029.md` |

A change can affect several documents. New Seed support, for example, can change
architecture, preparation and contract compatibility, provenance, and checkpoint
interpretation even when directory names remain unchanged. An internal refactor
may leave every existing description accurate: inspect the documents and state
that conclusion with its reason in the handoff; do not manufacture text edits.
Unrelated website or generic framework changes with no AutoResearch effect do
not require modifying these documents.

## Keep the explanation useful

Lead with current implemented behavior and link to its authoritative code or
command reference. Preserve research preferences, required operations, budgets,
permissions, and compatibility conditions. Keep historical contracts and results
under their recorded interpretation. Distinguish repository implementation from
active-run deployment, and successful CPU/synthetic checks from GPU or scientific
validation. Never update historical test counts as if they were rerun today.

Give each topic one primary home using the map above. Other documents provide
brief context and links. Cover the normal flow and consequential failure,
recovery and evidence boundaries; omit per-commit narration and exhaustive flag
catalogs. Put lasting rules into maintained sections rather than appending a
new dated temporary report at the top of `docs/`.

## Completion check

Check the final diff against the map, follow changed relative links and code
paths, and search affected documents for contradicted old terminology or values.
After a rename or deletion, check callers including documentation subdirectories;
preserve referenced paths when those callers are outside the authorized scope.
Run `git diff --check`. In the task handoff or PR, identify updated documents
(or the concrete no-impact reason), checks actually performed, and any remaining
limitation. Source edits alone do not deploy a live Run.
