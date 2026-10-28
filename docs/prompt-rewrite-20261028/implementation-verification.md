# Formal prompt application

Formal role prompts, shared workflow, independent Analyzer/Memory prompts and staged startup/resume messages are in English. Chinese drafts remain review material.

Validation after the English correction:

- Dynamic assignment/startup, delivered assets, Analysis recovery, research registration and Memory tests: **104 passed**.
- Ruff passed for changed Python files (existing test assertions excluded with `--ignore S101`); `git diff --check` passed.
- A regression check rejects Chinese text in formal role/prompt files and staged startup messages.
- Before translation, the staged successful lifecycle simulation passed using fake model/CPU execution outside the isolated process namespace. It did not run GPU experiments or compare research quality.

The two pre-existing unstaged legacy role deletions are excluded from the commit.

The user identified that the first English translation had converted approved
prose into bullets. This was corrected before deployment: all five prompts now
preserve the Chinese drafts' heading levels, paragraph groups, list groups/item
counts, tables and code blocks. Optionality, prohibitions and actor responsibilities
were checked paragraph by paragraph. Astra's separate fidelity review confirms the
corrections; `translation-structure.json` records the final structural comparison.
After these text corrections, **119 relevant tests passed** and `git diff --check`
passed. The separately authorized Analyzer context addition is excluded from
translation structure parity.

## Analyzer public context

New Analysis versions receive bounded snapshots of other public Node texts and
global navigation/summaries/wisdom. Retries reuse the original snapshot; revisions
capture newer context. Existing reports and unfinished legacy diagnostic tasks
retain their original evidence identity. No live notes root is mounted.

Diagnostic ledgers continue to bind the local formal input manifest; context is
bound separately to that hash and task version. The published Analysis manifest
contains both. Memory stages current raw evidence before historical context under
its existing shared limits.

- Expanded regression in a temporary source snapshot preserving committed legacy
  assets: **244 passed** (dynamic prompts, workflow/Parent eligibility, concurrent
  Analysis, recovery, citations, context, Memory, registration and delivered assets).
- Successful staged lifecycle with fake model and CPU execution: **1 passed,
  5 deselected in 42.96s**; no GPU training or model calls.
- Astra medium independent final regression: **75 passed**. Both review blockers
  were fixed; see `astra-analyzer-context-review.md`.
- Ruff for changed implementation and new/updated tests, plus `git diff --check`,
  passed. The user's pre-existing legacy deletions remain outside both commits.
