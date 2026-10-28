# Formal prompt application

Formal role prompts, shared workflow, independent Analyzer/Memory prompts and staged startup/resume messages are in English. Chinese drafts remain review material.

Validation after the English correction:

- Dynamic assignment/startup, delivered assets, Analysis recovery, research registration and Memory tests: **104 passed**.
- Ruff passed for changed Python files (existing test assertions excluded with `--ignore S101`); `git diff --check` passed.
- A regression check rejects Chinese text in formal role/prompt files and staged startup messages.
- Before translation, the staged successful lifecycle simulation passed using fake model/CPU execution outside the isolated process namespace. It did not run GPU experiments or compare research quality.

The two pre-existing unstaged legacy role deletions are excluded from the commit.
