# Engineer command guide

Operational reference for the Engineer role. This document owns managed diagnostic execution, formal timing, and final Analysis submission interfaces. Follow the Engineer role for objectives, semantic obligations, stopping conditions and Analysis requirements; use [the runtime reference](engineer_runtime.md) for performance diagnosis and optimization methods. Shared execution, evidence, permissions and recovery rules remain in `workflow_rules.md`. Current limits follow the host claim and saved evaluation contract.

On first entry, use `autoresearch engineering claim Nxxx` from a clean Worktree to confirm ownership, the fixed Research SHA and current `run_limits`. On recovery, inspect the existing claim, Evaluation/Attempts and completion state without resetting budgets.

Keep waiting on running commands through the shared **1800-second long-wait rule**, using the original terminal session. A lost reply reuses the original request identity and unchanged payload; deliberate new execution uses a new ID. See “Waiting and request identity” in `workflow_rules.md` for the full protocol.

## 1. Controlled GPU diagnostics

- An active, non-finalizing Engineer claim may use `autoresearch diagnostic run`. The host authenticates the calling process and its Node binding; no additional Analyzer is launched. `engineer_diagnostics: false` disables this capability.
- Commit tracked candidate changes first; an ordinary Git commit does not create a formal Attempt. Place scripts in your Worktree or `/tmp/<your-agent-id>/diagnostics/experiments/`. Prefer scratch so diagnostic instrumentation stays out of committed formal candidates.
- Example: `autoresearch diagnostic run --scripts-dir /absolute/path/to/experiments --gpus 4 --timeout 600 -- python experiments/probe.py --output results/probe.json`. The host stages scripts as `experiments/` in a separate workspace, freezes committed HEAD as `code/<sha>/`, and sets `PYTHONPATH` and `CORAL_DIAGNOSTIC_CODE` to that code. Read data only from host-configured runtime roots. Use paths relative to the diagnostic workspace, not your live Worktree.
- Only `experiments/`, `results/` and ephemeral sandbox `/tmp` are writable during execution. Put durable output in `results/`. The host retains the script/code snapshots, command, candidate/Research SHA, GPU identities, stdout/stderr and result files under `public/diagnostics/<node>-<agent-id>/records/<request-id>/`. These records are read-only to agents; the returned record and result paths identify them.
- The separate diagnostic budget is 10 successful and 20 total executions across resumes, 1–4 GPUs and at most 600 seconds per execution; GPU queue wait is at most 3600 seconds. Failures/timeouts consume total executions; requests that never start do not. Diagnostics share GPU allocation with formal evaluations but produce no official score and consume no formal-run slots.
- Inspect failed records, fix scripts and use a new request ID to execute again. Reuse `--request-id` with identical arguments only to retrieve an existing result. Cite both successful and failed diagnostics in your final Analysis; do not add profiler durations to claim formal step time. Wait for all diagnostics before `engineering finish`; the host seals their durable records with the final Analysis and rejects subsequent diagnosis.

## 2. Formal timing

- Read the claim's design source and use its fixed Research SHA as the anchor. Developed Nodes require the frozen selected iteration in `research.json`; Root uses its fixed baseline Spec/A0. Do not substitute the initial Spec or last iteration for a developed Node's selection.
- After semantic checks, commit the candidate and run `autoresearch engineering timing --sha <full-HEAD-SHA> --request-id <id>`. It waits for the grader's result without committing files, using the existing complete-cycle/event/hardware contract and formal budgets. Quality remains inherited, not remeasured. Inspect Evaluation/Attempts on resume.
- After a channel interruption repeat the same ID/SHA/flags to recover that execution. An intentional infrastructure retry uses `--retry` and a new ID, retaining the SHA and consuming the existing retry/formal allowances. Other failures require a corrected candidate; no empty commits or tuning mode.

## 3. Final Analysis and submission

After at least one terminal formal result and once the role's stopping condition or a formal-run limit is reached, write the final Analysis yourself. Wait for all diagnostics before submitting. Preserve self-check scripts and logs that need citation with `autoresearch evidence archive`; use the returned public-relative paths and follow the shared archive limits and evidence rules.

Write a JSON file in authorized scratch, outside the tracked Worktree, containing exactly these fields.

| Field | Content |
|---|---|
| `summary` | Nonempty summary |
| `findings` | Nonempty string array of key findings and evidence |
| `limitations` | Nonempty string array of limitations affecting judgment |
| `attempt_shas` | Every distinct formal candidate SHA in the current claim, exactly once, including failures and regressions |

Cite actual run/request IDs, persistent paths and check results inside the text fields. Include successful and failed diagnostics as well as every formal Attempt; do not claim profiler durations as formal step time. Quality remains `inherited (provisional)`.

Submit with `autoresearch engineering finish --analysis-file /absolute/scratch/final.json` and wait for host validation, versioned publication, and retirement. The host verifies ownership, terminal Attempts and complete SHA coverage, freezes the submission, seals the evidence envelope and publishes an immutable version before completing the claim. A `finalizing` response confirms persisted submission, not completed publication. Reuse the identical submission after interruption; do not submit new candidates or directly edit shared artifacts. No independent Engineering Analyzer is started.
