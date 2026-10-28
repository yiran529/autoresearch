# Researcher Execution Supplement

This supplement specifies registration, budget counting, code-version binding and Final recovery conditions not expanded in the Researcher role. Shared execution boundaries are in `workflow_rules.md`; request fields and examples are in `references/research_commands.md`.

## Registration and planning lease

- The current staged protocol uses `autoresearch research`. Report an unavailable protocol or configuration mismatch; do not switch to `coral eval`, legacy initialization/finish entry points or independent GPU work.
- Renew before expiry with `autoresearch node planning renew --token <token>`. Release an abandoned topic with `autoresearch node planning release --token <token>`. Failed registration retains the lease. After loss or expiry, reacquire it and reread the registry.
- The host generates the Spec and compatible handover fields from registration. No separate research intent is required.

## Budgets and counting

These default limits support research planning. The current run follows host records and its saved contract.

| Item | Supplementary rule |
|---|---|
| Development training trajectories | Hard limit 3; before executing a third, explain an evidence-supported correction or an important unknown affecting candidate selection |
| Designs | At most 3, including the initial design and designs screened only by timing |
| Development timing | 5 completed / 15 total |
| Research diagnostics | 10 successful / 20 total, at most 600 seconds each; failures, timeouts and interruptions count toward total |
| Infrastructure recovery | Only for host-classified infrastructure failures; default limit 2, subject to the corresponding execution counts |

- Failures, voluntary stops and rewind/restart consume training trajectories; pending work reserves capacity. Pause/resume does not reset steps, tokens, schedules or consumed capacity.
- Completed but incomparable timing still counts as completed. Failed timing and retries accumulate under the protocol.
- Preserve failures, regressions, incomparable and unavailable evidence; do not blindly rerun or use empty commits to manufacture retries.

## Code versions and continuation

- The first accepted training or timing request binds the design SHA. Subsequent code changes require an appended design. `research design` only appends a design record; it does not modify code or start training.
- Exact continuation at the same iteration/SHA requires restoring the model, optimizer, compressor/EF, schedule, data position and RNG. A training prefix may be reused only when method behavior and necessary state match.
- Compression policies must not change fixed data, training budgets, optimizers, seeds, precision or evaluation settings.
- Do not add profiling or gradient instrumentation to formal timing.

## Final recovery and Parent eligibility

- Final infrastructure recovery defaults to at most 2 retries per component. Preserve other completed components; reuse within a component is only for recovery of the same frozen evaluation. Vision retries start from scratch.
- Failed components and components that exhaust retries retain their terminal states. Other mandatory components must still finish or be explicitly skipped.
- Parent eligibility requires published Research Analysis and Engineering either completed or explicitly skipped by the host for quality failure/unavailability.
- U is advisory only. It does not determine research mode, Parent selection or candidate nomination.
