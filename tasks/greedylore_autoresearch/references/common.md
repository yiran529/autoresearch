# Common measurement semantics and evidence rules

**Use:** shared definitions and valid comparisons across communication algorithms. Experiment recipes belong in `initial_diagnostics.md`; symptom investigations belong in route refs. This is not a checklist to execute in full. Wall-clock, kernel/collective latency, overlap and runtime attribution belong to `engineer_runtime.md`, not the Research Analyzer.

## 1. Determine the actual communication semantics

| Axis | Question |
|---|---|
| When | every iteration, after H inner steps, event-triggered, stale, and actual dependencies |
| What | gradient, momentum, parameters, local model delta, optimizer statistics or mixture |
| How | dense, rank-r, quantized, sparse, composed representation; excluded tensors |
| Memory | EF, basis, momentum history, inner/outer optimizer state, worker replicas |
| Merge | all-reduce mean/sum, parameter averaging, outer optimizer, asynchronous apply |

Infer these from actual code, not an algorithm's name. For example, a DiLoCo outer step is not a per-step gradient All-Reduce. A low-rank module could compress accumulated model deltas rather than current gradients.

## 2. Unambiguous tensors and state

- `g_i,t`: raw local gradient at the relevant hook stage (clipping/unscale/accumulation matter); `gbar_t`: matching dense aggregate.
- `h_i,t`: actual local compressor input, potentially including EF/momentum; `C_i(h_i,t)`: local reconstruction; `cbar_t`: optimizer-visible global aggregate.
- `e_i,t`: residual state; `d_i,t`: actual tensor subtracted on local residual writeback; `x_i,t` and `Delta_i,t`: local weights and accumulated model delta.
- `u_t`: **actual** optimizer parameter update, not a synonym for gradient.

Only under `h=g+e, e_next=h-C(h)` does standard EF satisfy `C(h)-g=e-e_next`. Handle dense refresh, skipped layers, zero/inherited residual, and local/global variants separately. Generally `mean_i C_i(h_i) != C(mean_i h_i)`.

## 3. Six comparison-validity checks

1. Same object, shape, normalization, and the matching constraint (rank / support / real bit budget).
2. Same pre-step model, optimizer, EF, randomness, batch and hook iteration; if state is deliberately changed, label it.
3. Explicit worker-local/global, layer/bucket, ordinary/refresh/outer phase; never subtract incompatible errors.
4. State the actual representation constraints, including effective rank/bit budget/support/metadata where necessary for fair representation comparisons.
5. Match tokens/batches/schedules; separate inner training steps from outer synchronization rounds.
6. Multiple minibatches from one checkpoint are not independent training seeds; report ranges and uncertainty.

## 4. Separate evidence types

| Evidence | Directly supports | Does not automatically imply |
|---|---|---|
| Geometry | same-input representation/error difference | downstream training loss |
| Controlled mechanism | intervention changes an observation | universal efficacy |
| Optimizer local | same-state update or probe change | long-horizon convergence |
| Closed-loop training | quality change in the tested run | other tasks/topologies |

Synthetic matrices, mathematical counterexamples and empirical tests are useful but apply to different scopes. No observed problem is not proof of none; correlation is not causation.

## 5. Minimal evidence entry

`question → comparator/scope → actual measurement and source ID → competing causes → controlled test → support/counterevidence → downstream level → unresolved limit`. `No supported gap` / `inconclusive` are valid. Do not rewrite experiment predictions after results.

## 6. Foundation

- [Error Feedback Fixes SignSGD, ICML 2019](https://proceedings.mlr.press/v97/karimireddy19a.html): classic biased compression and EF; check the exact recurrence and assumptions.
- [EF21, 2021](https://arxiv.org/abs/2106.05203): alternative feedback/tracking view, not an identical plug-in formula.
