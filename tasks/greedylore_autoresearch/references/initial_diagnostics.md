# Initial Diagnostic Toolkit (D1–D5)

**Purpose:** provide standard, reusable *first-pass experiments* before a root cause is known. Select a small applicable subset under existing evidence and budgets; a logical experiment need not consume one distinct GPU execution. Route-specific Diagnosis Cards explain follow-up investigations.

## 0. Choose experiments by current situation

| Situation | Good first choice | Follow-up |
|---|---|---|
| Root or first characterization of a compressed representation | D1, perhaps D2 with available state | D4 if formal quality differs |
| Representation or direction selection changed | D1 | D4 |
| EF, basis reuse, refresh changed | D2 | D1/D4 |
| Worker count/aggregation changed | D3 | D1/D4 |
| Quality markedly worse | relevant D1 or D2 plus D4 | D3 |
| NaN/divergence/collective error | D5 | relevant route |
| Local SGD or DiLoCo | D2+D3 on workers/outer state | D4 |

Record input provenance first. Without real saved gradient/EF/basis/optimizer state, use existing logs or code-level invariants. Synthetic inputs can check algebra or counterexamples but cannot establish actual training bottlenecks. First pass produces **observations**, not preordained diagnoses.

## D1 — Representation Quality

- **Applies to:** lossy representation of vector/matrix/model delta.
- **Needs:** same-step real pre-encoder tensor `H`, actual reconstruction/aggregate, rank/bit/support constraints, shape/dtype and applicable oracle.
- **Action:** measure `||H-Hhat||²/||H||²` (handle zero norms separately); compare with a defensible matched-constraint reference. Low rank: LR-01 rank/SVD/basis/selection. Quantization: bias/variance under fixed input and declared bit constraint. Sparse: optimal same-k support vs actual support.
- **Record:** numeric value, comparator gap, local/global scope, samples/layers/steps.
- **Interpretation:** high oracle error may indicate capacity; large excess over oracle points to estimation/selection/implementation. Does **not** establish training degradation.

## D2 — Cross-Step/State Behavior

- **Applies to:** EF, reused basis, momentum, local/outer optimizer, refresh/sync cycles.
- **Needs:** real state snapshots across comparable steps or cycle phases, event markers and step definitions.
- **Action:** track error, residual, basis energy, worker dispersion or optimizer state over ordinary/refresh/outer phases; verify actual recurrence in code.
- **Record:** distribution by phase, reset events, missing state.
- **Interpretation:** spikes can arise from changing inputs, drift, EF reset or outer momentum. Adjacent **different** gradients do not make a matched-input intervention.

## D3 — Distributed Consistency

- **Applies to:** local compression followed by aggregation, local updates, independent support/scale/basis per worker.
- **Needs:** same-time local communication objects, local reconstruction and actual aggregate.
- **Action:** compare `mean_i C_i(H_i)` and true aggregate to `mean_i H_i`; inspect cancellation and support/scale/subspace disagreement. For Local SGD compute weighted worker model dispersion.
- **Record:** worker count, weights, local/global errors and shared structure.
- **Interpretation:** local fidelity does not automatically transfer to a global error guarantee; do not invent independent `1/N` residual reduction.

## D4 — Optimizer Update Impact

- **Applies to:** MSE/quality mismatch, suspected nonlinear preconditioning or historical state amplification.
- **Needs:** same model weights, complete pre-update optimizer/EF/basis state, identical batch/RNG, legitimate dense and compressed update inputs.
- **Action:** isolated one-step shadow optimizer comparison; update cosine, norm ratio, difference; optionally identical independent development probe batch for after-step loss.
- **Record:** optimizer, parameter groups, exact state reset, probe provenance.
- **Interpretation:** small gradient MSE with large update change points to optimizer/state interaction; one step does not imply long-run quality.

## D5 — Numerical/Boundary Behavior

- **Applies to:** new implementation sanity, NaN, rank deficiency, dynamic range, distributed shape errors.
- **Needs:** zero inputs, rank-deficient and tied-spectrum inputs, extremes, dtype/world-size cases.
- **Action:** check finite values, shapes, shared randomness, state transitions and communication order; optionally small exact algebraic comparisons.
- **Record:** reproducible input and failing rule.
- **Interpretation:** synthetic counterexample may refute a universal compressor claim; it does not automatically show a reachable LLM training trajectory.

## Important-question selection

After the first pass, choose by: direct relation to training quality or optimizer behavior; measurable **excess** beyond an appropriate reference; replication across steps/layers/workers; a feasible controlled discrimination within budget. Focus on one, occasionally two. `No supported gap` is valid.
