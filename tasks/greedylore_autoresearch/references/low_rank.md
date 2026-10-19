# Low-Rank Communication: From First-Pass Diagnostics to Mechanism Diagnosis

**Applicable to:** PowerSGD, GreedyLore, PowerSGD+, and low-rank compression of gradients or parameter deltas. Read `common.md` first and use this reference selectively. **This is a diagnostic reference, not a mandatory experiment checklist or a source of proposed mutations.** Runtime performance analysis is documented only in the Engineer reference.

## 1. Algorithm landscape

| Mechanism | Typical implementation | Principal diagnostic question |
|---|---|---|
| Rank-r representation | Communicate matrix factors or projected coefficients | Is rank-r sufficient for the **actual compressor input**? |
| Power iteration / warm start | Reuse prior factors as an initial subspace | Does the basis cover important current directions? |
| Greedy directions in a full basis | Select top-r directions from energy scores | How much is lost due to basis restriction vs. scoring/selection? |
| Periodic SVD refresh | Rebuild a more accurate basis at intervals | How does the representation error and residual evolve around refresh steps? |
| Error feedback | Carry uncommunicated information forward | Does the actual residual recurrence compensate correctly, including refresh steps? |
| Distributed aggregation | Shared basis and reduced coefficients, or independently compressed messages | Do local fidelity metrics describe the optimizer-visible global update? |

Compression may act on `g_i`, `g_i+e_i`, momentum, or an accumulated local parameter delta `Δx_i`. Never compare an oracle for raw `g` to the actual reconstruction of a different EF-augmented input and call the difference an approximation gap.

## 2. Diagnostic targets

| Object | Metric | Directly supports | Does **not** establish |
|---|---|---|---|
| Rank capacity | Singular value energy, best rank-r Frobenius residual | Matrix-level representation limit | Tail energy is useless for training |
| Available basis | Best within-basis rank-r error vs. SVD | Extra restriction of stored basis | Basis refresh is necessarily too infrequent |
| Direction selection | Exact vs. estimated energy and selected energy | Extra selection loss | Any particular alternative improves convergence |
| Across-step state | Error / retained energy by refresh phase | Temporal patterns | Changes across unmatched minibatches are causal |
| EF | Local/global residual and recurrence | Information carryover / resets | Large local residual means global information loss |
| Optimizer | Matched pre-state realized parameter update | Local effect on optimization behavior | One probe predicts final loss |

## 3. Initial diagnostic recipes

### LR-01 — Matched-input approximation decomposition (preferred baseline)

**Trigger:** First characterization of a low-rank candidate or change to basis/selection/encoding. **Inputs:** one or more actual, sampled compressor-input matrices `H`, shape, effective rank `r`, stored basis `U` when available, selected directions `Ŝ`, actual decoded output, and local/global scope. Stratify a small sample by layer/bucket and normal vs. refresh phase, reporting sample count and absolute input norms.

When `U` has orthonormal columns and the algorithm selects a rank-r subset of that basis with **orthogonal projection**, define:

- `E_svd = ||H-H*_r||_F² / ||H||_F²`, where `H*_r` is the best rank-r truncated SVD.
- `E_basis = min_{S⊂[d], |S|=r} ||H-U_S U_Sᵀ H||_F² / ||H||_F²`, computed by ranking exact energies `||u_jᵀ H||²`.
- `E_selected = ||H-U_Ŝ U_Ŝᵀ H||_F² / ||H||_F²`, the directions actually chosen by the method, evaluated using that same exact projection.
- `E_impl = ||H-Ĥ_actual||_F² / ||H||_F²`, the actual reconstructed output, recorded separately.

Under matched conditions, `E_svd ≤ E_basis ≤ E_selected`:

| Term | Meaning |
|---|---|
| `E_svd` high | Rank-r is intrinsically limited for this matrix under Frobenius reconstruction |
| `E_basis-E_svd` high | Restriction to the available basis incurs extra error |
| `E_selected-E_basis` high | Scoring/selection loses energy relative to exact within-basis selection |
| `E_impl` differs from `E_selected` | Investigate precision, projection implementation, aggregation and object definitions before attributing the difference |

**Inapplicability:** No full stored basis, non-orthonormal basis, different ranks, or reconstruction not in this family. Then report directly comparable SVD and actual residuals only, and explicitly mark the decomposition N/A. Handle `||H||=0` separately. SVD is an offline oracle, not free online communication.

**Deliverable:** Input and step definition, sample coverage, valid error components and the largest observed structural restriction—not a claim about final training performance.

### LR-02 — Refresh-window and EF-state profile

**Trigger:** Warm starts, periodic refresh or EF, or a temporal pattern in LR-01. **Inputs:** actual trajectory with `hook_step`, refresh phase/events, input/output, basis, per-worker `e_i`, and phase-dependent recurrence from code.

Record ordinary vs. refresh steps and comparable phases across windows, separating raw gradient, EF-augmented input, and optimizer-visible global output. For classical ordinary-step EF `h=g+e; e_next=h-c`, check `c-g=e-e_next`. If refresh clears/inherits residuals or performs dense aggregation, derive that branch separately. Report both `||mean_i e_i||` and `mean_i||e_i||` to expose cancellation.

A difference between two adjacent minibatch steps is **not** a same-input causal refresh comparison. Use fixed-input replay or a controlled intervention for stronger attribution.

### LR-03 — Optimizer update impact

**Trigger:** Significant structural gap with unknown optimization relevance, or formal quality and reconstruction results disagree. **Inputs:** restorable model, full optimizer, EF, basis, RNG and batch pre-update state.

With matched pre-state, compare real dense and compressed optimizer parameter increments (cosine, norm ratio, discrepancy); optionally measure a one-step loss on independent development probe data. Treat AdamW and Muon separately.

One-step gain is not proof of improved final training.

## 4. Symptom → diagnosis cards

### LR-RANK — Best rank-r SVD has a large residual

- **Alternatives:** The matrix has a heavy spectral tail; or the sample is an outlier layer/batch, rank/dimension ratios differ, or normalization distorts comparisons.
- **Discriminator:** Stratify real inputs by layer, stage and worker; measure `∑_{j≤r}σ_j² / ||H||²` under fixed r and link the observation to actual optimizer updates if relevant.
- **Supported:** A rank-r Frobenius representation limit for the measured inputs.
- **Not supported:** The spectral tail is noise, final quality must suffer, or a specific rank is optimal.

### LR-BASIS — Within-basis optimum is much worse than SVD

- **Alternatives:** Stale tracking; even a freshly built basis using the current method is unsuitable; or minibatch fluctuation.
- **Discriminator:** Fix *the same H*; compare stored basis, basis freshly generated by the existing algorithm's defined procedure, and exact SVD; examine repeated phase-conditioned retained energy, projector distances and spectral gaps.
- **Interpretation:** Old basis poor + fresh basis noticeably better supports tracking lag; fresh basis still poor does not specifically implicate staleness. With tied singular values, projector angles can be misleading; prioritize energy.

### LR-SELECTION — Actual choice is much worse than the within-basis optimum

- **Alternatives:** Sketch noise, nonlinear top-r selection bias, sorting/indexing/dtype/orthogonalization differences.
- **Discriminator:** With identical `H` and basis compute exact per-direction energies, compare estimated scores and selected retained energy; resample sketches with fixed input. Separate exact projection from BF16 implementation when relevant.
- **Interpretation:** Low index overlap need not mean substantial energy loss; confirmed energy loss need not imply training harm.

### LR-TEMPORAL — Error rises near end of window or jumps at refresh

- **Alternatives:** True subspace drift, EF accumulation/reset, phase-dependent compression inputs, or minibatch differences.
- **Discriminator:** Audit each code branch's inputs and state writes; replay the same H under different existing stored states; reproduce phase effects across windows and check exact EF bookkeeping.
- **Interpretation:** Phase correlation is not refresh causality; local EF size is not a global-loss metric.

### LR-UPDATE — Reconstruction and actual quality disagree

- **Alternatives:** Momentum/preconditioning changes direction; norm changes; stochastic variation or historical state; error lies in update-insensitive components.
- **Discriminator:** Full-state optimizer shadow update, direction vs. magnitude, limited independent probe; report whether short real closed-loop evidence exists.
- **Interpretation:** Report the reached evidence level, not a universal convergence claim.

## 5. Mathematical toolbox (conditional)

| Tool | Prerequisites | Purpose | Common misuse |
|---|---|---|---|
| Eckart–Young–Mirsky | Same matrix, rank, Frobenius objective | Minimal rank-r residual | Claims optimal training quality |
| Wedin / Davis–Kahan | Appropriate perturbation and nontrivial spectral gap | Subspace changes vs. matrix perturbations | Over-interprets angles for tied eigen/singular values |
| Random-sketch moments/selection probabilities | Correct sketch law and conditional independence | Diagnose energy score quality | Score unbiasedness ⇒ top-r unbiasedness |
| Contractive compressor theory | Conditional property of the **real**, history-dependent compressor | EF/error bounds | All rank-r outputs automatically contract like SVD |
| EF/window Lyapunov | Correct recurrence and refresh boundary | Multi-step residual control | Extends ordinary EF identity across residual reset |
| Smoothness/descent | Matched update/gradient hypotheses | Relate perturbation to local descent | Proves arbitrary AdamW/Muon training behavior |

## 6. Paper cards

1. **[PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization](https://arxiv.org/abs/1905.13727)** (Vogels et al., NeurIPS 2019). **Why:** Reduce communication with All-Reduce-compatible factors. **How:** Power iteration, warm start and EF. **Useful for:** factor aggregation, reconstruction-quality comparisons and historical basis tests. **Limits to check:** evolving-subspace coverage and difference from optimal SVD; do not import later PowerSGD+ guarantees.
2. **[Greedy Low-Rank Gradient Compression for Distributed Learning with Convergence Guarantees (GreedyLore)](https://arxiv.org/abs/2507.08784)** (Chen et al., 2025). **Why:** Combine greedy energy selection and convergence guarantees. **How:** Semi-lazy basis refresh, current direction scoring/selection, EF. **Useful for:** LR-BASIS, LR-SELECTION, randomized-selection analysis. **Limits to check:** sketch selection, stale basis, refresh-state behavior and optimizer/randomness assumptions in the actual revision.
3. **[From PowerSGD to PowerSGD+: Low-Rank Gradient Compression for Distributed Optimization with Convergence Guarantees](https://arxiv.org/abs/2509.11254)** (Xie et al., 2025). **Why:** Address convergence limitations of original PowerSGD. **How:** Periodic SVD refreshing and window-level analysis. **Useful for:** LR-TEMPORAL, residual bounds and counterexamples. **Limits to check:** refresh-state assumptions, window-length dependency and actual workload.
4. **[Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://proceedings.mlr.press/v97/karimireddy19a.html)** (Karimireddy et al., ICML 2019). **Why:** Understand biased compression and residual feedback. **How:** Error accumulation and virtual iterates. **Useful for:** true recurrence and conditional compressor assumptions. **Limits:** Do not assume the same guarantees for arbitrary AdamW/Muon implementations.

**Interpretation example (hypothetical, not experiment data):** `E_svd=0.08`, `E_basis=0.36`, `E_selected=0.41`. The basis restriction (`0.28`) exceeds the selection excess (`0.05`) *for this input*. It does **not** establish that refresh is too infrequent or that final model quality is impaired.
