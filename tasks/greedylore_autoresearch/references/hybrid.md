# Hybrid Communication: Cross-Mechanism Interaction Diagnosis

**Scope:** low rank + quantization, low rank + sparse correction, any representation compression + infrequent synchronization, and candidates that jointly change the communication object, frequency or retained state. Read `common.md` plus applicable route refs. **Combining known mechanisms is not automatically novel**; the diagnostic task is to determine whether their interaction is supported, complementary, conflicting or still unknown.

## 1. Algorithm landscape

| Combination | What changes | Diagnostic focus |
|---|---|---|
| Compression × local updates | communicated object may become accumulated model delta `Δ_i`, not `g_t` | target distribution, sync window, outer optimizer |
| Low rank × quantization | low-rank factor precision/scale may amplify reconstructed error | factor bitstreams and final matrix |
| Low rank × sparsification | residual spectrum and sparse support depend on application order | matched-object representation error |
| EF × outer optimizer | residual units, writeback and update cadence interact | inner/outer event accounting |
| Layer-wise mixed compressors | representation differs across layer/block | per-layer reconstruction and effective representation constraints |

**Key observation:** combinations may change each other's **input tensor**, statistics and state. A per-step gradient SVD oracle is not automatically an oracle for compressed Local-SGD accumulated deltas.

## 2. Diagnostic targets

| Target | Measurement | Supports | Does not prove |
|---|---|---|---|
| Representation complementarity | identical H and equal-byte reconstruction across individual/combined methods | whether two representations capture distinct structures | training improvement |
| Frequency-representation interaction | spectrum/sparsity/quantization error of actual target across H | local accumulation alters compressibility | better final quality |
| State interactions | EF, basis/mask/scale, outer optimizer timeline | whether accounting/reset/delay differs | state difference is inherently a bug |
| Quality interaction | comparable control/A/B/AB quality at matched training tokens | whether the mechanism effects interact | a single best score establishes synergy |

## 3. Initial recipes

### HY-01 — Same-object, equal-budget representation comparison (D1/D3)

**When:** two coding methods operate on the same object (e.g., quantized low-rank factors or sparse correction). **Inputs:** actual `H`, individual/combined implementations in matched state, real bitstreams including scales/indices. **Procedure:** measure final reconstruction, intermediate errors, application order and actual bytes under a stated budget. **Output:** whether information losses appear complementary. **Boundary:** reconstruction complementarity is not automatically training improvement.

### HY-02 — Frequency × compression window profile (D1–D4)

**When:** Local SGD/DiLoCo is combined with a compressor. **Inputs:** actual per-worker `Δ_i` or other communication target; precise local/outer states; matched token budget. **Procedure:** characterize the **actual deltas** with spectrum/sparsity/quantization measurements, then compare encoders on identical input. If necessary, use the factorial controls below. **Boundary:** cannot assert both columns of a factorial experiment communicate the same tensor.

## 4. Symptom → diagnosis cards

### HY-ADDITIVE — AB improves over A and B but interaction is unclear

- **Explanations:** independent benefits; new input distribution; changed compute budget or configuration.
- **Discriminator:** a carefully defined 2×2 comparison: dense/each-step A, dense/local B, compressed/each-step C, compressed/local D. State the estimand (loss at fixed tokens or optimizer-update discrepancy) and ensure comparator validity. Note that temporal changes may alter the transmitted object.
- **Inference:** absent interaction evidence, a combination may still be useful but does not establish a new coupling mechanism.

### HY-DELTA — Accumulated local delta is more/less compressible

- **Explanations:** structural averaging; norm/scale effects; changed momentum/learning-rate spectrum; worker drift.
- **Discriminator:** sample `Δ_i` and per-step `g_i` in the same real windows; use matched-rank/budget references for **each actual object**, and encode real `Δ_i` directly.
- **Inference:** object-specific structure only; not automatic improved training quality.

### HY-STATE — EF and outer updates interact unexpectedly

- **Explanations:** inconsistent units; writing residual at inner rather than outer steps; refresh clears buffers; worker weighting mismatch.
- **Discriminator:** event-by-event reconstruction of input → encoding → aggregation → outer update → residual writeback, with a short matched replay.
- **Inference:** a discrepancy in actual defined state transitions, not proof that every scheme should use textbook EF.

### HY-QUALITY — Components work individually, combined quality degrades

- **Explanations:** optimizer/state nonlinearity; correlated error; changed token/batch/scheduler; seed variability.
- **Discriminator:** audit conditions across four controls; inspect same-state true optimizer updates and authorized short closed-loop evidence.
- **Inference:** distinguish supported conflicts from unverified long-run effects.

## 5. Mathematical/experimental toolbox

| Tool | Conditions | Answers | Pitfall |
|---|---|---|---|
| Factorial A/B/C/D | matched resource budget and well-defined target | main effects, interaction and confounds | interaction depends on metric scale and actual semantics |
| Same-object compression oracle | identical communication input | representation complementarity | gradient oracle is not a model-delta oracle |
| State/EF recurrence | exact inner/outer order | whether omitted information is duplicated or dropped | textbook EF not universal |
| Shadow optimizer update and probe | full restorable state | how encoding/sync affects actual updates | one-step result != final quality |

## 6. Paper cards / closest prior art

1. **[LoCoDL: Communication-Efficient Distributed Learning with Local Training and Compression](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1ec69275e9f002ee068f5d68380f3290-Abstract-Conference.html)**, Condat et al., ICLR 2025. **Motivation:** combine local training with compression. **Method:** local updates plus a broad class of unbiased compressors with guarantees under particular strongly convex/heterogeneous assumptions. **Reuse:** prior art and joint rate analysis. **Limit:** not a general AdamW+LLM proof.
2. **[Maximizing Communication Efficiency for Large-scale Training via 0/1 Adam](https://arxiv.org/abs/2202.06009)**, Lu et al., ICLR 2023. **Motivation:** combine 1-bit communication and local steps for adaptive optimization. **Method:** approximate/stale optimizer states enabling communication reductions. **Reuse:** optimizer state interaction. **Limit:** nonlinear state and approximation assumptions are not a trivial Adam plug-in.
3. **[DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105)**, Douillard et al., 2023. **Motivation:** train LLMs with weak interconnects. **Method:** many inner steps followed by model-delta outer updates. **Reuse:** meaning of the transmitted object. **Limit:** depends on H, outer optimizer and tested data/network.
4. **[1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed](https://proceedings.mlr.press/v139/tang21a.html)**, Tang et al., ICML 2021. **Motivation:** communication savings with Adam-style convergence speed. **Method:** preconditioner warmup and compression of optimizer-specific state. **Reuse:** optimizer-state-aware evidence. **Limit:** specific warmup/freeze rules; not generic SGD compression.
