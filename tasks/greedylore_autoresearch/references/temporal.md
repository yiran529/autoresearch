# Temporal / Low-Frequency Communication: Diagnosis and Mechanism Investigation

**Scope:** Local SGD, periodic parameter averaging, SlowMo, DiLoCo, and schemes that change the inter-worker communication interval. Unlike a per-step gradient compressor, the communication object may be a **model delta after H inner steps**, `Δ_i`, not `g_t`. A research-tree Node/Island is not a worker/group within a training algorithm. Read `common.md` first. Across-step or synchronization-phase diagnostics describe **algorithmic state**; system performance diagnosis belongs to Engineer.

## 1. Algorithm landscape

| Mechanism | How it works | Main diagnostic objects |
|---|---|---|
| Local SGD/periodic averaging | Each worker runs `H` inner updates before synchronization | model disagreement, before/after average, loss |
| SlowMo | Slow momentum atop base optimizers with periodic synchronization | slow-momentum state and boundary update |
| DiLoCo | Inner AdamW-like steps; exchange accumulated deltas; outer momentum update | local `Δ_i`, averaged delta, outer optimizer state |
| Async/stale variants | worker updates applied at different model versions | delay, staleness, worker weighting; first verify true asynchrony |

Define **local step**, **outer step**, per-worker and total training tokens, gradient accumulation and scheduler progress precisely. Comparing 500 outer steps with 500 synchronous SGD steps is invalid.

## 2. Diagnostic targets

| Object | Metric | Supports | Does not prove |
|---|---|---|---|
| Worker divergence | `D_t=Σ_i w_i||x_i−x̄||²`, `x̄=Σ_iw_ix_i` | model disagreement in a sync window | divergence causes inferior final quality |
| Inner/outer updates | norm/cosine of averaged local delta, outer update and defined reference | where update direction/scale changes | two different trajectories should match per step |
| Window phase | states and loss at start/mid/before/after outer step | whether effects concentrate at boundaries | adjacent different inputs are causal controls |
| Data heterogeneity | worker-shard statistics and matched-parameter gradients | whether data distribution contributes to disagreement | one sample explains all drift |
| Checkpoint fidelity | inner/outer optimizer, scheduler, RNG, cursor, phase | whether replay/resume is meaningful | weights-only checkpoint is sufficient |

## 3. Initial recipes

### TM-01 — Window and worker-disagreement profile (D2/D3, default)

**When:** first Local SGD/DiLoCo candidate or change to `H`/inner optimizer. **Inputs:** actual worker weights, event markers, `H`, weights `w_i`, step/token counters, checkpoints at relevant phases; representative layer sampling is allowed if scope is declared.

**Procedure:** measure `D_t` at start/mid/end of a window and after the outer update; pair with worker/global probe loss if available, and record the exact communicated object. **Output:** recurrent growth or discontinuities, scope and affected layers. **Boundary:** disagreement alone does not establish training degradation.

### TM-02 — Inner/outer optimizer effect (D4)

**When:** boundary anomaly or lower quality at matched tokens. **Inputs:** complete pre-update model and inner/outer optimizer, RNG, scheduler, worker deltas and weights, defined synchronous reference. **Procedure:** separate `Δ̄=Σ_i w_iΔ_i` from the actual outer parameter update; measure direction/norm, layers and post-step development-probe loss. Document distinct trajectories when comparing with dense synchronous training. **Output:** which layer of the update mechanism introduces the observed difference. **Boundary:** one-step observation does not prove long-term cause.

## 4. Symptom → diagnosis cards

### TM-DRIFT — Worker disagreement increases with H

- **Explanations:** data-shard heterogeneity; independent optimizer trajectories; unequal states/steps; measurement scope.
- **Discriminator:** matched initial weights and optimizer; compare carefully matched shared-data versus real-shard windows, and distinguish gradients at local divergent weights from gradients at the same reference weights.
- **Inference:** magnitude and potential source of drift within the tested window. **Not:** proven quality harm or optimal H.

### TM-LOSS — Quality falls at the same number of training tokens

- **Explanations:** local-drift-induced update mismatch; outer update amplification; batch/scheduler definitions changed; stochastic variability.
- **Discriminator:** audit token/batch/inner/outer/scheduler semantics first; inspect averaged local delta and actual outer update in the same legal state; consult independent seed or short true closed-loop evidence.
- **Inference:** which component is supported as contributing to degradation. **Not:** a convergence theorem from one outer step.

### TM-OUTER — Abrupt update at synchronization boundary

- **Explanations:** slow/outer momentum; inner state mismatch; incorrectly weighted delta aggregation; lost restore state.
- **Discriminator:** hold actual worker deltas fixed, recompute weighted aggregate and outer transition; compare direction/norm and deterministic resume behavior.
- **Inference:** whether the discrepancy appears before or within the outer update. **Not:** a recommended outer hyperparameter.

### TM-WORKER — Training quality deteriorates when adding workers

- **Explanations:** changed global batch/data assignment, wrong aggregate weights, staleness, increased disagreement.
- **Discriminator:** matched total tokens and batch semantics; track weights, phases, data differences and model versions.
- **Boundary:** one worker-count experiment is not universal scaling evidence.

### TM-RESUME — Resumed trajectory differs from continuous training

- **Explanations:** omitted inner/outer optimizer, scheduler, RNG, data cursor or phase state.
- **Discriminator:** deterministic save/resume around synchronization boundaries on matched inputs; record state differences.
- **Boundary:** this is an implementation-state diagnosis, not an automatic theoretical counterexample.

## 5. Mathematical toolbox (only as needed)

| Tool | Preconditions | Helps answer | Common misuse |
|---|---|---|---|
| Local-SGD dispersion/consensus recursion | well-defined inner optimizer, aggregation and sampling | H/noise/heterogeneity vs average-path drift | exporting convex bounds to arbitrary AdamW LLM training |
| Smoothness-based drift bounds | appropriate Lipschitz gradients | parameter divergence vs gradient changes | treating measured drift as a loss lower bound |
| Inner/outer momentum recursion | exact update equations | how boundary state affects the update | equating accumulated delta with current stochastic gradient |
| Bounded staleness | actual version/delay semantics | additional stale-update mismatch | using shared-memory Hogwild claims for cross-node DiLoCo |

## 6. Paper cards

1. **[Local SGD Converges Fast and Communicates Little](https://arxiv.org/abs/1805.09767)**, Stich, ICLR 2019. **Motivation:** fewer synchronization rounds with suitable statistical efficiency. **Method:** H local SGD steps followed by averaging. **Reuse:** worker disagreement and round accounting. **Limitation:** main classical theory assumes convex objectives, not arbitrary AdamW+LLM.
2. **[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html)**, McMahan et al., AISTATS 2017. **Motivation:** communication-limited, possibly non-IID clients. **Method:** FedAvg/periodic model averaging. **Reuse:** data heterogeneity and worker weighting controls. **Limitation:** federated client assumptions differ from homogeneous datacenter workers.
3. **[SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum](https://arxiv.org/abs/1910.00643)**, Wang et al., ICLR 2020. **Motivation:** improve convergence/quality with infrequent communication. **Method:** slow momentum over periodic base-optimizer rounds. **Reuse:** outer-state diagnosis. **Limitation:** optimizer/state/benchmark assumptions must match.
4. **[DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105)**, Douillard et al., 2023. **Motivation:** LLM training across weakly interconnected groups. **Method:** many inner AdamW steps and outer momentum over model deltas. **Reuse:** window/worker/outer-update diagnostics. **Limitation:** optimizer-state preservation, outer-update and data assumptions.
5. **[0/1 Adam](https://arxiv.org/abs/2202.06009)**, Lu et al., ICLR 2023. **Motivation:** combine low-bit exchange and local optimizer steps for Adam. **Method:** approximate/stale optimizer state to support efficient communication. **Reuse:** state-drift and update diagnostics. **Limitation:** not a generic plug-in replacement for AdamW.
6. **[Streaming DiLoCo](https://arxiv.org/abs/2501.18512)**, Douillard et al., 2025. **Motivation:** reduce blocking/peak inter-group communication. **Method:** staggered model-fragment synchronization, communication/compute overlap and quantization. **Reuse:** communicating *which fragment when* as an algorithmic design dimension. **Limitation:** delayed partial updates change protocol semantics; speed/quality claims depend on the actual tested setup.

## 7. Additional mechanism contrasts (optional gap-discovery lenses)

| Contrast | Possible question | Boundary |
| --- | --- | --- |
| Fixed sync schedule vs. synchronized conditional decision | Can all ranks observe the same trigger at a guaranteed common event? | Independent local trigger branches can disagree on collective order and deadlock |
| Model dispersion vs. optimization loss | Is disagreement correlated with an actual outer-update or quality problem? | Dispersion is a parameter-space statistic, not an automatic loss certificate |
| Whole-model sync vs. fragment/streaming sync | Are peak bandwidth and critical path more important than aggregate bytes? | Partial updates introduce model-version and outer-optimizer dependencies; see Streaming DiLoCo |
| Fewer rounds vs. a changed communicated object | When the window changes, are the real model deltas still comparable to per-step gradient inputs? | A different object/trajectory needs its own oracle and matched-token evidence |

**Transferable insight:** communication frequency, event ordering, fragmentation, and state staleness can be research variables independent of a compression codec. A new trigger or schedule is meaningful only with a legal common protocol; it need not originate from a measured disagreement gap.
