# Stateful Optimizer & Compression: Mechanism and Diagnosis Reference

**Use for:** communication algorithms whose behavior depends on residual memory, gradient/parameter tracking, momentum, Adam-style preconditioning, optimizer state staleness, or cross-step protocol phases. Read `common.md` first; consult `quantization.md`, `temporal.md`, or `hybrid.md` for the actual communicated object and relevant interaction. **This is a selective mechanism reference, not a mandated optimizer modification or research direction.**

## 1. Algorithm landscape

| State mechanism | What it retains | Main question / non-equivalence to watch |
| --- | --- | --- |
| Classical error feedback (EF) | Locally uncommunicated compressor-input information | Are residual units, decoded local messages, and writeback events consistent? |
| Gradient tracking / EF21 | A recursively corrected gradient estimate | Does the estimator track its stated target, rather than simply storing a missing message? |
| MARINA-style difference estimation | A shared/worker estimator refreshed occasionally and changed with compressed gradient differences | Are full-refresh and difference stages using compatible model versions and legal shared decisions? |
| Momentum compression | Historical first-moment information | Is the compressed object the momentum (or a derived quantity) rather than the current gradient? |
| Preconditioner handling in 1-bit Adam | Warmed-up then frozen variance/preconditioner under the paper's algorithm | Does the assumed stability hold in the evaluated training regime? |
| Approximate/stale states and local steps in 0/1 Adam | Adaptive state estimates across fewer synchronization events | Are state approximation, model deltas, and synchronization windows explicitly defined? |

A state can be **local to worker** $i$, **shared and identical**, or **optimizer-owned but not synchronized**. The same variable name does not imply the same value, timestamp, or update rule on different ranks. A local gradient, EF-augmented input, gradient estimator, momentum, preconditioned update and accumulated model delta are different objects.

## 2. State semantics and diagnostic targets

| Object | Question / diagnostic observable | Supports | Does not establish |
| --- | --- | --- | --- |
| Residual $e_{i,t}$ | Units, local reconstruction, conservation at every real event | Correct/incorrect local EF accounting | Better final training merely from smaller residual norm |
| Estimate $z_{i,t}$ | Error against a matched target, drift and correction terms | Tracking behavior on tested states | EF and tracking algorithms are interchangeable |
| Momentum $m_{i,t}$ | Pre/post-compression moments and update contribution | State-dependent perturbation | Compression noise is harmless after accumulation |
| Preconditioner $D_t$ or $v_t$ | Drift over warmup, compressed phase and regime changes | Plausibility of a fixed/stale approximation | Stability for all layers, tasks or future phases |
| Synchronization phase | Exact order of local/collective/optimizer state transitions | Protocol legality and reproducibility | Lower network bytes mean lower wall-clock time |
| Optimizer-visible update $u_t$ | Same-state dense/compressed update difference | Local effect of state/codec interaction | End-to-end convergence or generalization |

**Classical local EF identity (conditional):** if, on a consecutive sequence of communication events,

$$
h_{i,t}=g_{i,t}+e_{i,t},\qquad
q_{i,t}=C_i(h_{i,t}),\qquad
e_{i,t+1}=h_{i,t}-q_{i,t},
$$

then $q_{i,t}-g_{i,t}=e_{i,t}-e_{i,t+1}$ and, with no hidden resets,

$$
\sum_{t=0}^{T-1}q_{i,t}=\sum_{t=0}^{T-1}g_{i,t}+e_{i,0}-e_{i,T}.
$$

Here $q_{i,t}$ means the **actual local decoded message** used for residual writeback, not the optimizer-visible global aggregate. This algebra does not prove small residuals, stable Adam updates, or legality of any collective. Refresh, skipped layers, changing units, stochastic decoding and outer steps require their own actual branch semantics.

**Tracking is a different program.** One possible *illustrative* gradient-estimate recurrence is

$$
z_{i,t+1}=z_{i,t}+C_i(g_{i,t+1}-z_{i,t}).
$$

Its tracked quantity, conditioning, and compressor property must be declared. It is not a universal specification for EF21, MARINA, or any optimizer, whose algorithms and assumptions must be read directly (see Paper cards 2–4).

## 3. Initial diagnostic recipes (select only where relevant)

### SO-01 — Residual conservation and state units (D2/D5)

**Inputs:** actual hook-stage input, local decoded message, residual before/after, refresh/skip flag, worker ID and optimizer event. **Action:** reconstruct every writeback expression and test the recurrence for several events, including boundaries. **Interpretation:** an algebra mismatch can identify missing or duplicated information; the identity alone says nothing about training quality.

### SO-02 — Tracking estimate versus target (D2/D3)

**Inputs:** stored estimator(s), matched real gradients where legitimately measurable, worker weights, common decision state, refresh events. **Action:** separate estimator error, target drift, compression error and mismatched model versions; audit which variables are known at each stage. **Interpretation:** a lower fixed-input estimator error is not a closed-loop advantage, and a locally accurate estimator need not yield an accurate weighted global update.

### SO-03 — Preconditioner and momentum phases (D2/D4)

**Inputs:** complete Adam-like optimizer snapshots before/after warmup, during compression and at any proposed state transition. **Action:** inspect $m_t$, $v_t$, bias corrections, weight decay, preconditioning and true parameter increments under the specified algorithm; compare frozen versus live-state update *on matched state* if that is the question. **Interpretation:** observed preconditioner stability is workload/phase dependent, not a license to freeze it for arbitrary optimizers.

**Muon example (130M development):**

- Identify whether compression acts on the gradient, momentum/Nesterov update, orthogonalization input or orthogonalized output; trace the actual order of aggregation and state updates.
- From identical model, momentum, compressor/EF and batch/RNG state, compare the realized parameter updates, including scaling and weight decay, rather than reconstruction error alone. Generally $\operatorname{Ortho}(\sum_i w_i M_i)\ne\sum_i w_i\operatorname{Ortho}(M_i)$ for worker inputs $M_i$; distributing orthogonalization work over shared inputs and gathering results is distinct from averaging independently orthogonalized local inputs.
- Moving compression across orthogonalization can change the algorithm itself. Declare that change and test its mechanism and training effects on 130M; neither lower gradient MSE nor Adam preconditioner stability establishes a Muon benefit.

### SO-04 — Local/outer-state interaction (D2–D4)

**Inputs:** exact local-update window, model versions, transmitted delta or momentum buffer, optimizer state, shared/worker weights. **Action:** reconstruct the communication event, aggregation, outer update, and all local-state resets or corrections. **Interpretation:** a periodic model delta is not the current gradient; replaying fixed deltas cannot establish a new closed-loop trajectory.

## 4. Symptom → diagnosis cards

### SO-RESIDUAL — Residual grows or suddenly vanishes

- **Explanations:** genuinely difficult compression inputs, scale drift, refresh/skip handling, inconsistent units, or a different intentionally defined memory scheme.
- **Discriminator:** verify actual per-event state transitions; separate raw $g$, encoded $h$, local $q$, weighted aggregate and optimizer update on the same timeline.
- **Boundary:** a large local EF norm is not necessarily global lost information or a proof of poor training.

### SO-TRACK — Tracking error and compressed-message MSE disagree

- **Explanations:** changed estimator target, target drift, biased/stale state, or different conditioning.
- **Discriminator:** evaluate both quantities at the same parameter/worker state and reconstruct the exact recurrence, including full-refresh branches.
- **Boundary:** a different estimator is not intrinsically a bug; the algorithm defines what should be maintained.

### SO-PRECOND — Low-bit reconstruction looks good but the adaptive update differs

- **Explanations:** nonlinear preconditioning, drift during a frozen-state phase, momentum mismatch, loss scaling/clipping, or incorrect bias-correction placement.
- **Discriminator:** full-state shadow optimizer, true before/after update and phase-specific state diagnostics; see `quantization.md` QT-ADAM.
- **Boundary:** a local update difference is not a long-horizon convergence theorem.

### SO-STATE-COMMS — Fewer synchronizations change the meaning of state

- **Explanations:** asynchronous/stale model versions, local optimizer drift, misweighted model delta, incorrect outer correction or resume state.
- **Discriminator:** compare event order and exact transmitted object at worker/local/outer scopes under a matched window.
- **Boundary:** fewer rounds may increase per-event payload or staleness and need not save wall-clock time.

## 5. Mathematical / construction toolbox (conditional)

| Tool or contrast | Preconditions | What it exposes | Frequent misuse |
| --- | --- | --- | --- |
| Telescoping EF identity | Exact local $h=g+e,\ e'=h-q$ and no unmodeled resets | Delayed correction accounting | Applying EF algebra directly to Adam's nonlinear parameter update |
| Contractivity or conditional unbiasedness | Exact compressor law conditioned on history/state | Possible state-error recurrences | Assuming every history-dependent codec satisfies a static bound |
| Tracking recursion / Lyapunov drift | Correct target, sampling, Markov state and refresh event | Whether state error contracts or accumulates | Importing EF21/MARINA rates to a different optimizer |
| Frozen preconditioner thought experiment | Specified stable $D_\star$ and momentum semantics | Linear map from moment input to schematic update | Calling the result exactly vanilla Adam after state freezing |
| One-step shadow optimizer | Full identical pre-update model and optimizer state | Perturbation of the actual realized parameter update | Predicting final training quality from one step |

### Transferable mechanism insights (not a candidate menu)

**Classical EF:** change the temporal accounting of omitted information; the residual is a specific local quantity. **EF21 / 3PC:** change the state that is tracked and its correction operator. **MARINA:** communicate the difference of a maintained estimate, occasionally restoring it with a more informative event. **1-bit Adam:** redesign the optimization/communication interface by handling the nonlinear preconditioner before compressing the momentum. **0/1 Adam:** jointly reason about optimizer approximation and communication frequency rather than composing two independent plug-ins.

These are different explanations of *how a mechanism can become possible*. None dictates that a new proposal must use EF, freezing, refresh, or any listed pattern. A claimed new state program needs its own equations, information dependencies, cost and phase handling.

## 6. Paper cards / closest prior art

1. **[Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://proceedings.mlr.press/v97/karimireddy19a.html)**, Karimireddy et al., ICML 2019. **Mechanism:** local residual correction and virtual iterates. **Reuse:** exact omitted-information accounting. **Limit:** the theorem's compressor/SGD assumptions do not automatically cover AdamW, Muon or altered refresh rules.
2. **[EF21](https://papers.nips.cc/paper_files/paper/2021/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html)**, Richtárik et al., NeurIPS 2021. **Mechanism:** a modern Markovian gradient estimate induced by compression. **Reuse:** tracked-state construction. **Limit:** not textbook EF residual writeback.
3. **[MARINA](https://proceedings.mlr.press/v139/gorbunov21a.html)**, Gorbunov et al., ICML 2021. **Mechanism:** occasional fuller communication plus compressed gradient differences with a maintained estimator. **Reuse:** estimator/refresh trade-off. **Limit:** exact gradient versions, randomness and compressor assumptions matter.
4. **[3PC: Three Point Compressors](https://proceedings.mlr.press/v162/richtarik22a.html)**, Richtárik et al., ICML 2022. **Mechanism:** state-dependent compressor family that includes EF21-type constructions. **Reuse:** broadening the state-dependent operator design space. **Limit:** a named 3PC assumption must be checked for the proposed method.
5. **[1-bit Adam](https://proceedings.mlr.press/v139/tang21a.html)**, Tang et al., ICML 2021. **Mechanism:** warmup/fixed preconditioner with momentum-oriented error-compensated 1-bit communication. **Reuse:** isolate optimizer nonlinearity before redesigning communication. **Limit:** phase and variance-stability assumptions.
6. **[0/1 Adam](https://arxiv.org/abs/2202.06009)**, Lu et al., ICLR 2023. **Mechanism:** optimizer-state approximations with low-bit communication and local steps. **Reuse:** temporal and nonlinear-state coupling. **Limit:** not equivalent to compressing standard Adam independently at every local step.
7. **[EF21 with Bells & Whistles](https://www.jmlr.org/papers/v26/24-0059.html)**, Fatkhullin et al., JMLR 2025. **Mechanism:** several EF21 extensions, including momentum and bidirectional compression, under stated analysis conditions. **Reuse:** examples of stateful mechanism composition. **Limit:** theory/algorithm variants are not universal LLM training guarantees.
