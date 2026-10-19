# Quantization Communication: Diagnosis and Mechanism Analysis

**Use for:** randomized or deterministic low-bit quantization, block scales, sign compression, 1-bit Adam, 0/1 Adam. Read `common.md` first. This is not a default experiment checklist. In particular, **unbiased QSGD** and **1-bit Adam's optimizer-state mechanism** require different diagnoses.

## 1. Landscape

| Family | Motivation / method | Principal risks to diagnose |
|---|---|---|
| QSGD / stochastic rounding | Explicit bits–variance trade-off | Variance, group size, randomization, and representation constraints |
| Ternary / sign coding | Very few discrete levels with scale | Sign distortion, dynamic range, clipping/saturation |
| Block quantizers | Per-block scales limit outlier dominance | Metadata/scale definitions, inconsistent worker scales, outliers |
| 1-bit Adam | Leverage approximate stability of preconditioning after warmup | Variance drift, frozen state, momentum/EF semantics |
| 0/1 Adam | Combine compression and local steps while treating Adam nonlinearity | Stale states, local training interaction; see `hybrid.md` |

Determine the **actual quantized object**, scale generation/synchronization, randomized rounding distribution, clipping, signed encoding, and metadata actually transmitted.

## 2. Initial recipes

### QT-01 — Matched-input quantization error (D1)
**Inputs:** Real message input `H`, quantizer output `Q(H)`, block/scale/bitwidth/zero-point, clip rule, RNG, wire format and local/global definition. **Run:** Fix H, repeat with independent quantizer RNG; measure `||Q(H)-H||²/||H||²`, empirical `E[Q(H)|H]-H` (bias), `E||Q-EQ||²` (variance), coordinate error quantiles, saturation/outlier rates and declared quantization levels, scales and clip rule.

**Interpretation:** Fixed-input repetitions isolate compression randomness, whereas changing minibatches also changes sampling noise. A near-zero empirical mean is **not** proof of unbiasedness for every H. Handle zero input and extreme ranges separately.

### QT-02 — Local/global and cross-step state profile (D2/D3)
**Inputs:** Same-sync worker inputs, quantizer scales/states and reconstructed optimizer-visible global result, plus relevant consecutive steps. **Run:** Compare `mean_i Q_i(H_i)` and actual aggregation with dense `mean_i H_i`; separate scale differences from worker gradient heterogeneity. For momentum quantization/frozen variance, track true `m_t`, `v_t`, preconditioner drift and warmup phases.

**Interpretation:** Local MSE is not automatically global error; variance stability is an empirical regime, not a universal invariant.

### QT-03 — Optimizer update response (D4)
**Inputs:** Full pre-update optimizer/model/EF state on matched batches. **Run:** Measure realized dense vs quantized update direction, magnitude and (if authorized) independent one-step probe loss; preserve the exact scale and quantizer-state semantics.

**Interpretation:** Small MSE does not establish final training quality.

## 3. Symptom → diagnosis cards

### QT-BIAS — Nonzero empirical mean error
- **Alternatives:** Truly biased quantizer; insufficient RNG trials/correlated seed; scale/clipping/overflow changes the implemented estimator.
- **Discriminator:** Keep H fixed, derive analytical conditional expectation where possible and compare with repeated seeds; audit clipping, zero input and precision.
- **Limit:** Finite empirical mean does not prove a theorem for all H.

### QT-OUTLIER — A few coordinates dominate quantization error/scale
- **Alternatives:** Per-tensor scale set by outliers; block grouping inappropriate; clipping bias; naturally large stochastic gradients.
- **Discriminator:** On the same H quantify extreme-value distribution and reconstruction under existing per-block/per-tensor reference definitions; separate rounding and clipping loss. References are diagnostic, not parameter optimization.

### QT-AGG — Local fidelity looks good but global update is wrong
- **Alternatives:** Different worker scales; incorrect order/weighting of aggregation; inherently heterogeneous worker gradients; overflow or metadata mismatch.
- **Discriminator:** Reconstruct local inputs/scales and true dense global average at the same event. No automatic independent-error `1/N` claim.

### QT-ADAM — Small compression error but unusual actual update/quality
- **Alternatives:** Momentum/preconditioner nonlinearities; continued variance drift after freeze; changed update length; stochastic training variation.
- **Discriminator:** Match full pre-step Adam states, compare true updates and `m,v` dynamics at warmup/early/late compression stages; optional independent probe.
- **Limit:** Missing optimizer state cannot be replaced by newly initialized Adam.

## 4. Conditional math tools

- **Conditional unbiasedness and variance:** `E_ω[Q(H;ω)|H]=H` only for the correctly specified random rounding law and conditioning; historical randomness does not suffice.
- **Biased compressors and EF:** require exact residual units, aggregation and contraction assumptions. Not every quantizer needs EF; EF does not automatically repair arbitrary Adam nonlinearities.
- **Optimizer:** gradient reconstruction error is not realized update error. Check the temporal regime in which frozen `v_t` is valid.

## 5. Paper cards

1. **[QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](https://proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html)** (Alistarh et al., NeurIPS 2017). **Why:** controllable bandwidth–variance compromise. **How:** randomized quantization and encoding with convergence analysis. **Limits:** variance, quantization assumptions and workload dependence; QT-01 can check whether implementation matches the randomized rule.
2. **[TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning](https://arxiv.org/abs/1705.07878)** (Wen et al., 2017). **Why:** very-low-bit gradient communication. **How:** scaled stochastic ternary representation. **Limits:** dynamic range, layer scales and noise; do not extrapolate to LLM Adam without evidence.
3. **[1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed](https://proceedings.mlr.press/v139/tang21a.html)** (Tang et al., ICML 2021). **Why:** nonlinear Adam cannot simply use classic gradient EF. **How:** warmup a preconditioner, then compress momentum communication. **Limits:** preconditioner drift, state transitions, QT-ADAM.
4. **[Maximizing Communication Efficiency for Large-scale Training via 0/1 Adam](https://arxiv.org/abs/2202.06009)** (Lu et al., 2022). **Why:** combine low-bit and infrequent sync with an adaptive optimizer. **How:** approximate stale-state/linearized Adam-like updates. **Limits:** stale state and nonlinear effects; significant prior art for quantization × temporal hybrids.
5. **[Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://proceedings.mlr.press/v97/karimireddy19a.html)** (Karimireddy et al., ICML 2019). **Why:** repair biased sign/compression through residual compensation. **How:** EF and virtual iterates. **Limits:** only under matching compressor/optimizer assumptions.
