# Sparsification Communication: Diagnosis and Mechanism Analysis

**Use for:** Top-k, Random-k, threshold selection, block sparse communication, DGC, sparse EF/momentum. Read `common.md`. Focus on **coordinate selection, residual history, multi-worker aggregation and effective sparse representation constraints**.

## 1. Landscape

| Mechanism | Why / how | Principal limitation to investigate |
|---|---|---|
| Top-k | Transmit highest-magnitude coordinates to retain L2 energy | Cross-step support variation, local/global top-k mismatch |
| Random-k | Select random coordinates; some scaled variants can be unbiased | High variance and possible long waiting time for important coordinates |
| DGC | Extreme sparsity with momentum correction, clipping/masking/warmup | Mechanism interaction with momentum and masking |
| Sparse EF | Carry omitted information into later sparse messages | Coordinate age, residual reset, optimizer behavior |
| Structured sparse | Send blocks/rows or shared masks | Extra representation restriction vs. unstructured sparsity |

## 2. Initial recipes

### SP-01 — Same-input sparse representation (D1)
**Inputs:** Actual sparse-compression object `H`, transmitted support and values, effective k, index encoding and local/global definition. **Run:** Compare true sparse reconstruction against the best k-sparse L2 oracle (exact top-k after vectorization); report retained energy, density, value quantiles and declared sparse representation constraints. **Interpretation:** Exact top-k is an oracle for **that sparse reconstruction objective**, not final model quality. If the algorithm itself implements exact top-k on the same H, an approximately zero oracle gap is expected.

### SP-02 — Cross-step support behavior and EF (D2)
**Inputs:** Support sets on the same parameter coordinate system over several steps, residuals, thresholds and inputs. **Run:** Measure support overlap, number of skipped steps since last transmission, EF carryover and residual dynamics, aligned across workers. **Interpretation:** Different minibatches naturally produce different supports; a long delay may or may not affect realized updates.

### SP-03 — Aggregation and optimizer effect (D3/D4)
**Inputs:** Per-worker inputs/supports/values and aggregate at a matched sync event. **Run:** Compare worker support unions/intersections and globally reconstructed sparse vs dense mean; measure global update difference; check index/value alignment and use a full-state optimizer shadow if relevant. **Interpretation:** Good local reconstructions do not imply good global approximation.

## 3. Symptom → diagnosis cards

### SP-CAPACITY — Even best k-sparse reconstruction loses much energy
- **Alternatives:** Flat coordinate energy distribution, small relative budget, nonrepresentative large-gradient layers/batches.
- **Discriminator:** Stratified oracle retained-energy distribution at matched k and dimension, then check relevance to realized updates.
- **Limit:** A small coordinate is not necessarily useless noise.

### SP-SELECT — Actual result is noticeably worse than best k-sparse oracle
- **Alternatives:** Actual method is not exact top-k of the current H; reused/grouped masks; approximate threshold/sorting or numerical mistakes.
- **Discriminator:** Same H and k, compare exact top-k, actual mask and threshold/indices, and verify true compressor input.
- **Limit:** More accurate reconstruction does not automatically improve training.

### SP-AGING — Some coordinates remain unsent or EF spikes
- **Alternatives:** Repeated competition with larger coordinates, normal delayed EF, reset, gradient-scale drift.
- **Discriminator:** Track coordinate age, residual and eventual transmission with global-average residual and actual recurrence across stages.
- **Limit:** Delay is not permanent loss; large residual alone is not a quality root cause.

### SP-WORKER — Local quality high but global update poor
- **Alternatives:** Disjoint worker supports, local gradient cancellation, wrong index coalescing or weighted mean.
- **Discriminator:** Construct matched dense and sparse worker inputs/aggregates with exact weights and index alignment; record both local and global error.
- **Limit:** Support overlap is only a proxy; actual global update is the target.

## 4. Conditional mathematical tools

- Exact top-k for a fixed vector minimizes L2 reconstruction error at k nonzeros; the tail squared-energy sum is the oracle residual.
- Random-k can be unbiased after a **specific** sampling-and-scaling rule. Check sampling probabilities and scaling.
- EF theory requires the exact residual recurrence and compressor assumptions; feedback does not guarantee every sparse optimizer converges.
- Local compression errors cannot be assumed independent across workers when estimating global errors.

## 5. Paper cards

1. **[Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)** (Lin et al., ICLR 2018). **Why:** Sparse communication in low-bandwidth distributed training. **How:** aggressive sparsity plus momentum correction, gradient clipping, masking and warmup. **Limits:** interconnected components, masking and optimizer-state interactions, workload dependence; attribute ablation effects carefully.
2. **[Sparsified SGD with Memory](https://proceedings.neurips.cc/paper/2018/hash/b440509a0106086a67bc2ea9df0a1dab-Abstract.html)** (Stich et al., NeurIPS 2018). **Why:** Understand convergence with sparse communication and residual memory. **How:** top-k/random-k sparsified SGD and memory. **Limits:** SGD-specific assumptions, not arbitrary Adam/residual resets or inconsistent aggregation.
3. **[The Convergence of Sparsified Gradient Methods](https://proceedings.neurips.cc/paper/2018/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html)** (Alistarh et al., NeurIPS 2018). **Why:** Mathematical justification of magnitude sparsification and error correction. **How:** Analyze delayed corrections under smoothness assumptions. **Limits:** Assumptions and training dynamics must be checked before applying to LLM optimizers.
4. **[Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://proceedings.mlr.press/v97/karimireddy19a.html)** (Karimireddy et al., ICML 2019). **Why:** Error compensation for biased compressors. **How:** EF analysis. **Limits:** Require correct state and actual optimizer assumptions.
