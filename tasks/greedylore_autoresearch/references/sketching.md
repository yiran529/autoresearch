# Sketching & Global Statistics for Distributed Communication

**Use for:** communication-efficient global statistics, shared random projections, candidate support/direction discovery, approximate aggregation decisions, and matrix/vector sketches. Read `common.md`; consult `low_rank.md` or `sparsification.md` for the target representation. **This is an optional mechanism and diagnosis reference, not a requirement to use a sketch or to optimize estimated scores.**

## 1. Mechanism landscape

| Mechanism | Information moved | Typical question / caveat |
| --- | --- | --- |
| Linear shared sketch | $S h_i$ for a common linear map $S$ | What global quantity can be estimated after legal aggregation? |
| CountSketch / heavy-hitter recovery | Compact linear summary of coordinates | Does a small candidate set cover important **global** coordinates? |
| Random directional statistics | Projections used to score candidate directions | How much sampling noise and nonlinear selection error arise? |
| Two-stage support/direction selection | Cheap candidate scores followed by exact/reduced values for candidates | Does improved selection offset the added dependent collective? |
| Reused/adaptive sketch state | Common seed, estimator, mask, or prior sketch | Does current data remain conditionally independent of sketch randomness? |
| Alternative atomic/coordinate representations | Sparse coefficients in a chosen basis/atom set | Is the chosen representation itself the hidden bottleneck? |

The goal need not be high-fidelity reconstruction of the entire original tensor. Sometimes the desired output is a **decision** (which support, direction, or update to use), a bound, or an approximate statistic. State that target separately from the eventual transmitted representation.

## 2. Common aggregation semantics

Let the actual worker inputs be $h_i$ and weights $w_i\ge 0$, $\sum_iw_i=1$. For a **common linear map** $S$ and compatible collective,

$$
y=\sum_i w_i(S h_i)=S\left(\sum_iw_i h_i\right)=S\bar h.
$$

This identity is what makes many shared sketches aggregatable. It does **not** imply that the nonlinear decoder $R(y)$ is unbiased, that independent local choices can be averaged, or that transmitted bytes/rounds improve runtime. For a nonlinear statistic $f$, generally

$$
\sum_iw_i f(S h_i)\ne f\left(\sum_iw_i S h_i\right).
$$

**Random-direction energy example:** fix a matrix $H=\bar H$, a candidate direction $u$ and a fresh $v\sim\mathcal N(0,I)$ independent of the conditioned input and basis. Set $Z=u^\top H v$. Then

$$
\mathbb E[Z^2\mid H,u]=\|u^\top H\|_2^2,
\qquad
\operatorname{Var}(Z^2\mid H,u)=2\|u^\top H\|_2^4.
$$

For $K$ conditionally independent draws with the same fixed $H,u$, averaging $Z_k^2$ divides the conditional variance by $K$. If $Z$ is built by first reducing $u^\top H_i v$ across workers and **then squaring**, it targets global energy; locally squaring and then reducing targets a different expression. Correlations, reused probes, and adaptive choices alter the conditioning. Even exactly unbiased energy estimates can produce a suboptimal selected support or direction set.

## 3. Diagnostic targets and initial recipes

| Object | Direct observation | Supports | Does not establish |
| --- | --- | --- | --- |
| Sketch $y=S\bar h$ | Fixed-input, same-seed aggregate and reconstruction | Legal merge and conditional estimation behavior | A training or wire-time gain |
| Candidate coverage | Global oracle support/direction energy contained in candidate set | Utility of candidate generation | Exact recovery from compressed values |
| Selection regret | True target utility of estimated choice vs. true best choice within the stated family | Whether estimator noise harms the decision | Final optimizer quality |
| Sketch state | Seed/basis/hash lifetime, synchronization and randomness | Correct conditional interpretation | Reusing randomness is harmless |
| Two-stage protocol | Bytes, indices, stages, memory, and critical path | True coordination cost | Lower end-to-end time from shorter messages |

### SK-01 — Mergeability and target-statistic check (D3/D5)

**Inputs:** matched per-worker $h_i$, weights, common sketch construction and actual reduced result. **Action:** test $\sum_iw_i Sh_i$ against $S\sum_iw_ih_i$ within declared precision; distinguish reduction before nonlinear decoding from nonlinear local decoding before reduction. **Interpretation:** identifies an incorrect aggregation target or implementation, not necessarily a research opportunity.

### SK-02 — Estimation versus selection (D1/D3)

**Inputs:** a fixed real communication input, candidate directions/supports, true global scores and the actual selection/sketch randomness. **Action:** repeat fresh sketches on *the same input*, compare score bias/variance and the true utility of the selected set. Where meaningful, compute selection regret

$$
\mathcal R=\sum_{j\in S^*}s_j-\mathbb E\left[\sum_{j\in\hat S}s_j\right]\ge 0,
$$

where $S^*$ is the exact size-$r$ maximizer of **fixed nonnegative utilities** $s_j$ in the same candidate family, and $\hat S$ is the random size-$r$ selection. **Interpretation:** low overlap between selected direction sets can coexist with small regret when candidate energies are tied or nearly equal. The regret is defined for this selection objective, not training loss.

### SK-03 — Candidate coverage and hidden communication (D1/D3)

**Inputs:** true global aggregate on a diagnostic snapshot, proposed candidates, support/score payload, second-stage value exchange and actual collective trace. **Action:** assess oracle inclusion or retained energy together with candidate set size, required broadcasts/AllReduce/AllGather, and metadata. **Interpretation:** a small candidate set can miss global coordinates; an accurate candidate set can still cost more than dense communication.

### SK-04 — Reused sketches and phase dependence (D2/D5)

**Inputs:** randomness lineage (shared/local, fresh/reused), previous selections, state refresh events and actual messages. **Action:** distinguish conditional estimation with fresh independent randomness from adaptively chosen sketches conditioned on previous data. **Interpretation:** a proof under fresh independent probes cannot automatically be applied to history-adaptive measurements.

## 4. Symptom → diagnosis cards

### SK-LOCAL — Local scores look reliable but the global choice is poor

- **Explanations:** cancellation across workers; locally nonlinear score aggregation; mismatched weights or probe seeds.
- **Discriminator:** compute the exact same-event global score, the reduced projected statistic and the locally aggregated nonlinear statistic.
- **Boundary:** a global sketch is one option, not a conclusion mandated by this diagnosis.

### SK-RANK — Unbiased estimates frequently choose weak directions

- **Explanations:** high variance, close score margins, adaptive selection effects, or implementation differences.
- **Discriminator:** fixed-input repeated randomness; plot true utility of selected sets, not only mean estimated score or index overlap.
- **Boundary:** a higher-accuracy sketch may also require more communication and still fail to change optimizer updates.

### SK-COST — Sketch bytes shrink but runtime does not

- **Explanations:** dependent rounds, heavy-hitter decoding, reconstruction, metadata and loss of overlap.
- **Discriminator:** measure true payloads and critical-path events using `engineer_runtime.md`; a single theoretical sketch dimension is not a network model.
- **Boundary:** local sketch compute and backend transport may dominate the resulting speed.

## 5. Mathematical / construction toolbox (conditional)

| Tool | Assumptions / inputs | Useful question | Common misuse |
| --- | --- | --- | --- |
| Linear sketch merge | Identical $S$, matching weights/dtypes and legal collective | What exact global linear statistic is available? | Equating nonlinear local scores with a global score |
| Gaussian projection moments | Proper Gaussian law and independence conditioned on the input | Direction-energy estimation variance | Unbiased score $\Rightarrow$ optimal top-$r$ selection |
| CountSketch / heavy-hitter recovery | Appropriate hash/sign layout and recoverability assumptions | Approximate global coordinate selection | Assuming candidate discovery is free or always complete |
| Sampling and decision regret | Defined fixed utilities and common feasible set | Selection loss beyond estimator bias | Treating support overlap as an optimization objective |
| Communication / information constraints | Full packet, seed, index, refresh and stage costs | Whether the protocol is legal and potentially economical | Counting encoded coefficients but ignoring collective dependencies |

### Transferable insights (not an exhaustive catalogue)

A sketch can change **what workers agree on**, not just how their tensors are stored. Linear summaries can be reduced before a nonlinear shared choice; a second message can recover values for a restricted set; a shared random view can replace a dense statistic with a conditional estimate. But shared randomness itself is not free of design constraints, and the most accurate statistic need not be the best training objective.

Consider other representation spaces, deterministic sufficient statistics, or protocols with no explicit sketch whenever those better match the target. A mathematically elegant oracle can remain useful as a reference even when it is too expensive or noncausal to deploy online.

## 6. Paper cards / closest prior art

1. **[Communication-efficient Distributed SGD with Sketching (Sketched-SGD)](https://papers.nips.cc/paper/2019/hash/75da5036f659fe64b53f3d9b39412967-Abstract.html)**, Ivkin et al., NeurIPS 2019. **Mechanism:** distributed linear sketches supporting a global sparse/heavy-hitter-oriented update. **Reuse:** sketch merge before support recovery. **Limit:** recovery cost and original convergence/SGD assumptions do not imply generic AdamW+LLM improvements.
2. **[Greedy Low-Rank Gradient Compression for Distributed Learning with Convergence Guarantees (GreedyLoRE)](https://arxiv.org/abs/2507.08784)**, Chen et al., 2025. **Mechanism:** randomized global direction scoring with shared state. **Reuse:** estimator target and order of aggregation/nonlinearity. **Limit:** conditional score unbiasedness does not prove exact selected top-$r$ directions.
3. **[PowerSGD](https://arxiv.org/abs/1905.13727)**, Vogels et al., NeurIPS 2019. **Mechanism:** a shared-factor protocol enabling aggregation of intermediate low-rank statistics. **Reuse:** lawful staging of linear collectives around nonlinear basis construction. **Limit:** low-rank factor communication is not automatically a one-message sketch.
4. **[ATOMO: Communication-efficient Learning via Atomic Sparsification](https://papers.nips.cc/paper_files/paper/2018/hash/33b3214d792caf311e1f00fd22b392c5-Abstract.html)**, Wang et al., NeurIPS 2018. **Mechanism:** unbiased sparsification over a chosen atomic decomposition. **Reuse:** rethink the coordinate system of a sparse representation. **Limit:** atom recovery/encoding costs and assumptions vary; it is not a drop-in global-sketch algorithm.
5. **[ScaleCom: Scalable Sparsified Gradient Compression](https://proceedings.nips.cc/paper_files/paper/2020/hash/9d58963592071dbf38a0fa114269959c-Abstract.html)**, Chen et al., NeurIPS 2020. **Mechanism:** worker-aware sparsification with a scalable communication structure. **Reuse:** coordinate selection and aggregation may be co-designed. **Limit:** practical/scaling claims depend on structure, backend and tested workloads.
