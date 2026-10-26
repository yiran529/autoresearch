# Algorithm Construction for Communication-Efficient Distributed Training

**Primary reader:** Researcher.

**Purpose:** construct candidate algorithms from an observed limitation **or a desired property under an explicit resource constraint**. A candidate does not need prior evidence of improved training quality to be proposed. It needs a concrete mechanism, an account of the information available at runtime, and a reason the change could matter.

Read **common.md** for communication objects and comparison semantics. Use the relevant family reference for established mechanisms, **initial_diagnostics.md** for D1–D5, **semantic_criteria.md** for correctness, and **engineer_runtime.md** for measured runtime and implementation attribution. These are canonical repository filenames. This reference adds no run quotas, submission procedures, or role permissions; the existing workflow and evaluation contract remain authoritative.

**Use selectively.** Start with Sections 1–3, then consult the most relevant construction pattern or example. Examples illustrate transformations, not an exhaustive menu. It is valid to use a construction not listed here. The examples, diagnostics, and patterns are **resources for reasoning, not a search policy**: neither the problem entry points nor C1–C6 bound the hypothesis space.

## Problem entry points
Start from the property or uncertainty to change. The links provide reusable components, not a required menu or evidence of success in the current run. Shared `notes/wisdom.md` contains the run's evidence-bounded lessons; published Analysis remains the source to check.

| Current problem | References and construction patterns | Check before transferring |
| --- | --- | --- |
| Representation loses useful information at a fixed budget | [Low rank](low_rank.md), [quantization](quantization.md), [hybrids](hybrid.md); C3 | Actual representable family, decoder agreement and total payload |
| Omitted information accumulates or returns too late | [Sparsification](sparsification.md), [temporal state](temporal.md); C4–C5 | State units, decay/coverage, phase changes and optimizer-visible effects |
| A fixed policy wastes budget across layers or training phases | [Temporal mechanisms](temporal.md); stateful protocol in §2, C4 and C6 | Observable online signal, synchronized decision, state transition and decision cost |
| A desirable decision depends on unavailable global information | [Sketching and global statistics](sketching.md), [common aggregation semantics](common.md), [low rank](low_rank.md); C2 | What is local/shared, statistic bias, decision error and added communication stages |
| Local reconstruction improves but training does not | [Stateful optimizer mechanisms](stateful_optimizer.md), [initial diagnostics](initial_diagnostics.md), [semantic criteria](semantic_criteria.md); C4, C6 | Actual optimizer update, state interaction, matched controls and training evidence |
| Two components may address complementary limitations | [Hybrid mechanisms](hybrid.md); §2, C1–C3 | Whether their benefits survive composition; state/merge compatibility and complete cost |
| Fewer bytes do not improve measured runtime | [Engineer runtime](engineer_runtime.md), [hybrids](hybrid.md) | End-to-end measurement, packing/metadata and dependent collectives |

A combination needs an explanation of why the components are complementary; evaluate their interaction rather than assuming the separate benefits add. An adaptive method needs an implementable signal and explicit action/state transitions; a fixed schedule is not the only allowed design. Neither construction requires multiple code Parents, and neither gains credibility merely by appearing in this index.

Distinguish three stages:

- **Idea:** a structural change and its motivation; unresolved details are explicit.
- **Executable candidate:** equations, communication, and state transitions are sufficiently specified to implement.
- **Supported finding:** a property or benefit established by the relevant derivation or experiment.

An untested idea may be proposed as a hypothesis. Do not promote a conjecture, a fixed-input identity, or a teaching example into a measured training result.

## 1. Begin with a property to change

Choose one primary design question. It may come from:

- **An observation:** selection loses energy despite a good within-basis oracle; a state drifts; an implementation has an algorithm-imposed dependency.
- **A desired property:** fewer dependent communication stages, less persistent state, coverage despite unreliable scores, or a larger representation family at a fixed message budget.
- **A resource constraint:** a different network regime, a memory limit, or a fixed communication budget changes which trade-offs are worthwhile.

Prior evidence that the baseline is defective is not required. State why the desired property matters for the current task.

Write a short construction brief:

> **Target property → limiting structure or assumption → changeable component → information needed → resource constraint → expected behavioral difference.**

Separate what is fixed by the evaluation contract from what Researcher may change. For example, total tokens and the model may be fixed while the compressor state transition is editable. Do not silently turn a quality comparison into a different-budget comparison.

When useful, generate a small set of structurally different candidates before selecting one. Changing the encoded object, projection construction, or state recurrence offers different mechanisms; three values of the same refresh period offer one mechanism with three settings. A valuable combination of known mechanisms remains a valid research candidate when its contribution is described accurately.

### Open-ended discovery lenses (optional, not a routing rule)

A potential gap is a mismatch between a desired property and what an algorithm actually guarantees, represents, observes, or pays for. It need not be a measured failure. These contrasts are **questions that can reveal assumptions**, not a finite taxonomy or instructions about which algorithm to construct.

| Possible viewpoint | Question worth asking | What would *not* follow automatically |
| --- | --- | --- |
| Information and coordination | Does the desired global decision require a nonlinear statistic that workers cannot compute consistently from local inputs? | A shared sketch is necessarily the right solution |
| Representational freedom | Which constraint is imposed by the chosen coordinates, factorization, scale, or update rule rather than by the budget itself? | The less constrained representation improves training |
| State and temporal structure | Is stored information a residual, an estimate, a predictor, an optimizer state, or a mixture that changes meaning across events? | Any single state recurrence is universally preferable |
| Equivalent descriptions | Could a different but equivalent optimization variable or protocol expose an aggregatable/cheaper operation? | Algebraic equivalence survives quantization, stale state, or finite precision |
| Desired properties without a known defect | Could a legal algorithm have fewer dependent collectives, better coverage, lower peak memory, or different quality–cost trade-offs even when diagnostics find no large error? | The target property is valuable on the present workload |

The absence of a large SVD/selection/quantization gap does not close unrelated research directions. Equally, a large proxy gap is not a mandate to optimize that proxy. Counterexamples, invariances, new objectives, and designs outside these examples are all legitimate entry points; distinguish supported facts from hypotheses.

## 2. Treat the algorithm as an editable stateful protocol

A useful abstraction is

$$
H_{i,t}=\operatorname{Input}(g_{i,t},s_{i,t},s_t^{\mathrm{shared}}),
$$

$$
(c_t,s_{1,t+1},\ldots,s_{N,t+1},s_{t+1}^{\mathrm{shared}})
=\operatorname{Protocol}_t(H_{1,t},\ldots,H_{N,t};s_t,\omega_t).
$$

The protocol can contain multiple local-compute and collective stages. This matters for methods such as PowerSGD: it is not always a collection of independent local encoders followed by one merge.

Use explicit worker weights \(w_i\geq0\), with \(\sum_i w_i=1\), when the target is a weighted aggregate. The worked examples use fixed weights across events. Name local state, shared state, randomness, and the optimizer-visible output separately.

| Editable component | Construction question | Coupling to account for |
| --- | --- | --- |
| Encoded object | Can predictable information be reconstructed from shared state, leaving a smaller innovation? | Predictor agreement, residual units, optimizer placement |
| Representation | Can a different coordinate system or a small relaxation represent the target better? | Matrix shape, basis sharing, decoder, state coordinates |
| Selection objective | Should selection optimize instantaneous energy, persistent information, coverage, or update sensitivity? | Estimator target, temporal lag, resource constraints |
| Persistent state | What should be remembered: omitted information, a prediction, an estimate, or a subspace? | Initialization, update event, reset and resume |
| Randomness | Is randomness used for estimation, coverage, or reconstruction? | Independence assumptions and coupling across workers, directions, and time |
| Communication stages | Which statistics must be shared before a nonlinear decision? | Legal merge operation, ordering, latency, critical path |
| Resource allocation | Where does the next unit of communication or memory provide the most useful improvement? | Real payload, metadata, extra collectives, layer interactions |
| Optimization metric | Which perturbations matter to the actual parameter update? | Optimizer state and validity of any local approximation |

For each new quantity, classify its availability:

1. Already local.
2. Already shared and identical at this event.
3. Available only after a new collective.
4. Estimated from a specified sample or sketch.
5. Offline oracle information unavailable to the online algorithm.

This classification often reveals the next construction step. A proposal depending on category 5 needs an implementable substitute before it becomes an executable candidate.

## 3. Reusable construction patterns

### C1 — Change the encoded object

**Target:** exploit information already available to the receiver.

Start from \(H=P+(H-P)\), with a reference \(P\) that sender and receiver can reconstruct consistently. Encode the innovation \(H-P\), then add \(P\) at reconstruction. Specify how \(P\) is produced and updated; “use a predictor” is not an algorithm.

**Investigate:** whether the innovation has a useful spectrum, dynamic range, or support; whether local heterogeneity makes a shared predictor ineffective; whether a linear compressor cancels the proposed change. Account for the predictor's memory and for stale predicted components. Use this pattern when prediction and its state costs are justified; it is not limited to any one codec.

### C2 — Replace an expensive decision with the statistics it needs

**Target:** approximate an attractive oracle without communicating its full input.

Write the exact decision first. Identify the smallest exact or approximate statistic sufficient to make it. Consider a sketch, sample, cached estimate, or a two-stage decision. Derive the statistic's target under the actual worker aggregation.

**Investigate:** bias, variance, selection dependence, and the gap between estimating a statistic and making a good nonlinear decision from it. Shared random projections can estimate global quantities; averaging nonlinear local statistics may estimate a different object. Example B in Section 5 uses a two-stage global selection; the same principle applies to sketches and low-rank scores. See `sketching.md` for conditional estimators, decision regret, and legal merge semantics; these are tools, not a required route.

### C3 — Relax one representation constraint

**Target:** enlarge the feasible representation family without simply increasing its nominal budget.

Identify the restriction: fixed coordinate support, selection of existing basis columns, a single scale, or a prescribed factorization. Relax one restriction through a small solvable subproblem. Examples include mixing a few basis directions or choosing an equivalent factor scaling before quantization.

**Investigate:** which degrees of freedom the decoder must know and how to communicate or reproduce them. A real-arithmetic equivalence can disappear after quantization. The representation may be relaxed in quantization as well: Example A separates outliers from bulk quantization under an explicit bit budget.

### C4 — Introduce state to maintain a useful quantity

**Target:** replace repeated expensive reconstruction with an explicitly maintained estimate.

Name what the state approximates, derive its update, and separate tracking error from target drift. A schematic design objective is to obtain a recurrence such as

$$
\mathbb E[E_{t+1}\mid\mathcal F_t]
\leq (1-\rho)E_t+\text{drift term}+\text{estimation term},
$$

where the conditioning and each term must be derived for the proposed method. This display is a design template, not a guarantee for an arbitrary state update.

**Investigate:** whether a correction, forgetting rule, or refresh controls the problematic term. State-tracking mechanisms such as EF21 and the broader 3PC framework provide related examples; their exact recurrences and assumptions matter [P4, P6]. `stateful_optimizer.md` contrasts residual memory, gradient estimation, and optimizer-owned state without treating them as interchangeable.

### C5 — Add coverage or correction for omitted information

**Target:** obtain a property that does not rely entirely on the accuracy of a greedy decision.

Define the omission: an unvisited coordinate, an unseen subspace, or information delayed across a window. Construct random exploration, a residual correction channel, or a coordinated schedule that addresses it. Derive a conditional coverage or window property where possible.

**Investigate:** what exploitation capacity is sacrificed, how long omitted information can wait, and whether the guarantee concerns coefficients, cumulative updates, or training. Coverage can be coordinate-wise, direction-wise, or temporal; its conditional assumptions must be re-derived for each mechanism.

### C6 — Reallocate a fixed budget or change the proxy objective

**Target:** spend resources where their marginal contribution is larger.

A possible construction problem is

$$
\min_{\{b_\ell\}}\sum_\ell a_\ell D_\ell(b_\ell)
\quad\text{subject to}\quad
\sum_\ell B_\ell(b_\ell)\leq B.
$$

Declare the distortion \(D_\ell\), real encoding cost \(B_\ell\), and any sensitivity weight \(a_\ell\). For a fixed-input SVD oracle, adding one rank removes \(\sigma_{r+1}^2\) of squared residual; its communication cost depends on the actual factor or shared-basis protocol. Marginal benefit per byte is a construction heuristic, not a general solution of a discrete allocation problem.

Alternatively, derive a proxy from the optimizer. Under a frozen linear update map \(M\), update distortion depends on \(\|M(\hat g-g)\|^2\). This can motivate transformed coordinates or different scores, provided the shared state, inverse map, and approximation regime are specified.

**Investigate:** whether the proxy predicts the intended behavior and whether additional decisions, small collectives, or packing erase the resource benefit.

## 4. Learn a derivation from a paper

The following are pedagogical reconstructions of mechanisms, not claims about the authors' private discovery sequence.

### PowerSGD: make the representation aggregatable

**Goal:** approximate the global input \(\bar H\) with low-rank communication.

**Obstacle:** independently factorizing each worker's input and averaging both factors generally does not reconstruct the average matrix.

**Construction:** condition on a common factor \(Q\). The product \(H_iQ\) is linear in the input, so its mean can be obtained by AllReduce. Orthogonalize the shared result to obtain \(P\); then aggregate \(H_i^\top P\). With common state and ideal arithmetic,

$$
P=\operatorname{orth}(\bar H Q),\qquad
\hat H=P(\bar H^\top P)^\top=PP^\top\bar H.
$$

**Transferable operation:** arrange a nonlinear construction as stages with shared intermediate state and aggregatable linear statistics. Warm starts exploit history, while dependent collectives and factor computation create costs. See Algorithm 1 and Appendix A.3 [P1].

### GreedyLoRE: estimate the correct global decision statistic

**Goal:** choose useful current directions from a stored common orthogonal basis.

**Obstacle:** exact direction energies of the global input require expensive information.

**Construction:** for direction \(u_j\), draw a fresh \(v_j\sim\mathcal N(0,I_n)\), shared across workers and independent of the conditioned input and basis. Form the local scalar \(u_j^\top H_i v_j\), aggregate the scalars, then square. Its conditional expectation is \(\|u_j^\top\bar H\|^2\). The paper uses these estimates for direction selection and periodically refreshes the basis [P2].

**Transferable operation:** identify the global statistic needed by a decision, preserve the order of aggregation and nonlinearity, and communicate an estimator rather than the full object. Unbiased scores do not make the resulting top-r decision exact.

### 1-bit Adam: change the optimizer representation before compression

**Goal:** reduce communication while retaining useful adaptive-optimizer behavior.

**Obstacle:** the Adam-type update combines first- and second-moment states through a time-dependent nonlinear preconditioner; applying classical error-compensated gradient compression as though the optimizer were linear can alter the intended state/update semantics.

**Construction insight:** for a schematic Adam-like update (omitting bias corrections, weight decay and implementation-specific ordering),

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^{\odot 2},
$$

$$
u_t=\frac{m_t}{\sqrt{v_t}+\epsilon}.
$$

If a warmup period produces a sufficiently stable preconditioner, a **deliberate algorithmic change** is to hold it fixed during a later compression phase. With $D_\star=\operatorname{diag}((\sqrt{v_\star}+\epsilon)^{-1})$, the schematic update becomes $u_t=D_\star m_t$; this enables a different momentum-communication and error-compensation design. The actual 1-bit Adam protocol specifies its own warmup, momentum, compression, and update rules [P8]. It is **not** exactly vanilla Adam with a compressed gradient, and frozen $v$ is an empirical/algorithm-specific condition rather than a universal property.

**Transferable operation:** identify which nonlinear dependence makes a naive codec unsuitable, then ask whether an equivalent description, phase-specific approximation, or newly maintained state can make a desired communication protocol legal. Account for the change to the optimizer itself, not only its message size. See `stateful_optimizer.md` and `quantization.md`.

### Error memory and tracking: make history part of the algorithm

**Goal:** prevent each communication event from discarding information with no later correction.

**Construction:** give omitted information or an approximation error an explicit state and derive its recurrence. Classical EF stores a residual; other designs maintain and correct a gradient estimate. These are different state programs, not interchangeable names [P3, P4].

**Transferable operation:** decide what the state represents before choosing its update. Compressing differences, as in MARINA, and the broader state-dependent construction in 3PC offer neighboring design ideas [P5, P6]. The constructions below demonstrate different routes rather than restating those established tracking algorithms.

## 5. Worked constructions across three different routes

**Status of Examples A–C:** these are *constructed teaching candidates*, not claims of literature novelty, executed results, or proven training improvements. Each example starts from a different edit point: **quantization changes the representation and bit allocation**, **sparsification changes who decides a shared support**, and **temporal communication changes when workers synchronize**. Low-rank examples remain in Section 4 and `low_rank.md`; hybrids are constructed by combining operations only after specifying their interactions (see `hybrid.md`).

The examples use weights \(w_i\geq0\) with \(\sum_iw_i=1\); where an ordinary gradient average is intended, the weighted aggregate is \(\bar H=\sum_iw_iH_i\). Do not infer an AllReduce-compatible encoding from an abstract codec.

### Example A — Quantization: separate outliers from low-bit bulk values

#### A1. Desired property and obstacle

Suppose a small number of unusually large coordinates determines a per-block quantizer's dynamic range. The bulk coordinates then receive a coarse grid even if their magnitudes are small. The construction goal is **lower rounding variance under a fixed real message budget**, not merely a lower nominal bitwidth.

For a local block \(h_i\in\mathbb R^d\), a uniform \(b\)-bit stochastic quantizer using the range \([-M_i,M_i]\) has \(L=2^b\) evenly spaced levels and step

$$
\Delta_i=\frac{2M_i}{L-1}.
$$

For a scalar between adjacent levels \(a\) and \(a+\Delta_i\), round to the upper level with probability \((x-a)/\Delta_i\), otherwise to the lower level. Conditional on the fixed input and the computed range, the decoded scalar \(Q(x)\) obeys

$$
\mathbb E[Q(x)\mid x,M_i]=x,\qquad
\mathbb E[(Q(x)-x)^2\mid x,M_i]\leq \frac{\Delta_i^2}{4}.
$$

These formulas assume a representable finite range, correct random rounding, and no clipping. An actual implementation may introduce cast or packing error. If \(M_i=0\), encode the all-zero block as a special case.

#### A2. Relax one representation restriction

Instead of forcing all values to use one low-bit grid, choose a set of \(k\) exceptional coordinates \(T_i\) (for example, the \(k\) largest absolute values of **the actual compressor input**). Encode those entries with a declared high-precision format, and quantize only the remaining entries with a new range

$$
M_i'=\max_{j\notin T_i}|h_{i,j}|,\qquad
\Delta_i'=\frac{2M_i'}{2^b-1}.
$$

Take high-precision exception entries to be exact in the *mathematical idealization*; if the implementation transmits FP16/BF16 or another rounded format, its error must be measured separately. Conditional on \(h_i,T_i\), the bulk's expected squared reconstruction error satisfies

$$
\mathbb E\|Q_{T_i}(h_i)-h_i\|_2^2
\leq\frac{d-k}{4}(\Delta_i')^2
$$

when exceptions are transmitted exactly. This bound exposes the possible benefit of narrowing the bulk range. It **does not** prove that this particular mixed codec beats a different quantizer under equal bytes, or that its compression noise improves training.

A useful extension is to choose \(T_i\) based on **expected reduction in actual bulk distortion per extra exception bit**, not solely magnitude. That extension requires a specific estimator and cost model.

#### A3. Make the message and merge legal

One straightforward executable protocol is:

1. At the same communication event, each worker selects \(T_i\), computes \(M_i'\), and quantizes the \(d-k\) bulk coordinates with fresh local random bits.
2. Pack the low-bit bulk codes in a deterministic order that skips the transmitted exception indices. Send the scale/range, exception indices, and exception values as metadata/payload.
3. **AllGather the encoded messages**; every worker decodes every message and forms \(\sum_iw_i\hat h_i\). A backend that only supports integer AllReduce is *not* automatically compatible with per-worker scales, variable exception supports, or bit-packed codes.
4. If using EF, write the next residual from the **actual local decoded reconstruction**, using the recurrence defined by the algorithm. Do not substitute the globally aggregated reconstruction.

An idealized per-worker payload estimate (excluding alignment, headers, and framing) is

$$
B_i\approx b(d-k)+k\,b_{\mathrm{exc}}+k\lceil\log_2 d\rceil+b_{\mathrm{scale}}\quad\text{bits}.
$$

The cost of supporting variable exception counts, index coding, packing, and the **collective topology** must be added. AllGather can scale differently from a compressed AllReduce as worker count grows. Compare against the actual baseline's *network bytes and critical path*, not only \(B_i\).

An alternative with shared scales/supports may permit another merge rule, but obtaining that shared information requires its own protocol and usually additional communication.

#### A4. First discriminator, failure, and repair

- **Oracle question:** on real saved compressor inputs, how much does \(M_i'\) decrease, and how much rounding error is removed by the exact-exception idealization?
- **Equal-budget question:** compare reconstructed MSE, conditional bias, realized message bytes, and actual aggregate error against a tuned baseline at the same wire budget. Repeat rounding RNG on *fixed* inputs.
- **Failure:** the exception indices and high-precision values consume more bits than they save, or different worker supports make AllGather expensive.
- **Possible repairs:** share support through a genuinely shared statistic, choose exceptions per larger block, or abandon exception coding when the measured outlier structure is weak. Each repair changes the cost and possibly the estimator's target.

**Reusable construction operation:** identify where a representation's shared precision resource is wasted; separate hard-to-represent values; derive both the distortion bound and the actual encoded/merged message.

### Example B — Sparsification: nominate locally, choose support globally

#### B1. Desired property and obstacle

Suppose each worker performs local Top-\(k\) sparsification, but the desired quality target concerns the **global mean** \(\bar h=\sum_i w_ih_i\), not each \(h_i\). Large local entries can cancel across workers. Separate local Top-\(k\) supports need not include coordinates that are important in \(\bar h\).

The offline fixed-\(k\) global L2 oracle keeps \(\operatorname{TopK}(|\bar h|)\), but computing \(\bar h\) densely would defeat communication sparsification. Ask instead: *which small candidate set is sufficient to approximate that oracle?*

#### B2. Construct a two-stage decision

For worker \(i\), form a local nomination set \(S_i\) of size \(q\) from the actual input. Make the common candidate set

$$
T=\bigcup_{i=1}^N S_i.
$$

Use **two different kinds of message**:

1. **Support stage:** AllGather nominated indices \(S_i\); sort and deduplicate to construct identical \(T\) on every worker.
2. **Value stage:** every worker extracts its **actual input values** \(h_i[T]\), including coordinates it did not nominate, and AllReduces the weighted values to obtain \(\bar h[T]\).
3. **Selection stage:** every worker deterministically selects \(K=\operatorname{TopK}_k(|\bar h[T]|)\) with a defined tie rule, and reconstructs \(\hat h_j=\bar h_j\) on \(K\) and zero elsewhere.

A complete and identical candidate set is essential before the second collective. Only reducing values contributed by nominating workers would give biased global values for \(T\), even before Top-\(k\) selection.

The result is exactly the best \(k\)-sparse approximation **restricted to \(T\)**:

$$
\hat h=\arg\min_{\operatorname{supp}(z)\subseteq T,\,\|z\|_0\leq k}
\|\bar h-z\|_2^2.
$$

If \(T\) contains the global oracle support, it matches the full global Top-\(k\) oracle; otherwise it does not. Local nomination has no general guarantee of containing that support. For example, local cancellation can cause large nominated coordinates to be irrelevant globally, while smaller but aligned coordinates fail to be nominated.

#### B3. Make the cost and EF semantics explicit

An idealized *logical* support payload consists of up to \(Nq\) indices before deduplication. The value stage communicates \(|T|\) dense values per worker, not merely \(k\), with \(|T|\leq\min(d,Nq)\). The final reconstructed result has support at most \(k\), but **final sparsity is not the wire budget**. Include index format, deduplication, collective latency, and actual AllReduce/AllGather wire traffic.

This design applies one **global** nonlinear selection after aggregating values. It is not equivalent to applying independent local sparse compressors and averaging their outputs. If EF is added, the location of the single global decision changes the meaning of local residual writeback; derive an actual state recurrence rather than blindly applying \(e_{i,t+1}=h_{i,t}-\hat h\). In particular, there may be no locally decoded sparse message corresponding to the global selector unless the algorithm explicitly defines one.

#### B4. First discriminator, failure, and repair

- **Oracle question:** how much true global Top-\(k\) energy lies in the nomination union \(T\) on real synchronized multi-worker inputs?
- **Protocol question:** does the resulting global sparse reconstruction improve matched global error over the current method at comparable true communication and collective costs?
- **Failure:** \(T\) becomes large, the second collective approaches dense communication, or useful global coordinates rarely appear in any local nomination.
- **Possible repairs:** replace magnitude nomination with a cheaper global candidate estimator, coordinate a shared random exploration subset, or allocate a fixed candidate budget across workers. Changing how \(T\) is formed requires new approximation analysis.

**Reusable construction operation:** move a nonlinear selection decision to the place where its target statistic is available, using a cheap preliminary stage and a more accurate restricted second stage.

### Example C — Temporal communication: synchronize when shared-anchor drift is large

#### C1. Desired property and unavailable information

For Local SGD or DiLoCo-style windows, reducing synchronization frequency saves communication but can let local models diverge. A tempting trigger is worker dispersion

$$
D_t=\sum_iw_i\|x_{i,t}-\bar x_t\|_2^2,
\qquad \bar x_t=\sum_iw_ix_{i,t}.
$$

But \(D_t\) is not available locally without communication. Triggering a collective independently when one worker crosses a local threshold would also lead to inconsistent collective order and possible deadlock.

Instead, use a **common anchor** \(a\) from the last synchronization and a fixed heartbeat schedule, so every worker knows when to communicate.

#### C2. Derive a cheap common decision statistic

At every \(J\) completed local optimizer updates (as defined by the algorithm), worker \(i\) computes

$$
r_i=\|x_i-a\|_2.
$$

AllReduce a scalar maximum to obtain \(R=\max_i r_i\), then all workers use the same decision. Weighted variance decomposition gives

$$
D=\sum_i w_i\|x_i-a\|_2^2-\|\bar x-a\|_2^2
\leq\sum_iw_i\|x_i-a\|_2^2\leq R^2.
$$

Thus at the **heartbeat event only**, \(R\leq\tau\) certifies \(D\leq\tau^2\) in this parameter-space metric. The inequality does not certify training loss, optimizer stability, or drift between heartbeats. It may be loose: identical movement by all workers gives \(D=0\) while \(R\) is large.

#### C3. An executable synchronization protocol

1. Begin with an agreed anchor \(a\) and fully specified local optimizer/outer state. All workers perform the same number \(J\) of inner updates before each heartbeat; specify behavior when updates are skipped or microbatch accumulation is used.
2. At the heartbeat, all workers AllReduce the scalar maximum \(R\). They make the *same* decision: synchronize if \(R>\tau\), or if the maximum window length \(H_{\max}\) has been reached; otherwise continue another \(J\) updates.
3. On synchronization, compute the prescribed weighted model average (or **the separately defined outer-delta/outer-optimizer transition**, if using a DiLoCo-style method). Only then reset the common anchor and window counter according to that method's semantics.
4. Keep data/token and scheduler accounting explicit. Resume must restore anchor, heartbeat phase, counters, inner state, and any outer state. Any worker or numerical exception must preserve collective order.

The scheme pays a scalar collective every \(J\) inner updates plus occasional full synchronization. The scalar heartbeat may dominate in a latency-sensitive network; counting only full synchronization messages would overstate savings. Setting \(H_{\max}\) to a multiple of \(J\) avoids an off-grid final event; other choices need a specified boundary rule.

#### C4. First discriminator, failure, and repair

- **Oracle question:** on short real windows, how does the observable bound \(R^2\) track actual model dispersion \(D\), and how often would it trigger?
- **Quality/cost question:** compare completed-window local/outer states, fixed-token quality, total communication, heartbeat latency, and eventual time-to-target with a matched fixed-window baseline.
- **Failure:** common-mode movement causes excessive synchronizations; divergence grows sharply between heartbeats; or heartbeat latency cancels the expected benefit.
- **Possible repairs:** use a tighter shared statistic (with its extra messages), change the heartbeat cadence, or construct a predictive bound from locally maintained state. A new trigger needs a new validity argument; changing \(J\) alone is a parameter change, not a new mechanism.

**Reusable construction operation:** start from an unavailable global condition, derive a cheap common sufficient signal, and make both the trigger and its communication schedule consistent across workers.

## 6. Convert failures into repair opportunities

Preserve the failed assumption and what still works. Distinguish a constructed counterexample from an actual run.

| Failure or limiting case | What it reveals | Possible structural repairs |
| --- | --- | --- |
| Independent factor averages reconstruct the wrong object | Missing common coordinates or an illegal merge | Share a factor, align coordinates, or aggregate a different intermediate statistic |
| A predictor construction reduces algebraically to its baseline | New state does not change observable behavior | Change the represented quantity or abandon the redundant state |
| Outlier exceptions cost more than they save | Representation error improves but on-wire budget does not | Co-design group size, exception selection, coding and legal aggregation |
| Local sparse nominations omit useful global support | Local selection does not match the global decision | Change nomination statistic, share candidates, or introduce exploration |
| Local synchronization triggers disagree across ranks | The global control decision is not common | Use a fixed heartbeat/collective and common trigger |
| Added state helps a proxy but not updates | The proxy or stored information may be misaligned | Derive a different objective or reconsider what the state tracks |

Record a failed attempt as:

~~~text
Attempt and scope:
Intended property:
Observed failure OR constructed counterexample:
Failed assumption:
Properties that still hold:
Possible repairs and their additional costs:
Conditions under which the original idea remains plausible:
Evidence or derivation:
~~~

A workload-specific failure is not a universal ban. Conversely, repeated failures under the same conditions should inform candidate selection.

## 7. Specify and compare a candidate

### Minimal candidate card

Use this card to communicate an **executable** candidate if helpful; it is not a prerequisite for brainstorming and not a scorecard for ranking exploratory hypotheses.

~~~text
Motivation:
  Observed limitation OR desired property, with the relevant resource constraint.

Structural change:
  Which object, representation, operator, state transition, or dependency changes?

Minimal algorithm:
  Equations or pseudocode; local/shared quantities; event and collective order.

Runtime information:
  What is already available, newly communicated, estimated, or only an oracle?

Communication and reconstruction:
  Payload fields, shape/dtype, weights, merge rule, decoder, real extra messages.

State:
  Initialization/conversion, writeback, refresh/reset branches, and resume state.
  Which event advances it: microbatch, synchronization, or successful update?

Why it might help:
  Separate directly derived properties, conjectures, and measured observations.

Distinct behavior:
  A small input/state where this differs from the baseline; degeneracy conditions.

Closest methods:
  Similarities and structural differences; prior-art checks still outstanding.

Cost and first discriminator:
  Added memory/compute/communication; the cheapest observation that changes the decision.
~~~

### Compare mechanisms, not names

Compare the encoded object, persistent state, update equations, and communication dependencies against the nearest methods. Identify limits in which the candidate reduces to an existing method. A useful composition can merit evaluation without being a new primitive.

At the idea stage, “closest prior art not yet checked” is acceptable. Before claiming originality, check the nearest literature and implementations. Do not treat a renamed recurrence, an equivalent factorization, or a parameter setting as evidence of novelty.

### Move from construction to evidence

An executable candidate should have a defined protocol, legal information dependencies, and a testable reason to matter. It need not have a complete convergence theorem or demonstrated training gain before its first experiment.

Use existing D1–D5 and family references selectively to test its specific assumptions. If moving to short closed-loop training under the evaluation contract, each branch recomputes gradients after its parameters diverge; replaying a saved gradient sequence studies state response to external inputs. Runtime attribution and semantics-preserving implementation work follow the Engineer reference.

Use results to revise the mechanism, change its supporting hypothesis, or stop the branch. Do not retrofit the original prediction after observing the outcome.

## 8. Sources and further construction examples

The family references remain the broader literature index. The citations below are precedents for construction *operations*, not originality claims for the Section 5 teaching candidates. See also `quantization.md`, `sparsification.md`, and `temporal.md` for route-specific closest prior art.

- **[P1] [PowerSGD, arXiv v3](https://arxiv.org/html/1905.13727v3)** — Algorithm 1 and Appendix A.3: shared intermediate state, linear aggregation, low-rank reconstruction.
- **[P2] [GreedyLoRE, arXiv v4](https://arxiv.org/html/2507.08784v4)** — Sections V-B/V-C and Algorithms 2/3: basis maintenance, global direction scoring, random statistics, and phase-dependent state. Check the exact implementation and theorem rather than importing a generic guarantee.
- **[P3] [Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://proceedings.mlr.press/v97/karimireddy19a.html)** — explicit residual memory and its role in compressed optimization.
- **[P4] [EF21](https://arxiv.org/abs/2106.05203)** — an alternative error-feedback/tracking construction; compare the actual recurrence with classical EF.
- **[P5] [MARINA](https://proceedings.mlr.press/v139/gorbunov21a.html)** — gradient-difference compression and estimator design.
- **[P6] [3PC: Three Point Compressors](https://proceedings.mlr.press/v162/richtarik22a.html)** — a state-dependent compressor framework that recovers EF21 and generates additional methods.
- **[P7] [Error Feedback Reloaded, arXiv v1](https://arxiv.org/abs/2402.10774v1)** — an explicitly described discovery sequence: an equivalent reformulation with impractical machine cloning, an executable weighted method, and improved analysis of the original method. Useful for studying how an impractical ideal construction can suggest a feasible one.
- **[P8] [1-bit Adam, ICML 2021](https://proceedings.mlr.press/v139/tang21a.html)** — optimizer preconditioning and momentum communication: the algorithmic object and state may need redesign before very-low-bit compression is meaningful.
- **[P9] [Sketched-SGD, NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/75da5036f659fe64b53f3d9b39412967-Abstract.html)** — use linear, mergeable sketches to approach a global sparsity decision with less direct communication.
- **[P10] [ATOMO, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/33b3214d792caf311e1f00fd22b392c5-Abstract.html)** — sparsification in alternative atomic representations; a coordinate choice is itself a design constraint.

For any paper-derived construction card, record the target and constraints, direct approach, obstacle, key transformation, required assumptions, new cost, and transferable operation. Keep the original result separate from extensions proposed in the current research.
