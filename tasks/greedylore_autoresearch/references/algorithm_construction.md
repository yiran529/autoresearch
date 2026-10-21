# Algorithm Construction for Communication-Efficient Distributed Training

**Primary reader:** Researcher.

**Purpose:** construct candidate algorithms from an observed limitation **or a desired property under an explicit resource constraint**. A candidate does not need prior evidence of improved training quality to be proposed. It needs a concrete mechanism, an account of the information available at runtime, and a reason the change could matter.

Read **common.md** for communication objects and comparison semantics. Use the relevant family reference for established mechanisms, **initial_diagnostics.md** for D1–D5, **semantic_criteria.md** for correctness, and **engineer_runtime.md** for measured runtime and implementation attribution. These are canonical repository filenames. This reference adds no run quotas, submission procedures, or role permissions; the existing workflow and evaluation contract remain authoritative.

**Use selectively.** Start with Sections 1–3, then consult the most relevant construction pattern or example. Examples illustrate transformations, not an exhaustive menu. It is valid to use a construction not listed here.

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

**Investigate:** whether the innovation has a useful spectrum, dynamic range, or support; whether local heterogeneity makes a shared predictor ineffective; whether a linear compressor cancels the proposed change. Account for the predictor's memory and for stale predicted components. Example A develops this pattern.

### C2 — Replace an expensive decision with the statistics it needs

**Target:** approximate an attractive oracle without communicating its full input.

Write the exact decision first. Identify the smallest exact or approximate statistic sufficient to make it. Consider a sketch, sample, cached estimate, or a two-stage decision. Derive the statistic's target under the actual worker aggregation.

**Investigate:** bias, variance, selection dependence, and the gap between estimating a statistic and making a good nonlinear decision from it. Shared random projections can estimate global quantities; averaging nonlinear local statistics may estimate a different object. Example B uses this distinction.

### C3 — Relax one representation constraint

**Target:** enlarge the feasible representation family without simply increasing its nominal budget.

Identify the restriction: fixed coordinate support, selection of existing basis columns, a single scale, or a prescribed factorization. Relax one restriction through a small solvable subproblem. Examples include mixing a few basis directions or choosing an equivalent factor scaling before quantization.

**Investigate:** which degrees of freedom the decoder must know and how to communicate or reproduce them. A real-arithmetic equivalence can disappear after quantization. Example B relaxes discrete column selection through a two-dimensional rotation.

### C4 — Introduce state to maintain a useful quantity

**Target:** replace repeated expensive reconstruction with an explicitly maintained estimate.

Name what the state approximates, derive its update, and separate tracking error from target drift. A schematic design objective is to obtain a recurrence such as

$$
\mathbb E[E_{t+1}\mid\mathcal F_t]
\leq (1-\rho)E_t+\text{drift term}+\text{estimation term},
$$

where the conditioning and each term must be derived for the proposed method. This display is a design template, not a guarantee for an arbitrary state update.

**Investigate:** whether a correction, forgetting rule, or refresh controls the problematic term. State-tracking mechanisms such as EF21 and the broader 3PC framework provide related examples; their exact recurrences and assumptions matter [P4, P6].

### C5 — Add coverage or correction for omitted information

**Target:** obtain a property that does not rely entirely on the accuracy of a greedy decision.

Define the omission: an unvisited coordinate, an unseen subspace, or information delayed across a window. Construct random exploration, a residual correction channel, or a coordinated schedule that addresses it. Derive a conditional coverage or window property where possible.

**Investigate:** what exploitation capacity is sacrificed, how long omitted information can wait, and whether the guarantee concerns coefficients, cumulative updates, or training. Example C allocates a fixed rank budget to coverage.

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

### Error memory and tracking: make history part of the algorithm

**Goal:** prevent each communication event from discarding information with no later correction.

**Construction:** give omitted information or an approximation error an explicit state and derive its recurrence. Classical EF stores a residual; other designs maintain and correct a gradient estimate. These are different state programs, not interchangeable names [P3, P4].

**Transferable operation:** decide what the state represents before choosing its update. Compressing differences, as in MARINA, and the broader state-dependent construction in 3PC offer neighboring design ideas [P5, P6]. Example A below is a separate teaching construction, not a restatement of one of these algorithms.

## 5. Worked constructions

**Status of Examples A–C:** constructed teaching candidates. Their displayed algebra is conditional on the stated definitions. They are not executed experiments, established improvements, or claims of literature novelty. Use their construction operations to generate other candidates.

In the matrix examples, \(H_i\in\mathbb R^{m\times n}\); \(\|\cdot\|_F\) denotes the Frobenius norm.

### Example A — Shared prediction and innovation compression

#### A1. Desired property and construction

Suppose part of the communicated input is predictable from a common historical state. Can the receiver reconstruct that part while the message describes only a correction?

For one synchronization event, let

$$
H_{i,t}=g_{i,t}+e_{i,t},
\qquad
v_{i,t}=H_{i,t}-p_t,
$$

where \(p_t\) is identical on every worker. Define a decoded innovation \(q_{i,t}=C_{i,t}(v_{i,t})\), and

$$
c_{i,t}=p_t+q_{i,t},\qquad
c_t=p_t+\sum_i w_iq_{i,t},
$$

$$
e_{i,t+1}=H_{i,t}-c_{i,t},\qquad
p_{t+1}=c_t.
$$

For a fresh start, initialize \(p_0=0\) and \(e_{i,0}=0\); a switch from another method requires an explicit state conversion. Every worker applies its defined optimizer to the same \(c_t\). The predictor is the reconstructed communication object, not the optimizer's parameter update.

#### A2. Online protocol

The codec must implement the weighted sum of decoded innovations. A generic nonlinear codec cannot be assumed to support an ordinary AllReduce over its codes.

A concrete special case uses a common column-orthonormal basis \(P_t\in\mathbb R^{m\times r}\), with \(P_t^\top P_t=I_r\):

$$
q_{i,t}=P_t a_{i,t},\qquad
a_{i,t}=P_t^\top(H_{i,t}-p_t).
$$

Reduce the coefficients \(a_{i,t}\), reconstruct \(c_t=p_t+P_t\bar a_t\), and compute each residual using the local coefficients. A local residual must not accidentally use another worker's or the globally averaged coefficients.

Here the rank-r correction has the usual coefficient shape. The reconstructed output can have rank greater than r because it also includes the predictor. A rank-r oracle for the complete output is therefore not the matched representation comparator.

#### A3. What follows, and what remains a hypothesis

The bookkeeping identity is

$$
c_{i,t}-g_{i,t}=e_{i,t}-e_{i,t+1}.
$$

This follows by substitution. It does not establish that the residual is bounded or that the predictor improves training. If the innovation compressor is contractive, its direct error bound concerns \(\|H_{i,t}-p_t\|\), so predictor mismatch enters the residual analysis.

The construction hypothesis is that the innovation is easier to encode under the real budget. Local heterogeneity or a rapidly changing target can make that hypothesis false.

**A stability stress case:** a contractive component is insufficient to certify the joint state program. For one worker and the scalar compressor \(C(z)=\tfrac32z\), the compression error satisfies \(|C(z)-z|^2=\tfrac14|z|^2\). Once the input gradient becomes zero, the proposed recurrence is

$$
\begin{pmatrix}e_{t+1}\\p_{t+1}\end{pmatrix}
=
\begin{pmatrix}-1/2&1/2\\3/2&-1/2\end{pmatrix}
\begin{pmatrix}e_t\\p_t\end{pmatrix}.
$$

Its eigenvalues are \((-1\pm\sqrt3)/2\); one has magnitude greater than one. Thus state excited by a preceding nonzero gradient can grow even with this contractive compressor. This is a constructed counterexample, not evidence that all practical innovation codecs fail. It identifies a repair question: which predictor update, correction, or forgetting rule controls the coupled state?

#### A4. A degeneracy that the candidate must pass

For a fixed linear compressor \(A\),

$$
p+A(H-p)=AH+(I-A)p.
$$

If \(A\) is a projection and \(p\) lies in its range, the proposed change exactly reduces to \(AH\). A fixed shared projection with \(p_0=0\) and the predictor update above stays in this degenerate case.

With a changing projection, the predictor can contribute outside the current range. That creates a real behavioral difference, but can also retain stale directions. A different name or extra state is not evidence of a different useful mechanism.

#### A5. Costs, boundary events, and first discriminator

- Account for the dense predictor state, additional memory traffic, and checkpoint state. Reusing a buffer requires a valid lifetime.
- Specify predictor and residual transitions on dense refresh, skipped updates, and any outer synchronization.
- First check for algebraic degeneracy. Then compare real innovations and absolute inputs under the same state and wire budget.
- Separate gains from easier innovations from gains due to additional historical representation. Either may be useful, but attribute the mechanism and charge its state costs. If neither helps under the intended constraint, revise the predictor or stop this branch.

**Reusable lesson:** change the object using shared information, then test whether the resulting state program changes behavior usefully and whether its correction is easier to communicate.

### Example B — Relax column selection through a small basis rotation

#### B1. Identify the restriction

Let \(U\in\mathbb R^{m\times m}\) be a full common orthonormal basis. Selecting r columns does not exhaust all r-dimensional subspaces: the full span is available, but the permitted choices are restricted to coordinate subspaces.

This motivates a different design variable: allow a few existing directions to combine while keeping the final rank fixed.

Consider an ordinary compression step with \(1\leq r<m\). Choose a common pair \(J=(a,b)\), where a is selected and b is unselected. Keep the other r-1 selected columns fixed. Pair selection must use information available to every worker; choosing a pair near a score boundary is only a heuristic.

#### B2. Solve a two-dimensional oracle problem

For the fixed global input \(\bar H=\sum_i w_iH_i\), define

$$
A=U_J^\top\bar H=
\begin{pmatrix}x^\top\\y^\top\end{pmatrix},
\qquad
K=AA^\top=
\begin{pmatrix}
\|x\|^2&\langle x,y\rangle\\
\langle x,y\rangle&\|y\|^2
\end{pmatrix}.
$$

Selecting one original column retains at most \(\max(K_{11},K_{22})\) of this pair's energy. Allowing a unit combination \(u'=U_Jv\) retains \(v^\top Kv\). Its optimum is

$$
\lambda_{\max}(K)
=\frac{K_{11}+K_{22}
+\sqrt{(K_{11}-K_{22})^2+4K_{12}^2}}{2}
\geq\max(K_{11},K_{22})\geq K_{11}.
$$

If \(x=y\neq0\), the direction \((u_a+u_b)/\sqrt2\) retains all the pair's energy, while either original column retains half. This is a constructed fixed-input example, not a claim about real training gradients.

The off-diagonal term identifies a potentially useful structure: correlation between projection rows. Marginal direction energies alone do not reveal it.

#### B3. Replace unavailable global information

Treat exact K as an oracle for the low-message candidate: obtaining it online requires an explicitly costed protocol, such as reducing both full projection rows. To estimate it with a smaller message, draw a new common

$$
R\in\mathbb R^{n\times s},\qquad
R_{jk}\sim\mathcal N(0,1/s).
$$

Use fresh randomness independent of the input and pair-selection randomness, conditional on the already selected pair and state. Every worker computes

$$
Z_i=U_J^\top H_iR.
$$

AllReduce the weighted sketches:

$$
Z=\sum_i w_iZ_i=AR,\qquad
\hat K=ZZ^\top.
$$

Since \(\mathbb E[RR^\top]=I\),

$$
\mathbb E[\hat K\mid H_1,\ldots,H_N,U,J]=K.
$$

Two protocol decisions are essential:

1. Reduce \(Z_i\) before forming the Gram matrix. Averaging local Gram matrices estimates a different target.
2. Use the same R for both directions. Independent direction-specific random vectors do not supply the required cross-correlation through their product.

Reusing the randomness that selected J requires a new selection-dependent analysis; the displayed conditional expectation does not justify it.

#### B4. An executable ordinary-step candidate

1. Construct J using the existing common selection result.
2. Obtain \(\hat K\) with the additional sketch reduction.
3. Choose a unit leading eigenvector v of \(\hat K\), with a specified tie rule.
4. Form \(u'=U_Jv\) and let P contain u' plus the unchanged r-1 columns.
5. Compute local \(a_i=P^\top H_i\), reduce \(\bar a=\sum_iw_ia_i\), and reconstruct \(c=P\bar a\).
6. If using ambient-space local EF with \(H_i=g_i+e_i\), write \(e_i'=H_i-Pa_i\). Apply the specified full-space optimizer to c.

Preserve the local \(a_i\), or finish the local residual update, before an in-place reduction overwrites the coefficient buffer.

The first version uses a temporary P and leaves the stored U unchanged. Since u' is a unit vector in the pair's plane, it is orthogonal to the unchanged columns. A later variant may persist a full two-column rotation, but that introduces a separate cross-step state change.

Keep the baseline's explicitly defined refresh branch for this minimal candidate. Low-dimensional optimizer states would require their own coordinate-transport rule and are outside this example.

All workers must use the same P. Use consistent eigenvector/tie handling or a designated worker and a small broadcast when local computations do not reliably agree.

#### B5. Cost and first discriminator

One pair adds a 2-by-s sketch reduction, local projection work, and a dependent decision before coefficient communication. The 2s elements describe the collective input payload, not total network traffic; wire bytes depend on dtype, worker count, and the collective implementation. Any required broadcast also counts. Final rank-r coefficients retain their shape, but total communication and latency are not automatically unchanged.

First distinguish two exact gains:

$$
\Delta_{\mathrm{baseline}}=\lambda_{\max}(K)-K_{11},\qquad
\Delta_{\mathrm{mixing}}=\lambda_{\max}(K)-\max(K_{11},K_{22}).
$$

The first measures improvement over the currently selected column a; the second isolates the benefit of mixing over choosing the better of a and b. Measure these on realistically selectable pairs from saved inputs. If the mixing gain is negligible, better column selection may suffice. If it is substantial, test whether the small sketch preserves it.

An unbiased \(\hat K\) does not guarantee that its eigenvector increases true retained energy. If the approximation loses the oracle benefit, change the estimator or reject the branch before expensive training.

**Reusable lesson:** relax a discrete representation restriction through a small continuous subproblem, then estimate only the information needed to solve that subproblem.

### Example C — Derive a coverage mechanism from a desired bound

#### C1. Desired property

Suppose a candidate should retain some information regardless of errors in its greedy scoring. This is a design target, not a claim that original GreedyLoRE lacks a contraction result; that paper already analyzes its own mechanism [P2].

Assume a full common orthonormal basis with m columns and a total rank budget r. Choose \(k\geq0\) greedy columns and \(s\geq1\) exploration columns, with \(r=k+s\leq m\). For a mixed example use \(0<k<r\).

#### C2. Construct the selection rule

Let G be any common set of k columns. Independently sample a size-s subset S uniformly without replacement from the remaining m-k columns, using shared fresh randomness. Let \(P_G=U_G\) and \(P=[U_G,U_S]\). Define

$$
C(H)=PP^\top H.
$$

Keep the selected coefficients unscaled: this is an orthogonal projection, not an inverse-probability-scaled unbiased estimator.

Condition on H, U, and G before the exploration draw. Each remaining basis direction is included with probability \(s/(m-k)\). Orthogonality gives

$$
\mathbb E_S\|H-C(H)\|_F^2
=\left(1-\frac{s}{m-k}\right)
\|H-P_GP_G^\top H\|_F^2
\leq\left(1-\frac{s}{m-k}\right)\|H\|_F^2.
$$

This is a conditional compressor property independent of the quality of the greedy scores. It relies on the complete basis and the specified sampling law. A thin basis cannot provide the same whole-space conclusion. Reusing an exploration subset across events can couple it to the current input; the displayed conditional result then needs a new argument.

#### C3. Online protocol and trade-off

All workers derive the same G and S, send \(P^\top H_i\), and reconstruct \(P\sum_iw_iP^\top H_i\). The final coefficient payload still has r directions. Include the cost of obtaining G, shared randomness, selection, and basis state.

The exploration slots replace some greedy slots. Consequently, the rule may retain less instantaneous energy than pure greedy selection at the same total r. Its appeal is the stated robustness property; whether that property improves the actual task remains a research question.

For r=1 there is no mixed allocation with both k and s positive. Randomly switching between rank-one greedy and exploratory steps is a different construction requiring its own derivation.

With a fixed basis and fresh draws, a particular direction not already selected by G has a per-step exploration opportunity. If the basis changes, a column index no longer identifies a persistent physical direction; do not turn this into an unqualified coordinate-age guarantee.

#### C4. First discriminator and further construction

Compare the fixed-budget loss of greedy energy with coverage of directions the intended application repeatedly omits. If omitted directions do not matter to updates, the guarantee may not justify the trade-off.

Potential next constructions include residual-informed coverage or coordinated coverage over windows. They change the sampling law or dependence structure and therefore require new bounds; the displayed result cannot simply be reused.

**Reusable lesson:** start from a desired property, identify a mechanism that supplies it, and allocate resources to that mechanism while exposing what it displaces.

## 6. Convert failures into repair opportunities

Preserve the failed assumption and what still works. Distinguish a constructed counterexample from an actual run.

| Failure or limiting case | What it reveals | Possible structural repairs |
| --- | --- | --- |
| Independent factor averages reconstruct the wrong object | Missing common coordinates or an illegal merge | Share a factor, align coordinates, or aggregate a different intermediate statistic |
| Predictor construction reduces algebraically to the baseline | New state does not change observable behavior | Change the represented quantity or abandon the redundant state |
| Full-basis column selection leaves a large oracle gap | Discrete subspace choices are restrictive | Small rotations, different basis maintenance, or another representation |
| Exact oracle helps but its sketch does not | The useful decision information is not recovered | Change sketch coupling, estimation target, or decision rule |
| A correction works locally but fails after aggregation | Local and global objectives differ | Reconstruct the global target statistic and redesign its protocol |
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

The family references remain the broader literature index. These sources are selected for the construction operations they illustrate.

- **[P1] [PowerSGD, arXiv v3](https://arxiv.org/html/1905.13727v3)** — Algorithm 1 and Appendix A.3: shared intermediate state, linear aggregation, low-rank reconstruction.
- **[P2] [GreedyLoRE, arXiv v4](https://arxiv.org/html/2507.08784v4)** — Sections V-B/V-C and Algorithms 2/3: basis maintenance, global direction scoring, random statistics, and phase-dependent state. Check the exact implementation and theorem rather than importing a generic guarantee.
- **[P3] [Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://proceedings.mlr.press/v97/karimireddy19a.html)** — explicit residual memory and its role in compressed optimization.
- **[P4] [EF21](https://arxiv.org/abs/2106.05203)** — an alternative error-feedback/tracking construction; compare the actual recurrence with classical EF.
- **[P5] [MARINA](https://proceedings.mlr.press/v139/gorbunov21a.html)** — gradient-difference compression and estimator design.
- **[P6] [3PC: Three Point Compressors](https://proceedings.mlr.press/v162/richtarik22a.html)** — a state-dependent compressor framework that recovers EF21 and generates additional methods.
- **[P7] [Error Feedback Reloaded, arXiv v1](https://arxiv.org/abs/2402.10774v1)** — an explicitly described discovery sequence: an equivalent reformulation with impractical machine cloning, an executable weighted method, and improved analysis of the original method. Useful for studying how an impractical ideal construction can suggest a feasible one.

For any paper-derived construction card, record the target and constraints, direct approach, obstacle, key transformation, required assumptions, new cost, and transferable operation. Keep the original result separate from extensions proposed in the current research.
