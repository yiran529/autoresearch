# Engineer Runtime & Communication Performance Reference

**Primary reader:** Engineer. This document owns **measured execution time and performance attribution**: step/wall-clock time, collective latency, encode/decode cost, critical path, communication–computation overlap, memory movement and performance regressions. It is separate from Research Analyzer's study of information loss, optimization updates and **algorithmic state evolution across training steps**.

**Use as a selective diagnosis toolkit, not a mandatory profiling checklist.** Follow the repository's trusted `workflow_rules.md`, `roles/engineer.md`, current `system_ref.md`, evaluation contract and semantic criteria. This reference does not override the existing claim/submit/finish controls or the **two distinct formal Engineering eval candidates per claim**. Profile only in authorized independent diagnostic/scratch settings; official timing runs must remain clean.

## 1. Decide what performance difference is real

First establish exact configuration and comparators: frozen Research SHA, selected Engineering SHA, device/world size, GPU occupancy and contention, model, optimizer, dtype, batch, warmup, timing window, repetition, network domain, and whether reported quality is **inherited (provisional)**. Use **complete ordinary/refresh/synchronization cycles** for amortized metrics and retain run-to-run spread. Compare candidate vs its *fixed* Research anchor, not a different algorithm/configuration. A profiler trace is not the official grader's timing result.

## 2. Initial performance diagnostics — pick only what explains the anomaly

| ID | Trigger | Minimum inputs | Measurement | How to interpret |
|---|---|---|---|---|
| E1 End-to-end timing | Unexpected speedup or slowdown | clean formal timing and matched baseline | step-time distribution, mean/dispersion, repetitions, warmup/measurement boundary | effect must survive noise and comparable hardware |
| E2 Communication ledger | Nominal compression not reflected in step time | collective trace, actual wire format, rank/worker topology | collective calls, sizes, real bytes including index/scale/basis/refresh/control, waiting/dependency | nominal payload reduction does not imply critical-path savings |
| E3 Compute & packing | Factor/Top-k/quantizer overhead suspected | separate representative trace, actual tensor shapes | GEMM/QR/SVD/scoring/sorting/pack/unpack/copy kernel durations, launches and workspaces | check whether the costly kernel lies on the end-to-end critical path |
| E4 Overlap & synchronization | Communication appears hidden or stalls appear | matched GPU timeline + stream/collective ordering | overlap periods, waits, idle gaps, serialization points | do not add overlapped durations mechanically |
| E5 Memory movement | GPU memory/bandwidth or allocation bottleneck | allocation and memcpy profile, state/buffer lifetimes | peak memory, copies, allocator sync, temporary workspaces, memory bandwidth | an in-place change may alter the algorithm's persistent state |
| E6 Periodic bursts | Refresh or outer synchronization yields unstable step time | trace across one or several **full** algorithmic periods | ordinary vs refresh/outer time, event frequency, peak stalls, amortized cost | sampling only ordinary steps can understate true time |

**Selection rule:** start from official time anomaly; choose the cheapest measurement that can distinguish *implementation overhead*, *algorithm-imposed dependency*, *resource contention* and *invalid comparison*. Compute/bytes ratios alone do not establish which factor dominates.

## 3. Symptom → follow-up diagnosis cards

### SYS-NOSPEED — Transmitted bytes fall, wall-clock does not
- **Plausible explanations:** encode/decode and factor work, extra small collectives, latency domination, baseline communication already hidden under compute, unsupported network target.
- **Discriminate:** compare official matched step timing with E2–E4, including complete refresh cycles; identify changed critical-path segments, not just isolated kernel benchmarks.
- **Evidence boundary:** if only a modeled network is available, call results predictions rather than measured speedups.

### SYS-REFRESH — Large periodic spikes
- **Plausible explanations:** full-matrix refresh communication, QR/SVD costs, blocking collectives, allocator events, data stalls.
- **Discriminate:** annotate exact code event, separately profile all components, measure full-cycle average and max; compare with an equivalent no-probe official trace.
- **Boundary:** reduction in a single spike does not ensure whole-training speedup.

### SYS-COLLECTIVE — Many small calls or poor effective bandwidth
- **Plausible explanations:** latency overhead, dependent collectives, small bucket shapes, bandwidth limitation, topology mismatch.
- **Discriminate:** actual message-size distribution, collective type/order, network and GPU traces, launch-to-complete latency. Compare against realistic same-hardware collective baselines.
- **Boundary:** a closed-form `α+bytes/β` is an approximate model; bandwidth is not universally constant.

### SYS-KERNEL — Heavy local compression compute
- **Plausible explanations:** QR/SVD/Top-k sort, low occupancy, memory layout, extra GPU synchronization, repeated allocation/packing.
- **Discriminate:** measured kernel shapes, allocation events, critical-path membership, scaling with layer dimensions and actual batch. An isolated kernel microbenchmark may not predict end-to-end change.
- **Boundary:** changing rank, compression rule, refresh frequency or optimizer is an **algorithm change**, not an Engineering optimization.

### SYS-OVERLAP — Isolated kernels faster but step time unchanged
- **Plausible explanations:** accelerated portion fully hidden, displaced work on another stream, serialization elsewhere, GPU contention.
- **Discriminate:** compare before/after end-to-end timelines, synchronized device measurements and time spent on the actual critical path.
- **Boundary:** profiler segment sums can exceed or under-represent elapsed step time.

### SYS-LOW-SYNC — Fewer synchronization events, limited wall-clock gain
- **Plausible explanations:** large periodic messages, outer optimizer compute, checkpoint/copy bursts, dependency stalls, network different from target deployment.
- **Discriminate:** full local/outer windows with exact inner/outer step and token meanings, transfer sizes and duration; report amortized time per training token and clean end-to-end results.
- **Boundary:** `1/H` fewer sync rounds does not imply `1/H` time or equivalent training quality. Engineering must not change `H` to improve timing without a new Research Node.

## 4. Typical per-family performance questions

| Family | First check | Common hidden costs |
|---|---|---|
| Low-rank | factor kernels and collective dependencies across ordinary and refresh steps | QR/SVD, factor broadcasts, two-stage reductions, temporary matrices |
| Quantization | actual encoded on-wire dtype/packing and reductions for scales | casts, pack/unpack, groupwise scales, small launches |
| Sparsity | whether real sparse collective handles index/value representation efficiently | Top-k sort, index bytes, coalescing, scatter/gather, noncontiguous buffers |
| Local SGD / DiLoCo | full local/outer synchronization window under actual topology | periodic model-delta bursts, outer optimizer compute, cross-worker waits |
| Hybrid | entire critical path after combining mechanisms | duplicated copies, extra state, inconsistent overlap, more collectives |

## 5. Optimization Toolkit — select a semantics-preserving implementation change

**Usage:** First verify a bottleneck using E1–E6 and a SYS diagnosis card. Choose **one primary critical-path target** for a bounded engineering change. The Engineer may design and implement optimizations; validate with small differential correctness checks and separate profiling before spending formal evaluation budget. The repository currently allows **two distinct formal Engineering eval candidates per claim**, not a parameter sweep.

Document each recipe as: `observed bottleneck + evidence → proposed implementation-only change → preserved semantics → minimum differential checks → end-to-end confirmation / stop condition`. **Technical feasibility is not evidence of either equivalence or speedup.**

| Recipe | Triggering evidence | Plausible implementation optimizations | Key semantic risks |
|---|---|---|---|
| **OPT-KERNEL** | E3: repeated GEMM, QR, Top-k, quantizer or elementwise kernels on critical path | vectorization, batching same-shape operators, removing duplicate casts, valid elementwise fusion and launch reduction | floating-point order, RNG consumption, stable tie handling, QR conventions |
| **OPT-MEMORY** | E5: large temporary tensors, repeated allocation/copy or allocator synchronizations | preallocation, reusable buffers/workspaces, improved layout/contiguity, proven-redundant copy elimination | overwriting cross-step EF/momentum state, async races, tensor aliasing |
| **OPT-COLLECTIVE** | E2: latency from many small messages/dependent communication launches | pack/aggregate compatible payloads, contiguous communication buffers, better bucket layout | inconsistent collective order, altered reduction order, changed reconstruction boundaries |
| **OPT-OVERLAP** | E4: independent compute and communication serialize unnecessarily | async collectives, stream/event scheduling, delay waits until dependencies, valid staging | consuming incomplete data, missing waits, inconsistent rank order |
| **OPT-ENCODE** | E2/E3: pack/unpack, index/value conversion or serialization dominates | exact bit packing, batched encoding, avoid repeated metadata conversion, more efficient scatter/gather | altered quantization code/scale, duplicate or lost indices, decode order or dtype |
| **OPT-PERIODIC** | E6: refresh / outer synchronization work creates major stalls | reusable workspaces, batching equivalent independent operations, removing redundant copies, dependency-safe staging | changed refresh frequency, outer-optimizer input, persistent state, synchronization points |

### OPT-KERNEL / OPT-MEMORY — local compute and data movement

- **Locate:** collect actual input shape, repetitions, dtype/layout, allocations, stack and critical-path placement. A slow kernel hidden under communication might not matter for wall-clock.
- **Try:** remove redundant casts/layout conversions/intermediates; batch structurally identical operations; fuse compatible elementwise work; reuse temporary workspace with verified lifetime. For low rank, inspect many small GEMM/QR and factor reconstruction; for sparsity, selection/sorting; for quantization, scaling/rounding/packing.
- **Preserve:** the compressor output and persistent state under the existing semantic criteria, including RNG/distribution where stochastic, EF writeback, and optimizer-visible updates. A floating-point reorder is not automatically equivalent.
- **Validate:** fixed-input/RNG output and multi-step state differentials plus edge cases; only then separate kernel profiling and clean official end-to-end timing. A faster isolated kernel with unchanged step time does not establish the intended gain.

### OPT-COLLECTIVE / OPT-OVERLAP — network operations and dependencies

- **Locate:** actual collective type, size/count, launch/completion order and dependency chain; identify legal overlap windows and whether the original communication was already masked.
- **Try:** pack compatible payloads, reduce needless launch overhead, improve contiguous buffer layout, use async communication and events only when data dependencies and per-rank order allow.
- **Preserve:** every worker's valid collective ordering; reconstructed/reduced result within accepted tolerances; no optimizer consumer may read incomplete data. Do not eliminate a required synchronization or defer it across an optimizer update boundary.
- **Validate:** multi-rank deadlock/consistency smoke tests, output and optimizer-update comparison, then actual critical path and official timing. Fewer API calls alone are not enough.

### OPT-ENCODE — codec- and representation-specific optimizations

- **Quantization:** verify low-bit payload is genuinely packed on wire; preserve scale, stochastic rounding, clipping, and generated code values. Accidentally reducing precision is an algorithm change.
- **Sparsification:** avoid repeated index tensor construction/workspace allocation; profile sorting, duplicate-index coalescing, alignment and true bytes. Changing Top-k support, threshold or EF semantics is not allowed.
- **Low rank:** reuse factor buffers, avoid demonstrably unnecessary full-size reconstructions, optimize QR/GEMM shapes and preserve two-stage communication dependencies. Cutting rank or subspace iterations changes the algorithm.
- **Validate:** trace encoded object, metadata, and decoded output with fixed inputs; cover empty/small tensors, non-divisible blocks, dtype boundaries and real network bytes. Only semantics-preserving end-to-end improvement qualifies.

### OPT-PERIODIC — refresh, bursts and outer steps

- **Locate:** over full cycles, distinguish communication bursts, basis refresh compute, model-delta construction, outer optimizer and memory transfer.
- **Try:** reuse safe workspaces, remove exact redundant copies, batch equivalent independent operations, and improve staging without changing update time/dependencies.
- **Preserve:** period length, trigger steps, communicated object, lifetime of EF/basis states and outer-optimizer state transitions.
- **Validate:** full-period amortized and worst-event timing plus state differentials across the refresh/outer boundary. A faster one-off refresh microbenchmark is not sufficient.

### Minimum evidence to graduate an optimization candidate

1. **Measured target:** point to a specific E1–E6 verified overhead and why it may affect end-to-end time.
2. **Bounded transformation:** document the implementation-only change and invariants; algorithm changes are outside an Engineering claim.
3. **Differential checks:** compare fixed-seed/input, degenerate cases, multi-rank results, persistent state and optimizer-visible updates; note numerical tolerances and coverage.
4. **Profiling then official eval:** use separate profiling for attribution and clean grader timing for speed evidence. Record variability, regressions and failures. A candidate failing the semantic checks is not a confirmed Engineering success.

## 6. Existing workflow and evidence requirements

- Use the assigned Engineer's **one active claim, clean worktree and fixed Research SHA**. An Engineering SHA may build on a preceding Engineering attempt as allowed by the host, while `base_commit` and `quality_anchor` still identify the original Research SHA.
- Report official efficiency and separate profiler findings with exact paths, run IDs, hardware and variation; note regressions and infrastructure faults. Official quality is **inherited, not freshly proven**.
- **Do not** change compression semantics, EF reset, optimizer schedule, communication frequency, rank, bitwidth, or model training objective as a hidden time optimization. If a structural algorithm change is required, explain that limitation in Engineering evidence; it belongs to a future Research Node.

## 7. Useful references

- [PyTorch DDP communication hooks](https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html) — bucket/hook interface, state and collective behavior; check actual installed PyTorch source/version.
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — CPU/GPU traces, operators, device memory; profiling has overhead and is not a clean official timing pass.
- [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — collective semantics and tuning/environment behavior; topology and backend version matter.
- [PowerSGD](https://arxiv.org/abs/1905.13727) — low-rank factor and aggregation structure; use for identifying algorithm-imposed communication dependencies.
- [DiLoCo](https://arxiv.org/abs/2311.08105) and [Streaming DiLoCo](https://arxiv.org/abs/2501.18512) — outer-step communication bursts, fragmentation/overlap as adjacent systems prior art; optimization semantics may differ.
- Repository `references/system_ref.md` — existing roofline/critical-path, collective cost, QR/GEMM/SVD, overlap, and memory analysis. Consult selectively alongside this diagnostic decision guide.
