# System Reference Toolkit — Distributed Communication Efficiency (V1)

**用途**：辅助 Engineer 选择优化方向、撰写分析并找出算法结构对 efficiency 的限制。静态、按需读取，不开发 Reference Mutation/Promotion。每条工具统一使用 **Applicable Conditions → System Principle → Metrics → Diagnostic Procedure → Potential Mutations**。

目标是无额外诊断开销的**端到端摊销平均步时**，包括通信、压缩、刷新、正交化及周期性开销。Profiler 只用于**独立诊断**；不把 profiler 各项加和当成正式 step time。

## 1. Systems Toolkit

### 1.1 Roofline / Critical Path

- **Applicable Conditions**：怀疑当前 step 的主要耗时在算力、带宽、同步或可重叠工作上。
- **System Principle**：kernel 的算术强度决定理想计算/内存带宽界；**实际端到端关键路径**还由事件依赖、通信和 overlap 决定。局部 kernel 加速可能不改变全局步时。
- **Metrics**：step time、kernel duration、FLOPs、读写字节、memory bandwidth、GPU utilization、关键路径长度。
- **Diagnostic Procedure**：在独立 profiling 中构造实际依赖图，找出关键路径；区分 compute-bound、memory-bound、communication-bound 与 synchronization-bound 区域。
- **Potential Mutations**：语义保持的 fusion、layout、buffer reuse、kernel 重写；若关键路径受算法依赖限制，再提出减少刷新/同步等算法改动。

### 1.2 Collective Cost Model

- **Applicable Conditions**：压缩通信、rank/bucket 划分、collective 编排可能主导步时。
- **System Principle**：粗略成本可写为 `T_comm ≈ n_collective·α + bytes/β_eff + topology/algorithm overhead`；仅作模型，实际 collective 类型、网络拓扑、并发和消息尺寸都会改变有效参数。
- **Metrics**：collective 次数、消息尺寸、实际字节、延迟、有效带宽、参与 worker、通信依赖序列。
- **Diagnostic Procedure**：列出每个通信操作的依赖与临界路径；测量按大小分组的延迟，不把减少字节数等同于必然加速。
- **Potential Mutations**：融合小消息、调整 bucket、减少轮次、拓扑感知交换；如果轮次来自算法的数据依赖，要作为 Research 问题，不伪装为 Engineering 微调。

### 1.3 Kernel Profiling：QR / GEMM / SVD 等

- **Applicable Conditions**：压缩/重建的低秩计算、正交化、刷新显著耗时。
- **System Principle**：算法引入的局部计算可能抵消压缩通信的收益；QR/SVD 类型、shape、dtype、内存布局影响代价。
- **Metrics**：各核耗时、调用次数、占用率、launch overhead、shape、workspace 和频率。
- **Diagnostic Procedure**：独立 profiling；将一次性或周期性计算摊入完整测量窗口；确认核确实位于端到端关键路径。
- **Potential Mutations**：kernel fusion、批处理、布局复用、减少临时分配；改变迭代/正交化频率属于潜在算法 mutation，需重新评估 quality。

### 1.4 Overlap Analysis

- **Applicable Conditions**：计算与通信可流水化，或某些同步阻断并行。
- **System Principle**：只有无真实依赖冲突的操作可以重叠；减少通信独立计时不保证缩短 step critical path。
- **Metrics**：依赖图、GPU stream 时间线、可隐藏区间、等待/空闲时间、重叠后的端到端步时。
- **Diagnostic Procedure**：在独立 trace 中检查发起时机、wait、collective 顺序、跨 rank 一致性；比较优化前后端到端时间，而非仅看单个通信操作。
- **Potential Mutations**：调整 scheduling、prefetch、异步通信、bucket 提交；若同步来自前一计算结果才能决定通信内容，则需要 Research 修改依赖结构。

### 1.5 Memory / I/O Analysis

- **Applicable Conditions**：重建张量、EF buffer、低秩因子或中间 workspace 造成额外带宽/显存压力。
- **System Principle**：显存读写和临时分配可成为 bottleneck；可复用内存与原地计算必须考虑跨步语义和生命周期。
- **Metrics**：峰值显存、allocated/retained memory、allocator 事件、拷贝次数、内存带宽、buffer live range。
- **Diagnostic Procedure**：在固定配置下分析内存流量与 buffer 生命周期，检查读写依赖与多 rank 不一致风险。
- **Potential Mutations**：buffer reuse、减少中间张量、layout 优化、in-place 重建；改变 EF 精度/保存策略通常属于算法语义改变。

## 2. 典型分析模板

### A. Cost Decomposition / Amortization

把一段完整的代表性训练窗口拆成 `compute + compress/decompress + communication + refresh/orthogonalization + synchronization`，**只用作归因，不机械相加为总 step time**。若每 `k` 步发生一次刷新，可先估算 `((k-1)T_ordinary + T_refresh)/k`，再通过完整窗口的正式端到端测量确认。对自适应刷新报告实际事件频率。

### B. Implementation vs Algorithm Limitation

| 观察 | 需进一步判断 | 下一步 |
|---|---|---|
| 小消息启动延迟高 | 是否可无语义变化地 fusion 或 batching？ | Engineer 测 scheduling/fusion |
| QR/迭代开销大 | 算法需要多少次迭代才能达到 quality 目标？ | Engineer 优化 kernel；Researcher 研究频率/迭代预算 |
| 通信与计算未重叠 | 是实现未排好，还是数据依赖本身阻止？ | 依赖图 + 可行 overlap 的对照 |
| 通信字节下降但 step time 不降 | 关键路径是否已转向压缩/刷新/同步？ | profile + 修订成本模型 |
| 更换网络/worker 数后表现变差 | 哪些结论依赖拓扑和通信模型？ | 明确适用范围，不外推 |

### C. Claim Calibration

只在有硬件模型、实际依赖和可复核证据时说“**在当前语义与条件下存在结构性成本**”；不要把“当前没有优化成功”直接说成“理论下界”。区分：

1. **Observed bottleneck**：trace / benchmark 直接支持。
2. **Conditional structural limitation**：在声明假设下由依赖/通信模型推导。
3. **Unknown**：缺少证据，指定最小可区分诊断。

## 3. Reference 使用与输出约定

- **Idea generation**：从 parent 的 quality/efficiency Analysis 找到实际未解决 gap，选择最相关的一个成本模型，写出可验证的工程或算法假设。
- **Engineering Analysis**：总结所有 Engineer Attempts（失败与退化也包括），关联 commit、语义自检、端到端步时、独立 profile；区分可在语义不变下优化的实现开销与需要算法改变的成本。
- **诊断隔离**：正式 efficiency 不跑 profiler；单独 diagnostic 的 code/config/设备拓扑/输入和证据引用必须明确。
- **适用范围**：所有性能结论注明硬件、网络、world size、workload、batch/shape、dtype、实现和评测版本。
