# Algorithm Reference Toolkit — Distributed Gradient Compression (V1)

**用途**：为 Researcher 提出可证伪的算法假设、Research Analyzer 解释 quality/gap 提供静态参考。按问题选读；**不要求每个 Node 跑全部诊断**，也不实现 Reference Mutation / Promotion。

每条工具按 **Applicable Conditions → Mathematical Principle → Metrics → Diagnostic Procedure → Potential Mutations** 组织。数学关系仅在各自假设下使用；诊断观察不是收敛定理。正式 quality/efficiency 训练**不得插入**额外梯度采集、oracle 或 profiling；诊断需独立执行并记录固定代码/模型状态/配置/比较对象。

## 0. 比较对象先定义清楚

- `G_t`：待说明是 local worker gradient、跨 worker 聚合梯度、某个 bucket/layer，还是加入 EF 前的目标更新。不要将它们混用。
- `C(X)`：实际算法在给定状态、rank、随机性与通信阶段下得到的有效重建结果；不能把压缩器单独运行的输出直接等同于训练中 optimizer 可见的更新。
- `R_t = G_t - C(G_t)`：**无 EF 或指定输入直接压缩**时的基准残差。若压缩输入是 `u_t = G_t + e_t`，应另报 `r_t = u_t - C(u_t)` 与 `C(u_t)-G_t`，不把两者都叫 `R_t`。
- 典型 EF 形式 `u_t=G_t+e_t, \tilde G_t=C(u_t), e_{t+1}=u_t-\tilde G_t` 满足 `\tilde G_t-G_t=e_t-e_{t+1}`；实际实现的符号/聚合位置/状态生命周期需要核对。
- 所有相对量先固定范数、归一化口径、shape、dtype、时间点和对照 checkpoint。局部 dense reference ≠ 独立 dense 训练轨迹。

## 1. Mathematical Toolkit

### 1.1 Eckart–Young–Mirsky：最佳 rank-r oracle

- **Applicable Conditions**：矩阵型梯度/残差可做低秩近似，关心给定 rank 的表示能力。
- **Mathematical Principle**：若 `G=UΣVᵀ`，最佳 rank-`r` Frobenius 误差满足 `min_{rank(X)≤r} ||G-X||_F² = Σ_{i>r} σ_i²`。
- **Metrics**：top-r 能量比例、oracle 重建误差、算法与 oracle 的误差差值、跨层 rank 收益。
- **Diagnostic Procedure**：在**独立诊断**的同状态梯度上计算 SVD oracle；与实际 compressor 的同 rank、同输入误差比较。
- **Potential Mutations**：rank allocation、分层 rank、自适应表示；若 oracle 很好而算法很差，优先改善子空间估计，而非贸然增加 rank。

### 1.2 Contractive / Unbiased Compressor：压缩性质

- **Applicable Conditions**：评估误差反馈或优化理论所要求的压缩假设。
- **Mathematical Principle**：contractive 形式可写为 `E[||C(G)-G||² | G] ≤ (1-δ)||G||²`，`δ>0`，期望仅对压缩器随机性；unbiased 是**另一条件** `E[C(G)|G]=G`，两者不可混同。
- **Metrics**：经验相对平方误差、保留能量、偏差随 step / layer / rank 的变化、极端失败样本。
- **Diagnostic Procedure**：在多输入与随机 seed 下记录经验统计，检查算法假设；无法由有限样本推出全局 contractivity 或 unbiased guarantee。
- **Potential Mutations**：混合压缩、控制偏差、error correction；若 oracle 本身差，应先质疑表示族。

### 1.3 Davis–Kahan / Wedin：子空间扰动与谱间隙

- **Applicable Conditions**：想区分 basis tracking 误差与真实 dominant subspace 漂移。
- **Mathematical Principle**：谱/奇异子空间扰动界依赖 perturbation 大小与适当的 spectral gap；Davis–Kahan 面向对称特征空间，Wedin 面向奇异子空间。
- **Metrics**：奇异值与 gap、principal angles、`||P Pᵀ-U_r U_rᵀ||_F`（`P,U_r` 均为正交列基且维度匹配）。
- **Diagnostic Procedure**：同 step 比较使用中的 basis 与 oracle 子空间；再比较连续 step 的 oracle 子空间，分开估计 tracking error 和真实 drift。
- **Potential Mutations**：adaptive refresh、oversampling、子空间再正交化、更稳健的投影策略。谱间隙很小时不要把 basis 角度变化直接解释成质量退化。

### 1.4 Power / Subspace Iteration：子空间估计

- **Applicable Conditions**：压缩器通过迭代近似 top singular subspace，使用 warm start 或周期性更新。
- **Mathematical Principle**：迭代收敛速度与谱结构、初始子空间、迭代次数及数值稳定性相关。
- **Metrics**：固定 rank 的重建误差、迭代/刷新开销、正交性偏差、warm start 与冷启动差距。
- **Diagnostic Procedure**：同一输入和随机初态下改变迭代步数、初始化和正交化；观察误差—运行成本曲线。
- **Potential Mutations**：iteration budget、warm start 规则、refresh 触发条件、block iteration。

### 1.5 Error Feedback：历史压缩误差

- **Applicable Conditions**：实现保存并补偿残差，或考虑 selective/decayed EF。
- **Mathematical Principle**：见第 0 节的 EF 状态递推。瞬时 `||r_t||` 不足以预测长期质量；状态/优化器交互很重要。
- **Metrics**：残差幅度、时间自相关、与后续梯度或更新方向的内积/夹角、跨步参数差异。
- **Diagnostic Procedure**：使用与正式训练分离的同状态短程对照，比较 EF 开关或策略；跟踪跨步状态，避免只看单步重建误差。
- **Potential Mutations**：selective EF、decayed EF、subspace-aware EF、分层残差修正。

### 1.6 Randomized Linear Algebra：随机投影与 sketch

- **Applicable Conditions**：希望以更小代价估计子空间、保留信息或做混合表示。
- **Mathematical Principle**：随机 sketch 的质量依赖投影维度、分布、输入谱与 oversampling；不可无条件当作精确 SVD。
- **Metrics**：sketch 重建误差、方差、随机 seed 敏感性、额外计算量/通信量。
- **Diagnostic Procedure**：对固定输入比较不同 sketch 维度与分布，并对照确定性低秩和 SVD oracle。
- **Potential Mutations**：sketch-assisted basis、hybrid low-rank + sparse、随机与历史子空间融合。

### 1.7 Smoothness / Descent Lemma：误差与训练更新

- **Applicable Conditions**：判断梯度方向误差是否可能影响下降性质，或解释质量退化。
- **Mathematical Principle**：若目标 `L` 的梯度为 `L_s`-Lipschitz，则 `L(x+s) ≤ L(x)+⟨∇L(x),s⟩+(L_s/2)||s||²`。该局部界**不能直接证明**实际非凸分布式训练的长期收敛。
- **Metrics**：压缩更新与参考更新的 alignment、norm ratio、局部 loss 变化、跨步质量曲线。
- **Diagnostic Procedure**：在固定 checkpoint 的独立诊断中比较候选更新及短程训练差异，说明优化器、学习率和状态影响。
- **Potential Mutations**：误差敏感层选择、更新方向修正、按训练阶段调整压缩强度。

## 2. Diagnostic Templates（按需选择，不构成固定必跑流程）

### A. Residual Structure Analysis

基于明确的 `G_t` 和 `C` 定义 `R_t=G_t-C(G_t)`；存在 EF 时按第 0 节分别分析输入残差与 optimizer 可见误差。

| 视角 | 可观测问题 | 潜在方向 |
|---|---|---|
| **Spectral** | Residual 的奇异值谱是否集中、是否仍低秩？ | 二次低秩修正、rank 重分配 |
| **Coordinate** | 大幅值坐标是否集中/稀疏？ | Low-Rank + Sparse Residual Correction |
| **Temporal** | Residual 是否平稳、可用历史信息预测？ | 预测性补偿、adaptive refresh |
| **Cross-worker** | 不同 worker 的 residual 是否存在一致方向或共享结构？ | 共享子空间、协同校正；注意额外通信成本 |

同时检查 residual 与**后续实际优化器更新**的相关性，避免“残差很大”就等价于“值得传输”。Cross-worker 诊断需区分 local 与聚合后的信号，并计算新方案的通信收益是否抵得过开销。

### B. Contractivity / Oracle-Gap Analysis

测量经验 `||C(G)-G||² / ||G||²`，与同输入同 rank 的最佳 SVD oracle 比较。

- Oracle 自身误差大 → 可能是表示族/rank 容量不足。
- Oracle 好、算法差 → 可能是 basis estimation、刷新或数值实现问题。
- EF 缓冲持续增大 → 检查压缩和补偿机制；**不能单凭增长断言理论不稳定**。

报告经验值及适用条件；不要把单个训练轨迹的能量保留率当作 compressor 的全局 contractivity guarantee。

### C. Subspace Drift Analysis

设使用中的正交 basis 为 `P_t`，同状态目标 dominant singular subspace 为 `U_t`：

`D_t = ||P_tP_tᵀ - U_tU_tᵀ||_F`。

比较 `D_t`（tracking error）和相邻 `U_t` 的 projector 距离（真实 drift），并报告 spectral gap：

- 子空间稳定、tracking error 大 → basis estimation / warm start 问题。
- 子空间变化快 → 检验 adaptive refresh。
- Gap 小 → oversampling、block iteration、稳定子空间选择可能更合适；慎解读单个 basis 的旋转。

### D. Error Feedback Analysis

按实际实现记录 `u_t=G_t+e_t, e_{t+1}=u_t-C(u_t)` 或注明变体。测残差幅度、时间相关性、future-gradient/update alignment、optimizer 状态与短程质量。候选改动可包括 selective EF、decayed EF、subspace-aware EF。对照实验须明确是否改变算法语义。

## 3. Reference 使用与输出约定

- **Idea generation**：先引用 parent Analysis 的具体观察，再选 1–2 条最相关工具，提出一个可证伪 design delta；不要套模板批量发散。
- **Analysis**：写出 Applicable Conditions、比较对象、原始证据、替代解释、可区分解释的最小独立诊断、Potential Mutation 及其 quality / efficiency 风险。
- **诊断权限**：在独立 diagnostic 中运行；正式训练不额外采样。结论必须关联 Node、Commit/Attempt、Run/log 与 checkpoint 来源。
- **参考论文方向**：PowerSGD、GreedyLore、Error Feedback、随机数值线性代数、低秩扰动理论。引用论文时核实具体算法的假设，不能将参考结果无条件套到现有实现。
