# Semantic / Correctness Criteria (Prompt 软约束)

## Researcher：新算法 correctness，不要求等价于旧算法
- 输入/输出 shape、dtype、device、参数分组与分布式 rank 一致性。
- 所有进程通信顺序一致，collective 不死锁，world-size 变化按设计处理。
- 压缩/重建值 finite；error feedback、低秩状态、optimizer 可见梯度与算法定义匹配。
- 若干连续 step 的状态更新符合设计；使用已有 small correctness/smoke tests。
- 记录**运行的命令、输入、结果、未覆盖项**；不能仅写“looks correct”。

## Engineer：相对研究实现的有限语义保持检查
- 固定初值/seed/配置，在同一语义定义下比较重建梯度、optimizer 可见值与更新。
- 比较误差反馈等跨 step 状态的**可观察行为**；不同内部低秩因子表示不必逐元素一致。
- 检查多 rank 和典型 world-size；注意浮点重排容差、非确定性与测试限制。
- 使用仓库已有 tests，必要时写小范围差分 smoke script；V1 不强制平台化。
- PASS/FAIL/NOT_RUN 必须配测试命令及未覆盖项。发现失败必须修复或停止该工程候选。
- 仅在可信度足够时声明“在测试覆盖范围内语义保持”；不得写“证明所有质量不变”。
