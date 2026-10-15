---
name: experiment-results-reporting
description: Use when a user asks to summarize, compare, audit, or document machine-learning experiment settings and metrics from run artifacts or experiment trackers.
---

# Experiment Results Reporting

把一次或一组机器学习实验整理成简洁、可复核的结果文档或 review。默认面向仓库内的
`docs/results.md`，但应遵循用户指定的输出路径。

先判断工作模式：

- **create/update**：用户要求生成或修改文档时才写文件；默认使用用户指定路径。
- **review**：用户只要求审查时只读，不修改文档；若用户另行授权，再进入 update。

## 工作范围

- 先阅读仓库的 `AGENTS.md`、`readme.md`、实验入口脚本和相关 worklog；仓库根目录的 `.slurm` 示例包含旧集群路径，应以实际启动记录为准。
- 先识别每个实验入口的结果契约，再读取运行产物。候选来源包括 JSON、CSV、日志、
  W&B、TensorBoard 和 worklog；`all_results.json` 只是某些入口的候选格式，不是通用必需文件。
- 建立 `experiment -> run identity -> source -> metric -> split/checkpoint/step` 映射；配置与
  硬件信息来自实际启动脚本或运行记录。
- 不补写未经证实的数字、超参数、硬件、数据规模或失败原因。找不到的内容省略，
  不用常识填充。
- 将 GreedyLore 论文结果、复现实验结果和 Muon 组合实验结果分开；论文中针对 Adam/MSGD 的收敛结论不能直接归于 Muon。单 seed 结果标为初步结果，不写成
  稳健统计结论。

## 默认文档结构

保持短小，通常只保留以下四部分（详见
[concise-template.md](references/concise-template.md)）：

1. 实验矩阵：实验编号、模型、方法、训练预算（steps/epochs/tokens 等）。
2. 核心设置：数据集、关键超参数、硬件和运行脚本；只保留影响复现或解释的字段。
3. 结果：用户要求的主要/次要指标，并注明 split、final/best、step 或 checkpoint。
4. 结论：同口径对照差值、最重要观察、复制次数和限制。

不要默认写入：W&B 地址、tokenizer 细节、依赖版本、原始数据清单、完整路径清单、
吞吐、timing、显存或通信量。只有用户明确要求，或它们是解释结果所必需的信息时才加入。

## 结果提取与比较

1. 识别 canonical 结果来源，优先级按“入口定义/用户指定”而不是文件扩展名判断：
   - 用户明确指定或训练入口定义的 final output；
   - 与 run identity 和完成状态匹配的机器可读 JSON/CSV；
   - 匹配的 W&B summary/history 或 TensorBoard event；
   - 带明确完成标记的日志；
   - worklog/项目说明仅作上下文或交叉核对。
2. 核验 run 是否完成、目标 step/epoch/token 是否达到，以及指标对应的 split。不能把
   崩溃前最后一条日志自动当作 final，也不能默认用 min/max 代替 best。
3. 若报告 best，记录选择规则、checkpoint/step 和 split；若报告 final，记录完成标记和
   最终 step/epoch/checkpoint。若来源冲突，不静默择值，应记录冲突并标记 unresolved。
4. 检查报告表格中的每个数字都能定位到 source 的 raw key/column/tag、run identity 和
   step/split；差值和相对变化用完整精度计算后再四舍五入展示。
5. 无压缩与压缩方法必须按相同模型规模、数据、优化器和评价口径比较；seed 应匹配预先规定的
   seed 集。多 seed 结果报告 `n` 以及离散度或置信区间；单 seed 只作初步结果。
6. 失败运行不隐藏：区分 observed error 与 confirmed root cause；若后续修复并重跑，
   简要记录首次失败、修复和重跑状态，不把失败目录当成成功结果。

## 指标口径

- 指标名称随任务变化：C4 可使用 loss/PPL，GLUE 可使用 accuracy/F1/Matthews，
  CIFAR-10 可使用 loss/accuracy；以实际训练入口的评价输出为准。
- PPL 只有在 loss 是自然对数、token-level causal cross-entropy 时才可用
  `exp(loss)` 推导；其他 loss 不得套用该公式。
- 超参数 sweep 选出的 best 配置要披露选择过程，不能把选择后的结果当作独立确认结果。
- 比较通信代价时，区分压缩 hook 和 Muon 优化器内部 collective；只有计数口径一致时才报告总量。

## Review 与验证

用户要求 review 时，请独立检查：

- 数值是否与实际 canonical source 一致；
- 实验设置、脚本和失败记录是否有证据；
- final/best、split、step/checkpoint 和复制次数是否明确；
- 结论是否超出证据支持的范围；
- 是否混入用户明确排除的指标或细节。

只有用户明确要求修订时，review 才可以修改文件；否则只返回问题和建议。

根据 review 修订后，至少执行：

```bash
git diff --check
```

并针对实际来源核对报告：JSON/CSV 用对应字段，日志核对明确 locator，W&B/TensorBoard
使用已授权的只读访问。检查报告文件本身存在且可解析，包括未跟踪的新文件；
`git diff --check` 只是附加格式检查，不是结果正确性证明。若用户要求精简结果，还应
搜索并确认报告不包含未请求的 timing、吞吐、显存、通信量或外部链接。

访问 W&B 等外部 tracker 前必须确认用户已授权且 run 在任务范围内；只读，不创建、修改
或恢复 run，不回显 API key、私有 URL 或其他敏感信息。

## 输出原则

- 遵循用户语言；仓库默认使用中文，除代码参数和指标名外避免堆砌术语。
- 先给核心结果，再给必要设置；每节只保留能影响复现或解释的内容。
- 用表格呈现重复字段，用短结论解释差异。
- 不为了“完整”复制日志、W&B 页面或数据 manifest。
