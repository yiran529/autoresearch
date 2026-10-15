# 简洁实验结果模板

```markdown
# <任务/数据集>实验结果

## 1. 实验矩阵

| 实验 | 模型 | 优化器/方法 | 训练预算 |
| --- | --- | --- | --- |
| <ID> | <model> | <method> | <steps/epochs/tokens> |

## 2. 核心设置

- 数据集：<dataset>
- 关键训练设置：<only requested or result-critical values>
- 优化器/方法超参数：<only requested or result-critical values>
- 硬件：<GPU count and model>
- 运行脚本：`<script path>`

## 3. 结果

| 实验 | 指标 | 数值 | split | final/best | step/checkpoint |
| --- | --- | ---: | --- | --- | --- |
| <ID> | <metric> | <value> | <validation/test> | <final/best> | <step/checkpoint> |

## 4. 结论

- <same-condition comparison>
- <main observation>
- <replication and configuration limitation>
```

若用户指定其他指标或结构，以用户要求覆盖模板。C4 可将 loss/PPL 作为指标示例；
GLUE 和 CIFAR-10 应使用各自入口定义的指标。不要自动添加 W&B 链接、吞吐、
timing、显存或通信量。
