---
name: greedy-lore-experiment-workflow
description: Use when planning, configuring, running, resuming, recording, or comparing GreedyLore and Muon communication experiments in the greedy_lore repository.
---

# GreedyLore 研究实验工作流

本仓库已有 GreedyLore 等梯度压缩通信 hook，并已接入 Muon。核心问题是：**GreedyLore 在 Muon 优化器下是否仍然有效，与相同 Muon 设置的 Dense 基线相比，模型指标相差多少，节省了多少通信？**主要对照是 Dense Muon 与 GreedyLore + Muon。实验尽可能沿用 GreedyLore 论文的任务、模型、数据、训练和压缩设置；仅对无法直接沿用的 Muon 特有超参数（尤其是矩阵与 scalar 参数的学习率）进行有记录的适配。实验细节以当次研究问题、训练入口和运行产物为准。

## 论文与适用范围

- [Greedy Low-Rank Gradient Compression for Distributed Learning with Convergence Guarantees](https://arxiv.org/pdf/2507.08784)（GreedyLore，arXiv:2507.08784）是方法与实验协议参考。按目标任务核对正文及附录中的配置；论文的 Adam 学习率不能直接当作 Muon 学习率。
- 论文的收敛分析针对 MSGD、Adam 等优化器；不能把其中的保证直接归于 Muon。将论文值、仓库默认值、计划覆盖值和实际运行值分开记录。
- GreedyLore 相关子空间压缩由 `comm_hooks/subspace_hook.py` 实现，经 `comm_hooks/utils.py` 中的 `top_subspace`、`top_rc` 等选项注册。`lore` 对应 `comm_hooks/lore_hook.py`，是另一条压缩路径。确认实际选项和 hook 后再给方法命名。

## 选择工作模式

- **设计方案**：明确问题、对照组、预算、指标和选择规则；设计请求本身不等于启动训练。
- **配置或实现**：检查对应训练入口、脚本和 hook，保留可切换的现有基线。
- **启动或续跑**：核对运行配置与输出目录后再执行。只有用户授权启动时才启动；续跑须确认当前入口真的保存且能恢复所需模型、优化器、调度器和训练进度状态。否则记为新 run。
- **结果审查或整理**：使用[实验结果汇报技能](../experiment-results-reporting/SKILL.md)，本技能只补充仓库的研究上下文。

## 建立实验约定

正式实验前阅读 `AGENTS.md`、`readme.md`、对应训练入口和现有运行脚本；已有同方法的 `docs/worklog/` 或 `docs/results.md` 时一并核对。根目录 `.slurm` 示例含旧集群路径，不可直接视为当前环境配置。

1. 主要比较 `--optimizer muon --compressor none`（Dense Muon）与相同 Muon 配置下的 `top_subspace` 或论文对应的 GreedyLore 子空间变体（GreedyLore + Muon）。报告相同训练预算和评价口径下的指标差值，并结合通信代价判断有效性。研究问题需要时再加入 `lore`、`powersgd`、`topk_sync`、`randk_sync` 或 AdamW 对照。
2. 以论文中的数据集、模型、训练预算、batch、scheduler、warmup、评价 split/指标及 GreedyLore 的压缩秩、投影更新频率、error feedback、压缩起点等设置为默认参照。这里的“尽可能沿用”是指：除 Muon 特有参数及实现或资源限制确实要求的调整外，不随意改动论文中的其他超参数；无法沿用时逐项说明原因与实际值，不从其他表格推断缺失设置。
3. Muon 特有参数另行设置或调优，尤其是矩阵参数 LR、AdamW fallback 的 scalar LR，以及必要的动量、正交化和 LR 缩放选项。预先记录候选范围与选择规则；主要对照的 Dense Muon 与 GreedyLore + Muon 使用相同的 Muon 参数分组和超参数，并保持 seed、batch、预算、scheduler 与评价方式一致。若为某一方法臂单独调参，将它标为另一项实验，不能并入只改变压缩方式的主比较。
4. Muon 正交化是非线性的。说明压缩作用于梯度、动量、正交化输入、正交化输出还是参数更新，并说明传输的张量。正交化前的有损压缩通常产生近似 Muon，不声称与无压缩 Muon 通信等价。论文对 Adam/MSGD 的结论也不直接验证 Muon。
5. 检查 hook 覆盖范围：DDP bucket 可能同时包含 Muon 矩阵参数和 AdamW fallback 参数的梯度。若没有选择性压缩，不写成“只压缩 Muon 参数”。`pytorch-cifar/main.py` 当前只有 SGD/Muon 优化器选项，没有 DDP 压缩 hook；不能将它当作已支持 GreedyLore + Muon 的入口。
6. 方法进入正式代码、配置或比较实验时，按 `AGENTS.md` 使用 `M###` 方法编号和 `CM###-...` 实验编号。各方法臂/run 保留唯一身份；输出目录、日志及可配置的 tracker 名称应能对应。入口无法控制名称时，在 worklog 中记录 run ID 与产物映射。
7. 单 seed 结果只作初步结果。若用 validation 挑选学习率或压缩参数，记录候选范围和选择规则，不把同一 validation 结果当独立确认。

## 配置与启动

启动前核对模型与数据路径、任务、seed、优化器、Muon 参数分组与学习率、scheduler、训练预算、batch、压缩参数、评估方式、输出目录及 tracking 模式。确认参数确实由目标入口支持；检查 GPU 占用、进程和磁盘。先做与改动规模相称的静态检查或 dry-run；有必要且已授权时做短 smoke test。不得干扰无关进程，也不要在 dry-run 中下载数据或启动训练。

长任务按环境使用 `tmux` 或作业调度系统并保留日志。使用 W&B 时，将 run 与本地实验编号关联；入口不能直接设置名称时记录 tracker ID。连接外部 tracker、下载大型资源和启动训练均应在用户授权范围内。

每次实际尝试后，按时间追加到 `docs/worklog/<M###-...>.md`；若目录尚不存在，首次需要时创建。记录目的、实际命令与配置、验证、结果、异常及产物路径；同一方法的重试仍记在同一文件。

## 结果与通信口径

`docs/results.md` 如需跨 run 汇总，可在结果整理任务中创建；原始日志、checkpoint 和机器可读结果仍保存在运行输出目录。报告时按[实验结果汇报技能](../experiment-results-reporting/SKILL.md)核实完成状态、实际预算、split、final/best 规则和每个数字的来源。来源冲突须标明，不能把最后一条日志自动当最终结果。

通信代价区分 DDP/压缩 hook 的梯度通信与 Muon 优化器内部 collective（如梯度布局检查和正交化结果 AllGather）。只有计数范围一致时才合计；注明估算或实测，缺失的部分标为未统计。

## 仓库路径速查

- GLUE：`glue/run_glue_no_trainer_HF.py`；运行示例 `run_glue.slurm`。
- CIFAR-10：`pytorch-cifar/main.py`、`pytorch-cifar/README.md`；目前无 DDP 压缩入口。
- C4/LLaMA：`c4/run_llama_pretraining.py`、`c4/configs/`、`c4/scripts/`。
- 通信注册与方法：`comm_hooks/utils.py`、`comm_hooks/subspace_hook.py`、`comm_hooks/lore_hook.py`、`comm_hooks/default_hooks.py`。
- Muon 优化器与参数分组：`optimizer/muon.py`、`optimizer/muon_utils.py`。
