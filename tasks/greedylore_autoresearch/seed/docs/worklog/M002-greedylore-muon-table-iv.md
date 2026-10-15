# M002：Table IV 60M GreedyLoRE + Muon

## 2026-09-25：通信实现与配置检查

- 目的：在 M001 的相同 Muon、模型、数据、seed 和预算下，仅改变 DDP 梯度通信，比较 rank 32 与 rank 128 的 validation PPL 和通信代价。
- 实验编号：`CM002-m002-greedylore-muon-llama60m-c4-r32-bf16-s1243` 与 `CM003-m002-greedylore-muon-llama60m-c4-r128-bf16-s1243`；均尚未启动正式训练。
- 代码和配置：`comm_hooks/subspace_hook.py`、`comm_hooks/utils.py`、`c4/configs/llama_60m_table_iv.json`、`c4/scripts/run_table_iv_60m_muon.bash`。
- 方法：真实通信路径为 `top_subspace`。压缩对象是 DDP bucket 中符合压缩门槛的参数梯度，发生在 Muon 动量和正交化前，属于近似 Muon；hook 可能同时覆盖 Muon 矩阵和 AdamW fallback 的矩阵梯度。embedding 与 lm_head 按现有规则不压缩。压缩 rank 为 32 或 128，EF14、前 1,000 个通信迭代 dense、投影更新间隔 200。
- 修复：原实现非 SVD 更新步的方向评分函数只有 `pass`。单个二维矩阵在第二步稳定复现 `TypeError: 'NoneType' object is not subscriptable`；现在对每个基向量使用独立高斯探针，平均各 rank 的有符号分数，再平方选方向。`lore` 仅复用固定 SVD 投影，`fake_top_subspace` 做完整梯度 AllReduce，不用于正式 GreedyLoRE 通信实验。
- rank 128 门槛：`hidden_size=256` 时默认最小压缩率 2.0 会跳过临界的 256 行/列矩阵；两组均显式使用 1.0，使实际被压缩的矩阵可进入 hook。运行时仍需报告实际压缩覆盖率。
- 训练设置随 M001 使用每卡 batch 128、GA=1、activation checkpointing、BF16；W&B online、不保存 checkpoint。
- 验证：单矩阵回归测试、双 rank Gloo 与双 rank NCCL/CUDA 小测试通过。正式训练未启动，当前无 PPL 或通信节省结论。
- 下一步：CM001 完成后，同一 `tmux` 脚本将串行启动 CM002、CM003。已用双 rank NCCL/CUDA 验证二维矩阵的投影更新和间隔压缩步一致性；正式结果仍待运行完成后检查。

## 2026-09-25：预算调整与串行任务中断

- CM001 在 5,220 步耗尽当前训练数据，随后因 W&B URL 接口报错退出，CM002/CM003 未启动。
- 用户将四组共同训练量明确为约 1.1B token 位置（含 padding）；脚本改为 8,393 步。数据补足方式待定，确定后再启动。

## 2026-09-25：数据量纠正

- 查明 C4 包装器与 Hugging Face 流式数据集对 DataLoader worker 重复分片，导致前次只使用约四分之一的记录。已修复并以多 worker 单元测试确认无丢样本。现有 30 个分片足够 8,393 步，无需下载或循环数据。

## 2026-09-25：130M 两组串行排队

- 实验编号：`CM006-m002-greedylore-muon-llama130m-c4-r32-bf16-s1243`、`CM007-m002-greedylore-muon-llama130m-c4-r256-bf16-s1243`。
- 与 CM005 完全共用模型 hidden 768、20,000 步、Muon 超参数、C4、seed、BF16、每卡 batch 128、GA=1、W&B online、无 checkpoint；仅改变通信为 `top_subspace`，rank 分别为 32 和 256。EF14、从第 1,000 步压缩、投影更新间隔 200、最小压缩率 1.0。
- 两组在 `greedylore_tableiv_130m_bf16` 内排在 CM005 后；若前序失败，`&&` 阻止后续运行。论文 Table IV 把 130M 的 rank 32 栏写为 `32/512`，与本仓库 130M hidden 768 不一致；此处固定 hidden 768，以保持方法间可比。

## 2026-09-26：Table III GLUE GreedyLoRE + Muon 两秩排队

- 与 M001 GLUE Dense Muon 共用模型、数据、Muon 超参数、batch、seed、训练预算和评估口径，仅改为 `--compressor top_subspace`、`--compress_rank 8/16`、`--use_error_feedback ef14`、`--start_compress_iter 1000`、`--update_proj_gap 200`。rank 8 为 `CM010/013/016/019/022/025/028/031`；rank 16 为 `CM011/014/017/020/023/026/029/032`，任务顺序见 M001。
- 论文附录对 rank 16 的 QNLI 使用 global batch 32；本比较统一 global batch 16，避免 batch 与 rank 同时变化。压缩作用于 Muon 动量和正交化前的 DDP 梯度；hook 可能覆盖 AdamW fallback 矩阵梯度，因此属于近似 Muon。
- 本地数据目录 `/home/wyr/.cache/huggingface/datasets/glue/` 已核对八任务 train/validation Arrow 文件，MNLI 同时具备 matched/mismatched validation。GLUE 入口增加本地 Arrow 读取，避免当前 `load_dataset("nyu-mll/glue")` 离线不命中缓存；未提供额外 test 数据，最终只报告 validation 指标。

## 2026-09-26：Table III rank 8/16 直接短试跑

- 按用户要求，直接在 GPU 0/1/2/7 上对 SST-2 的 `top_subspace` rank 8 和 rank 16 各执行 4 个更新步；短跑临时改为第 1 次通信迭代开始压缩、投影更新间隔 2，以覆盖投影更新和非更新的压缩路径。正式配置仍为第 1000 次开始、间隔 200。
- 两组均完成第 4 步及 validation，日志与结果分别为 `outputs/smoke-direct-tableiii-r8.log`、`outputs/smoke-direct-tableiii-r8/all_results.json` 和 `outputs/smoke-direct-tableiii-r16.log`、`outputs/smoke-direct-tableiii-r16/all_results.json`；未观察到 collective 错误或 NaN。两组均在第 1、3 次迭代记录投影更新。短跑指标不用于方法比较。

## 2026-09-26：130M 秩与压缩起点的独立尝试排队

- 目的：分别检验提高秩与延后开始压缩能否缩小 130M GreedyLoRE + Muon 对 Dense Muon 的 validation PPL 差距；两个实验各只改变一个变量。
- `CM033-m002-greedylore-muon-llama130m-c4-r64-bf16-s1243`：相对 CM006 仅将 `compress_rank` 从 32 提到 64，`start_compress_iter=1000`。
- `CM034-m002-greedylore-muon-llama130m-c4-r32_sci2000-bf16-s1243`：相对 CM006 保持 rank 32，仅将 `start_compress_iter` 从 1000 提到 2000。
- 其余沿用 `c4/scripts/run_table_iv_130m.bash`：hidden 768、20,000 步、4 GPU、global batch 512、每卡 batch 128、GA=1、BF16、Muon matrix/scalar LR 0.02/0.001、EF14、`update_proj_gap=200`、`min_compression_rate=1.0`、seed 1243、W&B online 项目 `GreedyLore-TableIV`，不保存 checkpoint。
- 使用 `c4/scripts/queue_table_iv_130m_followups.bash` 后台轮询：先等原 `greedylore_tableiv_130m_bf16` 会话结束，并核实 CM007、CM008 都有 `update_step=20000` 的结果；之后要求 GPU 3–6 连续三次（间隔 30 秒）利用率不超过 10%、显存占用不超过 1,500 MiB，再串行启动 CM033、CM034。每项启动前重新等待；第一项失败也记录状态并尝试第二项。队列状态在 `outputs/CM033-CM034-greedylore-130m-followups/status.tsv`，各 run 的训练日志在自身输出目录。
- 静态验证：两项 dry-run 均确认仅预期参数和 run 路径变化，W&B 项目、20,000 步、BF16、GA=1、无 checkpoint 设置保持一致；脚本通过 `bash -n`，输出目录尚不存在。当前原 130M 队列仍在跑 CM007，因此新队列不立即占用 GPU。
- 已创建 tmux 会话 `greedylore_tableiv_130m_followups`；`status.tsv` 显示 `waiting`，同时原 `greedylore_tableiv_130m_bf16` 会话仍在运行。当前尚未启动 CM033/CM034 的训练进程。

## 2026-09-26：后续队列未启动的原因与修正

- 原 CM007 在 20,000 步后正常写出 `all_results.json`，日志显示脚本成功结束；原 130M tmux 会话随后消失，但 CM008 没有输出目录或结果。会话为何在 CM008 启动前中断，目前没有直接证据。
- 新队列首次运行于 01:58 在 `status.tsv` 写入 `blocked` 后退出：它原先要求 CM007、CM008 均已有完整结果，因此没有抢占 GPU；CM033/CM034 均未启动。保留该状态记录。
- 修正 `c4/scripts/queue_table_iv_130m_followups.bash`：原会话结束且 CM007 完成后，若 CM008 未完成，则先经 GPU 3–6 连续空闲轮询启动既有 `adam_dense` 配置；仅当 CM008 完成 20,000 步，才继续 CM033、CM034。任何 CM008 失败均记录并阻止后续运行。新队列以追加方式写状态，不覆盖首次阻塞记录。
- 首次重启时 GPU 3–6 显存均仅约 3 MiB，但 10% 利用率阈值受瞬时波动影响，连续三次空闲检查尚未通过；将阈值改为 50%，仍要求显存低于 1,500 MiB、连续三次检查，并重启本队列。此前状态记录保留。
- 用户随后明确要求直接占用当前空闲的 GPU 3–6。确认四张卡显存均约 3 MiB 后，停止本次仍处于等待阶段的队列，并以 `--start-now` 重启同名 tmux 会话；09:06 已启动补跑 CM008。CM033、CM034 保持串行排在 CM008 完成校验后，两个尝试仍分别只改变秩或压缩起点。状态文件保留两次等待及首次阻塞记录。

## 2026-09-26 12:00：Table III GreedyLoRE 改为 W&B online

- 原 Table III CM010 rank 8 SST-2 在约 15,453/42,100 更新步中断，原批次没有 W&B online，不纳入正式比较。rank 16 尚未启动。
- 新的 rank 8/16 运行沿用 CM 编号、目录与 run 名追加 `-rerun1`，从头训练；W&B online 项目按任务命名为 `glue_no_trainer_<task>_greedylore`，不设置 checkpoint。与 Dense Muon 共用 GPU 0/1/2/7 空闲门槛、Muon 参数和训练预算，串行排在同一任务 Dense 后。

## 2026-09-29：完成结果核对

- C4 60M 的 CM002 r32、CM003 r128 均完成修订后的 8,393 步，final validation PPL 分别为 40.5794、37.4963；同预算 Dense Muon CM001 `-rerun1` 为 37.1255。C4 130M 的 CM006 r32、CM007 r256、CM033 r64、CM034 r32 且延后到第 2,000 步压缩，均完成 20,000 步，final PPL 分别为 36.4617、28.8555、33.5800、34.3349；同预算 Dense Muon CM005 为 27.0095。数值来自各 run 的 `all_results.json`，日志均有成功结束标记。
- 后续队列状态文件记录 CM008、CM033、CM034 均以 exit code 0 结束；各自的 `all_results.json` 又确认达到 20,000 步。CM033/CM034 是观察 CM006 后设计的单因素探索，不能把同一 validation 上的改善视作独立验证。
- GLUE rank 8 的 CM010/013/016/019/022/025/028/031 与 rank 16 的 CM011/014/017/020/023/026/029/032 均以 `-rerun1` 完成第 10 个 epoch。最终 validation 指标及源字段见 `docs/results.md`；原无 W&B 的 CM010 中断尝试继续排除在正式比较之外。所有结果只有一个 seed。

## 2026-09-30：130M r64 Muon LR 宽范围筛选

- 目的：在固定 GreedyLoRE r64、EF14 和通信设置时，分别筛选 Muon 矩阵 LR 与 scalar AdamW LR；门槛取既有 CM033 在第 5,000 步的 validation loss `3.743565253792393`，候选必须严格更低才继续到 20,000 步。CM033 与本轮使用的 C4 路径不同，这一门槛仅用于探索性筛选。
- 实验编号：CM036 (0.004/0.001)，CM037 (0.1/0.001)，CM038 (0.02/0.0002)，CM039 (0.02/0.005)；括号内为矩阵 LR/scalar LR。每项从头训练，seed 1243。
- 其余参数沿用 `c4/scripts/run_table_iv_130m.bash r64`：LLaMA 130M、C4、4 GPU、BF16、序列长 256、全局 batch 512、20,000 步 cosine 调度、warmup 2,000、rank 64、EF14、压缩起点 1,000、投影间隔 200、无 checkpoint、W&B online。数据路径固定为 `c4/c4_en`，GPU 4–7。
- 队列 `c4/scripts/sweep_table_iv_130m_lr.py` 读取每个 run 的 `train.log` 中 `Eval loss at step 5000`；候选 loss 等于或高于 CM033 时终止该候选进程组，低于则在原 20,000 步 LR 调度下继续同一次运行。状态记录于 `outputs/CM036-CM039-r64-lr-sweep/status.tsv`。不修改训练入口或优化器。
- 启动前核对：50 个训练分片和 8 个验证分片可由本地加载器读取；候选 dry-run 均保持 r64、EF14、20,000 步 cosine、warmup 2,000 等既有参数；相关单元测试、脚本语法与 `git diff --check` 通过。结果和运行状态以输出目录中的日志、`all_results.json` 和队列状态为准。
- 已启动 tmux 会话 `greedylore_130m_r64_lr_sweep`；队列状态记录旧 CM033 为 `reference`，当前 GPU 4–7 被其他任务占用，队列在等待连续空闲检查后启动 CM036。尚无本轮候选结果。
- 按用户后续要求，将正式队列的资源门槛改为：检查全部 GPU，每 5 分钟取样一次，连续两次取样中同样至少 4 张卡均满足显存 ≤1,500 MiB、利用率 ≤10% 才启动，实际选用的 4 张卡写入状态文件。等待控制器已在无候选训练时重启；CM036–CM039 配置和第 5,000 步 loss gate 不变。10 项相关单元测试通过；启动检查确认新控制器仍在等待，当前全部 GPU 被占用。

## 2026-10-01：终止 r64 LR sweep，改为 Dense 选 LR 后比较 r32/r256

- 用户观察降低 matrix LR 后的趋势，要求终止原 r64 sweep。CM036 在约第 12,000 步停止，未完成 20,000 步；CM037–CM039 未启动。CM036 的第 5,000 步与第 12,000 步 validation loss 分别为 3.48792、3.23633，仅为中断运行的观察值；保留原日志/W&B，不作为最终结果。状态见 `outputs/CM036-CM039-r64-lr-sweep/status.tsv`。
- 新实验 `CM042` r32 和 `CM043` r256 在 M001 的 CM040/CM041 完成后，使用最终 validation loss 更低的矩阵 LR。其余沿用 `c4/scripts/run_table_iv_130m.bash`、scalar LR 0.001、EF14、压缩从第 1,000 次通信迭代开始、投影间隔 200、20,000 步、C4 `c4/c4_en`、seed 1243、W&B online、无 checkpoint。完整 run ID 将含选定 LR。
- 串行队列脚本 `c4/scripts/queue_table_iv_130m_muon_lr.py`；每组启动前等待连续两次 5 分钟间隔的任意同一组 4 张 GPU 空闲。首组 Dense Muon 按用户单独授权直接用 GPU 0–3。队列状态在 `outputs/CM040-CM043-muon-matrix-lr/status.tsv`。

## 2026-10-02：r64 中断状态及 r32/r256 完成结果核对

- 目的与配置：记录最近学习率探索结果。CM036 沿用 r64 sweep 配置；CM042/CM043 使用 Dense 两候选选出的矩阵 LR 0.01、scalar LR 0.001，其他配置沿用上一条记录。实际 GPU 均为 0–3 的 4 张 RTX 4090。
- 中断尝试：CM036 的日志 `Eval loss at step 5000` / `12000` 分别为 3.48792 / 3.23633；虽通过旧 CM033 的 5,000 步筛选门槛，但用户改换实验后终止。没有最终结果文件或成功结束标记；CM037–CM039 取消且未启动。状态见 `outputs/CM036-CM039-r64-lr-sweep/status.tsv`，指标见 `outputs/CM036-m002-greedylore-muon-llama130m-c4-r64-lr0p004-slr0p001-bf16-s1243/train.log`。这两项是中间 validation loss，不作为 final。
- 完成结果：`CM042-m002-greedylore-muon-llama130m-c4-r32-lr0p01-bf16-s1243`、`CM043-m002-greedylore-muon-llama130m-c4-r256-lr0p01-bf16-s1243` 均完成 20,000 步；final validation loss/PPL 分别为 3.20127 / 24.5636、3.13895 / 23.0797。
- 验证：核对 `outputs/<上述完整 run ID>/all_results.json` 的 `update_step`、`final_eval_loss`、`final_eval_ppl`，日志的最终 loss 和成功结束标记，以及 `outputs/CM040-CM043-muon-matrix-lr/status.tsv` 的 completed/queue finished。运行配置确认同为 `c4/c4_en`、rank 32/256、EF14 与所选 LR。
- 结论：同设置 Dense CM040 的 PPL 为 22.9218；r32 高 1.6418（7.16%），r256 高 0.1579（0.69%）。r256 在本轮更接近 Dense。该比较只有 seed 1243 一次，并在同一 validation 上选择学习率，属于初步探索；不能将跨数据路径的旧 CM006/CM007 差异单独归因于 LR，也未核算总通信收益。
- 下一步：固定所选配置，用预先规定的多 seed 确认；完整比较摘要见 `docs/results.md`。


## 2026-10-02：CIFAR Figure 3 协议下的 Muon 比较

- 研究约定：探索性扩展实验，问题是相同 Muon 配置下 GreedyLoRE 的最终 test accuracy 与 Dense 相差多少、梯度通信减少多少；不是原论文 AdamW 的数值复现。主指标为第 40 epoch 的 test accuracy 和压缩组减 Dense 的百分点差，best 仅作辅助，不据此选择学习率。单 seed 1243，无调参；没有预先指定“相当”的容差，不将单次结果解释为统计等效。
- 用户已确认启动，并指定 matrix LR **0.02**、W&B **online**、CIFAR-10/100 分别用不同 project，**不保存 checkpoint**。其余沿用已确认方案。
- 论文协议：ResNet-18、CIFAR-10/100 官方 50,000 train / 10,000 test、40 epochs、4 张 RTX 4090、每卡 batch 32、全局 batch 128、cosine。GreedyLoRE rank 64，前 500/4000 次更新 dense，子空间间隔 750/1200。参照 https://arxiv.org/pdf/2507.08784 正文 VII-A、Figure 3 和附录 F。
- 实际适配：矩阵 Muon LR 0.02、momentum 0.95、Nesterov、5 步 Polar Express、spectral_norm scaling；scalar/head AdamW LR 分别 0.005/0.0005，betas (0.9,0.999)、eps 1e-8；weight decay 5e-4，bias/BatchNorm 为 0。FP32 训练，Muon 正交化内部 BF16。cosine 按 epoch 更新至 0，无 LR warmup。上述 Muon/精度/seed 设置不是论文值。
- 参照仓库最初提交 `92c5edb` 的 `pytorch-cifar/main.py`：沿用 CIFAR 版 ResNet、RandomCrop(32,padding=4)、水平翻转、normalization mean=(.4914,.4822,.4465)、std=(.2023,.1994,.2010)、test batch 100、weight decay 5e-4。原入口是 200 epochs + MultiStepLR，按用户批准的论文协议改用 40 epochs + cosine。CIFAR-100 沿用同一 normalization，记录为已批准的实现选择。
- 数据路径 `/home/wyr/ARC-TopK-release/data`：复用已存在且校验过的 CIFAR-10，CIFAR-100 从 torchvision 官方数据 URL 下载并校验为 50,000/10,000。每 epoch 391 updates、最后一批每卡 20，全程 15,640 updates；评估分片不补重复样本，完整 10,000 张。
- 新入口 `pytorch-cifar/train_ddp.py` 保留原 `main.py`，使用真实 DDP 和现有 Muon/GreedyLoRE 实现；复现脚本 `run_paper_muon.bash`、自动串行队列 `queue_paper_muon.bash`。不修改虚拟环境或关键依赖。启动时保存完整命令、git HEAD、关键代码 SHA256、config.json 和 tracker.json；没有 checkpoint，失败只能另建身份从头跑。
- 验证：15 项 CIFAR/hook/Muon 单元测试通过；4 卡 Dense 3-step 与 GreedyLoRE 6-step synthetic smoke 完成，后者 start=2/gap=2 覆盖首次 SVD、低秩传输、再次刷新，并检查各 rank 参数一致。smoke 仅测试实现，指标不作科学结果；产物 `outputs/cifar-smoke-dense/`、`outputs/cifar-smoke-greedylore/`。4 个正式命令 dry-run、bash 语法和 diff whitespace 检查通过。
- 资源编排：两个数据集同时占用不同的 4 卡，各自先 Dense 后 GreedyLoRE；前一组必须成功完成 40 epochs、15,640 updates、10,000 test 样本，才能启动后一组。队列失败即停止。共享主机的并行数据集任务可能影响时间测量；耗时视作诊断而非隔离 benchmark。
- 输出：每 run 的 `outputs/<run ID>/train.log`、`config.json`、`metrics.jsonl`、`all_results.json`、`tracker.json`；队列状态在 `outputs/CM044-CM047-cifar/status-cifar10.tsv` / `status-cifar100.tsv`。主结果尚未产生。

- 本方法臂：`CM045-m002-greedylore-muon-resnet18-cifar10-r64-lr0p02-fp32-s1243` 与 `CM047-m002-greedylore-muon-resnet18-cifar100-r64-lr0p02-fp32-s1243`；对应 project/GPU 与各自 Dense 相同，分别待 CM044/CM046 完成后自动启动。
- 压缩语义：`top_subspace` / EF14，梯度在 Muon momentum 和正交化前压缩；卷积展平为二维。分类头 `linear` 与一维参数 dense，收益门槛 `min_compression_rate=2`。`beta_ef=0`、`error_inherit=0`，子空间刷新时按既有实现清零 EF 并做 dense 聚合。第 501/4001 次更新为首次 dense SVD 刷新，第一步低秩传输是第 502/4002 次更新。因此是有损梯度下的近似 Muon，不宣称与 Dense 等价。
- 通信口径：DDP bits 是估算的每 rank collective 输入，计入 warmup、refresh 和每步 FP32 score probes，排除 network replication/DDP buffer broadcast；Muon collective 字段沿用优化器的单独计数（AllGather 的 world_size*(world_size-1)*input_bits 估计）。两者不是同一统计范围，不直接相加。

- 启动核对：两个 tmux 队列的 Dense 已完成首次更新并连接 W&B online；CM045/CM047 暂未启动，将在配对 Dense 通过完整预算门槛后自动从头开始。队列 session 名为 `greedylore_cifar10_muon` / `greedylore_cifar100_muon`。

- 额外验证：3 项既有训练入口 CLI 检查通过（运行时需要 `PYTHONPATH=.`，首次未设置而 C4 import 失败；未修改 C4）。共 18 项相关单元/入口检查通过。独立代码审查未发现训练/评估阻塞；按反馈明确记录 FP32 模型/梯度与 BF16 正交化，显式指定两组 GPU，保护既有队列状态不被覆盖并补充队列脚本 SHA256。
- 运行维护：普通长任务等待交给 tmux 和 shell 队列，停止 Agent 轮询。完成时每 run 自动写 final/best test 指标、通信口径、all_results.json 与 TRAINING_COMPLETED；失败保留日志并停下对应队列。以后整理结果须先核对完整预算和成功结束标记。


## 2026-10-02：CIFAR GreedyLoRE 完成结果核对

- CM045 / CIFAR-10 与 CM047 / CIFAR-100 均完成 40 epochs、15,640 updates，完整评估 10,000 张 test；final test accuracy / loss 分别为 94.29% / 0.31014942813664675 和 74.94% / 1.272592346072197。采用已批准的 matrix LR 0.02、rank 64、EF14，以及各自的压缩预热和子空间刷新间隔；与 Dense 保持同数据、seed、Muon 和训练预算。
- 来源：上述完整 run ID 对应 `outputs/<run ID>/all_results.json` 的 `final_test_accuracy`、`final_test_loss`；已核对 `status=completed`、`epoch=40`、`update_step=15640`、`test_samples=10000`，与 `metrics.jsonl` 最后一行及日志 `TRAINING_COMPLETED` 一致。两个队列状态均为 `queue_finished`，W&B 日志确认同步完成。
- 相对配对 Dense CM044/CM046，最终准确率低 0.14/0.97 个百分点；test loss 分别低约 0.00733 / 高约 0.00026。CIFAR-10 更接近 Dense，但每组只有 seed 1243 一次，不能声称统计等效。摘要见 `docs/results.md` 的 CIFAR 节；后续如需确认应做固定配置的多 seed 实验。

## 2026-10-02：Table V GreedyLore + Muon 测速队列

- 用户授权整张Table V改用Muon测速；M002计划臂为CM050 / CM053 / CM056 / CM059，对应60M / 130M / 350M / 1B。配置、论文来源、架构冲突及本地选择见 `docs/table-v-muon-protocol.md`；不能称作完全一致的论文架构复现。
- 统一rank32、`top_subspace`、EF14、从hook iter1000开始压缩、投影间隔200、min_compression_rate1、beta_ef0、error_inherit0。窗口更新1001–1500包含1001/1201/1401的投影刷新与首次初始化，不删慢步。压缩Muon正交化前的梯度，是近似Muon；embedding/head沿用hook排除规则。
- 其他设置与同模型Dense臂完全相同，包括Mu​​on分组、LR0.01/0.001、4GPU、真实C4、batch128（1B为64）、GA1、BF16、checkpointing、sequence256、seed1243。端到端时间包含Muon内部collective，不把hook通信量当总通信量。
- 验证：四卡130M batch128短smoke完成，NCCL确认SHM；相关19项单元测试通过。正式臂尚未产生完成结果，已排入 `greedylore_table_v_muon` 串行队列，产物位于 `outputs/CM048-CM059-table-v-muon/`。每臂实际开始/结束以status.tsv为准。

## 2026-10-02：暂停 Table V Muon 队列

- 用户要求暂停当前队列并改测 350M AdamW Dense/GreedyLore。350M GreedyLore Muon 的 CM056 已完成并保留；1B GreedyLore Muon CM059 未启动。

## 2026-10-03：GreedyLoRE timing setting matrix

- 用户授权比较9组系统设置，每组完整运行60M/130M/350M/1B的Dense/GreedyLoRE配对；GreedyLoRE臂为CM062–CM133中的奇数编号。完整配置、计时窗口和失败处理见 `docs/table-v-muon-setting-matrix.md`。
- GreedyLoRE保持现有`top_subspace`实现、rank32、EF14、压缩起点100、投影间隔200、min compression rate 1；不优化或修改hook。1024 MB DDP bucket仅通过独立timing入口的DDP构造参数设置。
- 预热101步包含首次投影，测量更新102–501的连续400步并包含两次周期投影。配置覆盖4/8 GPU、BF16/FP32、batch1/32/论文大batch、默认/单channel及Socket上界诊断；OOM保留并继续。
- 8卡60M FP32/batch1/单channel/1024 MB bucket短smoke中，临时将压缩起点设为0会在第二步报basis与gradient批维不一致；原因是首次投影发生在DDP首轮bucket重建之前。保持hook不变，将诊断压缩起点改为1后，第1步Dense、第2步投影、第3步压缩均完成。正式矩阵的压缩起点100会先稳定bucket，因此不采用或外推失败smoke的时序。
## 2026-10-03：严格阻塞通信测速

- 新增 CM134–CM165 矩阵中的 GreedyLoRE + Muon 奇数臂；每个配置只运行一次，配对 Dense 及完整四组设置见 `docs/table-v-muon-strict-blocking.md`。
- 不修改 `comm_hooks/subspace_hook.py`。独立 timing 包装器在 hook 入口同步、等待原 `top_subspace` Future 完成、再次同步，因而计入压缩、投影选择、collective 与解压，作为通信路径总开销而非纯 NCCL 时间。
- GreedyLoRE 固定 rank32、EF14、压缩起点100、投影间隔200；预热101步后测量400步。FP32、无 checkpointing、8192 MiB bucket cap、默认 channel、SHM 和 Muon 参数均与配对 Dense 相同。
- 8 卡 60M FP32/batch1 三步 smoke 覆盖首次投影及普通压缩步并完成；测量步记录 1 个 242,729,984-byte bucket，`mean_blocking_hook_seconds` 成功写入。正式结果待 `outputs/CM134-CM165-table-v-muon-strict-blocking/` 队列完成后补记。

### BF16 论文设置追加组

- 增加 CM167/169/171/173 GreedyLoRE + Muon，与追加 Dense 使用4卡、BF16、activation checkpointing和论文每卡batch；rank32、EF14、严格阻塞及其余系统配置保持配对一致，每臂一次。
- 追加组排在首批完整队列之后，产物写入 `outputs/CM166-CM173-table-v-muon-strict-blocking-bf16-paper/`。

### 系统变量精简矩阵

- 用户停止 BF16 论文追加组：GreedyLoRE CM167、CM169 完成，CM171/CM173 未启动；中断的 Dense CM170 不纳入比较。
- 仅350M的CM174–CM187队列在首个Dense臂未完成时停止，GreedyLoRE臂均未启动；不纳入正式比较。
- 正式CM188–CM243对每个setting运行60M/130M/350M/1B，统一batch1、rank32、EF14、严格阻塞和单一大bucket；排除原设计第3、6组，即两个FP32 Socket配置。
- 共7个setting、4模型、56臂，每臂一次；逐模型配对并交替方法运行顺序。Socket仅保留8卡BF16通信受限诊断。
- CM188–CM243已全部尝试：GreedyLoRE在60M/130M/350M的21个setting-model组合全部完成，三个8卡BF16 1B与全部FP32 1B均OOM；失败日志均明确记录CUDA OOM。
- 8卡FP32 SHM的130M/350M在默认及单channel下出现小幅严格阻塞加速；8卡BF16 Socket的130M/350M iter speedup为1.3871×/1.6343×、hook speedup为1.7670×/2.5906×。其余大部分配置慢于Dense；结果为n=1的通信受限诊断，完整证据见`docs/results.md`。

### 正常异步 DDP bucket sweep

- 新增GreedyLoRE CM245/CM247，与配对Dense共同使用350M、8卡、FP32、batch1、默认channel/SHM；bucket cap分别为256/1024 MiB，rank32、EF14、压缩起点100和投影间隔200不变。
- 不启用严格阻塞。被动包装器只记录每步实际bucket布局并返回原`top_subspace` Future，用端到端iteration time比较不同bucket下的正常overlap收益；每臂一次。
- CM245/CM247均完成，iteration time为0.314311/0.315635秒。配对Dense后，256 MiB下speedup为0.9589×（慢4.29%），1024 MiB下为1.0270×（快2.63%）；GreedyLoRE从1024降到256 MiB仅快约0.42%，未获得Dense同等的overlap收益。


## 2026-10-03：350M Table IV r32/r256正式比较计划

- 用户指定4卡Dense/r32并行60000步，之后4卡跑r256；共同matrix LR由先行Dense0.01/0.005、各10000步并行sweep决定，按端点validation loss选择，正式训练从头开始。M002身份CM251(r32)/CM252(r256)，配对Dense CM250。
- 全部使用96 train/8 validation固定C4分片、hidden1024仓库模型、BF16、activation checkpointing、seed1243、global batch512、每卡128、GA1、60000 schedule、6000 warmup、EF14、start1000、projection gap200。压缩梯度后执行Muon，属于有损通信适配。
- 详细合同、边界和失败规则见`docs/table-iv-350m-muon-protocol.md`；入口`c4/scripts/queue_table_iv_350m_muon.py`。数据准备和三臂smoke由队列先行；11项相关回归通过，正式结果待产生。不用并行吞吐估计方法加速，不将评估点当独立重复。

- 首次启动的编号检查误将控制器自身日志当作CM248已存在，在数据/训练前退出。已缩小检查到实验身份前缀并补充回归测试；失败日志保留为`outputs/CM248-CM252-table-iv-350m-muon.start-failed.log`，不计作训练run。

- 用户因流量不足明确禁止下载，已停止旧tmux队列。50个已有分片已校验；没有新增完整文件，残留一个201124976字节的部分下载。改为只用原`c4/c4_en`，离线模式，显式重复训练数据并丢弃尾部不足128的batch，保持60000步/global batch512。旧队列及日志保留；新控制目录`outputs/CM248-CM252-table-iv-350m-muon-existing-data/`。

- Dense四卡batch128短验证完成（5次更新，测量2次）；下一臂启动前GPU利用率检查立即失败。随后核实8卡显存/利用率均0且没有计算进程，符合短验证结束后的利用率读数残留。修正为控制器最多等待60秒再检查，不停止其他任务；加入回归测试并通过11项单元检查。增加仅前置验证阶段的`--resume`，保留原manifest、状态及日志，跳过有成功结果的Dense smoke，恢复剩余r32/r256验证。正式sweep尚未启动。

## 2026-10-06：CM251 GreedyLoRE + Muon r32 350M 完成结果核对

- 目的与配置：与 Dense CM250 比较相同模型、数据、Muon 和训练预算下 rank 32 梯度压缩的 validation 表现。CM251 从头训练 60,000 步，矩阵 LR 0.005 来自 CM248/CM249 的 10,000 步 Dense sweep；使用 `top_subspace`、EF14、压缩起点 1,000、投影间隔 200。已有 50 个 C4 train 分片被重复使用；完整设置见 `docs/table-iv-350m-muon-protocol.md` 和 run 的 `command.txt`。
- 验证与结果：`outputs/CM251-m002-greedylore-muon-llama350m-c4-r32-formal-lr0p005-bf16-s1243/all_results.json` 记录 `update_step=60000`，final validation loss/PPL 为 2.91212630/18.39587206，best loss/PPL 为 2.91208630/18.39513630（第 59,000 步）；final 评估覆盖 10,048,075 个有效 token。同目录 `train.log` 有 `Script finished successfully`，队列 `outputs/CM248-CM252-table-iv-350m-muon-existing-data/status.tsv` 记为 `completed`。
- 观察与下一步：相对 CM250，final validation loss 高 0.05716，PPL 高 1.02198（5.88%）；完整比较见 `docs/results.md`。两组均只有 seed 1243 一次训练，且 LR 选择和正式比较使用同一 validation；需固定配置并独立重复后再判断稳定性。

## 2026-10-06：CM252 GreedyLoRE + Muon r256 350M 完成结果核对

- 目的与配置：在 CM250/CM251 完成后，从头训练 rank 256，与相同模型、数据、Muon 配置和 60,000 步预算的 Dense CM250、r32 CM251 比较。CM252 使用由 CM248/CM249 选择的矩阵 LR 0.005、`top_subspace`、EF14、压缩起点 1,000、投影间隔 200；已有 50 个 C4 train 分片被重复使用。完整设置见 `docs/table-iv-350m-muon-protocol.md` 和 run 的 `command.txt`。
- 验证与结果：`outputs/CM252-m002-greedylore-muon-llama350m-c4-r256-formal-lr0p005-bf16-s1243/all_results.json` 记录 `update_step=60000`，final validation loss/PPL 为 2.86140490/17.48607581，best loss/PPL 为 2.86136158/17.48531838（第 59,000 步）；final 评估覆盖 10,048,075 个有效 token。同目录 `train.log` 有 `Script finished successfully`，队列 `outputs/CM248-CM252-table-iv-350m-muon-existing-data/status.tsv` 记为 `completed`。
- 观察与下一步：相对 CM250，final validation loss 高 0.00644，PPL 高 0.11218（0.65%）；CM252 的 final PPL 低于 CM251。三组均只有 seed 1243 一次训练，且 LR 选择和正式比较使用同一 validation；完整比较见 `docs/results.md`，需固定配置并独立重复后再判断稳定性。
