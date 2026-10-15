# M001：Table IV 60M Dense Muon

## 2026-09-25：配置与入口检查

- 目的：建立与 GreedyLoRE + Muon 仅压缩方式不同的 Dense Muon 对照。
- 实验编号：`CM001-m001-dense-muon-llama60m-c4-dense-bf16-s1243`；BF16 正式训练尚未启动。
- 代码和配置：`c4/run_llama_pretraining.py`、`c4/configs/llama_60m_table_iv.json`、`c4/scripts/run_table_iv_60m_muon.bash`。
- 数据：`/dev/shm/wyr_tmp/c4` 下的本地英文 C4 train/validation shards；已验证能读取一条样本。输出路径为 `outputs/<实验编号>/`。
- 论文参照：Table IV 的 60M 列为 `dmodel=256`、约 1.1B 有效训练 token；附录 F 为 10,000 更新步、warmup 1,000 步、global batch 512、序列长 256、cosine 到峰值学习率的 10%、weight decay 0、梯度裁剪 1.0。论文没有给出完整 60M 架构或 Muon 学习率。
- 模型架构推定：hidden 256、8 heads、56 layers、FFN 688，实际 60,682,496 参数；不是已核实的作者原始配置。
- Muon 起始配置：matrix LR 0.02、momentum 0.95、spectral-norm LR scaling；AdamW fallback LR 0.001、betas (0.9, 0.999)。采用 4 GPU、每卡 microbatch 128、累积 1、activation checkpointing、BF16、seed 1243。Muon 参数不来自论文，和 M002 保持一致。
- 用户指定 W&B online，且不保存 checkpoint。脚本的 `save_every=20000` 大于 10,000 步预算；`save_dir` 沿用入口默认值但不会写入。单卡合成数据 batch 128 的完整前向、反向、Muon 更新峰值显存为 14.12 GiB。
- 验证：`PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v` 16 项通过；运行脚本 `--dry-run` 可生成完整命令。正式指标为最低及最终 validation PPL。
- 下一步：跟踪正式训练结果；本入口不保存可续跑的完整 checkpoint，异常退出需视为新 run。

## 2026-09-25：FP32 启动失败与 BF16 更正

- 首次启动误用 FP32，目录 `outputs/CM001-m001-dense-muon-llama60m-c4-dense-s1243/`；在第 0 步之前因当前 Transformers 中 `model.generation_config is None` 而退出，没有训练指标，CM002/CM003 未启动。
- 按用户更正，将三组改为 BF16；入口改为设置模型 `config.pad_token_id`，仅当 generation config 存在时再同步它。失败日志保留，不并入 BF16 结果。

## 2026-09-25：BF16 正式训练启动

- 启动命令：`tmux new-session -d -s greedylore_tableiv_60m_bf16 'cd /home/wyr/greedy_lore && bash c4/scripts/run_table_iv_60m_muon.bash dense && bash c4/scripts/run_table_iv_60m_muon.bash r32 && bash c4/scripts/run_table_iv_60m_muon.bash r128'`。
- 运行环境：GPU 3、4、5、6；W&B online；无 checkpoint。CM001 的 W&B ID 是 `b35vplch`，本地日志在 `outputs/CM001-m001-dense-muon-llama60m-c4-dense-bf16-s1243/train.log`。
- 启动检查：四个 rank 已完成进程组初始化；训练完成第 1 次更新并进入首轮 validation。当前无可报告的最终结果。

## 2026-09-25：数据耗尽与退出故障

- CM001 在本地 30 个 C4 train 分片耗尽时只达到 5,220 更新步，未达到原定 10,000 步；这次运行是部分预算，不能当作 Table IV 的完整结果。该 run 的 W&B ID 为 `b35vplch`，最终 validation PPL 约 42.57，仅作故障排查记录。
- 入口随后调用当前 W&B 0.30.0 已不存在的 `run.get_url()`，报 `AttributeError`；串行任务因此没有启动 CM002/CM003。因为无 checkpoint，这次运行不能续跑。
- 按用户明确的 **1.1B token 位置（含 padding）** 预算，512×256×8,393 = 1,100,087,296，将四组计划统一改为 8,393 更新步。这个预算口径与之前按附录 F 设置的 10,000 步不同。原 30 个训练分片仍不够，新增数据或复用现有数据的选择待定。
- 修复训练入口：未达到目标步数即数据耗尽时抛错，防止写出貌似完成的结果；正常结束时使用 W&B `run.url`。19 项单元测试通过，脚本 dry run 通过。待数据方案确定后，CM001 从头启动，使用唯一的重试 run ID。

## 2026-09-25：纠正数据量判断与 DataLoader 双重切分

- 上一节“30 个 C4 分片不够”的判断已被推翻。30 个分片实际有 10,689,518 条记录，每个约 356,317 条。训练的 `PreprocessedIterableDataset` 对 DataLoader worker 再次使用 `islice`，但 Hugging Face 流式数据集已自行按 worker 分片；这使每个 worker 只读取被分配记录的约四分之一。16 条记录的复现测试在修复前只返回 4 条，修复后返回全部 16 条。
- 首个分片按每 356 条抽取一条、共 1,001 条，以训练所用 T5 tokenizer 估计：每分片约 1.91 亿原始 token；截断至 256 后约 6,969 万非 padding token；全部填充时约 9,122 万 token 位置。30 个分片的名义总容量约 27.37 亿 token 位置，超过 8,393 步的 11.00 亿预算，无需补充数据。抽样估计不是精确逐条 token 统计。
- 删除包装器内的第二次 worker 切分；20 项单元测试通过，`git diff --check` 通过。此前 CM001 仅 5,220 步的根因是代码丢样本，不是原始分片不足。
- 已重新启动四组串行 tmux 会话 `greedylore_tableiv_60m_bf16`，顺序为 Dense Muon、GreedyLoRE r32、GreedyLoRE r128、Dense AdamW；每组都从头训练。Dense Muon 重试 ID 为 `CM001-m001-dense-muon-llama60m-c4-dense-bf16-s1243-rerun1`，其余三组沿用尚未启动过的 CM002–CM004 ID。已确认 torchrun 与四个训练 rank 启动；目前尚无完成结果。
- 进一步以 30 shard、4 rank、每 rank 4 worker 的小型回归测试确认 120 条记录全部恰好读取一次。Dense Muon 重试已完成首个更新，正在执行第 1 步 validation。

## 2026-09-25：130M Dense Muon 启动

- 实验编号：`CM005-m001-dense-muon-llama130m-c4-dense-bf16-s1243`，使用仓库 `c4/configs/llama_130m.json`（hidden 768，约 134.1M 参数），与后续三组固定同一架构。
- 训练：本地 C4、4 GPU（3–6）、每卡 batch 128、GA=1、序列长 256、20,000 更新步（2,621,440,000 个含 padding 的名义 token 位置）、warmup 2,000、cosine 到 10%、BF16、activation checkpointing、seed 1243。Muon matrix LR 0.02、momentum 0.95、spectral-norm scaling；scalar AdamW LR 0.001。梯度裁剪 1.0，weight decay 0。
- 用户授权 W&B online、无 checkpoint；`save_every=40000` 高于训练预算。运行入口 `c4/scripts/run_table_iv_130m.bash`，串行会话 `greedylore_tableiv_130m_bf16`，输出 `outputs/<实验编号>/`。已确认 torchrun 及四个 rank 启动；尚无结果。
- 启动前检查：30 个本地 shard 有 10,689,518 条记录，20,000 步需要名义上 10,240,000 条；脚本静态检查及 21 项单元测试通过。数据余量约 4.4%，训练入口会在提前耗尽时明确报错。
- 启动后检查：CM005 的 W&B ID 为 `op0yd9v0`，日志显示 `Syncing run`；训练已完成第 1 次参数更新并进入第 1 步 validation。GPU 3–6 各使用约 11 GiB；输出目录下尚无 checkpoint。

## 2026-09-26：Table III GLUE Dense Muon 对照排队

- 实验编号：`CM009`、`CM012`、`CM015`、`CM018`、`CM021`、`CM024`、`CM027`、`CM030`，依次对应 SST-2、CoLA、MRPC、STS-B、RTE、QNLI、QQP、MNLI；完整运行目录名含任务和 seed。
- 与同任务的 M002 rank 8/16 组使用相同 RoBERTa-base、GLUE 本地 Arrow 数据、seed 1243、FP32、4 GPU、每卡 batch 4、10 epoch、序列长 256、cosine 到 0、LR warmup 10%、weight decay 0。矩阵 Muon LR `2e-4`、momentum 0.95、spectral-norm scaling；AdamW fallback LR `5e-5`。学习率未 sweep。
- 运行入口为 `glue/scripts/run_table_iii_muon.bash`；先等待 GPU 0/1/2/7 的 GPU 利用率均低于 20%，不检查显存利用率，再串行执行 24 个任务。结果口径为 validation，不是论文的 test。
- 用户进一步要求轮询时避免前序任务训练间隙的短暂低利用率导致抢占。启动器现先等待前序 `cm008-table2-randk-topk-muon-gpu0-1-2-7` tmux 会话结束，再要求四张 GPU 连续 3 次（间隔 30 秒）利用率均低于 20%；显存利用率不作为门槛。启动器会话 `greedylore_tableiii_muon`，日志 `outputs/table_iii_muon_launcher.log`。截至启动检查仍在等待前序会话，尚未开始 CM009。

## 2026-09-26：Table III 四卡直接短试跑及调度修复

- 按用户更正暂停正式启动器，直接在 GPU 0/1/2/7 执行 SST-2 Dense Muon 4 个更新步（`--max_train_steps 4`，每卡 batch 4），成功完成 validation；结果 `outputs/smoke-direct-tableiii-dense/all_results.json`，accuracy 仅约 0.491，不作为正式性能指标。
- 短跑暴露 GLUE 入口的 LR 调度器在 4 卡下每个 optimizer step 推进 4 次，记录为 0、`2e-4`、0、`2e-4`。设置 `Accelerator(step_scheduler_with_optimizer=False)` 后，同样 4 步的 LR 依次为约 `1.707e-4`、`1e-4`、`2.929e-5`、0；验证产物 `outputs/smoke-direct-tableiii-dense-scheduler-fix/`。
- 同时将总更新步和 warmup 的计算移到 Accelerate 数据切分之后。用本地 SST-2 train 的 64 条临时子集做 4 卡、1 epoch、4 更新步检查，日志 `outputs/smoke-direct-tableiii-scheduler-auto.log` 显示自动预算为 4，25% warmup 后 LR 为 `2e-4`、`1.5e-4`、`5e-5`、0。临时子集仅用于调度检查，正式运行仍使用完整本地 GLUE 数据。
- 静态检查、24 条正式命令 dry-run、24 项单元测试通过后，于 09:14:58 重新启动 `greedylore_tableiii_muon`；启动检查仍在等待前序 CM008 tmux 会话退出，正式训练尚未开始。
- 用户更正启动条件：不等待 CM008 会话。正式启动器仅轮询 GPU 0/1/2/7 的 `utilization.gpu`，每 300 秒采样一次，连续两次都低于 20% 即开始；显存利用率不参与判断。旧启动器已停用，并按新条件重启；此前“必须等 CM008 退出”不再适用。

## 2026-09-26 12:00：Table III 改为 W&B online 后从头重跑

- 用户要求立即停止原本不带 W&B 的正式训练，禁止事后上传，并且不保存 checkpoint。已中断 `greedylore_tableiii_muon` 原会话；旧 CM009 已完成、CM010 在约 15,453/42,100 更新步中断，这些结果不纳入本次 W&B online 正式比较。
- 更新 GLUE 入口：W&B run 名使用 `--wandb_run_name`，正常结束时读取当前 W&B `run.url`；启动器设置 `WANDB_MODE=online`、`--with_tracking --report_to wandb`。各任务项目名为 `glue_no_trainer_<task>_greedylore`，同一项目包含 Dense/r8/r16；run 名与本地目录同为原 CM 编号加 `-rerun1`。没有设置 `--checkpointing_steps` 或 `--resume_from_checkpoint`，因此不保存训练 checkpoint。
- 删除未使用的事后上传脚本及测试。24 条命令 dry-run 均含 online tracking 参数且无 checkpoint 参数；GLUE CLI 与本地数据测试 3 项通过。12:00:44 重新创建 `greedylore_tableiii_muon` 会话，按 GPU 0/1/2/7 每 300 秒连续两次低于 20% 的条件等待；12:00:45 首次检测四卡均为 0%，计数 1/2。新训练尚未开始。
- 12:05:45 第二次检测四卡均为 0%，随即从头启动 CM009 `-rerun1`；12:05:59 W&B 日志显示 `Syncing run`，run ID `g5iydejl`，链接 https://wandb.ai/yiran0834-huazhong-university-of-science-and-technology/glue_no_trainer_sst2_greedylore/runs/g5iydejl 。12:06:04 已完成前 13 个更新步；新输出目录只有 `train.log`，没有 checkpoint。

## 2026-09-29：完成结果核对

- 以各 run 的 `outputs/<完整 run ID>/all_results.json` 和 `train.log` 核对：CM001 `-rerun1` 完成修订后的 8,393 步，final C4 validation loss/PPL 为 3.61430 / 37.1255；CM005 完成 20,000 步，为 3.29619 / 27.0095。两组日志均有 `Script finished successfully`。早期只到 5,220 步的 CM001 仍是失败记录，不并入正式结果。
- GLUE Dense Muon 的 CM009、CM012、CM015、CM018、CM021、CM024、CM027、CM030 均以 `-rerun1` 完成第 10 个 epoch，并产生机器可读结果。最终 validation 指标和对应源字段已整理在 `docs/results.md`；MNLI matched 的最终指标来自 epoch 9 日志，JSON `eval_accuracy` 对应 mismatched。
- 上述结果均只有 seed 1243，属于初步结果；C4 60M 的 8,393 步与论文附录的 10,000 步不同，GLUE 是 validation 而非论文 test。

## 2026-10-01：130M Muon 矩阵 LR 两组比较

- 按用户要求，使用 `c4/scripts/queue_table_iv_130m_muon_lr.py` 串行运行 `CM040-m001-dense-muon-llama130m-c4-dense-lr0p01-bf16-s1243` 与 `CM041-m001-dense-muon-llama130m-c4-dense-lr0p005-bf16-s1243`，矩阵 LR 分别为 0.01、0.005；scalar LR 均为 0.001。
- 其余沿用 `c4/scripts/run_table_iv_130m.bash dense`：LLaMA 130M、C4、4 GPU、20,000 步、BF16、global batch 512、warmup 2,000、cosine、Muon momentum 0.95、spectral-norm scaling、seed 1243、W&B online、无 checkpoint。数据路径固定为 `c4/c4_en`，与随后 GreedyLoRE 两组一致。
- 以两组完成 20,000 步的 `all_results.json` 中 `final_eval_loss` 严格选较低者；相等时选择较小 LR。选中 LR 再用于 M002 的 r32/r256。队列状态与日志在 `outputs/CM040-CM043-muon-matrix-lr/`，训练结果在各自 `outputs/<run ID>/`。
- 用户授权首组在已确认空闲的 GPU 0–3 直接启动；后续各组需等待任意同一组 4 张 GPU 连续两次空闲（间隔 5 分钟，显存 ≤1,500 MiB、利用率 ≤10%）。
- 已启动 tmux 会话 `greedylore_130m_muon_matrix_lr`；`status.tsv` 确认 CM040 用 GPU 0–3、matrix LR 0.01 启动。训练日志显示完成首次更新并进入第 1 步 validation；其余三组等待自动串行执行。两个 LR 选择单元测试及脚本静态检查通过。

## 2026-10-02：130M Dense 学习率选择完成结果核对

- 目的：核对 CM040/CM041，并为同数据、同预算的 r32/r256 比较确定矩阵 LR。配置与运行入口沿用上一条记录；实际均使用 GPU 0–3 的 4 张 RTX 4090。
- 结果：`CM040-m001-dense-muon-llama130m-c4-dense-lr0p01-bf16-s1243` 完成 20,000 步，final validation loss/PPL 为 3.13209 / 22.9218；`CM041-m001-dense-muon-llama130m-c4-dense-lr0p005-bf16-s1243` 同预算为 3.14294 / 23.1718。CM040 的 loss 低 0.01085，队列因此选择矩阵 LR 0.01。
- 验证：逐项核对 `outputs/<上述完整 run ID>/all_results.json` 的 `update_step`、`final_eval_loss`、`final_eval_ppl`，日志的最终 loss 与成功结束标记，以及 `outputs/CM040-CM043-muon-matrix-lr/status.tsv` 的 completed/selection 记录。本地运行配置确认数据为 `c4/c4_en`。
- 结论与下一步：两候选中 0.01 的最终验证指标较低；每组仅 seed 1243 一次，且使用 validation 选 LR，不能称为稳健最优。与旧数据路径的 CM005 不作单因素 LR 比较。压缩组结果及限制见 `docs/results.md`，后续固定配置做多 seed 确认。


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

- 本方法臂：`CM044-m001-dense-muon-resnet18-cifar10-r64-lr0p02-fp32-s1243` (GPU 0–3 / project `GreedyLore-CIFAR10-Muon`)，`CM046-m001-dense-muon-resnet18-cifar100-r64-lr0p02-fp32-s1243` (GPU 4–7 / project `GreedyLore-CIFAR100-Muon`)。`compressor=none`；run ID 中 r64 是配对协议标识，不代表 Dense 在压缩。

- 启动核对：tmux `greedylore_cifar10_muon` / `greedylore_cifar100_muon` 已启动，显式隔离 GPU 0–3 / 4–7；CM044/CM046 的 config.json 确认 matrix LR 0.02、scalar LR 0.005/0.0005、global batch 128、391 steps/epoch、40 epochs、online、无 checkpoint，日志均已完成首个更新。
- W&B 映射：CM044 run `9gq5lr0g`，https://wandb.ai/yiran0834-huazhong-university-of-science-and-technology/GreedyLore-CIFAR10-Muon/runs/9gq5lr0g ；CM046 run `runsb22r`，https://wandb.ai/yiran0834-huazhong-university-of-science-and-technology/GreedyLore-CIFAR100-Muon/runs/runsb22r 。训练进行中，没有最终结果。

- 额外验证：3 项既有训练入口 CLI 检查通过（运行时需要 `PYTHONPATH=.`，首次未设置而 C4 import 失败；未修改 C4）。共 18 项相关单元/入口检查通过。独立代码审查未发现训练/评估阻塞；按反馈明确记录 FP32 模型/梯度与 BF16 正交化，显式指定两组 GPU，保护既有队列状态不被覆盖并补充队列脚本 SHA256。
- 运行维护：普通长任务等待交给 tmux 和 shell 队列，停止 Agent 轮询。完成时每 run 自动写 final/best test 指标、通信口径、all_results.json 与 TRAINING_COMPLETED；失败保留日志并停下对应队列。以后整理结果须先核对完整预算和成功结束标记。


## 2026-10-02：CIFAR Dense 完成结果核对

- CM044 / CIFAR-10 与 CM046 / CIFAR-100 均完成 40 epochs、15,640 updates，完整评估 10,000 张 test；final test accuracy / loss 分别为 94.43% / 0.31747798724099996 和 75.91% / 1.2723289066553116。参数沿用上一条：matrix LR 0.02，scalar LR 0.005/0.0005，4 GPU、global batch 128、seed 1243、W&B online、无 checkpoint。
- 来源：上述完整 run ID 对应 `outputs/<run ID>/all_results.json` 的 `final_test_accuracy`、`final_test_loss`；已核对 `status=completed`、`epoch=40`、`update_step=15640`、`test_samples=10000`，与 `metrics.jsonl` 最后一行及日志 `TRAINING_COMPLETED` 一致。两个队列状态均为 `queue_finished`，W&B 日志确认同步完成。
- 配对压缩组 CM045/CM047 的最终准确率分别低 0.14/0.97 个百分点。每组 n=1，属于初步观察；汇总见 `docs/results.md` 的 CIFAR 节，后续如需稳健结论应固定配置做多 seed 重复。

## 2026-10-02：Table V Muon 测速矩阵启动

- 用户要求按论文 Table V 尽可能一致地测速，并将 Adam 替换为 Muon。实验约定及所有未证实设置、架构偏差见 `docs/table-v-muon-protocol.md`。
- M001 Dense 臂为 CM048 / CM051 / CM054 / CM057，对应 60M / 130M / 350M / 1B；与 M002、M004 同模型三臂使用相同 Muon 设置、C4、seed、batch 和计时窗口。
- 实际配置：4 × RTX 4090（GPU 0–3）、NCCL SHM、BF16、sequence256、每卡 batch128（1B 为64）、GA1、checkpointing、matrix/scalar LR0.01/0.001、momentum0.95、spectral norm、WD0、clip1、先预热1000步，再测1001–1500的连续500步。Muon 内部 collective 计入；不做验证、不保存 checkpoint、不使用 tracker。

## 2026-10-02：暂停 Table V Muon 队列

- 按用户要求暂停 CM048–CM059 队列，准备改测 350M AdamW。CM048–CM056 已完成并保留；CM057 Dense Muon 1B 在运行中被终止，只有 `config.json`，不得作为完成结果。CM058–CM059 未启动。
- tmux 会话停止后发现 CM057 的 `torchrun` 成为孤儿进程，已只终止该已确认进程树并核对 GPU 释放。
- 新增 `c4/table_v_timing.py`、`c4/scripts/run_table_v_muon.py`、`tests/test_table_v_timing.py`。验证：19项相关单元测试通过，编译和diff检查通过；130M真实C4、batch128的四卡三路径smoke均完成3步测量；NCCL日志确认SHM。smoke不是正式结果。
- `greedylore_table_v_muon` tmux 已启动12臂串行队列；主目录 `outputs/CM048-CM059-table-v-muon/`，启动日志 `outputs/table-v-smoke/controller-launch.log`。当前无正式500步测速结果，后续状态由 `status.tsv` / `summary.json` 保存；最终自动生成 `summary.csv` / `results.md`。每臂失败会保留exit code与日志，不自动降低论文batch。

## 2026-10-03：GreedyLoRE timing setting matrix

- 用户授权比较9组系统设置，每组完整运行60M/130M/350M/1B的Dense与GreedyLoRE；Dense臂为CM062–CM133中的偶数编号。完整协议见 `docs/table-v-muon-setting-matrix.md`。
- 全部组使用Muon、C4、sequence256、rank32配对协议和1024 MB DDP bucket。设置覆盖4/8 GPU、BF16/FP32、每卡batch 1/32/论文大batch、默认/单NCCL channel、SHM/Socket；除论文大batch参照外关闭activation checkpointing。
- 计时预热101步并连续测量400步；OOM或超时记为失败并继续，不自动改变配置。入口只为独立timing路径增加DDP bucket参数，不修改原训练入口或GreedyLoRE hook。
## 2026-10-03：严格阻塞通信测速

- 新增 CM134–CM165 矩阵中的 Dense Muon 偶数臂，与每个 GreedyLoRE 臂使用相同模型、batch、GPU 数、FP32、Muon 和系统配置；每臂只运行一次。
- 独立 timing 入口在 DDP hook 内同步并等待原 Dense all-reduce Future，记录严格阻塞 hook 时间及实际 bucket 布局；原通信 hook 未修改。统一 `bucket_cap_mb=8192`、关闭 checkpointing、默认 NCCL channel、P2P 关闭、SHM 开启。
- 四组为论文工作负载参照（4 卡、每卡 batch 128，1B 为 64）、4 卡 batch32、4 卡 batch1、8 卡 batch1。预热 101 步，测量更新 102–501 共 400 步。完整协议见 `docs/table-v-muon-strict-blocking.md`。
- 静态测试和 dry-run 通过；8 卡 GreedyLoRE smoke 也验证了共用包装器与结果字段。正式产物写入 `outputs/CM134-CM165-table-v-muon-strict-blocking/`，结果待队列完成后补记。

### BF16 论文设置追加组

- 按用户要求增加 CM166/168/170/172 Dense Muon：4 卡、BF16、activation checkpointing、论文每卡 batch（前三个模型128，1B为64），其他严格阻塞与压缩对照设置不变；每臂一次。
- 追加组不打断或重跑 CM134–CM165，在首批结束后自动启动；输出位于 `outputs/CM166-CM173-table-v-muon-strict-blocking-bf16-paper/`。

### 系统变量精简矩阵

- 用户停止 BF16 论文追加组：Dense CM166、CM168 完成；CM170 在运行中终止，CM172 未启动。遗留 CM170 rank 已只按已确认 PID 终止并释放 GPU。
- 首次误按仅350M启动CM174，用户补充每个setting须跑四个模型后立即停止；CM174没有完成结果，CM175–CM187未启动，旧目录保留为中断尝试。
- 正式矩阵改为CM188–CM243，每个setting完整运行60M/130M/350M/1B，统一batch1、无checkpointing；比较4/8卡、FP32/BF16、默认/单channel及8卡BF16 Socket。按用户要求排除4卡与8卡的FP32 Socket配置。
- 每个系统配置和模型的Dense/GreedyLoRE各一次，并逐对交替执行顺序。完整矩阵与口径见 `docs/table-v-muon-strict-blocking.md`。
- CM188–CM243队列已结束：56臂中45个完成、11个日志确认CUDA OOM，形成21组完整配对；45个成功结果均完成400步严格阻塞窗口并实际记录单bucket。Dense的8卡BF16 1B三种模式均完成，FP32 1B均OOM。
- SHM中350M、8卡FP32的Dense/GreedyLoRE iter speedup在默认/单channel下为1.0780×/1.0892×；8卡BF16 Socket 350M为1.6343×。完整表、hook时间、失败项和来源核验见`docs/results.md`。

### 正常异步 DDP bucket sweep

- 新增Dense CM244/CM246，固定350M、8卡、FP32、batch1、默认channel/SHM和其余Muon设置，仅将DDP bucket cap设为256/1024 MiB；不用严格阻塞，测量正常异步端到端时间。
- timing入口的被动包装器只记录实际bucket布局并原样返回Dense all-reduce Future，不同步、不等待。预热101步、测量400步；完整协议见`docs/table-v-muon-strict-blocking.md`。
- CM244/CM246均完成。256/1024 MiB cap实际形成5/2个bucket，Dense iteration time为0.301390/0.324152秒；小bucket使Dense比1024 MiB快约7.02%，与更多通信/计算overlap一致。


## 2026-10-03：350M Table IV Muon LR筛选与正式比较

- 用户明确：每项4卡；Dense LR0.01/0.005各10000步sweep并行，胜出LR用于从头60000步Dense/r32并行，之后4卡r256。Dense身份CM248/CM249/CM250；M002身份CM251/CM252。
- 协议：`docs/table-iv-350m-muon-protocol.md`；队列`c4/scripts/queue_table_iv_350m_muon.py`。以Table VII为准：schedule60000、warmup6000、global batch512、每卡128、sequence256、cosine到10%、WD0、clip1。新增`--stop_after_steps`仅改变停止预算，scheduler horizon保持60000。
- 数据独立准备96 train/8 validation固定revision分片，复用已有文件并补充；精确计数和SHA256校验后做三臂短smoke，再自动启动实验。不循环数据，不覆盖已有结果；Muon和通信实现未改。
- 验证：11项相关单元/数据回归测试通过（多worker测试需沙箱外进程通信），dry-run和diff检查通过；实际正式训练和smoke结果尚未产生。
- 主指标best validation PPL、副指标final PPL；sweep按10000步final loss选LR，失败阻止下一阶段，n=1探索性适配。队列目录`outputs/CM248-CM252-table-iv-350m-muon/`，日志`outputs/CM248-CM252-table-iv-350m-muon.queue.log`。

- 首次启动的编号检查误将控制器自身日志当作CM248已存在，在数据/训练前退出。已缩小检查到实验身份前缀并补充回归测试；失败日志保留为`outputs/CM248-CM252-table-iv-350m-muon.start-failed.log`，不计作训练run。

- 用户因流量不足明确禁止下载，已停止旧tmux队列。50个已有分片已校验；没有新增完整文件，残留一个201124976字节的部分下载。改为只用原`c4/c4_en`，离线模式，显式重复训练数据并丢弃尾部不足128的batch，保持60000步/global batch512。旧队列及日志保留；新控制目录`outputs/CM248-CM252-table-iv-350m-muon-existing-data/`。

- Dense四卡batch128短验证完成（5次更新，测量2次）；下一臂启动前GPU利用率检查立即失败。随后核实8卡显存/利用率均0且没有计算进程，符合短验证结束后的利用率读数残留。修正为控制器最多等待60秒再检查，不停止其他任务；加入回归测试并通过11项单元检查。增加仅前置验证阶段的`--resume`，保留原manifest、状态及日志，跳过有成功结果的Dense smoke，恢复剩余r32/r256验证。正式sweep尚未启动。

## 2026-10-06：CM250 Dense Muon 350M 完成结果核对

- 目的与配置：为同设置的 GreedyLoRE r32（CM251）提供 Dense 对照。先以 CM248/CM249 的第 10,000 步 final validation loss 3.23100794/3.21209903 选择矩阵 LR 0.005；CM250 从头训练 60,000 步。使用已有 50 个 C4 train 分片重复数据、8 个 validation 分片、4 张 RTX 4090、BF16、seed 1243、全局 batch 512、warmup 6,000、cosine 调度、Muon scalar LR 0.001；完整设置见 `docs/table-iv-350m-muon-protocol.md` 和 run 的 `command.txt`。
- 验证与结果：`outputs/CM250-m001-dense-muon-llama350m-c4-dense-formal-lr0p005-bf16-s1243/all_results.json` 记录 `update_step=60000`，final validation loss/PPL 为 2.85496862/17.37389197，best PPL 同为 17.37389197（第 60,000 步）；final 评估覆盖 10,048,075 个有效 token。同目录 `train.log` 有 `Script finished successfully`，队列 `outputs/CM248-CM252-table-iv-350m-muon-existing-data/status.tsv` 记为 `completed`。
- 观察与下一步：与 CM251 同口径比较见 `docs/results.md`。本轮只做一次 seed 1243 训练，且训练数据重复；固定配置的独立重复才可检验差异是否稳定。
