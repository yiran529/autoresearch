# Table V：Muon 版本测速约定（2026-10-02）

目标是比较同模型、同数据和同 Muon 设置的 Dense、PowerSGD、GreedyLore 每次训练更新的墙钟时间；属于优化器替换后的探索性实验，不是 AdamW 数值复现或收敛验证。用户已授权启动。

论文来源：[GreedyLore arXiv:2507.08784v4](https://arxiv.org/pdf/2507.08784)，正文 VII-C、Table V 与附录 F / Table VII。主要指标为每个 rank 连续 500 次更新的总时间，取最慢 rank 除以 500；GreedyLore 加速比为同模型 Dense 时间 / GreedyLore 时间。每臂 seed 1243 一次；500 次迭代是时间样本，不是 500 次独立实验。

## 论文明确设置及实际值

| 项目 | 论文 | 实际 |
| --- | --- | --- |
| GPU / backend / transport | 4 × RTX 4090 24 GB，NCCL SHM | GPU 0–3，4 × RTX 4090；禁用 P2P、启用 SHM；smoke 日志确认 SHM/direct/direct |
| 统计窗口 | 连续 500 次迭代 | 更新步 1001–1500，完整连续 500 步，CUDA 每步同步，所有 rank 计时 |
| 压缩秩 | 32 | 两种压缩均 rank 32 |
| 每卡 batch | 60M / 130M / 350M 为 128，1B 为 64 | 相同，无梯度累积，全局 batch 512 / 256 |
| 数据 | C4 | 本地英文 C4 `c4/c4_en`；真实流式数据，T5 tokenizer，shuffle seed 42；无合成或循环填充 |
| 优化器 | AdamW | 全部换为现有 Muon；矩阵 LR 0.01、scalar AdamW LR 0.001、momentum 0.95、spectral-norm scaling、Polar Express；禁用 torch.compile |

## 必须保留的未知项和偏差

- Table V 的 `dmodel=256/512/768/1024` 与仓库模型不一致。模型选择曾向用户询问，执行时未收到回复，按建议采用现有可执行配置，**不能称作完全一致的 Table V 架构复现**。130M / 350M / 1B 使用仓库 JSON（hidden 768 / 1024 / 2048），不为匹配表格隐藏维度而自行猜测层数。60M 使用此前推定的 `llama_60m_table_iv.json`：hidden 256、56 layers、FFN 688、8 heads，60,682,496 参数；论文未给出完整架构。
- Table V 没有单独给出 dtype、序列长、checkpointing、测量起点、seed、LR 或 DDP bucket size。实际采用本地 C4 路径的 BF16 模型/梯度、FP32 fallback 状态、开启 activation checkpointing、序列长 256、默认 DDP bucket 25 MiB；这些是本地明确选择，不能当作论文 Table V 已证实设置。序列长 256 在附录 Table VII 的其他 C4 实验中出现。
- 预热 1000 次真实训练更新；压缩从 hook iter=1000 开始，因此测量窗口全程处于启用压缩的阶段。GreedyLore 在测量更新步 1001、1201、1401 更新投影，包含首次 SVD 与缓冲分配；不删掉慢步。论文 Table V 未说明是否排除了首次初始化。
- GreedyLore 用 `top_subspace` / EF14 / `update_proj_gap=200` / `min_compression_rate=1.0` / `beta_ef=0` / `sigma_type=0` / `error_inherit=0` / `scale=1`；投影更新间隔与 EF 有附录 C4 参照，其余以现有实现为准。这不是 `lore` hook。
- PowerSGD 用仓库 `powersgd` / EF14，保留其默认 warm-start、最小压缩率和正交化实现，不改成 `original_powersgd`。它与 GreedyLore 的压缩覆盖范围并不完全相同：GreedyLore 排除 embedding / lm_head；PowerSGD 沿用自己的矩阵筛选规则。
- 两种压缩作用于 Muon 正交化前的梯度，属于有损近似 Muon。Muon 内部梯度布局检查和正交化结果 AllGather 全部计入端到端时间；不声称与 Dense Muon 通信等价，也不声称只压缩 Muon 参数。
- LR 选择是适配：0.01 取自已有 130M Dense Muon 探索结果，不做本次测速专属调参。每个模型三臂完全相同。weight decay 0、gradient clipping 1、cosine 到 10%；调度总步数 / LR warmup 对 60M、130M、350M 为 10000/1000、20000/2000、60000/6000，参照附录 Table VII，**不是运行完整训练预算**。1B 的对应设置论文未给出，本地选 100000/10000，仅跑1500步。
- 测速入口排除评估、checkpoint 和外部 tracker，计入数据等待、H2D、forward/backward、DDP hook、clipping、Muon step/collectives、scheduler、zero_grad、CUDA 同步。窗口内不输出入口逐步日志，hook 自身的开销保留。
- 若论文 batch 显存不足、loss 非有限、数据耗尽或超过单臂 6 小时，记录失败并继续其他组，不自动调整 batch / sequence / dtype，不将部分窗口当作结果。

## 身份、产物和运行方式

| 模型 | Dense Muon (M001) | PowerSGD + Muon (M004) | GreedyLore + Muon (M002) |
| --- | --- | --- | --- |
| 60M | CM048 | CM049 | CM050 |
| 130M | CM051 | CM052 | CM053 |
| 350M | CM054 | CM055 | CM056 |
| 1B | CM057 | CM058 | CM059 |

独立入口 `c4/table_v_timing.py` 复用现有模型、数据处理、scheduler、Muon 和通信注册；不修改原训练入口或虚拟环境。串行控制器 `c4/scripts/run_table_v_muon.py`，会话 `greedylore_table_v_muon`；12 臂全部使用同一组卡，启动前由脚本等待空闲，不干扰其他进程。

产物目录 `outputs/CM048-CM059-table-v-muon/`：每臂命令、日志、唯一目录、`config.json`（包含模型实际参数量、Muon 分组、版本、git HEAD、关键源码 SHA256、NCCL 环境）、`all_results.json`（完整每 rank 每步时间、loss、窗口总时间、显存）。控制器保留 manifest、数据文件大小/mtime 清单、status.tsv，并逐臂更新 summary.json，最终生成 summary.csv 与 results.md。结果只采用 exit code 0、status completed、measured_steps=500 的臂。

```bash
PYTHONPATH=. .venv/bin/python c4/scripts/run_table_v_muon.py --dry-run
# 由 tmux 后台运行，启动日志保存在 outputs/table-v-smoke/controller-launch.log
PYTHONPATH=. .venv/bin/python c4/scripts/run_table_v_muon.py --gpus 0,1,2,3
```

静态测试验证最慢 rank 的连续窗口时间聚合、拒绝不完整/非有限数据；4 GPU 真实 C4、batch128 短 smoke 分别验证三种通信路径。短 smoke 的 3 次计时包含初始化，不能作为正式 Table V 结果或方法间加速结论。
