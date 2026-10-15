# Table IV 350M：Muon 适配实验

## 2026-10-03 预先约定

- 问题：固定 Muon 后，GreedyLoRE r32/r256 与 Dense 的验证 PPL 相差多少？
- 类型：论文协议下的探索性适配；论文使用 AdamW，不能将 Muon 数值称作原论文结果复现。
- 来源：[GreedyLoRE Table IV、附录 F / Table VII](https://arxiv.org/pdf/2507.08784)。以 Table VII 的更新步预算为准：60000步、warmup6000、global batch512、sequence256、linear warmup、cosine到10%、WD0、clip1、子空间间隔200。
- 本地模型：`c4/configs/llama_350m.json`，hidden1024、24层、FFN2736、16 heads，约368M参数。Table IV hidden1024与本地一致，但论文未给出完整架构，不能声称逐项架构完全一致。
- 硬件：每项4张4090、每卡batch128、GA1。两任务并行时一项GPU0–3、一项GPU4–7；r256单独GPU0–3。端到端速度可能受共享CPU/IO影响，不能直接用这轮并行日志声称方法加速。
- 本地控制项：BF16、activation checkpointing、seed1243、T5 tokenizer、padding/truncation256、数据shuffle42、4 workers/rank。三组一致。不修改虚拟环境或优化器/通信算法。
- Muon：matrix LR由Dense sweep选择；scalar AdamW LR0.001、betas(0.9,0.999)、eps1e-8、WD0；Muon momentum0.95、epsilon1e-8、spectral_norm scaling，分布式正交化。
- 压缩：`top_subspace`、EF14、rank32/256、投影间隔200、压缩起点1000、min_compression_rate1、beta_ef0、error_inherit0。压缩DDP梯度并聚合/重构后再做Muon动量及正交化，是有损梯度通信适配。
- 数据（用户更正）：只用已有`c4/c4_en`的50 train / 8 validation分片；不下载、不补数据。HF_HUB_OFFLINE和HF_DATASETS_OFFLINE开启。正式训练耗尽后从同一顺序重复已有数据，跳过不足每卡128的尾批，确保global batch512和60000更新步；sweep在10000步结束，无需重复。停止前已精确计数全部50分片，共17815864条。
- 原96分片准备队列已停止，没有新分片完整下载成功；一个新增分片留下201124976字节部分下载。保留失败/中断记录，原50分片未改。
- 名义正式预算为60000×512×256=7,864,320,000个含padding token位置；正文6.4B和附录token计数与算式不一致，保留说明，同时记录训练实际非padding token。

## 矩阵与决策规则

| 编号 | 方法 | matrix LR | 停止步 | GPU |
|---|---|---|---:|---|
| CM248 | M001 Dense sweep | 0.01 | 10000 | 0–3 |
| CM249 | M001 Dense sweep | 0.005 | 10000 | 4–7 |
| CM250 | M001 Dense正式 | 胜出值 | 60000 | 0–3 |
| CM251 | M002 GreedyLoRE r32正式 | 同一胜出值 | 60000 | 4–7 |
| CM252 | M002 GreedyLoRE r256正式 | 同一胜出值 | 60000 | 0–3 |

两组sweep并行，保留60000步scheduler horizon和6000 warmup，通过`--stop_after_steps 10000`提前结束。以第10000步final validation loss较低者选择共同LR，完全相等选0.005。必须两组退出成功、预算完整且loss有限；任何失败阻止后续阶段，不把不完整运行当候选。两组正式Dense/r32从头并行跑；二者成功完成后从头跑r256。无早期PPL筛选或按观察改配置。

主指标是固定评估频率下的`best_eval_ppl`及对应步数，副指标为final loss/PPL、token数及曲线。第1步、每1000步和结束时评估；沿用入口约10M validation target token、shuffle42、token加权loss。论文未明示相同评估频率，不能把跨论文差值仅归因于Muon。

每臂只有seed1243，独立重复单位是完整训练run；各评估点不是独立重复。不计算显著性或多seed置信区间。LR选择及正式比较使用同一validation，因此属于探索性。报告r32/r256相对Dense的loss差值和PPL相对变化，不预设等价界值或声称理论保证迁移到Muon。

## 执行与记录

入口`c4/scripts/queue_table_iv_350m_muon.py`，tmux会话`greedylore_tableiv_350m_muon`。
控制器目录`outputs/CM248-CM252-table-iv-350m-muon-existing-data/`保存queue日志、status.tsv、git HEAD和代码hash、选择结果；每run目录保存命令、protocol.json、train.log、W&B及all_results.json。W&B online，无checkpoint（save_every120000）；中断保留记录，重跑需新身份。

正式任务前串行检查Dense/r32/r256四卡batch128、5次更新的smoke；压缩从第2次通信迭代起，覆盖投影和低秩步骤。smoke使用已有`c4/table_v_timing.py`，只验证路径和显存，不用于PPL或性能结论。本轮显式开启重复数据；未开启该选项的已有训练行为仍为耗尽即失败。

失败保留日志；阶段启动前重新检查GPU空闲，若被其他任务占用，队列停止，不干扰该任务。训练和短验证由后台控制器编排，不用LLM轮询等待训练完成。


## 用户约束更正

因流量不足，用户要求停止下载、直接使用已有数据。正式训练保留60000步但重复数据，因此不能称为与论文独立数据曝光完全一致的复现；各臂重复规则一致，比较仍是本地探索性比较。

- GPU空闲检测最多等待60秒允许刚退出进程的利用率读数回落；仍占用则停止。前置验证可用`--resume`恢复，保留历史manifest并记录新代码hash；正式身份已使用时拒绝重启。
