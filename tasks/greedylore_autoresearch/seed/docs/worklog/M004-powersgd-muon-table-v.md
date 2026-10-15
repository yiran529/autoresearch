# M004：PowerSGD + Muon Table V 测速对照

## 2026-10-02：实现验证与正式矩阵排队

- 目的：Table V的PowerSGD行也替换为Muon，与同模型Dense和GreedyLore比较连续500次训练更新的墙钟时间。属于单seed探索性速度实验，不验证论文Adam收敛结论。
- 使用现有 `powersgd` hook，而非 `original_powersgd`；rank32、EF14、start_compress_iter1000。作用于正交化前梯度，属于有损近似Muon；保留PowerSGD默认矩阵筛选、warm start及正交化，与GreedyLore覆盖范围可能不同。
- 正式实验身份为CM049 / CM052 / CM055 / CM058（60M / 130M / 350M / 1B）。每臂与M001/M002共用Mu​​on分组、LR0.01/0.001、momentum0.95、spectral norm、seed1243、C4、BF16、checkpointing、sequence256、每卡batch128（1B为64）、GA1、NCCL SHM、4张4090。
- 先预热1000步，再计更新1001–1500；包含数据等待、DDP通信、Mu​​on内部collective及其余训练更新；无评估、checkpoint或tracker。详细约定及无法证实的论文设置见 `docs/table-v-muon-protocol.md`。
- 130M真实C4、每卡batch128的四卡PowerSGD短smoke完成3次测量；另外两臂smoke也完成，NCCL日志实际确认SHM；19项相关单元测试通过。原始验证在 `outputs/table-v-smoke/`，不能用其短窗口推断正式加速。
- 12组正式矩阵通过 `c4/scripts/run_table_v_muon.py` 排入tmux会话 `greedylore_table_v_muon`；当前无完成结果，状态和逐臂产物见 `outputs/CM048-CM059-table-v-muon/`。失败保留，不自动改变batch或隐藏失败。

## 2026-10-02：暂停 Table V Muon 队列

- 按用户要求在CM057运行期间停止队列。PowerSGD的CM049、CM052、CM055已完成并保留；1B CM058未启动。
