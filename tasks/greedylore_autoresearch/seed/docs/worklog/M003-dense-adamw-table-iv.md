# M003：Table IV 60M Dense AdamW

## 2026-09-25：追加对照并排队

- 目的：追加论文 Table IV 的 Dense AdamW 对照，和 CM001–CM003 使用相同 C4、推定的 LLaMA 60M 架构、BF16、全局 batch 512、每卡 batch 128、GA=1、seed 1243、10,000 更新步、评估口径及无 checkpoint 设置。
- 实验编号：`CM004-m003-dense-adamw-llama60m-c4-adam_dense-lr2p5e3-bf16-s1243`。
- 超参数：`--optimizer adamw --compressor none --lr 0.0025 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay 0`；warmup 1,000 步，cosine 降至峰值 LR 的 10%，梯度裁剪 1.0。用户从论文附录给出的 AdamW LR 候选集合 `{0.0015, 0.0025, 0.005}` 中选定单次运行值 0.0025；论文没有指出 Table IV 的 33.44 对应哪个候选值，不能将本次运行称为论文最佳学习率复现。
- 代码：`c4/scripts/run_table_iv_60m_muon.bash adam_dense`；排队入口 `c4/scripts/run_table_iv_60m_adam_after_muon.bash`。W&B online，run name 与实验编号一致。
- 依赖：等 `greedylore_tableiv_60m_bf16` 结束后，检查 CM003 `all_results.json` 中 `update_step=10000`，才启动 CM004；若前三组未全部完成，排队任务退出，不占用 GPU。
- 已创建 `greedylore_tableiv_60m_adam` tmux 排队会话；状态日志为 `outputs/CM004-m003-dense-adamw-llama60m-c4-adam_dense-lr2p5e3-bf16-s1243/queue.log`。当前无结果；完成后按最低 validation PPL 与 Muon 三组比较。

## 2026-09-25：前序任务失败与预算调整

- 前序 CM001 因 C4 数据耗尽及 W&B URL 接口报错而退出，CM002/CM003 未启动；AdamW 排队任务检查不到 CM003 结果，按设计退出，CM004 未启动。
- 用户将预算明确为约 1.1B token 位置（含 padding）。四组脚本及排队完成校验统一改为 8,393 更新步；AdamW LR 仍为 0.0025。数据补足方式待定。

## 2026-09-25：数据量纠正

- 前次 C4 提前耗尽的根因是训练包装器重复执行 worker 切分，现已修复；现有 30 个分片足够四组各自的 8,393 步预算。AdamW 继续排在三组 Muon 后运行。

## 2026-09-25：130M Dense AdamW 串行排队

- 实验编号：`CM008-m003-dense-adamw-llama130m-c4-adam_dense-lr2p5e3-bf16-s1243`。同 CM005–CM007 的 130M hidden 768、C4、seed、BF16、全局 batch 512、GA=1、20,000 步、warmup 2,000、cosine 到 10%、无 checkpoint、W&B online。
- Dense AdamW：LR 0.0025、betas (0.9, 0.999)、eps 1e-8、weight decay 0，梯度裁剪 1.0。该 LR 是用户此前从论文候选集合中选定的单次运行值；不能断言它是论文 130M 表项 24.73 使用的值。
- 在 `greedylore_tableiv_130m_bf16` 中排在三组 Muon 后，前序失败则不启动。

## 2026-09-26：130M CM008 原队列未启动后的补跑

- CM007 已完成 20,000 步且脚本成功结束，但原串行 tmux 会话在 CM008 创建输出目录前消失；中断原因未确认，CM008 此前没有训练记录。
- 用户确认 GPU 3–6 当前未使用并要求直接开始。经一次 GPU 占用检查，四张卡显存均约 3 MiB；09:06 由 `greedylore_tableiv_130m_followups` 会话使用原 `c4/scripts/run_table_iv_130m.bash adam_dense` 配置启动 CM008。已确认 torchrun 进程与 `outputs/CM008-m003-dense-adamw-llama130m-c4-adam_dense-lr2p5e3-bf16-s1243/train.log` 创建；最终结果尚未产生。
- 本次仍使用 W&B online 项目 `GreedyLore-TableIV`、20,000 步、BF16、GA=1、无 checkpoint。状态见 `outputs/CM033-CM034-greedylore-130m-followups/status.tsv`；CM008 完整结束后同一会话才尝试 CM033/CM034。

## 2026-09-29：完成结果核对

- CM004 Dense AdamW 完成修订后的 8,393 步，final C4 validation loss/PPL 为 3.50195 / 33.1799；CM008 补跑完成 20,000 步，为 3.16086 / 23.5908。来源是各自的 `outputs/<完整 run ID>/all_results.json`；日志均有 `Script finished successfully`，CM008 队列状态另记 exit code 0。
- 与同规模 Dense Muon 比较时必须保留优化器差异；本次 AdamW LR 0.0025 是单次选择，不能称为论文最佳 LR。全表和数据来源见 `docs/results.md`。

## 2026-10-02：350M Table V Dense AdamW 测速

- 新实验 CM060 使用仓库 `llama_350m.json`、C4、4×RTX 4090、BF16、sequence 256、每卡 batch 128、GA1、activation checkpointing、seed 1243；AdamW LR 0.001、betas (0.9,0.999)、eps 1e-8、weight decay 0、clip 1。LR 沿用仓库原 `c4/scripts/run_llama_pretraining_Adam.sh` 的 350M 值。
- 预热100次真实更新，测量更新101–600的连续500步，采用最慢rank连续墙钟时间/500。与CM061共用训练和计时配置，仅 compressor 不同；无评估、checkpoint或tracker。
- 入口只扩展新增的 `c4/table_v_timing.py` AdamW分支，不修改原训练入口、模型或通信hook。单元测试、dry-run、py_compile和四卡batch128短smoke通过；smoke只有2个测量步，不作速度结论。
- 正式串行队列已在tmux会话 `greedylore_table_v_adam350` 启动，CM060先运行，成功后自动运行CM061；状态与产物在 `outputs/CM060-CM061-table-v-adam-350m/`。
