# M005：GreedyLore + AdamW Table V 测速

## 2026-10-02：350M Dense配对实验

- 目的：判断移除Muon正交化与额外AllGather后，350M GreedyLore相对Dense AdamW的端到端训练更新时间。实验编号为CM061，配对Dense为M003的CM060。
- 共同配置：仓库 `llama_350m.json`、C4、4×RTX 4090、BF16、sequence 256、每卡batch128、GA1、activation checkpointing、seed1243；AdamW LR0.001、betas(0.9,0.999)、eps1e-8、WD0、clip1、60000步cosine调度和6000步LR warmup。
- GreedyLore配置：`top_subspace`、rank32、EF14、`start_compress_iter=100`、投影间隔200、`min_compression_rate=1`、`beta_ef=0`、`error_inherit=0`。测量更新101–600，因此包括101/301/501三次投影更新。
- 实现约束：不修改原训练入口、模型或通信hook；只给新增测速入口增加AdamW选择，并由 `c4/scripts/run_table_v_adam_350m.py` 串行运行CM060/CM061。第一次commit中的top_subspace评分缺失问题仍然存在，本实验使用当前分支后来补全的评分实现，不能称作第一次commit的逐字节原实现。
- 验证：相关4项unittest、dry-run、py_compile和四卡350M batch128短smoke通过。GreedyLore smoke越过首次投影及第一个低秩步，无OOM或死锁；短窗口结果不用于方法比较。
- 正式CM060→CM061串行队列已在tmux会话 `greedylore_table_v_adam350` 启动；CM061只在CM060子进程结束后启动，失败也会保留状态和日志。
