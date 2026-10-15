# Table V：350M AdamW Dense / GreedyLore 测速约定

本实验用于检查Muon是否是GreedyLore测速变慢的重要因素。CM060为Dense AdamW，CM061为GreedyLore + AdamW；两臂只改变通信compressor。

- 模型/数据：仓库 `llama_350m.json`（实际367,969,280参数）、本地C4、T5 tokenizer、sequence256。
- 设备：4×RTX4090，NCCL SHM，禁用P2P；每卡batch128，GA1。
- 训练：BF16、activation checkpointing、AdamW LR0.001、betas(0.9,0.999)、eps1e-8、WD0、clip1。
- 计时：预热100步，连续测量更新101–600共500步；取最慢rank总墙钟时间/500。窗口计入数据、H2D、forward/backward、DDP hook、clipping、optimizer、scheduler、zero_grad和CUDA完成时间。
- GreedyLore：top_subspace、rank32、EF14、压缩从hook iter100开始、投影间隔200；测量窗口包含101/301/501三次投影更新。
- 限制：论文Table V没有明确dtype、checkpointing和测量起点；当前350M配置的hidden size 1024也不等于论文350M列的dmodel 768。当前top_subspace使用后续commit补全的评分实现。

控制器：`c4/scripts/run_table_v_adam_350m.py`。产物：`outputs/CM060-CM061-table-v-adam-350m/`。
