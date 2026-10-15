# GreedyLoRE Muon timing setting matrix（2026-10-03）

## 研究问题

在不修改 GreedyLoRE 原始通信 hook 的前提下，系统比较模型规模、每卡 batch、
world size、梯度 dtype、NCCL channel 和传输后端对 Dense/GreedyLoRE 端到端
训练更新时间的影响，寻找能够合理体现梯度压缩通信收益的设置。实验是 Muon 扩展，
不是论文 AdamW 数值复现，也不在短 timing 窗口内判断收敛质量。

## 共同配置

- 数据为本地英文 C4，序列长 256，seed 1243，无梯度累积。
- 模型为仓库的 60M、130M、350M、1B 配置；所有配置组完整运行四种规模。
- 每组包含 Dense Muon 和 GreedyLoRE + Muon 两臂，共 9×4×2=72 臂，编号
  CM062–CM133。
- Muon matrix/scalar LR 为 0.01/0.001，momentum 0.95，spectral-norm scaling，
  weight decay 0，gradient clipping 1。
- GreedyLoRE 使用现有 `top_subspace` 实现、rank 32、EF14、
  `start_compress_iter=100`、`update_proj_gap=200`、
  `min_compression_rate=1`、`beta_ef=0`、`error_inherit=0`。
- DDP `bucket_cap_mb=1024`；禁用 P2P。SHM 组启用 SHM；单 channel 组同时固定
  NCCL min/max channels 和 CTAs 为 1。
- 先执行 101 个不计时更新，第一次投影刷新包含在预热中；随后连续测量更新
  102–501，共 400 步，包含更新 301、501 的两次周期刷新。主指标为最慢 rank
  连续窗口总时间除以 400。
- 不运行评估、不保存 checkpoint、不连接 tracker。OOM、超时和其他失败保留日志，
  控制器继续下一臂，不自动缩小模型、batch 或 dtype。

## 配置组

| 组 | GPU | dtype | 每卡 batch | checkpointing | NCCL channel | transport |
| --- | ---: | --- | ---: | --- | --- | --- |
| `gl-paper-batch` | 4 | BF16 | 128；1B 为 64 | 开 | 默认 | SHM |
| `ws4-bf16-b1-default` | 4 | BF16 | 1 | 关 | 默认 | SHM |
| `ws4-fp32-b1-default` | 4 | FP32 | 1 | 关 | 默认 | SHM |
| `ws4-fp32-b1-ch1` | 4 | FP32 | 1 | 关 | 1 | SHM |
| `ws8-fp32-b1-default` | 8 | FP32 | 1 | 关 | 默认 | SHM |
| `ws8-fp32-b1-ch1` | 8 | FP32 | 1 | 关 | 1 | SHM |
| `ws8-bf16-b1-ch1` | 8 | BF16 | 1 | 关 | 1 | SHM |
| `ws8-fp32-b32-ch1` | 8 | FP32 | 32 | 关 | 1 | SHM |
| `ws8-fp32-b1-ch1-socket` | 8 | FP32 | 1 | 关 | 1 | Socket/loopback |

首组保留 GreedyLoRE Table V 的4卡、大 batch、BF16和 checkpointing 本地参照。
其余组默认关闭 checkpointing。Socket 组是带宽受限上界诊断；主要结论优先使用
SHM 组。

## 入口与产物

控制器为 `c4/scripts/run_table_v_muon_setting_matrix.py`，独立测速入口为
`c4/table_v_timing.py`。输出根目录为
`outputs/CM062-CM133-table-v-muon-setting-matrix/`，保存 manifest、每臂命令、
NCCL 环境、日志、逐步 timing、显存、状态和 Dense/GreedyLoRE 配对 speedup。

```bash
PYTHONPATH=. .venv/bin/python c4/scripts/run_table_v_muon_setting_matrix.py --dry-run
PYTHONPATH=. .venv/bin/python c4/scripts/run_table_v_muon_setting_matrix.py --gpus 0,1,2,3,4,5,6,7
```
