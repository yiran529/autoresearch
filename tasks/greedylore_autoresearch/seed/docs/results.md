# GreedyLoRE + Muon 实验结果（2026-10-06 更新）

以下均为本仓库复现实验，**不是论文报告值**。每项只运行 seed 1243 一次，因此差异仅作初步观察。原始结果位于 `outputs/<完整 run ID>/all_results.json`，日志位于同目录的 `train.log`；完整 run ID 可由表中编号在 `outputs/` 下唯一定位。输出目录被 Git 忽略，本文保留可提交的结果摘要。

## C4 / LLaMA：GreedyLoRE Muon timing 设置矩阵（2026-10-03）

### 实验矩阵与计时口径

本轮是探索性系统实验，目标是在不修改 GreedyLoRE 原始通信 hook 的前提下，比较模型规模、每卡 batch、world size、梯度 dtype、NCCL channel 和通信后端对端到端训练更新时间的影响。结果不是 GreedyLoRE 论文 AdamW timing 的数值复现，也不评价短窗口内的收敛质量。

共同设置为本地英文 C4、序列长 256、seed 1243、Muon matrix/scalar LR 0.01/0.001、DDP `bucket_cap_mb=1024`、GreedyLoRE `top_subspace` rank 32、EF14、压缩起点 100、投影间隔 200。各臂先完成 101 个不计时更新，再连续测量更新 102–501 共 400 步；GreedyLoRE 窗口包含更新 301 和 501 的两次周期投影刷新。表中每步时间为最慢 rank 的连续窗口总墙钟时间除以 400，完整计入数据读取、forward/backward、DDP hook、梯度裁剪、Muon step、scheduler、zero_grad 和逐步 CUDA 同步。

`speedup = Dense 时间 / GreedyLoRE 时间`，大于 1 表示 GreedyLoRE 更快；耗时变化为 `(Dense−GreedyLoRE)/Dense`，正值表示节省。所有运行均为 seed 1243 的一次独立进程轨迹，400个更新步不是400次独立重复。

### 完整结果

| 配置组 | 模型 | GPU | dtype | batch/GPU | checkpoint | channel | 后端 | Dense ID | Dense (s/step) | GreedyLoRE ID | GreedyLoRE (s/step) | speedup | 耗时变化 |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| `gl-paper-batch` | 60M | 4 | BF16 | 128 | 开 | default | SHM | CM062 | 0.413976 | CM063 | 0.463586 | 0.8930× | -11.98% |
| `gl-paper-batch` | 130M | 4 | BF16 | 128 | 开 | default | SHM | CM064 | 0.475738 | CM065 | 0.493516 | 0.9640× | -3.74% |
| `gl-paper-batch` | 350M | 4 | BF16 | 128 | 开 | default | SHM | CM066 | 1.309372 | CM067 | 1.355402 | 0.9660× | -3.52% |
| `gl-paper-batch` | 1B | 4 | BF16 | 64 | 开 | default | SHM | CM068 | 2.213754 | CM069 | OOM | — | — |
| `ws4-bf16-b1-default` | 60M | 4 | BF16 | 1 | 关 | default | SHM | CM070 | 0.190665 | CM071 | 0.248289 | 0.7679× | -30.22% |
| `ws4-bf16-b1-default` | 130M | 4 | BF16 | 1 | 关 | default | SHM | CM072 | 0.084836 | CM073 | 0.102599 | 0.8269× | -20.94% |
| `ws4-bf16-b1-default` | 350M | 4 | BF16 | 1 | 关 | default | SHM | CM074 | 0.213076 | CM075 | 0.264482 | 0.8056× | -24.13% |
| `ws4-bf16-b1-default` | 1B | 4 | BF16 | 1 | 关 | default | SHM | CM076 | 0.649694 | CM077 | OOM | — | — |
| `ws4-fp32-b1-default` | 60M | 4 | FP32 | 1 | 关 | default | SHM | CM078 | 0.202492 | CM079 | 0.246487 | 0.8215× | -21.73% |
| `ws4-fp32-b1-default` | 130M | 4 | FP32 | 1 | 关 | default | SHM | CM080 | 0.115384 | CM081 | 0.128763 | 0.8961× | -11.59% |
| `ws4-fp32-b1-default` | 350M | 4 | FP32 | 1 | 关 | default | SHM | CM082 | 0.289404 | CM083 | 0.346010 | 0.8364× | -19.56% |
| `ws4-fp32-b1-default` | 1B | 4 | FP32 | 1 | 关 | default | SHM | CM084 | OOM | CM085 | OOM | — | — |
| `ws4-fp32-b1-ch1` | 60M | 4 | FP32 | 1 | 关 | one | SHM | CM086 | 0.206166 | CM087 | 0.250983 | 0.8214× | -21.74% |
| `ws4-fp32-b1-ch1` | 130M | 4 | FP32 | 1 | 关 | one | SHM | CM088 | 0.119089 | CM089 | 0.129550 | 0.9192× | -8.78% |
| `ws4-fp32-b1-ch1` | 350M | 4 | FP32 | 1 | 关 | one | SHM | CM090 | 0.302981 | CM091 | 0.352453 | 0.8596× | -16.33% |
| `ws4-fp32-b1-ch1` | 1B | 4 | FP32 | 1 | 关 | one | SHM | CM092 | OOM | CM093 | OOM | — | — |
| `ws8-fp32-b1-default` | 60M | 8 | FP32 | 1 | 关 | default | SHM | CM094 | 0.211907 | CM095 | 0.255914 | 0.8280× | -20.77% |
| `ws8-fp32-b1-default` | 130M | 8 | FP32 | 1 | 关 | default | SHM | CM096 | 0.132877 | CM097 | 0.129682 | 1.0246× | +2.40% |
| `ws8-fp32-b1-default` | 350M | 8 | FP32 | 1 | 关 | default | SHM | CM098 | 0.323979 | CM099 | 0.313733 | 1.0327× | +3.16% |
| `ws8-fp32-b1-default` | 1B | 8 | FP32 | 1 | 关 | default | SHM | CM100 | OOM | CM101 | OOM | — | — |
| `ws8-fp32-b1-ch1` | 60M | 8 | FP32 | 1 | 关 | one | SHM | CM102 | 0.220575 | CM103 | 0.256064 | 0.8614× | -16.09% |
| `ws8-fp32-b1-ch1` | 130M | 8 | FP32 | 1 | 关 | one | SHM | CM104 | 0.136509 | CM105 | 0.133722 | 1.0208× | +2.04% |
| `ws8-fp32-b1-ch1` | 350M | 8 | FP32 | 1 | 关 | one | SHM | CM106 | 0.342953 | CM107 | 0.319509 | 1.0734× | +6.84% |
| `ws8-fp32-b1-ch1` | 1B | 8 | FP32 | 1 | 关 | one | SHM | CM108 | OOM | CM109 | OOM | — | — |
| `ws8-bf16-b1-ch1` | 60M | 8 | BF16 | 1 | 关 | one | SHM | CM110 | 0.202079 | CM111 | 0.253615 | 0.7968× | -25.50% |
| `ws8-bf16-b1-ch1` | 130M | 8 | BF16 | 1 | 关 | one | SHM | CM112 | 0.096831 | CM113 | 0.111335 | 0.8697× | -14.98% |
| `ws8-bf16-b1-ch1` | 350M | 8 | BF16 | 1 | 关 | one | SHM | CM114 | 0.238803 | CM115 | 0.270175 | 0.8839× | -13.14% |
| `ws8-bf16-b1-ch1` | 1B | 8 | BF16 | 1 | 关 | one | SHM | CM116 | 0.729854 | CM117 | OOM | — | — |
| `ws8-fp32-b32-ch1` | 60M | 8 | FP32 | 32 | 关 | one | SHM | CM118 | 0.258105 | CM119 | 0.299888 | 0.8607× | -16.19% |
| `ws8-fp32-b32-ch1` | 130M | 8 | FP32 | 32 | 关 | one | SHM | CM120 | 0.292990 | CM121 | 0.290429 | 1.0088× | +0.87% |
| `ws8-fp32-b32-ch1` | 350M | 8 | FP32 | 32 | 关 | one | SHM | CM122 | OOM | CM123 | OOM | — | — |
| `ws8-fp32-b32-ch1` | 1B | 8 | FP32 | 32 | 关 | one | SHM | CM124 | OOM | CM125 | OOM | — | — |
| `ws8-fp32-b1-ch1-socket` | 60M | 8 | FP32 | 1 | 关 | one | SOCKET | CM126 | 0.400752 | CM127 | 0.352109 | 1.1381× | +12.14% |
| `ws8-fp32-b1-ch1-socket` | 130M | 8 | FP32 | 1 | 关 | one | SOCKET | CM128 | 0.566923 | CM129 | 0.345921 | 1.6389× | +38.98% |
| `ws8-fp32-b1-ch1-socket` | 350M | 8 | FP32 | 1 | 关 | one | SOCKET | CM130 | 1.554621 | CM131 | 0.743417 | 2.0912× | +52.18% |
| `ws8-fp32-b1-ch1-socket` | 1B | 8 | FP32 | 1 | 关 | one | SOCKET | CM132 | OOM | CM133 | OOM | — | — |

完成状态：72个实验臂中55个完成、17个失败，共形成26组完整配对。逐一核对17个失败臂的日志，均观察到 `CUDA out of memory`；表中因此标为 OOM。1B 的部分 BF16 Dense 臂完成，但对应 GreedyLoRE 臂 OOM，不能形成方法比较。

### 结论

- 在实际 SHM 后端中，最有利且完整的结果是350M、8 GPU、FP32、每卡 batch 1、单 channel：Dense/GreedyLoRE 分别为0.342953/0.319509 s，speedup为 **1.0734×**，端到端耗时降低 **6.84%**。相同配置使用默认 channel 时为1.0327×、降低3.16%。
- 同一350M单 channel设置改为BF16后，GreedyLoRE慢13.14%；4 GPU FP32 batch 1时也仍慢16.33%。这说明本轮观察到的SHM优势依赖8 GPU、FP32和低计算负载的组合，不能外推为所有训练设置均有优势。
- GreedyLoRE论文大 batch 本地参照中，60M/130M/350M分别慢11.98%/3.74%/3.52%。每卡batch 32时，60M慢16.19%，130M仅快0.87%，350M和1B双臂OOM。
- Socket/loopback带宽受限诊断中，60M/130M/350M的speedup分别为1.1381×/1.6389×/2.0912×，端到端耗时降低12.14%/38.98%/52.18%。该组用于显示通信受限上界，不作为常规本机SHM训练结论。
- 所有设置都只有一次独立运行，且配置是在查看先前 ARC-TopK 与 GreedyLoRE timing 后选择的探索性矩阵。6.84%的SHM优势是候选结果，仍需对固定配置做独立重复后才能作为稳健结论。

### 结果来源

- 运行代码版本：`d008aa5978b7531fce465ec2ef1991c1ab3be11e`。
- 完成状态、身份和主指标：`outputs/CM062-CM133-table-v-muon-setting-matrix/summary.json`。
- 每个完成臂的连续窗口、逐步时间和最慢 rank：对应 run 目录的 `all_results.json`。
- 失败原因：输出根目录下对应 `<run_id>.log`；17个失败均明确记录 CUDA OOM。
- 协议与设置：`docs/table-v-muon-setting-matrix.md`；控制器：`c4/scripts/run_table_v_muon_setting_matrix.py`。

## C4 / LLaMA：严格阻塞通信矩阵（CM188–CM243）

### 实验矩阵与口径

本轮固定本地英文C4、sequence256、每卡batch1、GA1、关闭activation checkpointing、
seed1243、Muon matrix/scalar LR 0.01/0.001，以及GreedyLoRE `top_subspace` rank32、
EF14、压缩起点100、投影间隔200。DDP `bucket_cap_mb=8192`，45个完成臂在测量
窗口中均实际记录为单bucket。每臂预热101步并连续测量更新102–501，共400步；每个
配置与模型只运行一次。

严格阻塞包装在每个DDP hook入口先CUDA同步，等待原hook Future完成后再次同步。
`iter`为完整训练步时间；`hook`包含Dense All-Reduce，或GreedyLoRE的评分、压缩、
collective、解压和投影刷新，不是纯NCCL wire time。`speedup = Dense / GreedyLoRE`，
大于1表示GreedyLoRE更快。Socket组同时使用单channel、关闭SHM并强制loopback
Socket，只作为通信受限诊断，不能将差异解释为纯channel效应。

### 完整结果

| 配置 | 模型 | GPU | dtype | channel/后端 | Dense ID | Dense iter (s) | Dense hook (s) | GreedyLoRE ID | GL iter (s) | GL hook (s) | iter speedup | hook speedup |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `ws4-fp32-default` | 60M | 4 | FP32 | default/SHM | CM188 | 0.210814 | 0.030056 | CM189 | 0.249546 | 0.066688 | 0.8448× | 0.4507× |
| `ws4-fp32-default` | 130M | 4 | FP32 | default/SHM | CM191 | 0.116889 | 0.052309 | CM190 | 0.127061 | 0.064700 | 0.9199× | 0.8085× |
| `ws4-fp32-default` | 350M | 4 | FP32 | default/SHM | CM192 | 0.298156 | 0.140076 | CM193 | 0.344055 | 0.185251 | 0.8666× | 0.7561× |
| `ws4-fp32-default` | 1B | 4 | FP32 | default/SHM | CM195 | OOM | OOM | CM194 | OOM | OOM | — | — |
| `ws4-fp32-ch1` | 60M | 4 | FP32 | one/SHM | CM196 | 0.216180 | 0.036293 | CM197 | 0.249620 | 0.066595 | 0.8660× | 0.5450× |
| `ws4-fp32-ch1` | 130M | 4 | FP32 | one/SHM | CM199 | 0.119475 | 0.057418 | CM198 | 0.131415 | 0.068621 | 0.9091× | 0.8367× |
| `ws4-fp32-ch1` | 350M | 4 | FP32 | one/SHM | CM200 | 0.316510 | 0.154594 | CM201 | 0.353045 | 0.192245 | 0.8965× | 0.8042× |
| `ws4-fp32-ch1` | 1B | 4 | FP32 | one/SHM | CM203 | OOM | OOM | CM202 | OOM | OOM | — | — |
| `ws8-fp32-default` | 60M | 8 | FP32 | default/SHM | CM204 | 0.218905 | 0.042892 | CM205 | 0.267620 | 0.087943 | 0.8180× | 0.4877× |
| `ws8-fp32-default` | 130M | 8 | FP32 | default/SHM | CM207 | 0.131965 | 0.067157 | CM206 | 0.130764 | 0.066431 | 1.0092× | 1.0109× |
| `ws8-fp32-default` | 350M | 8 | FP32 | default/SHM | CM208 | 0.335162 | 0.180162 | CM209 | 0.310899 | 0.157728 | 1.0780× | 1.1422× |
| `ws8-fp32-default` | 1B | 8 | FP32 | default/SHM | CM211 | OOM | OOM | CM210 | OOM | OOM | — | — |
| `ws8-fp32-ch1` | 60M | 8 | FP32 | one/SHM | CM212 | 0.229443 | 0.049234 | CM213 | 0.259736 | 0.076440 | 0.8834× | 0.6441× |
| `ws8-fp32-ch1` | 130M | 8 | FP32 | one/SHM | CM215 | 0.139529 | 0.073461 | CM214 | 0.134929 | 0.069078 | 1.0341× | 1.0634× |
| `ws8-fp32-ch1` | 350M | 8 | FP32 | one/SHM | CM216 | 0.351927 | 0.194516 | CM217 | 0.323101 | 0.164549 | 1.0892× | 1.1821× |
| `ws8-fp32-ch1` | 1B | 8 | FP32 | one/SHM | CM219 | OOM | OOM | CM218 | OOM | OOM | — | — |
| `ws8-bf16-default` | 60M | 8 | BF16 | default/SHM | CM220 | 0.211010 | 0.030079 | CM221 | 0.251648 | 0.070355 | 0.8385× | 0.4275× |
| `ws8-bf16-default` | 130M | 8 | BF16 | default/SHM | CM223 | 0.095222 | 0.034710 | CM222 | 0.109867 | 0.048421 | 0.8667× | 0.7168× |
| `ws8-bf16-default` | 350M | 8 | BF16 | default/SHM | CM224 | 0.234273 | 0.092570 | CM225 | 0.261808 | 0.123038 | 0.8948× | 0.7524× |
| `ws8-bf16-default` | 1B | 8 | BF16 | default/SHM | CM227 | 0.705021 | 0.321028 | CM226 | OOM | OOM | — | — |
| `ws8-bf16-ch1` | 60M | 8 | BF16 | one/SHM | CM228 | 0.209152 | 0.025031 | CM229 | 0.266598 | 0.079472 | 0.7845× | 0.3150× |
| `ws8-bf16-ch1` | 130M | 8 | BF16 | one/SHM | CM231 | 0.098735 | 0.036985 | CM230 | 0.111732 | 0.048398 | 0.8837× | 0.7642× |
| `ws8-bf16-ch1` | 350M | 8 | BF16 | one/SHM | CM232 | 0.248282 | 0.102471 | CM233 | 0.270685 | 0.125408 | 0.9172× | 0.8171× |
| `ws8-bf16-ch1` | 1B | 8 | BF16 | one/SHM | CM235 | 0.740529 | 0.343012 | CM234 | OOM | OOM | — | — |
| `ws8-bf16-socket` | 60M | 8 | BF16 | one/Socket | CM236 | 0.317298 | 0.106822 | CM237 | 0.314205 | 0.100043 | 1.0098× | 1.0678× |
| `ws8-bf16-socket` | 130M | 8 | BF16 | one/Socket | CM239 | 0.345583 | 0.219560 | CM238 | 0.249138 | 0.124254 | 1.3871× | 1.7670× |
| `ws8-bf16-socket` | 350M | 8 | BF16 | one/Socket | CM240 | 0.960729 | 0.606254 | CM241 | 0.587861 | 0.234025 | 1.6343× | 2.5906× |
| `ws8-bf16-socket` | 1B | 8 | BF16 | one/Socket | CM243 | 3.373184 | 2.158416 | CM242 | OOM | OOM | — | — |

完成状态：56个实验臂中45个完成、11个失败，形成21组完整配对。逐一核对11个
失败臂日志，均明确记录CUDA OOM。全部FP32 1B双臂OOM；8卡BF16的三个Dense 1B
完成，但对应GreedyLoRE均OOM，不能计算配对speedup。

### 结论

- SHM中只有8卡FP32的130M/350M出现完整配对加速。350M默认channel的iter/hook
  speedup为1.0780×/1.1422×，单channel为1.0892×/1.1821×；130M的增益较小。
- 4卡FP32的60M–350M以及8卡BF16 SHM的60M–350M均未显示压缩加速。减少channel
  会缩小部分FP32差距，但BF16下Dense通信字节减半，GreedyLoRE固定计算开销仍占主导。
- 8卡BF16 Socket的60M/130M/350M iter speedup分别为1.0098×/1.3871×/1.6343×，
  hook speedup为1.0678×/1.7670×/2.5906×。该结果展示通信受限上界，不代表常规SHM。
- 严格阻塞和单bucket有意取消反向计算与通信的外部重叠，因此结果支持“串行通信路径”
  的比较，不能直接外推为正常异步DDP吞吐。每臂仅seed1243一次，属于探索性系统结果。

### 正常异步 DDP bucket sweep（CM244–CM247）

为检验bucket大小对正常计算/通信overlap的影响，固定350M、8卡、FP32、每卡batch1、
默认channel/SHM和相同Muon/GreedyLoRE参数，只改变DDP bucket cap。该组不使用严格
阻塞；被动包装器只记录bucket布局并直接返回原hook Future。主指标为完整训练步时间，
不报告不可与严格阻塞口径等同的hook时间。

| Bucket cap | 实际bucket数 | Dense ID | Dense iter (s) | GreedyLoRE ID | GL iter (s) | iter speedup | 耗时变化 |
| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| 256 MiB | 5 | CM244 | 0.301390 | CM245 | 0.314311 | 0.9589× | -4.29% |
| 1024 MiB | 2 | CM246 | 0.324152 | CM247 | 0.315635 | 1.0270× | +2.63% |

256 MiB组实际bucket字节数为269,705,216、275,820,544、274,440,192、
268,808,192、383,102,976；1024 MiB组为1,076,191,232、395,685,888。Cap是软上限，
参数张量不拆分，因此实际bucket可略超上限。从1024 MiB减至256 MiB时，Dense每步时间
降低约7.02%，GreedyLoRE只降低约0.42%；小bucket主要帮助Dense把All-Reduce与剩余
backward重叠，GreedyLoRE的压缩与更多小collective抵消了大部分收益。两档均只有一次
运行；1024 MiB的1.0270×与先前同类异步结果约1.0327×接近，但2.63%仍是初步差异。

### 结果来源

- 运行代码版本：`9212f42c9acd629f73be7d4902f77ed91d0ac537`。
- 身份、完成状态与汇总指标：
  `outputs/CM188-CM243-table-v-muon-strict-blocking-system-matrix/summary.json`。
- 每个成功臂的连续窗口、逐步时间、hook时间与bucket布局：对应run目录的
  `all_results.json`；45个成功文件均核对`measured_steps=400`、
  `strict_blocking_communication=true`和summary数值一致。
- 失败原因：输出根目录下对应`<run_id>.log`，11项均确认CUDA OOM。
- 协议与设置：`docs/table-v-muon-strict-blocking.md`；控制器：
  `c4/scripts/run_table_v_muon_strict_blocking.py --matrix system`。
- Bucket sweep代码版本：`2d413db6f454d3cf64742f7a543b9e3622978ce5`；完成状态、
  iteration指标和实际bucket布局来自
  `outputs/CM244-CM247-table-v-muon-bucket-sweep/summary.json`及各run的
  `all_results.json`。4项均核对`measured_steps=400`、
  `strict_blocking_communication=false`、`bucket_layout_recorded=true`，队列状态为完成。

## C4 / LLaMA 350M：Table IV Muon 适配（CM250–CM252）

### 实验矩阵与核心设置

| 实验 | 模型 / 方法 | 完成预算 | 矩阵 LR |
| --- | --- | ---: | ---: |
| CM250 | LLaMA 350M / Dense Muon | 60,000 更新步 | 0.005 |
| CM251 | LLaMA 350M / GreedyLoRE + Muon r32 | 60,000 更新步 | 0.005 |
| CM252 | LLaMA 350M / GreedyLoRE + Muon r256 | 60,000 更新步 | 0.005 |

三组均使用本地 `c4/configs/llama_350m.json`、已有的 `c4/c4_en` C4 数据（50 个 train、8 个 validation 分片）、4 张 RTX 4090、BF16、activation checkpointing、序列长 256、每卡 batch 128、全局 batch 512、seed 1243。训练数据按固定顺序重复以达到 60,000 步；warmup 6,000 步，cosine 调度至峰值 LR 的 10%，weight decay 0，梯度裁剪 1.0。Muon momentum 0.95、spectral-norm scaling，scalar AdamW LR 0.001。CM251/CM252 使用 `top_subspace`、EF14、rank 32/256、压缩起点 1,000、投影间隔 200。运行入口为 `c4/run_llama_pretraining.py`，队列为 `c4/scripts/queue_table_iv_350m_muon.py`；CM250/CM251 从头并行训练，完成后 CM252 从头训练，均未保存 checkpoint。

共同矩阵 LR 由先行的 Dense CM248（LR 0.01）和 CM249（LR 0.005）在第 10,000 步的 final validation loss 选择：分别为 3.23101 和 3.21210，故选择 0.005。三组正式训练使用同一 validation 数据评估，属于探索性适配，并非论文 AdamW 结果的数值复现。

### 结果

| 实验 | best validation PPL（步） | final validation loss | final validation PPL（第 60,000 步） |
| --- | ---: | ---: | ---: |
| CM250 | 17.37389（60,000） | 2.85497 | 17.37389 |
| CM251 | 18.39514（59,000） | 2.91213 | 18.39587 |
| CM252 | 17.48532（59,000） | 2.86140 | 17.48608 |

best 是第 1 步、每 1,000 步及训练结束后的 validation 评估中 loss 最低的一次。CM251/CM252 在第 59,000 步达到 best，训练结束后的 final 评估略高；三组 final 评估均覆盖 10,048,075 个有效 token。`all_results.json` 的 `update_step` 均为 60,000，队列 `status.tsv` 均记为 `completed`，对应 `train.log` 均有 `Script finished successfully`。

### 结论与来源

- 同口径 final validation 下，CM251 比 CM250 的 loss 高 0.05716，PPL 高 1.02198（5.88%）。按各自 best 比较，CM251 的 PPL 高 1.02124；两个 best 所在步数不同。
- 同口径 final validation 下，CM252 比 CM250 的 loss 高 0.00644，PPL 高 0.11218（0.65%）。按各自 best 比较，CM252 的 PPL 高 0.11143；CM252 的 best 在第 59,000 步，CM250 在第 60,000 步。CM252 的 final PPL 低于 CM251，但尚不能据此推断 rank 效应稳定。
- 三组各只有 seed 1243 一次完整训练（n=1），LR 选择与正式比较复用 validation，不能据此推断稳定差异或统计等效。重复现有训练数据也使本轮数据曝光与论文协议不完全一致。
- 数值与步数来自 `outputs/CM250-m001-dense-muon-llama350m-c4-dense-formal-lr0p005-bf16-s1243/all_results.json`、`outputs/CM251-m002-greedylore-muon-llama350m-c4-r32-formal-lr0p005-bf16-s1243/all_results.json` 和 `outputs/CM252-m002-greedylore-muon-llama350m-c4-r256-formal-lr0p005-bf16-s1243/all_results.json` 的 `best_eval_ppl`、`best_eval_step`、`final_eval_loss`、`final_eval_ppl`、`final_eval_tokens`、`update_step`；运行设置来自同目录的 `command.txt`、`protocol.json`，LR 选择和完成状态来自 `outputs/CM248-CM252-table-iv-350m-muon-existing-data/selection.json`、`status.tsv`，成功结束由各自 `train.log` 核对。

## C4 / LLaMA：Table IV 相关实验

### 核心设置与结果

本地 C4、4 GPU（3–6）、BF16、序列长 256、全局 batch 512、seed 1243、cosine 调度、weight decay 0，均未保存 checkpoint。60M 使用 `c4/scripts/run_table_iv_60m_muon.bash`，预算为修订后的 8,393 步、warmup 1,000；130M 使用 `c4/scripts/run_table_iv_130m.bash`，预算 20,000 步、warmup 2,000。两种模型规模不能直接比较。Muon 矩阵 LR 0.02、momentum 0.95、spectral-norm scaling，scalar AdamW LR 0.001；Dense AdamW LR 0.0025。压缩组使用 `top_subspace`、EF14、投影间隔 200；默认从第 1,000 步开始，CM034 改为第 2,000 步。

| 实验 | 模型 | 方法 | 完成更新步 | final validation loss | final validation PPL |
| --- | --- | --- | ---: | ---: | ---: |
| CM001 `-rerun1` | LLaMA 60M | Dense Muon | 8,393 | 3.61430 | 37.1255 |
| CM002 | LLaMA 60M | GreedyLoRE + Muon r32 | 8,393 | 3.70326 | 40.5794 |
| CM003 | LLaMA 60M | GreedyLoRE + Muon r128 | 8,393 | 3.62424 | 37.4963 |
| CM004 | LLaMA 60M | Dense AdamW | 8,393 | 3.50195 | 33.1799 |
| CM005 | LLaMA 130M | Dense Muon | 20,000 | 3.29619 | 27.0095 |
| CM006 | LLaMA 130M | GreedyLoRE + Muon r32 | 20,000 | 3.59626 | 36.4617 |
| CM007 | LLaMA 130M | GreedyLoRE + Muon r256 | 20,000 | 3.36230 | 28.8555 |
| CM008 | LLaMA 130M | Dense AdamW | 20,000 | 3.16086 | 23.5908 |
| CM033 | LLaMA 130M | GreedyLoRE + Muon r64 | 20,000 | 3.51393 | 33.5800 |
| CM034 | LLaMA 130M | GreedyLoRE + Muon r32，第 2,000 步开始压缩 | 20,000 | 3.53616 | 34.3349 |

表中数值直接来自各 run 的 `all_results.json` 字段 `update_step`、`final_eval_loss`、`final_eval_ppl`；入口在完成目标步数并再次评估后写入该文件，日志均有 `Script finished successfully`。60M 的 8,393 步约为 1.1B 个含 padding 的 token 位置，与论文附录的 10,000 步不同。最初的 CM001（无 `-rerun1`）只运行到 5,220 步，不纳入表格。CM008 是原队列未启动后补跑的同配置实验；CM033/CM034 是在查看 CM006 结果后设计的探索性尝试。

在相同 130M Muon 设置下，CM006、CM007、CM033、CM034 的 final PPL 分别比 Dense Muon CM005 高 9.4522、1.8460、6.5705、7.3254。CM034 比同为 r32 的 CM006 低 2.1267，但它使用同一 validation 数据进行探索，不是独立确认。Dense AdamW 使用不同优化器，不属于仅改变压缩方式的 Muon 对照。

## C4 / LLaMA 130M：最近的学习率选择与压缩比较（2026-10-02 核对）

### 实验矩阵与核心设置

CM040–CM043 均完成 20,000 更新步，模型配置为 `c4/configs/llama_130m.json`。本轮使用 C4 `c4/c4_en`；运行配置记录的数据路径与早期 CM005–CM008、CM033–CM034 的 `/dev/shm/wyr_tmp/c4` 不同，因此不将跨批次差异单独归因于学习率。所有指标均为本地复现的 **final validation**，不是论文结果。

共同设置：4 张 NVIDIA GeForce RTX 4090（GPU 0–3）、BF16、序列长 256、每卡 batch 128、全局 batch 512、GA=1、seed 1243；warmup 2,000 步、cosine 到峰值 LR 的 10%、weight decay 0、梯度裁剪 1.0。Muon momentum 0.95、spectral-norm scaling，scalar AdamW LR 0.001。压缩组采用 `top_subspace`、EF14、从第 1,000 次通信迭代开始压缩、投影间隔 200，秩分别为 32/256。运行入口为 `c4/scripts/run_table_iv_130m.bash`，队列为 `c4/scripts/queue_table_iv_130m_muon_lr.py`；未保存 checkpoint。

### 结果

| 实验 | 模型 / 方法 | 矩阵 LR | 完成更新步 | final validation loss | final validation PPL |
| --- | --- | ---: | ---: | ---: | ---: |
| CM040 | LLaMA 130M / Dense Muon | 0.01 | 20,000 | 3.13209 | 22.9218 |
| CM041 | LLaMA 130M / Dense Muon | 0.005 | 20,000 | 3.14294 | 23.1718 |
| CM042 | LLaMA 130M / GreedyLoRE + Muon r32 | 0.01 | 20,000 | 3.20127 | 24.5636 |
| CM043 | LLaMA 130M / GreedyLoRE + Muon r256 | 0.01 | 20,000 | 3.13895 | 23.0797 |

每个编号在 `outputs/` 下唯一对应一个完整 run ID。表中步数和指标来自该目录 `all_results.json` 的 `update_step`、`final_eval_loss`、`final_eval_ppl`，并与 `train.log` 的 `Final eval loss`、`Script finished successfully` 交叉核对。共同设置以本地运行配置 `wandb/run-*/files/config.yaml` 和启动脚本核对；实际 GPU 编号及完成状态来自 `outputs/CM040-CM043-muon-matrix-lr/status.tsv`，GPU 型号来自本地 `wandb-metadata.json`。

### 结论与未完成尝试

- 先比较 CM040/CM041 在第 20,000 步的 final validation loss，再将较低者的矩阵 LR 用于 CM042/CM043；实际选中 0.01。CM040 比 CM041 的 loss 低 0.01085、PPL 低 0.2500。这是两候选的 validation 选择结果，不代表全局最优 LR。
- 同为矩阵 LR 0.01 时，r32 相对 Dense 的 loss 高 0.06918、PPL 高 1.6418（7.16%）；r256 的 loss 高 0.00686、PPL 高 0.1579（0.69%）。本轮 r256 的验证指标更接近 Dense，尚不能据此判断总通信收益。
- 每组只有 seed 1243 一次运行（n=1），学习率选择与比较使用同一 validation，属于探索性结果；下一步需固定配置，用预先约定的多 seed 重复确认。
- 先前 r64 sweep 的 CM036（矩阵 LR 0.004 / scalar LR 0.001）第 5,000 步 validation loss 为 3.48792，低于筛选门槛 CM033 的 3.74357，继续运行后在第 12,000 步评估完被用户终止；该步 loss 为 3.23633。两项来源均为 `outputs/CM036-m002-greedylore-muon-llama130m-c4-r64-lr0p004-slr0p001-bf16-s1243/train.log` 的 `Eval loss at step <step>`，不是 final。该目录没有 `all_results.json` 或成功结束标记；状态文件 `outputs/CM036-CM039-r64-lr-sweep/status.tsv` 记为 `interrupted`。CM037–CM039 记为取消且未启动。CM033 与 CM036 的数据路径不同，筛选门槛只作探索参考。

## CIFAR / ResNet-18：Figure 3 协议下的 Muon 比较

### 实验矩阵

| 实验 | 数据集 | 模型 / 方法 | 完成预算 |
| --- | --- | --- | --- |
| CM044 | CIFAR-10 | ResNet-18 / Dense Muon | 40 epochs / 15,640 更新步 |
| CM045 | CIFAR-10 | ResNet-18 / GreedyLoRE + Muon r64 | 40 epochs / 15,640 更新步 |
| CM046 | CIFAR-100 | ResNet-18 / Dense Muon | 40 epochs / 15,640 更新步 |
| CM047 | CIFAR-100 | ResNet-18 / GreedyLoRE + Muon r64 | 40 epochs / 15,640 更新步 |

### 核心设置

每个数据集使用官方 50,000 张 train / 10,000 张 test。共同设置：4 张 RTX 4090、每卡 batch 32、全局 batch 128、seed 1243、FP32 模型和梯度、cosine 按 epoch 更新至 0，无学习率 warmup，不保存 checkpoint；Muon 的 Polar Express 正交化内部使用 BF16。CIFAR-10 使用 GPU 0–3，CIFAR-100 使用 GPU 4–7，各自先 Dense 后 GreedyLoRE。

Muon 矩阵 LR **0.02**、momentum 0.95、Nesterov、spectral-norm scaling；scalar / 分类头 AdamW LR 在 CIFAR-10/100 上分别为 **0.005 / 0.0005**，betas=(0.9,0.999)、epsilon=1e-8。Weight decay 为 5e-4，bias 和 BatchNorm 参数为 0。两种数据集沿用原 CIFAR 入口的随机裁剪、水平翻转和 normalization，两方法臂保持一致。

压缩组使用 `top_subspace`、rank 64、EF14、`min_compression_rate=2`、`beta_ef=0`、`error_inherit=0`；CIFAR-10/100 分别在前 500/4000 次 dense 更新后启用，子空间更新间隔分别为 750/1200。首次子空间刷新仍为 dense 聚合，第一次低秩传输是第 502/4002 次更新。卷积梯度展平后压缩，分类头与一维参数保持 dense；压缩发生在 Muon 动量和正交化之前，因此属于近似 Muon。

训练预算、batch、rank、压缩起点和子空间间隔参照 GreedyLoRE Figure 3 / 附录 F；Muon 参数、精度、seed、数据处理及未明确的调度细节是本次适配，结果不是论文 AdamW 数值的直接复现。运行入口为 `pytorch-cifar/train_ddp.py`，配置脚本为 `pytorch-cifar/run_paper_muon.bash`，队列为 `pytorch-cifar/queue_paper_muon.bash`。W&B online 按数据集使用独立 project。

### 结果

| 实验 | final test accuracy (%) | final test loss |
| --- | ---: | ---: |
| CM044 | 94.43 | 0.31748 |
| CM045 | 94.29 | 0.31015 |
| CM046 | 75.91 | 1.27233 |
| CM047 | 74.94 | 1.27259 |

表中指标均为第 **40 epoch / 15,640 步**、完整 **10,000 张 test** 的最终值，不是 best。来源为各 run 的 `all_results.json` 字段 `final_test_accuracy`、`final_test_loss`，与 `metrics.jsonl` 最后一行交叉核对；同时核对 `status=completed`、`epoch`、`update_step`、`test_samples` 和日志 `TRAINING_COMPLETED`。完整 run ID 在 `outputs/` 下按实验编号及方法唯一定位，设置来自各 run 的 `config.json`；两个队列状态均为 `queue_finished`。

### 结论

- CIFAR-10 的 GreedyLoRE 最终 accuracy 比 Dense 低 **0.14 个百分点**，test loss 低 0.00733；CIFAR-100 的 accuracy 低 **0.97 个百分点**，test loss 高 0.00026。
- 本轮 CIFAR-10 的压缩组更接近 Dense，CIFAR-100 的准确率损失更明显。每组只有 seed 1243 一次运行（n=1），未进行 CIFAR 学习率搜索，不能据此声称统计等效或稳定差异。

## GLUE / RoBERTa-base：Table III 相关实验

本地 GLUE Arrow 数据、RoBERTa-base、4 GPU（0、1、2、7）、FP32、每卡 batch 4、10 epoch、seed 1243、cosine 调度和 10% warmup；入口为 `glue/scripts/run_table_iii_muon.bash`。Dense 与 r8/r16 使用相同 Muon 参数：矩阵 LR `2e-4`、scalar LR `5e-5`、momentum 0.95、spectral-norm scaling。r8/r16 使用 `top_subspace`、EF14、第 1,000 次通信迭代开始压缩、投影间隔 200。下表只用完成第 10 个 epoch 的 W&B online `-rerun1`；原 CM009 无 W&B 运行和中断的原 CM010 不参与比较。

数值是**最终 epoch 的 validation 指标**，不是每轮验证中的 best，也不是论文 test。每个单元格后的括号是实验编号。MRPC、QQP 显示 accuracy / F1，STS-B 显示 Pearson / Spearman；其他任务显示单一指标。MNLI 显示 matched / mismatched accuracy。

| 任务及指标 | Dense Muon | r8 + Muon | r16 + Muon |
| --- | ---: | ---: | ---: |
| SST-2，accuracy | 0.91628 (CM009) | 0.93005 (CM010) | 0.93119 (CM011) |
| CoLA，Matthews | 0.61570 (CM012) | 0.61339 (CM013) | 0.61833 (CM014) |
| MRPC，accuracy / F1 | 0.89216 / 0.92308 (CM015) | 0.87990 / 0.91358 (CM016) | 0.89216 / 0.92226 (CM017) |
| STS-B，Pearson / Spearman | 0.90674 / 0.90409 (CM018) | 0.90729 / 0.90444 (CM019) | 0.90645 / 0.90379 (CM020) |
| RTE，accuracy | 0.75090 (CM021) | 0.72924 (CM022) | 0.73285 (CM023) |
| QNLI，accuracy | 0.92422 (CM024) | 0.92385 (CM025) | 0.91909 (CM026) |
| QQP，accuracy / F1 | 0.91019 / 0.88036 (CM027) | 0.91365 / 0.88533 (CM028) | 0.91378 / 0.88538 (CM029) |
| MNLI，matched / mismatched accuracy | 0.85960 / 0.85724 (CM030) | 0.86816 / 0.86252 (CM031) | 0.86663 / 0.86252 (CM032) |

除 MNLI matched 外，表中数值来自相应 `-rerun1/all_results.json` 的 `eval_accuracy`、`eval_f1`、`eval_matthews_correlation`、`eval_pearson` 或 `eval_spearmanr`。MNLI 的 `eval_accuracy` 是最终 **mismatched**；最终 **matched** 来自对应 `train.log` 的 `[RANK 0] epoch 9, eval metric`。已逐项核对 24 个日志都有该行。入口还保存逐指标 `best_*`，但未保存这些最优值各自的 epoch，故本表不使用 best。

同任务对照中，压缩组有高于 Dense 的任务，也有低于 Dense 的任务，例如 SST-2 两个秩均较高，RTE 两个秩均较低。所有任务都只有一个 seed，且结果是 validation；不能据此声称稳定增益或复现论文 test 指标。本次也未统计与 Muon 内部 collective 同口径的总通信量。
