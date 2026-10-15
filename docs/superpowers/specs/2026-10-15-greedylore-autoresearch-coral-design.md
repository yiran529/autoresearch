# GreedyLoRE CORAL AutoResearch V1 设计

**日期：** 2026-10-15
**状态：** 已按用户 review 修订并批准
**上游 CORAL：** `Human-Agent-Society/CORAL@0123dfb939b35228cf2c1fde224cd0561e727408`
**GreedyLoRE 来源：** `/home/wyr/greedy_lore@d2c38bc2062e06fdfd73185c508e7cc652933de6`
**权威需求：** `/home/wyr/CORAL_AutoResearch_V1_Handoff_Refined.md`

## 1. 目标

在锁定的 CORAL 版本上实现一个最小、可运行的分布式通信算法
AutoResearch task。系统以 GreedyLoRE r32 + EF14 为 Seed Root `N000`，让
Researcher 和 Engineer 独立工作，并用统一 grader 评估：

- quality：LLaMA 60M + AdamW 完整训练，以及 LLaMA 130M + Muon 的前
  2,500 个 update；
- efficiency：LLaMA 130M + AdamW 在 4 GPU、每卡 batch 1、SHM、
  BF16/FP32 下的正常异步端到端训练 timing；
- 约束：两个 quality workload 的 final validation loss 均不得比各自
  Dense reference 高超过 10%；
- 优化目标：满足 quality gate 时，降低两个 timing cell 的归一化几何平均
  step time。

CORAL 继续负责 agent 生命周期、Git worktree、Attempt、grader daemon、共享
状态与 Codex runtime。领域适配器只增加评测、Node 索引、显式 Analysis/Review、
同 SHA 基础设施 retry 和必要的受控发布边界。

## 2. 非目标

- 不建立第二套 agent scheduler、Git/worktree pool、数据库或通用工作流引擎。
- 不建立运行时 Best/Promotion、MCTS、Node 自动评分或分支配额。
- 不把 CORAL `status=improved` 解释为 quality、semantic 或 Review 通过。
- 不在正式训练中插入 profiler、SVD oracle 或梯度诊断。
- 不把单 seed 结果称为统计等效或普适收敛结论。
- 不复制 `.git`、`.venv`、训练 outputs、旧日志、checkpoint、缓存或 C4 数据。
- 不在 V1 中做强语义证明；semantic 只作为带命令和覆盖范围的软证据。
- 不为假设性的罕见故障、未验证的平台变体或未来扩展提前建设通用防御/恢复框架。V1 的强校验集中在正式实验可信性、权限、GPU 争用和进程清理等直接风险；其他异常默认清晰失败并留存诊断，待真实证据或验收用例出现后再做最小增量。

## 3. 选定方案

采用“CORAL 最小 fork + task-local 薄适配器”。拒绝以下两种替代方案：

1. **纯 task、零 CORAL patch**：无法可靠关闭默认 heartbeat、无法对同一 SHA
   retry，也无法阻止 Agent 直接改写受保护的共享 artifact。
2. **把 Node/Analysis 做成 CORAL 原生领域对象**：会把特定研究语义扩散到
   通用框架，引入 handoff 明确排除的第二套状态机。

框架 patch 只提供显式禁用 heartbeat、validation 标记、grader daemon 内固定用途的
trusted writes 和粗粒度 SRT 保护。submission/artifact 校验、算法、评测矩阵、Node
规则、prompts 和 templates 全部留在 task 目录；不建立通用控制平面。

## 4. 目录布局

```text
/home/wyr/CORAL/
├── coral/                                  # 少量通用补丁
├── tests/                                  # CORAL patch 回归测试
└── tasks/greedylore_autoresearch/
    ├── task.yaml
    ├── seed/                               # GreedyLoRE 候选仓库内容
    │   ├── source_manifest.json
    │   ├── c4/
    │   ├── comm_hooks/
    │   ├── optimizer/
    │   ├── tests/
    │   └── ...
    ├── grader/
    │   ├── pyproject.toml
    │   └── src/greedylore_autoresearch_grader/
    │       ├── grader.py
    │       ├── contract.py
    │       ├── quality.py
    │       ├── efficiency.py
    │       ├── process.py
    │       └── results.py
    ├── autoresearch/
    │   ├── pyproject.toml
    │   └── src/autoresearch/
    │       ├── cli.py
    │       ├── client.py
    │       ├── submission.py
    │       ├── artifacts.py
    │       ├── nodes.py
    │       ├── evaluation.py
    │       ├── analysis.py
    │       ├── validation.py
    │       └── atomic.py
    ├── roles/
    │   ├── researcher.md
    │   └── engineer.md
    ├── prompts/
    │   ├── research_analysis.md
    │   ├── engineering_analysis.md
    │   └── reviewer.md
    ├── references/
    │   ├── semantic_criteria.md
    │   ├── algorithm_ref.md
    │   └── system_ref.md
    ├── templates/
    │   ├── node_spec.md
    │   ├── evaluation.md
    │   ├── research_analysis.md
    │   ├── engineering_analysis.md
    │   ├── review.md
    │   └── submission.yaml
    ├── taskdata/
    │   ├── eval_contract.yaml
    │   └── dense_references/
    │       ├── manifest.json
    │       ├── quality.json
    │       └── efficiency.json
    └── tests/
        ├── test_contract.py
        ├── test_grader.py
        ├── test_evaluation_index.py
        ├── test_node_rules.py
        ├── test_analysis_cli.py
        └── test_sandbox_boundaries.py
```

Handoff Appendix A/B/C 的内容原样抽取到对应 prompts、references 和 templates，
只替换与锁定 CORAL API 或固定 evaluation contract 有关的命令和路径。
`seed/` 不预置 `submission.yaml`；Root bootstrap 在 `S0` 之后从
`templates/submission.yaml` 受控生成该文件，从而保证 `S0→A0` 的唯一变更是
`submission.yaml`。

## 5. Seed 复制与来源固定

从 GreedyLoRE commit
`d2c38bc2062e06fdfd73185c508e7cc652933de6` 读取 `git ls-files`，复制其
119 个跟踪文件到 `tasks/greedylore_autoresearch/seed/`。复制使用一份可复核
脚本；脚本拒绝脏源仓库和 SHA 不匹配。

`source_manifest.json` 记录：

- source repository absolute path；
- source commit；
- 每个复制文件的相对路径和 SHA-256；
- 明确排除的 `.git`、`.venv`、`outputs`、cache、dataset 和日志；
- locked CORAL commit。

GreedyLoRE 原 `AGENTS.md` 保存为 `docs/greedylore-agent-guidance.md`，不占用
候选根目录 `AGENTS.md`，因为 CORAL 会为 Codex runtime 生成该文件。其研究
约束合并到 Researcher/Engineer role。

CORAL 首次创建 run repo 后产生可实际 checkout 的 seed commit `S0`。Node
`N000/spec.md` 同时记录 source commit 和 `S0`。Root 正式 Attempt `A0` 只允许
在 `S0` 上新增或修改 `submission.yaml`；不得改变算法、训练器或 benchmark。

## 6. 固定 Evaluation Contract

`taskdata/eval_contract.yaml` 是 grader 的唯一配置来源。所有正式结果记录
contract SHA-256、candidate commit、PyTorch/CUDA/NCCL 版本、GPU 型号、world
size、数据 manifest 和关键环境变量。

### 6.1 共用条件

- 数据：固定本地 C4 目录 `/home/wyr/greedy_lore/c4/c4_en/en`；
- tokenizer：固定已验证可离线加载的 `t5-base` snapshot
  `/home/wyr/.cache/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`；
- sequence length：256；
- world size：4；
- seed：1243；
- gradient accumulation：1；
- gradient clipping：1.0；
- tracker：正式 grader 强制 `WANDB_MODE=disabled`；
- GPU：不硬编码默认卡号；每个 run 从当前所有空闲 GPU 中原子预留任意 4 张，
  然后设置 `CUDA_VISIBLE_DEVICES`。记录物理 index、UUID 和型号；不足 4 张时
  fail fast 为 infrastructure failure；
- worker 数：4；
- evaluator 不下载模型、tokenizer 或数据；缺失本地资源属于基础设施失败。

### 6.2 Root compressor

Root `N000` 在两个 workload 都使用：

```yaml
compressor: top_subspace
compress_rank: 32
use_error_feedback: ef14
update_proj_gap: 200
min_compression_rate: 1.0
beta_ef: 0
error_inherit: 0
```

Quality 中 `start_compress_iter=1000`。Efficiency 的 warmup 只有 101 step，故
使用 `start_compress_iter=100`，保证整个正式 measurement window 已启用压缩。
这两个值属于不同 benchmark contract，不互相冒充。

### 6.3 Quality：LLaMA 60M + AdamW

- model：`c4/configs/llama_60m_table_iv.json`；
- dtype：BF16；
- activation checkpointing：开启；
- per-device batch：128；global batch：512；
- updates：8,393，必须完整结束；
- warmup：1,000；cosine schedule；`min_lr_ratio=0.1`；
- AdamW：LR 0.0025、betas `(0.9, 0.999)`、eps `1e-8`、weight decay 0；
- final metric：完整 C4 validation 的 `final_eval_loss` 和 `final_eval_ppl`。

### 6.4 Quality：LLaMA 130M + Muon

- model：`c4/configs/llama_130m.json`；
- dtype：BF16；
- activation checkpointing：开启；
- per-device batch：128；global batch：512；
- scheduler budget：20,000；实际执行前 2,500 updates；
- CLI 表达为 `--num_training_steps 20000 --stop_after_steps 2500`，从而保留
  原 20,000-step schedule 的前缀，而不是把 cosine schedule 压缩到 2,500 步；
- warmup：2,000；`min_lr_ratio=0.1`；
- Muon matrix：LR 0.01、momentum 0.95、eps `1e-8`、spectral-norm scaling、
  weight decay 0；
- scalar AdamW：LR 0.001、betas `(0.9, 0.999)`、eps `1e-8`、weight decay 0；
- final metric：2,500 updates 后完整 C4 validation 的 `final_eval_loss` 和
  `final_eval_ppl`。

### 6.5 Quality gate

Dense reference 使用相同 model、optimizer、schedule、batch、dtype、seed、数据
和 validation，仅设置 `compressor=none`。每项分别判定：

```text
candidate_final_eval_loss <= 1.10 * dense_reference_final_eval_loss
```

两项都通过时 `quality_gate=1`，否则为 0。报告始终保留原始 loss、PPL、ratio、
训练步数和日志；10% 是工程准入容差，不是统计等效声明。

### 6.6 Efficiency matrix

为节省评测成本，正式 efficiency 只包含两个 cell：

| Cell | Model | Optimizer | Dtype |
|---|---|---|---|
| `130m_adamw_bf16` | 130M | AdamW | BF16 |
| `130m_adamw_fp32` | 130M | AdamW | FP32 |

每个 cell 固定：

- 4 GPU；per-device batch 1；global batch 4；sequence 256；
- AdamW：LR 0.0025、betas `(0.9, 0.999)`、eps `1e-8`、weight decay 0；
- scheduler budget 20,000；warmup 2,000；`min_lr_ratio=0.1`；
- 无 activation checkpointing；
- `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、
  `NCCL_CUMEM_HOST_ENABLE=0`、默认 NCCL channel；
- 正常异步 DDP，不传 `--strict_blocking_communication`；
- 使用被动 `--record_bucket_layout`，不得 `.wait()` 或同步 inner Future；
- `ddp_bucket_cap_mb=1024`；
- `start_compress_iter=100`；
- warmup 101 updates，连续测量 400 updates；
- 每个 cell 独立运行 2 次；第一轮顺序为 BF16、FP32，第二轮为
  FP32、BF16；Dense 与 candidate 使用同一顺序；
- 每次输出 slowest-rank continuous-window wall time / 400；cell 正式值取两次
  `mean_iteration_seconds` 的算术平均，并同时报告两次原始值和 relative spread；
- repetition 之间强制重建 launcher/process group，不复用上一次的
  CUDA/NCCL 进程；
- 非 finite loss、不完整窗口、bucket layout 不一致或 NCCL 未实际使用 SHM
  均使该 run 无效，不自动改变配置重跑。

Dense timing reference 使用完全相同的两个 cell、2 repetitions 和 aggregation。
每个 cell 的 speedup 为：

```text
dense_mean_step_seconds / candidate_mean_step_seconds
```

综合 `efficiency_score` 是两个 speedup 的几何平均，使 BF16 和 FP32 在
归一化后等权。Dense 和 candidate 可以分配到不同物理 GPU index，但 GPU
型号、world size 和关键 NCCL 条件必须一致。

### 6.7 CORAL aggregate

Research Attempt：

```text
aggregated = efficiency_score  if quality_gate == 1
aggregated = 0.0               otherwise
```

Engineering Attempt 只运行 efficiency，`aggregated=efficiency_score`；quality
写为 `inherited (provisional)`，并绑定 `quality_anchor` 指向该 Node 唯一已完成的
Research SHA。Aggregate 只满足 CORAL 的排序接口，不构成 Node Selection、Review
或科学结论。

公开 named scores 至少包含：两项 quality loss/ratio/pass、两项 timing mean、
两项 speedup、`quality_gate` 和 `efficiency_score`。详细 repetition、命令、环境、
stdout/stderr、config 和 machine-readable result 放入 immutable eval logs。

## 7. Dense Reference 校准

真实 run 启动前，由受信任 operator 命令 `autoresearch calibrate-dense` 在固定
`S0` 上串行运行两项 dense quality 和两项 dense efficiency。输出进入 taskdata
并生成 manifest，包含 contract hash、source/S0 SHA、完整命令、数据 manifest、
硬件/软件 fingerprint 和原始日志 hash。

`taskdata/dense_references/` 必须通过 `task.yaml` 的 grader-private 配置仅向
trusted grader/operator 暴露；Agent 只能读取 grader 发布的按 Attempt 归一化结果，
不能读取或修改 raw dense references。

grader 启动时拒绝以下情况：

- reference contract hash 与当前 contract 不同；
- reference 缺 cell、步数不符或有非 finite 值；
- efficiency 当前 GPU UUID 集合与 reference 不同，且 GPU 类型、选定四卡互联拓扑、NUMA/NCCL fingerprint 不能证明等效；此时须在候选卡组补测 dense reference；
- source/S0 来源无法验证。

Dense calibration 不是 CORAL Node 或 Attempt。Root `N000` 仍通过统一 grader
重新执行 GreedyLoRE research evaluation。

## 8. Grader 行为

packaged `TaskGrader` 从固定 checkout 读取 committed `submission.yaml`，不读取
浮动 `tree.md` 推断模式。昂贵运行前依次检查：

1. YAML schema、mode、node ID、完整 SHA、base commit 和 quality anchor；
2. candidate HEAD 与 CORAL Attempt commit 一致；
3. Node spec 存在且 base/parent/Analysis 引用合法；
4. research：该 Node 尚无另一个不同 SHA 的正式 Research Attempt；
5. engineering：quality anchor 是该 Node 的唯一 terminal Research SHA；
6. Root：A0 相对 S0 的 diff 只包含根目录 `submission.yaml`；
7. GPU lock、GPU 空闲、数据/tokenizer、磁盘空间和环境可用。

`mode=research` 串行运行 60M quality、130M quality 和两个 efficiency cell。
quality 失败不跳过 efficiency，因为一次 Research Attempt 必须给出完整 quality +
efficiency 证据。进程崩溃、OOM、非 finite、数据耗尽和 contract 不符均按实际记录。

`mode=engineering` 只运行两个 efficiency cell。grader 不声称 semantic 等价；它
只复制 `submission.yaml.semantic_self_check` 到 evidence，并校验状态值、命令和
日志引用格式。

grader 使用 `.coral/public/eval_logs/<attempt_sha>/` 持久化输出。每个子进程以新
process group 启动；正常结束、timeout 和异常路径都执行 TERM、有限等待、KILL
和 CUDA/NCCL 子进程清理。`grader.parallel.max_workers=1`，并使用 task 级文件锁。
若选定 GPU 已被任何外部进程占用，Attempt 以 infrastructure failure 快速结束，
不得等待或干扰该进程。

## 9. Attempt、Node 与 retry

### 9.1 唯一 Research SHA

grader daemon 内的 trusted submit handler 在写 pending Attempt 前解析 committed `submission.yaml`，把
`mode`、`node_id`、`base_commit` 和 `quality_anchor` 写入 Attempt metadata。它在
同一原子锁内扫描 pending + terminal Attempts：同 Node 已有不同 research SHA
时直接拒绝，因此无需独立 `node_state.yaml`。客户端声明的 Agent、island、parent 均不授予权限；manager 把 Agent 根 PID、进程 start time、worktree realpath 和 island 原子登记在 `.coral/private/trusted_agents.json`，handler 将 `SO_PEERCRED` PID 沿 `/proc` 父链归属到仍存活且 start time 匹配的登记根进程。同 UID、无法归属、PID 复用或跨 Worktree SHA 均 fail closed。

普通 Git commit 和 `coral eval --tune` 不构成正式 Research Attempt；真实
research 必须使用默认 real budget class。Tune 只允许运行明确的 smoke contract，
且结果不能进入正式 `evaluation.md` 的 Research 行。

### 9.2 同 SHA 基础设施 retry

新增 `autoresearch retry <sha>`。它通过固定 trusted endpoint 请求，只接受 terminal Attempt metadata
中 `failure_class=infra` 的 SHA；candidate failure、quality failure、OOM 因算法
内存需求、非 finite 或用户代码异常不能 retry。

Attempt JSON 仍以 commit SHA 为主键。`metadata.grader_runs` 是 append-only 列表，
每项含 `run_id`、started/finished、status、failure class、contract hash 和 log dir。
retry 不创建 commit 或第二个 Research Attempt；最终视图可以反映最新有效结果，
但原始失败 run 永远保留。受保护 contract 固定 `max_infra_retries=2`（初始运行之外最多两次 retry）；每次实际 Grader 调用单独记录 run、时间和 GPU 资源消耗。dispatch、finalize 与 retry 在同一锁或等效原子事务中读改写 Attempt，避免丢失 run history。并发 retry、预算耗尽、contract 改变、来源 Agent/Worktree 不符或 checkout SHA 不同均拒绝。

## 10. 受控发布与 sandbox

### 10.1 当前上游行为

锁定版本的 Codex runtime 使用 `--dangerously-bypass-approvals-and-sandbox`，安全
边界依赖外层 CORAL SRT。现有 SRT 为支持 agent-side `coral eval`，允许写整个
`.coral/`，不足以满足 handoff 的受保护 artifact 要求。

### 10.2 写权限策略

SRT 允许 Agent 直接写自己的 worktree、scratch、candidate repo `.git`、runtime home 必需目录和 `/tmp`；允许读
同 island 已发布 evidence 和 sibling worktrees，但 sibling 只读。它不再 allow-write 整个 `.coral/`；仅允许连接 trusted socket，并 deny-write：

```text
.coral/public/attempts/
.coral/public/eval_logs/
.coral/public/notes/tree.md
.coral/public/notes/nodes/
.coral/.git/
.coral/private/
```

已发布普通 notes 可读，Agent 的过程记录留在 own worktree/scratch；所有 `.coral` 写入都由 grader daemon、manager 或 trusted endpoint 完成。跨 island 不开放读取。

### 10.3 固定用途的可信写入入口

现有 grader daemon 在进入 Attempt scan loop 前监听本 run 的 `.coral/public/trusted.sock`，并复用 daemon 的 start/stop/resume 生命周期；不新增 manager-owned server、`ControlConfig`、动态 Provider 或通用 operation registry。入口只支持：

- `submit_attempt`：从固定 commit 自行读取并校验 `submission.yaml`，在同一锁内完成 Research 唯一性检查和 pending Attempt 写入；
- `retry_attempt`：仅把同 SHA、terminal、`failure_class=infra` 且 contract 未改变的 Attempt 重新置为 pending；
- `publish_artifact`：只接受 kind、Node/mode/version 和 bounded content，由 task validator 映射 canonical target 后原子发布；没有任意路径或命令字段。

`coral eval` 仍在 Agent worktree 完成 Git add/commit 和 parent lookup，然后请求 `submit_attempt`；trusted handler 从受保护登记推导 Agent/worktree/island，验证 SHA 为该 worktree 当前提交，再执行 shared checkpoint、parent checkpoint provenance 和 pending write。Root bootstrap 使用固定 S0/N000/content 的独立受信任宿主入口，Agent 不能伪装。`autoresearch retry` 与 Node/Analysis CLI 使用后两个 operation。trusted validator 安装在 grader-private venv，服务端把所有 Agent payload 当作不可信输入。`metadata.grader_runs` 仅是同 SHA retry 所需的 append-only 最小历史，不扩展成独立运行管理系统。

### 10.4 主机前置条件

当前主机已有 `codex`、`uv`、`nvidia-smi`、`rg`，但缺少 `npx`、`bubblewrap`
和 `socat`，因此现在不能声称 SRT 已生效。真实启动前必须安装并验证
`@anthropic-ai/sandbox-runtime`、bubblewrap、socat；若检查失败，`coral start`
fail closed，不退化为无 sandbox 模式。

## 11. Heartbeat

增加显式 `agents.heartbeat_enabled: false`。关闭时：

- config preprocess 不补默认 reflect/consolidate/pivot/lint_wiki；
- hub 不强制恢复 protected consolidate；
- manager 不 seed local/global heartbeat 文件；
- resume 不重新引入动作；
- 手动 heartbeat mutation 返回明确禁用反馈。

该开关只改变 heartbeat，不移除 manager、grader、worktree 或 agent restart
reliability。Researcher/Engineer 的稳定角色来自各自 role file。

## 12. Producer、Analyzer 与 Reviewer

`task.yaml` 启动两个 Codex producer：一个 Researcher、一个 Engineer。两者同 island、独立 worktree、不同 role file；Node Selection 由角色根据 evidence 自主决定，不由 aggregate score 自动指定。

`autoresearch analyze --node Nxxx --mode research|engineering` 是 task-local 轻量 CLI：

1. 从 Node spec、`evaluation.md`、Attempts 和固定 commit 解析不可变 EvidenceBundle；
2. 无有效 Analysis 时启动 fresh Analyzer session，校验后通过 `publish_artifact` 原子发布；
3. 有 Analysis 而无对应 Review 时只启动独立 fresh Reviewer session；
4. 两者都有效时幂等返回；`--revise` 才分配下一版本并保留旧版本。

Analyzer/Reviewer 使用不同 scratch 和独立 `codex exec --ephemeral`，不复用 session、thread 或对话。Review decision 为 `pass|warning|fail`；`fail` 表示 Review 已完成但不表示通过，也不触发自动修订。Research Review pass/warning 后即可扩展 Child，不等待 Engineering Analysis。

Engineering Analysis 不要求已有 Engineering Attempt。它在开始时固定该 Node 当时已有的全部 Engineering Attempts（允许空列表）及各自正式 run/log/comparability 引用，并与 Review 共享同一个 evidence manifest hash；后续 Attempt 不会悄然改变已发布版本。没有时只绑定唯一 Research SHA、Research evaluation 和现有 profiling/evidence，不得声称未测量的提升。Analyzer/Reviewer 不能修改候选代码、历史 Artifact 或 raw evaluation，analyze 也不重跑正式训练。完成/恢复状态只由校验通过的版本化 Artifact 推导，不保存额外 workflow state。

## 13. Node artifact 与重建

权威共享路径保持 handoff 约定：

```text
.coral/public/notes/
├── tree.md
└── nodes/Nxxx/
    ├── spec.md
    ├── evaluation.md
    ├── research_analysis_vN.md
    ├── research_review_vN.md
    ├── engineering_analysis_vN.md
    └── engineering_review_vN.md
```

`evaluation.md` 从 Attempts、`metadata.grader_runs`、raw grader logs 和对应 commit
的 `submission.yaml` 重建。它显示唯一 Research SHA、同 SHA retries、全部
Engineering Attempts、失败/退化、contract/hardware 和 evidence。多个不同
Research SHA 时显式报告一致性错误，不覆盖。不可比 contract 不排序。

`tree.md` 只索引 Node 和最新已发布 Analysis/Review，可从 nodes 重建。没有
`node_state.yaml`、`engineering_history.md`、`runs/`、`iterations/` 或 findings
ledger。

## 14. Root Bootstrap

1. 复制并校验 GreedyLoRE source manifest。
2. 建立 run repo seed commit `S0`，记录 source SHA 与 S0。
3. 运行 trusted dense calibration，冻结 references。
4. 受控创建 `N000/spec.md`：`node_type=seed`、`parent_node=null`、
   `base_commit=S0`、provided GreedyLoRE r32/EF14。
5. 在候选根新增 Root `submission.yaml`；trusted submit 检查 S0→A0 diff 只含该文件。
6. 一次 `mode=research` eval 运行两项 quality 和两项 efficiency，A0 成为唯一
   Research SHA。
7. `autoresearch analyze --node N000 --mode research` 从 Artifact 解析 A0 并发布 Root
   Analysis 和 Review。
8. Review 为 pass/warning 后允许普通 child；Engineer 可同时继续 N000。

## 15. 错误分类与恢复

grader 为每个失败设置一个分类：

- `infra`：GPU 已占用、launcher/NCCL 外部故障、数据/cache 缺失、磁盘或 grader
  worker 故障；允许同 SHA retry；
- `candidate`：代码崩溃、deadlock、配置篡改、训练不足、非 finite、算法 OOM；
  不允许同 Node 新 research eval；先分析，再以 child 修复；
- `quality`：训练有效但超过 10% gate；保留完整 timing 和 Analysis；
- `invalid_submission`：schema、base、anchor、重复 Research SHA 或受保护文件违规；
  昂贵 benchmark 前拒绝。

所有异常保留 command、return code、stdout/stderr tail、完整日志路径和 cleanup
结果。缺失值写 `null`/`—`，不合成数字。

## 16. 验证策略

### 16.1 单元和集成测试

- contract 解析和 hash 稳定性；
- research/engineering submission schema；
- quality 10% 边界（等于 1.10 通过，超过失败）；
- timing 两次算术平均、relative spread 和两 cell geometric mean；
- incomplete/nonfinite/mismatched environment 拒绝；
- efficiency 使用同 GPU UUID 集合或验证通过的 topology/NUMA/NCCL 等效对照；
- Node 唯一 Research SHA，并发 reservation；
- infra-only same-SHA retry、受保护 retry 上限、append-only run history 与 finalize/retry 并发；
- 同 UID Agent 身份伪造、PID 复用与跨 Worktree SHA 提交被拒；
- `evaluation.md` 删除后可完全重建；
- Analysis/Review 幂等、partial recovery、`--revise` 版本化；
- heartbeat disabled 后 start/resume 均无动作；
- SRT deny-write、symlink、绝对路径和 sibling write 尝试；
- fixed trusted endpoint 不接受路径、任意命令、未知 operation 或未校验 Artifact；
- grader process cleanup 与 GPU lock。

### 16.2 分阶段 smoke

1. 原生锁定 CORAL 的 init/validate/start/eval smoke；
2. dummy task 验证 research 双 profile、engineering efficiency-only、Attempt/index；
3. 单进程 CPU contract/parser smoke；
4. 4 GPU、3-step dense/GreedyLoRE BF16 correctness smoke；
5. 动态预留 4 GPU、短 warmup + 3 measured step 的两 timing cell smoke，确认 SHM 和异步
   Future；
6. 缩短版 60M/130M quality smoke，确认 final result schema；
7. S1–S15 dummy acceptance；
8. dense calibration；
9. Root N000 正式 evaluation。

`coral validate` 给 Task metadata 增加只读 `coral_validation=true` 标记。该 task
grader 在 validation 时只跑便宜 smoke，不把它写成正式 Attempt；grader daemon
中的真实 Attempt 没有该标记，始终执行 committed submission 对应的正式 contract。

### 16.3 Handoff 验收映射

- S1/S2：trusted submit + retry history；
- S3/S9/S11：Artifact 驱动 analyze 的幂等、missing-review recovery 和 revision；
- S4/S5：child 引用固定 Analysis/Review 版本；
- S6/S7：evaluation rebuild 覆盖多 Engineering/零 Engineering；
- S8：Reviewer fixtures；
- S10：GPU lock、max_workers=1、async formal timing、diagnostic 隔离；
- S12：Root diff allowlist 和 S0/A0 双记录；
- S13：reference prompts 和 evidence validation；
- S14：selection rationale schema，不实现自动 selector；
- S15：SRT + trusted endpoint adversarial filesystem tests。

## 17. 实施顺序

1. 建立 task scaffold、复制脚本和 source manifest；
2. 增加 heartbeat disable 与 validation marker；
3. 增加 grader-owned trusted submit/retry/publish 和保护路径 sandbox；
4. 实现 evaluation contract、quality/efficiency runner 和 packaged grader；
5. 实现极简 Node Artifact、evaluation rebuild 和幂等 Root bootstrap；
6. 部署 prompts/references/templates；
7. 实现 Artifact 驱动的 Analyzer/Reviewer fresh-session 发布；
8. 完成 dummy S1–S15；
9. 安装/验证 SRT 主机依赖并做权限攻击测试；
10. 做动态 4 GPU smoke、dense calibration、Root N000 正式 eval。

实施时每完成一个可独立验证的大 step 就创建一个 Git commit。测试和检查在
commit 前完成；commit 动作本身不额外触发 push、长训练或 calibration。

长训练通过 grader daemon/tmux 管理，不用 Agent 回合轮询。任何尚未真实运行的
sandbox、GPU benchmark 或完整训练均不得在交付说明中声称通过。

## 18. 完成条件

V1 完成需要同时满足：

- 锁定 CORAL 与 GreedyLoRE 来源可复核；
- task validate/dummy acceptance 通过；
- Research/Engineering 分流和完整 evidence 可重建；
- 同 Node 不同 Research SHA 被拒绝，同 SHA infra retry 在预算内保留全部历史，且同 UID 不能伪造提交者；
- Analysis/Review fresh、版本化、幂等且受控发布；
- heartbeat 确实关闭；
- Agent 不能直接改写 protected shared artifacts，受控命令仍可工作；
- 四卡 SHM BF16/FP32 async timing smoke 有实际 NCCL 证据；
- dense references 完成并与 contract 匹配，efficiency 对照使用同 GPU UUID 集合或经验证等效的 topology/NUMA/NCCL 环境；
- Root N000 的唯一 Research Attempt、Analysis 和 Review 完成；
- 已知限制、未覆盖项和所有失败结果如实记录。
