# GreedyLoRE AutoResearch artifacts：组织、内容与保存规则

本文对应当前任务实现，说明一次研究如何关联代码、评估结果与分析文档。整体流程见 [架构文档](greedylore-autoresearch-architecture.md)。下文路径以单岛 run 为准；当前 GreedyLoRE 动态工作流只支持单岛。

## 1. 先区分四种身份

| 身份 | 示例 | 含义 |
| --- | --- | --- |
| Run | `<run>/` | 一次持续的研究工作区；停止、恢复进程可以继续使用同一个 Run |
| Node | `N001` | 一个研究假设及其 Spec、Research 和后续 Engineering；`N000` 是预定义 Root |
| Attempt / commit SHA | `<SHA_A>` | 固定代码版本上的一次正式提交；一个 Node 只有一个正式 Research SHA，Engineering 可以有自己的候选 SHA |
| Grader run ID | `<RUN_1>` | 对 Attempt 的一次实际评估调用；同 SHA 的基础设施重试产生新的 run ID |

关联关系是 `Run → Node → Attempt/SHA → grader run ID → 原始结果`。Analysis 还具有自己的版本号，例如 `research_analysis_v2.md`；它与测速重复轮次 `r2`、grader run ID 都不同。

本文使用 `<SHA_A>`、`<RUN_1>` 等占位符。磁盘上 SHA 是完整 Git commit hash，grader run ID 是生成的 UUID hex 字符串。所有示例数值均为说明用途，不是实际实验结论。标注“节选”的 JSON 可解析，但省略了其他真实字段，不能作为完整提交或 Analysis 输入。

## 2. Run 的目录布局

```text
<run>/
├── repo/                                  # 共享 Git 对象、提交历史
├── agents/<agent_id>/                     # 生产 Agent 的独立 worktree
└── .coral/
    ├── config.yaml                        # Run 配置，resume 从这里读取
    ├── private/                           # 宿主持有，不对生产 Agent 开放
    │   ├── autoresearch_state.json        # 权威节点、assignment、编号状态
    │   ├── root_bootstrap.json            # 固定 Root 身份与策略
    │   └── dense_references/
    │       ├── quality.json               # 同 contract 的 Dense 质量基线
    │       └── efficiency.json            # Dense 测速基线与硬件环境
    └── public/
        ├── attempts/<SHA_A>.json          # 正式提交记录与 grader run 历史
        ├── eval_logs/<SHA_A>/<RUN_1>/     # 本次评估持久证据
        ├── notes/
        │   ├── tree.md
        │   ├── nodes/N001/
        │   │   ├── spec.md
        │   │   ├── evaluation.md
        │   │   ├── research_analysis_v1.md
        │   │   └── engineering_analysis_v1.md
        │   ├── research_summary.md
        │   └── engineering_summary.md
        ├── diagnostics/                   # 可选 Analyzer 诊断归档
        └── logs/                          # Agent Session 日志
```

这些是按阶段可能出现的文件，不保证启动后立刻齐全。没有 Engineering 时不会生成对应 Analysis；Summary 需要达到记忆触发条件；失败评估可能只有部分日志。

Agent worktree 的运行时共享目录可以链接到公开产物，例如 `.codex/eval_logs/` 或 `.claude/eval_logs/`。反馈中的 `eval_logs/<SHA>/<run_id>/...` 是不带运行时前缀的相对定位方式，不是工作区中另一份独立结果。

## 3. 一次正式提交如何保存

1. Agent 经可信入口提交，固定代码 commit 与提交元数据，保存 `attempts/<SHA>.json`。
2. Grader dispatch 创建 run ID，并在 Attempt 的 `metadata.grader_runs` 追加运行记录，包含状态、起止时间、contract hash 和日志目录。
3. Grader 对固定 commit 创建隔离 checkout，直接将需保留的输出写入 `eval_logs/<SHA>/<run_id>/`。
4. 完成后将分数、反馈、失败类别和证据路径写回 Attempt；临时评分 checkout 会被删除。
5. 节点收尾时生成 `evaluation.md`，准备固定证据，再校验、发布版本化 Analysis。

因此，把结果只写在 grader 的 `codebase_path` 下不能长期保留。正式输出必须使用 `eval_logs_dir`，或通过框架结果记录、任务私有持久目录保存。

### Attempt 示例

`attempts/<SHA_A>.json` 的内容节选：

```json
{
  "commit_hash": "<SHA_A>",
  "agent_id": "research-N001",
  "metadata": {
    "node_id": "N001",
    "mode": "research",
    "contract_hash": "<CONTRACT_HASH>",
    "grader_runs": [
      {
        "run_id": "<RUN_1>",
        "status": "running",
        "failure_class": null,
        "log_dir": "eval_logs/<SHA_A>/<RUN_1>"
      }
    ]
  }
}
```

Attempt 是持续更新的索引记录，不是每个阶段各生成一个 JSON。完整记录还包括标题、时间、parent、状态、汇总分数、反馈等。`metadata.scores` 中保存各项评分；`metadata.evidence_paths` 用于定位评分证据。

Research 运行质量和效率评估；Engineering 只跑效率，质量指标标为 `inherited (provisional)`，依赖固定 Research anchor，不代表重新实测质量。Research 质量门槛失败时汇总分数为零；基础设施失败、硬件不可比等无法有效评分的情形不应解读为速度为零。

## 4. 评估证据的文件名与内容

一次完整 Research grader run 的主要目录如下：

```text
eval_logs/<SHA_A>/<RUN_1>/
├── gpu_resources.jsonl
├── gpu_resources.json
├── quality/
│   ├── llama60m_adamw/all_results.json
│   ├── llama60m_adamw.process/
│   │   ├── process.json
│   │   ├── <TOKEN>.stdout.log
│   │   └── <TOKEN>.stderr.log
│   ├── llama130m_muon/all_results.json
│   └── llama130m_muon.process/...
└── efficiency/
    ├── r1-130m_adamw_bf16/
    │   ├── config.json
    │   └── all_results.json
    ├── r1-130m_adamw_bf16.process/...
    ├── r1-130m_adamw_fp32/...
    ├── r1-130m_adamw_fp32.process/...
    ├── r2-130m_adamw_fp32/...
    ├── r2-130m_adamw_fp32.process/...
    ├── r2-130m_adamw_bf16/...
    └── r2-130m_adamw_bf16.process/...
```

Engineering 没有这里的 `quality/` 阶段。当前 timing 顺序为第一轮 BF16 → FP32，第二轮 FP32 → BF16。目录的 `r1` / `r2` 从 1 开始，而传入脚本并写入结果的 `repetition` 从 0 开始。因此 `r1-130m_adamw_bf16/all_results.json` 中是 `repetition: 0`。

### 4.1 Quality：训练后的质量

`quality/llama60m_adamw/all_results.json` 表示 LLaMA 60M + AdamW 的质量运行结果；`llama130m_muon` 表示另一项 130M + Muon workload。当前两项使用 BF16。

内容节选：

```json
{
  "status": "completed",
  "final_eval_loss": 4.1,
  "final_eval_tokens": 10000000,
  "best_eval_loss": 4.1,
  "best_eval_step": 4000,
  "update_step": 4000,
  "dtype": "bfloat16",
  "optimizer": "adamw",
  "world_size": 4,
  "contract_hash": "<CONTRACT_HASH>"
}
```

完整结果还有 final/best perplexity 和 `wandb_link`。评分使用 `final_eval_loss`，不使用最佳历史 loss、训练 loss 或 PPL 替代。当前探索协议中 60M 跑 4,000 步、130M 跑 2,500 步，但保留各自原学习率调度；具体以该 Run 固定的 contract 为准。

Quality 默认没有像 timing 一样单独写出的 `config.json`；完整命令可从兄弟目录 `llama60m_adamw.process/process.json` 查看。训练日志保存在 stdout/stderr 中，结果 JSON 不是逐步训练曲线。

### 4.2 Efficiency：端到端测速

`r2-130m_adamw_fp32` 表示第二轮、LLaMA 130M、AdamW、FP32 的测速 cell。

`config.json` 在初始化过程中写入并补全，包含命令行参数、模型配置、优化器分组、参数量、Git HEAD、关键源码 SHA256、PyTorch/GPU 信息和 NCCL 环境。它用于回答“具体测了哪份代码、什么配置”；初始化失败时可能只存在部分配置。

`all_results.json` 的内容节选：

```json
{
  "status": "completed",
  "repetition": 1,
  "dtype": "float32",
  "warmup_iterations": 101,
  "measured_steps": 400,
  "first_measured_update": 102,
  "last_measured_update": 501,
  "mean_iteration_seconds": 0.05,
  "rank_elapsed_seconds": [19.8, 20.0, 19.9, 19.8],
  "slowest_rank": 1,
  "strict_blocking_communication": false,
  "bucket_layout_recorded": true,
  "contract_hash": "<CONTRACT_HASH>"
}
```

主耗时是 `max(rank_elapsed_seconds) / measured_steps`，上述示例为 `20 / 400 = 0.05` 秒。该连续窗口包含取数据、H2D、前后向、DDP hook、裁剪、优化器、scheduler 和清梯度。

完整结果还包括各 rank 的逐步时间、loss、显存峰值、bucket 数量与布局、GPU UUID、NUMA、拓扑指纹和代码来源。当前正式协议为异步 DDP 加被动 bucket 记录，`strict_blocking_communication=false`；这些耗时不能当成单独通信耗时，bucket 字节也不能直接当成实际网络传输量。

每个精度 cell 聚合两轮的均值与 relative spread；最终效率分数是 BF16、FP32 两项 `Dense 耗时 / Candidate 耗时` 的几何平均。聚合评分写在 Attempt 中，各轮结果仍独立保留。

### 4.3 Process：执行证据

与结果目录相邻的 `<workload>.process/` 保存：

| 文件 | 内容 |
| --- | --- |
| `process.json` | 完整 argv、记录的环境变量、起止时间、returncode、timed_out、cleanup_signals、infra_failure 和日志路径 |
| `<TOKEN>.stdout.log` | 子进程标准输出；TOKEN 是本次进程调用生成的唯一标识 |
| `<TOKEN>.stderr.log` | 子进程标准错误，可能包含普通 logger/NCCL 输出，也可能包含异常堆栈 |

stderr 非空不等于失败，应结合退出码、超时状态和结果校验判断。正常命令返回后 grader 写 `process.json`；若 worker 被强制终止，该文件和最终结果可能尚未生成，但流式写入的日志仍可能存在。

### 4.4 GPU 资源事件

`gpu_resources.jsonl` 每行是一个 JSON 事件；`gpu_resources.json` 是最近事件的快照，并非完整历史。事件包含 `time`、`stage`、`state`、`elapsed_seconds`、运行 UID、资源策略版本与 options，以及本次事件附加字段，例如 GPU UUID、测速 cell、轮次或失败原因。

排查“为什么迟迟没有 all_results”时，先看资源事件是否仍在等待 GPU，再看进程日志。这两个文件不是连续 GPU 利用率或功耗曲线。

## 5. Notes：实际格式与更新规则

不要只按 `.md` 扩展名推断内容格式：

| 文件 | 内容和职责 | 更新规则 |
| --- | --- | --- |
| `nodes/N001/spec.md` | 当前正式 Spec 使用 YAML；记录 node、parent、base commit、假设、改动、风险和证伪条件 | 初始化后不可变 |
| `nodes/N001/evaluation.md` | Markdown 表格；从正式 Attempts 和原始日志重建，列出 mode、commit、submission、status、score、contract、comparable、run IDs、evidence | 可以刷新 |
| `research_analysis_v1.md` | schema 约束的 JSON；Research 证据封装及分析 | 单版本不可变，修订新增 v2 等 |
| `engineering_analysis_v1.md` | 同样是 JSON；固定 Research SHA 和 Engineering 候选证据上的分析 | 同上 |
| `tree.md` | 权威工作流状态的 Markdown 树视图 | 随状态重建 |
| `research_summary.md`、`engineering_summary.md` | YAML frontmatter + Markdown 正文；跨节点记忆 | 达到触发条件后更新 |

仓库 `templates/` 中供人阅读的 Markdown 提示骨架不等于实际持久 Analysis 的序列化格式；解析 Analysis 应使用 JSON。

普通节点 `spec.md` 的 YAML 示例：

```yaml
node_id: N001
parent_node: N000
parent_commit: "<PARENT_RESEARCH_SHA>"
base_commit: "<PARENT_RESEARCH_SHA>"
hypothesis: "调整子空间刷新策略可改善短预算质量。"
selection_rationale: "基于父节点 Analysis 中记录的证据缺口。"
delta: "修改刷新策略。"
risk: "刷新开销可能增加。"
falsification_criteria: "固定协议下质量未改善且速度下降。"
analysis_refs:
  parent_research: "<固定的父节点 Research Analysis 引用>"
  parent_engineering: null
```

Root `N000` 的 Spec 由 bootstrap 建立，有独立的来源身份要求，不能直接套用普通节点示例。

Analysis 内容节选：

```json
{
  "node_id": "N001",
  "mode": "research",
  "research_sha": "<SHA_A>",
  "engineering_attempts": [],
  "evidence_manifest_hash": "<MANIFEST_HASH>",
  "summary": "候选通过本次固定预算下的质量门槛。",
  "findings": ["具体观察及对应的固定证据引用。"],
  "limitations": ["短预算结果不能推断完整训练质量。"]
}
```

完整 schema 还要求 `evaluation_version`、`evidence_manifest`、`evidence_paths`、`evaluation_snapshot`。宿主将固定证据封装写入并校验；Analysis 保存自己的 Evaluation 快照和 manifest，因此以后刷新 `evaluation.md` 不会改写旧 Analysis 当时引用的证据身份。`evaluation_version` 也不应从当前 Evaluation 文件名猜测。

Summary 通过 frontmatter 记录已处理的 Analysis，例如：

```markdown
---
processed_analyses:
  - nodes/N001/research_analysis_v1.md
  - nodes/N002/research_analysis_v1.md
---
本阶段的研究结论、适用条件与后续方向。
```

这是格式示例，不表示两个节点就一定触发更新。当前默认 gate 按尚未被两份 Summary 共同覆盖的 Research Analysis 所属节点计数，达到 3 个节点时触发；不能用 Summary 是否更新来判断某个 Node 是否完成。

发布入口检查 canonical 路径、调用者身份与节点状态，拒绝路径穿越和符号链接。Spec 和版本化 Analysis 同路径、同内容重复发布可幂等返回；不同内容覆盖会拒绝。发布使用锁、临时文件和原子替换；Evaluation 的可信 authorizer 从正式记录刷新内容。Summary 由宿主记忆流程发布，普通 Agent 不能通过 artifact 路由伪造全局总结。

## 6. 可选 Analyzer 诊断归档

启用受控诊断后，按 Node、模式和 Analysis 版本组织：

```text
.coral/public/diagnostics/N001-research-v1/
├── workspace/
│   ├── code/                       # 固定 SHA 的只读代码
│   ├── evidence/                   # 只读证据快照
│   ├── experiments/                # 可写诊断脚本
│   └── results/                    # 可写输出
└── records/request-1/
    ├── record.json                 # 请求身份、命令、状态、时间和资源等
    ├── scripts/                    # 实际执行的脚本快照
    ├── input-results/              # 执行前已有 results 输入快照
    ├── execution/                  # 隔离执行所用可写输入副本
    ├── stdout.log
    ├── stderr.log
    └── results/                    # 本次输出归档
```

`request-1` 是显式使用 `--request-id request-1` 时的示例名称；默认请求 ID 为生成的 UUID hex 字符串，并非递增编号。它不是正式 Attempt 或 grader run ID。同请求 ID、同参数的重复请求返回原结果，复用 ID 却改变参数会被拒绝。取得 GPU 后冻结脚本与可写输入，再执行冻结副本；最终结果安全合并回 workspace。排队时的可写 workspace 不能代替持久执行记录来归因。

每类快照归档上限为 64 MiB / 4096 条目，超出明确记录归档失败，不保证完整保留超大输出。环境不可用、申请失败或执行中断时可能只有部分记录；诊断身份、预算及宿主控制状态另存于私有诊断目录。诊断结果用于支持 Analysis，不直接替代正式质量或速度评分。

## 7. 重试、分阶段运行与保留边界

同 SHA 的基础设施失败重试保留为：

```text
attempts/<SHA_A>.json                  # 同一 Attempt，grader_runs 累积历史
eval_logs/<SHA_A>/<RUN_1>/             # 第一次评估，可能失败/不完整
eval_logs/<SHA_A>/<RUN_2>/             # 显式重试的新目录
```

不能从文件名 `all_results.json` 存在就判断整个 Attempt 成功；某个 quality workload 完成后，后续 workload 或 timing 仍可能失败。应结合 Attempt 状态、对应 grader run 状态、contract 与硬件可比性核实。

分多次启动仍可使用同一个 Run：

```bash
coral stop --task <task> --run <run>
coral resume --task <task> --run <run> agents.dynamic.max_engineers=0
```

沿用原 `.coral`、repo 和固定评估协议，节点编号和 artifacts 的身份关联就能继续。适合停止的边界是正式评估和 Analysis 发布完成、Research assignment 已 completed。`max_engineers=0` 控制新增 Engineer 容量；已有活跃 assignment 不会因为降低容量自动消失。

`max_researchers=2` 只是并发容量，调度器会为完成者补充新节点；当前没有“本阶段只新增两个节点并自动停止”的持久配额。暂停 Run 与中断单次训练不同：正式协议不保存可恢复训练的模型/优化器 checkpoint，训练中断后不能从原更新步续跑。Quality 的 W&B 被禁用；默认也不生成独立结构化完整训练曲线、profiler trace 或模型权重。`activation_checkpointing` 是节省显存的重计算机制，不是权重保存功能。

改变训练预算、质量门槛等会改变 contract hash，旧基线和旧结果不能直接重标或混用。Dense 参考 JSON 是评分输入，包含质量基线或耗时与环境身份；其生成阶段也应保留原始日志。宿主准备基线时使用的外部 staging 目录名称不属于统一的公开 artifact 路径约定。

备份可恢复的 Run 时，应保留完整 Run 目录，包括 `.coral/private` 中的权威状态和基线、公开证据以及 Git 对象；只复制 `notes/` 无法重建正式身份与工作流。对外分享时则需单独选择公开产物，私有状态不属于 Agent 可见输入。

## 8. 阅读与核实入口

- 看节点定义与结论：`notes/nodes/<Node>/spec.md`、版本化 Analysis。
- 看该节点所有正式结果：`evaluation.md` → `attempts/<SHA>.json`。
- 核实某次测量：匹配 grader run ID，查看 `all_results.json`、`process.json`、stdout/stderr 和资源事件。
- 判断总体方向：参考 Summary，再回到其 `processed_analyses` 指向的证据。

对应实现：

- [持久评估目录](../coral/grader/task_grader.py)、[Attempt 存储](../coral/hub/attempts.py)、[dispatch 与可信发布](../coral/grader/trusted_writes.py)。
- [任务 grader](../tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/grader.py)、[Quality 命令](../tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/quality.py)、[Efficiency 命令](../tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/efficiency.py)。
- [质量结果构造](../tasks/greedylore_autoresearch/seed/c4/run_llama_pretraining.py)、[测速结果构造](../tasks/greedylore_autoresearch/seed/c4/table_v_timing.py)。
- [canonical artifact 路径](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/artifacts.py)、[Evaluation](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/evaluation.py)、[Analysis schema](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/validation.py)、[Memory](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/memory.py)。
- [诊断身份与记录](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/diagnostics.py)、[诊断执行与归档](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/diagnostic_sandbox.py)、[工作流状态与调度](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/state.py)。
