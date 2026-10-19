# CORAL × GreedyLoRE AutoResearch 架构

本文描述当前仓库中 GreedyLoRE AutoResearch 的实现，按职责划分为七个逻辑模块，说明各模块的输入、输出、代码位置和协作边界。这些模块并非七个独立部署的服务；部分职责由同一文件承担。

## 1. 总体结构

CORAL 提供 Agent 进程管理、Git worktree、评分队列、沙箱、共享状态和查询界面。GreedyLoRE 在此基础上增加研究树、任务状态机、Researcher / Engineer 分工、固定评估协议，以及独立分析和全局记忆。

任务配置入口是 [task.yaml](../tasks/greedylore_autoresearch/task.yaml)。任务专用的工作流实现在可独立安装的 [autoresearch 包](../tasks/greedylore_autoresearch/autoresearch/pyproject.toml)中；CORAL 核心通过 dynamic assignment 和可信命令接口接入它。

| 模块 | 核心问题 | 主要输出 |
| --- | --- | --- |
| 研究树与搜索策略 | 下一步研究什么、从哪个 Parent 出发？ | Node Spec、Parent 和代码基点 |
| 工作流调度 | 谁在何时执行哪个 assignment？ | assignment roster、生命周期状态 |
| Agent 执行管理 | 如何启动、隔离、恢复和停止 Agent？ | Agent 进程、worktree、Session 日志 |
| 可信提交与产物发布 | 谁可以提交或发布什么？ | 经校验的 Attempt 和共享产物 |
| 实验评估 | 候选是否满足质量要求、效率如何？ | ScoreBundle、原始测量和评估证据 |
| 分析与研究记忆 | 结果说明什么、下一轮应知道什么？ | Evaluation、版本化 Analysis、Summary |
| 状态存储与可观测性 | 如何保存、追溯和展示整个过程？ | 持久状态、实验历史、CLI 和 Dashboard 视图 |

```mermaid
flowchart TD
    A[研究树与搜索策略] --> B[工作流调度]
    B --> C[Agent 执行管理]
    C --> D[可信提交与产物发布]
    D --> E[实验评估]
    E --> F[分析与研究记忆]
    F --> A
    F -->|完成 assignment| B
    B -.-> G[状态存储与可观测性]
    D -.-> G
    E -.-> G
    F -.-> G
```

## 2. 研究树与搜索策略

该模块把实验组织成有显式 Parent 的研究节点。每个 Node 对应一个算法假设，Spec 记录 Parent、固定代码基点、选点理由、改动、风险和证伪条件。节点内部的实现过程与研究树上的一次扩展是不同概念。

运行从固定 Root `N000` 开始。Root bootstrap 固定来源信息、S0/A0、Spec 和基线 Attempt。动态调度验证这些身份后即可启动；Root 的 Research Attempt 终态且无剩余基础设施重试后，启动独立 Research Analysis 和质量达标时的 Engineering。N000 两条分支完成后才创建首批 Researcher。独立 baseline-only 工具只生成基线证据，不代表整个 Node 已完成。普通 Researcher 获得系统分配的 Node ID 后，自主选择已完成且有有效 Research Analysis 的 Parent，再调用 `autoresearch node initialize` 固定 Spec 和基点。

Parent 选择考虑预期收益或信息价值、已有证据、探索程度、验证成本和质量风险。当前没有集中式 UCT / MCTS 选点公式，也不强制从最高分节点扩展。质量失败仍可产生 Research Analysis，后续 child 可以研究修复或新假设。

输入是现有 Tree、Spec、Evaluation、Analysis 和代码快照；输出是新节点的研究定义。Engineering 在原 Node 上增加实现优化结果，不替换 Research 成果，也不创建新的研究树节点。Child 的代码基点取 Parent 的固定 Research SHA，Engineering Analysis 可以作为选点参考。

主要代码：[nodes.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/nodes.py)、[state.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/state.py)、[researcher.md](../tasks/greedylore_autoresearch/roles/researcher.md)。

## 3. 工作流调度

任务级调度器是 `WorkflowStore`。它管理节点预留、Research assignment、按 Node 绑定的 Engineering assignment、完成条件和编号分配。当前配置允许最多两个 Researcher 和一个 Engineer；容量由 `task.yaml` 指定。

CORAL Manager 周期性请求 `reconcile_workflow`。可信服务调用任务的 `handle_workflow()`，由 `WorkflowStore.reconcile()` 返回期望的 assignment roster。Manager 将 roster 与实际 Agent 对照，创建缺失的 Agent，保留存续者，并停止进入 finalizing 或已结束的生产 Agent。

对 Manager 而言，活跃 assignment 的主要阶段是 `reserved → active → finalizing`；完成后不再出现在所需 roster 中。finalizing 表示任务已请求收尾，禁止继续提交候选，宿主负责生成分析并完成状态转换。Research 在唯一正式 Attempt 已终态且没有剩余基础设施重试时冻结结果并退休，同时独立推进 Research Analysis 和条件 Engineering；Engineering 由 `autoresearch engineering finish` 发起收尾。Researcher 与 Engineer 的容量分别保留，Engineering 容量不足时按持久化进入队列的顺序等待，不能跳过。Node 只有在 Research Analysis 发布且 Engineering 完成或明确因质量失败/不可用而跳过后才可作 Parent。工作流状态使用 v2，v1 不自动迁移。

因此，调度分为三层：GreedyLoRE 决定研究 assignment，CORAL Manager 管理执行进程，grader daemon 管理评估队列。普通模式的固定长期 Agent 循环被 dynamic 分支替代，但进程管理和评分队列仍复用 CORAL。

主要代码：[state.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/state.py)、[lifecycle.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/lifecycle.py)、[dynamic.py](../coral/agent/dynamic.py)、[manager.py](../coral/agent/manager.py)。

## 4. Agent 执行管理

该模块将 assignment 转成实际运行环境。CORAL 为 Agent 建立独立 Git worktree，配置虚拟环境、共享目录链接、运行时设置和沙箱，再启动 Codex 等外部 CLI 子进程。动态 assignment 只提供身份、角色、Node 和状态；运行时和角色文件来自宿主配置。

Researcher 负责一个算法假设和一个正式 Research 候选。宿主把 Engineer 绑定到一个质量达标节点及其固定 Research SHA；`engineering claim` 仅确认绑定并准备 Worktree，不能自主更换目标。Engineer 不等待该 Node 的 Research Analysis，从固定 Research SHA 出发进行语义保持优化。Engineer 的角色规则规定每次认领最多两轮正式候选；提示词规则与服务端硬约束需要分别看待。

dynamic 模式生成精简的任务专用指令，指向对应角色文件。当前 Codex runtime 将其写为 worktree 根目录的 `AGENTS.md`，不会加载通用长版 `CORAL.md` 模板。角色文件再指引 Agent 阅读共享工作流规则。

进程异常退出时，Manager 尝试恢复相同身份与 Session；节点完成后则退休原生产 Agent，后续节点启动新身份。当前任务关闭通用 heartbeat。独立 Analyzer 使用新会话，不属于生产 Researcher / Engineer 的持续对话；受控诊断模式使用按 Node/mode/version 固定的独立 workspace，未启用时保留原临时目录模式。

主要代码：[manager.py](../coral/agent/manager.py)、[codex.py](../coral/agent/builtin/codex.py)、[project.py](../coral/workspace/project.py)、[worktree.py](../coral/workspace/worktree.py)、[coral_md.py](../coral/template/coral_md.py)、[角色规则](../tasks/greedylore_autoresearch/roles/workflow_rules.md)。

## 5. 可信提交与产物发布

该模块连接 Agent 的请求与宿主持有的权威状态。Agent 在自己的 worktree 中编辑代码，通过 `coral eval` 提交候选，通过 `autoresearch` 命令初始化节点、认领任务、请求收尾或发布产物。

可信服务通过本地 Unix socket 接收命令，结合调用进程身份和宿主登记的 Agent 信息确定归属。任务校验器读取固定 commit 中的 `submission.yaml`，检查 Node、mode、基点、quality anchor、证据引用等；授权器检查当前 assignment 及其是否已进入 finalizing。Research 同一节点的不同正式候选 SHA 也在提交入口受到限制。

Spec、Evaluation、Analysis 和 Summary 等产物经过目标路径和内容校验后发布。私有状态和评估数据由宿主持有，Agent 不直接修改权威 Tree 或原始评分记录。同 SHA 的基础设施故障使用 retry 路径，保留运行证据；算法或候选失败作为研究结果保留。

这部分不仅依赖提示词。`task.yaml` 中的 `trusted_workflow_*`、`trusted_submission_validator`、`trusted_artifact_*` 和 `trusted_memory_*` 入口将具体规则交给任务包，在可信执行环境中运行。

主要代码：[trusted_writes.py](../coral/grader/trusted_writes.py)、[trusted_worker.py](../coral/grader/trusted_worker.py)、[post_commit.py](../coral/hooks/post_commit.py)、[submission.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/submission.py)、[artifacts.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/artifacts.py)、[client.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/client.py)。

## 6. 实验评估

CORAL grader daemon 发现 pending Attempt 后，在对应 commit 的独立 detached worktree 中执行评分。评分程序通过独立 grader 虚拟环境加载，避免生产 Agent 后续编辑影响已提交候选。当前任务设置一个评估 worker，因此正式评估串行执行。

任务 grader 解释固定 `eval_contract.yaml`，组织训练质量检查和四卡效率测量。`gpu.py` 探测和预留 GPU，并校验 GPU 型号、拓扑、NUMA 和 NCCL 环境的可比性；`process.py` 管理评估子进程；`quality.py` 和 `efficiency.py` 构造固定命令。

当前探索预算：60M AdamW 保留 8,393 步学习率调度，只训练 4,000 步；130M Muon 保留 20,000 步调度，只训练 2,500 步。两项 Quality 均比较训练结束时的 validation loss（`final_eval_loss`），不使用 training loss、历史最低 loss 或 PPL 替代。Dense 对照必须采用相同截断步数和调度；修改预算会改变 contract hash，旧 8,393 步的 60M 基线不能直接复用为 4,000 步基线。这是短预算探索证据，不代表完整训练的最终质量。Timing 协议不变。

当前 Quality 门槛为两项 workload 的 final validation loss 均不超过对应 Dense 的 1.06 倍（最多退化 6%）；评分器读取 Contract 中的 `quality.max_relative_eval_loss`。门槛也参与 Contract hash，不能覆盖或重标此前 10% 门槛下的 Attempt 与 Dense 记录。角色和运行时提示使用英文；Analysis 仍保存为 schema 约束的 JSON，UTF-8 文本不转义非 ASCII 字符。

Research 模式同时评估质量与效率。质量门槛失败时，汇总分数为零；效率使用 BF16、FP32 相对 dense 基线加速比的几何平均。Engineering 模式只执行效率评估，质量指标标为 `inherited (provisional)`，不代表重新实测。硬件不可比时不产生有效汇总效率分数。

输出包括 `ScoreBundle`、反馈、失败类别、协议 hash、硬件信息和原始日志。持久证据写入专门的评估日志目录；评分 checkout 会被清理，不适合作为长期产物存储位置。

主要代码：[daemon.py](../coral/grader/daemon.py)、[subprocess_grader.py](../coral/grader/subprocess_grader.py)、[任务 grader](../tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/grader.py)、[gpu.py](../tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/gpu.py)、[固定协议](../tasks/greedylore_autoresearch/taskdata/eval_contract.yaml)。

## 7. 分析与研究记忆

该模块将评分证据转成下一轮可使用的知识。`evaluation.py` 从正式 Attempts 和原始日志重建 Node 的 `evaluation.md`；它是确定性的结果索引，包含失败和退化结果，不只展示最佳候选。

`lifecycle.py` 在现有后台 worker 中处理 Research Analysis 队列，准备固定证据并调用 `analysis.py`。独立 Analyzer 使用新会话，读取 Research SHA 的代码快照、Spec、Research Evaluation、相关 Analysis 和原始证据，根据 Research prompt 输出结构化正文。宿主校验并发布版本化 Analysis。Engineering Analysis 仍由 Engineer 自己撰写；短的 `reconcile()` 负责发布已封存的总结，不等待长时间分析。任务固定 SHA、证据 hash、版本和执行代次；失败后退避并继续后续任务，已封存输出和已发布 Artifact 可以恢复复用。停止/重启通过 PID start time 和进程组登记清理旧执行。

Research Analysis 关注质量变化、算法机制、竞争解释、局限和可证伪的 child 建议。Engineering Analysis 关注结构开销、可优化实现开销及尚未证实的性能归因，同时审视全部工程候选。Research Analysis 与 Engineering 独立推进；两者都收尾后才能扩展该 Node。Researcher 可以从其他已完成 Parent 继续研究。

### Analyzer 诊断实验

正常 Analyzer 启动时把本次 Attempt 记录的 `evaluation_context.json` 渲染为英文 prompt 段落，介绍真实 quality/timing 配置、quality 门槛和匹配的 Dense 比较值；不读取当前配置来填补历史缺失字段，也不开放 grader 私有目录。旧 Run 无此文件时明确报告 unavailable。

支持 checkpoint 的新 seed 在每个 quality workload 的最后一次更新后、最终验证前保存一个分布式 checkpoint，位于该 grader run 的 `checkpoints/<workload>/`。模型保存一份，各 rank 分别保存 optimizer/scheduler、可序列化的 compressor 状态、RNG 和最后一批训练输入。全部 rank 保存成功后才发布完整 manifest；磁盘不足、未知状态或单 rank 写入失败会使 checkpoint unavailable，不改变 quality 指标。Timing 不保存，也不启用 checkpoint hook 包装。旧候选缺少保存接口时保留原 quality 命令。

Analyzer 及其诊断进程只读挂载当前 Attempt 的完整 checkpoint。小型 manifest 进入固定证据，大型 tensor 文件不复制进 evidence/workspace 或 64 MiB 诊断归档。加载辅助函数 `diagnostic_checkpoint.load_checkpoint` 位于匹配 SHA 的代码快照；恢复须匹配模型、优化器、world size 和 DDP bucket 布局，不隐式清零 EF，也不承诺恢复原数据流位置。可读取现成 checkpoint，也可通过现有权威诊断命令进行预算内的小规模训练；实验设计仍以 Reference 为准。当前不实现 checkpoint 自动删除，已完成 Analysis 仍保留其 checkpoint。

关键证据不足时，Analyzer 可在同一 Session 中编写小型诊断脚本，通过权威入口申请 GPU；它不是正式 Attempt，不改变质量、评分或 Engineer 两轮规则：

```bash
autoresearch diagnostic run --gpus 4 --timeout 300 -- \
  python experiments/probe.py --output results/probe.json
```

每个分析版本最多成功执行 3 次、总执行 10 次（含失败、超时和执行中断），每次按需申请 1–4 卡、运行最多 600 秒，单次排队最多 3600 秒。排队失败不消耗执行次数；重复使用 `--request-id <原ID>` 查询同一请求不会再执行。失败后可查看日志、修复诊断脚本或启动命令，再用新 request ID 重试，所有失败、重试和退出码均被记录。重启同一分析版本不重置预算，正式 Analysis 已发布时仍直接复用。

工作区位于 `.coral/public/diagnostics/<Node>-<mode>-v<version>/workspace/`。只有 `experiments/` 和 `results/` 可写，固定 `code/`、`evidence/` 只读；兄弟任务、私有状态和未分配 GPU 不挂载。取得 GPU 后冻结可写输入，实验执行冻结副本，仍使用原来的命令和 workspace 路径，避免排队期间的修改污染证据。结果安全合并拷回 workspace，具体归因以本次持久记录为准。`records/<request-id>/` 保存脚本和输入快照、命令、固定证据身份、GPU、运行时间、stdout/stderr 及结果副本，供 Analysis 引用。每类归档上限 64 MiB / 4096 条目，超出时明确记录归档失败；不承诺完整保留超大输出。普通 Researcher/Engineer 不能改写诊断目录。

启用条件是 `grader.args.analyzer_diagnostics: true`、可用的 bubblewrap namespace，以及显式配置的公共运行依赖。`diagnostic_runtime_roots` 只读挂载所需 Python 环境（含其公开的解释器/链接目标）、CUDA 依赖、本地 C4 和 tokenizer；`diagnostic_path` 指向该环境的 bin 目录。不要挂载整个 home、run、其他 workspace、凭据或 grader 私有环境。它们是宿主配置，不由 Analyzer 请求指定；不会自动安装依赖或下载数据。未安装 bubblewrap 或 namespace 探测失败时保留普通分析并提示诊断未启用；已启用但运行环境不完整时诊断明确失败，不直接在宿主执行命令。

诊断执行器按已分配 GPU 的 UUID 从宿主 `/proc/driver/nvidia/gpus/*/information` 解析设备 minor，再挂载对应 `/dev/nvidiaN`；不能把 `nvidia-smi` index 当作设备节点编号。`CUDA_VISIBLE_DEVICES` 仍按已分配 UUID 筛选，记录同时保留 index、UUID 和 device minor。映射缺失或不唯一时释放预留并明确失败，不猜测或暴露额外 GPU。诊断沿用 `pretraining_allow_other_users` 共享开关；开启时不执行 timing 专用的独占检查，正式 timing 仍要求独占。

可通过 `analyzer_model` 和 `analyzer_reasoning_effort` 指定诊断 Analyzer 的模型和推理强度。Codex 从可写的 `results/` 启动，并通过 `--add-dir` 获准写 `experiments/`；外层 workspace 根目录及 `code/`、`evidence/` 仍只读。提示中提供 workspace 和固定输入的绝对路径；权威诊断命令仍以 workspace 根目录为工作目录，与 Codex 的初始目录不同。这样避免 Codex 内层 sandbox 在只读根目录创建 `.git`、`.aws` 等保护路径时失败。真实 Codex smoke 已验证读取固定输入、写脚本/结果及拒绝修改证据；该 smoke 不申请 GPU，不能替代 GPU 诊断端到端验证。仅通过 namespace 探针不代表模型工具可用，工具启动失败不得视为有效研究结论。

本机 Codex 0.160.0 的工具网络过滤器拒绝 Unix socket 创建，单路径 allow 配置在本机探针中未解除限制。可在单次隔离运行中显式设置 `analyzer_allow_unix_sockets: true`（任务模板默认 false）：启动器仅对该 Analyzer 启用 Unix socket 例外，保留工具网络代理和空外部域名允许列表，不关闭内层 sandbox。该例外允许所有可达 Unix socket，而非单路径授权，不能在非隔离进程中复用。真实 Codex CPU smoke 已验证可信请求、证据只读及工具 HTTP 被拒绝；v6 随后通过同一权威入口完成了四卡机制诊断。模型自行写出的请求失败说明不能替代服务端执行记录。公共 Python 运行时挂载还须包含解释器符号链接的中间路径，不能只挂载最终解析到的版本目录。

启动器显式转发已有的大小写 HTTP/HTTPS/ALL/NO_PROXY 环境变量给 Codex，以保留模型 API 的代理配置；不记录代理值或凭据。GPU 诊断执行器仍单独构造净化环境，不继承模型的 API 密钥或代理。代理类型/配置是否可用须以真实模型连接验证，不能由配置字段存在推断。

现阶段 CPU/可信 socket/预算与回归测试已覆盖；本机通过用户目录中的 bubblewrap 完成了真实 Codex namespace smoke 和一次四卡合成梯度诊断。该结果验证了本机的执行链路，不代表其他部署环境、所有代理配置或完整训练行为均已验证。

`memory.py` 在累计达到条件时整理全局记忆，发布 `research_summary.md` 和 `engineering_summary.md`，并记录所覆盖的 Analysis 引用。Summary 帮助新 Agent 定位证据，具体判断仍需追溯到固定 SHA、Analysis 和原始记录。

主要代码：[evaluation.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/evaluation.py)、[analysis.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/analysis.py)、[memory.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/memory.py)、[research_analysis.md](../tasks/greedylore_autoresearch/prompts/research_analysis.md)、[engineering_analysis.md](../tasks/greedylore_autoresearch/prompts/engineering_analysis.md)。

## 8. 状态存储与可观测性

Artifacts 的完整目录、文件名含义、内容格式、重试和分阶段恢复规则见 [artifacts 组织与保存说明](greedylore-autoresearch-artifacts.md)。该文包含 Quality、Timing、Attempt、Analysis 与诊断归档的示例。

系统以文件和 Git 保存状态，不依赖独立数据库。不同记录承担不同职责：

```text
<run>/
├── repo/                              # 共享 Git 对象与代码历史
├── agents/<agent_id>/                 # 生产 Agent 的独立 worktree
└── .coral/
    ├── private/
    │   ├── autoresearch_state.json    # 权威工作流状态
    │   ├── root_bootstrap.json        # 固定 Root 策略
    │   └── ...                        # grader 环境、私有数据等
    └── public/
        ├── attempts/                  # 正式实验及其运行元数据
        ├── eval_logs/                 # 持久评估证据
        ├── logs/                      # Agent Session 日志
        └── notes/
            ├── tree.md                # 工作流状态的树视图
            ├── nodes/<node_id>/       # Spec、Evaluation、版本化 Analysis
            ├── research_summary.md
            └── engineering_summary.md
```

`tree.md` 是私有工作流状态的公开投影，`evaluation.md` 是实验记录的结果投影；原始 Attempts 和日志用于核实评估证据，Git SHA 用于固定代码身份。Analysis 则在这些事实之上给出解释。恢复和审计时应区分这些层次，避免把摘要当成原始结果。

CORAL CLI 提供实验、日志、笔记和状态查询；Starlette 后端与 React Dashboard 提供排行榜、提交关系和日志视图，通过 SSE 推送变化。通用 Dashboard DAG 根据 Attempt 的 Git parent 画图，与任务按 Node Spec 定义的研究树不是同一张图；`improved` 标签也不等于通过质量门槛。

主要代码：[hub/](../coral/hub/)、[types.py](../coral/types.py)、[web/api.py](../coral/web/api.py)、[web/events.py](../coral/web/events.py)、[前端](../web/src/)。当前 GreedyLoRE 动态工作流只支持单岛，不能直接套用 CORAL 多岛迁移语义。

## 9. 一次完整研究循环

1. 宿主完成 Root bootstrap，固定 seed 和基线评估；Root Analysis 与条件 Engineering 收尾后开放首批 Researcher。
2. Manager 请求任务 roster，启动预留的 Researcher。
3. Researcher 选择 Parent 并 initialize，任务固定 Spec 和 Research 基点。
4. Agent 实现候选、记录自检，通过可信提交入口创建正式 Attempt。
5. Grader daemon 执行固定评估，保存分数、反馈与原始证据。
6. 任务 worker 在可收尾时冻结 assignment，生成 Evaluation 和 Research Analysis。
7. 节点完成，原 Agent 退休；新 Researcher 可扩展该节点。质量通过的非 Root 节点还可被 Engineer 认领。
8. Engineer 从固定 Research SHA 优化并评估，调用 finish 后由系统生成 Engineering Analysis 并退休。
9. 全局记忆达到触发条件时更新 Summary，供后续搜索使用。

## 10. 实现边界与阅读注意

七模块是职责划分，不表示代码已完全解耦：`state.py` 同时处理研究树约束和工作流调度，`lifecycle.py` 同时衔接任务完成、分析和记忆。角色规则也不都等价于服务端硬约束，例如 Engineering 两轮上限主要由角色指令表达。

通用机制与任务语义仍有需要注意的接缝：Research 提交校验目前要求候选直接 Git parent 等于 Spec 的 Parent SHA，与正式提交前允许普通 commit 的角色规则存在冲突；dynamic Agent 仍可能收到鼓励持续实验的通用恢复 prompt；长训练需要同时考虑评分超时、CLI 等待期限和 Agent watchdog。理解这些边界有助于排查工作流问题，不能仅依据 prompt 推断系统实际行为。
