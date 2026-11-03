# CORAL AutoResearch 产物：组织、内容与保存规则

更新：2026-11-02，依据仓库实现 `64aa372`。本文适用于 GreedyLoRE 和 ARC-TopK 的单岛任务。先从 [架构](greedylore-autoresearch-architecture.md) 理解分工，从 [工作流](greedylore-autoresearch-workflow.md) 理解阶段；本文说明如何定位和解释实际记录。

## 1. 身份与证据层次

| 身份 | 含义 |
| --- | --- |
| Run | 一个持续研究工作区，保存配置、Contract（固定评估协议）、代码和状态；停止／恢复进程可沿用同一 Run |
| Node，如 `N001` | 一个核心假设及其开发、最终评估、Analysis 和 Engineering；`N000` 是固定 Root |
| 设计轮次／轨迹 | Node 内的一份冻结设计及对应训练；暂停和原样续跑仍属同一轨迹 |
| 候选 SHA／Attempt | SHA 是固定代码版本的 Git 提交标识，Attempt 保存正式提交及其评估历史；一个开发 Node 可有多个候选，最终只选一个 Research SHA |
| 执行／请求 ID | 区分一次训练、timing、诊断或 grader 调用；分阶段任务使用显式请求 ID，普通 grader run 通常生成 UUID |
| Analysis 版本 | 如 `research_analysis_v1`，与训练轮次、测速 repetition、执行 ID 分别编号 |

`tree.md`、节点 README 和 `evaluation.md` 是导航或投影，不替代原始执行证据。公开研究档案记录设计、观察、决策和最终选择；Analysis 解释这些事实，Summary/Wisdom 汇总认识。核实结论时沿引用回到固定 SHA、Contract、执行记录和结果，不能只读摘要。

## 2. Run 布局

以下为主要路径，文件在对应阶段产生，并非启动时全部存在：

```text
<run>/
├── repo/                                # Git 对象及候选代码历史
├── agents/<agent_id>/                   # 生产 Agent worktree，可退休
└── .coral/
    ├── config.yaml                      # 保存的运行配置
    ├── private/                         # 宿主持有，Agent 不直接访问
    │   ├── eval_contract.yaml
    │   ├── dense_references/
    │   ├── root_bootstrap.json
    │   ├── autoresearch_state.json
    │   ├── research_records/<Node>.json  # 研究档案权威状态与请求身份
    │   ├── research_staged/<Node>.json   # 分阶段任务 journal
    │   ├── dynamic_artifacts.json
    │   ├── checkpoint_retention.json
    │   └── memory_state.json
    └── public/
        ├── attempts/<SHA>.json
        ├── eval_logs/<SHA>/<execution-id>/
        ├── diagnostics/
        ├── logs/
        └── notes/
            ├── tree.md
            ├── nodes/<Node>/
            │   ├── README.md            # 节点产物导航
            │   ├── spec.md              # YAML，初始定义固定
            │   ├── research.json        # 宿主发布的研究档案
            │   ├── evaluation.md
            │   ├── research_analysis_v1.{json,md}
            │   ├── engineering_analysis_v1.{json,md}
            │   └── handover.json
            ├── toolkit/                 # 角色操作参考及资源目录
            ├── research_summary.md
            ├── engineering_summary.md
            └── wisdom.md
```

Agent 通过运行时共享目录访问公开文件，如 `.codex/notes/`。返回的 `eval_logs/...`、`notes/...` 引用通常相对于本 Run 的 `.coral/public/`，不表示 worktree 中又存了一份结果。代码应通过 Run 的 Git 历史和固定 SHA 查找，不依赖已退休 Agent 的 `source_worktree` 路径。

v5 Contract 的 `seed` 声明包含 ID、Root 描述、上游仓库／commit 和导入文件哈希。候选内的 `source_manifest.json` 区分 GreedyLoRE 原始快照、ARC 导入源、适配后哈希与原因。`root_bootstrap.json` 保持既有 v1 结构，其中 `source_commit` 是适配后的本地初始代码快照 S0，不能当成 ARC 上游 SHA。Seed ID 与训练随机种子也是不同概念。独立 Seed 的 Dense 目录及旧 Contract 兼容规则见工作流文档。

## 3. 研究档案与正式结果

节点研究档案位于 `<run>/.coral/public/notes/nodes/<Node>/research.json`，新注册采用 version-3 格式（档案格式版本）。`origin` 保存研究起点及来源；`iterations[]` 保存各轮设计 `design`、固定代码版本 `code_sha`、执行、观察、测时和决策；`selected_iteration` 记录最终选中的轮次，`selection_rationale` 记录选择理由，`final_evaluation` 追加冻结后的各项最终评估证据。旧设计、负结果和修正理由不覆盖。Agent 经托管命令请求宿主更新档案，不能直接编辑公开文件。

job 指一次训练、测时或诊断任务；journal 是宿主保存其请求、状态与结果的执行账本。它们与可计分的正式提交 Attempt 不一一对应，例如暂停训练或运行诊断都不等于完成正式评估。`research finish` 冻结候选后完成最终评估并发布相应正式结果，开发质量通过不等于最终评分成功。Root 的固定候选直接走正式评估。Engineering 只重新测效率，质量继承自固定 Research 代码版本，标为 `inherited (provisional)`，表示没有重新实测质量。

正式 Attempt 位于 `<run>/.coral/public/attempts/<SHA>.json`，保存状态、评分、反馈、候选归属及 `metadata.grader_runs` 等运行历史。普通 grader 的同 SHA 基础设施重试追加执行记录与日志目录，不抹去原失败。分阶段 job 的关联及恢复以请求 journal 和 `research.json` 为准；预算看宿主记录，不能按日志目录数量推算。

定位测量时，先看节点目录中的 `research.json`／`evaluation.md` 及 Attempt 的证据引用。下面表格中的相对路径以 `<run>/.coral/public/eval_logs/<SHA>/<执行ID>/` 为父目录：`<run>` 是运行目录，`<SHA>` 是候选代码版本，`<执行ID>` 区分这份代码的具体执行或重试。在 Codex Agent 的 worktree 内，通常可通过 `.codex/eval_logs/<SHA>/<执行ID>/` 访问同一份公开证据。

| 内容 | 作用 |
| --- | --- |
| `quality/<workload>/all_results.json` | `<workload>` 为评估任务名，如 `llama130m_muon`；保存语言最终 validation loss、步数及训练配置，vision 保存对应 validation 指标 |
| `efficiency/r1-350m_adamw_bf16/` 等 | 350M 模型、AdamW、BF16 下的测时原始结果及配置；`r1` 是第一轮，第二轮 `r2` 按相反精度顺序执行 |
| `*.process/` 中的 `process.json`、stdout/stderr | 实际命令、退出状态及超时情况；stdout/stderr 分别是标准输出／错误输出日志。部分 staged 执行还在执行目录根部保存 `process.json` |
| `evaluation_context.json`（相应评估路径提供时） | 该次固定评估的任务配置、门槛及匹配 Dense 信息 |
| GPU 资源事件、来源身份及 checkpoint manifest | 记录使用的 GPU、实际代码版本，以及 checkpoint 文件清单和哈希；具体文件位置按该次执行返回的证据路径查找 |

实际目录名由该次执行返回的证据路径确定；旧 v1–v3 timing 仍可能含 `130m_adamw_*`。staged 执行 ID 可来自请求 ID 并带阶段后缀，不必是 UUID。某个 `all_results.json` 已存在、stderr 非空或单个组件 completed，都不足以判断整个评估成败，应结合终态、失败类别、Contract、完整配方与硬件可比性。Timing 是异步 DDP 端到端 step time，不是独立通信耗时，bucket 字节也不等于实际网络流量。

## 4. Analysis、Handover 与全局记忆

新 Research/Engineering Analysis 各有不可变完整 `.json` 和确定性生成的 `.md` 阅读视图，位于节点的 `notes/nodes/<Node>/` 目录。JSON 保存证据 manifest（文件清单及哈希）、Evaluation 快照及哈希；Markdown 提供结论和导航。宿主先写 JSON，再完成或修复 Markdown，验证二者一致后才算发布成功；模型 `output.json` 存在不表示发布完成。旧版把 JSON 内容保存在 `.md` 文件中（JSON-in-MD），仍可读取，不自动转换历史报告。

`handover.json` 由宿主从 Analysis 生成，绑定来源文档与哈希，供后续选题导航。Engineering 作者提交的简短总结由宿主补入固定证据并封存，不另调用 Engineering Analyzer。修订使用新版本，不能覆盖已被引用的结论。

引用来源包括固定 Git 文件、冻结输入证据及已完成的本任务受管诊断文本／脚本／日志。临时 `experiments/`、`results/`、`/workspace/...` 工作路径需先对应到持久归档；缺失证据应明确说明。文档哈希与 evidence manifest 哈希使用的 JSON 序列化规则不同，复核应使用 [共享规则](../tasks/greedylore_autoresearch/roles/workflow_rules.md) 和实现，不能互相套用。

每个 Node 完成后异步排队 Memory，单 worker 发布三份文档：研究 Summary、工程 Summary、Wisdom；不是“累计三个节点才更新”。宿主维护三者的 `processed_analyses`，全部发布成功才确认本次整理。已封存的输出可用于中断恢复而不再调用模型。覆盖记录表示已纳入的 Analysis 版本，不保证每次读取全部历史；旧 Run 缺少 Wisdom 时可补入既有覆盖。

## 5. 诊断与补充证据

受管 Analyzer 的典型归档如下，路径相对于 `<run>/.coral/`：

```text
public/diagnostics/<Node>-research-v<version>/
├── workspace/
│   ├── code/                  # 固定 SHA，只读
│   ├── evidence/              # 固定证据，只读
│   ├── experiments/           # 工作脚本
│   └── results/               # 工作输出
└── records/<request-id>/
    ├── record.json
    ├── scripts/
    ├── input-results/
    ├── stdout.log
    ├── stderr.log
    └── results/
```

这是逻辑视图；节点的大型 workspace 可位于宿主管理的动态根。引用优先使用返回的持久 `record_ref` 和归档文件。请求 ID 相同且参数相同返回原任务；改变脚本或参数需要新 ID。排队中的可写脚本不等于最终执行输入，获准执行时冻结的快照才用于归因。归档有限额，缺失或超额文件会明确记录；不保证超大输出完整保留。

Analyzer 的进程会话日志与某次诊断的 stdout/stderr 是两层记录。普通非诊断 runner 的系统临时目录不属于稳定证据归档，ephemeral 会话也不承诺完整可恢复的对话。Engineer 使用 claim 返回的本 Run 专属 scratch；Researcher 诊断另由 staged job 留存。路径和身份边界见 [运行时文档](greedylore-runtime-reliability-design-20261029.md)。

通过 `autoresearch evidence archive` 上传的自检附件带作者、SHA 和内容哈希，但属于 Agent 提供的材料；不等于宿主验证过的实际执行。受管诊断记录与附件需分别解释。

## 6. Checkpoint、清理与备份

大型 tensor、候选快照、编译及临时产物位于 `dynamic_artifact_root` 下按 Run/Node 组织的动态空间，默认 `/dev/shm/wyr_tmp`。公开目录保存小型 manifest、配置、指标、日志和归档；公开 manifest 与私有存储账本关联，不能据其存在承诺 tensor 仍在。

开发期间保留当前最佳候选及正在执行任务所需的 payload，其他终态候选可被裁剪；冻结后为 Analysis 保留选中候选的终态 checkpoint。无合格候选时保留失败代表供分析，不因此获得 Engineering 资格。普通 Node 完成且 grader、诊断与 Analyzer 排空后清理动态产物和评估 tensor；Root 保留 baseline checkpoint。宿主先记完成与删除账本，再清理，错误留待后续重试。

清理后 Spec、研究档案、Attempt、指标、日志、Analysis 和 Git 身份继续保留；Node 可作为代码／Analysis Parent，但被清理的 tensor 不能再用于续训或诊断。checkpoint 描述中的 `access`、`retention`、`diagnostic_path` 说明当前权限和可用性：manifest 可公开读取，tensor 仅由宿主选择并挂载到受管诊断。

恢复需要完整 Run 的配置、私有状态与基线、公开证据和 Git 对象，单独复制 `notes/` 不足以恢复工作流；动态 tensor 还受外部存储寿命约束。对外分享应选择公开产物，不分发凭据或私有会话目录。共享状态的 Git 快照排除 tensor 和运行缓存，仍保留 manifest、指标、日志与分析；历史大对象不会由当前忽略规则自动回收。

用户明确废弃错误产物并要求重做时，先停止所属控制进程及作业，再清除正式报告、输出缓存、派生 Node、工作树和对应动态 tensor，重建导航及交接。保留 Root 基线身份与操作审计；删除产物不恢复额度：Engineer worker 的 `prior_budget`、Research 私有档案的 `prior_trajectories` 以及 staged journal 的 `prior_budget` 分别继承已消耗的工程评测、轨迹、timing 和诊断计数。这些仅由操作方在可信状态中写入，正常请求不能指定；重新注册后的公开研究档案显示既往轨迹消耗。

## 实现入口

研究档案与执行：[research_record.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/research_record.py)、[staged_execution.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/staged_execution.py)。产物发布与导航：[artifacts.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/artifacts.py)、[navigation.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/navigation.py)、[validation.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/validation.py)。存储与清理：[runtime_artifacts.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/runtime_artifacts.py)、[checkpoint_retention.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/checkpoint_retention.py)、[node_cleanup.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/node_cleanup.py)。
