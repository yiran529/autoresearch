# CORAL AutoResearch 产物：组织、内容与保存规则

更新：2026-11-02，依据仓库实现 `64aa372`。本文适用于 GreedyLoRE 和 ARC-TopK 的单岛任务。先从 [架构](greedylore-autoresearch-architecture.md) 理解分工，从 [工作流](greedylore-autoresearch-workflow.md) 理解阶段；本文说明如何定位和解释实际记录。

## 1. 身份与证据层次

| 身份 | 含义 |
| --- | --- |
| Run | 一个持续研究工作区，保存配置、Contract、代码和状态；停止／恢复进程可沿用同一 Run |
| Node，如 `N001` | 一个核心假设及其开发、最终评估、Analysis 和 Engineering；`N000` 是固定 Root |
| 设计轮次／轨迹 | Node 内的一份冻结设计及对应训练；暂停和原样续跑仍属同一轨迹 |
| 候选 SHA／Attempt | SHA 固定代码，Attempt 保存正式提交及其评估历史；一个开发 Node 可有多个候选，最终只选一个 Research SHA |
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

v5 Contract 的 `seed` 声明包含 ID、Root 描述、上游仓库／commit 和导入文件哈希。候选内的 `source_manifest.json` 区分 GreedyLoRE 原始快照、ARC 导入源、适配后哈希与原因。`root_bootstrap.json` 保持既有 v1 结构，其中 `source_commit` 是适配后的本地 S0，不能当成 ARC 上游 SHA。Seed ID 与训练随机种子也是不同概念。独立 Seed 的 Dense 目录及旧 Contract 兼容规则见工作流文档。

## 3. 研究档案与正式结果

新注册生成 version-3 `research.json`。`origin` 保存研究起点及来源；`iterations[]` 保存各轮 `design`、固定 `code_sha`、执行、观察、timing 和决策；`selected_iteration` 与选择理由记录开发结果，`final_evaluation` 追加冻结后的最终组件证据。旧设计、负结果和修正理由不覆盖。Agent 经托管命令请求变更，不能直接编辑公开档案。

分阶段 Research 的训练、timing、diagnostic job 有独立 journal 和证据，不应假设“每个 job 都是一份可计分 Attempt”，也不应把某个候选的开发质量完成当成最终正式评分成功。`research finish` 冻结候选后完成最终评估并发布相应正式结果。Root 的固定候选直接走正式评估。Engineering 只重新测效率，质量来自固定 Research anchor，标为 `inherited (provisional)`。

正式 Attempt 位于 `attempts/<SHA>.json`，保存状态、评分、反馈、候选归属及 `metadata.grader_runs` 等运行历史。普通 grader 的同 SHA 基础设施重试追加执行记录与日志目录，不抹去原失败。分阶段 job 的关联及恢复以请求 journal 和 `research.json` 为准；不能从目录数反推消耗预算。

定位测量时，先看 `research.json`／`evaluation.md` 的引用和 Attempt，再进入相应执行目录。常见内容包括：

| 内容 | 作用 |
| --- | --- |
| `quality/<workload>/all_results.json` | 语言最终 validation loss、步数及配方；vision 保存对应 validation 指标 |
| `efficiency/r1-350m_adamw_bf16/` 等 | BF16/FP32 timing 原始结果及配置；第二轮按反向顺序执行 |
| `*.process/` 中的 `process.json`、stdout/stderr | 命令、退出状态、超时与原始输出 |
| `evaluation_context.json`（相应评估路径提供时） | 该次固定评估的任务配置、门槛及匹配 Dense 信息 |
| GPU 资源事件、来源身份及 checkpoint manifest | 资源、候选身份和恢复／保留信息 |

实际目录名由该次执行返回的证据路径确定；旧 v1–v3 timing 仍可能含 `130m_adamw_*`。staged 执行 ID 可来自请求 ID 并带阶段后缀，不必是 UUID。某个 `all_results.json` 已存在、stderr 非空或单个组件 completed，都不足以判断整个评估成败，应结合终态、失败类别、Contract、完整配方与硬件可比性。Timing 是异步 DDP 端到端 step time，不是独立通信耗时，bucket 字节也不等于实际网络流量。

## 4. Analysis、Handover 与全局记忆

新 Research/Engineering Analysis 各有不可变完整 `.json` 和确定性生成的 `.md` 阅读视图。JSON 保存证据 manifest、Evaluation 快照及哈希；Markdown 提供结论和导航。宿主先写 JSON，再完成或修复 Markdown，验证二者一致后才算发布成功；模型 `output.json` 存在不表示发布完成。旧版 JSON-in-MD 仍可读取，不自动转换历史报告。

`handover.json` 由宿主从 Analysis 生成，绑定来源文档与哈希，供后续选题导航。Engineering 作者提交的简短总结由宿主补入固定证据并封存，不另调用 Engineering Analyzer。修订使用新版本，不能覆盖已被引用的结论。

引用来源包括固定 Git 文件、冻结输入证据及已完成的本任务受管诊断文本／脚本／日志。临时 `experiments/`、`results/`、`/workspace/...` 工作路径需先对应到持久归档；缺失证据应明确说明。文档哈希与 evidence manifest 哈希使用的 JSON 序列化规则不同，复核应使用 [共享规则](../tasks/greedylore_autoresearch/roles/workflow_rules.md) 和实现，不能互相套用。

每个 Node 完成后异步排队 Memory，单 worker 发布三份文档：研究 Summary、工程 Summary、Wisdom；不是“累计三个节点才更新”。宿主维护三者的 `processed_analyses`，全部发布成功才确认本次整理。已封存的输出可用于中断恢复而不再调用模型。覆盖记录表示已纳入的 Analysis 版本，不保证每次读取全部历史；旧 Run 缺少 Wisdom 时可补入既有覆盖。

## 5. 诊断与补充证据

受管 Analyzer 的典型归档为：

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

## 实现入口

研究档案与执行：[research_record.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/research_record.py)、[staged_execution.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/staged_execution.py)。产物发布与导航：[artifacts.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/artifacts.py)、[navigation.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/navigation.py)、[validation.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/validation.py)。存储与清理：[runtime_artifacts.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/runtime_artifacts.py)、[checkpoint_retention.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/checkpoint_retention.py)、[node_cleanup.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/node_cleanup.py)。
