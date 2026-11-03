---
name: starting-coral-autoresearch
description: 用于准备 CORAL AutoResearch 的 Root 基线、启动正式研究、从已有 Root 继续运行，或切换运行中的角色模型。
---

# 启动 CORAL AutoResearch

使用保存的合同、代码和运行状态启动研究。区分配置已修改、进程已启动和任务实际推进，保留已有实验的身份与证据。

## 启动前确认

先读取现有配置和会话中的明确选择，只询问尚未确定的内容：

| 项目 | 确认内容 |
| --- | --- |
| 启动范围 | 使用哪个 seed、新建还是恢复哪个 run；只执行 Root，还是运行完整研究流程 |
| 角色模型 | Researcher、Analyzer、Engineer、Memory／Summary／Wisdom 的完整模型 ID 和推理强度 |
| Node 并行度 | 最多同时推进多少个 Node |
| 任务并行度 | 正式评测同时执行多少个 Attempt，以及托管训练、测速、诊断的实际并行方式 |

模型与强度一次询问完整，可以统一设置或按角色分别设置。例如：

> 这次 Researcher、Analyzer、Engineer 和 Memory 分别使用什么模型与推理强度？Node 并行度和正式评测任务并行度各设多少？

新 run 在模型选择明确后启动。恢复已有 run 时展示已保存的角色配置；用户明确沿用时直接采用。本会话已指定的选择不重复询问，不以 CLI 默认模型填补缺失选择。将 `mid` 转换为实际配置值 `medium`，保存模型的完整 ID。

Node 并行度、正式评测并行度、实际 GPU 作业并行度分别核对。`grader.parallel.max_workers` 不一定限制所有托管训练或诊断；GPU reservation 和每个作业所需卡数会约束实际同时执行的作业数。若 Node 模式会覆盖角色容量设置，说明真正生效的限制，避免询问无效参数。

## 准备与检查

- 查看仓库指令，以及 [工作流与评估协议](../../../docs/greedylore-autoresearch-workflow.md)；用当前源码核对入口和配置含义。
- 对照开发仓库、部署快照和 run，记录 Contract、seed 来源、候选 SHA、部署补丁及已接受请求的身份。
- 自行检查离线数据、Python 环境、CLI 认证、sandbox 依赖、磁盘和 checkpoint 空间，以及本次 run 的进程和 GPU reservation。
- 记录 checkpoint 位置与保留方式。`/dev/shm` 中的张量不能保证跨机器重启恢复；持久状态和日志应留在磁盘。
- 使用本次 run 的独立 CLI 配置，沿用必要认证，避免继承无关用户设置；认证内容不进入日志或提交。

优先使用仓库提供的准备、评测和恢复入口。[prepare_seed_run.py](../../../tasks/greedylore_autoresearch/scripts/prepare_seed_run.py) 只准备 Root 和待执行 Attempt；准备成功不表示训练或 Agent 已经启动。

### 诊断环境检查

首次启动完整研究（包括从 Root-only 转入），或更换部署代码、解释器、运行库、挂载、PATH／socket 配置时，按以下步骤验证。机器路径保存在本 Run 的 `.coral/config.yaml`；优先参考已验证 Run 的配置，核实本机路径和依赖后再采用。

1. 核对 `grader.args.diagnostic_runtime_roots`、`diagnostic_path` 和 `analyzer_allow_unix_sockets`。公共 ML venv 及其实际 Python 安装目录必须在诊断沙箱中可见，保留解释器的符号链接目标；私有 grader venv 不作为公共诊断环境。Analyzer 的内层 socket 权限与 Researcher／Engineer 的 SRT 权限分别检查。
2. 在实际诊断命名空间内检查 Python 和所需 ML 依赖。复用 [diagnostic_runtime.py](../../../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/diagnostic_runtime.py) 的 CPU probe；需要补齐配置时可使用准备入口的 `--diagnostic-python <public-venv>/bin/python`，活动 Run 先按 `updating-live-runs` 停止。更换环境时一并检查旧挂载与 PATH。主机导入成功或正式 Root 评测成功不能代替诊断沙箱验证。
3. 在首个获准启动的诊断角色中，通过其正常入口执行最小托管诊断，核对请求被宿主接收、调度执行、成功退出，以及返回的 `record_ref`、日志与结果归档。核实本次启用角色各自的控制通道；CPU probe 不证明 Analyzer socket 或完整托管链路可用。优先让首个实际诊断同时承担此验证，执行遵循原有 GPU 调度与诊断预算。
4. 在 Run 的持久操作记录中保存部署版本、解释器与环境配置、检查命令及输出、退出状态和托管诊断引用，并给出准确恢复入口。只有链路验证通过后才报告诊断环境可用；这些记录不作为正式质量或性能结论。

同一环境恢复时，核对部署与上述配置、解释器及依赖、挂载路径仍与成功记录一致，并轻量复查 CPU probe；能够核验时复用已有托管诊断证据，不额外消耗诊断额度。环境变化或缺少可核验证据时，重新完成上述检查。Root-only 仅检查本次评测所需环境，诊断链路验证留到转入完整研究时，不为检查启动 Analyzer／Engineer。

## 建立或接续 Root

新 Root 按固定合同建立匹配的 Dense 参考并执行评测。接续已有 Root 前，逐项检查已完成、失败和仍在执行的组件。

复用结果要求完成状态、Contract、预算、指标与来源身份均可核验。代码发生变化时，还需验证对应执行路径等价，并保留原始测量 SHA、结果哈希和复用依据。补测一个组件不能直接记作完整 Root 成功。

候选身份变化时保留失败记录，使用正式恢复接口衔接。必要的操作方修复应在明确的中断边界执行，保存修复前状态、使用相应锁并留下审计记录；不改写历史测量身份，不重置预算，也不放宽准入检查。

## 切换配置与会话

切换活动 run 时使用现有 `updating-live-runs` skill。修改文件不会改变已启动会话的模型或已加载代码。

用户要求立即切换时，停止本次 run 所属的相关会话及必要控制进程，保留任务、代码、请求、评测结果和预算，再部署并恢复。

检查实际启动参数：

- Researcher／Engineer：`agents` 默认值及各角色 assignment 的模型、`model_reasoning_effort`。
- Analyzer：`grader.args.analyzer_model`／`analyzer_reasoning_effort`，同时检查带诊断和只读入口。
- Memory／Summary／Wisdom：`memory_model`／`memory_reasoning_effort`，确认参数实际传到模型命令。

CLI 配置目录变化后，核实旧 session 能否恢复。失效时清理恢复索引并启动新会话，保留任务与历史记录，避免不断重试同一个失效 ID。

用户要求废弃旧 Analysis 时，移除正式报告与会被自动采用的旧输出缓存，再安排重做；保留训练、测速和预算。仅修改模型配置不会自动重做已发布的 Analysis。

## 启动并验证

- Root-only：完成指定 Root 评测后停止，不启动 Analysis、Engineering 或后续研究节点。
- 完整流程：Root 满足工作流准入条件后，启动 Analysis、Engineering 和后续研究。

分别核实保存配置、部署代码、实际模型参数、有效模型响应，以及受管作业的日志或步数推进。PID 存活或状态字段为 `running` 不能单独证明任务正常执行。

交付时说明当前阶段、已启动角色及实际模型、证据和日志入口，以及使用同一环境恢复该 run 的准确入口。区分启动流程已开始与所有角色已经实际工作，不把 CPU／模拟检查当作正式 GPU 质量或性能结论。
