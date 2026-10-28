# AutoResearch Shared Workflow Rules

## 1. 作用与权威来源

本文件承载 Researcher 与 Engineer 共用的执行协议。角色正文说明研究或工程目标，命令参考定义请求接口，宿主记录当前状态、预算与评估契约。各角色据此自主选择问题范围内的操作。

- 以下预算为当前协议默认值；实际运行使用其保存契约、宿主状态和 claim `run_limits`，新文档不重标历史结果。
- 当前 staged Research 使用 `autoresearch research`，不以 `coral eval`、旧初始化／完成入口或独立 GPU 工作绕过协议；配置不匹配时报告问题。

## 2. 写入、资源与证据边界

在自己的 Worktree／授权 scratch 中开发代码和请求，通过任务命令发布共享记录。正式训练、测时和 GPU 诊断由宿主管理；宿主在开发及全部 Final 组件期间持有 Research 四卡 reservation，任务串行借用。

- 不直接改 Tree、Spec、Attempts、正式日志、历史 Artifact 或共享结论；private 评估材料与凭据不在授权范围。
- 使用宿主提供的 artifact／checkpoint 路径。大 tensors 按动态存储规则保存，不复制到根分区或任意位置，不擅改 retention；空间或恢复失败报告宿主。
- 正式 timing 不加入 profiling／梯度 instrumentation。完整周期、关键事件、测量窗口和硬件可比性决定成本证据的范围。
- 可选 subagent 承担实质、有界任务，继承 sandbox、评估和发布权限；简单工作自行处理。

## 3. Research 登记与设计

选题前通过 `autoresearch node planning acquire --wait` 取得 planning lease，结合最新 registry、进行中的设计与证据选择一个合格代码 Parent，判断研究重叠。

实施前用 `autoresearch research register` 保存 `origin`、初始 `design` 和 planning controls。宿主生成 Spec 及兼容 handover 字段；Researcher 不另写 research intent。设计和判断写入追加式档案，以保留原预测及其变化。

- lease 到期前续租；放弃选题时释放。登记成功释放锁，失败仍持有；失锁后重新获取并重读 registry。具体 renew／release 命令见 `references/research_commands.md`。
- `origin` 固定 Parent/base_commit、主要问题、核心假设、选择依据及 Parent／Engineering 响应；核心机制／假设改变需要新 Node。
- 设计在训练前保存，修订追加不覆盖；已声明自适应策略的正常动作不算修订。
- 使用宿主指定的 REFINE／EXPLORE 模式，跨恢复和重试保持，不重新抽样。
- 研究可从观察／问题、资源约束下的期望性质或机制假设出发。回应真实 Parent 问题才引用对应 `problem_ref`；其他起点用 null 并说明联系。
- 实施前查阅可用 Engineering Summary 与 Parent Engineering Analysis，将复用／暂缓、混杂或资料不可用原因写入 `origin.engineering_review`。

## 4. Research 开发与预算

通过 `research train`、`research diagnostic` 和 `research timing` 训练、诊断和测量，通过 `research decision` 保存继续、修订或停止的判断。开发 timing 按需使用，能在训练前、暂停后或完成后开展。

| 项目 | 默认限制与计数 |
|---|---|
| 开发训练轨迹 | 通常不超过 2，硬上限 3；第三条需在执行前说明证据支持的修正或影响选择的重要未知 |
| 设计 | 最多 3 份，包含初始设计及仅 timing 筛选的设计 |
| 开发 timing | 5 completed／15 total，重复和失败重试按协议累计 |
| Research 诊断 | 10 successful／20 total，每次最多 600 秒，失败／timeout／中断计入 |
| 基础设施恢复 | 仅宿主分类后允许，默认上限 2，受对应计数约束 |

- 失败、主动停止、rewind／restart 消耗轨迹，pending 预留容量；暂停／恢复不重置 steps、tokens、schedule 或已用额度。
- 同 iteration／SHA 精确续跑属于原轨迹，需要完整恢复 model、optimizer、compressor/EF、schedule、数据位置和 RNG。新设计从头训练，初始化与数据顺序匹配，方法状态按设计初始化。
- Parent checkpoint 换方法续训只支持局部证据，不能替代从头训练质量；只有恢复原数据位置和必要状态才能称为精确 continuation。
- 复用训练前缀仅在方法行为和必要状态一致时成立。
- 第一笔获准训练或 timing 绑定设计 SHA；改变代码需追加设计。`decision` 不执行后续动作，`design` 不修改代码或开始训练。
- 诊断只针对暂停／完成轨迹，分支不污染候选或提供额外候选训练。
- timing 默认复用匹配宿主验证的结果；刻意重测附 `repeat_reason`，使用最新完成的匹配结果，不挑最快值。完成但不可比较的 timing 仍计 completed。
- `submission.yaml` 提交 `mode: research`、Spec 引用及实际事件 telemetry；新提交的 timing 声明完整有效 `period_steps`（非周期为 null），其中 `requested_steps` 为可选补充，legacy 按保存契约处理。
- 训练前运行适用正确性检查，失败先修复；Research 新算法不要求与 Parent 语义等价。
- 保留失败、回退、不可比较和不可用证据；不改写原预测。`improved`、局部机制成功、advisory U 和未来 Engineering 预期不构成实际成功证据。

## 5. 冻结、Final evaluation 与 Node 生命周期

用 `research finish` 提名一个已完成且通过开发 validity／quality／resource 检查的 iteration，或 null。宿主先冻结选定 SHA、设计、压缩策略和 workload 映射，再运行全新 mandatory Final timing、固定语言迁移和 ResNet18/CIFAR100。

- 开发 timing 不是提名前置条件；Final timing 有独立预算，不复用开发测量，也不新增加速幅度门槛；继续检查周期／事件和硬件有效性。
- 所有开发工作先进入终态，通过 `decision` 明确停止 planned／paused／infra-failed 轨迹；无已执行候选时按接口提供代表代码 SHA。
- 候选在 `method_policy.py` 的 `configure_compression(args, workload)` 和 shared hook 定义压缩映射，适用于矩阵与卷积梯度；不改变固定数据、预算、optimizer、seed、precision 等训练／评估设置。
- 冻结后不切候选、不用 Final 反馈调参或重开开发。仅宿主认定的基础设施故障允许恢复，默认每组件最多 2 次；保留其他已完成组件，同组件复用仅用于同次冻结评估恢复。Vision 重试从头运行。
- 没有合格候选时明确跳过并保留负结果；失败或耗尽重试的组件保留终态，其他必需组件仍需终态或明确跳过。
- Research reservation 保持到全部 Final 组件终态或明确跳过后释放，随后退休 Researcher、刷新 Evaluation，独立运行完整记录上的 Research Analysis 与质量门槛控制的 Engineering。
- Root 仅一次固定 baseline Attempt A0；历史已完成 Node 不重开。Researcher 不再领取其他 Node，也不另写最终报告。
- Research Analysis 与 Engineering 独立进行；Engineering 等待容量不构成合法 skip 原因。
- Parent 资格需要 Research Analysis 发布，且 Engineering 完成或因质量失败／不可用而明确跳过。宿主最多保留 2 个未完成 Node，等待者先恢复已有任务和预算。
- 全 Node 结束且没有任务依赖大产物后，宿主清理动态产物，保留小型证据与报告；Researcher 结束不等于可提前清理。

## 6. Engineering 执行与完成

首次通过 `autoresearch engineering claim Nxxx` 确认绑定和 Research 锚点，按 Engineer 角色开展语义保持优化，用 `autoresearch engineering timing --sha <full-HEAD-SHA> --request-id <id>` 正式测时。

- 冻结 `selected_iteration` 与对应 Research SHA 是开发 Node 的依据；Root 使用 Spec／A0。handover 必须存在且一致，自适应决策、触发、状态与随机语义保持。
- `submission.yaml` 为 `mode: engineering`；`base_commit` 和 `quality_anchor` 始终是固定 Research SHA，后续可基于前一个 Engineering Commit。
- 默认 4 successful／10 total；有效得分的回退计 successful，失败和允许的重试计 total，pending 预留容量，跨恢复／rewind 不重置。任一上限达到即停。
- 当前 claim `run_limits` 优先于启动旧预算。预算内持续到没有明确、实质的语义保持端到端改进机会；一次改善不单独构成停止依据。
- 至少一个正式 Attempt 终态，且满足停止条件或上限后，由 Engineer 自己撰写覆盖全部 claim Attempts 的 Analysis，并用 `autoresearch engineering finish --analysis-file /absolute/scratch/final.json` 提交。
- 提交前等待诊断完成并归档必要自检；输出字段见 Engineer。宿主封存、版本化发布后才完成 claim，不运行独立 Engineering Analyzer。
- `finalizing` 不是发布完成；提交后不改总结或新增候选，中断复用相同内容。

## 7. 等待、重试与证据归档

执行命令默认等待已发布结果。连接中断后，Research 使用 `research wait --node Nxxx --request-id <原ID>` 重新连接，`research status` 找回状态，`research recover` 请求处理中断的宿主任务。

- 使用当前工具支持的阻塞等待，任务返回后继续，避免另写 watcher、短轮询或周期读日志来等待。
- lost reply 复用相同 ID 和 payload；新的执行使用新 ID。先检查失败原因，保留失败记录，不盲目重跑或空提交重试。
- Engineering 同 SHA 基础设施重试使用 `--retry` 与新 ID；丢失回复复用原 ID／SHA／flags。Research／Final 恢复按相同冻结版本，不重跑已完成工作。

使用 `autoresearch evidence archive --node Nxxx --request-id <id> --files <路径...>` 归档后续需要引用的小型自检脚本和日志。Research 在托管实验前且 assignment 活跃时归档，Engineering 在 finish 前归档。

- 每 bundle 为小型 regular files、basename 唯一、合计最多 32 KiB；每 assignment 最多 64 bundles。更大的执行证据通过托管诊断保存。
- 保存返回的 public-relative 路径，日志说明实际命令、输入和范围；临时／已退休 Worktree 不是证据锚点。
- 宿主记录认证上传者、其声明的代码上下文 SHA 和内容 hash。hash 校验内容，宿主确认 SHA 存在，但不认证这些字节实际执行过；附件作为 agent-provided evidence 进入角色冻结 manifest。

## 8. 评估与证据导航

Node README 提供 Spec、Evaluation、Analysis 版本、候选 SHA 和证据入口。固定代码可用 `git show <SHA>:<file>` 查阅；`autoresearch refresh-evaluation --node Nxxx` 重建诊断所需 Evaluation。Analysis Markdown 适合阅读，同名 JSON 保存完整冻结记录，按需查看。

- 当前 HEAD 与历史 `source_worktree` 不是恢复锚点；诊断 ID 指向 immutable records，旧 Analysis `.md` 可能包含 JSON。
- 语言任务使用同契约 dense-relative final loss ≤ 1.05，vision 默认最多下降 1.0 percentage point，可由 operator 配置；固定任务与当前契约版本以 `references/evaluation_tasks.md` 和保存契约为准。
- 新契约需要匹配 dense／Root 证据，历史 seed 不作新 reference，不编造哈希。U 仅为 advice，不决定探索、Parent 选择或提名。
- timing 使用完整周期或非周期默认窗口，通常 400、最多 800 steps；无效事件、缺失罕见事件或未匹配 reference 窗口限制长期可比结论。
- Final 结果供后续节点使用时属于已知迁移／稳健性证据，不能称为独立 campaign held-out。
- 独立 Research Analysis 回答初始与迭代预测，并提供有边界的增量发现与问题；宿主派生 `handover.json`，来源报告保持权威，child Spec 保留原快照。
