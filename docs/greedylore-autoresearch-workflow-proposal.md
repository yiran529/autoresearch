# GreedyLoRE AutoResearch 工作流修改方案

更新：2026-10-26。状态：设计已更新，待实施；先补代码机制，暂不替换正式 prompt。

目标是在有限搜索预算下发现 quality 与 efficiency 更好的算法。保留 Researcher、独立 Analyzer、Engineer 三个角色；Researcher 在同一上下文中完成 130M 开发、按需诊断与少量迭代，每节点开发结束后对一个冻结且合格的候选做 60M 迁移检验。使用单一研究档案串联设计与证据，将 wisdom 与 summary 分开。先采用简单规则。

本方案基于 `root-reuse-20261022` 的复盘及后续讨论，描述目标机制，不表示当前运行时已具备。现有实现见 [架构](greedylore-autoresearch-architecture.md)、[产物](greedylore-autoresearch-artifacts.md)与 [GPU 协调](../tasks/greedylore_autoresearch/GPU_COORDINATION.md)。早期方案中的开发中 Analyzer 反馈、整轮末才运行 60M，已由以下流程替代。

## 1. 研究闭环与职责

```text
选题、注册研究起点与首轮设计，释放选题锁
        ↓
申请并持有 4 卡，开始 130M 开发
        ↓
Researcher：训练 → 观察/按需诊断 → 判断预测
        ├─ 原方案：从自身完整状态继续训练
        ├─ 有依据的修正：追加设计，重新实现、自检并从头训练
        └─ 主动结束或耗尽开发额度
        ↓
冻结研究档案与一个已完成的候选；无合格候选则明确记录
        ↓
宿主：130M 合格才做一次 60M 迁移检验，否则跳过
        ↓
释放 Research 阶段的 4 卡，启动一次独立 Analysis
        ↓
按宿主规则完成或明确跳过 Engineering，发布必要证据
        ↓
节点完成、子进程退出后清理动态产物；更新 wisdom 与导航
```

- **Researcher：**从证据/问题提出方法，并检验“机制解释 → 设计回应 → 机制变化 → 质量与成本收益”的推理；自行完成开发诊断，无须等待 Analyzer。
- **Analyzer：**在开发冻结且 60M 终态或明确跳过后，独立复核机制与实际收益，形成有范围的认识及重要研究问题。复用充分证据，仅为关键未知补有界诊断，不规定新算法、不重开当前节点。
- **Engineer：**对值得优化的候选做保持算法语义的实现优化。完整 Engineering 的选择与跳过规则属于后续细化内容，不增加开发中的 Agent 交接。

两侧均可发现机制。Researcher 的诊断服务于当前开发决策，Analyzer 的诊断服务于结论可信度和后续研究。无新增 finding 也是有效结果。

## 2. 单一研究档案

新协议以宿主管理的 `research.json` 合并初始 Spec 与开发记录，避免分别维护工作笔记、实验反思和最终研究报告。代码、提交声明和原始证据仍按各自用途保存。

| 内容 | 要求 |
| --- | --- |
| 研究起点 | 固定 Node、Parent/base_commit、研究模式、主要问题；`origin` 保存证据、初始核心假设、选择理由、父问题回应、Engineering 复用/推迟依据 |
| `iterations[]` | 每项对应一条候选训练轨迹，包含轮次 ID、`design`、`observations[]`、`decisions[]` |
| 本轮设计 | 方法及改动、机制解释与竞争解释、通信与持久状态、机制/质量/成本预测、对照与证伪条件；训练前保存并冻结 |
| 观察与决策 | 追加代码 SHA、run/diagnostic ID、持久证据、预测判断、continue/revise/stop 及理由；保留负结果和不确定性 |
| 最终选择 | 开发结束后固定 `selected_iteration`、选择理由、实际 SHA、参数、动态策略及迁移配置；无合格候选时明确记录 |

同一轮的暂停、诊断和原样续跑追加在原轮；修改方案后追加下一轮，不覆盖旧设计或观察。核心机制/假设或主要问题改变，仍属于新 Node。已定义 adaptive policy 的正常动作变化不算修改方案。

Researcher 通过宿主入口提交档案变更，不能直接修改公开档案。全部开发与诊断运行终态后冻结整份记录，供独立 Analyzer 使用。Analyzer 只提交一份结构化 Analysis；其 Markdown 展示和 `handover.json` 由宿主生成。质量结果、Attempt、日志和 Evaluation 也由系统维护，不重复撰写。

**待实现标记：**`[RESEARCH_RECORD_NOT_IMPLEMENTED]`。注册、追加、提交与设计快照关联、冻结、证据封装需一起支持。旧节点保留不可变 Spec 与既有记录，不直接修改历史协议。

## 3. 方法构造与探索空间

保留一个代码 Parent，允许引用其他分支的机制、证据和构件。选择方法时，按相关性考虑父方法内部限制、其他机制的互补作用，以及不同状态是否适合不同动作；不强制组合或 adaptive，不要求实现多个方向。

设计写清“证据 → 解释及相关竞争解释 → 设计回应与副作用 → 可证伪预测”。说明被改组件原本的有用功能，并回应相关反例。有动机的组合、迁移和参数研究有效，但组合已有构件本身不能证明创新。

固定研究假设可以是一条状态相关策略，需明确可观测输入、决策、状态转移和开销。压缩启动时刻可属于设计；从第 0 步压缩的方法执行自身完整路径。仅当前缀行为和必要状态一致时复用计算。从父方法最终 checkpoint 换算法续训，只支持局部机制判断，不能替代候选从头训练成绩。

这些主要属于后续 prompt 与 ref 的改进；代码侧提供公平评估、非周期策略和真实动作记录能力，不强制算法家族或固定动作。

## 4. 130M 开发与预算

通常不超过两条、最多三条 130M 候选训练轨迹。算法失败、主动提前终止、有意回退重跑均消耗轨迹；pending 预留额度。仅宿主认定的基础设施故障可按同版本恢复规则处理，重试仍有独立上限并计入实际计算。

同一方案从最新有效 checkpoint 精确续跑仍为原轨迹，不重置步数、token、诊断或重试额度。每条轨迹使用固定训练上限与 schedule；诊断采用节点级累计预算。第三条轨迹须有证据支持的修复或影响方法选择的关键未知，训练前记录理由；可以少于两条就结束。

每轮按三层判断：

| 层次 | 检查内容 |
| --- | --- |
| 实现 | 方法是否真正执行，动作是否触发，状态转移是否正确 |
| 机制 | 预测变化是否出现，有哪些竞争解释或副作用，现有证据能否区分 |
| 实际收益 | loss 与完整周期成本是否支持改进，局部收益是否转化为训练收益 |

实现错误可修复；机制预测失败时保留原预测，修正解释或在核心假设内调整方案。机制改善而 loss 无收益，需要检查副作用或停止。质量改善而解释错误，保留候选价值，标明事后认识。机制解释成立不作为晋级硬门槛。

在有意义的 checkpoint 检查即可，不要求每轮重复完整机制实验，也不在机制尚未触发时仅凭早期 loss 淘汰。修正后返回实现、自检和提交阶段，再开始下一条轨迹。

**待实现标记：**`[EVAL_V2_NOT_IMPLEMENTED]`。新权威入口需区分暂停、完成、失败与取消；部分训练不算完成的质量成绩。新轨迹预算与旧“三次成功计分、十次正式执行”口径不同，需版本化实施，不能只改 prompt 或绕过旧双 workload 评估。

## 5. Checkpoint：固定周期保存与可靠恢复

默认采用简单的固定周期方案：每隔 K 个 optimizer update 保存一次；K 仅由宿主权威入口的 `grader.args.research_checkpoint_interval_steps` 配置，默认 **1000**，Researcher 请求不能覆盖，在主动暂停和训练结束时补存；同一步的重复保存可复用。保存发生在完成 optimizer/scheduler 更新、清梯度后的安全边界，不以 microbatch 计数。周期保存不自动暂停训练，Researcher 可在已支持的检查点选择暂停。

初版不必增加复杂的事件触发保存器。固定周期足够支持恢复和一般诊断；动作触发、refresh/reset 等事件另记轻量 telemetry，需要特定状态时可主动保存。周期 checkpoint 本身不保证覆盖罕见事件，也不保证恢复正确。

每份可恢复 checkpoint 至少包含：

- 模型、optimizer、scheduler、compressor/EF 及方法相关持久状态；各 rank 的状态及 bucket 重建所需信息。
- Python/NumPy/CPU/CUDA RNG、optimizer update 计数、token 数与实际 schedule 位置。
- 数据顺序、sampler/迭代器位置及必要的预取缓冲恢复信息；流式数据若暂不支持精确恢复，必须明确标记，不能宣称原训练流续跑。
- 代码 SHA、设计轮次、契约、配置、world size、文件完整性信息，以及正确配对的输入与 labels（若保存诊断 batch）。

修复复盘中的 batch/label 配对问题，并验证连续训练与分段恢复在相同条件下的状态和输出一致。先写临时文件，全部 rank 成功后再发布完整 manifest；未完成快照不可用于正式恢复。恢复失败不能悄悄重置 EF、optimizer 或数据位置。

节点结束前默认保留已发布有效 checkpoint，数量由周期、训练上限和三条轨迹预算共同约束；启动前估算空间。不默默覆盖已引用快照。若另启用轮转保留，需显式固定策略并保留诊断依赖；初版可以只做容量检查和明确报错，节点结束后统一清理。

## 6. Researcher 托管诊断

扩展现有受控诊断入口给当前节点的 Researcher，复用资源隔离、超时、持久预算与脚本/结果归档。Researcher 在本上下文内完成判断，不增加每轮 Analyzer 反馈。诊断无明确用途时复用已有证据。

可做的简单实验包括：特定昂贵/理想干预是否有任务 loss 收益、两个机制是否互补、动作效果是否随状态反转。干预条件、额外计算和副作用需明确；它不是未经证明的性能上界，也不保证存在廉价实现。

恢复该结论需要的状态。更新或 continuation 对照包括 optimizer/compressor 历史，匹配数据和 label；各分支独立演化，用分离 probe 比较任务 loss。探索选中的干预在预算内用新数据复核，做不到则标为探索性。静态代数检查无需恢复用不到的训练状态。

诊断分支不得污染主轨迹的模型、RNG 或数据位置，不提供免费训练步。Synthetic 检查仅支持已测性质，短 continuation 仅支持相应状态和时域。对于 adaptive 方法确认真实触发；必要时比较固定动作及相同动作比例但时机随机/错位的对照，计入判断和切换成本。

## 7. 评估的 fidelity、efficiency 与候选保留（先不做）

正式开发、baseline 和对照使用相同的固定预算与数据协议。复盘运行的 130M 为 2500 步、warmup 2000 步，后段证据有限；是否延长训练或调整 schedule 需校准后确定，改变时建立新契约并补匹配基线，不重标历史结果。

- 持久化本地 loss、学习率、训练时间与实际动作记录，不依赖外部 W&B 才能分析。
- 开发验证可校准后使用较小子集，减少无决策价值的初始验证；先确认它能保持代表性候选的差值方向、分辨有用收益，暂不固定 token 数。
- 分开报告算法训练成本与搜索成本。checkpoint、诊断、失败、恢复及重试开销进入搜索成本；算法成本的测量边界由契约统一规定。正式 timing 不加入梯度 profiling。
- 效率判断覆盖目标 workload 与完整有效周期；对非周期方法保存真实动作和累计成本，短窗漏掉罕见事件时限制结论。保留现有 null period 支持并检查其覆盖范围。
- 固定 seed 和数据顺序支持配对比较，不提供独立训练重复。验证 token 数、训练长度和独立 seed 分别解决不同不确定性。

候选选择同时体现质量与成本，保留负结果及有价值的取舍。有限 Pareto 候选保留、后续继承规则和按需 Engineering 曾讨论过，但选择规则尚未定型，可在基础闭环之后实施。第一版仍只选一个候选做节点末 60M。

## 8. 每节点 60M 与后续 Engineering

主动结束开发或耗尽额度后，冻结一个已完成 130M、通过预先固定的有效性/质量门槛和资源约束的候选。允许选择先前完成的轮次，不限最后一轮。无合格候选则跳过 60M，保留失败供 Analysis。

宿主运行该候选的一次 60M AdamW 迁移检验，预算独立于三条 130M 轨迹；只允许既定的同版本基础设施恢复，不换候选、不因结果不理想追加参数试验、不重开当前节点。迁移所需的尺寸或参数映射在结果揭晓前固定。

结果可供 Analyzer、后继节点与后续 wisdom 使用，因此是冻结后的跨 workload 迁移检验，不是整轮搜索的独立 held-out。130M Muon 到 60M AdamW 同时改变模型规模和优化器，不能分别归因，也不替代同任务新 seed 或完整 schedule 确认。

完整 Engineering 集中于值得优化的候选，选择/跳过规则另行收敛。Engineering 产物如用于最终性能主张，仍需相应质量复核，不能永久只继承旧代码成绩。少数最终候选的新 seed、较长训练或额外保留条件，可另设预算；目前未确定，不作为第一版每节点的额外必跑阶段。

## 9. 动态产物目录与节点清理

新增可配置的动态产物根目录，默认 `/dev/shm/wyr_tmp`。按稳定 run ID、Node、轮次/任务和执行 ID 组织，避免不同 run 或重试互相覆盖：

```text
/dev/shm/wyr_tmp/coral/<run-id>/nodes/N001/
├── code/<candidate-sha>/...
├── runs/<request-id>/
│   ├── checkpoints/llama130m_muon/step-00001000/...
│   ├── quality/...
│   └── timing/...
├── runs/<finish-request-id>-transfer/...
├── diagnostics/<analysis-or-engineer-task-id>/...
├── analysis/<session-id>/...
└── caches/...
```

- checkpoint tensor、诊断大输出/中间 tensor、临时执行副本及本节点其他大体积动态文件进入该目录，避免在 run、Agent worktree、`/tmp` 或根分区重复留副本。checkpoint 不进入 Git。
- 小型研究档案、Analysis、指标、脚本、必要日志、manifest、校验信息与清理记录保存在 run 的持久证据目录。节点清理前固化 Analysis 实际依赖的小型结果；大 tensor 清理后保留其身份与已清理状态。
- 宿主记录存储根与归属，证据读取、checkpoint 发现、sandbox 只读挂载、完整性校验、归档和清理共用受控路径映射。当前路径校验不能直接接受任意外部路径，不能仅靠新增符号链接绕过。
- 节点的 Research、Analysis 和 Engineering 已结束或按规则跳过，且没有在跑/等待使用这些文件的进程时，由宿主清理该 Node 的动态目录。Researcher 结束不等于 Node 结束，不能提前删掉 Analyzer 所需状态。
- 正常完成、取消或终态失败均需在所属任务退出、必要证据落盘后清理；暂停/可恢复状态保留产物。清理只处理宿主确认归属的目录，可重试，不扫其他 run。跨节点保留的基线缓存若确有需要，应单独声明 run 级归属、预算和结束清理时机。

`/dev/shm` 是内存文件系统，需要计入主机 RAM 与实际 tmpfs/容器限制。启用前检查可写性、可用空间和安全余量，预算覆盖所有并发节点及诊断副本；不足时明确拒绝写入或暂停，不静默回退根分区。若需要主机重启后的恢复，可将同一配置指向有容量的持久数据盘；默认 tmpfs 不承诺跨重启保存。

本次会话检查：根分区使用率约 97%，`/dev/shm` 显示约 252 GiB 可用，但当前会话挂载视图为只读。实际运行宿主必须重新检查；本文没有创建或写入该目录，也不据此推定宿主已经具备所需权限。

## 10. Research 阶段持续预留 4 卡

可以实现。将当前每个 quality/timing/diagnostic 阶段单独申请、释放的 reservation，扩展为宿主持有的 Research 阶段 reservation：注册完成后、进入 GPU 开发时一次性申请 4 张卡，获卡后持续保留至 130M 开发和条件性的 60M 检验均结束或明确跳过。不要持有全局选题锁等待 GPU。

期间 Researcher 的训练、诊断、修改代码和思考均不释放这组卡；正式子进程由宿主串行复用固定 UUID，不重新进入公共队列申请同一组卡。持有 reservation 不要求训练进程始终存活，也不需要占位进程消耗显存。暂停后可结束训练进程，诊断完成后从 checkpoint 重启并续跑原轨迹。

最小支持包括：

- reservation 属于宿主的 Node/Research assignment，保留可信身份、UUID 与存活状态；由宿主把卡交给已授权的训练/诊断子任务。Agent 不能自行选择卡、伪造归属或绕过托管入口。
- 子任务复用已有 reservation，不再次竞争自己的锁或运行名额；同一 reservation 默认串行执行。保留 run 级执行名额限制，持续持卡的节点占用一个名额。
- 正式 timing 前清理本节点其他计算进程，再按原规则检查空闲条件、硬件可比性和外部干扰。持续预留不代表某个训练进程可与 timing 同时占卡。
- 开发及 60M 确认结束后释放，不继续占到 Analyzer/Engineer 完成。它们按自身授权申请资源；GPU 释放与动态产物清理分别按各自阶段处理。
- 正常结束、取消、宿主或 assignment 失效时，停止自己的子任务并释放 reservation。中断恢复保留研究预算，但重新核实/申请实际 GPU，不把旧状态文件当作仍持有锁。

此模式减少反复排队并保持卡身份稳定，代价是思考和代码调整期间也占着 4 卡。高优先级 Analyzer/Engineer 无法抢占已持有的 reservation，现有优先级只能影响新申请；并发节点数需与 GPU 组数和执行名额匹配。第一版按用户要求保留整段 reservation，不做短时闲置回收。

现有文件锁主要协调同宿主、同 UID 且遵循公共 allocator 的 CORAL run，不能阻止任意外部进程或其他用户启动 GPU 任务。若要求机器级硬独占，还需作业调度器或主机资源管理配合；继续保留实际干扰监测。

## 11. 独立 Analyzer 的输入与输出

输入为冻结研究档案、选定代码、全部尝试的正式结果与提供的诊断证据，并仅使用实际提供的历史/父节点资料。复核每轮预测、修改依据与最终收益，也保留发现意外机制的空间，不局限于确认 Researcher 的故事。

输出仍为一份符合现有 schema 的 Analysis：机制与 practical verdict 分开；findings 包含来源、对照、条件、竞争解释、反证和限制；new_information 只记增量；最多两个重要后续问题，附最小区分实验。handover 是同一文档中的交接内容，不另写一份报告。

没有关键未知时跳过新增诊断，仍完成判断与输出。表示/几何、受控机制、实际 optimizer update、闭环质量的证据层次适用于全部分析，不随诊断章节跳过。可报告有益干预、互补关系、条件效应与被削弱的解释，不直接给新算法或调参建议。已有成本证据可用于判断，不新增 runtime profiling。

## 12. Ref、summary 与全局 wisdom

全局 wisdom 独立为 `notes/wisdom.md`，不混入 summary，不增加复杂知识系统。

| 内容 | 职责 |
| --- | --- |
| `notes/wisdom.md` | 少量可复用认识、适用条件、重要反例与直接证据 |
| `notes/research_summary.md` | 研究进展、代表结果、候选与证据导航 |
| `notes/engineering_summary.md` | 实现结果、性能比较与证据导航 |
| 现有 `references/` | 文献方法、方程、假设和可复用构件 |
| 研究档案、Analysis 与原始记录 | 初始设计、完整迭代、独立论证和失败证据 |

Wisdom 的单位是会影响设计、解释或实验选择的认识；同一认识的新证据合并，不能因重复就自动提升为普遍规律。不要求覆盖每个节点，没有新增认识就不更新。分数、状态和完整日志不放入 wisdom，强制流程保留在 workflow rules。

Memory 读取当前 wisdom、新节点 Analysis，必要时回查直接证据，统一发布更新；Researcher/Analyzer 不直接修改共享 wisdom。宿主补充发布、输入 staging 与读取入口。每节点 60M 迁移结果已属于可供后续使用的已知证据，记录其来源与范围。

Ref 保留家族文件，增加一个简短的问题入口，链接表示容量、历史积累、预算分配、阶段变化等问题对应的构件；同一内容可被多个入口引用，不复制正文。跨方向组合不要求多 Parent 状态管理。

## 13. 实施顺序、验收与待定参数

优先补代码机制，正式 prompt 留待能力可用后统一启用：

1. **分段训练基础：**checkpoint 配对与完整状态、固定周期保存、精确恢复、本地曲线，以及动态产物根与容量检查。验证连续训练与暂停恢复一致，保存不完整时拒绝恢复。
2. **开发执行与资源：**130M 开发入口、Researcher 托管诊断、节点累计预算、持续 4 卡 reservation；验证子任务复用资源不死锁，暂停与重启不刷新预算，取消后回收自己的进程与锁。
3. **研究档案与收尾：**注册/追加/冻结、设计与实际提交关联、选定候选 60M、独立 Analysis 的完整输入；验证冻结后不可调参，失败和跳过均可形成完整分析。
4. **产物生命周期：**清理触发与证据固化、恢复与过期目录处理；验证 active 节点不误删，终态清理可重试且不影响其他 run，保留的证据仍可核查。
5. **评估与知识整理：**校准开发验证开销及 schedule，接入独立 wisdom 与 ref 问题入口；后续再细化 Pareto 保留、按需 Engineering 和少数最终候选的额外确认。

首版保持现有训练契约：130M 2500 步、60M 4000 步，schedule 不变；checkpoint 默认 1000 次更新，Researcher 诊断累计最多 5 次完成/15 次请求，同版本基础设施重试默认最多 2 次。仍需校准训练 schedule、验证划分/token 数及空间预算；GPU 持有阶段的整体超时与并发、候选保留/按需 Engineering、新 seed 和长程确认预算后续确定。

新入口与存储机制未启用前继续执行原契约。历史 Spec、Attempt、评估与 Analysis 保留原版本；不自动迁移运行中的旧实验，不重标历史结果。本次实现代码机制，正式 prompt 和现有 run 配置不修改；新协议默认关闭。

## 14. 代码改动边界与模拟跑验收

实施时先保留正式 prompt；新协议以显式配置启用，旧 run 继续原分支。模拟 Agent 使用脚本调用新入口，使代码机制可以在修改 prompt 前独立验证。第一批交付覆盖开发闭环、产物生命周期和资源归属；不同时引入复杂候选选择规则。

| 交付 | 主要代码位置（任务目录内） | 必须跑通的结果 |
| --- | --- | --- |
| 产物根、周期 checkpoint 与恢复 | `seed/diagnostic_checkpoint.py`、`seed/c4/run_llama_pretraining.py`、checkpoint 发现与 retention；共享受控路径解析 | 大产物进入配置目录，真实小模型保存/恢复一致，暂停不被判为完成质量评估 |
| 研究档案 | `autoresearch` 的 nodes、submission、artifacts、状态与 CLI | 注册起点、冻结本轮设计、追加观察/决策、关联实际 SHA、最终冻结；历史内容不能覆盖 |
| 分阶段评估与预算 | grader 的 contract、quality、grader；autoresearch 状态与提交入口 | 130M 暂停/续跑/重跑，轨迹累计最多三条，开发结束后单个候选有条件地运行 60M |
| 持续 4 卡与 Researcher 诊断 | grader 的 GPU allocator、宿主 assignment 生命周期、diagnostics 与 sandbox | 宿主持卡，子任务串行复用且不重复申请，正式 timing 保持原隔离检查，退出时释放 |
| 分析交接与节点清理 | Analysis 输入/证据封装、Evaluation、节点完成及清理逻辑 | 冻结档案和全部有效证据交接；60M 结束/跳过后只发一次独立 Analysis；完整节点结束后安全清理 |
| 后续知识整理 | Memory 发布与输入 staging、ref 文档 | wisdom 独立于 summary；新增证据可追溯。放在核心闭环之后验收 |

测试复用 `tests/test_workflow_simulation.py` 与 `tests/simulation_support.py` 的真实流程框架，扩展已有 checkpoint、GPU allocator、diagnostics、节点清理与发布测试。保持真实 CLI、可信服务、状态转换、预算、发布与进程生命周期；替换模型调用、耗时训练和 GPU inventory 等外部边界。不能通过模拟代码直接填好新状态或假造清理成功来绕过待测实现。

**完整模拟跑是合入必要条件。** 至少覆盖以下场景：

| 场景 | 关键断言 |
| --- | --- |
| 正常闭环 | 注册 → 首轮 130M → 中途保存/诊断 → 原样续跑 → 冻结 → 60M → 一次 Analysis → Engineering 终态 → 清理；阶段顺序和证据一致 |
| 调整与额度 | 第二轮按新设计、SHA 和固定起始协议重跑；第三轮消耗额度，第四轮拒绝；失败和提前终止计数，恢复/归档不刷新额度 |
| 无合格候选 | 跳过 60M，仍产出包含失败的 Analysis，节点能进入合法终态 |
| 迁移失败 | 60M 算法失败不切换候选、不重开开发；发布实际失败结果及 Analysis，后续资格按新质量契约确定 |
| 中断与重试 | checkpoint 或结果写入中断、宿主重启、同 SHA 基础设施恢复；完整快照可恢复，部分写入不能被使用，预算和已完成阶段保留 |
| 幂等与权限 | 重复请求不重复训练/扣额/运行 60M/发 Analysis；不能修改冻结设计、伪造 SHA、操作其他节点或绕过托管诊断 |
| 持卡与并发 | 思考和改代码的模拟间隔内仍持有同组 UUID；训练/诊断复用不死锁；另一节点或 run 不能拿到相同卡；取消/退出只回收本任务进程和 reservation |
| 空间与清理 | 所有大动态文件位于配置根，无根分区回退；只读/空间不足明确失败；活跃节点和其他 run 不被误删；终态清理可恢复且保留可核查的小型证据 |
| Legacy 回归 | 原 Spec、双 workload、预算、Analysis 发布和恢复行为在旧协议下保持不变 |

模拟存储根使用测试专用可写临时目录，不要求当前 `/dev/shm` 可写，也不占真实 4 卡。GPU 竞争测试使用真实跨进程文件锁和可控 inventory；不能仅以设置环境变量代表已获得卡。测试结束断言没有模拟子进程和本测试持有的 reservation 遗留。

Checkpoint 正确性单独用真实 CPU PyTorch 小模型执行：使用变化的数据序列和随机状态，比较连续训练与多次保存/恢复后的模型、optimizer、scheduler、compressor/EF、RNG、数据位置及下一 batch/labels；扩展现有两 rank Gloo、AdamW/Muon 用例，不能只比较相同固定 batch 上的一步更新。不完整 rank 保存和 manifest 校验失败也必须覆盖。

验收产物包含 `report.json`、事件时间线和失败日志，明确标记 simulated，列出执行的测试命令、通过/失败/跳过及覆盖范围。原有正常/故障完整流程与新增场景均需实际通过；关键测试因缺依赖被跳过时，不算完成对应验收。模拟与 CPU 测试验证流程和所测恢复语义，不证明真实 CUDA/NCCL 等价或算法性能；需要这些结论时，另做短程真实 GPU 冒烟并单独报告。


## 15. 首批实现的配置与入口

代码机制以 `grader.args.staged_research: true` 显式启用；默认 `false`，不迁移运行中的旧 run。宿主配置如下：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `research_checkpoint_interval_steps` | `1000` | optimizer update 保存周期；正整数，仅宿主可调整 |
| `dynamic_artifact_root` | `/dev/shm/wyr_tmp` | 本 run/node 的 tensor、训练/诊断副本和缓存根 |
| `dynamic_artifact_min_free_bytes` | `1073741824` | 开始 GPU 工作前要求的剩余空间；不足明确报错 |
| `research_infra_retry_limit` | `2` | 同一轨迹/冻结迁移候选的基础设施重试上限 |
每个动作使用 `autoresearch research <action> --node N001 --file <data.json> --request-id <唯一ID>`。`status`、`recover` 无需文件。执行请求立即返回排队状态，使用 `status` 读取结果；通道超时重用同一 ID，不能为同一操作换 ID 后盲目重试。宿主重启后的 `recover` 负责收尾中断请求与重放已完成结果的发布。

| action | JSON data |
| --- | --- |
| `register` | `registration`：原初始化字段；`design`：`method/mechanism/predictions/tests` |
| `design` | 下一轮 `method/mechanism/predictions/tests`；追加后不能覆盖上一轮 |
| `train` | `iteration`、完整 `sha`、可选绝对更新端点 `pause_after_steps` |
| `diagnostic` | `iteration`、`argv`、`timeout`、`scripts_dir`；串行复用已持有的 4 卡 |
| `decision` | `iteration`、`action`（`continue/revise/stop`）、`reason` |
| `finish` | `selected_iteration`（可为 null）、`reason`；从未训练时另提供已提交的 `sha` 作为未执行代码证据 |
| `status/recover` | 空对象 |
注册仍遵守原 Parent、选题租约、Spec/提交自检与 timing declaration 约束。`research.json` 是可演进设计与证据的权威投影，旧 Spec 保留为初始来源兼容快照；不新增 Researcher 终稿。第三条轨迹是硬上限，不会自动替 Researcher 选定候选。主动结束前须显式终止尚处于计划/暂停/基础设施失败状态的轨迹。没有合格候选时跳过 60M；即使从未启动训练，也必须有明确标记未执行的已提交代码供独立分析，不伪造质量成绩。
130M 暂停结果不计为正式成绩，恢复沿用原 SHA、固定数据流和训练上限；完成后沿用原 timing 协议检查资格。冻结后，单个候选的 60M 不接受改方法或切换候选。迁移算法失败和基础设施重试耗尽都终止迁移并保留失败分析；发布失败可以恢复，不重跑已完成的迁移。
4 卡由独立宿主 keeper 持有，借用任务不再次排队。Agent/宿主身份变化、显式取消或迁移终态会释放自己的进程和 reservation；它是同 UID 的 CORAL 调度锁，不阻止其他用户直接启动 CUDA。长开发节点应由操作者配置足够的 Agent 总超时，避免原 `agents.timeout` 提前终止 Researcher。独立 Analyzer 和 Engineer 后续单独申请资源。
完整节点终态才清理动态目录。保留小型 JSON/曲线/脚本/日志、checkpoint manifest、原始证据索引和最终 Analysis；大文件、超额文本明确标记未保留。Analyzer/Engineer 的临时诊断目录也进入本节点动态根，引用使用持久化的小型证据。数据流恢复保持契约规定的 worker 数（当前为 4），使用确定性、按序交付的 DataLoader 回放；恢复时校验 worker、预取等设置，并在回放后恢复训练 RNG。需要固定 C4 内容、顺序与确定性预处理，不承诺任意随机数据变换的恢复；tmpfs 不能提供重启后的 tensor 持久性。
模拟验证保留真实 CLI、可信 PID 身份、状态、SHA 冻结、Analysis 哈希、Engineering 收尾和节点清理，替换耗时训练/GPU 边界。另用真实跨进程 flock 验证持卡，以及真实 PyTorch CPU/Gloo 验证恢复。当前不声称验证 CUDA/NCCL 等价或算法收益；缺少 bubblewrap 时诊断明确返回 unavailable，单独的 CPU 执行器测试验证借卡而不重新申请资源。


## 16. 验证记录（2026-10-26）
- 集成回归：任务测试加框架 trusted-write/dynamic-agent 测试共 724 项通过、2 项跳过；跳过项依赖当前环境缺失的 bubblewrap。包含取消清理、证据保留和多 worker 恢复；另补注册前干净工作区及固定 Parent checkout 检查，定向测试通过。
- 新入口完整模拟：4 个场景通过，覆盖正常 Engineering 收尾、60M 质量失败、60M 执行失败、无合格候选与三轨迹上限；保持真实 CLI、宿主身份绑定、发布、证据哈希及清理逻辑。
- Checkpoint 用真实 CPU/Gloo 小模型验证，GPU reservation 用真实跨进程文件锁验证。另补原始连续 DataLoader 与恢复数据流的 2/4 worker 对照，避免改变 dense 比较的数据协议。
- Astra medium review 发现的中断恢复、发布重放、档案引用、路径越界和失败证据问题均已修复并复查。
- 模拟报告、时间线和日志：`/tmp/coral-staged-implementation-20261026/simulation-final/`。训练/GPU 边界是模拟，不能据此声称 CUDA/NCCL 等价或算法收益；诊断借卡的真实 CPU 进程测试通过，实际 bubblewrap namespace 隔离仍需在具备依赖的宿主验证。

代码提交：`aeb0d21`（checkpoint/恢复）、`23616cd`（档案/产物归属）、`cee6ec8`（持续持卡）、`eed52eb`（分阶段工作流/交接清理）、`b0ececd`（保持原 worker 数据协议）。正式 prompt 未修改。后续 review 按用户要求使用 `gpt-5.6-sol`、`high`。
