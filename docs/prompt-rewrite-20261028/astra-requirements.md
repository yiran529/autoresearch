# Astra 独立语义盘点（第一阶段）

日期：2026-10-28。此文件是原文要求基线，不是改写稿的通过证明。审查人按父任务指定，以 Astra medium 独立阅读；不以新旧、长短判断优劣。

## 覆盖、来源与限制

已读仓库根 `AGENTS.md`；完整阅读 `tasks/greedylore_autoresearch/roles/{researcher,engineer,workflow_rules}.md`、`prompts/{research_analysis,global_memory_manager}.md`、`references/{research_commands,common,initial_diagnostics,evaluation_tasks,semantic_criteria}.md`。阅读 `engineer_runtime.md` 的执行、计时和诊断用途/解释边界，未逐项审查全部优化配方。核实 `coral/agent/dynamic.py`、`coral/template/coral_md.py` 的动态入口，以及 `autoresearch/src/autoresearch/analysis.py::_prompt`、`memory.py::_consolidate_node`、`autoresearch/schemas/{analysis,memory}.schema.json`。未运行训练、评估或完整 runtime 测试；未核实每项 host enforcement。

第二阶段需对各改写文件逐项比对，并标明保留、移至可达引用、弱化、遗漏、冲突或有意调整。原文中的机械顺序可以改，但不能因它是旧规则而默认删除；实验工具包不是必须执行的研究路线。

## 上下文可达性与优先级

- Researcher/Engineer 启动消息要求读角色和生成的指令；生成指令再引向 toolkit/workflow。两者可以依靠明确的共享引用承载重复规则，但使用者仍须知道何时读、读什么以及必须用的入口。
- Analyzer 的 `_prompt` 拼接独立角色、候选清单、固定 evidence JSON；Memory 的 `_consolidate_node` 拼接独立角色与 required/covered refs。两者不是 Researcher 对话延续，也没有自动拼接 shared workflow；关键边界不得仅移至 shared workflow。
- 动态启动代码仍有非 staged 的旧流程文本，staged 分支由 `staged_research_scope` 覆盖；不要把旧 3 成功/10 总 Research formal runs 或 `node initialize/complete` 混入 staged 协议。
- recorded evaluation contract 权威。原 shared workflow 有 v3 表述，evaluation_tasks 已说明 v4、350M timing 与分阶段协议；这属于已有版本差异，改写不能把旧版事实错误固化，也不能重标历史结果。

## Researcher：研究方向与职责

- 对一个分配 Node 负责算法设计、实现、实验、诊断和 refinement；开发期间自己判断，不等待独立 Analyzer。
- 背景不是“找缺陷”：可从观察、资源约束下的目标性质或机制假设起步。建立可证伪问题、证据→机制→方法→预测→质量/成本的联系。
- Parent 依据研究价值、连续性、信息增量与验证成本；先评估候选 Parent handover 对源证据的支持。当前低秩和稀疏优先。其他分支可提供证据/组件。
- host mode 固定，缺省 REFINE，不重采样或自行提交 mode。REFINE 深入或挑战 Parent 方法/机制；EXPLORE prefer a different technical route from Parent **and existing branches**，例低秩、稀疏、量化、local/decentralized，研究动机优先于新奇。
- 直接回答 Parent 问题时 `problem_ref` 必须真实；否则 null 且 `parent_response` 解释联系。并非必须从 handover 取题。
- 相比 Parent、注册中方法和最近 prior work 交代贡献，避免重复；组合要考虑互补，迁移/参数研究要有动机，dynamic-adaptive 明确可观测输入、动作、状态转移与成本。不得将此弱化为无方向的超参搜索。

## Researcher：注册、设计与开发条件

- 先读 Tree/Node README、wisdom/summaries 的源 Analysis、状态/Attempts/Worktree；恢复现有阶段、变更、提交和消耗，不重选 Node。
- 设计前读 construction 基本入口，family refs 按需；相应操作前读 command guide。host 未启用 staged 时报告，不自行更改配置、协议。
- Parent/方法选择前 planning acquire；锁内检查当前 registry、进行中设计、相关 Analysis/Evaluation/固定 SHA 代码；干净 Worktree、Parent Research SHA 注册成功后才实现。成功释放，失败保留；续租、放弃释放、失锁重新获取并重读。
- `origin` 固定 Parent/base_commit、主问题与初始核心假设；包括 selection rationale、delta、risk、falsification、parent response、Engineering review。先读 available Engineering Summary/parent Analysis，记录复用、推迟、confound 或不可用理由。
- 每份 design 必须有四个非空字段：method（通信对象、同步聚合、optimizer/state/对应组件）、mechanism（证据、解释、对手解释、旧组件有效作用与副作用）、predictions（机制/质量/完整周期成本分别可证伪）、tests（匹配对照、可观测判别、支持/削弱结果）。保存后训练，append-only；核心机制/假设改变要新 Node。
- 区分算法、参数、Engineering 变化；Researcher correctness 不要求 Parent 等价。实际 semantic 检查失败修复，记录命令/输入/结果/未覆盖。
- 提交 `submission.yaml` research 模式、Spec refs、完整 `timing.period_steps`（非周期 null）、维护事件 telemetry；`method_policy.py::configure_compression(args, workload)` 与 shared hook 声明各 workload 策略并支持矩阵/卷积梯度。不能改训练字段。
- 在 managed experiments 前归档后续 Analysis 所需 self-check；附件仅证明内容哈希，不认证执行。
- train 使用 iteration/full SHA；首个 timing 或 train 将设计 pin 到 SHA。检查设计代码一致，在机制有机会发生后再否定；候选成本与整个搜索成本分开。
- 诊断针对会改变决策的不确定性，优先现有真实日志/状态。仅 paused/completed 轨迹可诊断，不能污染候选或追加免费训练；恢复相关状态、独立演化分支、单独 probe data，可行时 fresh-data 确认探索发现。
- optional development timing 可训练前/暂停/完成后；默认复用同 SHA/protocol 的最新完成结果，主动重复需 repeat_reason，不选最快；检查 hardware、variability、完整 cycle/event、动态决策/切换成本。
- decision.reason 对照原机制预测，标记支持/反驳/未测试，引用 IDs/持久证据并解释下一动作；保留失败及解释失败时仍实测的收益，改解释标 post hoc。
- continue 恢复模型、optimizer、compressor/EF、schedule、data position、RNG 且同 iteration/SHA；revise 先结束非 terminal 轨迹，再 append design、实现检查提交，从头训练并匹配初始化/数据顺序；声明的 adaptive 动作不算 revision。decision 本身不执行下一动作。
- 区分实现失败、机制不成立和机制成立无实际收益；区分算法固有成本与实现开销。proxy、improved、U、预期未来优化均不能证明实际成功。

## Researcher：预算、freeze 与 Final

- 通常最多 2 轨迹、硬上限 3；第三条需执行前有证据的修正或 selection-relevant uncertainty。总 3 designs 包括初始和 timing-only；少于 2 合法。失败/停止/rewind/restart 计数、pending 占位、exact continuation 不新增，恢复不重置预算或固定 steps/tokens/schedule。
- development timing 5 completed/15 total，diagnostics 10 successful/20 total、600 秒；失败、超时、中断计 total，host 分类 infrastructure recovery 受自身 cap 约束。不能把计时/诊断预算并入训练或 Final。
- finish 前全部开发 train/timing/diagnostic 终止；planned/paused/infra_failed 显式 decision stop。提名一个 completed 且开发 validity/quality/resource passing 的迭代，可选较早迭代；机制确认和 development timing 均不是新增 gate。无合格候选 null；从未执行候选则提供 committed representative SHA。
- 先 freeze selected SHA/design/policy/workload mapping，再 host fresh mandatory Final timing + 固定 language transfer + ResNet18/CIFAR100。Final timing 独立预算、不复用开发数据，保留 validity/cycle/hardware 检查、没有新增 speedup magnitude 门槛。
- freeze 后不能换候选、反馈调参或 reopen；仅 host-classified infra recovery，默认每 component 2 retries；保留已完成 component，同 component reuse 只 recovery；vision 从头重跑。
- 所有 Final terminal/explicit skipped 后退休 Researcher，独立 Analysis 与 quality-gated Engineering 启动。无资格候选明确 skip + negative evidence。Researcher 不另写 final report、不认领另一 Node。
- Final 是开发选择后 transfer/robustness 证据；被后续 Node 复用后不再是独立 campaign held-out。

## Engineer：职责、测量与完成

- 仅 host-bound Node，提高端到端效率且保持 Research 算法语义；不创建/换 Node、不取消 required Engineering、不替换 Research 结果、不等 Research Analyzer。
- 首次干净 Worktree `engineering claim Nxxx`，以 claim 当前 run_limits 为准；恢复保留进度。developed Node 锚定 frozen selected_iteration 与匹配 Research SHA，不用 initial Spec/last iteration 替代；Root 是 baseline Spec/A0、不是 S0，保持 seed、不重 bootstrap。缺失/矛盾 handover 是错误，不能猜替代。
- 主优化目标来自正式证据/代码/历史优化/可用 profile，区分实现开销、算法成本与归因未决；selection_rationale 1–3 句连接成本、端到端影响、改动和预期。
- 诊断按需、小而代表性，覆盖 ordinary/refresh；先 commit candidate，managed `diagnostic run` 独立 snapshot，持久 results，formal candidate 无 instrumentation。成功失败 ID/SHA/结果均保留，局部/合成范围明确，操作数减少不证明 critical-path 缩短。
- semantic/differential checks 对 Research，包含通信、optimizer、持久状态；adaptive 输入/阈值/触发/动作/状态/RNG 分布与 trigger/non-trigger/cross-step，不可冻结策略或换阈值换速度。记录容差、未覆盖；有限测试不是所有质量不变证明。
- commit engineering submission，base_commit/quality_anchor 始终固定 Research SHA，即便工程候选相互继承；仅 `engineering timing --sha <full-HEAD-SHA> --request-id <id>` 正式测量，命令不 stage/commit；quality inherited (provisional)，不是重训。
- 按正式结果+语义+profile 判断保留/撤回，含 dtype、变异性、新增成本；最高 score 不等于有支持的收益。预算内直到无明显 substantial semantics-preserving 端到端机会，一次 improved 不够。
- 4 successful/10 total，valid scored regression 算 successful，失败/retry 算 total、pending 占位、跨恢复保留，任一 cap 停。仅 host-classified infra 同 SHA --retry 新 ID；lost reply 原 ID/SHA/flags。
- 诊断独立 10/20、1–4 GPUs、600 秒/执行、3600 秒 queue；不替代 formal run。finish 前全部 diagnostic 结束并归档需要的 self-check。
- 至少一 terminal formal Attempt 且停止条件/cap 达成后，本人写 final Analysis，覆盖全 claim 成功失败回退、SHAs、实测和真实检查、停止理由与剩余空间。profiler durations 不可相加当 step-time；当前开销不等于算法下界。
- scratch（tracked Worktree 外）JSON 恰含 summary、findings、limitations、attempt_shas；前者非空，两个 arrays 非空，每个 distinct formal candidate SHA 恰一次，包括失败/回退。
- `engineering finish --analysis-file /absolute/scratch/final.json`；finalizing 是 durable submission 非 published completion。中断相同内容重送，提交后无新候选/改 summary，等待 host versioned publish/retirement。无独立 Engineering Analyzer。

## Independent Analyzer：可研究的内容与不可跨越的边界

- fresh session、冻结后全记录独立解释，不插入 development、不 reopen。研究不是仅审计：追踪机制、意外收益、交互、条件效应、反例及会改变下次实验/设计的理解。
- 输入是 supplied research.json、初始 Spec、Evaluation、全部 Attempts/Final components、固定 SHA、raw results、managed diagnostics。Parent/history 仅 supplied；Root S0/A0 区分；Researcher 解释待核实。
- 比较 contract/budgets/schedule/seed/data/model/optimizer/comparator/gates；区分算法/参数/实现改动、pause/continuation/diagnostic/restart、相关状态与 contamination。每个 Final component 明确 pass/fail/skip/missing；中间 checkpoint 非最终质量，重试/probes 非独立 seeds；同时改变 task/model/optimizer 无法分别归因。
- 有清楚关键问题直接研究；否则 initial_diagnostics 少量适用 first-pass；不制造 gaps、不跑全菜单。优先会改变判断的 cheap discriminator，执行前 purpose/conditions/observables/implications；可用日志、数学、受控实验。
- 恢复 claim-relevant inputs/labels、optimizer/compressor history、bucket layout；声明缺失/故意干预；独立分支；真实 phases/interactions/trigger、fresh probe。decision timing 可能需 fixed-action/matched-mix control。
- 区分 geometry、controlled effect、optimizer-update、closed-loop quality；构造 replay 不等于真实 stream continuation，局部误差小不等于长期收益；理想干预既非认证 upper bound 也非便宜实现证明；数学条件、合成边界、rare-event/single-checkpoint 缺口要限定范围。
- 有效结论可为 no supported gap/inconclusive/replication；不以 bookkeeping/generic audit 填 findings、不强填两个问题。保留实用正负收益即便机制未知，保留解释失败时真实收益。
- summary 回答问题及迭代原预测/实际质量成本；findings 给 claim/comparator/直接证据/scope/意义；limitations 说明具体限制哪项结论；handover new_information 表明新增/排除/收窄，纯重复可空；问题给重要性、对手解释和最小区分实验。
- **独立保留**：不得新算法、推荐 mutations、代码修复/超参选择；controlled intervention 解释用，不是可部署建议。不得新增 wall-clock/latency/critical-path profiling/runtime attribution（Engineer 负责），可使用已验证成本。
- **独立保留**：仅 supplied snapshots/approved roots；不改 Research code、正式行为、eval data、历史报告、wisdom/private tests；sandbox scripts，不 instrumentation official eval。host 支持的 `autoresearch diagnostic run` 唯一入口，10 successful/20 total、600 秒，无 off-path GPU/额外大规模训练；缺能力写 limitation。
- 失败先读 record/stdout/stderr；新执行新 ID，原 ID 只取原结果；长等，不短轮询/周期读日志；持久成功失败均保留。subagent bounded substantive task 且同权限。
- 引用只 supplied inventory + 本任务 persistent diagnostics；固定 SHA/真实 ID，允许 staged/local aliases，仅 frozen files；不随意 workspace/tmp；attachments hash 不认证执行。
- **schema/语言不改**：英文、一个 JSON，无 extra fields/独立报告；summary/string arrays findings/limitations 非空；tags `[OBSERVED] [SUPPORTED] [REFUTED] [REPLICATION] [INCONCLUSIVE] [NO_SUPPORTED_GAP]` 不加字段。handover 的两个 verdict ∈ solved/partially_solved/not_solved/inconclusive；无 evidence refs 时都 inconclusive；new_information/follow_up_questions 可空；后者最多 2、unique Q1/Q2、id/question/importance/discriminator。

## Global Memory：派生知识与来源

- one-shot read-only analyst，来源是 published Analysis；产物为 derived views，不是新实验或独立证明。只有 Node Research+Analysis complete 且 Engineering complete/explicit skip 才 consolidate。
- 读当前 wisdom/两个 summary、required_refs；legacy bootstrap 可能要求历史一次。其余 supplied history 仅解决相关冲突/范围；Markdown 入口、同名 JSON metadata，旧 .md 可 JSON。仅检查 supplied small evidence。
- covered_refs 是 host 累积 provenance，不等于本次 reread；保留 inherited 覆盖与 fresh inspection 差别。未提供材料是 limitation，不是不存在。
- wisdom 是能改变设计/实践判断/实验选择的少量机制、交互、条件、反例、可靠实际收益/质量成本取舍及 Engineering 支持的实现知识；按问题组织，不按 family/Node 顺序。
- 合并同问题、可保留 IDs；反例导致收窄/未决/撤回并留原因来源；保留有用负结果与判别问题。无新可复用理解则正文不变，无条目可明确说无。
- 每条 Finding / Boundary / Design implication / Sources；exact canonical `nodes/Nxxx/{research,engineering}_analysis_vN.md`，仅 covered_refs 源；区分机制、质量、实测效率、假说/结论。重复不增泛化，不让错误广义结论与修正并列成独立 lessons。
- 不发明算法/无支持参数选择，不每 Node 强制一条、不把 generic audit 变 wisdom；wisdom 不塞 scores/status/logs/workflow。
- Research Summary 按问题导航到原 Analysis/wisdom，保留 handover IDs、机制/实践 verdict 差别；Engineering Summary 记录实现进展、测量条件/比较、保留和否决方向/边界；保留历史导航、不可用信息、区分常规 replication/seed 补测与改变研究路线的问题。
- 不从 Research Analysis 推断 runtime causes、不把未决 validation 升格知识。只一个 JSON，恰好 research_summary/engineering_summary/wisdom 三个非空 Markdown 字符串，body only，无 YAML/processed_analyses；host 累积 coverage/校验源/发布。不得改 Node reports/toolkit/source/commit，不训练评估或诊断。

## 必须可找到的真实命令与共享边界

- planning acquire/renew/release；research register/design/train/timing/diagnostic/decision/finish/status/wait/recover；形式 `autoresearch research <action> --node Nxxx --file /absolute/scratch/data.json --request-id <id>`，status/wait/recover 无 file，wait 用原 ID。不能暗示所有动作都需 file。
- `autoresearch engineering claim Nxxx`、`autoresearch engineering timing --sha <full-HEAD-SHA> --request-id <id>`、`autoresearch engineering finish --analysis-file /absolute/scratch/final.json`。
- `autoresearch diagnostic run --scripts-dir /absolute/path/to/experiments --gpus 4 --timeout 600 -- python experiments/probe.py --output results/probe.json`（Researcher staged diagnostic 用其独立 research 入口）。scripts staged 到 experiments、持久 outputs 在 results、host interpreter/roots，不能随意宿主路径。
- `autoresearch evidence archive --node Nxxx --request-id self-check-1 --files /absolute/scratch/check.py /absolute/scratch/check.log`；小 regular files/unique basenames、32 KiB/bundle、64 bundles/assignment，返回 public-relative anchors；不可用 tmp 当最终证据。
- `git show <SHA>:<file>` 查固定版本；`autoresearch refresh-evaluation --node Nxxx` 更新 Evaluation（host command，不手改）。Tree/README 是导航，current HEAD/source_worktree 非历史恢复锚。
- 直接写仅授权 Worktree/scratch，共享通过 host；不改 Specs/frozen/history/logs/credentials/private tests。managed GPU 唯一入口；Research 四 GPU reservation 开发与 Final 全程保留、serial borrow，不自取 group。
- artifact 只用 supplied dynamic paths（默认 /dev/shm/wyr_tmp）；大 tensors 不复制 root FS、不自改 retention，space/recovery 错误报告；小 evidence 持久，整 Node 完成 host 清理大件。
- transport lost reply 原 ID+同 payload；新执行新 ID，inspect failure，不盲重试/空 commit。wait/recover 不混淆：前者等原请求，后者 host recovery。发布 replay 不应重跑完成工作。
- 原文指定 terminal session 1800 秒 long waits、无 watcher/短轮询/周期日志；改写至少应保持真实支持的长期等待入口与恢复方式，不能只写“等待完成”而删掉 reconnect。
- lifecycle：最多两 unfinished Nodes，slot 贯穿 Research+published Analysis+Engineering complete/合法 skip；capacity pending 不可当 Engineering skip，child 需要 Analysis published 与 Engineering 完成/质量失败或不可用 skip。Root fixed A0、不重开 historical Nodes。

## 第二阶段优劣判据

更好稿应让背景、目的、输入、权限、可操作动作先可见；每节先解释用途，正文放任务、bullet 放条件边界；减少强制思考顺序而保留可证伪研究与决策依据。若短稿把“可以做什么/何时做/哪个命令”或关键统计/权限边界藏到不可达引用，即使更顺口也更差。结论按角色逐份给出，指出具体损失和收益，不宣称文案变化已提升实际研究性能。
