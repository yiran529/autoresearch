# Astra 最终定向复核

2026-10-28。复核对象为当前六份中文评审稿及 writing-rules；本次根据 round1 的 B1/B2/G1–G5 定向读取实际修订文本，没有重新开展整轮研究或修改运行入口。以下判断接续 `astra-requirements.md` 与 `astra-review-round1.md` 的原文、参考和接口审核。

## 逐项结论

| 项目 | 当前承载位置 | 判断与理由 |
|---|---|---|
| B1 方法设计要求 | Researcher 67–77 行四字段表 | **认可，关闭。** method 明确通信对象、聚合／同步、optimizer、持久状态和回应问题的组件；mechanism 有竞争解释、原有效作用及副作用；predictions 三类可证伪；tests 有对照、判别和支持／削弱结果。四字段非空字符串保留，不再仅把未完整定义的要求委托 command guide。 |
| B2 Memory 归因边界 | Memory 60 行 | **认可，关闭。** 恢复“不从 Research Analysis 推断 runtime 原因”，实现归因依据 Engineering Analysis，未继续保留“未经支持”这一可放宽职责的限定。 |
| G1 参考与操作契约、梯度支持 | writing-rules 16–18 行；Researcher 83、100 行 | **认可，关闭。** 明确区分研究配方按需与操作前核对接口／权限／契约；register 前核对 origin/design 必填字段；“验证对矩阵和卷积梯度的支持”恢复实际要求。没有强制研究路线或配方执行顺序。 |
| G2 Analyzer 实际语义与诊断有效性 | Analyzer 19、56、60–61 行 | **认可，关闭。** 实际通信至 optimizer update 的语义链、主张相关训练阶段／交互、故意改变／未恢复状态及其结论范围均明确。时机效应要求区分动作本身与选择时机，控制方式仍按需，不变成每次都做的实验清单。 |
| G3 共享规则精度 | workflow 25、51、94 行 | **认可，关闭。** renew/release 明确引用真实 command guide；额外的持锁等 GPU 禁句删除，原注册前后依赖仍在；timing 字段与 legacy 契约限定清楚；authenticated uploader、declared code SHA、content hash、非执行认证分别表达。 |
| G4 启动阅读依赖 | startup 20 行 | **认可，关闭。** 生成说明明确要求执行前阅读 shared workflow，再按角色参考核对接口／契约。精简重复预算后，剩余约束有显式阅读入口；不要求 Analyzer/Memory 依赖此启动模板。 |
| G5 Engineer 小处 | Engineer 35、78、82 行及字段表 | **认可，关闭。** profiler 时长不能相加冒充正式 step time，已移除“重叠的”狭窄限定；JSON 恰含四字段明确。selection_rationale 的 1–3 句改“简要记录”是有意取消句数微操，保留成本→影响→改动→预期内容，符合用户偏好，不改变 schema 或预算。 |

本次没有不认可或仍待修复的已报告项。没有因为文件变新、变短或中文结构更长而判定修订更好；依据是具体要求恢复后，目标、能力、条件与交付之间的关系更清楚。

## 六稿最终前后比较

| 稿件 | 前后优劣结论 | 具体收益及保留条件 |
|---|---|---|
| Researcher | **当前新稿更好** | 先解释实际效率和研究认识的目的，Parent 偏好及研究模式清楚；操作有用途与阶段条件，设计实质字段已补齐。真实预算、幂等与冻结通过显式可达 workflow 承载，不把可选开发 timing 变提名门槛。 |
| Engineer | **当前新稿更好** | 优化目标、工具用途与停止／交付更分明；固定 selected iteration/Research SHA/A0、adaptive semantics、inherited quality、正式预算、本人 Analysis 和发布等待保留。放松句数而保留所需论据是有益自主空间。 |
| Independent Analyzer | **当前新稿更好** | 输入范围、独立判断与可用调查手段更易定位；恢复实际算法语义和状态控制后，未用泛泛自主性代替证据要求。权限、诊断入口、英文 JSON/schema 与有效结论边界独立保留。 |
| Memory | **当前新稿更好** | 角色用途、知识筛选、合并／收窄／撤回与进展导航更清楚；来源归因禁区已恢复，covered_refs 不等于 fresh inspection，纯复现或无新增不强造 wisdom。 |
| Shared workflow | **当前新稿更好** | 预算、冻结、生命周期、幂等与证据分别组织，默认值和保存契约关系更准确。关键非显然约束没有靠“宿主会管”省掉；可达参考承担命令细节。 |
| Startup | **作为 staged 中文入口设计，当前新稿更好** | 身份、角色路径、进度和恢复动作直接可见；不重复整套角色流程，shared 阅读依赖已明确。此结论不等于模板生成和实际消息投递已验证。 |

## 要求、约束与倾向覆盖

在本次已审范围内，**没有已知未修复的实质遗漏、错误接口或意外权限扩张**。这不是穷尽所有宿主分支的形式化等价证明。

- 研究倾向仍准确：Parent 的研究价值／连续性／信息增量／验证成本、优先评估 handover、当前低秩与稀疏、REFINE 深入／挑战、EXPLORE 区别 Parent **及已有分支**、有动机的组合／迁移／动态自适应、单纯超参搜索非优先；偏好未变禁令。
- 执行条件保留：锁与注册、append-only 设计、固定初始核心假设、同 SHA 完整状态续跑与新设计从头训练、三类独立开发额度、Final 冻结／新计时／禁止反馈调参、基础设施恢复、固定 Research 工程锚点、正式计数和提交后等待。
- 跨角色边界保留：Researcher/Engineer 的共享规则有显式入口；Analyzer/Memory 自带 fixed-evidence、private/write、角色能力和输出契约，不依赖它们未被保证读取的共享文件。
- 证据要求保留：失败、反证、原预测、持久来源、附件非执行认证、机制与实际收益分开、Final 复用不再独立 held-out、历史契约不重标。输出字段与语言、枚举、非空条件、问题额度均在此前审核后保留。

## 保留限制

本次是定向文字复核。第一轮已核对原文快照及相关代码/schema；本轮未重新穷尽 runtime 或所有引用文件。运行中的 task prompt、生成入口与 schema 未在本审查中更改。

Startup 尚无生成内容、真实投递、路径注入和 mode default/validation 的执行测试；落地时应验证 staged 分支确实投递角色／共享文件，保持 host 默认 REFINE 与 legacy 分支行为。当前文本不能代替该验证。

没有执行 prompt A/B、训练、GPU 评估或 Agent 行为实验，因此不声称研究质量、实验成功率、运行效率或 token 消耗获得实际提升。最终认可的是中文稿的语义覆盖、职责组织和可执行说明质量。
