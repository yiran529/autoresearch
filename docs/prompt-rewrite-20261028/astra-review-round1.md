# Astra 独立审核：第一轮中文稿

日期：2026-10-28。对象是中文评审稿，不是已经替换运行入口的实现。以下行号对应本轮读取版本。只修改本审查文件，不修改改稿。

## 同轮修订复核（最新结论）

主编收到先行反馈后进行了修订。本审查人重新读取实际文件，确认：Researcher 新增四字段内容表，完整恢复 B1 的 method／mechanism／predictions／tests 要求；Memory 恢复不从 Research Analysis 推断 runtime 原因，并将实现归因指向 Engineering Analysis，B2 关闭；Analyzer 补实际通信对象、聚合／同步、state/EF、optimizer update，以及故意改变／无法恢复状态和结论范围，G2 的主要问题关闭。

**截至该次复读，没有未解决的阻塞替换项。** 下列初审发现保留作变更依据，不应误读为修订后的现存缺陷。G1、G3、G4、G5 和 G2 的更细训练阶段／控制提示仍属非阻塞改进。

| 当前稿 | 修订后与原稿比较 |
|---|---|
| Researcher | 新稿更好：关键语义已恢复，目的、偏好和可用操作更清楚；可达共享承载真实预算与执行边界 |
| Engineer | 新稿更好：保留锚点、语义、正式测量与交付，职责和操作用途更易辨认 |
| Analyzer | 新稿更好：独立权限／输出完整，关键方法语义和状态恢复限定已恢复，诊断按问题自主选择 |
| Memory | 新稿更好：归因边界已恢复，知识合并／撤回、来源和进展导航更清楚 |
| Shared workflow | 新稿更好：真实协议与预算集中、层次清楚；仍有附件身份表述的小精度问题 |
| Startup | 新稿作为 staged 入口设计更好：去掉重复流程但保留身份、角色入口、状态与恢复；生成／投递验证仍未执行 |

“更好”是语义覆盖后的文字与职责组织判断，绝非实际 Agent 研究表现已提升的结论。

## 结论

新稿在目的、输入地图、可用操作与条件、跨角色阅读上有真实改善；研究偏好基本完整且强度适当。未发现主要预算、Final 协议、输出 schema 或命令拼写被错误改写。尚不建议整组直接作为“无语义损失替换”：两个事项需先修，另有数处可低成本恢复的研究要求与入口提示。原稿在方法语义要求与部分诊断有效性细节上仍更精确。

这里的“阻塞替换”指本次宣称保留既有要求的评审验收，不表示当前生产运行出错，也不表示 schema 必然拒绝新稿生成的输出。

| 稿件 | 当前前后比较 | 判断依据 | 修订后预期 |
|---|---|---|---|
| Researcher | 新稿组织与能力说明更好，旧稿方法设计要求更完整；暂不判新稿全面更优 | 研究倾向准确，操作表/决策与执行分开；但 B1 的方法含义未完整迁移 | 修 B1、补 G1 后可优于原稿 |
| Engineer | 新稿总体更好 | 主目标和可用手段更清楚；selected iteration/A0、固定质量锚点、预算、adaptive semantics、本人 finish 契约都保留 | 少量措辞收紧即可 |
| Analyzer | 新稿可读性更好，原稿调查有效性要求略强；总体接近 | 独立权限/schema 完整，诊断示例真实；实际语义追踪与故意状态变动标注较弱 | 补 G2 后可优于原稿 |
| Memory | 暂优先原稿的边界精确性；新稿结构更好 | 唯一实质退步是 B2 的 runtime attribution 边界松动；其余知识更新逻辑保持 | 修 B2 后新稿更好 |
| Shared workflow | 新稿总体更好 | 分离默认值与保存契约、并列预算、Final 与生命周期清楚；未丢关键幂等/冻结/证据条件 | 补 G3 的引用与精确性 |
| Startup | 作为 staged 设计稿，新稿更好；实际投递等价性未验证 | 身份/路径/恢复入口可读，角色与共享承担细节，legacy 明确排除 | G4 加强显式阅读依赖，落地时测生成文本 |

## 替换前应修复

### B1 — Researcher 的方法描述要求没有完整落到委托引用

位置：`drafts/roles/researcher.md:65`、`:68`；原文 researcher §2.4 的 `method` 字段。

新稿仅要求“方法变化”和四字段非空，字段定义指向 research_commands。该 reference 的注册示例确实保留了 communication/persistent state、mechanism 的 rival、tests 的 matched control/discriminator/falsification，不能把这些误报为完全丢失。但它未完整承载原角色 method 中明确的 **aggregation/synchronization、optimizer behavior、哪个组件回答研究问题**。新稿实现节要求实现符合方法，并不替代登记设计应先说明这些内容。其他可选算法配方也不能稳定承担注册字段的语义要求。

建议在四字段旁加短定义，或至少加正文：“`method` 说明实际通信对象、聚合／同步、optimizer 行为和持久状态，以及哪个组件回应研究问题；`mechanism` 说明证据与主要竞争解释；`predictions` 分开给可证伪的机制、质量和完整周期成本预期；`tests` 给匹配对照、可观测判别及支持／削弱解释的结果。”保留已有“改变组件原有有效作用、副作用”句即可，无须恢复旧的固定思考顺序。

### B2 — Memory 将职责禁区改成了证据充足时可做

位置：`drafts/prompts/global_memory_manager.md:60`；原文 Memory §3 最后一条。

原文是“不从 Research Analysis 推断 runtime causes”；改稿成为“不从 Research Analysis 推断**未经支持的** runtime 原因”。后者允许 Memory 自行认为证据充分后从 Research Analysis 归因。这与“归因归 Engineering、Memory 汇总已发布认识”的角色分工相比是权限/职责弱化，不是普通中文润色。

建议改为：“runtime 原因的归纳依据 Engineering Analysis；不从 Research Analysis 自行推断 runtime 原因，也不把未完成验证变成既定知识。”这不妨碍引用 Research 已验证质量／成本结果，只限制新归因。

## 一般改进与需明确的取舍

### G1 — Researcher 的契约条件可更直接，避免‘按需阅读’吞掉操作必读项

位置：Researcher `:29`、`:65`、`:92`，writing-rules `:16`。

原文 design 前读 construction basic guidance、相应操作前读 research_commands。用户要求去掉不必要路线和固定阅读顺序，所以将配方改按需是合理变化，不据此阻塞。不同的是 schema/权限/操作接口本来是执行前需知道的条件。建议规则写“研究配方与延伸参考按需；执行前确认相应操作接口和契约”，角色首次 register 前明确“先核对命令参考的 origin/design 必填字段与当前评估契约”。

`:92` 的“考虑矩阵和卷积梯度的适用性”比原文“verify support”弱；共享 §5 有“适用于矩阵与卷积梯度”，已承载要求，但改成“验证支持矩阵与卷积梯度”更直接。无需重复 fixed workload 全表。

### G2 — Analyzer 明示实际算法语义与不完整状态恢复

位置：Analyzer `:16`–`:18`、`:56`、`:60`。原文 §1.2 明确 trace communication objects/aggregation/synchronization/state/EF/actual optimizer update；§3.2 明确 identify deliberate changes or missing restoration，实际训练 phases/component interactions；decision-timing claims may need fixed/matched-action controls。

新稿仍引向可达 `common.md`（其定义表覆盖语义轴），也保留 relevant restoration、真实触发、缺失状态局限，因此不判为全丢失或不可达。问题在于“了解方法/恢复相关状态”没有明确要求记录故意状态变动及未恢复项；“时机主张可使用”也比按 claim 需要控制略软。

建议加两句：“从固定代码确认通信对象、聚合／同步、EF 等持久状态及实际 optimizer 更新，不由方法名推定。”、“说明故意改变或未能恢复的状态，覆盖主张涉及的训练阶段与组件交互；时机效应需与固定动作／匹配动作比例等替代解释区分。”这规定证据有效性，不规定诊断顺序或强制每次做同一实验。

原文要求机制解释失败也保留实测收益；新稿开头和 findings 已表达机制／实际分别判断及未解释可靠收益，未算遗漏。若追求最小歧义，可在 §4 加“即使原解释被反驳，也保留已验证实际收益”。

### G3 — 共享协议的两个精度问题

位置：workflow `:25` 与 `:94`（附件行以文本“上传者声明 SHA”定位）。

1. 新增“不在持锁期间等待 GPU”未在旧五 prompt 中明确出现，但与“成功注册释放后才实施”一致，是合理强化而非改变研究路线。其目的为避免串行 planning 阻塞，建议说明这一用途；或者删掉新增句，依赖已有注册依赖。续租/释放只有动作、未展示命令；Researcher 可达 research_commands 有真实 renew/release 命令，因此不是缺接口。共享在这里加“具体 renew/release 命令见 research_commands”可改善自足性。
2. “上传者声明 SHA 和内容 hash 证明身份”不够准确。应区分宿主记录的 authenticated uploader、uploader-declared context SHA、content hash。建议“宿主记录上传者身份、声明的代码 SHA 和内容 hash；hash 校验内容，SHA 存在不证明这些字节执行过。”当前后半句已经正确限制执行认证，属精度改进。

workflow `:50` 说“只有 requested_steps 是可选补充”若按上下文限定 timing 是可接受的，但 legacy submission 原文可 omit timing；本稿开头保存契约优先已覆盖。若未来抽取片段使用，建议写“新提交的 timing 中……；legacy 以保存契约为准”，避免把整份 submission 其他 optional fields 意外变成必填。

### G4 — 精简启动消息后，将共享规则阅读依赖写成明确动作

位置：startup `:14`、`:20`、`:51`；Researcher 背景资料与 Engineer references。

新 startup 初始消息明确读 instruction/role，shared workflow 只有“共享协议为……”路径。两个角色又说“规则见……”或“承载共享执行规则”。引用可达且没有隐藏关键材料，但删除重复预算后，这个依赖比原来更重要。

建议生成 AGENTS 输入段加“执行前阅读共享协议；具体操作按角色指向的参考核对”。不是重引入每步思考顺序，只确保默认预算、private 边界、幂等和锁规则确实被读。不存在要求 Analyzer/Memory 去读这里的问题，它们仍自带边界。

Research mode 缺省 REFINE 的生成逻辑未在文字模板体现。当前稿明确不改 assignment 字段/宿主调度，宿主代码仍负责 default/validation，故不视为 prompt 缺失；落地时必须保留该逻辑，不将 `{research_mode}` 渲染成 None/空值。

### G5 — Engineer 小处

位置：Engineer `:31` 与 `:79`。

原 selection_rationale 1–3 句变成“简要记录”，非硬协议/schema 变化，作为去写作微操可接受，建议在最终差异记录标作有意放松。原文不把 profiler durations 相加当 step time，新稿写“不把重叠的 profiler 时长相加”，字面略窄；建议删“重叠的”，因为即便看似不重叠的 profile segments，其和也不自动等于正式端到端 step time。

## 覆盖结论

| 要求族 | 承载与结论 |
|---|---|
| Parent 价值/连续性/信息增量/成本、handover 优先 | Researcher §3，完整；不是 handover 强制出题 |
| 低秩/稀疏优先；REFINE 挑战；EXPLORE Parent **及现有分支**不同路线 | Researcher §§2–3，完整，prefer 未被变硬禁令 |
| 组合/迁移/adaptive、非纯超参优先 | Researcher §3，完整；“不是优先方向”忠实偏好强度 |
| origin/design/register/planning | 主依赖和字段保留，B1 方法含义缺口；锁命令完整定义仍在可达 reference |
| 开发 train/diagnostic/timing/decision/design | 操作目的、条件、真实入口、decision 不执行均保留；SHA pin、从头/完整状态续跑在 workflow |
| 预算/失败/pending/跨恢复 | Researcher+workflow 和 Engineer 完整；没有发现混淆三类 Research 预算或放宽 hard cap |
| freeze/Final/quality gate/恢复 | 完整；开发 timing optional 与 Final mandatory 区分明确；vision restart、完成 component 保留、无反馈调参均保留 |
| fixed SHA/selected iteration/Root A0 | Engineer 及独立 Analyzer 保留；无 initial Spec/last iteration 替代错误 |
| 托管 GPU/private/write/evidence/idempotence | Researcher/Engineer 在可达 workflow；Analyzer/Memory 自身保留，不依赖不可达共享引用 |
| Analyzer 独立性/禁止 runtime profile/new algorithms | 自身保留；实际方法追踪及状态变动标注是 G2 的弱化 |
| Analysis/Engineering/Memory 输出 | 核心字段、非空、枚举、Q1/Q2/上限、无 extra outputs 均保留；Analyzer 仍英文 JSON |
| Memory provenance/继承与 fresh inspection/合并撤回 | 完整；B2 是来源归因边界例外 |
| 生命周期/Parent eligibility/容量 | workflow 保留；等待容量非 skip，host 完成发布，Researcher 不再认领 |
| 启动去重与 legacy | staged scope 引向角色/共享可达；legacy 排除明确，未作实际生成测试 |

## 写作规则是否忠实于用户偏好

总体忠实。十二条中目的先行、输入权限、偏好强度、能力用途与条件、真实命令、正文任务/bullet 边界、独立角色自足、覆盖审核均与任务一致；特别明确“单纯超参搜索不是优先方向”而非禁令，保留 EXPLORE 的 **AND existing branches**。

规则 §2 的“参考资料按需”建议按 G1 限定研究配方，不误伤真实协议必读；§8 的“宿主负责计数”不能推导为角色不需要知道计数（当前稿没有这样删）；§9 将固定 1800 秒改为当前工具支持的阻塞等待，保留 wait/recover 和不轮询，符合可执行性与当前工具约束，不应硬保留不可用参数。规则未声称真实研究表现或 token 效率提升，合适。

## 核实方法与限制

逐份读本轮六稿、README、writing-rules，与第一阶段五主 prompt 和 dynamic/coral_md 原文要求清单比对；再次独立验证 manifest 七份 source/snapshot 的 SHA-256 全部相等。检查 Analysis/Memory schema 与输入拼接；核实 Analyzer 的 `diagnostic run` 示例由 analysis.py 原样提供，**无需 Engineer 专属 `--scripts-dir`**，1–4 GPU/3600 秒 queue 来自宿主 diagnostic guidance，不是改稿发明。

未执行 GPU 实验、prompt A/B 或运行入口替换；未穷尽所有宿主 enforcement 分支。结论仅针对文字语义、可达上下文与所查接口。文中行号在修订后会变化；复审应按事项 ID 和对应句核对。
