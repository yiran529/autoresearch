# 英文正式 prompt 忠实性独立复核

2026-10-28。基准为 `docs/prompt-rewrite-20261028/drafts/` 五份中文定稿，目标为 `tasks/greedylore_autoresearch/roles/{researcher,engineer,workflow_rules}.md` 和 `prompts/{research_analysis,global_memory_manager}.md`。本审查只写审查文档，不改正式文件。

## 最终结论

已重新读取修订后的实际文件，下面所列语气及四处语义差异均已修复。**认可当前五份英文稿为保持中文定稿结构、范围和强度的翻译；没有已知未修复的忠实性阻塞项。** Analyzer 的 public-context 内容是另行授权增量，不计作翻译出来的原有内容。

最终确认 Memory 使用 `important replication/applicability boundaries` 与 `actual question`；workflow 使用 `the fixed language-transfer task`，并以 `The host retains ... then releases it` 明确资源操作主体。Memory 的可查阅／可进入 wisdom 保持 `may consult`／`may also`，每条内容要求使用 `For each entry, describe`，没有额外升级 must；参考查阅仍保持可选语义。

| 正式英文文件 | 独立复核结论 |
|---|---|
| Researcher | 结构对应，选择偏好及能力／约束强度忠实；非额外报告要求已恢复 |
| Engineer | 结构对应，锚点、优化目标、检查、预算和提交条件忠实 |
| Shared workflow | 结构对应，宿主资源主体、单一语言迁移、恢复／证据／计数语义忠实 |
| Analyzer | 原有部分结构对应，可选诊断恢复可选语气；新增公开上下文范围单独明确 |
| Memory | 结构对应，来源、边界、知识增量、角色权限及可选阅读语义忠实 |

## 对之前判断的纠正

此前 context 实现审查把“中文任务正文被翻成英文 bullets”视为非阻塞文风问题，这一判断不符合用户已经明确的忠实翻译要求。正文、条件 bullets 与段落分组是已审定结构的一部分；翻译不能自行重排。此项须修复后才能认可英文落地，不能用关键词存在、测试通过或整体意义相近替代。

## 方法与结构结果

逐段阅读中英文，核对动作主体、条件、可选／必须／禁止强度、对象范围、研究偏好、命令和输出语义；不是只做词汇匹配。另独立运行 block-sequence 比较，不仅采信主代理的结构 audit。

Analyzer 新获授权的 `Public research context` 小节及专门的 frozen-context 引用 bullet 单独排除；原证据段中增加 context 路径为该功能的必要增量。其余结构比较结果如下：

| 文件 | 中文／英文 blocks | 结构结论 |
|---|---:|---|
| Researcher | 78 / 78 | 一致 |
| Engineer | 36 / 36 | 一致 |
| Shared workflow | 29 / 29 | 一致 |
| Analyzer | 40 / 40 | 排除上述授权增量后一致 |
| Memory | 30 / 30 | 一致 |

比较包含标题级别、正文段落位置、list 分组和每组条数、表格行数、code fence 序列及语言。换行宽度和中英文标点不作为语义变化。当前五稿已恢复“正文承载任务、bullets 承载条件边界”，不再有前轮系统性把正文列表化的问题。

## 已核实的语气修复

- Analyzer 的“可从 initial_diagnostics 选择”恢复为 `you may choose`，未强制先跑工具包。
- Analyzer 的“可通过干预研究”恢复为 `You may study`，未要求每次实施干预。
- Researcher 的“无须另写重复最终报告”恢复为 `No separate duplicate final report is required`；“不再认领其他 Node”的禁令单独保留。
- Researcher `engineering_review` 段移除中文该段没有的额外 `confounds`；共享协议自身的混杂要求仍正确保留。
- Researcher 资料段移除额外的 `before operating` 命令式阅读条件，恢复中文该段的引用用途；这不删除启动入口已有的共享协议阅读要求。

## 本轮逐段语义发现

以下四处已通知主代理，并通过实际落盘文本复读确认关闭；保留发现过程供追溯：

1. Memory“重要复现／适用边界”曾译为 `important replications, and applicability boundaries`，把复现本身与边界并列。应译为 `consequential replication/applicability boundaries`，保持 wisdom 收录的是这些边界。
2. Memory“按实际问题组织”曾译为 `practical question`，容易偏成仅实践问题；应为 `actual issue/question`，包含机制与理论问题。
3. workflow reservation 曾用 `Keep ... then release it. The host then ...` 指令，容易把资源释放主体转给 Agent。应明确 host retains/releases，再由 host retirement/refresh/Analysis，保持宿主托管资源边界。
4. workflow“固定语言迁移”曾译为 `fixed language-transfer tasks`；当前 Final 只有一个语言迁移任务，应使用 singular `task`，不扩大评估任务数。

除以上所列，在本轮逐段阅读范围内未发现其他实质范围、强度、主体或必需内容变化。Engineer 的固定锚点、语义保持、计数和本人交付；workflow 的冻结、幂等、恢复和证据身份；Memory 的累计来源、合并／撤回和无新增时不造条目均对应中文。

Researcher 的 Parent 价值／连续性／信息增量／验证成本、handover 优先、低秩／稀疏优先、REFINE 深入／挑战、EXPLORE 区别 Parent **and existing branches**、组合／迁移／dynamic-adaptive 与纯超参非优先均保持原强度；没有把偏好升级为禁令。Analyzer 的独立性、真实语义追踪、状态限制、机制／实用判断分开及英文输出契约保留。

## 限制

本报告评价翻译忠实性，不替代运行或生成投递验证。结构相同本身不证明语义相同，所以已另作上述逐段审查；语义审查也不是形式化的等价证明。授权新增 public context 按独立实现审查处理，不伪称它是原中文已有内容。没有进行 prompt A/B、GPU 或实际研究表现比较。
