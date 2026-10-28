# CORAL AutoResearch 中文 prompt 改写与评审

本目录保存根据用户讨论形成的写作规则、中文改稿、原文快照和 astra mid 独立审查。目标是保留任务要求、约束、研究倾向与真实接口，同时使研究目的、可用能力和交付更清楚。

这些文件保存中文评审稿及改写前的对照材料。正式 task prompt 和动态启动入口采用英文，保留已审核稿的要求、约束和研究倾向；中文仅作评审资料。schema、预算和评估协议保持不变，运行实例同步另行记录。

## 写作规则

[完整规则](writing-rules.md) 的核心是：先说明目的和期望结果；给可用输入、ref、权限和能力；保留用户偏好及其强度；把研究顺序和方法选择交给 Agent；必做操作给真实命令；正文承载任务、bullets 承载约束；避免重复但确保独立上下文可达；用语义覆盖评审而非字数决定优劣。

## 改稿与原文

| 内容 | 中文改稿 | 原文来源 |
|---|---|---|
| Researcher | [Researcher](drafts/roles/researcher.md) | [原角色](originals/tasks/greedylore_autoresearch/roles/researcher.md) |
| Engineer | [Engineer](drafts/roles/engineer.md) | [原角色](originals/tasks/greedylore_autoresearch/roles/engineer.md) |
| Independent Research Analyzer | [Analyzer](drafts/prompts/research_analysis.md) | [原 prompt](originals/tasks/greedylore_autoresearch/prompts/research_analysis.md) |
| Global Memory Manager | [Memory](drafts/prompts/global_memory_manager.md) | [原 prompt](originals/tasks/greedylore_autoresearch/prompts/global_memory_manager.md) |
| Shared Workflow Rules | [共享协议](drafts/roles/workflow_rules.md) | [原规则](originals/tasks/greedylore_autoresearch/roles/workflow_rules.md) |
| 动态 AGENTS／启动与恢复消息 | [入口模板](drafts/startup.md) | [dynamic.py](originals/coral/agent/dynamic.py)、[coral_md.py](originals/coral/template/coral_md.py) |

Researcher 沿用本次对话中已形成的中文结构，补齐可达的关键接口与约束。其他角色使用相同写法，按各自职责保留不同的能力和边界。

共享协议保留现有预算、冻结、Parent 资格、重试和证据规则；启动模板减少与角色正文重复的完整流程。Analyzer 和 Memory 是独立输入包，其权限和输出契约在各自稿中直接保留。

## 审查与验证

[astra 原文要求盘点](astra-requirements.md) 提取要求、约束、倾向、接口和输入可达性；[第一轮审核](astra-review-round1.md) 保留初稿发现与逐稿比较；[修订说明](revision-notes.md) 记录已修复事项和有意的文字调整；[最终复核](astra-review-final.md) 逐项确认 B1／B2／G1–G5 已修复，并给出六稿前后优劣结论。

最终复核在已审范围内未发现仍未修复的实质遗漏、接口错误或意外扩权，认为六份修订稿的目的、能力、条件和交付比原文更清楚。Startup 的结论限于 staged 入口设计，尚未执行生成投递测试。

[source-manifest.json](source-manifest.json) 记录文字审核阶段的源 commit、改写前快照路径和 SHA-256。它及历史 verification 证明当时评审未改运行源文件；正式应用后源文件按预期变化。原工作区已有的两份 legacy 角色删除不属于本次修改。

[verification.json](verification.json) 记录最终文件结构、链接、必需 schema 词项、Research 命令名和七份源快照核对结果；它不代替人工语义审查。

本轮为文字与接口审核，不执行 GPU 实验；前后 Agent 行为、研究质量或 token 效率的比较需要另行运行验证。
