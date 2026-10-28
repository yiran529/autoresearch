# Dynamic Assignment 启动入口中文草稿

以下模板对应 staged Research／Engineering 的生成说明和初始／恢复消息。占位符由宿主填入，预算和完整角色内容不在多个入口反复展开。

## 1. 生成的 AGENTS 内容

```markdown
# {task_name}: {agent_id}

{task_description}

## 当前 assignment

你负责 Node `{node_id}` 的 `{workflow_role}` 工作。角色目标、研究倾向、可用操作与交付见 `{shared_dir}/roles/{agent_id}.md`。

{role_context}

## 输入与执行协议

当前 Node 的证据入口在 `{shared_dir}/notes/nodes/{node_id}/README.md`，Tree 和进展汇总在 `{shared_dir}/notes/`。执行前阅读共享协议 `{shared_dir}/notes/toolkit/workflow_rules.md`；具体操作按角色指向的同目录 `references/` 核对接口与契约。

根据角色目标和保存进展自主推进，通过任务命令发布共享记录。

- 代码编辑使用自己的 Worktree，scratch 与资源遵循宿主授权。
- 中断后恢复同一 assignment／Codex Session，保留修改、已冻结提交和已消耗预算。
```

### Research 的 role_context

```markdown
研究模式为宿主指定的 `{research_mode}`，含义见角色正文。使用 `autoresearch research status --node {node_id}` 读取登记、iteration 和阶段，再从已有进展继续。

- 研究模式跨恢复和重试固定，不重新抽样。
- 新研究按共享 planning／register 协议开始；开发中的训练、诊断、测时与 finish 入口见角色。
```

### Engineering 的 role_context

```markdown
目标由宿主绑定。首次在干净 Worktree 中运行 `autoresearch engineering claim {node_id}` 确认 Research 锚点和当前 run_limits；已开始的 assignment 保留修改并恢复现有进展。Research Analysis 独立进行，无需等待。

- 不切换目标、重置计数或取消必需 Engineering。
- 正式测量使用 `autoresearch engineering timing`；达到角色停止条件或宿主上限后自己提交 Engineering Analysis。
```

## 2. 初始或恢复消息

```markdown
{开始或恢复}宿主绑定的 `{workflow_role}` assignment，Node `{node_id}`。

阅读 `{instruction_file}` 和角色文件 `{shared_dir}/roles/{agent_id}.md`，理解目标、输入、可用操作和交付。结合现有 Node 记录与 Worktree 继续推进，具体实验或优化安排由你判断。

{recovery_hint}

- 不重做已完成步骤，不切换 Node，不重置预算。

{如有：本 assignment 的宿主反馈}
```

Research 的 `recovery_hint` 保留状态和幂等重连入口：

```markdown
使用 `autoresearch research status --node {node_id}` 查阅当前任务和结果；等待连接中断时，以 `autoresearch research wait --node {node_id} --request-id <原ID>` 连接已有执行。
```

Engineering 的 `recovery_hint` 提醒读取 claim／Attempts 和提交阶段；尚未发布的 finish 仍属于原 assignment。

## 3. 对应代码与迁移边界

当前重复来源为 `coral/agent/dynamic.py` 的 `staged_research_scope`／`prompt_for_assignment`，以及 `coral/template/coral_md.py` 的 dynamic 分支。实际替换应让生成入口与启动消息只各自承担上述内容，角色和共享协议共同覆盖原 staged scope。

- 本稿不改变 legacy／非动态模板，也不改变 assignment 字段、宿主调度或投递权限。
- 独立 Analyzer 和 Memory 各自保留边界与输出契约，不依赖本启动模板获得权限。
