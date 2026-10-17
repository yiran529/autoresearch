# AutoResearch Automatic Finalization Implementation Plan

**Goal:** 自动完成 Evaluation 刷新、Analyzer 启动、assignment 完成和 Memory 检查，保留科研决策与 Engineering 两轮 Prompt 软约束。

**Architecture:** Manager 现有 reconcile 触发配置的 grader-owned 一次性 workflow worker；worker 以非阻塞 flock 串行扫描任务，在 trusted lock 内刷新 Evaluation 并固定收尾意图，锁外运行 Analyzer，再校验并完成 assignment。Engineer 显式请求结束，Research 在不可再 infra retry 的评测终态后自动请求结束。Memory 在每次扫描后检查门槛，复用现有独立 worker。普通静态任务不配置此入口。

- [x] 在 workflow state 测试加入 Research 自动收尾、pending/infra 阻塞、Engineer 请求结束、收尾提交禁用、Analyzer 失败恢复、worker 并发排他与 Memory 门槛用例；运行并观察 RED。
- [x] 实现 `WorkflowStore.request_finish()`、task-local `lifecycle.run_worker()`，复用 `analyze()` 与可信固定路径发布；新增固定 `finish_assignment` 操作和 `engineering finish` CLI。
- [x] reconcile 启动可配置的一次性 workflow worker，复用 `start_task_worker()`；移除 task Researcher Memory startup Hook，更新角色 Prompt、生成指令模板和设计说明。finalizing roster 停止 Producer 并保留资源，完成后回收。
- [x] 运行 workflow、trusted writes、dynamic lifecycle、Memory 回归和 Ruff，再运行完整无 GPU 测试。检查 diff 后提交。

Engineer 轮数不进入 Scheduler 或可信提交限制。现有手动 Analysis/complete API 保留兼容；自动收尾标记后禁止 Agent 发布 Analysis 或提交新 Attempt。同 SHA 可重试 infra failure 不触发 Research 收尾。Analysis 失败保留 finish_requested，下一次 reconcile 可恢复。

验证：完整 CORAL suite 813 passed / 1 skipped；task suite 141 passed；dynamic/template 20 passed；Ruff、diff check 和 `coral validate` 通过。socket 测试和 grader 依赖安装使用允许本地 socket/联网的执行环境。未运行真实 Codex 或 GPU workload。
