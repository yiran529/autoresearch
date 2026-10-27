# GreedyLoRE v4 正式运行恢复方案（2026-10-27）
目标：修复真实 staged 运行暴露的环境问题；复用有效 dense/Root，贯通一个真实节点后持续搜索，不设六小时时限。
- [x] 依赖：复用宿主 uv 缓存，候选快照先离线准备环境；环境初始化失败记录为 infra，阻断新节点派发。checkpoint/临时与编译产物保持外部节点归属。
- [x] 持卡：根据真实 keeper 错误和进程身份定位跨请求失败，保留四卡整个研究期间的所有权；验证真实进程与 GPU 借用。
- [x] Analyzer：修正实际可执行文件/沙箱依赖可见性；真实模型读取随机证据并核对内容，不能用模型返回成功替代工具有效性。
- [ ] 新 run：固定修复后的源码，保持评估契约与种子不变；只导入验证过的 dense、Root 质量/工程结果及完整封存证据，不导入失败节点的资格或预算。一个真实节点通过入口、训练、诊断/续跑、最终评估、Analyzer和清理后放开两节点搜索。
- [ ] 清理：确认旧 run 无活进程并验证新 run 引用独立后，清除旧失败节点的 Worktree、无用候选分支与动态大产物；保留小型失败档案/错误统计、有效 Root 与参考证据。记录范围、大小及清理结果。
约束：每项重大代码修改单独提交；本轮不做独立 review。先定向回归/简单模拟，再真实运行验收。环境修复不改变训练配方、U、质量门槛或历史结果。

进展：依赖环境修复定向77项通过；真实旧候选快照使用共享缓存离线安装成功，两个短CUDA进程连续借用同一四卡成功（`/tmp/coral-runtime-recovery-probe-20261027/report.json`）。持卡问题目前未在此真实进程探针复现；继续在完整节点验证，不凭推测改变所有权规则。

Analyzer 验收：发现两处嵌套 tmpfs 可见性问题，非缺少系统安装包。Codex 临时 home 的 helper 路径须位于宿主小型私有会话目录；模型使用只读 `/workspace` 视图，并只开放 experiments/results 写入，真实产物仍在节点 tmpfs。真实模型成功执行 shell 并返回未在提示词中暴露的随机文件内容。证据：`/home/wyr/coral-experiments/analyzer-runtime-probe-20261027/report.json`。

最终定向回归：114 passed、2 skipped，包含实际进程持卡测试与一个完整 staged 成功模拟；额外的环境故障阻断新节点准入检查通过。真实离线安装、两次四卡CUDA借用和真实Analyzer随机证据读取均通过。新run不设置六小时时限；先保留单节点准入闸门，通过真实节点后放开两节点并清理旧失败Worktree/动态产物。

正式首节点进一步定位到租约取消原因：例行 `stop_workflow_worker(cancel_diagnostics=False)` 轮换 Analyzer 时曾短暂发布全局 `workflow_stopping`，令并行 Researcher keeper 退出。改为轮换专用 `workflow_recovering`，仅暂停模型准入；全局关闭继续使用原停止标记并释放研究资源。真实 keeper/注册任务回归先复现失败再通过，相关84项测试通过。新 run 保留首节点方法与失败记录，修复后恢复同一会话，不重置实验预算。

恢复运行：`/home/wyr/coral-experiments/greedylore-v4-restart-20261027/run`，执行源码固定为 `085c43f356d3ae15fe5d558bbf22f62d850f5ec8`。真实四卡 CUDA 探针在 Analyzer 轮换期间两次连续借卡均通过；首节点 N001 已进入真实开发训练并持续记录 loss，GPU 租约正常。首节点完整验收与旧失败节点清理仍待后台监督器执行，不能据此声称完整科学评估已通过。

新增全链路成功模拟在宿主完成（1 passed），覆盖入口、暂停/同 SHA 续跑、独立最终 timing、语言/视觉最终评估、Analyzer、Engineering 和动态产物清理。受限执行沙箱中的同一模拟曾在 Root 初始化处超时，未作为通过证据；进程身份/IPC 集成验收使用宿主执行结果。新的搜索、启动记录和 run 契约均不设六小时时限。
