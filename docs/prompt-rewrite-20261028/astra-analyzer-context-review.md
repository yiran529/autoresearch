# Analyzer public context 实现审查

2026-10-28。只读审查当前未提交的 `analysis.py`、新增 `analysis_context.py`、`test_analyzer_public_context.py` 与英文 Analyzer prompt；复核现有 `analysis_sources.py`、`validation.py`、`memory.py` 路径。未改生产代码或测试；用户原有两份 legacy 删除不算本变更。

## 修订后复核：最终结论

已重新读取当前 diff 与补充测试，**两项集成阻塞均已修复，未发现其他必修的上下文复用问题**。下文初审问题和失败记录保留作审查历史，不代表当前状态。

- `analyze` 在扩展前保留 `diagnostic_evidence`，`create_task` 绑定其 local manifest hash；`seal_sources` 收到同一 base evidence 校验执行诊断。最终 Analysis 的完整 manifest 仍包含 context。`read_source`、`_artifact_evidence`、`_reference_path` 未改变，恢复时 local current 与 ledger 身份一致。
- `memory._stage_public_inputs` 先聚合所有 required reports 的 frozen records，再统一将非 context 排在 context 前；不是仅逐 report 排序。跨报告满 256 context 的测试确认当前 raw result 先保留，只使用剩余容量复制 255 个历史文件，原预算和 hash 检查仍生效。
- 限额 fixture 已修复。新增集合覆盖 legacy unfinished task、同版本／新版本、并发单次收集、snapshot 身份／路径／symlink／缺失篡改、异常中断不发表 partial 和 Memory 优先级。
- `test_handover_evidence` 的五种诊断状态同时引用 context 与真实 task.run 生成的 diagnostic records，断言 base ledger identity 并恢复已发表结果，覆盖初审普通 runner 测试遗漏的冲突。

本审查人独立执行四文件回归：

```bash
PYTHONPATH=tasks/greedylore_autoresearch/autoresearch/src .venv/bin/python -m pytest -q tasks/greedylore_autoresearch/tests/test_analyzer_public_context.py tasks/greedylore_autoresearch/tests/test_handover_evidence.py tasks/greedylore_autoresearch/tests/test_analysis_recovery.py tasks/greedylore_autoresearch/tests/test_knowledge_memory.py
```

结果 **75 passed in 1.93s**。主代理另报告更广六文件 105 passed 与 ruff passed；本审查人未独立复跑那两项，不将其混入上述亲自执行结果。

认可当前实现进入第二次提交。范围仍是 CPU 隔离／恢复／引用与 Memory 集成验证，不声称目标 run 已加载新代码或实际研究表现提升；同步时仍需核实 import path 与新 worker 行为，避免打断正在进行的 timing。

## 初审结论（以下保留审查历史）

核心快照设计认可：先冻结、按 version 复用，未扩张 live notes mounts。当前需修复两个集成问题：Memory 配额优先级，以及下述诊断 ledger identity。另有一个新测试 fixture 失败，需要修正并复跑。

## 补充：诊断 ledger 不能直接绑定扩展 context hash

主代理在集成中发现并告知：当前 `create_task` 若绑定 extended manifest hash，已发表 Analysis 的 `_artifact_evidence` 却传入 local-only current evidence 给 `read_source`，其 diagnostic ledger identity 校验会失败。经核对既有代码逻辑认可这个发现，是恢复路径的阻塞问题，不能由仅无诊断 runner 的上下文恢复测试证明通过。

认可修法：保持 DiagnosticTask.evidence_hash 绑定原本 local/formal evidence；context metadata 绑定该 base hash 和 node/mode/version 且不可变；最终 Analysis manifest 独立包含 context records/hash，恢复时仍验证全部 context。如此既保留证据联系，也兼容旧 manager 的 local current validator。

较简单的实现可在 analyze 保留扩展前 `base_evidence`，供 create_task 及 seal_sources 的 authenticated diagnostic read 使用；seal_sources 可用可选参数承载，默认保留旧行为。若采用 projection，必须只剔除 exact own context prefix 与对应 evidence_paths，不得混入 seal 后追加 code/diagnostic 的 hash。需实际 diagnostic runner 执行→发表→恢复的回归，不能只测试无 GPU 的普通 runner 输出。

## 需修复：Memory 中原始证据的优先级

位置：`analysis_context.py::with_public_context` 按路径排序合并 evidence_records；`memory.py::_stage_public_inputs` 遍历该数组并使用统一 `256 files / 8,000,000 bytes` 预算。

context 自身最多 256 个小文本和一个 metadata，按字典序 `diagnostics/analysis-context-*` 往往先于当前 Node 的 `eval_logs`、`research_diagnostics`。积累较多历史文档后，Memory 会先复制导航、历史 summary 和其他 Node 记录，耗尽预算，然后静默跳过本 Node 真正的实验结果／诊断。这样扩展 Analyzer 的背景知识反而削弱 downstream curation 对当前结果的核查能力。

建议在 Memory 中将非 context 原始证据优先于 `diagnostics/analysis-context-*`，context 仅使用剩余配额；或者给 context 单独明确且有界的配额，保留现有 raw evidence 配额。不要扩大整个日志读取权限。测试应放置满额 context 和至少一个排在其后的当前 Node 原始结果，确认后者仍进入 Memory staging，并维持 hash 校验与总体边界。

## 已认可的实现性质

- `_open_directory` 使用逐级 `O_DIRECTORY/O_NOFOLLOW`，`_read` 用文件描述符读取一次并校验 regular/text/UTF-8/size；source symlink 和 ancestor symlink 不会泄漏目标内容。非白名单文件、当前 Node 副本、其他 Node code/log/checkpoint 未被递归引入。
- inventory 记录缺失、不安全与容量截断；文件数、单文件、总 bytes 和目录枚举有边界；metadata 有独立 bound。
- 私有 per-task flock 串行首次收集，临时目录完整验证后 rename 成最终路径；retry 校验旧 snapshot，不重收集。失败清理 staging，不将 partial 当有效上下文。
- `base_evidence_hash` 加 node/mode/version 绑定背景与当前本地证据；旧未完成 diagnostic task 没有 context 时维持旧 hash；已发表 Analysis 走既有恢复分支，不因 live summary 变化失效。
- `analysis-context-*` 不匹配 DiagnosticTask 身份正则，也不匹配 `_artifact_evidence` 的本 Analysis 执行诊断特别分支；作为已有 diagnostics namespace 的普通 manifest record 校验，旧 manager validator 不需认识新前缀。
- context 不进入 runtime 大产物清理范围；Memory 既有 diagnostics prefix、suffix/hash 校验能读取其被引用的小文件，无需开放任意 live notes。
- `seal_sources` 对预先冻结 record 的 canonical path 可接纳；未供应的 notes alias 仍拒绝。未为任意 diagnostics 路径放宽 authenticated `read_source`。
- 新 prompt 明确本 Node 为对象、其他未完成 Node 只代表当时公开进展、summary/wisdom 为派生认识、不跟随未授权链接、精确快照引用、不把快照当实验执行认证。原独立分析权限、诊断预算、证据边界和英文 JSON/schema 要求仍在。

## 正式 prompt 语言与偏好核对

本次扫描五份正式角色／分析 prompt 未发现中文字符；Researcher 保留 Parent 研究价值／连续性／信息增量／验证成本、先评估 handover、低秩与稀疏优先、REFINE 深入／挑战、EXPLORE 与 Parent **and existing branches** 不同路线、组合／迁移／dynamic-adaptive 与 pure hyperparameter search 非优先。英文迁移没有把这些偏好变为禁令。

Analyzer 增量没有删掉状态恢复、实际语义链、机制与实用效果分开、禁止新增 runtime attribution／算法建议等要求。英文稿有较多任务正文改为 bullets，相比用户原“任务正文、条件 bullets”的文风偏好略弱，但不构成这次证据权限实现的阻塞项。

## 本次实际测试

命令：

```bash
PYTHONPATH=tasks/greedylore_autoresearch/autoresearch/src .venv/bin/python -m pytest -q tasks/greedylore_autoresearch/tests/test_analyzer_public_context.py tasks/greedylore_autoresearch/tests/test_analysis_recovery.py tasks/greedylore_autoresearch/tests/test_knowledge_memory.py
```

第一次执行 **52 passed / 1 failed**。失败为 `test_bounded_context_inventory_reports_omissions`：fixture 设置 total cap 40、first accepted file 24 bytes，后续现有文件都因剩余额度不足而拒绝，实际只接受一个文件，尚未达到 file cap 2；因此 `file_limit` 和两个 present 的断言不成立。是测试没有触发所预期边界，不是观察到实现越限。可增加一个 ≤16-byte 文件使第二个文件被接纳，或将 file cap 与 total cap 分开测试。

主代理仍在补修测试，因此以上仅记录已执行状态，不宣称最终全通过。应在修复 Memory 优先级及 fixture 后再次跑相关测试。

## 限制与部署

未实测目标 run 的进程热加载。保持 `_reference_path/_artifact_evidence` 不变可避免旧 manager 因新 namespace 拒绝证据，但新 worker 的实际 import path、不同模块版本混用仍须在同步时检查。已有工程 timing 不应因此被打断；文件更新和已运行会话采用新 prompt 是不同事件。

未执行 GPU、A/B 或全仓测试。当前认可的是隔离、版本冻结和可追溯引用设计；Memory 集成问题修复前不建议把第二阶段标为完全完成。
