# Analyzer 跨 Node 公开上下文：只读设计审查

本轮仅阅读代码并提出方案，没有修改生产文件或测试。范围为同 run／同公开 notes root 的其他 Node 文档与全局导航；不扩展代码 SHA、checkpoint、原始日志或私有数据权限。

## 主要发现

`analysis.resolve_evidence` 当前仅加入本 Node research.json、相关运行证据及 Spec 指定的 parent Analysis；`_reference_path` 的 notes 白名单也只允许 versioned Analysis 与 research.json。`_prepare_evidence_snapshot` 将记录复制到隔离 evidence roots。这个隔离模式应保留。

不能仅扩白名单并将 live tree／summary／wisdom／其他 Node evaluation、research.json 填入 raw_references。`_artifact_evidence` 恢复旧 Analysis 时会重新读取每个 present record 的当前源文件并核对 hash；全局 summary 的正常更新会令旧 Analysis 变成 “published Analysis evidence changed”。`state.py` 在 Parent eligibility、发布／恢复中调用这套验证，因此影响不限于旧文档读取。

另一个陷阱是未发表 Analysis 的重试：`diagnostics.create_task` 固定 `evidence_hash` 身份。如果重新收集已经变化的公开上下文，diagnostic task 身份改变，即使本 Node Research 没变也无法正常恢复。

## 建议的最小可靠方案

新增一个独立的小模块管理 **每个 Analysis version 一次性冻结的公开上下文**；使用已有 EvidenceRecord 结构，尽量不改模型输出 schema 和 evidence envelope 的字段形状。

1. 在确定 node/mode/version 后、创建 diagnostic task 前，创建或加载同一 version 的持久快照。例如 public `analysis_context/N001/research/v1/notes/...`，具体目录名可按仓库惯例。首次用 staging directory + atomic rename 发布完整快照和清单；相同版本重试只读旧快照，`revise` 的新版本才取新公开状态。文件保留到 Analysis 长期可验证，不能随大 tensor/runtime artifacts 清理。
2. 仅复制白名单普通文本：`tree.md`、`research_summary.md`、`engineering_summary.md`、`wisdom.md`；`nodes/Nxxx/spec.md`、`evaluation.md`、`research.json`、已发布 `research_analysis_vN.md/json`、`engineering_analysis_vN.md/json`。不要递归整个 notes 或跟随文档链接扩权限。Node README 未在本次要求清单，可不新增。
3. 源限制到当前 context 的 notes_root，同 run／island；拒绝 symlink 文件和任一 symlink ancestor、路径穿越、非普通文件。读取一次 bounded bytes，再用同一 bytes hash+写快照，避免 hash 与 copy 间版本变化。设明确文件／总字节上限，超限或缺失在清单中明确说明，不悄悄暗示全部覆盖。
4. 公开中的 in-progress research.json 是“快照时已公开进展”，不是冻结完成研究。应可读但标记这一性质；不要为阅读其他 Node 而刷新它的 Evaluation、打开 private record 或修改其状态。论文式结论依然要源 Analysis／实际证据支持。
5. 将不可变快照路径作为 present EvidenceRecord 纳入当前 evidence；snapshot inventory 给出 original public path → immutable citation path 映射。模型看到的是隔离 evidence 下的这些小文件，而非整个 live notes_root。tree/summary/wisdom 明确为导航／派生认识，其链接若未供应不自动授权进一步读。
6. 恢复已发表 Analysis 时按其 manifest 中的快照路径验证，不与 live summary 比较。旧 manifest 没有新路径时保持原行为，不 retroactively 重收集上下文。未发表且已有 diagnostic task 的旧版本也必须保留原身份：可继续旧上下文，或只在没有旧 task 时启用扩展；不要不兼容地给旧 task 加 hash。

如需要最小落地，不必为了全部历史文本建立复杂内容寻址存储；version-scoped immutable directory + manifest/hash 已足够。但首次创建与恢复必须明确，不应每次 resolve_evidence 都产生副作用或偷偷创建新快照。

## 各文件修改点

| 文件 | 建议 |
|---|---|
| analysis.py | 保持 resolve_evidence 可供 state/engineering 复用的原本本地证据解析；在新 Research Analysis session 分支、版本确定后增加“加载／冻结上下文并扩展 evidence”步骤。`_reference_path` 为新快照前缀添加严格解析；`_prepare_evidence_snapshot` 读 immutable files；恢复分支不用新上下文替代旧 manifest。 |
| analysis_sources.py | 已在 evidence_records 的 present 路径可由 seal_sources 既有 allowed 集接纳，通常无需开放任意 live notes 的 read_source。若支持便捷 notes alias，只能用首次冻结映射 canonicalize，不能返回时再读 live 文件或接受未列入上下文的路径。保留 code SHA／authenticated diagnostic 规则。 |
| validation.py | 当前 validate_analysis 用 evidence allowlist，新的 immutable records 可自然进入；无需改 output schema。避免新增“任意 notes 路径都可引用”的宽白名单。现有 narrative citation 检查主要拒 tmp，不应声称所有正文引用已逐一认证。 |
| memory.py | `_stage_public_inputs` 目前仅接受有限 prefix，notes 仅允许 research.json。若 Memory 后续需要检查 Analysis 引用的新快照，显式允许该严格前缀并进行 regular-file、hash、预算校验。复制到原 canonical 路径，不重取 live summary，不递归导入所有历史 contexts。 |
| research_analysis prompt | 说明可读的其他 Node published context 与 tree/summary/wisdom，以及冻结时间／进展性质；保持本 Node 为分析对象，历史仅用于相关比较、重复性及边界判断。summary 是导航，不能冒充独立证据；新增上下文不授权其他 Node 代码／checkpoint／诊断训练。 |

已发布 Analysis 的 markdown 可能链接其 JSON sidecar，两者均应供应。公开目录出现单个半发布文件时必须选择明确策略：只纳入已完成发布的版本，或标记缺 sidecar，而不是将缺文件当作未来可变的 present 输入。不可让版本枚举中途新增文件改变已冻结集合。

## 必要回归测试

1. 新 session 能读另一个非 Parent Node 的白名单 spec/evaluation/research.json/versioned Analysis 与四个全局文档；不能读 unrelated-secret、private、credentials、额外 logs/tensors／其他 Node code。覆盖 symlink 文件、ancestor symlink、`..` 和跨 root。
2. 分析中修改 live summary 或其他 Node record：模型输入与发表 manifest 仍对应最初复制的 bytes，引用被 canonicalize 到同一 immutable 文件。
3. 发表后更新／删除 live summary、追加 Node/Analysis 版本：`analyze` 默认恢复及 `WorkflowStore` 的 Parent/发布验证仍成功，不再次调用模型；快照本身篡改则失败。
4. 未发表任务中断后公开上下文变化：重试沿用同一 snapshot/hash/diagnostic identity 与 budget；旧任务（扩展前创建、无 context）有兼容路径；`revise=True` 新版才得到新上下文，v1 仍可验证。
5. 引用校验：已冻结新上下文引用成功，任意 live notes/private/新增文件/其他 version snapshot 引用失败；alias 若提供必须固定映射；无 handover refs 的 verdict 规则不变。
6. Memory 只复制所需、匹配 hash 的新快照证据，live summary 更新不会替换历史内容；总预算、缺失文件和超限行为可预期；不跟随嵌套 manifest 无限递归。
7. 并发首次冻结只产生一个完整快照；故障留下 staging partial 不成为正式输入；恢复不把后来出现文件补入旧快照。
8. 原测试 `test_fixed_snapshot_includes_only_referenced_parent_analysis` 的“只 Parent”语义将变化，但保留 unrelated-secret 隔离断言；`test_research_snapshot_and_roots_exclude_engineering_evidence` 必须继续保护当前 Node 正式 Research anchor、全部 live roots 与本地工程 raw results，不要因可读他 Node published Engineering Analysis 而全删此隔离测试。

优先测试现有 `test_analysis_recovery.py`、`test_analysis_cli.py`、`test_parallel_analysis.py`、Analysis citation/diagnostic tests、`test_knowledge_memory.py`／`test_memory.py` 以及覆盖 Parent eligibility 的 workflow tests。无需 GPU 测试，重点是快照生命周期与权限。

## 同步到运行目录的风险

同步目标已明确，但文件同步不是进程热更新。`state.py` 顶层从 analysis 导入 `_artifact_evidence`／`resolve_evidence`，`lifecycle.py` 顶层导入分析入口，`memory.py` 也导入 runner；长寿命 manager/trusted handler 若缓存模块，会继续用旧函数。新 worker 可能从磁盘读新代码，而旧 manager 用旧 `_reference_path` 拒绝新前缀，导致 Analysis 已发表但 Parent 不合格或 reconcile 失败。不要只 reload analysis 一个模块并假定所有 from-import 绑定更新。

正式同步前需核实目标环境实际 import path、editable install/venv、manager/worker PID 和 active Analysis/diagnostic 状态；安全边界处按项目现有 stop/restart/recover 流程切换依赖进程，保留 session、已冻结提交、IDs 和预算，不杀正在使用的 GPU 工作来“快速生效”。若不能立刻安全重启，应区分“文件已同步”与“新行为已生效”，避免新旧验证代码混用。

同步应只带已提交任务/框架源文件及确需的公开 toolkit/prompt 模板，不能覆盖目标 run 的 `.coral/private`、research.json、Attempts、frozen Analyses、runtime state、已有 Worktree 修改或预算。已启动 agent 不会因模板文件更新自动替换其当前上下文；独立新 Analyzer 才会读新 prompt。旧 evidence 不应为适应新规则被重写。

本审查没有检查目标 run 当前进程状态或执行任何同步，以上为基于代码导入关系的部署风险，落地前需由主代理核实。
