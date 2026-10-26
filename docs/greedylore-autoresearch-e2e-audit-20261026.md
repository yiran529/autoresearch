# GreedyLoRE AutoResearch 全链路模拟检查清单

日期：2026-10-26（Asia/Shanghai）。被测代码：`8532a37`。本次只新增验收记录，没有修改生产代码。

## 结果与验证边界
- 全量相关回归：**864 passed、2 skipped**，耗时 267.88 秒；JUnit 共记录 866 项，0 failures、0 errors。
- 完整流程模拟：**6 个新契约 staged 场景 + 2 个兼容多节点场景**，全部通过。
- 补充贯通检查：将本次 staged 成功场景清理后保留的真实发布产物接入真实 Memory 流程，验证最终证据交接、三份文档、增量必读范围和幂等；通过。
- 除 GPU/训练/模型输出等耗时边界外，完整模拟保留真实 CLI、可信 PID 身份绑定、请求、宿主状态、SHA、发布、Analysis 证据哈希与清理。补充检查只脚本化 Memory 撰写内容，未替换知识整理及发布逻辑。
- CPU/Gloo 小模型与真实跨进程文件锁测试另外验证 checkpoint/数据流恢复、vision runner 和持卡逻辑。它们不验证 CUDA/NCCL、实际大模型训练或算法收益。

## 关键检查清单
| 状态 | 检查项 | 本次检查内容与证据 |
| --- | --- | --- |
| [x] | Root 与新契约 | Root 固定 A0 经正式 grader 执行三项质量任务和四次 timing；语言停止步数为 4000/5000，schedule 保持 20000/8393、warmup 保持 2000/1000；旧预算 reference 在申请 GPU 前被拒绝；不完整 vision 和非对象 JSON 正常形成 candidate failure。对应 `test_root_evaluation.py`、contract/reference/grader 测试。 |
| [x] | 注册、选题和历史 | 选题锁、Parent/干净 checkout、`origin + design`、性质/机制起点、首轮设计哈希、Spec 兼容字段、同请求幂等；旧注册形状及改写设计被拒绝。对应 origin/property/planning/record 测试和 staged CLI。 |
| [x] | Researcher 开发循环 | 暂停后保持同 SHA，恢复最终完成 4000 步；训练/设计上限及诊断累计预算跨恢复保留；Researcher 自己观察、诊断和决策。托管接口与材料交付经过模拟，真实模型能否遵循诊断思路不在本次验证范围。 |
| [x] | Checkpoint、dynamics、存储与持卡 | 周期和最终 checkpoint、默认周期与入口覆盖参数、loss/LR 记录；真实 CPU 多 worker/多 rank 恢复；外部产物目录、公开小型证据、终态清理；真实跨进程四卡文件锁在借用进程结束后继续保留。整链 GPU inventory 是模拟，不代表实际 CUDA 排他或容量压力测试。 |
| [x] | 开发 timing | 可在训练前、暂停后或训练完成后独立调用；完成训练不自动测时；首次执行绑定设计/SHA、默认复用匹配的最新记录、主动复测须说明、5 completed/15 total 累计额度；不消耗 diagnostic 额度。对应 staged timing 测试与 CLI 场景。 |
| [x] | 冻结后的最终评估 | 原始时间线复核：冻结态下产生新 timing，随后语言迁移、vision，最后释放 Research lease；开发 timing 不替代最终 timing；最终评估不占开发 timing 额度；无合格候选时不执行最终组件。具体预算在命令与发布结果中核验。 |
| [x] | 任务与方法边界 | ResNet18 固定训练配方、40 epoch/15640 update/10000 test 的结果校验；使用最终 epoch 准确率，默认相对 dense 最多下降 1pp；rank/start/gap 由候选策略定义，策略接口拒绝训练字段；dense 不压缩。元数据校验与 CPU 小模型执行通过，模拟并未真实训练 40 epoch。 |
| [x] | 失败、重试与发布恢复 | 同 ID 重发只取原结果；按 final component 保留已完成结果；基础设施默认两次重试，耗尽仍标 infra；已完成外部训练可验证进程/完整命令后恢复到私有缓存；发布中断只补发布；不切候选、不重开开发。完整场景验证终态路径，精细中断窗口另由故障注入测试验证。 |
| [x] | Analyzer 与 Engineer | Analyzer 独立读取冻结 record、最终评估证据与固定代码；返回单一结构化结果，宿主生成 Markdown/JSON/handover 并核验证据哈希。Engineer 继承选中轮次而非最后一轮，经统一 timing 入口执行，保持 3 成功/10 总次数、重试幂等和统一收尾；不启动独立 Engineering Analyzer。模型的机制判断与语义保持能力未验证。 |
| [x] | 知识、权限、调度与清理 | research summary、engineering summary、wisdom 分开发布；覆盖范围由宿主记录；新节点只要求读新增 Analysis，历史来源仍可用；重复 Memory 不再次调用。跨节点越权、伪造来源、证据改写等被拒绝；兼容多节点并发、Analyzer 失败恢复、进程清理通过。实际 namespace 隔离见下方跳过项。 |

## 完整模拟场景
| 场景 | 新的最终 timing | 语言迁移 / vision 调用 | Engineer | 清理 |
| --- | --- | --- | --- | --- |
| staged 成功 | 1 | 1 / 1 | 完成 | 完成 |
| staged 语言质量失败 | 1 | 1 / 1 | 正确跳过 | 完成 |
| staged 语言执行失败 | 1 | 1 / 1 | 正确跳过 | 完成 |
| staged 无合格候选 | 0 | 0 / 0 | 正确跳过 | 完成 |
| staged vision 执行失败 | 1 | 1 / 1 | 正确跳过 | 完成 |
| staged vision 质量失败 | 1 | 1 / 1 | 正确跳过 | 完成 |
| 兼容流程 normal | 兼容协议 | 三个 Research 节点 | 9 次 Engineer Attempts，3 个节点完成 Engineering | 无遗留模拟进程 |
| 兼容流程 faults | 兼容协议 | 同 SHA 基础设施重试、会话及 Analyzer 恢复 | 9 次 Engineer Attempts，3 个节点完成 Engineering | 无遗留模拟进程 |

额外直接读取六个 staged 原始时间线，核验实际 timing `frozen=true`、最终组件顺序、释放 lease 的位置与 4000/5000 命令参数，没有仅依赖报告中的布尔汇总。

## v2 节点到 Memory 的补充贯通检查
1. 复制本轮 staged 成功场景的已发布证据与宿主节点终态，保留原场景不变。
2. 用实际 `consolidate_memory` 依次处理 N000、N001；仅替换撰写文本的模型调用。
3. N001 输入中实际读取到冻结 `research.json`：开发完成 4000 步，含新 timing、5000 步语言迁移及 vision 最终证据。此时大产物已清理，公开证据仍可用。
4. N001 `required_refs` 仅含 N001 的两份新 Analysis；Root Analysis 和先前 wisdom 仍可用，Root 研究档案不作为新增原始证据重复搬入。
5. 三份知识文档均记录同一组累计四份 Analysis 来源；再次调用返回 `not_due`，不再调用模型，三个文档 hash 均不变化。

## 未验证项
- 两项测试因环境缺少 bubblewrap 而跳过：`test_diagnostics.py:345`（namespaces unavailable）及 `:559`（未安装，真实 namespace 隔离未验证）。
- 未运行真实 GPU 训练、CUDA/NCCL、正式大模型推理或真实规模资源压力测试；不能据此判断算法 quality/efficiency、评估统计置信度或模型研究能力。
- 正式使用新契约仍需实测匹配的 dense/Root 参考，本次所有科学指标是测试夹具，不是可发布的研究结果。

## 可复核产物
目录：`/tmp/coral-e2e-audit-20261026-8532a37/`
- `pytest.log`、`tests.xml`：本次完整输出和逐项结果。
- `audit.json`：额外时间线核验、场景摘要和 Memory 贯通结果。
- `staged/<scenario>/`：六个新契约场景的报告、原始时间线、日志、提示交付记录及发布产物。
- `workflow/normal/`、`workflow/faults/`：兼容多节点全流程报告、交接和提示审计。
- `check_staged_memory.py`、`memory_followthrough/report.json`：补充贯通脚本和结果；`memory_followthrough/.coral/public/notes/` 保留三份发布文档。

复现完整回归：在仓库中设置 `CORAL_STAGED_SIMULATION_DIR` 和 `CORAL_ARTIFACT_REVIEW_DIR` 指向新的输出目录，运行 `.venv/bin/python -m pytest -q -rs --junitxml=<输出目录>/tests.xml tasks/greedylore_autoresearch/tests tests/test_dynamic_agents.py tests/test_trusted_writes.py`。本机 IPC/CPU Gloo 测试需要可用的本地进程通信权限。
