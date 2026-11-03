# CORAL AutoResearch 历史验证索引

整理：2026-11-02。本文保留既有引用路径，汇总历史验证的版本、证据与限制；**不表示这些检查在当前 HEAD 重新执行过，也不表示活跃 Run 已部署最新代码**。当前机制见 [架构](greedylore-autoresearch-architecture.md)、[工作流](greedylore-autoresearch-workflow.md)、[产物](greedylore-autoresearch-artifacts.md) 和 [运行时](greedylore-runtime-reliability-design-20261029.md)。

## 1. 验证记录与可追溯来源

原顶层临时报告已合并到本索引。完整原文保存在整理前的 Git commit `64aa372`，可按下表路径读取，例如：

```bash
git show 64aa372:docs/greedylore-autoresearch-e2e-audit-20261026.md
```

| 日期／来源文件（`docs/` 下） | 当时的主要证据 | 验证边界 |
| --- | --- | --- |
| 10-26，`greedylore-autoresearch-e2e-audit-20261026.md`，被测 `8532a37` | 864 passed、2 skipped；6 个 staged 场景及 2 个兼容多节点场景；另有清理后产物接入 Memory 的增量／幂等检查 | 训练、GPU、模型输出为替身；缺 bubblewrap 的两项跳过不作为隔离成功 |
| 10-27，`greedylore-v4-recovery-plan-20261027.md` | 短 TMPDIR、IPC 快速失败、来源 SHA、checkpoint JSON 配置规范化及嵌套诊断路径修复；定向检查与真实 CPU/Gloo、局部 CUDA／模型探针分别记录 | 多次分阶段检查不是同一全量回归；局部探针不等于真实首节点完整科学评估；记录末尾仍有未完成的运行验收／清理 |
| 10-29，本文件原版，范围 `e0c3302..8d206cb` | 传输、资源视图、预检、受理回执；6 个 staged 场景通过，补充 97 项检查通过；真实 UDS、SRT/bwrap 与 CPU/Gloo 探针 | 首次受影响套件 234 passed、2 failed，缺 SRT PATH；补齐后 SRT 专项 2 passed。兼容副本的 115+1 项使用历史 legacy 资产，不等于主工作区全量通过 |
| 10-29，`greedylore-luna-behavior-validation-20261029.md` | fresh Luna 自编诊断与 Engineering 总结；生产者退休后独立消费者读取公开证据并核对哈希 | 首轮暴露跨 Run scratch 污染、未归档自由文本引用及哈希规则误读；重复和辅助澄清均保留，不记为首次盲测成功 |
| 10-29，`greedylore-luna-runtime-fixes-validation-20261029.md` | Run 专属 scratch、持久引用校验及清理后消费；199 项唯一相关检查；20 个 Engineer／18 个 Analyzer 已存在证据逐一核验 | GPU／科学输入合成；Analyzer 使用桥接启动器和未启用 authorizer 的测试 publisher；两条原始评估引用缺失，清理后的 checkpoint 不可由公开消费者重算 |

10-29 初始可靠性范围的独立代码审查记录为 0 Critical、0 Important 和 1 个已修正文档问题；审查者没有重跑广泛测试。后续 scratch 与引用修复有独立归档，不能将前一范围的审查自动套用于后续所有提交。

## 2. 保留的证据位置

本地持久归档在 `/home/wyr/coral-validation-artifacts/`：

- `greedylore-luna-behavior-20261029.tar.gz`：初轮行为、失败、干净重复、消费者与辅助哈希复核。原报告记录 SHA-256 为 `45b293051091a980ed4e9f006f50491e0f0bbfa6bfa8b4d2f47fe4341fdb28ee`。
- `greedylore-luna-runtime-fixes-20261029.tar.gz` 及相邻 `.sha256`：修复后的 fixture、生产者记录、退休后证据、首次与修正消费者报告。
- `greedylore-runtime-fixes-reviews-20261029.tar.gz` 及相邻 `.sha256`：修复审查材料。

原 10-26 审计目录为 `/tmp/coral-e2e-audit-20261026-8532a37/`；10-29 staged 输出为 `/tmp/coral-runtime-reliability-staged-final/`。这些是历史临时位置，可能已被清理，不视为跨机器持久接口。其他探针位置与精确命令保留在 Git 原报告中；本次整理未重跑历史实验或逐一重验归档内容。

## 3. 如何解释这些检查

脚本化模拟可覆盖真实 CLI、可信进程身份、冻结、恢复、发布、哈希与清理，同时替换昂贵执行边界。实际模型行为测试另外验证角色能否使用接口，但桥接启动器、合成发布身份及缺失历史输入仍限制结论。CPU/Gloo 状态 roundtrip 不自动证明完整训练中断与连续执行的轨迹等价；任一层检查都不能代替真实 GPU/NCCL 质量和性能测量。

初轮 Luna 暴露的问题已形成 Run 专属 scratch、持久引用校验和哈希规则说明；这不追溯改变初轮失败或宣称所有自由文本文件自动归档。修复后独立消费者首次混用哈希规则并把两个 fixture 当成同一谱系，澄清后的复核及其缺失输入限制仍一并保留。

## 4. 后续实现的验证入口

- 2026-11-03，`638a6fe` 的公共诊断环境与预算继承实现曾通过 **129 项检查，254.61s**，包含真实 CPU 依赖加载、本地 socket 往返和 staged 模拟；昂贵训练及模型输出使用测试替身。用户随后明确整轮作废重跑应重置额度，因此预算继承实现和对应 `test_discarded_budget.py` 已撤销；保留 [诊断环境测试](../tasks/greedylore_autoresearch/tests/test_diagnostic_runtime.py)，原有轨迹、工程评测、timing 与诊断预算按正常账本计数。历史通过数不表示撤销后的版本已重新执行同一套检查。

  撤销后重新执行诊断环境、研究轨迹、staged timing、staged research 与真实诊断往返检查，**87 passed in 7.25s**；另以真实 `authorize_attempt` 验证原有工程评测上限的 7 个准入边界（4 次成功／10 次总执行）。五个预算模块逐字节恢复到 `638a6fe` 前的版本，没有新增重置或继承接口。活动 Run 的角色产物与账本由一次性操作方清理后重新启动，Root／Dense 的 105 个文件哈希保持一致；这不表示新一轮研究已经完成。

- `6fe3573`：减少常规 reconcile 对已完成历史证据的重复检查；对应 [test_workflow_hot_path.py](../tasks/greedylore_autoresearch/tests/test_workflow_hot_path.py)，发布／Parent／清理的完整校验仍保留。
- `64aa372`：声明式 GreedyLoRE／ARC-TopK v5 Seed；对应 [test_arctopk_seed.py](../tasks/greedylore_autoresearch/tests/test_arctopk_seed.py)、[test_root_bootstrap.py](../tasks/greedylore_autoresearch/tests/test_root_bootstrap.py)、[staged 模拟](../tasks/greedylore_autoresearch/tests/test_staged_workflow_simulation.py) 及任务 README 的验证范围。此处列出可查入口，不新增通过数量或 GPU 效果结论。
- 2026-11-02，`9ede28a` 后的 launcher 参数修正：首次实际 Root 准备在训练前暴露 `torchrun` 对 ARC `--r` 的参数缩写歧义；三个真实单进程 CPU launcher 用例先复现失败，加入脚本前的 `--` 后，与 Seed、Root、契约、命令、策略、manifest、最终契约和 timing 的相关检查合计 **121 passed in 20.71s**。用例只运行参数探针，不加载模型或占用 GPU；不证明后续正式质量或性能。失败校准日志保留在 `/home/wyr/coral-experiments/arctopk-root-20261102/.coral/private/dense_reference_attempts/attempt-001-launcher-argparse/`，固定 A0 与 Contract 未重建。
- 2026-11-03，视觉入口校验修正：扩展 [ARC CPU/Gloo 用例](../tasks/greedylore_autoresearch/tests/arctopk_cpu_cases.py)，经过实际 `main()` 先复现 `unsupported vision compressor`，再验证两 rank、6 updates、跨压缩起始步的 Muon 训练，以及最终 checkpoint 中的 ARC 状态。仅替换 GPU 初始化、模型规模及数据来源等测试边界，使用真实 DDP、压缩 hook 和 checkpoint 实现；相关 ARC、视觉、manifest、策略、Root 与命令回归 **51 passed in 23.58s**。这不代表正式 GPU 评测成功；原 Run 的失败 A0 未替换或重试。

当前主工作区保留两份 legacy 角色文件的既有删除；依赖旧资产的测试可能受影响。阅读旧 Contract 的能力与当前能否初始化相应协议应分别核实。以后新增验证记录只记实际执行的版本、命令、结果与范围；例行文字修改无需复制历史通过数量或新建一份日期报告。

2026-11-03，角色模型设置传递：只读 Analyzer 与 Memory 的实际命令构造先复现未传 `--model`，补齐参数后，`test_analysis_cli.py`、`test_analysis_recovery.py`、`test_memory.py`、`test_analyzer_context.py` 合计 **65 passed in 0.95s**。测试替换模型进程边界，不调用真实模型服务；另行检查的 `test_parallel_analysis.py` 有 7 项因上述既有 legacy 文件缺失而失败，未恢复用户删除的资产。

2026-11-03，开场动态上下文外置：以 `9ee9ef0` 为修改前基准，通过 [离线回放脚本](../tasks/greedylore_autoresearch/tests/replay_prompt_artifacts.py) 读取 ARC-TopK Run 的 N000/N001/N002 已发布 Analysis，核对 91/306/181 条证据记录、源文件哈希及完整 manifest 往返一致。Analyzer 角色正文加动态 manifest／评估配置部分分别从 380,321／610,227／406,530 字符减至 16,240／16,837／16,589 字符（减少 95.7%／97.2%／95.9%）；加上首次读取的完整索引后分别为 28,071／55,477／38,225 字符。这是已发布产物的载荷回放，含分析后追加引用，不是历史开场消息精确重建；不含 checkpoint 说明、runner 后缀、后续证据读取或模型行为，不能据此推断整场 token 或科学质量改善。原 Run 只读，保存结果位于 `/home/wyr/coral-prompt-audit-20261103/verified-replay/`。

相关 Analyzer、Memory、公共上下文、checkpoint、引用路径与 CLI 回归 **137 passed、5 skipped**；CPU/Gloo 用例在允许 localhost 的环境通过，5 项真实 namespace 检查因环境无 bubblewrap 跳过，不能视为隔离验收成功。当前 staged 全流程模拟 **7 项通过**（6 个常规场景及 ARC-TopK success），使用脚本化 agent 和合成评估，验证实际启动输入、独立 Analysis、发布及清理，不调用真实模型或 GPU。旧协议全流程模拟的 2 项在启动前因既有 `workflow_rules_legacy.md` 缺失而失败；没有恢复已删除的 legacy 资产。GPT-6.1 Sol（high）独立检查改动及回放，核对 manifest 与索引无遗漏，并复核修正后的必读范围及按字段读取指引，无剩余阻塞项。实现未部署到活跃 Run。

复现载荷回放（输出目录需为新目录）：

```bash
.venv/bin/python tasks/greedylore_autoresearch/tests/replay_prompt_artifacts.py \
  --run /home/wyr/coral-experiments/arctopk-root-20261102 \
  --output /tmp/coral-prompt-replay \
  --baseline 9ee9ef033fda6348a84b7cb016b069099f521696
```
