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

- `6fe3573`：减少常规 reconcile 对已完成历史证据的重复检查；对应 [test_workflow_hot_path.py](../tasks/greedylore_autoresearch/tests/test_workflow_hot_path.py)，发布／Parent／清理的完整校验仍保留。
- `64aa372`：声明式 GreedyLoRE／ARC-TopK v5 Seed；对应 [test_arctopk_seed.py](../tasks/greedylore_autoresearch/tests/test_arctopk_seed.py)、[test_root_bootstrap.py](../tasks/greedylore_autoresearch/tests/test_root_bootstrap.py)、[staged 模拟](../tasks/greedylore_autoresearch/tests/test_staged_workflow_simulation.py) 及任务 README 的验证范围。此处列出可查入口，不新增通过数量或 GPU 效果结论。

当前主工作区保留两份 legacy 角色文件的既有删除；依赖旧资产的测试可能受影响。阅读旧 Contract 的能力与当前能否初始化相应协议应分别核实。以后新增验证记录只记实际执行的版本、命令、结果与范围；例行文字修改无需复制历史通过数量或新建一份日期报告。
