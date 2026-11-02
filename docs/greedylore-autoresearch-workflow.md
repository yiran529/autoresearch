# CORAL AutoResearch 工作流与评估协议

更新：2026-11-02，依据仓库实现 `64aa372`。本文描述 GreedyLoRE/ARC-TopK 的当前分阶段流程；各 Run 保存的 Contract、配置和代码快照是其权威依据。系统分工见 [架构](greedylore-autoresearch-architecture.md)，逐项命令与请求示例见 [Researcher 命令参考](../tasks/greedylore_autoresearch/references/research_commands.md) 和 [Engineer 命令参考](../tasks/greedylore_autoresearch/references/engineer_commands.md)。

## 1. 选择 Seed 与建立 Root

在 CORAL 仓库中准备声明式 v5 Run：

```bash
.venv/bin/python tasks/greedylore_autoresearch/scripts/prepare_seed_run.py \
  --seed arctopk --output /absolute/path/to/new/arc-run

.venv/bin/python tasks/greedylore_autoresearch/scripts/prepare_seed_run.py \
  --seed greedylore --output /absolute/path/to/new/greedy-run
```

准备命令复制共享 seed，选择对应 task 和 v5 Contract，固定来源并创建 `N000` 与其待执行 Attempt；它不启动 worker 或申请 GPU。返回结果提供正式 `coral.cli resume` 命令。`prepare_seed_run.py --resume --output <existing-run>` 核验并复用已保存 Run，不重置 Root、Attempt 或预算，额外的 `--seed` 也不会改变已有 Run 的 Seed。

两份 v5 Contract 的 Dense reference 命名空间分别位于 `taskdata/arctopk/`、`taskdata/greedylore/`，当前随仓库交付的目录为空。解释正式成绩前需要实际建立同 Contract、同硬件协议的 Dense 参考。Root 从固定候选执行三项质量任务及正式 timing，不进行普通节点的选题、开发或候选选择；Root Analysis 与条件 Engineering 完成后才开放后续研究。

原 `task.yaml` 继续选择 GreedyLoRE v4。v5 在 v4 固定训练和 timing 上增加 Seed 身份及方法适配；保存的 v1–v4 Contract 保持原哈希和解释，不自动迁移或重标结果。

## 2. 选题与注册

宿主分配 Node ID 和研究模式，初始 `p_explore=0.20`。`REFINE` 深化 Parent 方法或机制，`EXPLORE` 优先不同机制；模式跨恢复保留。两种模式均可从已有观察、期望性质或待检验机制出发，不必引用一个现成的父问题。

Researcher 先通过 `autoresearch node planning acquire --wait` 获得选题锁，阅读登记方法、相关 Analysis、Wisdom 与工程经验，再注册 `{origin, design, planning_token, overlap_review}`。锁租约默认 1800 秒，可续期或释放；注册原子地释放锁，不持锁等 GPU。宿主保证排他选题，不用字符串相似度代替研究者对方法重叠的判断。

`origin` 固定问题、核心假设、Parent/base commit、证据、选点理由和工程经验的复用或推迟依据；首份 `design` 写明方法、机制及竞争解释、预测和检验。宿主生成初始 Spec 和 version-3 `research.json`，Researcher 不另写一份重复的最终报告。一个 Node 使用一个代码 Parent；核心假设改变另建 Node，固定自适应策略的正常动作变化不算修改设计。

## 3. Development experiments

Researcher 自行实现、检查、提交代码，使用托管训练、按需 timing 和诊断检验预测，记录 `continue / revise / stop` 的证据与理由。`decision` 记录判断，`design` 追加设计；二者都不直接训练或改代码。首次获准训练或 timing 固定该轮 SHA，修改候选需追加设计并从头训练；同 SHA 精确续跑仍属原轨迹。

| 项目 | 当前默认边界 |
| --- | --- |
| 开发轨迹与设计 | 通常不超过两条，最多三条轨迹及三份追加式设计；仅 timing 即淘汰也占设计额度 |
| 第三条轨迹 | 需要有证据支持的修复，或会影响方法选择的关键未知，训练前记录理由 |
| 可选开发 timing | 每 Node 5 次 completed／15 次总执行，跨轮次与恢复累计 |
| 托管诊断 | 10 次成功／20 次总执行，每次至多 600 秒；失败和修正重试保留记录 |
| Checkpoint | 默认每 1000 optimizer updates 保存，并在请求暂停及训练结束时补存；间隔仅由宿主配置 |

训练失败、主动停止和有意重跑消耗轨迹；pending 预留额度。基础设施恢复由宿主分类并按保存协议处理，不能靠换请求或恢复会话重置预算。开发 timing 默认复用同 SHA、同上下文最新完成记录；主动复测需理由，保留全部结果，不能只选最快值。

判断同时考虑实现是否生效、机制预测及竞争解释、质量与完整周期成本。机制解释成立不增加晋级门槛；解释被推翻但候选有效时保留实用价值，并明确认识的修正。暂停后的诊断服务于当前开发决策，不等待 Analyzer 逐轮反馈。

可恢复 checkpoint 保存模型、optimizer/scheduler、方法状态、各 rank RNG、步数与数据流等必要信息。精确续跑要求相同 SHA、兼容配置/world size/bucket 身份及支持的数据流回放；不承诺任意随机预处理或机器重启后的 tmpfs 恢复。具体恢复检查见运行时文档和命令参考。

## 4. 冻结与 Final evaluation

开发结束后，`research finish` 选择一条**已经完整完成且通过开发有效性、质量及资源检查**的轨迹，或明确没有合格候选。开发 timing 为可选，不是提名的前置条件。宿主先冻结选中轮次、SHA、设计、压缩策略和 workload 映射，再依次执行：

1. 全新的强制最终 timing；不复用开发 timing，也不占开发 timing 额度。
2. 固定语言迁移任务。
3. ResNet18/CIFAR100 最终视觉评估。

最终反馈不允许重开开发、调参或换候选。无合格候选时明确跳过最终组件，失败与未知仍进入 Analysis。每个最终组件默认最多两次基础设施重试，已完成组件仅为恢复当前冻结评估而复用；vision 重试从头训练，不声称精确续跑。所有组件终态或明确跳过后，宿主释放 Research 四卡 reservation，退休 Researcher，启动独立 Research Analysis 和质量达标时的 Engineering。

## 5. 当前固定评估任务

以下适用于当前 v4/v5 模板；完整字段以 [evaluation_tasks.md](../tasks/greedylore_autoresearch/references/evaluation_tasks.md) 及对应 [GreedyLoRE](../tasks/greedylore_autoresearch/taskdata/greedylore/eval_contract.yaml)／[ARC-TopK](../tasks/greedylore_autoresearch/taskdata/arctopk/eval_contract.yaml) Contract 为准。

| 阶段 | 固定任务 | 质量／有效性依据 |
| --- | --- | --- |
| 开发 | Llama130M + Muon，BF16，4000 updates；保留 20000 步 schedule | 完成时 validation loss ≤ 同 Contract Dense 的 1.05 倍 |
| 最终语言迁移 | Llama60M + AdamW，BF16，5000 updates；保留 8393 步 schedule | 同上，不使用训练 loss、历史最佳 loss 或 PPL 替代 |
| 最终视觉 | ResNet18/CIFAR100 + Muon，FP32，40 epochs／14080 updates | 最终 epoch 的 validation accuracy 相对 Dense 默认最多下降 1.0 个百分点 |
| timing | Llama350M + AdamW，BF16/FP32，4 GPU，每卡 batch 1 | 两轮反向顺序重复；完整周期、事件与硬件可比性检查 |

CIFAR100 从官方训练集固定按类留出 5000 张作 validation，其余 45000 张训练；日常搜索不加载官方 test。搜索后由操作方冻结少量候选再做官方 test 确认，当前没有自动 campaign-test 阶段。最终评估用于考察开发所选方法的迁移与稳健性；被后续节点反复使用后，不再是整场研究独立的 held-out 证据。

候选通过提交的 `method_policy.py` 定义压缩设置及每任务映射，不能改固定数据、训练预算、优化器、精度或评估设置。GreedyLoRE 的 rank、prefix 和 refresh 默认值属于方法设置；ARC 初始为 `group_topk`、`r=4`、保留比例 0.2、EF14。两者共享 hook 构造入口，Dense 关闭压缩；这些 Root 设置不代表已证明最优。

语言、视觉与 timing 的宿主命令在 `torchrun` 启动参数和训练脚本之间使用 `--`，确保 ARC 的 `--r` 等方法参数传给训练脚本，不被 launcher 的参数缩写规则误解析。分隔符不改变保存的 Contract、训练设置或指标解释。

timing 默认 warmup 101 步、测量 400 步，最多 800 步，需按声明周期对齐，并保留各 rank 的实际事件。固定周期包含相互作用的维护动作；无固定周期时声明 `period_steps: null`。ARC 每压缩步采用新随机投影，使用 null，不伪造 refresh 事件。窗口外稀有动作的长期成本不由一次测量证明。正式速度使用匹配 Dense 的 BF16/FP32 几何平均 speedup；有效的速度退步仍是结果，不增加 speedup 大小晋级门槛。

辅助 utility 为 `U = S × exp(-mean(Lcandidate − LRoot))`：两项语言 loss 差等权、有符号，`S` 为相对阶段匹配 Root 的测时 speedup。Vision 单独设门槛，U 不决定选题、Parent 资格或候选选择。缺失或不兼容的证据返回 null 并说明原因；历史 v1/v2 的旧 U 语义保持不变。

## 6. Analysis、Engineering 与节点完成

独立 Analyzer 读取冻结档案、最终结果、固定代码及封存证据，区分机制与实用结论、反例、未知和后续问题。可以按需进行有界受控诊断，不重开当前节点，也不规定下一代算法。没有新增 finding 是有效结果。

Engineer 从宿主绑定的选中 Research SHA 开始，不等 Research Analysis 才工作。正式运行上限为 **4 次成功／10 次总运行**；有效计分的退步算成功，失败及获准重试算总执行，pending 预留额度。达到上限只停止评测，仍需作者提交涵盖所有候选、语义检查、归因及停止原因的最终 Analysis。质量沿用固定 Research anchor，标为 `inherited (provisional)`。

`autoresearch engineering finish --analysis-file /absolute/scratch/final.json --wait` 等待宿主校验、发布和退休。不带 `--wait` 时接受请求不表示发布完成；恢复复用相同内容。没有独立 Engineering Analyzer。

Research Analysis 已发布且 Engineering 完成或因质量失败／终态不可用明确跳过后，Node 才完成并可作 Parent。宿主随后异步更新 Summary/Wisdom，普通节点在执行与诊断排空后清理大产物，保留代码、记录和可引用的小型证据。Root baseline checkpoint 保留。

## 7. 历史兼容与验证范围

Contract 版本、研究档案版本和工作流状态版本是不同概念。旧 Contract 可读取不代表当前工作区具备运行每个旧协议的所有资产；现有 legacy 角色文件删除会影响依赖它们的初始化测试。工作流状态 v1 不自动迁移为 v2，恢复需使用匹配的运行时和保存配置。不得通过替换哈希、改门槛或改默认 Seed 重解释历史结果。

CPU/Gloo、脚本化工作流与隔离行为检查分别验证其覆盖的执行边界；通过这些检查不证明真实 GPU 质量、性能或算法创新。历史证据与限制集中在 [验证索引](greedylore-runtime-reliability-validation-20261029.md)。
