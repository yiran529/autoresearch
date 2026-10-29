# GreedyLoRE 运行时可靠性：接口与资源访问设计

状态：2026-10-29 各阶段已实施并提交，独立全范围代码审查已完成，唯一文档问题已修正；尚未部署到正在运行的实验。

## 1. 问题、范围与已有设计

本轮延续 [工作流方案](greedylore-autoresearch-workflow-proposal.md)、
[现有架构](greedylore-autoresearch-architecture.md) 和
[artifact 组织说明](greedylore-autoresearch-artifacts.md)。
处理问题 2、3、4、5、9、10，以及问题 11 的环境与入口预检部分：
状态响应超过 64 KiB；数据链接目标未挂载；隔离环境中 localhost 启动失败；
宿主路径被当成 agent 可读路径；阶段与资格含义混淆；提交受理反馈不明确；
必要元数据、入口或运行资源缺失发现过晚。

本轮不变更科学评估契约、方法、种子、质量门槛、计数、持卡政策或已封存证据。
问题 6 的持久任务身份/租约重绑定另行设计，问题 7 的 FP32 对照异常单独诊断，
问题 12 仅保持认证回归。本轮预检不宣称证明完整方法语义接入正确。

## 2. 单一维护来源

| 内容 | 权威来源 | 消费方式 |
|---|---|---|
| 传输限额、响应格式、状态投影 | 实现代码 | 普通客户端、隔离客户端、CLI 和字段说明共用定义 |
| 当前数据、checkpoint、访问视图 | 宿主配置与资源账本 | 同一解析结果生成挂载、公开响应和会话资源说明 |
| 跨角色等待/恢复/资源/发布约定 | `roles/workflow_rules.md` | 角色引用，独立会话装配必要边界 |
| Researcher 操作接口 | `references/research_commands.md` | 命令、参数、输出与恢复示例 |
| Engineer 操作接口 | 新增 `references/engineer_commands.md` | 从 runtime 参考迁入执行和提交接口 |
| 性能诊断与优化方法 | `references/engineer_runtime.md` | 保留分析方法、诊断卡和优化工具箱 |
| 角色目标、判断与交付 | `roles/researcher.md`、`roles/engineer.md` | 少量更新接口引用，不复制共享协议 |

预算与历史实验解释依从宿主保存的契约。保留低秩/稀疏偏好、新方法与机制偏好、
组合互补性和自适应依据的要求，以及完整的 1800 秒阻塞等待、原会话/原 request 恢复要求。

## 3. 通信与可解释状态（阶段 1）

请求继续限制为 64 KiB，响应独立限制为 4 MiB。两种客户端使用同一份无额外依赖的
协议实现；Analyzer 仅复制所需协议模块，不依赖私有 grader 环境。
服务端发送前检查序列化后的响应，超限返回有界、可解析的错误；客户端区分超限、
空帧/截断、无效 JSON 与任务拒绝。换行帧须完整，错误不能悄悄变成任务失败。

默认 status 保留已有关键结构，增加派生的 `progress`；显式 `--summary` 提供有界摘要，
`wait --request-id` 继续读取单项详情。摘要不更改私有 journal，不删除原始证据，
不把进程绑定和完整协议/计时数组重复放入摘要。超大详情可经现有 evidence 引用定位。

阶段说明区分开发质量、Final 各组件、发布状态和正式分数资格。正式资格来自权威
Attempt/评分记录；不能仅凭一个组件 completed 或开发质量通过推断。历史字段保持原义。

## 4. 资源声明与访问视图（阶段 2）

继续使用已配置的 `dynamic_artifact_root`，本轮不迁移活跃 run 的 `/dev/shm` 数据。
短 IPC 临时目录与 tensor 存储独立。宿主位置与消费环境可读入口是不同信息。

新增宿主配置的诊断资源声明，描述资源 ID、原始根、可选稳定视图、允许的链接目标根。
现有 `diagnostic_runtime_roots` 继续用于运行库等直接挂载。数据链接的目标只在显式允许
范围内被解析，目标缺失、循环、越界、过宽授权和预算超限均在 GPU 执行前失败。
按有界规则解析声明的数据目录，自动挂载必要的目标；保留原路径兼容 checkpoint 中
保存的数据路径及原数据流，额外视图仅改变访问地址。

资源解析结果同时供 Researcher、Engineer、Analyzer 及实际诊断进程构造环境。
在创建独立 `/dev/shm` 后安装已授权挂载，并从同一结果生成会话资源清单。
不能通过候选参数或任意软链接获得额外宿主访问。

checkpoint 的公开 manifest 与宿主账本继续保持身份/哈希关系。公开响应提供可读
manifest、稳定引用和访问/保留状态；需要 tensor 的受管诊断只挂载宿主选择的快照。
原始执行记录和私有 journal 留存宿主路径，面向 agent 的投影不原地改写这些证据。
兼容旧 checkpoint 字段时可指向公开元信息目录，并显式说明 tensor 需经受管诊断访问。
资源已清理或裁剪时返回对应状态，不制造仍可读取 tensor 的承诺。

## 5. 诊断环境与预检（阶段 3）

隔离诊断提供最小 localhost 解析环境；宿主有意挂载的解析配置不被覆盖。
增加可复用 IPv4 DDP 启动入口，支持自动端口；普通脚本仍可在该环境启动 torchrun。
入口为 `python /run/coral-ipv4-launch.py --nproc-per-node 2 --port 0 experiments/probe.py`。
已安装 Torch 2.4.0+cu121 的 c10d 动态 rendezvous 会另用 getfqdn 生成 worker MASTER_ADDR，
因此本实现采用单节点 static rendezvous、显式 127.0.0.1。自动端口通过 AF_INET bind(0)
选取后释放，torchrun 绑定前存在短竞态；可用 `--port` 指定宿主管理端口，失败时按既有诊断
请求规则重试。exec 保留 torchrun 退出状态和直接信号传递；正式训练/评分命令不变。
预检在相同挂载与网络命名空间验证文件读取、localhost 和本地连接，并且不占 GPU。

预检复用已有 submission 验证器，提供显式检查入口，并在需要启动工作的边界执行必要
检查。报告候选 SHA、契约、实际入口模块来源和失败阶段。静态加载只在隔离环境执行，
不得在有凭据的可信控制进程导入候选代码。元数据错误与环境资源缺失应区分。
不静默修改 submission，不把预检成功当成方法等价或科学质量证明。

两份 legacy 角色文件的删除属于本轮开始前的工作区状态。预检应明确指出被选择协议
需要但不存在的资产；处理兼容引用须有记录，不能悄悄恢复用户删除的文件或把旧协议
改名为新协议。已有 staged 数据与历史证据读取继续兼容。

## 6. 提交反馈（阶段 4）

提交先捕获候选 SHA，认证并持久化 Attempt 后立即输出受理事件，再等待最终评分。
普通 CLI 的最终 JSON/结果格式保持兼容，进度反馈写入独立的进度通道（如 stderr）。
`--no-wait` 仅跳过评分等待，仍须完成受理。受理前 HEAD 改变仍拒绝，且拒绝不能消耗
正式执行容量。断线时使用原 SHA/request 查已有受理结果，不重复执行。

## 7. 验证与交付

每阶段先补可复现失败测试，完成实现、定向回归和任务审查后提交。设计状态和验证结果
随阶段更新，最后同步现有架构/artifact 文档中的相关说明。

验收包含：大于 64 KiB 的真实 UDS 响应；超限和截断错误；只读状态及旧记录兼容；
数据软链接目标在嵌套沙箱可读且越界目标不可读；普通 agent 的公开 manifest 与受管
诊断 tensor 视图；清理/裁剪后的状态；localhost/IPv4 DDP；提交 HEAD 竞争；冻结
Analysis 的证据集合和哈希不被状态查询或资源说明改写。

开发和提交发生在 `/home/wyr/CORAL` 的 feature 分支。活跃实验源码、进程和存储保持
原状；本轮完成后另行安排常驻服务与客户端的兼容升级边界。

## 8. 实施记录

- 基线：通信、staged 恢复、路径与 handover 定向测试 93 passed、2 skipped。
- 已知工作区状态：两份 legacy 角色文件已删除；涉及旧协议初始化的部分测试会因此
  失败，须单独归因，不作为本轮新问题隐瞒。
- 文档拆分：Engineer 命令、诊断、计时和 Analysis 提交接口已迁入独立参考；性能方法保留在 runtime 参考。定向文档与 handoff 检查 11 passed；toolkit 状态测试因工作区预先删除的 legacy 规则文件失败。隔离完整工作流成功模拟 1 passed、5 deselected（48.88s）；捕获的实际 toolkit 内容确认独立 Engineer 命令参考已交付。
- 阶段 1：请求限额 64 KiB、响应限额 4 MiB；状态增加只读阶段进度与显式有界摘要，正式资格来自发布的 Attempt。定向回归见 [验证记录](greedylore-runtime-reliability-validation-20261029.md)。
- 阶段 2：宿主 `diagnostic_resources` 声明由同一有界解析器生成挂载与 `public/notes/toolkit/runtime_resources.json`（会话准备时发布，状态读取不写入）。Researcher、Engineer、Analyzer 和实际诊断执行共用其结构，保留原始路径，允许额外 `/resources/<id>` 视图。每次环境准备解析一次；所有实际挂载在 `/dev/shm` tmpfs 之后。每份声明最多 64 个目标根，总目录扫描最多 4096 个条目、深度 64、每条链接最多 40 跳，拒绝缺失、循环、越界、非普通文件和过宽根。显式目标根授权整个只读目录；仅选中被链接引用的目标根，且扫描其内容。稳定视图拒绝逃出源根的相对链接，应改用共同父根或省略 view。
- 阶段 2 实际模板：C4 源 `/home/wyr/greedy_lore/c4/c4_en/en`，目标仅 `/dev/shm/wyr_tmp/c4-half`，视图 `/resources/c4`；tokenizer 声明 model 专用 `models--t5-base` 目录（包含 snapshots/blobs，使相对链接保真），视图 `/resources/tokenizer`；CIFAR100 源 `/home/wyr/ARC-TopK-release/data`，视图 `/resources/cifar100`。当前宿主 C4 链接目标缺失：不修复、不下载、不更改运行中实验，部署前需宿主恢复已授权数据。
- 阶段 2 checkpoint：公开响应按私有账本映射公开 manifest，给出稳定 ref、manifest hash、access 和 retention；兼容 checkpoint_path 仅表示元数据目录；diagnostic_path 复用原有稳定 tensor view 地址供获准的诊断脚本使用，清理后为 null。普通角色通过现有 SRT 本 run public 只读白名单读取；不扩展核心 home 权限。Researcher 仅挂载当前 iteration 的宿主选中 checkpoint，Analyzer 仅挂载固定 Attempt 的选中 checkpoint；Engineer 保留无 tensor 选择的现有边界。available 不等于任意诊断均可访问。原 journal/cache/evidence 的宿主路径与哈希不改写；读取不新增证据文件。pruned/cleaned/unavailable 明确区分，选择已裁剪目标不再抛文件不存在错误。
- 阶段 2 验证：主工作区定向回归 153 passed；隔离 `/tmp` 源码副本（仅该副本补齐 HEAD 的两个 legacy 测试资产）最终资源、路径、Engineer、staged、handover 联合回归 227 passed，包含真实 SRT、嵌套 bwrap 与 UDS；无 GPU、训练或模型调用。工作区原有两个删除保持未暂存。最终整体验证见 [验证记录](greedylore-runtime-reliability-validation-20261029.md)。

- 阶段 3：增加 `autoresearch preflight --sha <full-sha>`，经已有 UDS 进程身份认证，仅接受 SHA；复用 immutable submission validator 的宿主 worktree/contract/parent/island 上下文。顺序报告 metadata、ownership、protocol_assets、resources、entry、namespace；输出候选 SHA、契约 hash、实际固定入口 Git blob/source/hash。静态 AST 不导入候选代码、不加载依赖、不认证方法等价。Researcher 与 Engineer 按各自 active assignment 检查，不调用 formal budget authorizer，不创建 Attempt、lease、GPU reservation 或 journal 记录。Engineer 仍验证冻结 Research anchor。
- 阶段 3 环境：同一 `build_sandbox(resources=...)` 供 CPU 预检与诊断使用；host-owned 最小 hosts/nsswitch 不覆盖显式挂载。命名空间实际读取声明资源中的普通文件首字节（4096 个 inode 有界），验证 localhost 与 IPv4 loopback，环境不继承凭据或 PYTHONPATH，且无 GPU 节点。暂存探针目录在 `/tmp` 清理；原有 staged CPU tensor IPC 检查独立保留。元数据/缺失协议资产不会自动改写或切换协议。
- 阶段 3 打包：离线 wheel 检查发现 ancestor `autoresearch/` ignore 会使 wheel 仅含 metadata；对已限定 `src/autoresearch` 包的 hatch 构建关闭 VCS ignore、排除 pycache，确保 host-owned localhost 文件和 launcher 随包交付。离线 wheel 内容逐字节验证及其实际 bwrap 探针通过。
- 阶段 3 验证：最终主工作区 namespace/environment/submission/CLI/trusted-write 定向回归 178 passed；隔离 `/tmp` 历史资产兼容副本上的资源、Engineer、handoff、contract 与真实 SRT 检查 66 passed。含真实 authenticated UDS、只读/凭据隔离、CPU 两 rank Gloo all-reduce、worker 失败与 SIGTERM；无 GPU、训练、模型或包网络下载。两份 legacy 删除仍保持未暂存。

- 阶段 4：`coral eval` 在持久受理后、等待评分前向 stderr 输出完整 SHA；Engineering timing 同时输出 request ID。`--no-wait` 仍完成受理；拒绝不输出受理事件。Engineering `finish --analysis-file <file> --wait` 以同一内容重放已受理提交直到发布完成；默认仍返回 finalizing，断线时使用同一文件重试。
- 资源目录发布遇到缺失或无效声明时写入 `status: unavailable` 和错误，允许已有冻结 Analysis 发布、清理和 roster 恢复；预检及实际诊断独立解析资源，仍在 GPU 前 fail closed。
- 四阶段独立操作演练及其边界见 [验证记录](greedylore-runtime-reliability-validation-20261029.md)；采用临时合成数据，不代表在线实验或 GPU 验证。
- 最终离线回归：六种 staged 情景全部通过（真实 CLI、认证、控制器与发布，训练、GPU 和模型边界合成）；真实 SRT 双场景通过，当前源码加历史角色资产的独立兼容副本 115 项通过。主工作区两份原有 legacy 删除仍未恢复，在线服务和数据未部署或修改。

## 9. Luna 行为测试后的修补

依据 [独立 Luna 测试](greedylore-luna-behavior-validation-20261029.md) 的实测结果，本轮修补两项已获授权的问题：跨 run 复用 `/tmp/<agent-id>` 会混入旧脚本；新 Analysis 正文仍能把未归档临时文件写成证据引用。工作保留在当前 CORAL feature 分支，原有两份 legacy 删除不恢复、不暂存；不部署到在线 run。

接口与权限保持原有边界：诊断仍用 `--scripts-dir`，Engineer 仍可提交本 Worktree 的脚本，不新增 tensor 权限、不改变预算、科学契约或已有 Analysis。新目录由宿主生成并保存在当前 run 的私有工作流状态。新提交的临时引用检查只识别明确引用和已知临时别名；普通路径描述、说明证据缺失及合法持久文件的同名 basename 不应被误判。

### Task 1: 持久化并强制使用本 run 的 Engineer scratch

**实现范围**：`autoresearch/src/autoresearch/engineer_scratch.py`（必要时新增的小型路径模块）、`state.py`、`engineer_diagnostics.py`、`cli.py`；权威操作说明为 `references/engineer_commands.md`。代码路径均相对本 task 根目录。

- 每个 Engineer worker 获得宿主生成、持久化的随机 scratch 身份；不得仅用 `.coral` 路径 hash，因为删除状态后在同一路径创建的新 run 也必须获得新目录。目录形式为 `/tmp/coral-engineer/<32位随机十六进制身份>/<agent-id>/`。同一 run 的 worker 恢复和重复 claim 返回相同目录；另一 run 的同名 agent 获得不同目录。
- `engineering claim` 响应增加 `scratch_dir`。新 worker 在分配时记录身份；旧 active worker 可在 claim 或正常 reconcile 时补齐，保留所有预算、request 和已受理提交。目录内容只通过现有脚本暂存/归档接口使用，不自动导入旧 `/tmp/<agent-id>` 内容。
- 新执行仅允许本 Worktree 或宿主返回的 scratch 子目录，继续拒绝 `..`、软链接和别的 run 的目录。错误给出当前合法目录/claim 入口；不得悄悄改用另一数据流或重置执行次数。
- 终态 request 的同 ID、同 payload 恢复仍须返回原结果；收紧新执行的目录规则不应让已经受理的历史 request 无法恢复。不同 payload 继续拒绝。路径检查应在新执行前完成，失败不消费诊断/正式预算。
- 不删除用户已有的通用 scratch；本轮不引入会破坏 `finish --analysis-file` 重放文件的自动清理。
- 先写并观察失败测试：同名 worker 的跨 run 不同目录；同一路径重建 run 目录不同；恢复路径稳定；当前目录脚本可执行、旧目录和别的 run 目录拒绝且不执行/不扣费；旧 request 在规则变化后仍可恢复；已有 Worktree 脚本和预算/封存行为不回归。使用真实 WorkflowStore/handler，GPU 执行可用既有 CPU 测试替身。
- 运行相应 Engineer/workflow/CLI/handoff 回归与 Ruff，更新本节实施结果，通过后单独 commit。缺失 legacy 测试资产只在临时源码兼容副本补齐，主工作区删除保留。

### Task 2: 新 Analysis 的明确引用与归档规则

**实现范围**：`validation.py`，视调用需要最小修改 `analysis.py`、`analysis_sources.py`、`engineering.py`；权威 Analyzer 说明为 `prompts/research_analysis.md`，跨角色通则放 `roles/workflow_rules.md`，Engineer 专有命令继续引用 `references/engineer_commands.md`，不在多个入口重复全文规则。

- 保留已有临时 `/tmp` 引用校验，在新提交中识别 Markdown 文件链接和 `Evidence:`/`Sources:` 证据子句里的工作文件引用。至少覆盖测试中漏掉的 `Evidence: ...; prompt.txt`、显式链接到 `prompt.txt`、`output.json`、`results/<file>`、`experiments/<file>` 和 `/workspace/results|experiments/...`；不要把已归档的 `diagnostics/.../results/output.json`、合法固定代码文件或普通文字里的未归档/缺失说明误拦。
- 可用既有冻结 inventory、当前 task 的持久 records 和已有 alias 规则验证明确引用；保持已经支持的 `records/<request>/...` 与 `evidence/...` 语义，优先让新正文使用返回的公共相对 `record_ref`。不要自动把任意文件从外层工作区发布成可信证据，也不要对任意自然语言文件名做通用推断。
- 无法作为持久证据的明确引用应在新提交封存前拒绝，错误应说明具体引用及修正办法：把支持材料放入受管诊断的 `results/` 后重新执行并引用记录，或用支持的 evidence archive；Analyzer 不具备 Engineer archive 命令时只提示其实际可用操作。
- 对历史已封存 Analysis 不追溯重验证；相同已受理 Engineering body 的恢复保持幂等。新提交失败不能创建正式 Analysis、冻结 Engineering summary 或损坏已存在证据。
- 在 Analyzer 权威提示中明确：诊断执行时写入 `results/` 的小型输出随记录保存；模型在外层另写的日志和最终输出文件不是自动发布的诊断证据。需要长期引用的日志应在诊断内生成并引用实际持久路径。提交 schema 与自主研究判断保持不变。
- 按 prompt-writing skill 保留研究偏好、角色职责、可选诊断和 1800 秒等待约束；不把归档规则变成额外科学 gate。可顺手补充当前 reader 文档里 `sha256`（文件字节）和 `document_sha256`（规范化 JSON 文档）的已有语义，不改 hash 算法。
- 先写并观察上述失败案例的回归测试，再实现；包括有效同名持久输出通过、缺失说明不误拦、真实 analyze/Engineering submit 拒绝后可修正、历史恢复不回归。通过定向回归、Ruff 和独立审查后单独 commit。

### Task 3: 真实操作模拟与最终验证

- 使用本轮更新后的生产代码和权威提示搭建新隔离环境。仅给 fresh Luna 正常角色入口、资源清单和任务目标，不提供正确 scratch 路径、脚本或预期数值。
- 必须覆盖：Luna 从 claim 读取 `scratch_dir` 并真正使用；预先残留的通用旧目录/另一 run 目录不能进入新脚本快照；受管诊断产生持久结果与日志，Engineer 自主 archive/finish；Analyzer 的明确临时引用由真实校验器拒绝，修正提交后成功发布。
- 模型行为成功与独立宿主故障注入分别记录。校验器反例测试不冒充模型首次就会犯错/修复；如需反馈后续跑，保留首次结果并标记。
- 清理生产者自己的动态/角色 scratch 后，由 fresh consumer 读取实际公共引用并核对哈希；保存失败、修复、实际脚本目录和公开证据清单。科学输入和 GPU reservation 可合成，真实 CLI、SRT/bwrap、UDS 和发布不可用手写产物替代。
- 更新本轮验证结果、进行独立最终审查并 commit；只清理本轮拥有的模拟进程/目录，保留审计归档。在线同步不在本轮授权范围。
