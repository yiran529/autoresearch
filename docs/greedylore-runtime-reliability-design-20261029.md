# GreedyLoRE 运行时可靠性：接口与资源访问设计

状态：2026-10-29 用户批准实施，分阶段验证与提交；尚未部署到正在运行的实验。

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
- 阶段 1：请求限额 64 KiB、响应限额 4 MiB；状态增加只读阶段进度与显式有界摘要，正式资格来自发布的 Attempt。定向回归见任务报告。
- 阶段 2：宿主 `diagnostic_resources` 声明由同一有界解析器生成挂载与 `public/notes/toolkit/runtime_resources.json`（会话准备时发布，状态读取不写入）。Researcher、Engineer、Analyzer 和实际诊断执行共用其结构，保留原始路径，允许额外 `/resources/<id>` 视图。每次环境准备解析一次；所有实际挂载在 `/dev/shm` tmpfs 之后。每份声明最多 64 个目标根，总目录扫描最多 4096 个条目、深度 64、每条链接最多 40 跳，拒绝缺失、循环、越界、非普通文件和过宽根。显式目标根授权整个只读目录；仅选中被链接引用的目标根，且扫描其内容。稳定视图拒绝逃出源根的相对链接，应改用共同父根或省略 view。
- 阶段 2 实际模板：C4 源 `/home/wyr/greedy_lore/c4/c4_en/en`，目标仅 `/dev/shm/wyr_tmp/c4-half`，视图 `/resources/c4`；tokenizer 声明 model 专用 `models--t5-base` 目录（包含 snapshots/blobs，使相对链接保真），视图 `/resources/tokenizer`；CIFAR100 源 `/home/wyr/ARC-TopK-release/data`，视图 `/resources/cifar100`。当前宿主 C4 链接目标缺失：不修复、不下载、不更改运行中实验，部署前需宿主恢复已授权数据。
- 阶段 2 checkpoint：公开响应按私有账本映射公开 manifest，给出稳定 ref、manifest hash、access 和 retention；兼容 checkpoint_path 仅表示元数据目录；diagnostic_path 复用原有稳定 tensor view 地址供获准的诊断脚本使用，清理后为 null。普通角色通过现有 SRT 本 run public 只读白名单读取；不扩展核心 home 权限。Researcher 仅挂载当前 iteration 的宿主选中 checkpoint，Analyzer 仅挂载固定 Attempt 的选中 checkpoint；Engineer 保留无 tensor 选择的现有边界。available 不等于任意诊断均可访问。原 journal/cache/evidence 的宿主路径与哈希不改写；读取不新增证据文件。pruned/cleaned/unavailable 明确区分，选择已裁剪目标不再抛文件不存在错误。
- 阶段 2 验证：主工作区定向回归 153 passed；隔离 `/tmp` 源码副本（仅该副本补齐 HEAD 的两个 legacy 测试资产）最终资源、路径、Engineer、staged、handover 联合回归 227 passed，包含真实 SRT、嵌套 bwrap 与 UDS；无 GPU、训练或模型调用。工作区原有两个删除保持未暂存。阶段 4 与最终验证待实施。

- 阶段 3：增加 `autoresearch preflight --sha <full-sha>`，经已有 UDS 进程身份认证，仅接受 SHA；复用 immutable submission validator 的宿主 worktree/contract/parent/island 上下文。顺序报告 metadata、ownership、protocol_assets、resources、entry、namespace；输出候选 SHA、契约 hash、实际固定入口 Git blob/source/hash。静态 AST 不导入候选代码、不加载依赖、不认证方法等价。Researcher 与 Engineer 按各自 active assignment 检查，不调用 formal budget authorizer，不创建 Attempt、lease、GPU reservation 或 journal 记录。Engineer 仍验证冻结 Research anchor。
- 阶段 3 环境：同一 `build_sandbox(resources=...)` 供 CPU 预检与诊断使用；host-owned 最小 hosts/nsswitch 不覆盖显式挂载。命名空间实际读取声明资源中的普通文件首字节（4096 个 inode 有界），验证 localhost 与 IPv4 loopback，环境不继承凭据或 PYTHONPATH，且无 GPU 节点。暂存探针目录在 `/tmp` 清理；原有 staged CPU tensor IPC 检查独立保留。元数据/缺失协议资产不会自动改写或切换协议。
- 阶段 3 打包：离线 wheel 检查发现 ancestor `autoresearch/` ignore 会使 wheel 仅含 metadata；对已限定 `src/autoresearch` 包的 hatch 构建关闭 VCS ignore、排除 pycache，确保 host-owned localhost 文件和 launcher 随包交付。离线 wheel 内容逐字节验证及其实际 bwrap 探针通过。
- 阶段 3 验证：最终主工作区 namespace/environment/submission/CLI/trusted-write 定向回归 178 passed；隔离 `/tmp` 历史资产兼容副本上的资源、Engineer、handoff、contract 与真实 SRT 检查 66 passed。含真实 authenticated UDS、只读/凭据隔离、CPU 两 rank Gloo all-reduce、worker 失败与 SIGTERM；无 GPU、训练、模型或包网络下载。两份 legacy 删除仍保持未暂存。

- 阶段 4：`coral eval` 在持久受理后、等待评分前向 stderr 输出完整 SHA；Engineering timing 同时输出 request ID。`--no-wait` 仍完成受理；拒绝不输出受理事件。Engineering `finish --analysis-file <file> --wait` 以同一内容重放已受理提交直到发布完成；默认仍返回 finalizing，断线时使用同一文件重试。
- 资源目录发布遇到缺失或无效声明时写入 `status: unavailable` 和错误，允许已有冻结 Analysis 发布、清理和 roster 恢复；预检及实际诊断独立解析资源，仍在 GPU 前 fail closed。
- 独立操作演练记录在本地 `.superpowers/sdd/2026-10-29-greedylore-runtime-reliability/simulation-task1.md`、`simulation-task2.md`、`simulation-task3.md`：分别覆盖状态/通信、资源与 checkpoint 视图、真实 namespace 与 CPU Gloo 预检；采用临时合成数据，不代表在线实验或 GPU 验证。
- 最终离线回归：六种 staged 情景全部通过（真实 CLI、认证、控制器与发布，训练、GPU 和模型边界合成）；真实 SRT 双场景通过，当前源码加历史角色资产的独立兼容副本 115 项通过。主工作区两份原有 legacy 删除仍未恢复，在线服务和数据未部署或修改。
