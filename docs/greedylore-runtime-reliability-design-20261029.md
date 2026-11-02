# CORAL AutoResearch 运行时、诊断与资源访问

更新：2026-11-02，依据仓库实现 `64aa372`。本文件保留历史路径以兼容既有设计引用，正文说明当前机制。整体分工、评估和产物分别见 [架构](greedylore-autoresearch-architecture.md)、[工作流](greedylore-autoresearch-workflow.md)、[产物说明](greedylore-autoresearch-artifacts.md)；历史测试结论见 [验证索引](greedylore-runtime-reliability-validation-20261029.md)。仓库已实现与活跃 Run 已部署是不同状态，本文不据此宣称任何历史 Run 已升级。

## 1. 权威来源与身份

| 内容 | 权威位置 |
| --- | --- |
| Seed、评估任务、门槛和固定参数 | Run 保存的 `private/eval_contract.yaml`、配置与代码快照 |
| assignment、阶段、预算、请求及资源账本 | 宿主私有状态；公开文件为受控投影 |
| 角色目标与研究判断 | 任务 `roles/`、`prompts/shared_research_goal.md` |
| 执行约定与实际操作 | [workflow_rules.md](../tasks/greedylore_autoresearch/roles/workflow_rules.md)、[workflow_research.md](../tasks/greedylore_autoresearch/roles/workflow_research.md)、[Researcher 命令](../tasks/greedylore_autoresearch/references/research_commands.md)、[Engineer 命令](../tasks/greedylore_autoresearch/references/engineer_commands.md) |

可信 Unix socket 服务结合已登记进程及其身份校验请求归属。候选中的 Node 字段、路径或环境变量不能自行授予跨节点权限。Agent 只在授权 worktree/scratch 工作，通过命令请求正式状态与产物变更；私有 journal、评分和预算不由 Agent 直接改写。

## 2. 状态、回执与等待

命令请求限制为 64 KiB，响应独立限制为 4 MiB；两类客户端共用有界传输与错误处理。`research status` 保留完整 journal 并提供 `progress`；`--summary` 返回有界投影，最多最近 256 个 job，长文本裁剪，完整结果通过 `research wait --request-id` 查询。

状态区分开发质量、最终各组件、发布状态和正式资格。单个组件 completed、开发质量通过或请求已接受，都不表示候选已取得最终正式分数。公开状态查询不改变冻结记录或消耗执行预算。

正式提交的 acceptance receipt 在 pending Attempt 持久化后立即给出完整 SHA，再进入等待；拒绝不产生成功回执。`engineering finish` 先持久化作者提交并返回 `finalizing`，`--wait` 才等待验证、发布和退休。发布失败或连接中断后复用原请求／相同内容，不能换身份、body 或 SHA 猜测成功。

长任务沿用原终端会话，以 **1800 秒阻塞等待**（支持该参数的终端调用为 `yield_time_ms=1800000`）；结束可提前返回并立即继续。连接中断时先按原 request ID 查询状态；同 ID、同内容重放只恢复原请求。确认未知请求后才按原 ID 重发；故意启动新执行或修正脚本使用新 ID，并计入对应预算。`recover` 请求宿主恢复工作，`wait` 仅查询／等待，二者不等价。

## 3. GPU 所有权与进程恢复

选题锁与四卡 reservation 分离。注册释放选题锁后申请 GPU，Research 阶段持有四卡直至开发与所有最终评估组件终态或明确跳过；训练、timing、诊断串行借用，不各自争抢另一组卡。Engineering 和独立 Analyzer 按受控入口排队，诊断本身不替代正式评分。

Researcher/Engineer 用 Manager 已有 Session resume 和退出恢复机制；Analyzer/Memory 独立模型任务有各自的私有恢复记录。每 Node 的 Analysis 使用独立锁，可并行推进，状态变更与发布只持短共享锁。例行 worker 恢复与全局停止使用不同标志，不能因为轮换 Analyzer 释放其他 Researcher 的 reservation。

当前 Agent `timeout: 10800` 是无输出 watchdog，不是总运行时限；正式 grader `timeout: 0`，pending 评估有 176400 秒 watchdog 豁免。GPU 队列、诊断执行和客户端等待仍有各自期限，见 [GPU_COORDINATION.md](../tasks/greedylore_autoresearch/GPU_COORDINATION.md)。保持原请求可避免重复执行与无意义短轮询。

## 4. 数据与 checkpoint 的可读视图

宿主 `diagnostic_resources` 声明资源 ID、原始路径、允许的链接目标根和可选稳定视图。模板包含 C4、tokenizer、CIFAR100；实际清单在公开 `notes/toolkit/runtime_resources.json`。`diagnostic_runtime_roots` 用于运行库等直接只读挂载。资源解析结果同时供角色会话和实际诊断环境使用。

只有显式允许的链接目标会被解析和挂载；目标缺失、循环、越界或过宽授权在使用 GPU 前拒绝。保留原路径以兼容保存的数据流，另外提供 `/resources/...` 视图；不能用任意软链接获得额外宿主访问。宿主存在的路径不一定在模型内层沙箱可读，临时目录被重新隔离后尤其如此，应使用受管命令与返回的资源视图。

checkpoint 的公开 manifest 保存固定身份及哈希，私有账本记录实际 tensor 位置。普通响应提供公开元信息和访问／保留状态；只有受管诊断可读宿主为本任务选定的只读 tensor 挂载。`diagnostic_path` 是诊断命名空间地址，不是任意 shell 的本地路径。裁剪或清理后保留 manifest，明确报告 tensor 不可用，不改写既有原始证据。

资源目录不可用时公开投影记录错误，但已冻结产物的发布与清理仍可进行；新执行及 preflight 仍拒绝缺失的必要资源。动态存储默认使用专用 `/dev/shm/wyr_tmp` 子树，短 IPC 临时目录与 tensor 存储分离；不把整个宿主共享内存目录开放给 Agent。

## 5. 诊断环境与预检

`autoresearch preflight --sha <full-commit-sha>` 核验提交元数据、Contract、assignment、协议资产、声明资源和已提交入口的静态语法／来源，并做 CPU 诊断命名空间检查；不创建 Attempt、GPU lease 或诊断预算扣费。环境准备还会在 GPU 启动前检查真实 CPU 多进程 tensor IPC。

静态检查不在可信控制进程中导入候选代码，也不证明动态依赖、算法语义或科学等价性。旧协议缺少 legacy 文件时明确失败，不悄悄恢复用户删除的资产或切换协议。恢复训练继续检查固定 SHA、配置、world size 和方法／bucket 状态，不能用重置 EF 等办法绕过不兼容。

受管诊断提供 localhost 解析和 IPv4 loopback。需要单机 DDP 时可在托管命令 argv 中使用：

```text
python /run/coral-ipv4-launch.py --nproc-per-node 2 --port 0 experiments/probe.py
```

该 helper 使用 `127.0.0.1` 的 static rendezvous，保留进程退出状态；不申请 GPU，也不修改正式训练配方。自动端口在探测后释放，torchrun 绑定前存在短竞态；可指定宿主管理的端口，冲突按诊断请求规则处理。实际解释器以任务环境为准。

## 6. Scratch、归档与引用

Engineer 首次 claim 返回 `/tmp/coral-engineer/<scratch_identity>/<engineer-id>/`，identity 持久化于本 Run 的 worker。重复 claim 和同 Run 恢复沿用它，不同 Run 即使 Agent 名相同也不同。脚本需来自授权 worktree 或当前 claim 的 scratch；通用旧目录和其他 Run 的 scratch 不被接纳。

诊断在获准执行时冻结代码、脚本和必要输入，保存受管记录、stdout/stderr 和结果。Analyzer 的 `/workspace` 代码／证据只读，`experiments/` 与 `results/` 是工作区，`/workspace/records` 可读本任务记录。命令返回持久 `record_ref` 及有界日志尾部；应核对失败、修正和实际归档，不以工作文件仍存在代替发布。

新 Analysis 的明确引用需对应冻结证据、固定 Git 文件或本任务终态受管诊断的持久归档。临时 `/workspace/results/...` 等未归档引用在封存前拒绝；确实缺失的证据可作为限制说明，不能暗示已核实。附件声明、实际受管执行和人工注入测试分别记录。发布与下游读取使用同一哈希规则；详细格式交给产物文档和代码维护。

## 实现入口

传输与回执：[client.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/client.py)、[trusted_writes.py](../coral/grader/trusted_writes.py)、[提交 hook](../coral/hooks/post_commit.py)。环境与资源：[runtime_resources.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/runtime_resources.py)、[preflight.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/preflight.py)、[diagnostic_sandbox.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/diagnostic_sandbox.py)。身份与证据：[engineer_scratch.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/engineer_scratch.py)、[analysis_sources.py](../tasks/greedylore_autoresearch/autoresearch/src/autoresearch/analysis_sources.py)。
