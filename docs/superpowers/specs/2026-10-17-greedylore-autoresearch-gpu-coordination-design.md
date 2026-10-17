# GreedyLoRE AutoResearch — GPU 共享与跨 Run 协调设计

**日期：** 2026-10-17

**状态：** 已完成 GPT-6.1-Sol（medium）设计初审、意见修订和复核，待用户审阅；尚未实现。

**代码基线：** `74e3de25b1168ce2d040ae3545386f4560461978`。

**适用范围：** Linux 单机、同一宿主 UID 启动的多个独立 CORAL run，各自保留自己的评测队列。

## 1. 修改目的与边界

用户希望同时满足以下要求：

1. 预训练任务（本任务的 quality workloads）通过开关允许与其他用户的 GPU 计算进程共用显卡。
2. 不主动把任务放到已有自己计算进程的卡上。“自己”指宿主 Linux UID，不限当前 run、Agent 或代码仓库。
3. Timing 只在空闲卡上启动，不允许有其他计算进程共用；运行中发现外部进程加入时，不发布受干扰的计时成绩。
4. 多个独立 CORAL run 使用同一个 GPU 锁域，避免 GPU 尚未出现计算进程时被重复分配。
5. 当不同 run 各自的资源条件都满足时，允许同时执行；资源不足时自动等待，而非立即消耗一次 infra retry。

本次不提高单个 run 的评测并发：`grader.parallel.max_workers: 1` 保留。Agent 容量、Node/Attempt 归属、Engineering 次数、评测公式和训练预算均不变。不建设统一的跨 run 服务、全局 FIFO 队列或多机调度器。

## 2. 当前实现与问题

| 当前代码 | 行为 | 本次需要改变的原因 |
|---|---|---|
| `gpu.py:GpuInventory.from_nvidia_smi()` | 获取 GPU 指标、compute PID 和拓扑，不获取进程 UID | 无法区分自己与其他用户 |
| `gpu.py:GpuDevice.idle()` | 根据进程数、显存、利用率判断空闲 | quality 与 timing 需要不同策略 |
| `gpu.py:reserve_idle_gpus()` | 一次查询、逐卡加锁、二次查询；不足四卡立即报 infra | 需要统一锁域及等待机制 |
| `grader.py:Grader._formal()` | 使用 `private_dir/gpu_locks`，一次预留贯穿 quality 和 timing | 各 run 的锁互不可见；共享 quality 不能直接转入 timing |
| `process.py:run_process_group()` | 创建独立进程组、等待、超时清理 | 需要运行中检查及取消时清理 |
| `coral/grader/daemon.py` | 每 run 独立 daemon，按各自 `max_workers` 执行队列 | 独立 daemon 已可并发，无须新增全局队列 |

现有空闲条件来自 `taskdata/eval_contract.yaml`：无计算进程、显存占用不超过 512 MiB、GPU 利用率不超过 5%。继续作为严格空闲策略使用。

## 3. 方案选择

选择**任务侧公共文件锁 + 各 run 自主等待资源**。与新增全局调度 daemon 相比，它能复用现有队列、`flock` 和进程清理逻辑，避免增加服务生命周期、集中队列持久化和授权接口。

仅提高 `max_workers` 不解决独立 run 之间的锁冲突；仅移动锁目录也不能解决 quality/timing 策略不同及资源不足立即失败的问题，因此这些改动需要一起完成。

保证范围是遵守同一协议的新版 grader。同 UID 的旧版 run、手动任务和其他用户进程不受文件锁强制约束；GPU 实际进程检查仍然必需。上线前应让旧版评测退出，再统一启动新版，不能宣称新旧锁协议互通。

## 4. 配置与兼容

在 `task.yaml` 的 `grader.args` 增加以下任务侧配置，读取后严格校验类型和范围：

```yaml
pretraining_allow_other_users: false
gpu_lock_dir: null
gpu_poll_interval_seconds: 5
gpu_wait_timeout_seconds: 86400
timing_monitor_interval_seconds: 1
```

- 共享开关默认关闭；开启只影响 quality，不放宽 timing。
- `gpu_lock_dir: null` 解析为 `~/.coral/gpu-locks/<host-id>/uid-<uid>/`，各 run 不得回退到自己的 `private_dir`。`host-id` 使用统一的宿主 machine-id 摘要；同一宿主所有 run 必须解析到同一实际目录。
- 锁目录应处于支持本机 `flock` 的文件系统，目录归当前 UID 所有且权限为 `0700`，锁文件不开放继承。自定义目录必须是所有相关 run 共用的宿主绝对路径，运行时记录解析结果；不同目录不会互相协调。
- 默认路径避开当前 sandbox 明确放行的 `~/.cache`、`~/.local` 和 `/tmp`。Agent 不获得公共锁目录的额外读写权限；grader 在 Agent sandbox 外管理锁。
- 轮询与监测间隔为有限正数；等待超时为有限非负数，`0` 表示只尝试一次，正值表示每阶段允许等待的秒数。不提供默认无限等待。
- 24 小时是初始运维默认值，不是任务得分协议。等待时间不计入训练步时或候选子进程执行超时；仍受显式配置的整个 grader timeout 约束。当前任务的整个 grader timeout 为 `0`。

现有 immutable `eval_contract.yaml` 已包含 `hardware.selection: any_idle_four` 和空闲阈值，且参与 contract hash，不能声称资源策略从未属于 contract。本增量保留文件及 hash：`any_idle_four` 和既有阈值继续约束 timing 及共享关闭时的 quality；仅当 `pretraining_allow_other_users: true` 时，明确允许 quality 选卡以任务配置覆盖这一默认调度策略。该例外不改变固定训练参数、quality 判定、训练预算或 timing 的严格空闲条件。证据必须记录独立的 `allocation_policy_version`、开关值及每阶段实际策略，不能只凭 contract hash 宣称新旧资源环境相同；历史结果不追溯改标。

共享 quality 本身不自动要求重建 Dense references。新增 timing 监测是否允许复用旧 efficiency reference 则单独验收：在代表性空闲环境中，用相同 GPU、固定 Dense 命令和交错重复，比较原控制器与新增监测控制器，记录步时、重复离散度和监测开销；在采样前固定等价容差，不能看到结果后再扩大容差。复用要求差异及其不确定性满足该容差，且资源/监测环境有可追溯记录；不能因为“未发现显著差异”或 hash 相同就默认兼容。

如果没有上述证据、扰动超过容差，或并行环境使对照不可比，则在发布正式 speedup 前通过既有参考数据生成流程补测 Dense reference，并记录与候选一致的监测策略版本和环境。单元测试不要求 GPU 性能实验，但上线前的 reference 兼容验收不能省略。无需仅为这个运维记录修改训练 contract；应最小增加 reference 侧策略元数据及读取/校验逻辑，旧 reference 缺少元数据时按需补充有来源的兼容证据，不能悄悄给历史测量补写未经验证的标签。

配置修改应通过启动新 run 生效，不承诺对已启动 daemon 热更新。Validation 分支继续只构建命令，不申请 GPU、不等待资源。

## 5. GPU 与进程归属信息

保留 GPU UUID、物理索引、型号、显存占用、利用率、NUMA 和拓扑；增加各 compute PID 的宿主 UID 与进程身份信息。

进程身份至少包括 PID 与 `/proc/<pid>/stat` 的启动时间，避免只凭 PID 判断而误认复用的进程。UID 从宿主 `/proc` 获得；读取前后核对进程身份。PID namespace 不一致、权限不足、身份变化或其他不可确认情况，都不得推断为“其他用户”。进程在查询过程中退出时重新取样；本轮不能确认的卡不参与分配。

“自己的进程”按当前 grader 的宿主 UID 判断。卡上同时存在自己和其他 UID 的进程时仍拒绝。不存在进程但显存或利用率不符合空闲条件的卡，也不因开启共享而自动放行；共享例外要求实际存在且全部可确认的其他 UID 计算进程。

## 6. 分阶段选卡规则

| 卡上状态 | quality：共享关闭 | quality：共享开启 | timing 启动 |
|---|---|---|---|
| 满足既有严格空闲条件 | 允许 | 允许，优先 | 允许 |
| 只有已确认的其他 UID 计算进程 | 拒绝 | 允许，作为空闲卡不足时的补充 | 拒绝 |
| 存在当前 UID 的计算进程 | 拒绝 | 拒绝 | 拒绝 |
| 进程归属或 inventory 无法确认 | 拒绝 | 拒绝 | 拒绝 |
| 被其他新版 CORAL run 持锁 | 拒绝 | 拒绝 | 拒绝 |

先选择严格空闲卡，再考虑可共享卡；同一优先级按物理索引排序，保持行为可解释。允许混合选择空闲卡和其他用户占用的卡，但每阶段仍需完整四卡。

共享 quality 的补充卡不应用 512 MiB/5% 的空闲门槛，否则共享开关没有实际作用。本阶段不新增 GPU 显存估计器，不保证共享卡剩余显存能容纳模型；OOM 按既有候选执行失败路径处理，同时保留共享证据，不凭 stderr 猜测并自动改写失败类别。

quality 完成申请后，自己的训练进程必然出现在卡上；此时不能再次套用“存在自己 UID 就拒绝”的分配规则。该规则用于主动加入前的筛选及后续阶段的重新申请，不用于终止本次训练自身。

## 7. 跨 run 锁与等待算法

公共目录包含一个短时 `allocator.lock` 和每个 GPU UUID 的长期锁文件。锁文件一旦创建不在常规释放时 unlink，避免同路径被重建为不同 inode 而绕开 `flock`。退出、异常和正常释放均关闭锁 FD；锁的有效性由内核持锁状态决定，不由文件存在或元数据中的 PID 决定。

每次申请：

1. 在可取消、受截止时间约束的循环中尝试非阻塞获取 `allocator.lock`。持有时间仅覆盖本轮查询与选卡，不覆盖等待或训练；`nvidia-smi` 查询本身也必须有有限超时。
2. 查询 inventory，按阶段规则排序候选卡，非阻塞获取 UUID 锁。
3. 无法取得完整四卡则释放本轮所有 UUID 锁及 `allocator.lock`，记录等待原因后退出本轮。不能持有两张卡睡眠等待另外两张。
4. 取得四卡后再次查询，重新确认选中的 UUID、阶段规则及身份；确认失败则释放全部锁，稍后重试。
5. 确认成功后释放 `allocator.lock`，把四张卡的 UUID 锁交给阶段 reservation 持有。通过 `CUDA_VISIBLE_DEVICES` 传入物理索引。
6. 失败本轮后按轮询间隔并加入小幅随机抖动等待，期间仍检查取消与截止时间。查询失败与“资源暂时繁忙”分别记录，不能把坏配置、锁目录权限错误等永久错误当作普通等待无限重试。

正常资源繁忙期间 Attempt 保持现有非终态 `pending`，不新增 Attempt、不调用 `autoresearch retry`、不增加 infra retry 次数。每阶段写出 `waiting / reserved / running / released` 的资源状态证据，不把这些状态冒充新的 Attempt 状态机。

等待超过期限时返回有明确原因的 infra failure；用户随后仍可按既有同 SHA retry 流程操作。锁目录不可用、配置非法、inventory 持续不可用等错误应区分并给出诊断，不等待完整 24 小时才报告配置问题。

各 daemon 竞争资源，不保证跨 run FIFO、公平性或 timing 优先级。分配锁、统一获取顺序和失败时释放全部卡解决协议内死锁；轮询抖动降低抢锁冲突，但不提供严格无饥饿保证。

## 8. Quality 与 Timing 生命周期

Research Attempt 的流程：

```text
等待 quality 的四张卡
  -> 预留 -> 顺序执行全部 quality workloads
  -> 确认本次子进程组已清理 -> 释放 quality reservation
等待 timing 的四张严格空闲卡（允许换卡）
  -> 预留 -> 顺序执行完整 timing 矩阵
  -> 清理子进程 -> 释放 timing reservation
  -> 使用 timing 的实际硬件环境校验可比性并评分
```

Engineering Attempt 直接进入 timing 阶段。等待时不得保留上一阶段的 GPU 锁。刚退出训练的利用率或显存指标未回落时，按严格空闲规则继续等待，不能绕过检查。

完整 timing 矩阵的两种 precision 和重复实验使用同一组 GPU，reservation 不在矩阵内部更换；每次启动 timing 子进程前复查选中卡无外部计算进程。硬件 fingerprint、NUMA、GPU 名称及 UUID 从 timing reservation 计算，不能沿用 quality 的数据。

取得 timing reservation 后、启动第一个计时子进程前，使用现有 `validate_efficiency_comparability()` 对照 Dense environment 做一次硬件可比性预检，并检查第4节的监测/reference 兼容证据。已取得的四卡若不兼容，释放 reservation 并明确返回 `incomparable_hardware` 等 infra 原因，不执行完整矩阵后才发现不能评分，也不把已知不兼容错误当作普通资源繁忙无限等待。本增量不实现任意四卡组合搜索器；异构机器需使用与 reference 匹配的资源配置，后续再根据实际需求扩展选卡约束。计时结束后仍保留环境及结果一致性校验。

quality 已成功但 timing 暂时无卡时，在同一存活 worker 中保留已完成结果并等待，不重跑 quality。若 worker 已退出、等待超时或同 SHA retry，则先保留原始证据，后续按现有 Attempt/retry 隔离机制重新执行；本次不新增跨 worker 的 quality checkpoint 或自动复用。

## 9. Timing 干扰检查与进程清理

`run_process_group()` 保留现有独立进程组和 SIGTERM→宽限期→SIGKILL 清理方式，将一次长 `wait()` 改成支持检查回调的短轮询等待。回调由 grader 绑定 GPU reservation 和本次启动的进程身份，不把任意配置字符串执行成回调。

运行中仅允许本次 timing 创建的进程树/受管进程组使用选中的 GPU；同 UID 的无关进程也属于干扰。不能只凭 UID 放行，更不能把本次 torchrun ranks 误认成抢卡者。使用启动时间和进程组/会话归属校验 PID，覆盖 torchrun 的 launcher、rank 启动和退出过渡；出现未知进程或无法确认归属时按干扰处理。

检测到外部 compute PID，或关键进程 inventory 查询无法可靠完成时：

1. 记录检测时间、选中卡、外部 PID/UID 或查询错误。
2. 只清理本次创建的进程组，不发送信号给外部进程。
3. 返回带明确 `interference` 原因的执行结果，由 `_completed_json()` 在普通退出码检查之前转为 `InfraFailure`，不能误分类为 `CandidateError`。
4. 即使候选已经生成结果文件，也不发布本次 timing 的有效速度分数；保留已完成单元与失败证据，继续走既有同 SHA infra retry。

完整 timing reservation 内的子进程间隙也要检查，最后一个子进程结束后执行最终检查。监测期间不以“存在 GPU 利用率”或高显存判断干扰，因为本次 timing 自身必然产生这些指标。

轮询监测不能阻止外部进程启动，也不能保证捕获两次采样之间的短暂占用。因此此方案保证的是协作式独占及发现干扰后的无效化，不是驱动级强隔离。监测成本也需实际验证：检查在宿主控制器执行，不向候选训练循环插桩；保持对照与候选监测设置一致，不能未经验证宣称零测量扰动。不同卡的并行任务仍可能竞争 CPU、内存带宽、磁盘或互连资源，本次“禁止同卡共享”不等于整机独占。

## 10. 停止、崩溃与重启

当前 daemon 的 stop event 不直接传入 grader 子进程，manager 等待约 10 秒后才会终止未退出的 daemon。设计不假定“等待函数有 cancellation callback”就已完成停止链路。

任务侧 worker 启动时记录直接父进程（grader daemon）的稳定身份，在资源等待与子进程监控中检查父进程是否仍存活；优先使用 Linux pidfd，兼容路径使用 PID + 启动时间。采样等待需要分片，生命周期检查间隔不超过 1 秒，不受 `gpu_poll_interval_seconds` 放大；inventory 查询使用有限超时，不把阻塞外部命令排除在取消延迟验收之外。daemon 退出后 worker 停止申请资源，清理自己的运行进程组并释放锁，不在后台继续训练或等待。收到可处理的 SIGTERM/SIGINT 也走相同的清理路径。

具体接线放在任务侧 `Grader.grade()` 的主线程异步入口：注册生命周期上下文和信号处理器，再调用 `super().grade()`；在 finally 恢复原处理器并关闭 pidfd。`TaskGrader.grade()` 会把 `evaluate()` 放进线程池，因此不得在 `evaluate()` 内调用 `signal.signal()`。信号处理器只设置线程安全的取消标志，不直接执行查询、写日志或清理；由资源等待与进程执行循环观察标志并进入现有 finally 清理路径。生产 worker 在主线程调用 grade；不满足该条件的嵌入调用需显式验证并报告限制，不能悄悄忽略取消机制。

该方案沿用现有 manager 停止宽限期，不承诺 stop event 触发后立即取消，也不需要先改通用 daemon 的协议。实现测试必须覆盖实际父进程死亡及 manager 强制结束 daemon 的路径；若任务侧方案无法满足清理验收，再提出有证据的最小 framework 变更，不能静默扩大范围。

worker 被 SIGKILL 时不能承诺执行 finally 清理；内核释放它持有的 FD 锁，但遗留训练进程可能仍占卡。后续 allocator 必须看到这些同 UID 计算进程并跳过，不能仅凭锁释放判定卡空闲。若存在可确认的孤儿，清理只针对本次可证明归属的进程；不扩展为杀死其他 run 的 GPU 进程。

## 11. 证据与可观测性

在每次 Attempt 自己的 `eval_logs_dir` 下新增资源记录，例如 `gpu_resources.jsonl` 和当前状态快照，至少记录：

- run/Attempt、quality 或 timing 阶段、时间、等待累计时长与原因；
- `allocation_policy_version`、策略配置、当前 UID、公共锁目录、选中 GPU 索引与 UUID；
- 申请时是否共享、可见 compute PID/UID、关键空闲指标；
- reservation 获取/释放、取消、超时、inventory 错误与 timing 干扰；
- query/monitor 开销的可测信息及监测间隔，便于判断环境差异。

等待日志按状态变化及有界周期记录，避免每秒反复写相同内容。对外资源状态由 grader 生成，不授权 Agent 修改锁或状态。公共锁目录中的诊断信息不作为权威 Attempt 记录；不记录其他用户的命令行、环境或凭据。

ScoreBundle 保持现有主要字段与失败类别，只增加必要的资源证据引用/失败原因。已有 Analyzer 和 Evaluation 渲染应继续工作；如新增原因字段被白名单丢弃，应最小适配对应的序列化/展示测试，不重做发布协议。

## 12. 预期修改位置

| 文件 | 修改职责 |
|---|---|
| `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/gpu.py` | GPU/PID 身份、阶段策略、公共锁、完整分配、等待与干扰检查 |
| `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/grader.py` | 配置解析、分阶段 reservation、生命周期检查接线、证据及 infra 分类 |
| `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/process.py` | 可检查/取消的执行循环、干扰原因、可靠进程组清理 |
| `tasks/greedylore_autoresearch/task.yaml` | 新配置及默认值；保留 `parallel.max_workers: 1` |
| `tasks/greedylore_autoresearch/tests/test_gpu_allocator.py` | 策略矩阵、跨进程锁、等待/超时/取消、完整四卡分配 |
| `tasks/greedylore_autoresearch/tests/test_grader.py` | research/engineering 分阶段、GPU 切换、环境证据与 infra 分类 |
| `tasks/greedylore_autoresearch/tests/test_process.py` | 监测回调、进程归属、取消及仅清理本次进程组 |
| `tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py`、`test_handoff_assets.py` | 公共目录隔离和配置资产兼容（按实际现有检查扩展） |
| Dense reference 生成/校验证据及对应 fixtures | 记录监测策略与兼容证据；需要时补测，不修改质量标准或自动认可旧 timing 环境 |
| 本设计及必要的任务配置使用说明 | 共享开关、启动迁移、等待状态及实际独占边界 |

如 `gpu.py` 职责明显膨胀，可把 PID 身份/生命周期帮助函数放入一个任务侧辅助模块；不预先建设通用资源管理框架。不修改 role prompts 来代替实际调度约束；不修改 `eval_contract.yaml` 的计分协议。

## 13. 验收与测试策略

优先使用可控 inventory、时钟和不同 OS 进程测试，不启动真实长预训练，不杀死现有 GPU 任务。

1. **策略：** 共享关闭保持旧行为；开启后优先空闲卡、允许外 UID、拒绝本 UID、拒绝未知归属；混合 UID、PID 复用和 PID 消失均有覆盖。Timing 无条件拒绝外部计算进程。
2. **跨 run 互斥：** 两个进程使用不同 private_dir、同一公共锁目录，在训练进程尚未启动时也不能拿到重叠 UUID。使用进程同步屏障验证，不只做同进程 mock。
3. **并发：** 模拟八张可用卡，两独立 run 同时持有互不重叠的四卡 reservation；第三个等待，前一个释放后自动取得四卡。四卡环境只能同时有一个 reservation。
4. **无部分持锁等待：** 本轮不足四卡、二次检查失败、query 超时、回调异常时释放全部已获锁；下一个进程仍能使用它们。锁文件不删除，进程退出释放锁。
5. **阶段：** quality 可共享而 timing 必须等待；阶段之间释放锁；全部 timing 单元使用一组固定 GPU，评分使用 timing 环境；不兼容 reservation 在第一个 timing 命令前拒绝；engineering 不跑 quality；validation 不碰 GPU。
6. **等待：** 正常忙碌时 Attempt 不终结、不增加 retry；截止时间到期明确为 infra；坏配置立即失败；持续 query 故障有有限重试预算；等待与子进程执行超时分开。
7. **监测：** 本次 launcher/ranks 不误报；外 UID 或同 UID 的无关进程加入时中止并标记 infra；监测失败、末次检查与矩阵间隙覆盖；不向外部 PID 发信号。
8. **生命周期：** 正常退出、SIGTERM、父 daemon 死亡均停止等待/清理本次子进程并释放锁；worker 被强杀后的遗留同 UID GPU 进程会被后续分配拒绝；PID 复用不误认父进程或 rank。
9. **兼容：** contract hash 与训练数值不变；策略版本与 quality 例外可追溯；reference 缺少监测兼容证据时不能自动认可，补测/补证据路径有覆盖；既有命令构建、进程超时、Evaluation/Analysis 及 sandbox 边界检查继续通过。

实现后可由 run owner 在空闲机器上选择做两个独立 run 的短烟雾验证，确认实际 CUDA 可见卡不重叠、等待状态和释放后的启动行为；真实 GPU 计时对监测开销及并行环境的敏感性需另外测量，不以单元测试替代性能证据。

## 14. 实施顺序与 review 关注点

建议顺序是先完成公共锁和跨进程测试，再完成 UID 策略与等待，随后拆分评测阶段，最后接入 timing 监测、生命周期与证据。每步先验证对应行为，再启用共享开关；默认关闭共享仍应能验证跨 run 并发。

独立 review 重点：目的是否忠实于用户需求；任务侧方案是否足够；公共锁域、进程归属和阶段切换是否闭合；等待是否引入停止/重启问题；“timing 不共享”的实际保证是否被夸大；改动成本是否与收益相称。

本次交付只包含设计文档及 review 后的文档修订，不执行上述实现，不改当前 run 的配置，不启动、停止或重试实际训练任务。

## 15. 独立设计 Review 记录

2026-10-17，按用户要求由 **GPT-6.1-Sol，medium reasoning** 对本文初稿及相关实际代码做只读审查。

初审结论：修改目的与用户需求一致，任务侧公共 `flock`、分阶段申请和各 run 自主等待总体合理；保留每 run 的单评测队列合适；未发现架构级 Critical 问题。初审建议先补充两项 Important 再进入实现规划：

| 初审意见 | 文档处理 |
|---|---|
| Python 信号处理不能在 `evaluate()` 线程注册 | 第10节明确在任务侧 `Grader.grade()` 主线程入口注册，处理器只置取消标志，轮询负责清理 |
| 旧 Dense reference 与新增 timing 监测之间缺少兼容条件 | 第4节补充监测开销等价验收、策略元数据和必要时补测；不再笼统承诺复用全部旧 references |
| 现有 contract 已包含 `any_idle_four`，不能泛称资源策略在 contract 外 | 第4节明确 quality 共享是可追溯的配置例外，默认 quality 与 timing 仍受原策略约束 |
| 换卡后可能到矩阵结束才发现硬件不兼容 | 第8节增加首个 timing 命令前的可比性预检及失败释放路径 |

上述修订是对审查意见的设计回应，不代表已实现或已通过真实 GPU 验收。初审无架构推翻建议；后续实现规划应以修订后的正文为准。

同一 reviewer 随后对修订内容做聚焦复核：两项 Important 在文档层面已闭合，quality 调度例外和 timing 预检的说明足够，本设计范围不要求预先修改 contract schema/hash；未发现新增 Critical、Important 或 Minor 问题。最终结论为**可以进入后续实现规划**，不代表实现或真实 GPU 兼容验收已经完成。
