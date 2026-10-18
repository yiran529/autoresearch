# GPU 分配与跨 run 协调

GreedyLoRE grader 在同一 Linux 宿主、同一 UID 的多个独立 CORAL run 之间协调 GPU。每个 run 可以同时保持多个等待 GPU 的评测 worker，实际占用 GPU 的阶段数仍由 `grader.parallel.max_workers` 限制；八张满足条件的卡可以支持两个独立 run 并发执行四卡阶段。Agent 并发数、等待 worker 数和 GPU 执行名额独立。

## 配置

在启动 run 前设置 `task.yaml` 的 `grader.args`：

```yaml
pretraining_allow_other_users: false
gpu_lock_dir: null
gpu_poll_interval_seconds: 5
gpu_wait_timeout_seconds: 86400
timing_monitor_interval_seconds: 1
```

- `pretraining_allow_other_users`：默认关闭。开启后，quality 优先用空闲卡，不足时允许选择只有其他 UID 计算进程的卡。已有本 UID 计算进程、归属未知或被其他 run 预留的卡仍会跳过。共享不保证有足够显存，OOM 继续按候选执行失败处理。
- `gpu_lock_dir`：默认使用 `~/.coral/gpu-locks/<machine-id 摘要>/uid-<uid>/`。自定义时必须是所有相关 run 共用的宿主绝对路径、归当前 UID 所有、权限为 `0700`，并支持 `flock`。不要放进 Agent 可以写入的位置，不要在运行时删除锁文件或切换目录。
- `gpu_poll_interval_seconds`：资源不足后的重试间隔，加入少量随机抖动。
- `gpu_wait_timeout_seconds`：每阶段资源等待上限；`0` 表示只尝试一次。等待不计入候选执行时间，但仍受整个 grader timeout 限制。
- `timing_monitor_interval_seconds`：timing 运行中的检查间隔，必须大于零。改变它需要重新验证 Dense reference 监测条件。

配置不热更新。升级时先让旧 grader 退出，再启动新版；旧版使用 run 私有锁，与公共锁不互通。需同步升级 CORAL 框架和任务 grader：框架提供可信父进程启动身份，缺少该上下文的旧框架会被任务 grader 明确拒绝。不同自定义锁目录、不同 UID 或不共享相同宿主文件系统的容器不会通过这些锁互相协调。本实现要求宿主 PID/UID 和 procfs 视图一致，不支持把 NVML 宿主 PID 与隔离容器 PID 混用。

## 角色优先级与等待名额

固定优先级为 **Analyzer = Engineer > Researcher**，同级按 GPU 请求入队顺序执行，不估算耗时。Researcher 的 quality 和 timing 都属于低优先级；Engineer 正式 timing 和 Analyzer 受控诊断属于同一个高优先级。角色由可信宿主入口确定，诊断请求不能自行选择优先级。

任务的配置为：

```yaml
grader:
  parallel:
    max_workers: 1
    max_waiting_workers: 2
```

这允许一个 run 最多保持三个评测宿主 worker，其中低优先级 worker 最多占两个，保留一个接纳后来提交的 Engineer。daemon 持续检查新提交，不必等当前 Researcher 结束才派发。这里的三个 worker 并不表示可以同时跑三个 GPU 实验：公共 allocator 以 canonical `.coral` 路径标识 run，并限制同一 run 同时持有的 GPU 阶段 reservation 数不超过 `max_workers`。Analyzer 诊断使用相同的 run 名额。

例如 Researcher 已经在等卡，Engineer 后提交：Engineer 可以进入公共 GPU 队列；卡空闲时高优先级先拿卡，Researcher 保持存活等待。没有退出/重启评测，没有新增正式 Attempt、infra failure 或 retry，也不会丢失或重跑已完成的 quality。已经持有 GPU 的实验不被抢占；Researcher 在 quality 与 timing 之间释放 reservation 后重新排队。

优先级只在资源条件满足的请求之间应用。高优先级请求暂时凑不齐严格空闲卡时，可让符合共享规则的 quality 使用其他可用卡；八卡环境也允许不同 run 同时获准使用不重叠的卡。持续有高优先级请求时，Researcher 可能长期等待；本策略没有自动老化提权。

队列记录位于公共锁目录的 `requests/`，记录 PID/启动时间、角色、入队时间、等待期限、共享规则及执行名额。等待超时/取消和正常释放会清理请求；其他申请者也会清理已死亡或过期的等待者。活跃 reservation 不因原等待期限到期而被抢占。更新时需要同步重启使用该锁域的旧版 run，避免旧调用方没有登记角色/执行组而绕开新策略。

`max_waiting_workers` 默认是 `0`，其他任务继续使用原有派发行为。只有在 grader 实现了协作式资源名额控制时才应启用它；单独增加这个值不能保证其他 grader 的实际执行仍串行。

新 run 会复制本任务中已启用的配置。已有 run 使用自己的 `.coral/config.yaml`，需要由 run owner 同步其中的 `grader.parallel.max_waiting_workers: 2`，并重启更新后的 daemon；仅修改任务模板不会热更新已有 run。

## 分配与阶段

每轮申请先拿短时分配锁，按规则筛卡并非阻塞获取各 UUID 的独占锁。只有取得完整四卡且二次状态检查通过才开始执行。失败则释放本轮全部锁再等待，不持有部分卡等待剩余卡。资源暂时忙碌不会生成终态 Attempt 或消耗 infra retry；等待截止、查询持续失败等返回明确 infra failure。

Research 先预留 GPU 执行全部 quality workloads，清理本次子进程后释放；随后重新等待严格空闲的四张卡执行完整 timing 矩阵。Engineering 直接申请 timing。Timing 的空闲条件继续使用 contract：无计算进程、显存占用 ≤512 MiB、利用率 ≤5%。取得卡后先验证与 Dense reference 的硬件可比性；不兼容则释放并报 infra，不启动计时。

Quality 开启共享是对 `hardware.selection: any_idle_four` 默认调度策略的显式例外；训练参数、质量标准、计分公式及 contract hash 保持不变。实际策略版本 `greedylore-gpu-v1`、开关与选卡状态会进入资源证据。

## Timing reference 的监测证据

新增宿主监测可能影响 timing，旧 reference 不能只凭相同 contract hash 自动认可。`dense_references/efficiency.json` 需增加：

```json
{
  "monitoring": {
    "allocation_policy_version": "greedylore-gpu-v1",
    "interval_seconds": 1,
    "evidence_path": "monitoring.json"
  }
}
```

保留原有 `contract_hash`、`environment`、`cells`。`evidence_path` 必须指向该 `dense_references` 目录内可读取的 JSON，不允许指向目录外。

实际证据文件需要包含以下字段：

| 字段 | 内容 |
|---|---|
| `kind` | `monitored_reference` 或 `legacy_equivalence` |
| `contract_hash` | 与 efficiency reference 相同 |
| `environment` | 与 efficiency reference 的环境对象完全相同 |
| `allocation_policy_version` | `greedylore-gpu-v1` |
| `interval_seconds` | 与本次配置相同的监测间隔 |

`monitored_reference` 只用于确实在相同监测控制器下重新测量的 Dense reference。不要给旧测量补写这一标签。

若复用旧 reference，应先固定容差，在代表性环境用相同 GPU、Dense 命令和交错重复测量比较旧/新控制器，并保留测量来源。`legacy_equivalence` 证据还必须包含：`predeclared: true`、介于 0 和 1 之间的 `relative_tolerance`，以及 `relative_difference_interval: [下界, 上界]`。整个差异区间必须落在正负容差内。区间应反映真实测量的不确定性，不能用两个点估计冒充区间。缺少证据或未达到等价要求时，先补测参考数据；grader 会在占用 GPU 前拒绝不合格证据。

字段校验只能验证证据一致性，不能代替 run owner 对测量真实性、预先固定容差和统计方法的责任。本次实现不自动生成 GPU 性能证据，不修改历史 reference。

## 干扰、停止与证据

Timing 允许的计算进程必须属于本次创建的进程组/session，不能仅因同 UID 就放行。发现外部计算进程、未知归属或监测查询失败时，只清理本次进程组，返回 infra failure，不发布本次有效速度分数；不终止其他用户或其他 run 的进程。

文件锁不能阻止外部任务启动，采样也无法保证捕捉两次检查之间的短暂占用。不同卡上的任务还可能竞争 CPU、IO、内存带宽或互连；需要实际性能验证，不能把不同卡解释为整机隔离。

启动方在创建 worker 前通过可信上下文记录原父 daemon 的 PID 和启动时间，worker 初始化时先核对，再在等待与执行期间继续监测。父进程提前退出而导致 worker 被收养时也不能继续运行。父进程退出或 worker 收到 SIGTERM/SIGINT 后取消并清理自己的任务。沿用现有 manager 停止宽限期；强杀 worker 无法保证执行清理，遗留的本 UID 计算进程会让后续 allocator 跳过对应卡，不凭锁释放就认定卡空闲。

每个 Attempt 的 `eval_logs` 下保存：

- `gpu_resources.jsonl`：阶段、等待/预留/执行/释放/失败历史，等待重复信息限频。
- `gpu_resources.json`：最新资源状态快照。
- 各命令的 `process.json` 和 stdout/stderr：包括监测/取消导致的 `infra_failure`。

同 SHA infra failure 继续使用既有 retry 流程。Quality 完成后等待 timing 不会在同一存活 worker 内重跑 quality；worker 退出后的 retry 仍重新执行，不新增跨 worker checkpoint。
