# GPU 分配与跨 run 协调

GreedyLoRE grader 在同一 Linux 宿主、同一 UID 的多个独立 CORAL run 之间协调 GPU。每个 run 保留自己的串行评测队列；八张满足条件的卡可以支持两个并发的四卡阶段。Agent 并发数与评测队列并发数独立。

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
