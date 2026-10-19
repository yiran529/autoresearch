"""Wait budgets for trusted calls and asynchronous evaluation clients."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coral.config import CoralConfig

TRUSTED_WORKER_TIMEOUT = 3600
TRUSTED_CLIENT_TIMEOUT = 3660
DIAGNOSTIC_EXECUTION_TIMEOUT = 600
DIAGNOSTIC_QUEUE_TIMEOUT = 3600
DIAGNOSTIC_WORKER_TIMEOUT = 4500
DIAGNOSTIC_CLIENT_TIMEOUT = 4800


def default_eval_wait(config: CoralConfig) -> float:
    """Allow at least an hour plus two GPU acquisition stages when configured.

    Quality and timing may each acquire GPUs. Preserve longer grader budgets;
    this client budget never changes the grader's own execution timeout.
    """
    execution = max(3600, max(0, config.grader.timeout) * 2 + 60)
    queue = config.grader.args.get('gpu_wait_timeout_seconds', 0)
    if type(queue) not in (int, float) or not math.isfinite(queue) or queue < 0:
        queue = 0
    return execution + 2 * queue
