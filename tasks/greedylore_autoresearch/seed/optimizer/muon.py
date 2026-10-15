"""DDP-compatible Muon with AdamW fallback for non-matrix parameters.

The matrix update follows dion's default Nesterov + Polar Express path. Gradient
communication remains in the model's DDP hook. This optimizer may issue an
AllGather to share orthogonalization work.
"""

import math
from collections import defaultdict
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer


_POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def polar_express(gradient: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Approximate the polar factor of a matrix or batch of matrices."""
    if gradient.ndim < 2:
        raise ValueError("Polar Express expects matrices")

    x = gradient.to(torch.bfloat16)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + epsilon)
    if gradient.size(-2) > gradient.size(-1):
        for a, b, c in _POLAR_EXPRESS_COEFFS:
            gram = x.mT @ x
            polynomial = b * gram + c * (gram @ gram)
            x = a * x + x @ polynomial
    else:
        for a, b, c in _POLAR_EXPRESS_COEFFS:
            gram = x @ x.mT
            polynomial = b * gram + c * (gram @ gram)
            x = a * x + polynomial @ x
    return x


class Muon(Optimizer):
    """Muon matrix updates and AdamW scalar updates in one optimizer."""

    def __init__(
        self,
        params,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        adjust_lr: Optional[str] = "spectral_norm",
        orthogonalize: Optional[Callable[[Tensor, float], Tensor]] = None,
        compile_orthogonalization: bool = False,
        distributed_orthogonalization: bool = True,
        process_group=None,
    ):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and non-negative")
        if not math.isfinite(mu) or not 0 <= mu < 1:
            raise ValueError("mu must be finite and in [0, 1)")
        if len(betas) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in betas):
            raise ValueError("betas must be finite and in [0, 1)")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError("adjust_lr must be spectral_norm, rms_norm, or None")
        if orthogonalize is not None and compile_orthogonalization:
            raise ValueError("compile_orthogonalization cannot be combined with custom orthogonalize")

        defaults = dict(
            lr=lr,
            mu=mu,
            betas=betas,
            epsilon=epsilon,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            algorithm="muon",
        )
        super().__init__(params, defaults)
        self._orthogonalize = orthogonalize or polar_express
        if compile_orthogonalization:
            self._orthogonalize = torch.compile(
                self._orthogonalize, dynamic=False, fullgraph=True
            )
        self._distributed_orthogonalization = distributed_orthogonalization
        self._process_group = process_group
        self._communication_bits = {
            "gradient_presence": 0,
            "orthogonalization_results": 0,
        }
        self._communication_bits_this_step = {
            "gradient_presence": 0,
            "orthogonalization_results": 0,
        }

        for group in self.param_groups:
            algorithm = group["algorithm"]
            if algorithm not in ("muon", "adamw"):
                raise ValueError(f"Unknown Muon parameter-group algorithm: {algorithm}")
            if not math.isfinite(group["weight_decay"]) or group["weight_decay"] < 0:
                raise ValueError("weight_decay must be finite and non-negative")
            for param in group["params"]:
                if algorithm == "muon":
                    if param.ndim < 2:
                        raise ValueError("Muon requires matrix parameters")
                    self.state[param]["momentum"] = torch.zeros_like(param)
                else:
                    state = self.state[param]
                    state_dtype = (
                        torch.float32
                        if param.dtype in (torch.bfloat16, torch.float16)
                        else param.dtype
                    )
                    state["exp_avg"] = torch.zeros_like(param, dtype=state_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(param, dtype=state_dtype)
                    state["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._communication_bits_this_step = {
            "gradient_presence": 0,
            "orthogonalization_results": 0,
        }
        self._validate_distributed_grad_layout()
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def load_state_dict(self, state_dict):
        """Load state while preserving FP32 AdamW moments for low-precision parameters."""
        preserved_moments = []
        saved_groups = state_dict.get("param_groups", [])
        saved_state = state_dict.get("state", {})
        if len(saved_groups) == len(self.param_groups):
            for current_group, saved_group in zip(self.param_groups, saved_groups):
                if current_group.get("algorithm") != "adamw":
                    continue
                for param, saved_param_id in zip(current_group["params"], saved_group["params"]):
                    state = saved_state.get(saved_param_id, {})
                    if param.dtype in (torch.bfloat16, torch.float16) and {
                        "exp_avg",
                        "exp_avg_sq",
                    }.issubset(state):
                        preserved_moments.append(
                            (
                                param,
                                state["exp_avg"]
                                .detach()
                                .to(device=param.device, dtype=torch.float32)
                                .clone(),
                                state["exp_avg_sq"]
                                .detach()
                                .to(device=param.device, dtype=torch.float32)
                                .clone(),
                            )
                        )
        super().load_state_dict(state_dict)
        for param, exp_avg, exp_avg_sq in preserved_moments:
            self.state[param]["exp_avg"] = exp_avg
            self.state[param]["exp_avg_sq"] = exp_avg_sq

    def communication_bits_stats(self):
        return {
            "total": dict(self._communication_bits),
            "this_step": dict(self._communication_bits_this_step),
        }

    def _record_all_gather(self, category: str, tensor: Tensor, world_size: int):
        bits = world_size * (world_size - 1) * tensor.numel() * tensor.element_size() * 8
        self._communication_bits[category] += bits
        self._communication_bits_this_step[category] += bits

    def _validate_distributed_grad_layout(self):
        if (
            not self._distributed_orthogonalization
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return

        process_group = self._process_group or dist.group.WORLD
        if dist.get_world_size(process_group) == 1:
            return

        relevant = []
        for group in self.param_groups:
            if group["algorithm"] != "muon":
                continue
            shape_counts = defaultdict(int)
            for param in group["params"]:
                shape_counts[
                    (param.shape[0], math.prod(param.shape[1:]), param.dtype, param.device)
                ] += 1
            relevant.extend(
                param
                for param in group["params"]
                if shape_counts[
                    (param.shape[0], math.prod(param.shape[1:]), param.dtype, param.device)
                ]
                > 1
            )
        if not relevant:
            return

        presence = torch.tensor(
            [param.grad is not None for param in relevant],
            dtype=torch.uint8,
            device=relevant[0].device,
        )
        gathered = [
            torch.empty_like(presence) for _ in range(dist.get_world_size(process_group))
        ]
        dist.all_gather(gathered, presence, group=process_group)
        self._record_all_gather("gradient_presence", presence, len(gathered))
        if any(not torch.equal(layout, gathered[0]) for layout in gathered[1:]):
            raise RuntimeError(
                "Muon gradient presence differs across ranks; distributed "
                "orthogonalization requires identical used-parameter layouts"
            )

    def _step_muon_group(self, group):
        batches = defaultdict(list)
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            matrix_shape = (param.shape[0], math.prod(param.shape[1:]))
            batches[(matrix_shape, param.dtype, param.device)].append(param)

        for params in batches.values():
            momentums = [self.state[param]["momentum"] for param in params]
            grads = [param.grad.to(dtype=momentum.dtype) for param, momentum in zip(params, momentums)]
            torch._foreach_mul_(momentums, group["mu"])
            torch._foreach_add_(momentums, grads)
            if group["nesterov"]:
                updates = torch._foreach_mul(momentums, group["mu"])
                torch._foreach_add_(updates, grads)
            else:
                updates = momentums

            matrix_updates = torch.stack([update.reshape(update.shape[0], -1) for update in updates])
            orthogonalized = self._orthogonalize_batch(matrix_updates, group["epsilon"])
            if orthogonalized.shape != matrix_updates.shape:
                raise ValueError("Orthogonalization changed matrix shape")

            matrix_shape = matrix_updates.shape[-2:]
            if group["adjust_lr"] == "spectral_norm":
                lr_scale = math.sqrt(matrix_shape[0] / matrix_shape[1])
            elif group["adjust_lr"] == "rms_norm":
                lr_scale = 0.2 * math.sqrt(max(matrix_shape))
            else:
                lr_scale = 1.0

            torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
            shaped_updates = [
                update.reshape_as(param).to(dtype=param.dtype)
                for update, param in zip(orthogonalized.unbind(0), params)
            ]
            torch._foreach_add_(params, shaped_updates, alpha=-group["lr"] * lr_scale)

    def _orthogonalize_batch(self, updates: Tensor, epsilon: float) -> Tensor:
        if (
            not self._distributed_orthogonalization
            or not dist.is_available()
            or not dist.is_initialized()
            or updates.shape[0] == 1
        ):
            return self._orthogonalize(updates, epsilon)

        process_group = self._process_group or dist.group.WORLD
        world_size = dist.get_world_size(process_group)
        if world_size == 1:
            return self._orthogonalize(updates, epsilon)

        rank = dist.get_rank(process_group)
        original_count = updates.shape[0]
        pad_count = (-original_count) % world_size
        if pad_count:
            updates = torch.cat(
                (updates, updates.new_zeros((pad_count, *updates.shape[1:]))), dim=0
            )
        per_rank = updates.shape[0] // world_size
        mine = updates.narrow(0, rank * per_rank, per_rank)
        local_result = self._orthogonalize(mine, epsilon).contiguous()
        gathered = [torch.empty_like(local_result) for _ in range(world_size)]
        dist.all_gather(gathered, local_result, group=process_group)
        self._record_all_gather("orthogonalization_results", local_result, world_size)
        return torch.cat(gathered, dim=0)[:original_count]

    def _step_adamw_group(self, group):
        beta1, beta2 = group["betas"]
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("AdamW fallback does not support sparse gradients")
            state = self.state[param]
            grad = grad.to(dtype=state["exp_avg"].dtype)
            state["step"] += 1
            state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            param.mul_(1 - group["lr"] * group["weight_decay"])
            correction1 = 1 - beta1 ** state["step"]
            correction2 = 1 - beta2 ** state["step"]
            denominator = (
                state["exp_avg_sq"]
                .sqrt()
                .div_(math.sqrt(correction2))
                .add_(group["epsilon"])
            )
            update = state["exp_avg"].div(correction1).div_(denominator)
            param.add_(update.to(dtype=param.dtype), alpha=-group["lr"])
