"""Model parameter selection and CLI options for the local Muon optimizer."""

from typing import Optional

from torch import nn

from .muon import Muon


_OUTPUT_HEAD_NAMES = {"lm_head", "classifier", "score", "fc", "linear"}
_NORM_MODULES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
)


def add_muon_args(parser, *, scalar_lr_default=None, scalar_weight_decay_default=None):
    parser.add_argument("--muon_mu", type=float, default=0.95, help="Muon momentum")
    parser.add_argument(
        "--muon_epsilon",
        type=float,
        default=1e-8,
        help="Polar Express normalization epsilon",
    )
    parser.add_argument(
        "--muon_scalar_lr",
        type=float,
        default=scalar_lr_default,
        help="AdamW fallback LR; defaults to base LR",
    )
    parser.add_argument("--muon_scalar_beta1", type=float, default=0.9)
    parser.add_argument("--muon_scalar_beta2", type=float, default=0.999)
    parser.add_argument("--muon_scalar_eps", type=float, default=1e-8)
    parser.add_argument(
        "--muon_scalar_weight_decay",
        type=float,
        default=scalar_weight_decay_default,
        help="AdamW fallback weight decay; defaults to base weight decay",
    )
    parser.add_argument(
        "--muon_adjust_lr",
        choices=("spectral_norm", "rms_norm", "none"),
        default="spectral_norm",
        help="Matrix-size learning-rate scaling",
    )
    parser.add_argument(
        "--muon_compile", action="store_true", help="Compile Polar Express after first use"
    )
    parser.add_argument(
        "--muon_local_orthogonalization",
        action="store_true",
        help="Orthogonalize all matrices on every rank instead of sharing work",
    )


def build_muon_optimizer(
    model: nn.Module,
    *,
    lr: float,
    scalar_lr: Optional[float] = None,
    mu: float = 0.95,
    weight_decay: float = 0.01,
    scalar_weight_decay: Optional[float] = None,
    scalar_betas: tuple[float, float] = (0.9, 0.999),
    scalar_epsilon: float = 1e-8,
    muon_epsilon: float = 1e-8,
    adjust_lr: Optional[str] = "spectral_norm",
    compile_orthogonalization: bool = False,
    distributed_orthogonalization: bool = True,
) -> Muon:
    """Assign each trainable parameter to matrix Muon or AdamW fallback."""
    if scalar_lr is None:
        scalar_lr = lr
    if scalar_weight_decay is None:
        scalar_weight_decay = weight_decay

    matrix_ids = set()
    excluded_ids = set()
    norm_ids = set()
    for module_name, module in model.named_modules():
        head = any(part in _OUTPUT_HEAD_NAMES for part in module_name.split("."))
        if isinstance(module, _NORM_MODULES):
            norm_ids.update(id(param) for param in module.parameters(recurse=False))
        if isinstance(module, nn.Embedding) or head:
            excluded_ids.update(id(param) for param in module.parameters(recurse=False))
        if isinstance(module, (nn.Linear, nn.Conv2d)) and not head:
            weight = getattr(module, "weight", None)
            if weight is not None and weight.requires_grad and weight.ndim >= 2:
                matrix_ids.add(id(weight))
    matrix_ids.difference_update(excluded_ids)

    matrix_params = []
    scalar_decay = []
    scalar_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in matrix_ids:
            matrix_params.append(param)
        elif param.ndim < 2 or id(param) in norm_ids or name.endswith("bias"):
            scalar_no_decay.append(param)
        else:
            scalar_decay.append(param)

    if not matrix_params:
        raise ValueError("Model has no Muon-eligible matrix weights")

    groups = [
        {
            "params": matrix_params,
            "algorithm": "muon",
            "lr": lr,
            "weight_decay": weight_decay,
            "epsilon": muon_epsilon,
        }
    ]
    if scalar_decay:
        groups.append(
            {
                "params": scalar_decay,
                "algorithm": "adamw",
                "lr": scalar_lr,
                "weight_decay": scalar_weight_decay,
                "epsilon": scalar_epsilon,
            }
        )
    if scalar_no_decay:
        groups.append(
            {
                "params": scalar_no_decay,
                "algorithm": "adamw",
                "lr": scalar_lr,
                "weight_decay": 0.0,
                "epsilon": scalar_epsilon,
            }
        )

    return Muon(
        groups,
        lr=lr,
        mu=mu,
        betas=scalar_betas,
        epsilon=muon_epsilon,
        weight_decay=weight_decay,
        adjust_lr=adjust_lr,
        compile_orthogonalization=compile_orthogonalization,
        distributed_orthogonalization=distributed_orthogonalization,
    )
