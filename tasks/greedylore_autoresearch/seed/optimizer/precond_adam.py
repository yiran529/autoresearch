# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0


from typing import cast, List, Optional, Tuple, Union
from torch import Tensor

import torch
import numpy as np
import torch.distributed as dist



class PrecondAdam(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        bias_correction=True,
        eps_inside_sqrt=False,
    ):

        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay)

        super(PrecondAdam, self).__init__(params, defaults)

        self.param_groups = list(params)
        self.eps_inside_sqrt = eps_inside_sqrt
        self.amsgrad = amsgrad

        self.cur_step = 0
        self.hook_state = None

        print(f"PrecondAdam optimizer initialized!!")

        assert dist.is_initialized(), "Please initialize the torch distributed backend."

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        self.cur_step += 1

        # print(f"PrecondAdam step {self.cur_step}, Adam freeze key: {self.hook_state.adam_freeze_key}")
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if self.amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                beta1, beta2 = group['betas']

                # bias correction
                bias_correction1 = 1 - beta1 ** self.cur_step
                bias_correction2 = 1 - beta2 ** self.cur_step

                # update the exponential moving averages
                # assert isinstance(self.hook_state, HookState), "Please set the hook state before calling the optimizer."
                if not self.hook_state.adam_freeze_key:
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                else:
                    # adam is frozen, grad is momentum accually
                    exp_avg.set_(grad)
                    # if self.cur_step == 422 and grad.shape[0] == 50265:
                    #     # and id(p) in [139833935791984, 140650538108688, 140358931679088, 140432279881488]
                    #     print(f"Rank{dist.get_rank()}, first 10 item of exp_avg(before): {exp_avg.flatten()[0:10]}") 
                    #     print(f"Rank{dist.get_rank()}, first 10 item of grad: {grad.flatten()[0:10]}")
                    #     print(f"Rank{dist.get_rank()}, beta1: {beta1}")
                    # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # if self.cur_step == 422 and grad.shape[0] == 50265:
                    #     print(f"Rank{dist.get_rank()}, first 10 item of exp_avg(final): {exp_avg.flatten()[0:10]}") 
                    # exit()
                    bias_correction2 = 1 - beta2 ** self.hook_state.start_compress_iter
                
                # ams_grad choice and epsilon choice
                final_exp_avg_sq = torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq) if self.amsgrad else exp_avg_sq
                denom = (final_exp_avg_sq + group['eps']).sqrt() if self.eps_inside_sqrt else (final_exp_avg_sq.sqrt()).add_(group['eps'])
                
                # weight decay
                p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                # update the parameters
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

